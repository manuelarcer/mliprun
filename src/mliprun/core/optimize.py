"""Geometry optimization engine using ASE."""
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch, GPMin, MDMin

from mliprun.core.utils import calc_fmax
from mliprun.core.run_record import RunContext, RunRecord, collect_provenance

logger = logging.getLogger(__name__)

OPTIMIZER_MAP = {
    "fire": FIRE,
    "bfgs": BFGS,
    "lbfgs": LBFGS,
    "bfgsls": BFGSLineSearch,
    "gpmin": GPMin,
    "mdmin": MDMin,
}


def _wrap_for_cell_relaxation(atoms):
    """Wrap atoms with a cell filter so the optimizer sees cell DOFs too.

    Prefers ASE 3.23+'s :class:`ase.filters.FrechetCellFilter` (better-behaved
    for soft modes); falls back to :class:`ase.constraints.ExpCellFilter` on
    older ASE.
    """
    try:
        from ase.filters import FrechetCellFilter as _Filter
    except ImportError:
        from ase.constraints import ExpCellFilter as _Filter
    return _Filter(atoms)


def run_optimization(
    atoms,
    optimizer: str = "bfgs",
    fmax: float = 0.05,
    max_steps: int = 200,
    trajectory: str = "opt.traj",
    logfile: str = "opt.log",
    output_dir: str | Path = ".",
    model_name: str = "mlip",
    verbose: bool = True,
    relax_cell: bool = False,
    plot: bool = False,
    run_context: Optional[RunContext] = None,
    device_requested: str = "auto",
    device_resolved: str = "auto",
    uma_task: Optional[str] = None,
    mace_head: Optional[str] = None,
) -> bool:
    """Run geometry optimization on an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with calculator attached.
    optimizer : str
        Optimizer algorithm: ``'fire'``, ``'bfgs'``, ``'lbfgs'``,
        ``'bfgsls'``, ``'gpmin'``, ``'mdmin'``.
    fmax : float
        Force convergence criterion (eV/Ang). When ``relax_cell=True`` the
        criterion is applied to the combined atomic forces + cell virials
        emitted by the cell filter.
    max_steps : int
        Maximum number of optimization steps.
    trajectory : str or Path
        Trajectory filename.
    logfile : str or Path
        Log filename.
    output_dir : str or Path
        Directory for output files.
    model_name : str
        Name of MLIP model for parameter file.
    verbose : bool
        If True, show optimization progress table.
    relax_cell : bool
        If True, also relax the simulation cell (positions + cell, like VASP
        ISIF=3). Uses ``ase.filters.FrechetCellFilter`` when available,
        otherwise ``ase.constraints.ExpCellFilter``.
    plot : bool
        If True, write the ``*_convergence.png`` figure. Defaults to False:
        the matplotlib figure/save is per-structure IO that dominates short
        relaxations (e.g. frozen-surface site scans), so plotting is opt-in.
        The ``*_convergence.csv`` is always written, so the data is retained
        either way and can be plotted later.
    run_context : RunContext, optional
        Declares the command, batch identity, and where each parameter value
        came from. When omitted the record still gets written, with every
        parameter tagged ``unspecified``.
    device_requested : str
        The device as asked for (e.g. ``'auto'``), recorded for provenance.
    device_resolved : str
        The device actually used (e.g. ``'cuda'``).
    uma_task : str, optional
        UMA task actually used, recorded for provenance. Ignored for
        non-UMA models.
    mace_head : str, optional
        MACE head actually used, recorded for provenance. Ignored for
        non-MACE models.

    Returns
    -------
    bool
        Whether optimization converged.

    Raises
    ------
    ValueError
        If ``optimizer`` is not recognised.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    traj_file = output_path / trajectory
    log_file = output_path / logfile

    logfile_stem = Path(logfile).stem
    csv_file = output_path / f"{logfile_stem}_convergence.csv"
    convergence_plot = output_path / f"{logfile_stem}_convergence.png"
    final_structure = output_path / f"{logfile_stem}_final.vasp"
    # CONTCAR mirror of the final structure so a follow-up DFT run managed by
    # asetools can restart from this directory (it reads OUTCAR or CONTCAR).
    contcar_file = output_path / "CONTCAR"

    optimizer_name = optimizer.lower()
    if optimizer_name not in OPTIMIZER_MAP:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. "
            f"Available: {list(OPTIMIZER_MAP.keys())}"
        )

    OptimizerClass = OPTIMIZER_MAP[optimizer_name]

    record = RunRecord.begin(
        output_path,
        command="optimize",
        stage_kind="optimize",
        parameters={
            "optimizer": optimizer_name,
            "fmax": fmax,
            "max_steps": max_steps,
            "relax_cell": relax_cell,
            "trajectory": trajectory,
            "logfile": logfile,
            "plot": plot,
            "verbose": verbose,
        },
        inputs={
            "n_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
        },
        provenance=collect_provenance(
            mlip_model=model_name,
            device_requested=device_requested,
            device_resolved=device_resolved,
            uma_task=uma_task,
            mace_head=mace_head,
        ),
        run_context=run_context,
    )

    # When relax_cell, the optimizer sees the filtered object (atoms + cell
    # DOFs). Trajectory frames are still written from the underlying atoms.
    opt_target = _wrap_for_cell_relaxation(atoms) if relax_cell else atoms

    traj = Trajectory(str(traj_file), "w", atoms)

    log_data = {"step": [], "energy(eV)": [], "fmax(eV/A)": []}

    def log_convergence():
        step = opt.nsteps
        energy = atoms.get_potential_energy()
        # opt_target.get_forces() includes cell virials when relax_cell is on,
        # matching what the optimizer's fmax convergence is checking against.
        fmax_val = calc_fmax(opt_target.get_forces())
        log_data["step"].append(step)
        log_data["energy(eV)"].append(energy)
        log_data["fmax(eV/A)"].append(fmax_val)

    try:
        if verbose:
            opt = OptimizerClass(opt_target, trajectory=str(traj_file), logfile=str(log_file))
            opt.attach(log_convergence, interval=1)
            logger.info("Starting optimization with %s (fmax=%.4f, max_steps=%d, relax_cell=%s)",
                        optimizer.upper(), fmax, max_steps, relax_cell)
            converged = opt.run(fmax=fmax, steps=max_steps)
        else:
            with open(log_file, "w") as lf:
                opt = OptimizerClass(opt_target, trajectory=str(traj_file), logfile=lf)
                opt.attach(log_convergence, interval=1)
                converged = opt.run(fmax=fmax, steps=max_steps)

        final_energy = atoms.get_potential_energy()
        final_fmax = calc_fmax(opt_target.get_forces())
    except Exception as exc:
        record.complete(status="failed", results={"error": str(exc)})
        raise

    logger.info("Optimization complete (converged=%s, steps=%d, energy=%.6f eV, fmax=%.6f eV/Ang)",
                converged, opt.nsteps, final_energy, final_fmax)

    record.complete(
        status="converged" if converged else "not_converged",
        steps=int(opt.nsteps),
        results={
            "converged": bool(converged),
            "final_energy_eV": float(final_energy),
            "final_fmax_eV_per_A": float(final_fmax),
        },
    )

    write(str(final_structure), atoms, format="vasp")
    write(str(contcar_file), atoms, format="vasp")

    df = pd.DataFrame(log_data)
    df.to_csv(csv_file, index=False)

    # Plot convergence (skippable: the figure + savefig is per-structure IO that
    # dominates short relaxations; the CSV above retains the same data).
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        ax1.plot(df["step"], df["energy(eV)"], marker="o", markersize=4, linewidth=1.5)
        ax1.set_xlabel("Optimization Step")
        ax1.set_ylabel("Energy (eV)")
        ax1.set_title(f"Energy Convergence ({optimizer.upper()})")
        ax1.grid(True, alpha=0.3)

        ax2.plot(df["step"], df["fmax(eV/A)"], marker="o", markersize=4, linewidth=1.5, color="orange")
        ax2.axhline(y=fmax, color="r", linestyle="--", label=f"fmax target = {fmax}")
        ax2.set_xlabel("Optimization Step")
        ax2.set_ylabel("Max Force (eV/Ang)")
        ax2.set_title("Force Convergence")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

        plt.tight_layout()
        plt.savefig(convergence_plot, dpi=150)
        plt.close()

    return converged
