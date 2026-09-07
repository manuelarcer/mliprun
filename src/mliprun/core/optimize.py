"""Geometry optimization engine using ASE."""
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch, GPMin, MDMin

from mliprun.core.committee.calculator import (
    CommitteeTraceWriter,
    uncertainty_summary,
    write_peratom_sigma,
)
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


def _committee_parameters(committee, committee_config, threshold) -> dict:
    """Committee-only entries for the record's parameter block."""
    if committee is None:
        return {}
    parameters = {"uncertainty_threshold": threshold,
                  "n_members": len(committee.members)}
    if committee_config is not None:
        parameters["committee_file"] = committee_config.source_path
    return parameters


def _committee_provenance(committee, committee_config):
    """The ``provenance.committee`` block, declared values plus measured ones.

    ``committee_config`` carries only what ``committee.yaml`` DECLARED. The
    versions each member reported from inside its own env once it had loaded
    -- interpreter, ASE, torch, MLIP package and its version -- are collected
    by ``worker._versions``, returned by ``CommitteeCalculator.start()`` and
    kept on ``member_versions``. Merging them here is the only reason a
    committee record can say what actually ran rather than what was asked
    for; without it a committee record carries no MLIP version at all, where
    a single-model record carries three.

    ``getattr`` rather than an attribute access: ``run_optimization`` accepts
    any object with the committee protocol, and the test doubles in this
    repo's suite predate ``member_versions``.
    """
    if committee_config is None:
        return None
    return committee_config.as_provenance(
        measured_versions=getattr(committee, "member_versions", None))


def _plot_convergence(df, fmax: float, optimizer: str, committee_rows=None):
    """Build the convergence figure.

    Two panels normally -- energy and max force. A committee run gets a third
    carrying the per-atom force disagreement, on the same log scale as the
    force panel so the two are read against each other: where sigma_max
    approaches fmax, the minimum sits inside the committee's own noise.

    Returns
    -------
    matplotlib.figure.Figure
        The caller saves and closes it.
    """
    n_panels = 3 if committee_rows else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4 * n_panels))
    ax1, ax2 = axes[0], axes[1]

    ax1.plot(df["step"], df["energy(eV)"], marker="o", markersize=4,
             linewidth=1.5)
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title(f"Energy Convergence ({optimizer.upper()})")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["step"], df["fmax(eV/A)"], marker="o", markersize=4,
             linewidth=1.5, color="orange")
    ax2.axhline(y=fmax, color="r", linestyle="--",
                label=f"fmax target = {fmax}")
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Max Force (eV/Ang)")
    ax2.set_title("Force Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    if committee_rows:
        ax3 = axes[2]
        steps = [row["step"] for row in committee_rows]
        sigma_max = [row["sigma_max_eV_per_A"] for row in committee_rows]
        sigma_mean = [row["sigma_mean_eV_per_A"] for row in committee_rows]
        ax3.plot(steps, sigma_max, marker="o", markersize=4, linewidth=1.5,
                 label="sigma_max")
        ax3.plot(steps, sigma_mean, marker="s", markersize=3, linewidth=1.0,
                 label="sigma_mean")
        ax3.axhline(y=fmax, color="r", linestyle="--",
                    label=f"fmax target = {fmax}")
        ax3.set_xlabel("Optimization Step")
        ax3.set_ylabel("Force disagreement (eV/Ang)")
        ax3.set_title("Committee Disagreement")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # sigma is non-negative by construction. A plain log scale silently
        # drops every non-positive value with no warning (matplotlib just
        # excludes them from autoscaling), so an exactly-agreeing committee
        # -- sigma identically 0, this project's own sanity check -- would
        # render as an empty panel. Never clamp a true zero to a fake
        # epsilon: that falsifies the figure. Instead pick the scale from
        # what is actually being plotted.
        positive_values = [v for v in sigma_max + sigma_mean if v > 0]
        if not positive_values:
            # No disagreement at any step: linear keeps both flat-zero
            # traces on screen, and the annotation says why the panel is
            # flat rather than leaving the reader to guess.
            ax3.set_yscale("linear")
            ax3.text(0.5, 0.5,
                     "sigma is identically zero: committee members agree "
                     "exactly at every step",
                     transform=ax3.transAxes, ha="center", va="center",
                     fontsize=9, style="italic",
                     bbox=dict(boxstyle="round", facecolor="white",
                               alpha=0.85))
        elif len(positive_values) < len(sigma_max) + len(sigma_mean):
            # Mixed: some steps agree exactly, others don't. A log scale
            # would drop precisely the exact-agreement steps while plotting
            # the rest normally -- more misleading than an empty panel,
            # since it looks like clean data. symlog keeps every point by
            # treating |sigma| <= linthresh linearly.
            ax3.set_yscale("symlog", linthresh=min(positive_values))
        else:
            # Normal case: every value is a real, positive disagreement.
            ax3.set_yscale("log")

    fig.tight_layout()
    return fig


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
    sevennet_task: Optional[str] = None,
    committee=None,
    committee_config=None,
    uncertainty_threshold: Optional[float] = None,
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
        Ignored on a committee run: both device fields become the string
        ``'committee'``, because the driver process resolves no device at all
        and each member's own ``device``/``gpu`` is recorded per member.
    device_resolved : str
        The device actually used (e.g. ``'cuda'``). See ``device_requested``
        for the committee case.
    uma_task : str, optional
        UMA task actually used, recorded for provenance. Ignored for
        non-UMA models.
    sevennet_task : str, optional
        SevenNet inference task, recorded in the run record when the model is
        a ``7net*`` tag. No default: see ``validate_mlip``.
    mace_head : str, optional
        MACE head actually used, recorded for provenance. Ignored for
        non-MACE models.
    committee : CommitteeCalculator, optional
        The committee driving this relaxation, when there is one. It must
        already be started and attached as ``atoms.calc``; this function only
        reads its per-step statistics and writes the trace. Teardown belongs
        to whoever built it -- an API caller may reuse one loaded committee
        across many structures, exactly as ``optimize batch`` reuses one
        calculator.
    committee_config : CommitteeConfig, optional
        The parsed ``committee.yaml``, for the run record: member list, envs,
        resolved levels of theory, and the file's SHA-256.
    uncertainty_threshold : float, optional
        sigma_max above which the final configuration is flagged as
        high-disagreement. Defaults to ``fmax``: if the models disagree about
        the forces by more than the convergence tolerance, the located
        minimum sits inside the committee's own noise and the geometry is not
        resolved. The value applied and where it came from are both recorded.

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

    committee_csv = output_path / f"{logfile_stem}_committee.csv"
    committee_peratom_csv = output_path / f"{logfile_stem}_committee_peratom.csv"

    # Default threshold is fmax itself -- self-scaling, and physically
    # motivated. UNCALIBRATED: no same-level sigma_F measurement exists yet
    # (see the design's "Flagging rule"), so both the value and its origin
    # travel with the flag.
    threshold = (float(uncertainty_threshold)
                 if uncertainty_threshold is not None else float(fmax))
    threshold_source = "explicit" if uncertainty_threshold is not None else "fmax"

    optimizer_name = optimizer.lower()
    if optimizer_name not in OPTIMIZER_MAP:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. "
            f"Available: {list(OPTIMIZER_MAP.keys())}"
        )

    OptimizerClass = OPTIMIZER_MAP[optimizer_name]

    if committee is not None:
        # A committee may be reused across structures (the docstring above
        # invites exactly that), and it still carries the PREVIOUS structure's
        # statistics. If this run fails before its first successful evaluation
        # -- a member rejecting the geometry, which is the failure preflight
        # exists for -- the exception handler below would otherwise write the
        # previous structure's sigma, its atom index and its flag into THIS
        # run's record, wearing this structure's element symbols. Clearing here
        # is safe: `log_convergence` guards on `latest is not None`, the
        # optimizer's first `get_potential_energy()` repopulates it before the
        # observer fires, and `uncertainty_summary(rows=[], latest=None)`
        # returns all-null with `flagged: False`, which is the honest output.
        committee.latest = None
        committee.latest_uncertainty_summary = None

        # The driver process is designed to have no torch at all (ADR 0001), so
        # `resolve_device("auto")` returns "cpu" here on ImportError even when
        # every member is on its own GPU. Recording that would be a false claim
        # about what hardware ran the calculation. The authoritative per-member
        # `device` and `gpu` live in the committee provenance block; these two
        # fields say only that device selection happened per member.
        device_requested = "committee"
        device_resolved = "committee"

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
            **_committee_parameters(committee, committee_config, threshold),
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
            sevennet_task=sevennet_task,
            committee=_committee_provenance(committee, committee_config),
        ),
        run_context=run_context,
    )

    # When relax_cell, the optimizer sees the filtered object (atoms + cell
    # DOFs). Trajectory frames are still written from the underlying atoms.
    opt_target = _wrap_for_cell_relaxation(atoms) if relax_cell else atoms

    traj = Trajectory(str(traj_file), "w", atoms)

    log_data = {"step": [], "energy(eV)": [], "fmax(eV/A)": []}

    trace_writer = (
        CommitteeTraceWriter(committee_csv, committee.member_names,
                             committee.mixed_theory)
        if committee is not None else None
    )

    def log_convergence():
        step = opt.nsteps
        energy = atoms.get_potential_energy()
        # opt_target.get_forces() includes cell virials when relax_cell is on,
        # matching what the optimizer's fmax convergence is checking against.
        fmax_val = calc_fmax(opt_target.get_forces())
        log_data["step"].append(step)
        log_data["energy(eV)"].append(energy)
        log_data["fmax(eV/A)"].append(fmax_val)
        if trace_writer is not None and committee.latest is not None:
            # Both calls above hit the (cached) committee at this geometry, so
            # `latest` is this step's evaluation. Flushed per row, so a run
            # that dies at step 300 keeps its first 300 steps.
            trace_writer.write_step(step, committee.latest, fmax_val)

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
        # Partial results survive: the trace is already on disk, and the
        # record says what the committee had seen when the run died.
        results = {"error": str(exc)}
        if trace_writer is not None:
            trace_writer.close()
            summary = uncertainty_summary(
                trace_writer.rows, committee.latest, threshold=threshold,
                threshold_source=threshold_source,
                symbols=atoms.get_chemical_symbols())
            results["committee_uncertainty"] = summary
            committee.latest_uncertainty_summary = summary
        record.complete(status="failed", results=results)
        raise

    logger.info("Optimization complete (converged=%s, steps=%d, energy=%.6f eV, fmax=%.6f eV/Ang)",
                converged, opt.nsteps, final_energy, final_fmax)

    results = {
        "converged": bool(converged),
        "final_energy_eV": float(final_energy),
        "final_fmax_eV_per_A": float(final_fmax),
    }
    if trace_writer is not None:
        trace_writer.close()
        write_peratom_sigma(committee_peratom_csv,
                            atoms.get_chemical_symbols(),
                            committee.latest["sigma_per_atom"])
        summary = uncertainty_summary(
            trace_writer.rows, committee.latest, threshold=threshold,
            threshold_source=threshold_source,
            symbols=atoms.get_chemical_symbols())
        results["committee_uncertainty"] = summary
        committee.latest_uncertainty_summary = summary
        if summary["flagged"]:
            # INFO, not WARNING: with no logging configured anywhere in this
            # codebase (confirmed by grep for basicConfig/addHandler/setLevel/
            # dictConfig/fileConfig), a WARNING-level record reaches the
            # terminal on its own via `logging.lastResort` -- printing the
            # same message the CLI already echoes from `results`. This line
            # stays for anyone running with verbose logging configured; the
            # CLI echo (reading `committee.latest_uncertainty_summary`, not
            # this call) is the one terminal report at default settings.
            logger.info(
                "High committee disagreement at the final geometry: "
                "sigma_max = %.4f eV/Ang > %.4f (%s). The located minimum "
                "sits inside the committee's own noise; this configuration "
                "deserves a DFT check. Worst atom: %s (%s).",
                summary["sigma_max_final_eV_per_A"], threshold,
                threshold_source, summary["worst_atom"],
                summary["worst_atom_symbol"])

    record.complete(
        status="converged" if converged else "not_converged",
        steps=int(opt.nsteps),
        results=results,
    )

    write(str(final_structure), atoms, format="vasp")
    write(str(contcar_file), atoms, format="vasp")

    df = pd.DataFrame(log_data)
    df.to_csv(csv_file, index=False)

    # Plot convergence (skippable: the figure + savefig is per-structure IO that
    # dominates short relaxations; the CSV above retains the same data).
    if plot:
        figure = _plot_convergence(
            df, fmax, optimizer,
            committee_rows=(trace_writer.rows if trace_writer is not None
                            else None))
        figure.savefig(convergence_plot, dpi=150)
        plt.close(figure)

    return converged
