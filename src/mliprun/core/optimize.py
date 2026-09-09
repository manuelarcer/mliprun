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
    force panel so the two are read against each other: where
    sigma_max_free approaches fmax, the minimum sits inside the committee's
    own noise.

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
        sigma_max_free = [row["sigma_max_free_eV_per_A"]
                          for row in committee_rows]
        sigma_mean_free = [row["sigma_mean_free_eV_per_A"]
                           for row in committee_rows]
        ax3.plot(steps, sigma_max_free, marker="o", markersize=4,
                 linewidth=1.5, label="sigma_max_free")
        ax3.plot(steps, sigma_mean_free, marker="s", markersize=3,
                 linewidth=1.0, label="sigma_mean_free")
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
        positive_values = [v for v in sigma_max_free + sigma_mean_free
                           if v > 0]
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
        elif len(positive_values) < len(sigma_max_free) + len(sigma_mean_free):
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


def _uncertainty_traces(committee_rows):
    """The numbers behind the uncertainty figure, with no drawing in them.

    Separated from ``_plot_uncertainty`` so the arithmetic -- which zero the
    energy is measured from, where the force band is clipped -- can be
    asserted exactly, rather than read back out of a polygon.

    Parameters
    ----------
    committee_rows : list of dict
        Rows as ``CommitteeTraceWriter`` writes them.

    Returns
    -------
    dict
        ``steps``, ``energy`` (relative to step 0) with ``energy_lo`` /
        ``energy_hi``, and ``fmax`` with ``force_lo`` / ``force_hi``.
    """
    steps = [int(row["step"]) for row in committee_rows]
    energies = [float(row["energy_mean_eV"]) for row in committee_rows]
    spreads = [float(row["energy_spread_aligned_eV"])
               for row in committee_rows]
    fmax_values = [float(row["fmax_eV_per_A"]) for row in committee_rows]
    sigmas = [float(row["sigma_max_free_eV_per_A"]) for row in committee_rows]

    # The band is measured after each member's own step-0 energy is removed,
    # so the centre has to share that zero or the two are not commensurate:
    # absolute committee energies carry a per-package offset of tens of eV
    # while the band is ~0.01 eV wide, and the band would render as a line.
    reference = energies[0]
    energy = [value - reference for value in energies]

    return {
        "steps": steps,
        "energy": energy,
        "energy_lo": [e - s for e, s in zip(energy, spreads)],
        "energy_hi": [e + s for e, s in zip(energy, spreads)],
        "fmax": fmax_values,
        # A force magnitude cannot be negative. sigma above fmax means the
        # members disagree about the force by more than its own size -- the
        # regime worth looking at, since the minimum then sits inside the
        # committee's own noise -- and its honest lower edge is zero.
        "force_lo": [max(f - s, 0.0) for f, s in zip(fmax_values, sigmas)],
        "force_hi": [f + s for f, s in zip(fmax_values, sigmas)],
    }


def _plot_uncertainty(committee_rows, fmax: float):
    """Build the committee uncertainty figure.

    One panel, two y-axes against the optimizer step: mean energy relative to
    step 0 with a +/- aligned-spread band on the left, max force with a
    +/- ``sigma_max_free`` band on the right. The point of putting them on one
    panel is the comparison the fmax target line makes possible -- whether the
    force is converging into the committee's own disagreement.

    The band on the force axis is an upper bound rather than the uncertainty
    of the plotted number: ``sigma_max_free`` is the largest disagreement
    anywhere in the free region, and the atom carrying it need not be the atom
    carrying fmax.

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` when the trace has fewer than two steps -- a run that
        converged at step 0 would otherwise get a one-point figure whose
        zero-width band says nothing and implies a lot. The caller saves and
        closes the figure.
    """
    if not committee_rows or len(committee_rows) < 2:
        return None

    traces = _uncertainty_traces(committee_rows)
    steps = traces["steps"]

    fig, ax_energy = plt.subplots(figsize=(8, 5))
    ax_force = ax_energy.twinx()

    energy_color, force_color = "tab:blue", "tab:orange"

    ax_energy.plot(steps, traces["energy"], marker="o", markersize=4,
                   linewidth=1.5, color=energy_color,
                   label="energy (committee mean)")
    ax_energy.fill_between(steps, traces["energy_lo"], traces["energy_hi"],
                           color=energy_color, alpha=0.25, linewidth=0,
                           label="+/- energy spread")
    ax_energy.set_xlabel("Optimization Step")
    ax_energy.set_ylabel("Energy - Energy(step 0) (eV)", color=energy_color)
    ax_energy.tick_params(axis="y", labelcolor=energy_color)
    ax_energy.grid(True, alpha=0.3)

    ax_force.plot(steps, traces["fmax"], marker="s", markersize=4,
                  linewidth=1.5, color=force_color, label="max force")
    ax_force.fill_between(steps, traces["force_lo"], traces["force_hi"],
                          color=force_color, alpha=0.25, linewidth=0,
                          label="+/- sigma_max_free")
    ax_force.axhline(y=fmax, color="r", linestyle="--", linewidth=1.0,
                     label=f"fmax target = {fmax}")
    ax_force.set_ylabel("Max Force (eV/Ang)", color=force_color)
    ax_force.tick_params(axis="y", labelcolor=force_color)

    # Energy relative to step 0 is negative for any relaxation that went
    # downhill, so this axis can only ever be linear.
    ax_energy.set_yscale("linear")

    # The force axis is log, matching the convergence figure's force panel: a
    # relaxation spans orders of magnitude in fmax, and a linear axis buries
    # every step after the first few. The band complicates that -- its lower
    # edge is clipped to zero wherever sigma >= fmax -- and matplotlib drops
    # non-positive vertices from a log axis with no warning, which does not
    # merely hide that edge but deforms the whole polygon. So the scale adapts
    # exactly as the sigma panel's does, for the same reason.
    force_values = traces["fmax"] + traces["force_lo"] + traces["force_hi"]
    positive_values = [value for value in force_values if value > 0]
    if not positive_values:
        # A converged, exactly-agreeing committee: nothing to put on a log
        # axis, and linear keeps the flat trace on screen.
        ax_force.set_yscale("linear")
    elif len(positive_values) < len(force_values):
        # symlog treats |y| <= linthresh linearly, so a band edge sitting at
        # exactly zero is drawn where it belongs instead of vanishing. It is
        # symmetric about zero by definition, though, and autoscaling then
        # offers decades of negative force -- a quantity that does not exist.
        ax_force.set_yscale("symlog", linthresh=min(positive_values))
        ax_force.set_ylim(bottom=0.0)
    else:
        ax_force.set_yscale("log")

    handles = ax_energy.get_legend_handles_labels()
    twin_handles = ax_force.get_legend_handles_labels()
    ax_energy.legend(handles[0] + twin_handles[0],
                     handles[1] + twin_handles[1],
                     loc="upper right", fontsize=8)

    ax_energy.set_title("Committee Energy and Force with Uncertainty")

    fig.tight_layout()
    # Below the axes, not inside them: on a real relaxation both traces flatten
    # into the bottom of the panel, which is where an in-axes note lands.
    fig.subplots_adjust(bottom=0.19)
    # Without this note a band opening from zero reads as a run that started
    # certain and got worse. Step 0 is where each member's offset is measured,
    # so the spread there is zero by construction, not by agreement.
    fig.text(0.01, 0.015,
             "energy band is zero at step 0 by construction: that step is "
             "where each member's energy offset is measured",
             ha="left", va="bottom", fontsize=7, style="italic")
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
    uncertainty_plot: bool = False,
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
    uncertainty_plot : bool
        If True, write the ``*_uncertainty.png`` figure: committee mean energy
        with its spread band on the primary y-axis, max force with its
        ``sigma_max_free`` band on the secondary. Independent of ``plot`` --
        either, both or neither. Ignored without a committee, since there is
        no disagreement to draw; the CLI rejects that combination outright
        rather than leaving a caller waiting for a figure. Nothing is written
        when the trace has fewer than two steps.
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
        sigma_max_free above which the final configuration is flagged as
        high-disagreement. No default: same-level committees disagree by
        0.11-0.15 eV/Ang against typical fmax targets of 0.02-0.05, so
        defaulting to fmax flagged ordinary healthy relaxations. With no
        threshold, sigma is still reported but no verdict is asserted
        (``threshold_source: "none"``, ``flagged: None``). The value applied
        and where it came from are both recorded.

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
    uncertainty_plot_path = output_path / f"{logfile_stem}_uncertainty.png"
    final_structure = output_path / f"{logfile_stem}_final.vasp"
    # CONTCAR mirror of the final structure so a follow-up DFT run managed by
    # asetools can restart from this directory (it reads OUTCAR or CONTCAR).
    contcar_file = output_path / "CONTCAR"

    committee_csv = output_path / f"{logfile_stem}_committee.csv"
    committee_peratom_csv = output_path / f"{logfile_stem}_committee_peratom.csv"

    # No default. A same-level committee disagrees by 0.11-0.15 eV/A against
    # convergence targets of 0.02-0.05, so defaulting to fmax flagged
    # ordinary healthy relaxations; and those numbers predate the exclusion
    # of constrained atoms from sigma, so they are not a calibration either.
    # The run reports sigma and its ratio to fmax; a verdict is asserted only
    # when a caller chooses a threshold. See the 2026-09-08 design note.
    threshold = (float(uncertainty_threshold)
                 if uncertainty_threshold is not None else None)
    threshold_source = ("explicit" if uncertainty_threshold is not None
                        else "none")

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
        # is safe: every reader guards on `latest is not None`
        # (`log_convergence` and the summary block at the end), and
        # `uncertainty_summary(rows=[], latest=None)` returns all-null with
        # `flagged: None` -- no verdict, which is the honest output when
        # nothing was ever evaluated.
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
            "uncertainty_plot": uncertainty_plot,
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
        # The summary is written either way -- `uncertainty_summary` turns a
        # `latest` of None into an honest all-null block -- so the record is
        # always completed. Everything else here reads the final evaluation
        # and needs it to exist.
        summary = uncertainty_summary(
            trace_writer.rows, committee.latest, threshold=threshold,
            threshold_source=threshold_source,
            symbols=atoms.get_chemical_symbols())
        results["committee_uncertainty"] = summary
        committee.latest_uncertainty_summary = summary
        if committee.latest is not None:
            write_peratom_sigma(
                committee_peratom_csv,
                atoms.get_chemical_symbols(),
                committee.latest["sigma_per_atom_all"],
                sigma_free=committee.latest["sigma_per_atom_free"],
                free_mask=committee.latest["free_mask"])
            # sigma is reported unconditionally now -- there is no default
            # threshold to compare it against, so a caller with no opinion
            # about what counts as "too much disagreement" still learns the
            # number. A verdict (the second line) prints only when a
            # threshold was actually applied and it was exceeded.
            logger.info(
                "Committee disagreement at the final geometry: "
                "sigma_max_free = "
                "%.4f eV/Ang over %s free atoms (worst: %s #%s), "
                "sigma_mean_free = %.4f eV/Ang.",
                summary["sigma_max_free_final_eV_per_A"],
                summary["n_free_atoms"],
                summary["worst_atom_free_symbol"], summary["worst_atom_free"],
                summary["sigma_mean_free_final_eV_per_A"])
            if summary["flagged"]:
                # INFO, not WARNING: with no logging configured anywhere in
                # this codebase (confirmed by grep for basicConfig/addHandler/
                # setLevel/dictConfig/fileConfig), a WARNING-level record
                # reaches the terminal on its own via `logging.lastResort` --
                # printing the same message the CLI already echoes from
                # `results`. This line stays for anyone running with verbose
                # logging configured; the CLI echo (reading
                # `committee.latest_uncertainty_summary`, not this call) is
                # the one terminal report at default settings.
                logger.info(
                    "sigma_max_free exceeds the chosen threshold %.4f eV/Ang; "
                    "this configuration deserves a DFT check.", threshold)

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

    # The uncertainty figure needs a committee to have something to draw, and
    # at least two steps to draw it against.
    if uncertainty_plot and trace_writer is not None:
        figure = _plot_uncertainty(trace_writer.rows, fmax)
        if figure is None:
            logger.info(
                "uncertainty plot skipped: the committee trace has %d step(s), "
                "and a band needs at least two.", len(trace_writer.rows))
        else:
            figure.savefig(uncertainty_plot_path, dpi=150)
            plt.close(figure)

    return converged
