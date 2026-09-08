import contextlib
import csv
import signal
import sys
import threading
import time
import traceback
import typer
from pathlib import Path
from ase.io import read
from mliprun.core.optimize import run_optimization, OPTIMIZER_MAP
from mliprun.core.run_record import BatchInfo, RunContext, new_batch_id
from mliprun.core.committee.calculator import CommitteeCalculator, CommitteeError
from mliprun.core.committee.config import CommitteeConfigError, load_committee
from mliprun.core.committee.remote import (
    DEFAULT_CALC_TIMEOUT_S,
    MemberError,
    RemoteMember,
)
from mliprun.cli.utils import (
    DEVICE_HELP,
    MACE_HEAD_HELP,
    MLIP_HELP,
    PLOT_HELP,
    UMA_TASK_HELP,
    _resolve_device,
    build_calculator,
    detect_mlip,
    param_sources_from_ctx,
    setup_calculator,
    validate_mlip,
    SEVENNET_TASK_HELP,
)

app = typer.Typer(help="Run geometry optimization on structures.")


def _find_input_structure(subdir: Path, pattern: str) -> Path:
    """Return the single input structure in ``subdir`` matching ``pattern``.

    The platform's own optimization outputs (``*_final.vasp``) are excluded so
    a batch can be safely re-run or resumed without the output being mistaken
    for a fresh input. Raises ``ValueError`` if zero or more than one candidate
    remains.
    """
    candidates = [
        p for p in sorted(subdir.glob(pattern))
        if p.is_file() and not p.name.endswith("_final.vasp")
    ]
    if not candidates:
        raise ValueError(f"no input structure matching '{pattern}'")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(
            f"multiple structures match '{pattern}' ({names}); "
            f"narrow --input-name to pick one"
        )
    return candidates[0]


#: Model-selection options the committee file owns. Passing any of them with
#: --committee is an error rather than a silent precedence rule. ``device``
#: is included because each member's device comes from committee.yaml
#: (``spec.device``); a stray ``--device`` here would silently do nothing.
_COMMITTEE_CONFLICTS = ("mlip", "uma_task", "mace_head", "sevennet_task",
                        "device")


def _reject_conflicting_options(ctx) -> None:
    """Fail if a model-selection option was typed alongside --committee.

    Checked against the *parameter source*, not the value: --mlip defaults to
    'auto', so comparing values would either miss a typed 'auto' or reject
    every committee run.
    """
    sources = param_sources_from_ctx(ctx)
    typed = [name for name in _COMMITTEE_CONFLICTS
             if sources.get(name) == "user"]
    if typed:
        flags = ", ".join("--" + name.replace("_", "-") for name in typed)
        typer.echo(
            f"❌ {flags} cannot be combined with --committee.\n"
            f"   The committee file declares the model, task and head for "
            f"every member; a flag here would silently do nothing.")
        raise typer.Exit(1)


def _build_committee(config, output_dir: Path,
                     member_timeout: float) -> CommitteeCalculator:
    """Turn a parsed committee.yaml into a started CommitteeCalculator.

    Each member gets its own worker log next to the run's other outputs --
    that is where library banners and remote tracebacks land, and every error
    message points at it.
    """
    members = [
        RemoteMember(
            spec.name,
            spec.python_exe,
            mlip=spec.mlip,
            uma_task=spec.uma_task,
            mace_head=spec.mace_head,
            sevennet_task=spec.sevennet_task,
            device=spec.device,
            gpu=spec.gpu,
            log_path=output_dir / f"committee_{spec.name}.log",
            timeout=member_timeout,
        )
        for spec in config.members
    ]
    return CommitteeCalculator(members, mixed_theory=config.mixed_theory,
                               levels=config.levels)


class _Terminated(KeyboardInterrupt):
    """SIGTERM, raised in the main thread so the stack actually unwinds.

    A subclass of ``KeyboardInterrupt`` deliberately: Ctrl+C teardown is the
    path that is already tested and was re-confirmed on cos-cluster, so
    SIGTERM joins it rather than opening a second one.
    """


@contextlib.contextmanager
def _sigterm_as_interrupt():
    """Make SIGTERM unwind the stack instead of killing the process outright.

    SIGTERM's default disposition terminates the interpreter *without*
    unwinding, so a plain ``kill`` on a committee driver skips both the
    teardown around the run and the ``atexit`` backstop in
    ``committee.remote``. The workers are left running, holding their CUDA
    contexts, and cos-cluster has no scheduler to reap them (verified by pid
    on 2026-09-07: driver dead, two workers still holding 2948 MiB).

    The worker's own defence -- exiting when the driver closes its end of the
    pipe -- does not cover this, because a worker only reaches
    ``stdin.readline()`` *between* calculations. During a relaxation it
    spends nearly all of its wall clock inside the model's forward pass,
    where the driver's death is invisible to it.

    Installed here, in the CLI, and removed on the way out:
    ``run_optimization`` is also a Python API entry point, and must not
    change signal behaviour for a caller who installs their own handler.

    Exit code 143 is 128 + SIGTERM, so a shell still reads a terminated run
    as terminated rather than as a clean exit.
    """
    if threading.current_thread() is not threading.main_thread():
        # Only the main thread may install a handler, and only the main
        # thread ever runs one. A caller driving this command from a worker
        # thread keeps the process's existing disposition.
        yield
        return

    def _raise(signum, frame):
        raise _Terminated("terminated by SIGTERM")

    previous = signal.signal(signal.SIGTERM, _raise)
    try:
        yield
    except _Terminated:
        # Teardown already ran, on the way out of the inner block.
        typer.echo("\n⛔ Terminated (SIGTERM). Committee workers shut down.")
        raise typer.Exit(143) from None
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextlib.contextmanager
def _started_committee(config, output_dir: Path, member_timeout: float,
                       atoms):
    """Start every member, and guarantee teardown on every exit path.

    Every path out of this block closes the workers: a failed load, a failed
    preflight, an error mid-relaxation, Ctrl+C, and -- through
    :func:`_sigterm_as_interrupt` -- a plain ``kill``. The guard is armed
    before the first worker is spawned, because a model load is a long
    window (22.8 s for UMA in the 2026-09-04 probe) in which a member is
    already holding GPU memory.
    """
    with _sigterm_as_interrupt():
        calc = _build_committee(config, output_dir, member_timeout)
        try:
            typer.echo("⚙️  Starting committee members "
                       "(one model load each)...")
            try:
                calc.start()
                # Every member evaluates the input geometry once, up front:
                # members disagree about what input is valid (fairchem's UMA
                # calculator rejects a pbc=(T,T,F) slab that the others
                # accept), and that must surface now, not on step 400
                # tonight.
                calc.preflight(atoms)
            except (MemberError, CommitteeError) as exc:
                typer.echo(f"❌ {exc}")
                raise typer.Exit(1)
            yield calc
        finally:
            calc.close()


def _report_committee_uncertainty(committee_calc) -> None:
    """Echo the committee's disagreement at the final geometry.

    Always printed: with no threshold there is no verdict to give, and the
    numbers are the deliverable. The warning below it appears only when a
    caller chose a threshold and the run exceeded it.

    This is the single terminal report at default settings:
    ``run_optimization``'s own log calls are INFO-level (silent unless a
    caller configures logging below WARNING), precisely so this echo is not
    a duplicate of them.

    Reads ``committee_calc.latest_uncertainty_summary`` -- the exact
    ``uncertainty_summary(...)`` dict ``run_optimization`` already computed
    and stored in the run record -- rather than recomputing it, so the
    printed number can never diverge from the recorded one.
    """
    if committee_calc is None:
        return
    summary = committee_calc.latest_uncertainty_summary
    if summary is None or summary["sigma_max_final_eV_per_A"] is None:
        return
    ratio = summary["sigma_max_over_fmax_final"]
    ratio_text = "" if ratio is None else f", {ratio:.1f}x the final fmax"
    typer.echo(
        f"\n📊 Committee disagreement at the final geometry: "
        f"sigma_max = {summary['sigma_max_final_eV_per_A']:.4f} eV/Å"
        f"{ratio_text}, sigma_mean = "
        f"{summary['sigma_mean_final_eV_per_A']:.4f} eV/Å over "
        f"{summary['n_free_atoms']} free atoms. Worst atom: "
        f"{summary['worst_atom_symbol']} (#{summary['worst_atom']}).")
    if summary["unhandled_constraints"]:
        typer.echo(
            f"   Note: constraint type(s) "
            f"{', '.join(summary['unhandled_constraints'])} are not masked, "
            f"so sigma is over-reported for their atoms.")
    if summary["flagged"]:
        typer.echo(
            f"\n⚠️  sigma_max exceeds the threshold you set "
            f"({summary['threshold_eV_per_A']:.4f} eV/Å). The located "
            f"minimum sits inside the committee's own noise; this "
            f"configuration deserves a DFT check.")


@app.command()
def run(
    ctx: typer.Context,
    structure: Path = typer.Option(..., prompt=True, help="Structure file (.vasp)"),
    mlip: str = typer.Option("auto", help=MLIP_HELP),
    uma_task: str = typer.Option(None, help=UMA_TASK_HELP),
    device: str = typer.Option("auto", help=DEVICE_HELP),
    mace_head: str = typer.Option(None, help=MACE_HEAD_HELP),
    sevennet_task: str = typer.Option(None, help=SEVENNET_TASK_HELP),
    committee: Path = typer.Option(
        None, "--committee",
        help="Path to a committee.yaml declaring two or more MLIP members, "
             "each in its own env. Drives the relaxation with their mean "
             "force and reports their disagreement as a per-configuration "
             "uncertainty. Mutually exclusive with --mlip and the head/task "
             "options: the file owns model selection."),
    member_timeout: float = typer.Option(
        DEFAULT_CALC_TIMEOUT_S, "--member-timeout",
        help="Seconds a single committee member may take per single-point "
             "before it is killed and the run aborts. Model loading has its "
             "own, much longer budget."),
    uncertainty_threshold: float = typer.Option(
        None, "--uncertainty-threshold",
        help="Flag the final configuration when the committee's per-atom "
             "force disagreement exceeds this (eV/Å). NO DEFAULT: without "
             "it the run reports sigma and its ratio to fmax but asserts no "
             "verdict. There is no calibrated value yet -- same-level "
             "committees disagree by ~0.1 eV/Å. See docs/OUTPUTS.md."),
    optimizer: str = typer.Option("bfgs", help=f"Optimizer algorithm: {', '.join(OPTIMIZER_MAP.keys())}"),
    fmax: float = typer.Option(0.05, help="Force convergence threshold (eV/Å)"),
    max_steps: int = typer.Option(200, help="Maximum optimization steps"),
    relax_cell: bool = typer.Option(
        False, "--relax-cell",
        help=("Relax the simulation cell as well as positions (VASP "
              "ISIF=3-equivalent). Wraps atoms in ASE's FrechetCellFilter "
              "(or ExpCellFilter on older ASE)."),
    ),
    trajectory: str = typer.Option("opt.traj", help="Trajectory filename"),
    logfile: str = typer.Option("opt.log", help="Log filename"),
    verbose: bool = typer.Option(True, help="Show optimization progress table (forces, energies)"),
    plot: bool = typer.Option(False, "--plot/--no-plot", help=PLOT_HELP),
):
    """
    Run geometry optimization using a supported MLIP model.

    Optimizes atomic positions to minimize forces until fmax convergence is reached.
    """
    # Read structure
    atoms = read(structure)
    typer.echo(f"📂 Loaded structure: {structure.name}")
    typer.echo(f"   Atoms: {len(atoms)}, Formula: {atoms.get_chemical_formula()}")

    committee_config = None
    committee_calc = None
    if committee is not None:
        _reject_conflicting_options(ctx)
        try:
            committee_config = load_committee(committee)
        except CommitteeConfigError as exc:
            typer.echo(f"❌ {exc}")
            raise typer.Exit(1)
        mlip = "committee"
        typer.echo(f"🧠 Committee of {len(committee_config.members)} members "
                   f"from {committee}")
        for spec in committee_config.members:
            typer.echo(f"   {spec.name}: {spec.mlip} "
                       f"[{spec.level_of_theory}] gpu={spec.gpu} "
                       f"env={spec.env}")
        if committee_config.mixed_theory:
            typer.echo(f"\n⚠️  {committee_config.mixed_theory_warning()}\n")
    else:
        # Detect or validate MLIP
        if mlip == "auto":
            mlip = detect_mlip()
            typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
            # An auto-detected tag still has to satisfy its own task rules.
            validate_mlip(mlip, sevennet_task, uma_task, mace_head)
        else:
            validate_mlip(mlip, sevennet_task, uma_task, mace_head)
            typer.echo(f"🧠 Using MLIP: {mlip}")

    # Validate optimizer
    if optimizer.lower() not in OPTIMIZER_MAP:
        typer.echo(f"❌ Unknown optimizer: {optimizer}")
        typer.echo(f"   Available: {', '.join(OPTIMIZER_MAP.keys())}")
        raise typer.Exit(1)

    # Output directory
    output_dir = structure.parent

    # A leaked worker holds a CUDA context that makes the GPU look busy to
    # everyone else on the node, and cos-cluster has no scheduler to reap
    # orphans. So the committee's whole lifetime -- model loads included --
    # lives in one scope that closes it on every exit path, SIGTERM included.
    with contextlib.ExitStack() as committee_session:
        if committee_config is not None:
            committee_calc = committee_session.enter_context(
                _started_committee(committee_config, output_dir,
                                   member_timeout, atoms))
            atoms.calc = committee_calc
        else:
            # Assign calculator
            typer.echo(f"⚙️  Attaching {mlip} calculator (device={device})...")
            if mlip.startswith("uma-"):
                typer.echo(f"   UMA task: {uma_task}")
            if mlip.startswith("mace-mh-"):
                typer.echo(f"   MACE head: {mace_head}")
            if mlip.startswith("7net"):
                typer.echo(f"   SevenNet task: {sevennet_task}")
            atoms = setup_calculator(atoms, mlip, uma_task, device=device,
                                     mace_head=mace_head,
                                     sevennet_task=sevennet_task)

        # Run optimization
        typer.echo(f"\n🔧 Optimizer: {optimizer.upper()}")
        typer.echo(f"   fmax = {fmax} eV/Å")
        typer.echo(f"   max_steps = {max_steps}")
        typer.echo(f"   Output dir: {output_dir.resolve()}\n")

        run_context = RunContext(
            command="optimize",
            mode="one-off",
            param_sources=param_sources_from_ctx(ctx),
        )
        run_context.extra_inputs = {
            "structure": structure.name,
            "structure_abspath": str(structure.resolve()),
        }

        converged = run_optimization(
            atoms=atoms,
            optimizer=optimizer,
            fmax=fmax,
            max_steps=max_steps,
            trajectory=trajectory,
            logfile=logfile,
            output_dir=output_dir,
            model_name=mlip,
            verbose=verbose,
            relax_cell=relax_cell,
            plot=plot,
            run_context=run_context,
            device_requested=device,
            device_resolved=_resolve_device(device),
            uma_task=uma_task,
            mace_head=mace_head,
            sevennet_task=sevennet_task,
            committee=committee_calc,
            committee_config=committee_config,
            uncertainty_threshold=uncertainty_threshold,
        )

    # Save parameters
    _write_params(output_dir / "opt_params.txt", mlip, uma_task, mace_head,
                  device, relax_cell, structure.name, optimizer, fmax,
                  max_steps, converged, output_dir,
                  sevennet_task=sevennet_task,
                  committee_config=committee_config,
                  uncertainty_threshold=uncertainty_threshold)

    # Print output summary
    typer.echo("\n✅ Optimization complete. Output files:")

    # Extract prefix from logfile for convergence filenames
    logfile_stem = Path(logfile).stem
    output_files = [
        trajectory,
        logfile,
        f"{logfile_stem}_convergence.csv",
        f"{logfile_stem}_final.vasp",
        "CONTCAR",
        "opt_params.txt"
    ]
    if committee_config is not None:
        output_files.insert(3, f"{logfile_stem}_committee.csv")
        output_files.insert(4, f"{logfile_stem}_committee_peratom.csv")
    if plot:
        output_files.insert(3, f"{logfile_stem}_convergence.png")
    for file in output_files:
        typer.echo(f"   📄 {(output_dir / file).resolve()}")

    if not converged:
        typer.echo("\n⚠️  Warning: Optimization did not converge. Consider:")
        typer.echo("   - Increasing max_steps")
        typer.echo("   - Relaxing fmax threshold")
        typer.echo("   - Trying a different optimizer")

    _report_committee_uncertainty(committee_calc)


@app.command()
def batch(
    ctx: typer.Context,
    parent: Path = typer.Option(..., prompt=True,
        help="Parent directory; each immediate subdirectory holds one input structure."),
    input_name: str = typer.Option("*.vasp", "--input-name",
        help="Glob for the input structure inside each subdirectory. Default "
             "'*.vasp' expects exactly one .vasp file per subdir (the platform's "
             "own *_final.vasp outputs are ignored). Use e.g. 'POSCAR' or "
             "'init.vasp' for a fixed name."),
    mlip: str = typer.Option("auto", help=MLIP_HELP),
    uma_task: str = typer.Option(None, help=UMA_TASK_HELP),
    device: str = typer.Option("auto", help=DEVICE_HELP),
    mace_head: str = typer.Option(None, help=MACE_HEAD_HELP),
    sevennet_task: str = typer.Option(None, help=SEVENNET_TASK_HELP),
    optimizer: str = typer.Option("bfgs", help=f"Optimizer algorithm: {', '.join(OPTIMIZER_MAP.keys())}"),
    fmax: float = typer.Option(0.05, help="Force convergence threshold (eV/Å)"),
    max_steps: int = typer.Option(200, help="Maximum optimization steps"),
    relax_cell: bool = typer.Option(
        False, "--relax-cell",
        help="Relax the simulation cell as well as positions (applied to every structure)."),
    trajectory: str = typer.Option("opt.traj", help="Trajectory filename (per subdirectory)"),
    logfile: str = typer.Option("opt.log", help="Log filename (per subdirectory)"),
    skip_existing: bool = typer.Option(
        False, "--skip-existing",
        help="Skip subdirectories that already contain a CONTCAR (resume a partial batch)."),
    verbose: bool = typer.Option(False, help="Show per-structure optimization progress table"),
    plot: bool = typer.Option(False, "--plot/--no-plot", help=PLOT_HELP),
):
    """
    Relax a series of structures, loading the MLIP model only once.

    Discovers one input structure per immediate subdirectory of PARENT, builds
    the calculator a single time, and reuses it across every relaxation. Each
    structure is optimized in place (outputs written into its own subdirectory,
    exactly as ``optimize run``). A structure that errors or fails to converge
    is logged and the batch continues. A ``batch_summary.csv`` is written into
    PARENT.
    """
    if not parent.is_dir():
        typer.echo(f"❌ Not a directory: {parent}")
        raise typer.Exit(1)

    if optimizer.lower() not in OPTIMIZER_MAP:
        typer.echo(f"❌ Unknown optimizer: {optimizer}")
        typer.echo(f"   Available: {', '.join(OPTIMIZER_MAP.keys())}")
        raise typer.Exit(1)

    # Detect or validate MLIP once for the whole batch.
    if mlip == "auto":
        mlip = detect_mlip()
        typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
        # An auto-detected tag still has to satisfy its own task rules.
        validate_mlip(mlip, sevennet_task, uma_task, mace_head)
    else:
        validate_mlip(mlip, sevennet_task, uma_task, mace_head)
        typer.echo(f"🧠 Using MLIP: {mlip}")

    subdirs = sorted(d for d in parent.iterdir() if d.is_dir())
    if not subdirs:
        typer.echo(f"❌ No subdirectories found in {parent.resolve()}")
        raise typer.Exit(1)

    batch_info = BatchInfo(
        batch_id=new_batch_id(),
        driver="mliprun optimize batch",
        argv=list(sys.argv),
        root=str(parent.resolve()),
    )
    batch_sources = param_sources_from_ctx(ctx)

    # Build the calculator ONCE and reuse it for every structure. This is the
    # whole point of the batch command: the model load happens a single time.
    typer.echo(f"⚙️  Loading {mlip} calculator once (device={device})...")
    if mlip.startswith("uma-"):
        typer.echo(f"   UMA task: {uma_task}")
    if mlip.startswith("mace-mh-"):
        typer.echo(f"   MACE head: {mace_head}")
    if mlip.startswith("7net"):
        typer.echo(f"   SevenNet task: {sevennet_task}")
    calc = build_calculator(mlip, uma_task, device=device, mace_head=mace_head,
                            sevennet_task=sevennet_task)

    typer.echo(f"\n🔧 Optimizer: {optimizer.upper()} | fmax={fmax} eV/Å | max_steps={max_steps}")
    typer.echo(f"   {len(subdirs)} subdirectories under {parent.resolve()}\n")

    results = []
    for subdir in subdirs:
        if skip_existing and (subdir / "CONTCAR").exists():
            typer.echo(f"⏭️  {subdir.name}: CONTCAR present, skipping")
            results.append({"subdir": subdir.name, "status": "skipped",
                            "converged": "", "steps": "", "energy_eV": "",
                            "walltime_s": "", "detail": "CONTCAR present"})
            continue

        try:
            structure = _find_input_structure(subdir, input_name)
        except ValueError as exc:
            typer.echo(f"⚠️  {subdir.name}: {exc} — skipping")
            results.append({"subdir": subdir.name, "status": "no_input",
                            "converged": "", "steps": "", "energy_eV": "",
                            "walltime_s": "", "detail": str(exc)})
            continue

        typer.echo(f"▶️  {subdir.name}: relaxing {structure.name}")
        t0 = time.perf_counter()
        try:
            atoms = read(structure)
            atoms.calc = calc  # reuse the single loaded model

            run_context = RunContext(
                command="optimize",
                mode="batch",
                batch=batch_info,
                param_sources=batch_sources,
            )
            run_context.extra_inputs = {
                "structure": structure.name,
                "structure_abspath": str(structure.resolve()),
            }

            converged = run_optimization(
                atoms=atoms,
                optimizer=optimizer,
                fmax=fmax,
                max_steps=max_steps,
                trajectory=trajectory,
                logfile=logfile,
                output_dir=subdir,
                model_name=mlip,
                verbose=verbose,
                relax_cell=relax_cell,
                plot=plot,
                run_context=run_context,
                device_requested=device,
                device_resolved=_resolve_device(device),
                uma_task=uma_task,
                mace_head=mace_head,
                sevennet_task=sevennet_task,
            )
            walltime = time.perf_counter() - t0
            energy = atoms.get_potential_energy()

            _write_params(subdir / "opt_params.txt", mlip, uma_task, mace_head,
                          device, relax_cell, structure.name, optimizer, fmax,
                          max_steps, converged, subdir,
                          sevennet_task=sevennet_task)

            status = "converged" if converged else "not_converged"
            icon = "✅" if converged else "⚠️"
            typer.echo(f"   {icon} {status} in {walltime:.1f}s "
                       f"(E={energy:.4f} eV)")
            results.append({"subdir": subdir.name, "status": status,
                            "converged": converged, "steps": "",
                            "energy_eV": f"{energy:.6f}",
                            "walltime_s": f"{walltime:.2f}", "detail": ""})
        except Exception as exc:  # noqa: BLE001 -- continue-on-failure by design
            walltime = time.perf_counter() - t0
            typer.echo(f"   ❌ error: {exc}")
            traceback.print_exc()
            results.append({"subdir": subdir.name, "status": "error",
                            "converged": "", "steps": "", "energy_eV": "",
                            "walltime_s": f"{walltime:.2f}",
                            "detail": str(exc)})

    # Write batch summary.
    summary_path = parent / "batch_summary.csv"
    fieldnames = ["subdir", "status", "converged", "steps", "energy_eV",
                  "walltime_s", "detail"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    n_ok = sum(1 for r in results if r["status"] == "converged")
    n_total = len(results)
    typer.echo(f"\n📊 Batch complete: {n_ok}/{n_total} converged.")
    typer.echo(f"   Summary: {summary_path.resolve()}")


def _write_params(param_file, mlip, uma_task, mace_head, device, relax_cell,
                  structure_name, optimizer, fmax, max_steps, converged,
                  output_dir, sevennet_task=None, committee_config=None,
                  uncertainty_threshold=None):
    """Write the per-structure opt_params.txt (matches ``optimize run``)."""
    with open(param_file, "w", encoding="utf-8") as f:
        f.write("Geometry Optimization Parameters\n")
        f.write("=================================\n")
        f.write(f"MLIP model:        {mlip}\n")
        if committee_config is not None:
            f.write(f"Committee:         {committee_config.source_path}\n")
            f.write(f"  sha256:          {committee_config.sha256}\n")
            f.write(f"  mixed theory:    {committee_config.mixed_theory}\n")
            for spec in committee_config.members:
                f.write(f"  - {spec.name}: {spec.mlip} "
                        f"[{spec.level_of_theory}] gpu={spec.gpu} "
                        f"env={spec.env}\n")
            # Mirrors `threshold_source` in the run record (Task 5): no
            # default. Writing the removed fmax default here would put a
            # verdict nobody asked for into this artifact even though the
            # run record correctly recorded none.
            if uncertainty_threshold is None:
                f.write("Uncertainty thr.:  none\n")
            else:
                f.write(f"Uncertainty thr.:  {uncertainty_threshold} "
                        f"(explicit)\n")
        if mlip.startswith("uma-"):
            f.write(f"UMA task:          {uma_task}\n")
        if mlip.startswith("mace-mh-"):
            f.write(f"MACE head:         {mace_head}\n")
        if mlip.startswith("7net"):
            f.write(f"SevenNet task:     {sevennet_task}\n")
        f.write(f"Device:            {device}\n")
        f.write(f"Relax cell:        {relax_cell}\n")
        f.write(f"Structure:         {structure_name}\n")
        f.write(f"Optimizer:         {optimizer.upper()}\n")
        f.write(f"fmax (eV/Å):       {fmax}\n")
        f.write(f"Max steps:         {max_steps}\n")
        f.write(f"Converged:         {converged}\n")
        f.write(f"Output dir:        {output_dir.resolve()}\n")


if __name__ == "__main__":
    app()
