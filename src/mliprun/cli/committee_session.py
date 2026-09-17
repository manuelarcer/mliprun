"""Committee lifecycle for CLI commands: startup, teardown, and reporting.

Extracted from ``cli/commands/optimize.py`` so that ``optimize``,
``singlepoint`` and ``freq`` share one copy. The SIGTERM path here is the fix
from PR #47: a plain ``kill`` on a driver used to leave every worker running
and holding a CUDA context on a node with no scheduler to reap them. Three
copies of that fix would be three chances for it to drift.
"""
import contextlib
import signal
import threading
import typer
from pathlib import Path
from mliprun.core.committee.calculator import CommitteeCalculator, CommitteeError
from mliprun.core.committee.remote import MemberError, RemoteMember
from mliprun.cli.utils import param_sources_from_ctx


#: Model-selection options the committee file owns. Passing any of them with
#: --committee is an error rather than a silent precedence rule. ``device``
#: is included because each member's device comes from committee.yaml
#: (``spec.device``); a stray ``--device`` here would silently do nothing.
COMMITTEE_CONFLICTS = ("mlip", "uma_task", "mace_head", "sevennet_task",
                        "device")


def reject_conflicting_options(ctx) -> None:
    """Fail if a model-selection option was typed alongside --committee.

    Checked against the *parameter source*, not the value: --mlip defaults to
    'auto', so comparing values would either miss a typed 'auto' or reject
    every committee run.
    """
    sources = param_sources_from_ctx(ctx)
    typed = [name for name in COMMITTEE_CONFLICTS
             if sources.get(name) == "user"]
    if typed:
        flags = ", ".join("--" + name.replace("_", "-") for name in typed)
        typer.echo(
            f"❌ {flags} cannot be combined with --committee.\n"
            f"   The committee file declares the model, task and head for "
            f"every member; a flag here would silently do nothing.")
        raise typer.Exit(1)


def build_committee(config, output_dir: Path,
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


class Terminated(KeyboardInterrupt):
    """SIGTERM, raised in the main thread so the stack actually unwinds.

    A subclass of ``KeyboardInterrupt`` deliberately: Ctrl+C teardown is the
    path that is already tested and was re-confirmed on cos-cluster, so
    SIGTERM joins it rather than opening a second one.
    """


@contextlib.contextmanager
def sigterm_as_interrupt():
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
        raise Terminated("terminated by SIGTERM")

    previous = signal.signal(signal.SIGTERM, _raise)
    try:
        yield
    except Terminated:
        # Teardown already ran, on the way out of the inner block.
        typer.echo("\n⛔ Terminated (SIGTERM). Committee workers shut down.")
        raise typer.Exit(143) from None
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextlib.contextmanager
def started_committee(config, output_dir: Path, member_timeout: float,
                       atoms):
    """Start every member, and guarantee teardown on every exit path.

    Every path out of this block closes the workers: a failed load, a failed
    preflight, an error mid-relaxation, Ctrl+C, and -- through
    :func:`sigterm_as_interrupt` -- a plain ``kill``. The guard is armed
    before the first worker is spawned, because a model load is a long
    window (22.8 s for UMA in the 2026-09-04 probe) in which a member is
    already holding GPU memory.
    """
    with sigterm_as_interrupt():
        calc = build_committee(config, output_dir, member_timeout)
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


def report_committee_uncertainty(committee_calc) -> None:
    """Echo the committee's disagreement at the final geometry.

    Printed on every run that finishes, threshold or not: with no threshold
    there is no verdict to give, and the numbers are the deliverable. The
    warning below it appears only when a caller chose a threshold and the
    run exceeded it. A run that *failed* never reaches here -- the summary
    is written to the run record and the exception re-raised before this
    call -- so a failed run's numbers are in `mliprun_run.json`, not on the
    terminal.

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
    if summary is None or summary["sigma_max_free_final_eV_per_A"] is None:
        return
    ratio = summary["sigma_max_free_over_fmax_final"]
    # Printed unconditionally, so it is worth knowing where it is not
    # like-for-like: on a --relax-cell run the denominator carries the cell
    # virials the optimizer converges against, while sigma is atomic forces
    # only. See "The flagging rule" in docs/OUTPUTS.md.
    ratio_text = "" if ratio is None else f", {ratio:.1f}x the final fmax"
    typer.echo(
        f"\n📊 Committee disagreement at the final geometry: "
        f"sigma_max_free = "
        f"{summary['sigma_max_free_final_eV_per_A']:.4f} eV/Å"
        f"{ratio_text}, sigma_mean_free = "
        f"{summary['sigma_mean_free_final_eV_per_A']:.4f} eV/Å over "
        f"{summary['n_free_atoms']} free atoms. Worst atom: "
        f"{summary['worst_atom_free_symbol']} "
        f"(#{summary['worst_atom_free']}).")
    if summary["unhandled_constraints"]:
        typer.echo(
            f"   Note: constraint type(s) "
            f"{', '.join(summary['unhandled_constraints'])} are not masked, "
            f"so sigma is over-reported for their atoms.")
    if summary["flagged"]:
        typer.echo(
            f"\n⚠️  sigma_max_free exceeds the threshold you set "
            f"({summary['threshold_eV_per_A']:.4f} eV/Å). The located "
            f"minimum sits inside the committee's own noise; this "
            f"configuration deserves a DFT check.")
