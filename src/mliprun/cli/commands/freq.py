"""Frequency CLI command: vibrational frequencies by finite differences."""
import contextlib
from pathlib import Path

import typer
from ase.io import read

from mliprun.cli.committee_session import (
    reject_conflicting_options,
    report_committee_uncertainty,
    started_committee,
)
from mliprun.cli.utils import (
    DEVICE_HELP,
    MACE_HEAD_HELP,
    MLIP_HELP,
    SEVENNET_TASK_HELP,
    UMA_TASK_HELP,
    _resolve_device,
    detect_mlip,
    param_sources_from_ctx,
    setup_calculator,
    validate_mlip,
)
from mliprun.core.committee.config import CommitteeConfigError, load_committee
from mliprun.core.committee.remote import DEFAULT_CALC_TIMEOUT_S
from mliprun.core.run_record import RunContext
from mliprun.core.vibrations import (
    VALID_DIRECTIONS,
    VALID_METHODS,
    VALID_NFREE,
    FrequencyCacheError,
    parse_indices,
    run_frequencies,
)

app = typer.Typer(
    help="Vibrational frequencies by finite differences of forces.")


@app.callback()
def _callback() -> None:
    """No shared setup: this exists only to keep ``run`` addressable by name.

    A ``typer.Typer`` with exactly one command collapses onto that command
    when invoked directly (no subcommand name needed) -- fine for ``mlip
    freq run ...`` since ``add_typer`` in ``cli/main.py`` always builds a
    real group there, but it would make a direct ``freq.app`` invocation (as
    in the unit tests) silently swallow the word ``run`` as a stray
    argument.
    """


@app.command()
def run(
    ctx: typer.Context,
    structure: Path = typer.Option(..., prompt=True,
                                   help="Structure file (.vasp)"),
    mlip: str = typer.Option("auto", help=MLIP_HELP),
    uma_task: str = typer.Option(None, help=UMA_TASK_HELP),
    device: str = typer.Option("auto", help=DEVICE_HELP),
    mace_head: str = typer.Option(None, help=MACE_HEAD_HELP),
    sevennet_task: str = typer.Option(None, help=SEVENNET_TASK_HELP),
    committee: Path = typer.Option(
        None, "--committee",
        help="Path to a committee.yaml. Builds one Hessian per member from "
             "a single displacement sweep and reports each member's "
             "frequencies plus the per-mode spread. Mutually exclusive with "
             "--mlip and the head/task options."),
    member_timeout: float = typer.Option(
        DEFAULT_CALC_TIMEOUT_S, "--member-timeout",
        help="Seconds a single committee member may take per single-point "
             "before it is killed and the run aborts."),
    uncertainty_threshold: float = typer.Option(
        None, "--uncertainty-threshold",
        help="Flag the input geometry when the committee's force "
             "disagreement over the free atoms exceeds this (eV/Å). NO "
             "DEFAULT: without it sigma is reported and no verdict "
             "asserted."),
    indices: str = typer.Option(
        None, "--indices",
        help="Atoms to displace, e.g. '0,1,5' or '12-30' (inclusive), or a "
             "mix. Default: every atom not held by a FixAtoms constraint, "
             "which is the structure's own answer."),
    delta: float = typer.Option(0.01, help="Displacement in Å"),
    nfree: int = typer.Option(
        2, help="2 (three-point) or 4 (five-point stencil, twice the cost)"),
    direction: str = typer.Option(
        "central", help="central, forward or backward"),
    method: str = typer.Option(
        "standard",
        help="standard, or frederiksen for the acoustic sum-rule correction"),
    write_modes: str = typer.Option(
        "imaginary", "--write-modes",
        help="Which modes get an animated trajectory: none, imaginary "
             "(default) or all."),
    expect_fmax: float = typer.Option(
        None, "--expect-fmax",
        help="Warn when fmax at the input geometry exceeds this (eV/Å). "
             "Never stops the run. Default: the fmax a converged optimize "
             "stage in the structure's directory actually met, if there is "
             "one."),
    output_dir: Path = typer.Option(
        None, "--output-dir",
        help="Directory for this run's outputs. Default: next to the input "
             "structure. Use it to keep a frequency run out of the "
             "optimization folder that produced the structure — a run record "
             "is replaced by the next command that writes in the same "
             "directory."),
    prefix: str = typer.Option("freq",
                               help="Stem for this command's output files"),
):
    """Compute vibrational frequencies and the zero-point energy."""
    atoms = read(structure)
    typer.echo(f"📂 Loaded structure: {structure.name}")
    typer.echo(f"   Atoms: {len(atoms)}, "
               f"Formula: {atoms.get_chemical_formula()}")

    # Validated here rather than deep in the engine, so a typo costs nothing
    # instead of failing after the first displacement.
    if nfree not in VALID_NFREE:
        typer.echo(f"❌ --nfree must be 2 or 4, got {nfree}.")
        raise typer.Exit(1)
    if direction not in VALID_DIRECTIONS:
        typer.echo(f"❌ --direction must be one of "
                   f"{', '.join(VALID_DIRECTIONS)}; got {direction!r}.")
        raise typer.Exit(1)
    if method not in VALID_METHODS:
        typer.echo(f"❌ --method must be one of "
                   f"{', '.join(VALID_METHODS)}; got {method!r}.")
        raise typer.Exit(1)
    if write_modes not in ("none", "imaginary", "all"):
        typer.echo("❌ --write-modes must be none, imaginary or all; "
                   f"got {write_modes!r}.")
        raise typer.Exit(1)
    try:
        chosen = parse_indices(indices, n_atoms=len(atoms))
    except ValueError as exc:
        typer.echo(f"❌ {exc}")
        raise typer.Exit(1)

    committee_config = None
    committee_calc = None
    if committee is not None:
        reject_conflicting_options(ctx)
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
        if mlip == "auto":
            mlip = detect_mlip()
            typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
        else:
            typer.echo(f"🧠 Using MLIP: {mlip}")
        validate_mlip(mlip, sevennet_task, uma_task, mace_head)

    # Separate from the structure's own directory on purpose: a prior run
    # record (e.g. from the optimize stage that produced this structure)
    # stays where it is, and this run writes elsewhere. `structure_dir`
    # (passed to run_frequencies below, for the fmax-expectation lookup)
    # stays structure.parent regardless of --output-dir.
    output_dir = output_dir if output_dir is not None else structure.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = RunContext(
        command="freq",
        mode="one-off",
        param_sources=param_sources_from_ctx(ctx),
    )
    run_context.extra_inputs = {
        "structure": structure.name,
        "structure_abspath": str(structure.resolve()),
    }

    with contextlib.ExitStack() as session:
        if committee_config is not None:
            committee_calc = session.enter_context(
                started_committee(committee_config, output_dir,
                                  member_timeout, atoms))
            atoms.calc = committee_calc
        else:
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

        typer.echo(f"\n〰️  Displacing "
                   f"{len(chosen) if chosen is not None else 'the free'} "
                   f"atoms, delta = {delta} Å, nfree = {nfree}\n")

        # A cache this run must not trust is a user-fixable situation, not a
        # bug: the message already names the directory and both remedies, so
        # print it rather than letting a traceback carry it. `run_frequencies`
        # has already completed the run record as `failed` by this point.
        try:
            results = run_frequencies(
                atoms=atoms,
                output_dir=output_dir,
                prefix=prefix,
                model_name=mlip,
                indices=chosen,
                delta=delta,
                nfree=nfree,
                direction=direction,
                method=method,
                write_modes=write_modes,
                expect_fmax=expect_fmax,
                structure_dir=structure.parent,
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
        except FrequencyCacheError as exc:
            typer.echo(f"\n❌ {exc}")
            raise typer.Exit(1)

    typer.echo(f"\n〰️  {results['n_modes']} modes over "
               f"{results['n_displaced_atoms']} displaced atoms "
               f"({results['n_force_calls']} force calls)")
    typer.echo(f"   ZPE: {results['zpe_eV']:.6f} eV")
    # Both populations, never one: the free value is what the expectation
    # below is compared against (it is the constrained criterion an
    # `optimize` stage converged against), while the all-atom value is what
    # the model predicts before anything is held fixed. On a slab with
    # frozen layers they differ by an order of magnitude.
    typer.echo(f"   fmax at the input geometry: "
               f"{results['fmax_at_input_free_eV_per_A']:.6f} eV/Å over the "
               f"free components, "
               f"{results['fmax_at_input_all_eV_per_A']:.6f} eV/Å over all")
    if results["fmax_expectation_source"] == "none":
        typer.echo("   (no relaxation provenance next to this structure, so "
                   "nothing to compare it against)")
    elif results["fmax_warning"]:
        typer.echo(
            f"\n⚠️  That is above the {results['fmax_expectation']:.6f} eV/Å "
            f"expected from {results['fmax_expectation_source']}. A geometry "
            f"that is not a stationary point produces spurious imaginary "
            f"modes; the run continued.")
    if results["n_imaginary"]:
        typer.echo(f"\n⚠️  {results['n_imaginary']} imaginary mode(s). ZPE "
                   f"above counts the real modes only.")
    if results["unhandled_constraints"]:
        typer.echo(
            f"\n⚠️  Constraint type(s) "
            f"{', '.join(results['unhandled_constraints'])} could not be "
            f"honoured: ASE selects whole atoms, so those atoms were "
            f"displaced in full and their held components are in the "
            f"Hessian as if free. Use --indices to exclude them.")

    typer.echo("\n✅ Frequencies complete. Output files:")
    written = [f"{prefix}_frequencies.csv", f"{prefix}_summary.txt",
               f"{prefix}_vibrations.json", f"{prefix}_cache.json",
               "mliprun_run.json"]
    if committee_config is not None:
        written.insert(1, f"{prefix}_committee_frequencies.csv")
    for name in written:
        typer.echo(f"   📄 {(output_dir / name).resolve()}")
    # Globbed, not predicted: --write-modes imaginary on a clean minimum
    # writes nothing, and a listing that names a file which is not there is
    # worse than a short listing.
    for path in sorted(output_dir.glob(f"{prefix}.*.traj")):
        typer.echo(f"   📄 {path.resolve()}")
    # Deleting the directory alone is enough: with no entries left,
    # `{prefix}_cache.json` is rewritten with the next run's own identity.
    typer.echo(f"   📁 {(output_dir / prefix).resolve()}  "
               f"(displacement cache — delete to force a full recompute; "
               f"{prefix}_cache.json records what it was swept under)")

    report_committee_uncertainty(committee_calc, "the input geometry")


if __name__ == "__main__":
    app()
