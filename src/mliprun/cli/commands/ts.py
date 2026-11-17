"""Transition-state CLI command: verify a TS by vibrational analysis."""
from pathlib import Path

import typer
from ase.io import read

from mliprun.cli.utils import (
    DEVICE_HELP,
    MACE_HEAD_HELP,
    MLIP_HELP,
    SEVENNET_TASK_HELP,
    UMA_TASK_HELP,
    _resolve_device,
    build_calculator,
    detect_mlip,
    param_sources_from_ctx,
    reject_aqcat_unsupported,
    validate_mlip,
)
from mliprun.core.run_record import RunContext
from mliprun.core.ts_verify import (
    CHECKS_BY_CHOICE,
    load_neb_for_verification,
    neb_tangent,
    verify_ts,
)
from mliprun.core.vibrations import (
    IMAGINARY_FLOOR_CM1,
    VALID_NFREE,
    FrequencyCacheError,
    parse_indices,
)

app = typer.Typer(help="Transition-state tools.")


def echo_ts_verification(results: dict, output_dir: Path) -> None:
    """Print the verdict, its reason, the checks and the report path."""
    typer.echo(f"\nTS verification: {results['verdict']}")
    typer.echo(f"   Reason: {results['reason']}")
    if "archived_previous" in results:
        typer.echo(f"   A verification of a different TS geometry was found "
                   f"here and moved to {results['archived_previous']}/.")
    typer.echo(f"   Imaginary modes: {results['n_imaginary']}"
               + (" (" + ", ".join(f"{f:.1f}i cm⁻¹" for f
                                   in results["imaginary_frequencies_cm-1"])
                  + ")" if results["n_imaginary"] else ""))
    typer.echo(f"   TS fmax (free components): "
               f"{results['ts_fmax_free_eV_per_A']:.4f} eV/Å "
               f"(limit {results['ts_fmax_limit_eV_per_A']:.4f})")
    overlap = results.get("overlap")
    if overlap is not None:
        value = ("not computed" if overlap["value"] is None
                 else f"{overlap['value']:.4f}")
        typer.echo(f"   Overlap with {overlap['reference']}: {value} "
                   f"(threshold {overlap['threshold']:.2f}) -> "
                   f"{overlap['status']}")
    relax = results.get("relax")
    if relax is not None:
        typer.echo(f"   Displace-and-relax: "
                   f"{relax['result'] or relax['status']}")
        if "note" in relax:
            typer.echo(f"   Note: {relax['note']}")
    typer.echo(f"\n📄 {(output_dir / 'ts_verify.txt').resolve()}")


@app.callback()
def _callback() -> None:
    """No shared setup: this exists only to keep ``verify`` addressable by name.

    A ``typer.Typer`` with exactly one command collapses onto that command
    when invoked directly, which would swallow the word ``verify`` as a stray
    argument (same reason as in ``freq.py``).
    """


def _fail(message: str) -> None:
    typer.echo(f"❌ {message}")
    raise typer.Exit(1)


@app.command()
def verify(
    ctx: typer.Context,
    neb_dir: Path = typer.Option(
        None, "--neb-dir",
        help="A finished NEB folder (A2B.traj, neb_parameters.txt). The TS "
             "is its highest-energy image, the reference direction is the "
             "NEB tangent there, and the model is the one the NEB used. "
             "Mutually exclusive with --ts/--initial/--final."),
    ts: Path = typer.Option(None, "--ts", help="TS structure file."),
    initial: Path = typer.Option(None, "--initial",
                                 help="Initial-state structure file."),
    final: Path = typer.Option(None, "--final",
                               help="Final-state structure file."),
    check: str = typer.Option(
        "overlap", "--check",
        help="overlap (mode vs reference direction), relax (displace along "
             "the mode and relax both ways) or both."),
    mlip: str = typer.Option(
        None, help=MLIP_HELP + " With --neb-dir the default is the NEB's "
                               "own model."),
    uma_task: str = typer.Option(None, help=UMA_TASK_HELP),
    mace_head: str = typer.Option(None, help=MACE_HEAD_HELP),
    sevennet_task: str = typer.Option(None, help=SEVENNET_TASK_HELP),
    device: str = typer.Option(None, help=DEVICE_HELP),
    indices: str = typer.Option(
        None, "--indices",
        help="Atoms to displace, e.g. '0,1,5' or '12-30'. Default: every "
             "atom not held by a FixAtoms constraint."),
    delta: float = typer.Option(0.01, help="Displacement in Å for the "
                                           "finite-difference Hessian"),
    nfree: int = typer.Option(2, help="2 (three-point) or 4 (five-point)"),
    imaginary_floor: float = typer.Option(
        IMAGINARY_FLOOR_CM1, "--imaginary-floor",
        help="cm⁻¹. A negative-curvature mode counts as imaginary only "
             "above this magnitude."),
    ts_fmax_limit: float = typer.Option(
        0.03, "--ts-fmax-limit",
        help="eV/Å. The TS is not a stationary point above this fmax over "
             "the free components."),
    overlap_threshold: float = typer.Option(
        0.7, "--overlap-threshold",
        help="Minimum |overlap| of the imaginary mode with the reference."),
    displacement: float = typer.Option(
        0.1, "--displacement",
        help="Å. Size of the displacement along the mode (relax check)."),
    match_tolerance: float = typer.Option(
        0.3, "--match-tolerance",
        help="Å. A relaxed structure matches an endpoint within this "
             "distance (relax check)."),
    relax_fmax: float = typer.Option(
        0.01, "--relax-fmax", help="eV/Å. Relaxation convergence (relax check)."),
    relax_optimizer: str = typer.Option(
        "bfgs", "--relax-optimizer", help="bfgs, lbfgs or fire."),
    relax_max_steps: int = typer.Option(
        200, "--relax-max-steps", help="Step limit per relaxation."),
    output_dir: Path = typer.Option(
        None, "--output-dir",
        help="Output directory. Default: <neb-dir>/ts_verify, or ts_verify "
             "beside the TS file."),
):
    """Verify a transition state: one imaginary mode along the reaction path."""
    files = {"--ts": ts, "--initial": initial, "--final": final}
    given = [name for name, value in files.items() if value is not None]
    if neb_dir is not None and given:
        _fail(f"--neb-dir cannot be combined with {', '.join(given)}.")
    if neb_dir is None and not given:
        _fail("Give either --neb-dir, or all of --ts, --initial and --final.")
    if neb_dir is None and len(given) != 3:
        missing = [name for name, value in files.items() if value is None]
        _fail(f"--ts, --initial and --final go together; missing "
              f"{', '.join(missing)}.")
    if check not in CHECKS_BY_CHOICE:
        _fail(f"--check must be one of {', '.join(CHECKS_BY_CHOICE)}; got {check!r}.")
    if nfree not in VALID_NFREE:
        _fail(f"--nfree must be 2 or 4, got {nfree}.")

    # Everything below that can fail on user input runs before a model loads.
    reference_direction = None
    reference_label = None
    extra_inputs = None
    if neb_dir is not None:
        try:
            loaded = load_neb_for_verification(neb_dir)
        except ValueError as exc:
            _fail(str(exc))
        images = loaded["images"]
        ts_index = loaded["ts_index"]
        params = loaded["params"]
        typer.echo(f"📂 NEB band: {len(images)} images in {neb_dir}; "
                   f"TS is image {ts_index}")
        if loaded["last_stage_status"] not in (None, "converged"):
            typer.echo(f"⚠️  The last NEB stage did not finish converged "
                       f"(status: {loaded['last_stage_status']}; not "
                       f"converged): the highest image may not be a saddle. "
                       f"Continuing.")
        if loaded["last_stage_climb"] is False:
            typer.echo("⚠️  The last NEB stage was not climbing: the highest "
                       "image is not driven to the saddle. Continuing.")
        reference_direction = neb_tangent(images, ts_index)
        reference_label = "neb_tangent"
        ts_atoms = images[ts_index].copy()
        initial_atoms = images[0].copy()
        final_atoms = images[-1].copy()
        mlip = mlip or params["mlip"]
        uma_task = uma_task or params.get("uma_task")
        mace_head = mace_head or params.get("mace_head")
        sevennet_task = sevennet_task or params.get("sevennet_task")
        device = device or params.get("device") or "auto"
        extra_inputs = {
            "neb_dir": str(neb_dir.resolve()),
            "ts_image_index": ts_index,
            "n_images": len(images),
        }
        default_output = neb_dir / "ts_verify"
    else:
        ts_atoms = read(ts)
        initial_atoms = read(initial)
        final_atoms = read(final)
        mlip = mlip or "auto"
        device = device or "auto"
        extra_inputs = {
            "ts": str(ts.resolve()),
            "initial": str(initial.resolve()),
            "final": str(final.resolve()),
        }
        default_output = ts.parent / "ts_verify"

    try:
        chosen = parse_indices(indices, n_atoms=len(ts_atoms))
    except ValueError as exc:
        _fail(str(exc))

    reject_aqcat_unsupported(mlip, "ts verify")
    if mlip == "auto":
        mlip = detect_mlip()
        typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
    else:
        typer.echo(f"🧠 Using MLIP: {mlip}")
    validate_mlip(mlip, sevennet_task, uma_task, mace_head)

    output_dir = output_dir if output_dir is not None else default_output
    run_context = RunContext(
        command="ts-verify",
        mode="one-off",
        param_sources=param_sources_from_ctx(ctx),
    )

    def calculator_factory():
        return build_calculator(mlip, uma_task, device=device,
                                mace_head=mace_head,
                                sevennet_task=sevennet_task)

    typer.echo(f"⚙️  Verifying with {mlip} (device={device}), "
               f"check={check}...")
    try:
        results = verify_ts(
            ts_atoms, initial_atoms, final_atoms,
            calculator_factory=calculator_factory,
            checks=CHECKS_BY_CHOICE[check],
            reference_direction=reference_direction,
            reference_label=reference_label,
            output_dir=output_dir,
            model_name=mlip,
            indices=chosen,
            delta=delta,
            nfree=nfree,
            imaginary_floor=imaginary_floor,
            ts_fmax_limit=ts_fmax_limit,
            overlap_threshold=overlap_threshold,
            displacement=displacement,
            match_tolerance=match_tolerance,
            relax_fmax=relax_fmax,
            relax_optimizer=relax_optimizer,
            relax_max_steps=relax_max_steps,
            provenance={
                "device_requested": device,
                "device_resolved": _resolve_device(device),
                "uma_task": uma_task,
                "mace_head": mace_head,
                "sevennet_task": sevennet_task,
            },
            extra_inputs=extra_inputs,
            run_context=run_context,
        )
    except (FrequencyCacheError, ValueError) as exc:
        _fail(str(exc))

    echo_ts_verification(results, output_dir)

if __name__ == "__main__":
    app()
