"""Single-point CLI command: evaluate one structure and stop."""
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
from mliprun.core.singlepoint import run_singlepoint

app = typer.Typer(help="Evaluate a structure once: energy, forces, stress.")


@app.callback()
def _callback() -> None:
    """No shared setup: this exists only to keep ``run`` addressable by name.

    A ``typer.Typer`` with exactly one command collapses onto that command
    when invoked directly (no subcommand name needed) -- fine for ``mlip
    singlepoint run ...`` since ``add_typer`` in ``cli/main.py`` always
    builds a real group there, but it would make a direct
    ``singlepoint.app`` invocation (as in the unit tests, and a later `freq`
    sibling under this same app) silently swallow the word ``run`` as a
    stray argument.
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
        help="Path to a committee.yaml. Reports a consensus energy and the "
             "members' disagreement for this one configuration, with no "
             "relaxation. Mutually exclusive with --mlip and the head/task "
             "options."),
    member_timeout: float = typer.Option(
        DEFAULT_CALC_TIMEOUT_S, "--member-timeout",
        help="Seconds a single committee member may take before it is "
             "killed and the run aborts."),
    uncertainty_threshold: float = typer.Option(
        None, "--uncertainty-threshold",
        help="Flag this configuration when the committee's force "
             "disagreement over the free atoms exceeds this (eV/Å). NO "
             "DEFAULT: without it sigma is reported and no verdict "
             "asserted."),
    stress: bool = typer.Option(
        None, "--stress/--no-stress",
        help="Ask the calculator for the stress tensor. Default: attempted "
             "only when the cell is periodic in all three directions, "
             "because a slab's stress along the vacuum means nothing."),
    prefix: str = typer.Option("singlepoint",
                               help="Stem for this command's output files"),
    output_dir: Path = typer.Option(
        None, "--output-dir",
        help="Directory for this run's outputs. Default: next to the input "
             "structure. Use it to keep a single-point out of the "
             "optimization folder that produced the structure -- a run "
             "record is replaced by the next command that writes in the "
             "same directory."),
):
    """Evaluate a structure once and report energy, forces and stress."""
    atoms = read(structure)
    typer.echo(f"📂 Loaded structure: {structure.name}")
    typer.echo(f"   Atoms: {len(atoms)}, "
               f"Formula: {atoms.get_chemical_formula()}")

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
    # record stays where it is, and this run writes elsewhere.
    output_dir = output_dir if output_dir is not None else structure.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = RunContext(
        command="singlepoint",
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

        results = run_singlepoint(
            atoms=atoms,
            output_dir=output_dir,
            prefix=prefix,
            model_name=mlip,
            stress=stress,
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

    typer.echo(f"\n⚡ Energy: {results['energy_eV']:.6f} eV")
    # No free atom means no worst one: say so rather than printing
    # "None #None", which reads as a bug in the report.
    worst_free = (
        "no free atom" if results["worst_force_atom_free"] is None
        else f"worst: {results['worst_force_atom_free_symbol']} "
             f"#{results['worst_force_atom_free']}")
    typer.echo(f"   fmax_free = {results['fmax_free_eV_per_A']:.6f} eV/Å "
               f"over {results['n_free_atoms']} free atoms "
               f"({worst_free})")
    typer.echo(f"   fmax_all  = {results['fmax_all_eV_per_A']:.6f} eV/Å "
               f"(worst: {results['worst_force_atom_all_symbol']} "
               f"#{results['worst_force_atom_all']})")
    if results["stress_GPa"] is None:
        typer.echo(f"   stress: not reported "
                   f"({results['stress_unavailable_reason']})")
    else:
        typer.echo(f"   stress (GPa, Voigt): "
                   f"{[round(v, 4) for v in results['stress_GPa']]}")
    if results["unhandled_constraints"]:
        typer.echo(
            f"   Note: constraint type(s) "
            f"{', '.join(results['unhandled_constraints'])} are not masked, "
            f"so the free statistics over-report for their atoms.")

    typer.echo("\n✅ Single point complete. Output files:")
    written = [f"{prefix}_forces.csv", "mliprun_run.json"]
    if committee_config is not None:
        written.insert(1, f"{prefix}_committee_peratom.csv")
    for name in written:
        typer.echo(f"   📄 {(output_dir / name).resolve()}")

    report_committee_uncertainty(committee_calc,
                                 "the evaluated configuration")


if __name__ == "__main__":
    app()
