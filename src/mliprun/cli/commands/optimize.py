import csv
import sys
import time
import traceback
import typer
from pathlib import Path
from ase.io import read
from mliprun.core.optimize import run_optimization, OPTIMIZER_MAP
from mliprun.core.run_record import BatchInfo, RunContext, new_batch_id
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


@app.command()
def run(
    ctx: typer.Context,
    structure: Path = typer.Option(..., prompt=True, help="Structure file (.vasp)"),
    mlip: str = typer.Option("auto", help=MLIP_HELP),
    uma_task: str = typer.Option("omat", help=UMA_TASK_HELP),
    device: str = typer.Option("auto", help=DEVICE_HELP),
    mace_head: str = typer.Option("omat_pbe", help=MACE_HEAD_HELP),
    sevennet_task: str = typer.Option(None, help=SEVENNET_TASK_HELP),
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

    # Detect or validate MLIP
    if mlip == "auto":
        mlip = detect_mlip()
        typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
        # An auto-detected tag still has to satisfy its own task rules.
        validate_mlip(mlip, sevennet_task)
    else:
        validate_mlip(mlip, sevennet_task)
        typer.echo(f"🧠 Using MLIP: {mlip}")

    # Validate optimizer
    if optimizer.lower() not in OPTIMIZER_MAP:
        typer.echo(f"❌ Unknown optimizer: {optimizer}")
        typer.echo(f"   Available: {', '.join(OPTIMIZER_MAP.keys())}")
        raise typer.Exit(1)

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

    # Output directory
    output_dir = structure.parent

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
    )

    # Save parameters
    _write_params(output_dir / "opt_params.txt", mlip, uma_task, mace_head,
                  device, relax_cell, structure.name, optimizer, fmax,
                  max_steps, converged, output_dir,
                  sevennet_task=sevennet_task)

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
    if plot:
        output_files.insert(3, f"{logfile_stem}_convergence.png")
    for file in output_files:
        typer.echo(f"   📄 {(output_dir / file).resolve()}")

    if not converged:
        typer.echo("\n⚠️  Warning: Optimization did not converge. Consider:")
        typer.echo("   - Increasing max_steps")
        typer.echo("   - Relaxing fmax threshold")
        typer.echo("   - Trying a different optimizer")


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
    uma_task: str = typer.Option("omat", help=UMA_TASK_HELP),
    device: str = typer.Option("auto", help=DEVICE_HELP),
    mace_head: str = typer.Option("omat_pbe", help=MACE_HEAD_HELP),
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
        validate_mlip(mlip, sevennet_task)
    else:
        validate_mlip(mlip, sevennet_task)
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
                  output_dir, sevennet_task=None):
    """Write the per-structure opt_params.txt (matches ``optimize run``)."""
    with open(param_file, "w", encoding="utf-8") as f:
        f.write("Geometry Optimization Parameters\n")
        f.write("=================================\n")
        f.write(f"MLIP model:        {mlip}\n")
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
