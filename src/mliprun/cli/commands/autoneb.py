"""AutoNEB CLI command."""
from pathlib import Path

import typer
from ase.io import read
from ase.optimize import FIRE

from mliprun.core.neb import CustomNEB
from mliprun.core.params_io import write_parameters_file, write_endpoint_results
from mliprun.core.run_record import RunContext
from mliprun.cli.utils import MACE_HEAD_HELP, MLIP_HELP, SEVENNET_TASK_HELP, UMA_TASK_HELP, param_sources_from_ctx, parse_relax_atoms, resolve_mlip

app = typer.Typer()


@app.command()
def run(
    ctx: typer.Context,
    initial: Path = typer.Option(..., prompt=True, help="Initial structure file (.vasp)"),
    final: Path = typer.Option(..., prompt=True, help="Final structure file (.vasp)"),
    n_max: int = typer.Option(9, help="Maximum number of images (including endpoints)"),
    n_simul: int = typer.Option(1, help="Number of parallel relaxations (requires MPI for n_simul > 1)"),
    fmax: float = typer.Option(0.05, help="Force convergence threshold (eV/Ang)"),
    mlip: str = typer.Option("auto", help=MLIP_HELP),
    uma_task: str = typer.Option("omat", help=UMA_TASK_HELP),
    mace_head: str = typer.Option("omat_pbe", help=MACE_HEAD_HELP),
    sevennet_task: str = typer.Option(None, help=SEVENNET_TASK_HELP),
    climb: bool = typer.Option(True, help="Enable climbing image NEB"),
    k: float = typer.Option(0.1, help="Spring constant"),
    space_energy_ratio: float = typer.Option(0.5, help="Preference for geometric (1.0) vs energy (0.0) gaps"),
    interpolate_method: str = typer.Option("idpp", help="Interpolation method: 'linear' or 'idpp'"),
    maxsteps: int = typer.Option(10000, help="Maximum steps per relaxation"),
    prefix: str = typer.Option("autoneb", help="Prefix for output files"),
    relax_atoms: str = typer.Option(None, help="WARNING: Comma-separated atom indices to relax. May not work well with AutoNEB!"),
    optimize_endpoints: bool = typer.Option(True, help="Optimize initial and final structures before AutoNEB"),
    endpoint_fmax: float = typer.Option(0.01, help="Force threshold for endpoint optimization (eV/Ang)"),
    endpoint_optimizer: str = typer.Option("bfgs", help="Optimizer for endpoints: 'bfgs', 'lbfgs', 'fire'"),
    endpoint_max_steps: int = typer.Option(200, help="Maximum steps for endpoint optimization"),
):
    """Run AutoNEB calculation with dynamic image insertion.

    AutoNEB automatically adds intermediate images until n_max is reached,
    making it ideal for complex reaction pathways where the optimal number
    of images is unclear.
    """
    atoms_initial = read(initial, format="vasp")
    atoms_final = read(final, format="vasp")

    if len(atoms_initial) != len(atoms_final):
        typer.echo("Error: Initial and final structures must have the same number of atoms.")
        raise typer.Exit(code=1)

    mlip = resolve_mlip(mlip, sevennet_task)
    if mlip.startswith("uma-"):
        typer.echo(f"   UMA task: {uma_task}")
    if mlip.startswith("mace-mh-"):
        typer.echo(f"   MACE head: {mace_head}")
    if mlip.startswith("7net"):
        typer.echo(f"   SevenNet task: {sevennet_task}")

    output_dir = Path.cwd()

    relax_indices = None
    if relax_atoms:
        relax_indices = parse_relax_atoms(relax_atoms, len(atoms_initial))
        typer.echo(f"WARNING: Highly-constrained mode with AutoNEB may not work as expected!")
        typer.echo(f"   Relaxing only atoms: {relax_indices}")

    if n_simul > 1:
        typer.echo("\nWARNING: n_simul > 1 requires MPI (parallel execution).")
        typer.echo("   Use 'mpirun -np N autoneb ...' or set --n-simul 1 for serial mode.\n")

    typer.echo(f"\nRunning AutoNEB with:")
    typer.echo(f"   n_max:              {n_max} (target images including endpoints)")
    typer.echo(f"   n_simul:            {n_simul} (parallel relaxations)")
    typer.echo(f"   fmax:               {fmax} eV/Ang")
    typer.echo(f"   climb:              {climb}")
    typer.echo(f"   k:                  {k}")
    typer.echo(f"   space_energy_ratio: {space_energy_ratio}")
    typer.echo(f"   interpolate_method: {interpolate_method}")
    typer.echo(f"   maxsteps:           {maxsteps}")
    typer.echo(f"   prefix:             {prefix}")
    typer.echo(f"   optimize_endpoints: {optimize_endpoints}")
    if optimize_endpoints:
        typer.echo(f"   endpoint_fmax:      {endpoint_fmax}")
        typer.echo(f"   endpoint_optimizer: {endpoint_optimizer}")
    typer.echo(f"   output_dir:         {output_dir}\n")

    # Save parameters
    param_dict = {
        "MLIP model:": mlip,
        **({f"UMA task:": uma_task} if mlip.startswith("uma-") else {}),
        **({f"MACE head:": mace_head} if mlip.startswith("mace-mh-") else {}),
        **({f"SevenNet task:": sevennet_task} if mlip.startswith("7net") else {}),
        "Initial:": str(initial),
        "Final:": str(final),
        "n_max:": n_max,
        "n_simul:": n_simul,
        "fmax:": fmax,
        "climb:": climb,
        "k:": k,
        "space_energy_ratio:": space_energy_ratio,
        "interpolate_method:": interpolate_method,
        "maxsteps:": maxsteps,
        "prefix:": prefix,
        "Optimize endpoints:": optimize_endpoints,
    }
    if optimize_endpoints:
        param_dict["Endpoint fmax:"] = endpoint_fmax
        param_dict["Endpoint optimizer:"] = endpoint_optimizer
        param_dict["Endpoint max steps:"] = endpoint_max_steps
    param_dict["output_dir:"] = str(output_dir)
    if relax_indices:
        param_dict["relax_atoms:"] = relax_indices

    write_parameters_file(output_dir / "autoneb_parameters.txt", "AutoNEB Run Parameters", param_dict)

    # Create CustomNEB instance
    neb = CustomNEB(
        initial=atoms_initial, final=atoms_final,
        num_images=5,  # Dummy value, not used by AutoNEB
        fmax=fmax, mlip=mlip, uma_task=uma_task, mace_head=mace_head,
        sevennet_task=sevennet_task,
        output_dir=output_dir, relax_atoms=relax_indices,
    )

    # Optimize endpoints if requested
    if optimize_endpoints:
        endpoint_results = neb.optimize_endpoints(
            endpoint_fmax=endpoint_fmax, optimizer=endpoint_optimizer,
            max_steps=endpoint_max_steps,
        )
        write_endpoint_results(output_dir / "endpoint_optimization.txt", endpoint_results)

    # Run AutoNEB
    run_context = RunContext(
        command="autoneb",
        mode="one-off",
        param_sources=param_sources_from_ctx(ctx),
    )
    neb.run_autoneb(
        n_simul=n_simul, n_max=n_max, k=k, climb=climb,
        optimizer=FIRE, space_energy_ratio=space_energy_ratio,
        interpolate_method=interpolate_method, maxsteps=maxsteps,
        prefix=prefix, run_context=run_context,
    )

    typer.echo(f"\nAutoNEB output files:")
    typer.echo(f"   {output_dir / f'{prefix}*.traj'} - Individual image trajectories")
    typer.echo(f"   {output_dir / 'AutoNEB_iter'} - Iteration history folder")
    typer.echo(f"   {output_dir / 'autoneb_parameters.txt'} - Run parameters")
    typer.echo("\nNote: Custom convergence plots (CSV/PNG) are not generated in AutoNEB mode.")
    typer.echo("   Check AutoNEB_iter/ folder for detailed iteration history.")


if __name__ == "__main__":
    app()
