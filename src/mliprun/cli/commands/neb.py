"""NEB CLI command."""
import logging
from datetime import datetime
from pathlib import Path
import shutil

import typer
from ase.io import read
from ase.optimize import FIRE, MDMin, BFGS, LBFGS

from mliprun.core.neb import CustomNEB
from mliprun.core.params_io import write_parameters_file, write_endpoint_results
from mliprun.core.run_record import RunContext
from mliprun.cli.utils import MACE_HEAD_HELP, MLIP_HELP, PLOT_HELP, SEVENNET_TASK_HELP, UMA_TASK_HELP, param_sources_from_ctx, parse_relax_atoms, resolve_mlip

logger = logging.getLogger(__name__)

app = typer.Typer()

NEB_OPTIMIZER_MAP = {"fire": FIRE, "mdmin": MDMin, "bfgs": BFGS, "lbfgs": LBFGS}


def create_backup_folder(output_dir: Path):
    """Create timestamped backup folder and move all NEB output files.

    Parameters
    ----------
    output_dir : Path
        Current output directory.

    Returns
    -------
    tuple[Path, list[str]]
        Path to created backup folder and list of moved files/folders.
    """
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    backup_dir = output_dir / f"bkup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    files_to_backup = [
        "A2B.traj", "A2B_full.traj", "neb.log",
        "neb_convergence.csv", "neb_convergence.png",
        "neb_energy.png", "neb_data.csv", "idpp.traj", "idpp.log",
        "endpoint_optimization.txt",
        "initial_opt.traj", "initial_opt.log",
        "final_opt.traj", "final_opt.log",
    ]

    moved_files = []

    for filename in files_to_backup:
        filepath = output_dir / filename
        if filepath.exists():
            shutil.move(str(filepath), str(backup_dir / filename))
            moved_files.append(filename)

    # Move POSCAR folders (00/, 01/, 02/, ...)
    for poscar_dir in sorted(output_dir.glob("[0-9][0-9]")):
        if poscar_dir.is_dir():
            shutil.move(str(poscar_dir), str(backup_dir / poscar_dir.name))
            moved_files.append(poscar_dir.name + "/")

    # Copy (not move) neb_parameters.txt — needed for restart but backup should
    # have a complete record of the original run
    params_file = output_dir / "neb_parameters.txt"
    if params_file.exists():
        shutil.copy(str(params_file), str(backup_dir / "neb_parameters.txt"))
        moved_files.append("neb_parameters.txt (copied)")

    return backup_dir, moved_files


def _handle_restart(output_dir, *, mlip, uma_task, mace_head, sevennet_task,
                    fmax, log, k, climb,
                    dyneb, scale_fmax, neb_optimizer, neb_max_steps, device):
    """Handle NEB restart mode.

    Returns
    -------
    tuple[CustomNEB, dict]
        NEB instance loaded from restart and the resolved parameter dict.
    """
    import time

    typer.echo("RESTART MODE")

    typer.echo(f"Loading restart files from: {output_dir}")

    if mlip is not None:
        typer.echo(f"   MLIP override detected: {mlip}")

    try:
        neb_instance, loaded_params = CustomNEB.load_from_restart(
            output_dir=output_dir, mlip=mlip, uma_task=uma_task,
            mace_head=mace_head, sevennet_task=sevennet_task,
            fmax=fmax, logfile=log, k=k, climb=climb,
            neb_optimizer=neb_optimizer, neb_max_steps=neb_max_steps,
            device=device,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)

    # Show loaded parameters
    typer.echo("\nParameters loaded from neb_parameters.txt:")
    typer.echo(f"   - MLIP model:          {loaded_params['mlip']}")
    if loaded_params.get("uma_task"):
        typer.echo(f"   - UMA task:            {loaded_params['uma_task']}")
    if loaded_params.get("mace_head"):
        typer.echo(f"   - MACE head:           {loaded_params['mace_head']}")
    if loaded_params.get("sevennet_task"):
        typer.echo(f"   - SevenNet task:       {loaded_params['sevennet_task']}")
    typer.echo(f"   - Intermediate images: {loaded_params['num_images']}")
    typer.echo(f"   - Total images:        {loaded_params['total_images']}")
    if loaded_params.get("relax_atoms"):
        typer.echo(f"   - Relax atoms:         {loaded_params['relax_atoms']}")

    # Show overrides
    overrides = []
    if mlip is not None:
        overrides.append(f"MLIP: {mlip}")
    if fmax is not None:
        overrides.append(f"fmax: {fmax}")
    if k is not None:
        overrides.append(f"k: {k}")
    if climb is not None:
        overrides.append(f"climb: {climb}")
    if dyneb is not None:
        overrides.append(f"dyneb: {dyneb}")
    if scale_fmax is not None:
        overrides.append(f"scale_fmax: {scale_fmax}")
    if neb_optimizer is not None:
        overrides.append(f"optimizer: {neb_optimizer}")
    if neb_max_steps is not None:
        overrides.append(f"max_steps: {neb_max_steps}")
    if overrides:
        typer.echo(f"\nParameter overrides: {', '.join(overrides)}")

    # Validate image count consistency
    from ase.io import read as ase_read

    full_traj_path = output_dir / "A2B_full.traj"
    all_frames = ase_read(str(full_traj_path), index=":")
    expected_total = loaded_params["total_images"]

    if len(all_frames) % expected_total != 0:
        typer.echo(f"\nWARNING: Image count mismatch!")
        typer.echo(f"   Expected multiples of {expected_total} images, "
                   f"but found {len(all_frames)} frames.")
        typer.echo("   Press Ctrl+C within 15 seconds to cancel...")
        for i in range(15, 0, -1):
            typer.echo(f"   Continuing in: {i}...", nl=False)
            time.sleep(1)
            typer.echo("\r" + " " * 50 + "\r", nl=False)
        typer.echo("   Proceeding with restart...\n")

    # Create backup
    typer.echo("Creating backup of previous results...")
    backup_dir, moved_files = create_backup_folder(output_dir)
    typer.echo(f"   Backup created: {backup_dir.name}")
    typer.echo(f"   Moved {len(moved_files)} files/folders")

    # Resolve parameters (override or loaded)
    params = {
        "mlip": mlip or loaded_params["mlip"],
        # No "<default>" fallbacks: a head must never be invented on
        # restart, or the band resumes on a different energy zero (C3).
        "uma_task": uma_task or loaded_params.get("uma_task"),
        "mace_head": mace_head or loaded_params.get("mace_head"),
        # No default fallback: a SevenNet task must never be invented on
        # restart, or the band resumes on a different energy zero (C3).
        "sevennet_task": sevennet_task or loaded_params.get("sevennet_task"),
        "fmax": fmax if fmax is not None else loaded_params["fmax"],
        "k": k if k is not None else loaded_params.get("k", 0.1),
        "climb": climb if climb is not None else loaded_params.get("climb", True),
        "dyneb": dyneb if dyneb is not None else loaded_params.get("dyneb", False),
        "scale_fmax": scale_fmax if scale_fmax is not None else loaded_params.get("scale_fmax", 0.0),
        "neb_optimizer": neb_optimizer or loaded_params.get("neb_optimizer", "fire"),
        "neb_max_steps": neb_max_steps if neb_max_steps is not None else loaded_params.get("neb_max_steps", 600),
        "log": log or loaded_params.get("log", "neb.log"),
        "num_images": loaded_params["num_images"],
        "total_images": loaded_params["total_images"],
        "relax_atoms": loaded_params.get("relax_atoms"),
        "device": device if device is not None else loaded_params.get("device", "cpu"),
        "optimize_endpoints": False,
    }

    # Write updated parameter file
    write_parameters_file(output_dir / "neb_parameters.txt", "NEB Run Parameters (RESTART)", {
        "Restarted from:": backup_dir.name,
        "MLIP model:": params["mlip"],
        **({f"UMA task:": params["uma_task"]} if params["mlip"].startswith("uma-") else {}),
        **({f"MACE head:": params["mace_head"]} if params["mlip"].startswith("mace-mh-") else {}),
        **({f"SevenNet task:": params["sevennet_task"]} if params["mlip"].startswith("7net") else {}),
        "Device:": params["device"],
        "Initial:": "(from restart)",
        "Final:": "(from restart)",
        "Intermediate images:": params["num_images"],
        "Total images:": params["total_images"],
        "IDPP fmax:": "(from restart)",
        "IDPP steps:": "(from restart)",
        "Final fmax:": params["fmax"],
        "Spring constant (k):": params["k"],
        "Climb:": params["climb"],
        "Dynamic NEB:": params["dyneb"],
        "Scale fmax:": params["scale_fmax"],
        "NEB optimizer:": params["neb_optimizer"],
        "NEB max steps:": params["neb_max_steps"],
        "Optimize endpoints:": "False (restart)",
        "Log file:": params["log"],
        "Output dir:": str(output_dir),
        **({f"Relax atoms:": params["relax_atoms"]} if params["relax_atoms"] else {}),
    })

    return neb_instance, params


def _handle_new_neb(output_dir, initial, final, *, num_images, interp_fmax,
                    interp_steps, fmax, mlip, uma_task, mace_head,
                    sevennet_task, log, k, climb,
                    dyneb, scale_fmax,
                    neb_optimizer, neb_max_steps, optimize_endpoints,
                    endpoint_fmax, endpoint_optimizer, endpoint_max_steps,
                    relax_atoms_str, device):
    """Handle normal (non-restart) NEB mode.

    Returns
    -------
    tuple[CustomNEB, dict]
        NEB instance and the resolved parameter dict.
    """
    if initial is None or final is None:
        typer.echo("Error: --initial and --final are required for new NEB calculation")
        typer.echo("   Use --restart to continue from previous calculation")
        raise typer.Exit(code=1)

    # Defaults
    num_images = num_images or 5
    interp_fmax = interp_fmax or 0.1
    interp_steps = interp_steps or 100
    fmax = fmax or 0.05
    mlip = mlip or "auto"
    log = log or "neb.log"
    k = k or 0.1
    climb = climb if climb is not None else True
    dyneb = dyneb if dyneb is not None else False
    scale_fmax = scale_fmax if scale_fmax is not None else 0.0
    neb_optimizer = neb_optimizer or "fire"
    neb_max_steps = neb_max_steps or 600
    optimize_endpoints = optimize_endpoints if optimize_endpoints is not None else True
    endpoint_fmax = endpoint_fmax or 0.01
    endpoint_optimizer = endpoint_optimizer or "bfgs"
    endpoint_max_steps = endpoint_max_steps or 200
    device = device or "cpu"

    atoms_initial = read(initial, format="vasp")
    atoms_final = read(final, format="vasp")

    if len(atoms_initial) != len(atoms_final):
        typer.echo("Error: Initial and final structures must have the same number of atoms.")
        raise typer.Exit(code=1)

    mlip = resolve_mlip(mlip, sevennet_task, uma_task, mace_head)
    if mlip.startswith("uma-"):
        typer.echo(f"   UMA task: {uma_task}")
    if mlip.startswith("mace-mh-"):
        typer.echo(f"   MACE head: {mace_head}")
    if mlip.startswith("7net"):
        typer.echo(f"   SevenNet task: {sevennet_task}")

    # Parse relax_atoms
    relax_indices = None
    if relax_atoms_str:
        relax_indices = parse_relax_atoms(relax_atoms_str, len(atoms_initial))
        typer.echo(f"HIGHLY CONSTRAINED MODE: Relaxing only atoms: {relax_indices}")

    total_images = num_images + 2

    params = {
        "mlip": mlip, "uma_task": uma_task, "mace_head": mace_head,
        "sevennet_task": sevennet_task,
        "fmax": fmax, "k": k,
        "climb": climb, "dyneb": dyneb, "scale_fmax": scale_fmax,
        "neb_optimizer": neb_optimizer,
        "neb_max_steps": neb_max_steps, "log": log,
        "num_images": num_images, "total_images": total_images,
        "relax_atoms": relax_indices, "optimize_endpoints": optimize_endpoints,
        "device": device,
    }

    # Write parameter file
    param_dict = {
        "MLIP model:": mlip,
        **({f"UMA task:": uma_task} if mlip.startswith("uma-") else {}),
        **({f"MACE head:": mace_head} if mlip.startswith("mace-mh-") else {}),
        **({f"SevenNet task:": sevennet_task} if mlip.startswith("7net") else {}),
        "Device:": device,
        "Initial:": str(initial),
        "Final:": str(final),
        "Intermediate images:": num_images,
        "Total images:": total_images,
        "IDPP fmax:": interp_fmax,
        "IDPP steps:": interp_steps,
        "Final fmax:": fmax,
        "Spring constant (k):": k,
        "Climb:": climb,
        "Dynamic NEB:": dyneb,
        "Scale fmax:": scale_fmax,
        "NEB optimizer:": neb_optimizer,
        "NEB max steps:": neb_max_steps,
        "Optimize endpoints:": optimize_endpoints,
    }
    if optimize_endpoints:
        param_dict["Endpoint fmax:"] = endpoint_fmax
        param_dict["Endpoint optimizer:"] = endpoint_optimizer
        param_dict["Endpoint max steps:"] = endpoint_max_steps
    param_dict["Log file:"] = log
    param_dict["Output dir:"] = str(output_dir)
    if relax_indices:
        param_dict["Relax atoms:"] = relax_indices

    write_parameters_file(output_dir / "neb_parameters.txt", "NEB Run Parameters", param_dict)

    # Create NEB instance
    neb_obj = CustomNEB(
        initial=atoms_initial, final=atoms_final,
        num_images=num_images, interp_fmax=interp_fmax,
        interp_steps=interp_steps, fmax=fmax, mlip=mlip,
        uma_task=uma_task, mace_head=mace_head,
        sevennet_task=sevennet_task, output_dir=output_dir,
        relax_atoms=relax_indices, logfile=log, device=device,
    )

    # Optimize endpoints
    if optimize_endpoints:
        endpoint_results = neb_obj.optimize_endpoints(
            endpoint_fmax=endpoint_fmax, optimizer=endpoint_optimizer,
            max_steps=endpoint_max_steps,
        )
        neb_obj.images = neb_obj.setup_neb()
        write_endpoint_results(output_dir / "endpoint_optimization.txt", endpoint_results)

    # IDPP interpolation
    typer.echo(" Interpolating with IDPP...")
    neb_obj.interpolate_idpp()

    return neb_obj, params


@app.command()
def run(
    ctx: typer.Context,
    restart: bool = typer.Option(False, "--restart", help="Restart from previous NEB calculation"),
    initial: Path = typer.Option(None, help="Initial structure file (.vasp)"),
    final: Path = typer.Option(None, help="Final structure file (.vasp)"),
    num_images: int = typer.Option(None, help="Number of intermediate images (excluding initial and final)"),
    interp_fmax: float = typer.Option(None, help="IDPP interpolation fmax"),
    interp_steps: int = typer.Option(None, help="IDPP interpolation steps"),
    fmax: float = typer.Option(None, help="Final NEB force threshold"),
    mlip: str = typer.Option(None, help=MLIP_HELP),
    uma_task: str = typer.Option(None, help=UMA_TASK_HELP),
    mace_head: str = typer.Option(None, help=MACE_HEAD_HELP),
    sevennet_task: str = typer.Option(None, help=SEVENNET_TASK_HELP),
    relax_atoms: str = typer.Option(None, help="Comma-separated list of atom indices to relax (e.g. '0,1,5'). If set, others are fixed."),
    log: str = typer.Option(None, help="Name for the NEB iteration log file (default: neb.log)"),
    k: float = typer.Option(None, help="Spring constant for NEB"),
    climb: bool = typer.Option(None, help="Enable climbing image NEB"),
    dyneb: bool = typer.Option(None, "--dyneb/--no-dyneb", help="Use DyNEB (dynamic NEB): freeze images already converged below fmax to save force calls (default: off)"),
    scale_fmax: float = typer.Option(None, help="DyNEB only: loosen per-image convergence with distance from the highest-energy image (0 = uniform criterion)"),
    neb_optimizer: str = typer.Option(None, help="NEB optimizer: 'fire', 'mdmin', 'bfgs', or 'lbfgs'"),
    neb_max_steps: int = typer.Option(None, help="Maximum steps for NEB optimization"),
    optimize_endpoints: bool = typer.Option(None, help="Optimize initial and final structures before NEB"),
    endpoint_fmax: float = typer.Option(None, help="Force threshold for endpoint optimization (eV/Ang)"),
    endpoint_optimizer: str = typer.Option(None, help="Optimizer for endpoints: 'bfgs', 'lbfgs', 'fire'"),
    endpoint_max_steps: int = typer.Option(None, help="Maximum steps for endpoint optimization"),
    device: str = typer.Option(None, help="Compute device for the MLIP calculator: 'cpu' (default) or 'cuda'"),
    plot: bool = typer.Option(False, "--plot/--no-plot", help=PLOT_HELP),
):
    """Run Nudged Elastic Band (NEB) calculation."""
    output_dir = Path.cwd()

    if restart:
        # Validate forbidden parameters
        forbidden = {
            "initial": initial, "final": final, "num_images": num_images,
            "relax_atoms": relax_atoms, "optimize_endpoints": optimize_endpoints,
        }
        provided = [name for name, val in forbidden.items() if val is not None]
        if provided:
            param_names = ", ".join(f"--{n.replace('_', '-')}" for n in provided)
            typer.echo(f"Error: Cannot specify {param_names} with --restart")
            typer.echo("   These parameters are loaded from neb_parameters.txt")
            raise typer.Exit(code=1)

        neb_obj, params = _handle_restart(
            output_dir, mlip=mlip, uma_task=uma_task, mace_head=mace_head,
            sevennet_task=sevennet_task, fmax=fmax, log=log,
            k=k, climb=climb, dyneb=dyneb, scale_fmax=scale_fmax,
            neb_optimizer=neb_optimizer,
            neb_max_steps=neb_max_steps, device=device,
        )
        typer.echo("Skipping interpolation (loaded from restart)")
    else:
        neb_obj, params = _handle_new_neb(
            output_dir, initial, final,
            num_images=num_images, interp_fmax=interp_fmax,
            interp_steps=interp_steps, fmax=fmax, mlip=mlip,
            uma_task=uma_task, mace_head=mace_head,
            sevennet_task=sevennet_task, log=log, k=k, climb=climb,
            dyneb=dyneb, scale_fmax=scale_fmax,
            neb_optimizer=neb_optimizer, neb_max_steps=neb_max_steps,
            optimize_endpoints=optimize_endpoints, endpoint_fmax=endpoint_fmax,
            endpoint_optimizer=endpoint_optimizer,
            endpoint_max_steps=endpoint_max_steps,
            relax_atoms_str=relax_atoms, device=device,
        )

    # ---- Run NEB optimization ----
    neb_optimizer_name = params.get("neb_optimizer", "fire")
    neb_opt = NEB_OPTIMIZER_MAP.get(neb_optimizer_name.lower(), FIRE)
    climb_val = params.get("climb", True)
    max_steps = params.get("neb_max_steps", 600)

    engine = "DyNEB" if params.get("dyneb") else "NEB"
    typer.echo(f"Running {engine} optimization (optimizer={neb_optimizer_name.upper()}, "
               f"climb={climb_val}, max_steps={max_steps})...")
    run_context = RunContext(
        command="neb",
        mode="one-off",
        param_sources=param_sources_from_ctx(ctx),
    )
    neb_obj.run_neb(optimizer=neb_opt, climb=climb_val, max_steps=max_steps,
                    k=params.get("k"), dyneb=params.get("dyneb", False),
                    scale_fmax=params.get("scale_fmax", 0.0),
                    plot=plot, run_context=run_context,
                    append=restart)

    typer.echo("Processing results...")
    df = neb_obj.process_results()
    if plot:
        neb_obj.plot_results(df)

    typer.echo("Exporting POSCARs...")
    neb_obj.export_poscars()

    total = params["total_images"]
    log_name = params.get("log", "neb.log")
    typer.echo("NEB complete. Output written to:")
    output_files = [log_name, "neb_convergence.csv",
                    "A2B.traj", "A2B_full.traj", "idpp.traj", "idpp.log",
                    "neb_data.csv", "neb_parameters.txt"]
    if plot:
        output_files += ["neb_convergence.png", "neb_energy.png"]
    for f in output_files:
        typer.echo(f" - {output_dir / f}")
    for i in range(total):
        typer.echo(f" - {output_dir / f'{i:02d}' / 'POSCAR'}")


if __name__ == "__main__":
    app()
