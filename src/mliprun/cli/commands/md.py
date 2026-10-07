import typer
from pathlib import Path
from ase.io import read
from mliprun.core.md import (
    DEFAULT_BAROSTAT_MASK,
    DEFAULT_COMPRESSIBILITY_PER_GPA,
    PFACTOR_GPA_FS2,
    default_pfactor,
    normalize_barostat_mask,
    run_md,
)
from mliprun.core.run_record import RunContext
from mliprun.cli.utils import (
    DEVICE_HELP,
    MACE_HEAD_HELP,
    MLIP_HELP,
    PLOT_HELP,
    UMA_TASK_HELP,
    _resolve_device,
    detect_mlip,
    param_sources_from_ctx,
    setup_calculator,
    validate_mlip,
    SEVENNET_TASK_HELP,
)

app = typer.Typer(help="Run molecular dynamics simulations.")

BAROSTAT_MASK_HELP = (
    "Axes the barostat may change, as three 0/1 values, e.g. \"0,0,1\" to "
    "relax only z while the in-plane lattice stays fixed (slab-liquid "
    "interface). Default \"1,1,1\" (isotropic). NPT only."
)


def _format_barostat_mask(mask) -> str:
    """Render a mask the way ``--barostat-mask`` accepts it back."""
    return ",".join(str(int(v)) for v in mask)


def _parse_barostat_mask(raw: str) -> tuple:
    """Parse ``"0,0,1"`` into ``(0, 0, 1)``.

    Raises ``typer.BadParameter`` rather than letting a ValueError escape:
    a wrong mask silently changes which cell axes move, and the user needs a
    message naming the flag, not a traceback.
    """
    fields = [field.strip() for field in str(raw).split(",")]
    try:
        values = [int(field) for field in fields]
    except ValueError:
        raise typer.BadParameter(
            f"expected three 0/1 values separated by commas, e.g. \"0,0,1\"; "
            f"got {raw!r}",
            param_hint="--barostat-mask",
        ) from None

    try:
        return normalize_barostat_mask(values)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--barostat-mask") from None


@app.command()
def run(
    ctx: typer.Context,
    structure: Path = typer.Option(..., prompt=True, help="Structure file (.vasp)"),
    ensemble: str = typer.Option("nvt", help="Ensemble: 'nve', 'nvt', 'npt'"),
    steps: int = typer.Option(1000, help="Number of MD steps"),
    temperature: float = typer.Option(300, help="Temperature in K (required for NVT, NPT)"),
    pressure: float = typer.Option(0.0, help="Pressure in GPa (required for NPT)"),
    timestep: float = typer.Option(1.0, help="Timestep in fs"),

    # Thermostat/Barostat selection
    thermostat: str = typer.Option("langevin", help="Thermostat for NVT: 'langevin', 'nose-hoover', 'berendsen'"),
    barostat: str = typer.Option("npt", help="Barostat for NPT: 'npt' (ASE NPT, Melchionna/Nosé-Hoover-Parrinello-Rahman), 'berendsen'"),
    barostat_mask: str = typer.Option(
        _format_barostat_mask(DEFAULT_BAROSTAT_MASK),
        "--barostat-mask",
        help=BAROSTAT_MASK_HELP,
    ),

    # Advanced thermostat/barostat parameters
    friction: float = typer.Option(0.01, help="Langevin friction coefficient (1/fs)"),
    ttime: float = typer.Option(25.0, help="Nosé-Hoover/NPT time constant (fs)"),
    taut: float = typer.Option(100.0, help="Berendsen temperature coupling time (fs)"),
    taup: float = typer.Option(1000.0, help="Berendsen pressure coupling time (fs)"),
    compressibility: float = typer.Option(
        DEFAULT_COMPRESSIBILITY_PER_GPA,
        "--compressibility",
        help=(
            "Berendsen barostat compressibility in 1/GPa. Default 0.457 "
            "(liquid water); solids are ~0.005-0.02. With --taup it sets how "
            "fast the cell responds. --barostat berendsen only."
        ),
    ),
    pfactor: float = typer.Option(
        None,
        "--pfactor",
        help=(
            "Barostat constant for --barostat npt, in GPa*fs^2: ptime^2 * B "
            "(e.g. 75 fs and 100 GPa -> 5.6e5). Default: auto from --ttime, "
            "2.28e6 at 25 fs. --barostat npt only."
        ),
    ),

    # MLIP options
    mlip: str = typer.Option("auto", help=MLIP_HELP),
    uma_task: str = typer.Option(None, help=UMA_TASK_HELP),
    device: str = typer.Option("auto", help=DEVICE_HELP),
    mace_head: str = typer.Option(None, help=MACE_HEAD_HELP),
    sevennet_task: str = typer.Option(None, help=SEVENNET_TASK_HELP),

    # Resume
    resume: bool = typer.Option(
        False,
        "--resume",
        help=(
            "Resume from an existing md.traj + md_energy.csv in the structure's "
            "directory. The provided --structure is ignored for positions/momenta "
            "(the last trajectory frame is used instead) but is still required "
            "to locate the output directory. --steps is additional steps on top "
            "of the prior run."
        ),
    ),
    csv_flush_every: int = typer.Option(
        100,
        "--csv-flush-every",
        help=(
            "Append buffered MD log rows to md_energy.csv every N log calls "
            "(default 100). Set 0 to disable incremental writes and only "
            "flush at end of run."
        ),
    ),
    log_interval: int = typer.Option(
        10,
        "--log-interval",
        help=(
            "Append a row to md_energy.csv every N steps (also drives the "
            "stdout MDLogger). At dt=0.5 fs the default 10 → one row every "
            "5 fs."
        ),
    ),
    traj_interval: int = typer.Option(
        100,
        "--traj-interval",
        help=(
            "Write a frame to md.traj every N steps. At dt=0.5 fs the default "
            "100 → one frame every 50 fs. Lower for finer dynamics; raise to "
            "shrink disk."
        ),
    ),
    plot: bool = typer.Option(False, "--plot/--no-plot", help=PLOT_HELP),
):
    """
    Run molecular dynamics simulation using a supported MLIP model.

    Supports NVE, NVT, and NPT ensembles with various thermostats and barostats.
    """
    output_dir = structure.parent
    if resume:
        traj_file = output_dir / "md.traj"
        if not traj_file.exists():
            raise typer.Exit(f"❌ --resume specified but {traj_file} not found.")
        atoms = read(traj_file, index=-1)
        typer.echo(f"🔁 Resuming from {traj_file} (last frame).")
    else:
        atoms = read(structure)
    ensemble = ensemble.lower()

    # Validate ensemble
    if ensemble not in ['nve', 'nvt', 'npt']:
        raise typer.Exit(f"❌ Unknown ensemble: {ensemble}. Use 'nve', 'nvt', or 'npt'.")

    # Validate parameters for each ensemble
    if ensemble in ['nvt', 'npt'] and temperature <= 0:
        raise typer.Exit(f"❌ Temperature must be > 0 for {ensemble.upper()} ensemble.")

    if ensemble == 'npt' and pressure is None:
        raise typer.Exit("❌ Pressure must be specified for NPT ensemble.")

    barostat_mask_value = _parse_barostat_mask(barostat_mask)
    # Refused, not ignored: NVE and NVT never scale the cell, so a user who
    # masked axes there would believe they had constrained something.
    if ensemble != 'npt' and barostat_mask_value != DEFAULT_BAROSTAT_MASK:
        raise typer.BadParameter(
            f"only applies to --ensemble npt, not '{ensemble}'. The cell "
            f"does not change in NVE or NVT, so masking its axes has no "
            f"effect.",
            param_hint="--barostat-mask",
        )

    # Each knob belongs to one barostat. An explicit value the chosen
    # barostat never reads is refused, same as the mask outside NPT: the
    # user would otherwise believe they had tuned the run.
    sources = param_sources_from_ctx(ctx)
    uses_berendsen = ensemble == 'npt' and barostat.lower() == 'berendsen'
    uses_npt = ensemble == 'npt' and barostat.lower() == 'npt'
    if sources.get("compressibility") == "user" and not uses_berendsen:
        raise typer.BadParameter(
            "only applies to --ensemble npt --barostat berendsen.",
            param_hint="--compressibility",
        )
    if pfactor is not None and not uses_npt:
        raise typer.BadParameter(
            "only applies to --ensemble npt --barostat npt.",
            param_hint="--pfactor",
        )
    if uses_berendsen and not compressibility > 0:
        raise typer.BadParameter(
            f"must be positive (1/GPa), got {compressibility}.",
            param_hint="--compressibility",
        )
    if pfactor is not None and not pfactor > 0:
        raise typer.BadParameter(
            f"must be positive (GPa*fs^2), got {pfactor}.",
            param_hint="--pfactor",
        )
    # Human unit on the CLI, ASE units in the core API.
    pfactor_ase = None if pfactor is None else pfactor * PFACTOR_GPA_FS2
    if pfactor is None:
        pfactor_label = (f"{default_pfactor(ttime) / PFACTOR_GPA_FS2:.4g}"
                         f" (auto from ttime)")
    else:
        pfactor_label = f"{pfactor:.4g}"

    # Detect or use specified model
    if mlip == "auto":
        mlip = detect_mlip()
        typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
        # An auto-detected tag still has to satisfy its own task rules.
        validate_mlip(mlip, sevennet_task, uma_task, mace_head)
    else:
        validate_mlip(mlip, sevennet_task, uma_task, mace_head)
        typer.echo(f"🧠 Using MLIP: {mlip}")

    if mlip.startswith("uma-"):
        typer.echo(f"   UMA task: {uma_task}")
    if mlip.startswith("7net"):
        typer.echo(f"   SevenNet task: {sevennet_task}")

    # Display ensemble information
    typer.echo(f"\n🔬 MD Simulation Setup:")
    typer.echo(f"   Ensemble:    {ensemble.upper()}")
    if ensemble == 'nvt':
        typer.echo(f"   Thermostat:  {thermostat}")
        if thermostat == 'langevin':
            typer.echo(f"   Friction:    {friction} fs⁻¹")
        elif thermostat == 'nose-hoover':
            typer.echo(f"   Time const:  {ttime} fs")
        elif thermostat == 'berendsen':
            typer.echo(f"   Tau T:       {taut} fs")
    elif ensemble == 'npt':
        typer.echo(f"   Barostat:    {barostat}")
        typer.echo(f"   Barostat mask: {_format_barostat_mask(barostat_mask_value)}"
                   f" (x,y,z; 1 = axis free)")
        typer.echo(f"   Pressure:    {pressure} GPa")
        if barostat == 'npt':
            typer.echo(f"   Time const:  {ttime} fs")
            typer.echo(f"   pfactor:     {pfactor_label} GPa fs^2")
        elif barostat == 'berendsen':
            typer.echo(f"   Tau T:       {taut} fs")
            typer.echo(f"   Tau P:       {taup} fs")
            typer.echo(f"   Compressibility: {compressibility} 1/GPa")

    if ensemble in ['nvt', 'npt']:
        typer.echo(f"   Temperature: {temperature} K")
    typer.echo(f"   Steps:       {steps}")
    typer.echo(f"   Timestep:    {timestep} fs")

    # Assign calculator
    typer.echo(f"\n⚙️  Attaching {mlip} calculator (device={device})...")
    if mlip.startswith("mace-mh-"):
        typer.echo(f"   MACE head: {mace_head}")
    atoms = setup_calculator(atoms, mlip, uma_task, device=device,
                              mace_head=mace_head,
                              sevennet_task=sevennet_task)

    # Save parameters before the run starts so the file exists for long runs
    # (500k+ steps) and is still present if the job dies mid-trajectory.
    # Appended on resume so the full invocation chain is preserved.
    param_file = output_dir / "md_params.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(param_file, "a" if resume else "w", encoding="utf-8") as f:
        if resume:
            f.write("\n--- Resume invocation ---\n")
        else:
            f.write("MD Run Parameters\n")
            f.write("===================\n")
        f.write(f"MLIP model:        {mlip}\n")
        f.write(f"Device:            {device}\n")
        if mlip.startswith("uma-"):
            f.write(f"UMA task:          {uma_task}\n")
        if mlip.startswith("mace-mh-"):
            f.write(f"MACE head:         {mace_head}\n")
        if mlip.startswith("7net"):
            f.write(f"SevenNet task:     {sevennet_task}\n")
        f.write(f"Structure:         {structure.name}\n")
        f.write(f"Ensemble:          {ensemble.upper()}\n")

        if ensemble == 'nvt':
            f.write(f"Thermostat:        {thermostat}\n")
            if thermostat == 'langevin':
                f.write(f"Friction (1/fs):   {friction}\n")
            elif thermostat == 'nose-hoover':
                f.write(f"Time constant (fs): {ttime}\n")
            elif thermostat == 'berendsen':
                f.write(f"Tau T (fs):        {taut}\n")

        elif ensemble == 'npt':
            f.write(f"Barostat:          {barostat}\n")
            f.write(f"Barostat mask:     "
                    f"{_format_barostat_mask(barostat_mask_value)}"
                    f"   (x,y,z; 1 = axis free to change)\n")
            f.write(f"Pressure (GPa):    {pressure}\n")
            if barostat == 'npt':
                f.write(f"Time constant (fs): {ttime}\n")
                f.write(f"pfactor (GPa fs^2): {pfactor_label}\n")
            elif barostat == 'berendsen':
                f.write(f"Tau T (fs):        {taut}\n")
                f.write(f"Tau P (fs):        {taup}\n")
                f.write(f"Compressibility (1/GPa): {compressibility}\n")

        if ensemble in ['nvt', 'npt']:
            f.write(f"Temperature (K):   {temperature}\n")
        f.write(f"Number of steps:   {steps}\n")
        f.write(f"Timestep (fs):     {timestep}\n")
        f.write(f"Log interval:      {log_interval} steps\n")
        f.write(f"Traj interval:     {traj_interval} steps\n")
        f.write(f"Output dir:        {output_dir.resolve()}\n")

    # Run MD
    # The record keys carry their unit, so the CLI source tags are copied
    # across; without this both would be recorded as "unspecified".
    for cli_name, record_key in (("compressibility", "compressibility_per_GPa"),
                                 ("pfactor", "pfactor_GPa_fs2")):
        if cli_name in sources:
            sources[record_key] = sources[cli_name]
    run_context = RunContext(
        command="md",
        mode="one-off",
        param_sources=sources,
    )
    run_context.extra_inputs = {
        "structure": structure.name,
        "structure_abspath": str(structure.resolve()),
    }

    run_md(
        atoms=atoms,
        ensemble=ensemble,
        thermostat=thermostat,
        barostat=barostat,
        barostat_mask=barostat_mask_value,
        temperature=temperature,
        pressure=pressure,
        timestep=timestep,
        friction=friction,
        ttime=ttime,
        taut=taut,
        taup=taup,
        compressibility=compressibility,
        pfactor=pfactor_ase,
        steps=steps,
        log_interval=log_interval,
        traj_interval=traj_interval,
        output_dir=output_dir,
        model_name=mlip,
        resume=resume,
        csv_flush_every=csv_flush_every,
        plot=plot,
        run_context=run_context,
        device_requested=device,
        device_resolved=_resolve_device(device),
        uma_task=uma_task,
        mace_head=mace_head,
        sevennet_task=sevennet_task,
    )

    # List output files
    typer.echo("\n✅ MD complete. Output written to:")
    output_files = ["md.traj", "md_energy.csv"]

    if plot:
        output_files.extend(["md_energy.png", "md_temperature.png"])
        if ensemble == 'npt':
            output_files.extend(["md_pressure.png", "md_volume.png"])

    output_files.append("md_params.txt")

    for file in output_files:
        typer.echo(f"   📄 {(output_dir / file).resolve()}")

if __name__ == "__main__":
    app()
