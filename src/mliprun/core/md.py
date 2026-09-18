"""Molecular dynamics engine using ASE."""
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import units
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from mliprun.core.utils import GPA_TO_EV_PER_ANG3
from mliprun.core.run_record import RunContext, RunRecord, collect_provenance

logger = logging.getLogger(__name__)

# Try to import Nose-Hoover (may not be available in all ASE versions)
try:
    from ase.md.nose_hoover import NoseHoover
    NOSEHOOVER_AVAILABLE = True
except ImportError:
    NOSEHOOVER_AVAILABLE = False


DYNAMICS_MAP = {
    "nve": {"velocityverlet": VelocityVerlet},
    "nvt": {
        "langevin": Langevin,
        "nose-hoover": NoseHoover if NOSEHOOVER_AVAILABLE else None,
        "berendsen": NVTBerendsen,
    },
    "npt": {
        # The class each barostat keyword resolves to with the default,
        # isotropic mask. A non-default ``barostat_mask`` keeps the keyword
        # but swaps ``berendsen`` for the Inhomogeneous_NPTBerendsen
        # subclass, which scales each axis separately; ``npt`` (MTK) takes
        # the mask on the same class. See :func:`setup_dynamics`.
        "npt": NPT,
        "berendsen": NPTBerendsen,
    },
}

#: Every axis coupled to the barostat -- the historical, isotropic behaviour.
DEFAULT_BAROSTAT_MASK = (1, 1, 1)


def normalize_barostat_mask(barostat_mask) -> tuple:
    """Validate a barostat mask and return it as a 3-tuple of ints.

    Parameters
    ----------
    barostat_mask : sequence
        Three values, each 0 or 1, in Cartesian ``(x, y, z)`` order. ``1``
        couples that axis to the barostat; ``0`` holds its length fixed.

    Returns
    -------
    tuple
        ``(x, y, z)`` as plain Python ints.

    Raises
    ------
    ValueError
        If the mask is not exactly three values, or any value is not 0 or 1.
        Rejected rather than coerced: a silently truncated or rounded mask
        would change which axes of the cell are allowed to move, and the
        trajectory is the only place that error would show up.
    """
    try:
        values = list(barostat_mask)
    except TypeError:
        raise ValueError(
            f"barostat_mask must be three values (x, y, z), got {barostat_mask!r}"
        ) from None

    if len(values) != 3:
        raise ValueError(
            f"barostat_mask must have exactly 3 values (x, y, z), "
            f"got {len(values)}: {barostat_mask!r}"
        )

    normalized = []
    for axis, value in zip("xyz", values):
        # bool is a subclass of int, so True/False are accepted as 1/0.
        # Strings and floats are not: they are far more likely to be a typo
        # than an intent, and 0.5 has no meaning here.
        if not isinstance(value, (bool, int, np.integer)) or int(value) not in (0, 1):
            raise ValueError(
                f"barostat_mask {axis} value must be 0 or 1, "
                f"got {value!r} in {barostat_mask!r}"
            )
        normalized.append(int(value))

    return tuple(normalized)


def setup_dynamics(
    atoms,
    ensemble: str = "nvt",
    thermostat: str = "langevin",
    barostat: str = "npt",
    barostat_mask=DEFAULT_BAROSTAT_MASK,
    temperature: float = 300,
    pressure: float = 0.0,
    timestep: float = 1.0,
    friction: float = 0.01,
    ttime: float = 25.0,
    pfactor=None,
    taut: float = 100.0,
    taup: float = 1000.0,
    compressibility: float = 4.57e-5,
    set_velocities: bool = True,
):
    """Set up MD dynamics with specified ensemble and parameters.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with calculator attached.
    ensemble : str
        Ensemble type: ``'nve'``, ``'nvt'``, ``'npt'``.
    thermostat : str
        Thermostat for NVT: ``'langevin'``, ``'nose-hoover'``, ``'berendsen'``.
    barostat : str
        Barostat for NPT: ``'npt'`` (MTK), ``'berendsen'``.
    barostat_mask : sequence of 3 ints
        Which Cartesian axes the barostat may change, ``(x, y, z)``. ``1``
        couples that axis; ``0`` holds its length fixed. Defaults to
        ``(1, 1, 1)`` -- isotropic coupling, identical to the behaviour
        before this parameter existed. ``(0, 0, 1)`` relaxes only z, which
        is what a slab-liquid interface needs: the in-plane lattice stays at
        its relaxed bulk value while the liquid finds its own density.
        Meaningful only for ``ensemble='npt'``; a non-default mask with any
        other ensemble raises rather than being silently ignored.
    temperature : float
        Temperature in Kelvin.
    pressure : float
        Pressure in GPa (for NPT).
    timestep : float
        Timestep in fs.
    friction : float
        Langevin friction coefficient (1/fs).
    ttime : float
        Nose-Hoover/NPT time constant (fs).
    pfactor : float or None
        NPT pressure coupling factor (auto-calculated if None).
    taut : float
        Berendsen temperature coupling time (fs).
    taup : float
        Berendsen pressure coupling time (fs).
    compressibility : float
        Berendsen compressibility (1/GPa).

    Returns
    -------
    ASE dynamics object.

    Raises
    ------
    ValueError
        If ensemble/thermostat/barostat is unknown.
    ImportError
        If Nose-Hoover is requested but not available.
    """
    ensemble = ensemble.lower()
    barostat_mask = normalize_barostat_mask(barostat_mask)

    # A mask outside NPT is a silent no-op in ASE: neither NVE nor NVT has a
    # cell-scaling step for it to reach. Refusing is the only way the caller
    # learns that the axes they asked to hold were never being moved anyway.
    if ensemble != "npt" and barostat_mask != DEFAULT_BAROSTAT_MASK:
        raise ValueError(
            f"barostat_mask={barostat_mask} is only meaningful for "
            f"ensemble='npt', not '{ensemble}'. The cell does not change in "
            f"NVE or NVT, so masking its axes would have no effect."
        )

    if set_velocities and ensemble in ["nvt", "npt"] and temperature > 0:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    timestep_ase = timestep * units.fs

    if ensemble == "nve":
        return VelocityVerlet(atoms, timestep=timestep_ase)

    elif ensemble == "nvt":
        thermostat = thermostat.lower()

        if thermostat == "langevin":
            return Langevin(
                atoms, timestep=timestep_ase,
                temperature_K=temperature, friction=friction / units.fs,
            )
        elif thermostat == "nose-hoover":
            if not NOSEHOOVER_AVAILABLE:
                raise ImportError("Nose-Hoover thermostat not available in this ASE version")
            return NoseHoover(
                atoms, timestep=timestep_ase,
                temperature_K=temperature, ttime=ttime * units.fs,
            )
        elif thermostat == "berendsen":
            return NVTBerendsen(
                atoms, timestep=timestep_ase,
                temperature_K=temperature, taut=taut * units.fs,
            )
        else:
            raise ValueError(f"Unknown thermostat: {thermostat}. Use 'langevin', 'nose-hoover', or 'berendsen'")

    elif ensemble == "npt":
        barostat = barostat.lower()
        pressure_ase = pressure * GPA_TO_EV_PER_ANG3
        externalstress = pressure_ase * np.ones(6)

        isotropic = barostat_mask == DEFAULT_BAROSTAT_MASK

        if barostat == "npt":
            if pfactor is None:
                pfactor = (ttime * 75 * units.GPa) ** 2
            return NPT(
                atoms, timestep=timestep_ase,
                temperature_K=temperature, externalstress=externalstress,
                ttime=ttime * units.fs, pfactor=pfactor,
                # None, not (1, 1, 1): ASE's own default, so an unmasked run
                # takes exactly the call it took before this parameter
                # existed. NPT.set_mask turns a 3-vector into its outer
                # product, so (0, 0, 1) frees the zz strain alone -- no
                # in-plane strain and no xz/yz shear.
                mask=None if isotropic else barostat_mask,
            )
        elif barostat == "berendsen":
            if isotropic:
                return NPTBerendsen(
                    atoms, timestep=timestep_ase,
                    temperature_K=temperature, taut=taut * units.fs,
                    pressure_au=pressure_ase, taup=taup * units.fs,
                    compressibility_au=compressibility / units.GPa,
                )
            # NPTBerendsen itself takes no mask: the anisotropic version is a
            # separate subclass that scales each axis independently.
            return Inhomogeneous_NPTBerendsen(
                atoms, timestep=timestep_ase,
                temperature_K=temperature, taut=taut * units.fs,
                pressure_au=pressure_ase, taup=taup * units.fs,
                compressibility_au=compressibility / units.GPa,
                mask=barostat_mask,
            )
        else:
            raise ValueError(f"Unknown barostat: {barostat}. Use 'npt' or 'berendsen'")

    else:
        raise ValueError(f"Unknown ensemble: {ensemble}. Use 'nve', 'nvt', or 'npt'")


def _md_statistics(df, n_atoms: int) -> dict:
    """Raw, assumption-free summary of one MD segment.

    Deliberately excludes any equilibration-window detection: choosing a
    production region is a methodological decision, and MD samples are
    autocorrelated, so a naive standard-error criterion would truncate too
    early and understate the uncertainty. Decile block means are provided so
    equilibration can be judged by eye without reopening the CSV.
    """
    if df.empty:
        return {"n_samples": 0}

    total = df["total_energy(eV)"]
    stats = {
        "n_samples": int(len(df)),
        "mean_temperature_K": float(df["temperature(K)"].mean()),
        "std_temperature_K": float(df["temperature(K)"].std(ddof=1)) if len(df) > 1 else 0.0,
        "mean_total_energy_eV": float(total.mean()),
        "std_total_energy_eV": float(total.std(ddof=1)) if len(df) > 1 else 0.0,
        "mean_potential_energy_eV": float(df["potential_energy(eV)"].mean()),
        "std_potential_energy_eV": float(df["potential_energy(eV)"].std(ddof=1)) if len(df) > 1 else 0.0,
    }

    # Drift per atom per ps: the number that says whether an NVE run is
    # trustworthy. Time is logged in fs.
    span_fs = float(df["time(fs)"].iloc[-1] - df["time(fs)"].iloc[0])
    if span_fs > 0 and n_atoms > 0:
        drift = float(total.iloc[-1] - total.iloc[0])
        stats["total_energy_drift_eV_per_atom_per_ps"] = (
            drift / n_atoms / (span_fs / 1000.0)
        )
    else:
        stats["total_energy_drift_eV_per_atom_per_ps"] = 0.0

    # Ten equal blocks over the segment, so equilibration is eyeballable.
    n = len(df)
    stats["decile_mean_total_energy_eV"] = [
        float(total.iloc[(i * n) // 10:((i + 1) * n) // 10].mean())
        if ((i + 1) * n) // 10 > (i * n) // 10 else float(total.iloc[-1])
        for i in range(10)
    ]

    if "pressure(GPa)" in df:
        stats["mean_pressure_GPa"] = float(df["pressure(GPa)"].mean())
        stats["mean_volume_A3"] = float(df["volume(A^3)"].mean())
    return stats


def run_md(
    atoms,
    ensemble: str = "nvt",
    thermostat: str = "langevin",
    barostat: str = "npt",
    barostat_mask=DEFAULT_BAROSTAT_MASK,
    temperature: float = 300,
    pressure: float = 0.0,
    timestep: float = 1.0,
    friction: float = 0.01,
    ttime: float = 25.0,
    pfactor=None,
    taut: float = 100.0,
    taup: float = 1000.0,
    compressibility: float = 4.57e-5,
    steps: int = 1000,
    log_interval: int = 10,
    traj_interval: int = 100,
    log_path=None,
    output_dir: str | Path = ".",
    model_name: str = "mlip",
    resume: bool = False,
    csv_flush_every: int = 100,
    plot: bool = False,
    run_context: Optional[RunContext] = None,
    device_requested: str = "auto",
    device_resolved: str = "auto",
    uma_task: Optional[str] = None,
    mace_head: Optional[str] = None,
    sevennet_task: Optional[str] = None,
) -> None:
    """Run molecular dynamics simulation.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with calculator attached.
    ensemble : str
        Ensemble type: ``'nve'``, ``'nvt'``, ``'npt'``.
    steps : int
        Number of MD steps.
    log_interval : int
        Append a row to md_energy.csv every N steps (also drives stdout
        MDLogger). At dt=0.5 fs, 10 → 5 fs/row.
    traj_interval : int
        Write a frame to md.traj every N steps. At dt=0.5 fs, 100 → 50 fs/frame.
    log_path : str or None
        Path to log file.
    output_dir : str or Path
        Output directory.
    model_name : str
        MLIP model name.
    csv_flush_every : int
        Append buffered rows to md_energy.csv every N log_properties calls so
        the file grows incrementally instead of appearing only at end of run.
    plot : bool
        If True, write the md_energy/md_temperature (and NPT md_pressure/
        md_volume) PNGs. Defaults to False -- plotting is opt-in; md_energy.csv
        is always written and can be plotted later.
    run_context : RunContext, optional
        Declares the command, batch identity, and where each parameter value
        came from. When omitted the record still gets written, with every
        parameter tagged ``unspecified``.
    device_requested : str
        The device as asked for (e.g. ``'auto'``), recorded for provenance.
    device_resolved : str
        The device actually used (e.g. ``'cuda'``).
    uma_task : str, optional
        UMA task actually used, recorded for provenance. Ignored for
        non-UMA models.
    sevennet_task : str, optional
        SevenNet inference task, recorded in the run record when the model is
        a ``7net*`` tag. No default: see ``validate_mlip``.
    mace_head : str, optional
        MACE head actually used, recorded for provenance. Ignored for
        non-MACE models.

    (Other parameters as in :func:`setup_dynamics`.)
    """
    # Validated before the record is opened, not at setup_dynamics time: a
    # malformed mask must not leave behind a record stuck in state "running".
    barostat_mask = normalize_barostat_mask(barostat_mask)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    traj_file = output_path / "md.traj"
    csv_file = output_path / "md_energy.csv"
    energy_plot = output_path / "md_energy.png"
    temp_plot = output_path / "md_temperature.png"

    if resume and not (traj_file.exists() and csv_file.exists()):
        raise FileNotFoundError(
            f"Cannot resume: expected both {traj_file} and {csv_file} to exist."
        )

    log_data = {
        "step": [], "time(fs)": [], "temperature(K)": [],
        "total_energy(eV)": [], "potential_energy(eV)": [], "kinetic_energy(eV)": [],
    }
    if ensemble == "npt":
        log_data["pressure(GPa)"] = []
        log_data["volume(A^3)"] = []

    prior_steps = 0
    n_prior_rows = 0
    if resume:
        prior_df = pd.read_csv(csv_file)
        n_prior_rows = len(prior_df)
        for col in log_data:
            if col in prior_df.columns:
                log_data[col].extend(prior_df[col].tolist())
        prior_steps = int(prior_df["step"].iloc[-1]) if len(prior_df) else 0
        logger.info("Resuming MD from step %d (prior frames: %d)", prior_steps, len(prior_df))

    record = RunRecord.begin(
        output_path,
        command="md",
        stage_kind="md-resume" if resume else "md",
        parameters={
            "ensemble": ensemble, "steps": steps, "temperature": temperature,
            "pressure": pressure, "timestep": timestep,
            "thermostat": thermostat, "barostat": barostat,
            "barostat_mask": barostat_mask,
            "friction": friction, "ttime": ttime, "taut": taut, "taup": taup,
            "log_interval": log_interval, "traj_interval": traj_interval,
        },
        inputs={"n_atoms": len(atoms), "formula": atoms.get_chemical_formula()},
        provenance=collect_provenance(
            mlip_model=model_name,
            device_requested=device_requested,
            device_resolved=device_resolved,
            uma_task=uma_task,
            mace_head=mace_head,
            sevennet_task=sevennet_task,
        ),
        run_context=run_context,
        append=resume,
    )

    dyn = setup_dynamics(
        atoms, ensemble=ensemble, thermostat=thermostat, barostat=barostat,
        barostat_mask=barostat_mask,
        temperature=temperature, pressure=pressure, timestep=timestep,
        friction=friction, ttime=ttime, pfactor=pfactor,
        taut=taut, taup=taup, compressibility=compressibility,
        set_velocities=not resume,
    )
    if resume:
        dyn.nsteps = prior_steps

    traj_mode = "a" if resume else "w"
    traj_writer = Trajectory(str(traj_file), traj_mode, atoms)
    dyn.attach(traj_writer.write, interval=traj_interval)

    stress_log = (ensemble == "npt")
    dyn.attach(MDLogger(dyn, atoms, sys.stdout, header=True, stress=stress_log), interval=log_interval)
    if log_path:
        dyn.attach(MDLogger(dyn, atoms, log_path, header=True, stress=stress_log), interval=log_interval)

    # Incremental CSV writer: append a batch of buffered rows every
    # csv_flush_every log calls so the file grows during the run and survives
    # mid-trajectory crashes. On a fresh run start with a header; on resume
    # we leave the existing CSV alone (prior rows already on disk).
    flush_buffer = {k: [] for k in log_data}
    csv_columns = list(log_data.keys())
    if not resume:
        pd.DataFrame(columns=csv_columns).to_csv(csv_file, index=False)
    flush_counter = [0]

    def _flush_csv():
        if not flush_buffer["step"]:
            return
        pd.DataFrame(flush_buffer).to_csv(
            csv_file, mode="a", header=False, index=False,
        )
        for k in flush_buffer:
            flush_buffer[k].clear()

    def log_properties():
        step = dyn.get_number_of_steps()
        row = {
            "step": step,
            "time(fs)": step * timestep,
            "temperature(K)": atoms.get_temperature(),
            "total_energy(eV)": atoms.get_total_energy(),
            "potential_energy(eV)": atoms.get_potential_energy(),
            "kinetic_energy(eV)": atoms.get_kinetic_energy(),
        }
        if ensemble == "npt":
            stress = atoms.get_stress(voigt=False)
            pressure_ase = -stress.trace() / 3.0
            row["pressure(GPa)"] = pressure_ase / GPA_TO_EV_PER_ANG3
            row["volume(A^3)"] = atoms.get_volume()

        for k, v in row.items():
            log_data[k].append(v)
            flush_buffer[k].append(v)

        flush_counter[0] += 1
        if csv_flush_every > 0 and flush_counter[0] >= csv_flush_every:
            _flush_csv()
            flush_counter[0] = 0

    dyn.attach(log_properties, interval=log_interval)

    try:
        dyn.run(steps)
    except Exception as exc:
        traj_writer.close()
        _flush_csv()
        record.complete(status="failed", results={"error": str(exc)})
        raise
    traj_writer.close()
    _flush_csv()  # final tail of buffered rows

    df = pd.DataFrame(log_data)

    # Statistics describe THIS segment only: on resume, log_data was seeded
    # with the prior run's rows, so df holds the whole history.
    record.complete(
        status="converged",
        steps=int(dyn.get_number_of_steps()),
        results=_md_statistics(df.iloc[n_prior_rows:], len(atoms)),
    )

    # Plotting is opt-in. md_energy.csv is already written above, so returning
    # here just skips the PNGs (which are IO that can dominate short runs).
    if not plot:
        return

    # Plot energy: potential/total on the left axis, kinetic on a twin
    # right axis -- kinetic is orders of magnitude smaller than |E_pot|,
    # so a shared axis flattens every trace into a featureless line.
    # Offset the two y-ranges so the traces occupy separate halves.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["time(fs)"], df["total_energy(eV)"], label="Total", linewidth=1.5)
    ax.plot(df["time(fs)"], df["potential_energy(eV)"], label="Potential", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(df["time(fs)"], df["kinetic_energy(eV)"], color="tab:purple",
             alpha=0.7, label="Kinetic (right)")
    lo1 = min(df["total_energy(eV)"].min(), df["potential_energy(eV)"].min())
    hi1 = max(df["total_energy(eV)"].max(), df["potential_energy(eV)"].max())
    r1 = float(hi1 - lo1) or 1.0
    ax.set_ylim(lo1 - 0.05 * r1, hi1 + 1.15 * r1)
    lo2 = df["kinetic_energy(eV)"].min()
    hi2 = df["kinetic_energy(eV)"].max()
    r2 = float(hi2 - lo2) or 1.0
    ax2.set_ylim(lo2 - 1.15 * r2, hi2 + 0.05 * r2)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Energy (eV)")
    ax2.set_ylabel("Kinetic energy (eV)", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_title(f"MD Energy ({ensemble.upper()})")
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(energy_plot, dpi=150)
    plt.close(fig)

    # Plot temperature
    plt.figure(figsize=(8, 5))
    plt.plot(df["time(fs)"], df["temperature(K)"], color="orange", linewidth=1.5)
    if ensemble in ["nvt", "npt"]:
        plt.axhline(y=temperature, color="r", linestyle="--", label=f"Target: {temperature} K", alpha=0.5)
    plt.xlabel("Time (fs)")
    plt.ylabel("Temperature (K)")
    plt.title(f"MD Temperature ({ensemble.upper()})")
    if ensemble in ["nvt", "npt"]:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(temp_plot, dpi=150)
    plt.close()

    # NPT-specific plots
    if ensemble == "npt":
        pressure_plot = output_path / "md_pressure.png"
        volume_plot = output_path / "md_volume.png"

        plt.figure(figsize=(8, 5))
        plt.plot(df["time(fs)"], df["pressure(GPa)"], color="blue", linewidth=1.5)
        plt.axhline(y=pressure, color="r", linestyle="--", label=f"Target: {pressure} GPa", alpha=0.5)
        plt.xlabel("Time (fs)")
        plt.ylabel("Pressure (GPa)")
        plt.title("MD Pressure (NPT)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pressure_plot, dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(df["time(fs)"], df["volume(A^3)"], color="green", linewidth=1.5)
        plt.xlabel("Time (fs)")
        plt.ylabel("Volume (Ang^3)")
        plt.title("MD Volume (NPT)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(volume_plot, dpi=150)
        plt.close()
