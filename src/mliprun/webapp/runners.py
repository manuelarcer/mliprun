"""One runner per calc type: call the core engine, then build the result payload.

Each runner writes into ``output_dir = run_dir`` and streams live progress
through an :class:`ObserverAdapter`, which turns the core observer payloads
into ``events.jsonl`` lines. AutoNEB has no core hook; it gets the same live
data from a FIRE subclass (:func:`make_observed_fire`) passed as its
optimizer.
"""
from __future__ import annotations

import csv
import glob
import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional

from mliprun.webapp.events import encode_frame

logger = logging.getLogger("mliprun.webapp")

_MD_COLUMNS = {
    "time(fs)": "time_fs",
    "potential_energy(eV)": "epot",
    "kinetic_energy(eV)": "ekin",
    "total_energy(eV)": "etot",
    "temperature(K)": "temperature",
    "pressure(GPa)": "pressure_GPa",
    "volume(A^3)": "volume_A3",
}

_FRAME_KEYS = ("frame", "final_frame", "band")


class ObserverAdapter:
    """Turn core observer payloads into events.

    Parameters
    ----------
    writer : EventWriter
        Where events go.
    structure_every : int
        Attach positions on call ``k`` of each event type iff
        ``structure_every > 0`` and ``k % structure_every == 0``.
    """

    def __init__(self, writer, structure_every: int = 5):
        self.writer = writer
        self.structure_every = int(structure_every or 0)
        self.counts: dict = {}
        self.cancelled = False

    def _with_frame(self, event_type: str) -> bool:
        k = self.counts.get(event_type, 0)
        self.counts[event_type] = k + 1
        return self.structure_every > 0 and k % self.structure_every == 0

    def _check_cancel(self) -> None:
        # SIGTERM is the cancellation mechanism (it raises in the main
        # thread); this flag is only a cooperative backstop.
        if self.cancelled:
            from mliprun.webapp.worker import JobCancelled

            raise JobCancelled("cancelled by user")

    def opt(self, payload: dict) -> None:
        """``run_optimization`` observer -> ``opt_step``."""
        self._check_cancel()
        event = {"step": payload["step"], "energy": payload["energy"],
                 "fmax": payload["fmax"]}
        if self._with_frame("opt_step"):
            event["frame"] = encode_frame(payload["positions"], payload["cell"])
        self.writer.emit("opt_step", **event)

    def md(self, payload: dict) -> None:
        """``run_md`` observer -> ``md_step``."""
        self._check_cancel()
        row = payload["row"]
        event = {"step": int(row["step"])}
        for column, key in _MD_COLUMNS.items():
            if column in row:
                event[key] = row[column]
        if self._with_frame("md_step"):
            event["frame"] = encode_frame(payload["positions"], payload["cell"])
        self.writer.emit("md_step", **event)

    def freq(self, payload: dict) -> None:
        """``run_frequencies`` observer -> ``freq_progress``."""
        self._check_cancel()
        self.writer.emit("freq_progress", done=int(payload["done"]),
                         total=payload["total"])

    def neb(self, payload: dict, autoneb: Optional[dict] = None) -> None:
        """``CustomNEB.run_neb`` observer (and AutoNEB callback) -> ``neb_iter``."""
        self._check_cancel()
        energies = list(payload["energies"])
        e0 = energies[0] if energies else None
        rel = [None if (e is None or e0 is None) else e - e0 for e in energies]
        known = [e for e in energies if e is not None]
        highest = energies.index(max(known)) if known else None  # first argmax
        barrier = payload.get("barrier")
        if barrier is None and known and e0 is not None:
            barrier = max(known) - e0
        event = {"iteration": int(payload["step"]), "energies": energies,
                 "rel_energies": rel, "fmax": payload["fmax"],
                 "barrier": barrier, "highest_image": highest,
                 "n_images": len(energies)}
        if self._with_frame("neb_iter"):
            cell = payload["cell"]
            event["band"] = [encode_frame(p, cell) for p in payload["positions"]]
        if autoneb is not None:
            event["autoneb"] = autoneb
        self.writer.emit("neb_iter", **event)


def make_observed_fire(adapter: ObserverAdapter):
    """A FIRE subclass whose instances report every step to ``adapter``.

    ASE's AutoNEB builds a fresh optimizer per sub-band relaxation, so the
    callback is attached in ``__init__``. Energies are read from each
    image's calculator cache only (never triggering a new calculation); a
    missing one is reported as ``None``.
    """
    from ase.optimize import FIRE

    from mliprun.core.utils import calc_fmax

    counter = {"iteration": 0}

    class ObservedFIRE(FIRE):
        def __init__(self, atoms, *args, **kwargs):
            self._observed_target = atoms
            super().__init__(atoms, *args, **kwargs)
            self.attach(self._report, interval=1)

        def _report(self):
            target = self._observed_target
            energies = []
            for img in target.images:
                try:
                    value = img.calc.get_property("energy", img,
                                                  allow_calculation=False)
                    energies.append(None if value is None else float(value))
                except Exception:  # noqa: BLE001 - an unreadable energy is reported as null
                    energies.append(None)
            try:
                fmax = calc_fmax(target.get_forces())
            except Exception:  # noqa: BLE001
                fmax = None
            payload = {"step": counter["iteration"], "fmax": fmax, "barrier": None,
                       "energies": energies,
                       "positions": [img.get_positions() for img in target.images],
                       "cell": target.images[0].get_cell().array}
            counter["iteration"] += 1
            adapter.neb(payload, autoneb={"sub_step": int(self.nsteps)})

    return ObservedFIRE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_record(run_dir: Path) -> dict:
    """The run's ``mliprun_run.json`` (empty dict when missing or unreadable)."""
    try:
        return json.loads((Path(run_dir) / "mliprun_run.json").read_text())
    except (OSError, ValueError):
        return {}


def record_common(run_dir: Path, calc_type: str) -> dict:
    """The common result keys taken from the run record."""
    rec = read_record(run_dir)
    stages = rec.get("stages") or [{}]
    return {"calc_type": calc_type,
            "record_status": rec.get("status"),
            "record_results": stages[-1].get("results"),
            "walltime_s": (rec.get("provenance") or {}).get("walltime_s"),
            "_record_steps": stages[-1].get("steps")}


def _truthy(text: str) -> bool:
    return str(text).strip().lower() in ("true", "1", "yes")


def _float_or_none(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def strip_frames(result: dict) -> dict:
    """``result`` without positions (for ``summary.json``)."""
    return {k: v for k, v in result.items() if k not in _FRAME_KEYS}


class RunInputs:
    """Everything a runner needs for one job."""

    def __init__(self, job: dict, run_dir: Path, spec: dict, device_resolved: str,
                 adapter: ObserverAdapter, structures: dict,
                 phase: Callable[[str, str], None]):
        self.job = job
        self.run_dir = Path(run_dir)
        self.params = job.get("params") or {}
        self.spec = spec
        self.device_resolved = device_resolved
        self.adapter = adapter
        self.structures = structures
        self.phase = phase
        self.calc = None

    def common_kwargs(self) -> dict:
        return {"model_name": self.spec["name"],
                "device_requested": self.spec.get("device") or "auto",
                "device_resolved": self.device_resolved,
                "uma_task": self.spec.get("uma_task"),
                "mace_head": self.spec.get("mace_head"),
                "sevennet_task": self.spec.get("sevennet_task")}

    def run_context(self, command: str):
        from mliprun.core.run_record import RunContext

        inputs = self.job.get("inputs") or {}
        names = {role: meta.get("filename") for role, meta in inputs.items()}
        extra = {"source": "webapp", "job_id": self.job.get("job_id"),
                 "structure_filename": names.get("structure") or names.get("band")
                 or f"{names.get('initial')} -> {names.get('final')}"}
        return RunContext(command=command, mode="one-off",
                          param_sources=dict(self.job.get("param_sources") or {}),
                          extra_inputs=extra)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_singlepoint_job(ri: RunInputs) -> dict:
    """Single point via ``core.singlepoint.run_singlepoint``."""
    from mliprun.core.singlepoint import run_singlepoint

    atoms = ri.structures["structure"]
    atoms.calc = ri.calc
    stress = {"auto": None, "on": True, "off": False}[ri.params.get("stress", "auto")]
    res = run_singlepoint(atoms, output_dir=ri.run_dir, prefix="singlepoint",
                          stress=stress, run_context=ri.run_context("singlepoint"),
                          **ri.common_kwargs())
    ri.phase("postprocess", "collecting results")
    forces = []
    with open(ri.run_dir / "singlepoint_forces.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            forces.append({
                "index": int(row["index"]), "symbol": row["symbol"],
                "fx": float(row["fx"]), "fy": float(row["fy"]), "fz": float(row["fz"]),
                "f_norm": float(row["f_norm"]),
                "free": any(_truthy(row[k]) for k in ("free_x", "free_y", "free_z"))})
    out = record_common(ri.run_dir, "singlepoint")
    out.update({"energy_eV": res.get("energy_eV"),
                "fmax_free_eV_per_A": res.get("fmax_free_eV_per_A"),
                "fmax_all_eV_per_A": res.get("fmax_all_eV_per_A"),
                "stress_GPa": res.get("stress_GPa"),
                "stress_unavailable_reason": res.get("stress_unavailable_reason"),
                "forces": forces,
                "final_frame": encode_frame(atoms.get_positions(), atoms.get_cell())})
    return out


def run_optimize_job(ri: RunInputs) -> dict:
    """Geometry optimization via ``core.optimize.run_optimization``."""
    from mliprun.core.optimize import run_optimization

    atoms = ri.structures["structure"]
    atoms.calc = ri.calc
    p = ri.params
    converged = run_optimization(
        atoms, optimizer=p["optimizer"], fmax=p["fmax"], max_steps=p["max_steps"],
        trajectory="opt.traj", logfile="opt.log", output_dir=ri.run_dir,
        verbose=True, relax_cell=p["relax_cell"], plot=p["plot"],
        run_context=ri.run_context("optimize"), observer=ri.adapter.opt,
        **ri.common_kwargs())
    ri.phase("postprocess", "collecting results")
    out = record_common(ri.run_dir, "optimize")
    results = out["record_results"] or {}
    out.update({"converged": bool(converged),
                "final_energy_eV": results.get("final_energy_eV"),
                "final_fmax_eV_per_A": results.get("final_fmax_eV_per_A"),
                "steps": out["_record_steps"],
                "final_frame": encode_frame(atoms.get_positions(), atoms.get_cell())})
    return out


def _mode_index(path: str) -> int:
    match = re.search(r"\.(\d+)\.traj$", path)
    return int(match.group(1)) if match else -1


def run_freq_job(ri: RunInputs) -> dict:
    """Vibrational frequencies via ``core.vibrations.run_frequencies``."""
    from mliprun.core.vibrations import run_frequencies
    from mliprun.webapp.schema import parse_index_list

    atoms = ri.structures["structure"]
    atoms.calc = ri.calc
    p = ri.params
    indices = (parse_index_list(p["indices"], len(atoms))
               if p.get("indices") is not None else None)
    res = run_frequencies(
        atoms, output_dir=ri.run_dir, prefix="freq", indices=indices,
        delta=p["delta"], nfree=p["nfree"], direction=p["direction"],
        method=p["method"], write_modes=p["write_modes"],
        imaginary_floor=p["imaginary_floor"], expect_fmax=p.get("expect_fmax"),
        run_context=ri.run_context("freq"), observer=ri.adapter.freq,
        **ri.common_kwargs())
    ri.phase("postprocess", "collecting results")
    modes = []
    with open(ri.run_dir / "freq_frequencies.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            modes.append({"index": int(row["mode_index"]),
                          "frequency_cm1": _float_or_none(row["frequency_cm-1"]),
                          "energy_meV": _float_or_none(row["energy_meV"]),
                          "imaginary": _truthy(row["imaginary"])})
    files = sorted((Path(f).name for f in glob.glob(str(ri.run_dir / "freq.*.traj"))),
                   key=_mode_index)
    out = record_common(ri.run_dir, "freq")
    out.update({"modes": modes, "n_modes": res.get("n_modes"),
                "n_imaginary": res.get("n_imaginary"),
                "n_imaginary_raw": res.get("n_imaginary_raw"),
                "imaginary_floor_cm1": res.get("imaginary_floor_cm-1"),
                "zpe_eV": res.get("zpe_eV"),
                "fmax_at_input_free_eV_per_A": res.get("fmax_at_input_free_eV_per_A"),
                "fmax_warning": res.get("fmax_warning"),
                "mode_files": files,
                "final_frame": encode_frame(atoms.get_positions(), atoms.get_cell())})
    return out


_MD_KWARGS = ("ensemble", "thermostat", "barostat", "temperature", "pressure",
              "timestep", "friction", "ttime", "taut", "taup", "compressibility",
              "steps", "log_interval", "traj_interval", "plot")


def run_md_job(ri: RunInputs) -> dict:
    """Molecular dynamics via ``core.md.run_md``. Hidden params use core defaults."""
    from mliprun.core.md import PFACTOR_GPA_FS2, run_md
    from mliprun.webapp.schema import parse_mask3

    atoms = ri.structures["structure"]
    atoms.calc = ri.calc
    p = ri.params
    kwargs = {k: p[k] for k in _MD_KWARGS if k in p}
    if "barostat_mask" in p:
        kwargs["barostat_mask"] = parse_mask3(p["barostat_mask"])
    if p.get("pfactor") is not None:
        kwargs["pfactor"] = p["pfactor"] * PFACTOR_GPA_FS2  # GPa fs^2 -> ASE units
    run_md(atoms, output_dir=ri.run_dir, resume=False,
           run_context=ri.run_context("md"), observer=ri.adapter.md,
           **kwargs, **ri.common_kwargs())
    ri.phase("postprocess", "collecting results")
    out = record_common(ri.run_dir, "md")
    results = out["record_results"] or {}
    for key, value in results.items():
        out.setdefault(key, value)
    out.update({"steps": out["_record_steps"], "ensemble": p["ensemble"],
                "final_frame": encode_frame(atoms.get_positions(), atoms.get_cell())})
    return out


def _make_custom_neb(ri: RunInputs, initial, final, num_images: int, relax=None,
                     interp_fmax=0.1, interp_steps=100):
    from mliprun.core.neb import CustomNEB

    spec = ri.spec
    neb = CustomNEB(initial=initial, final=final, num_images=num_images,
                    interp_fmax=interp_fmax, interp_steps=interp_steps,
                    fmax=ri.params["fmax"], mlip=spec["name"],
                    uma_task=spec.get("uma_task"), mace_head=spec.get("mace_head"),
                    sevennet_task=spec.get("sevennet_task"), output_dir=ri.run_dir,
                    relax_atoms=relax, logfile="neb.log", device=ri.device_resolved)
    if spec["name"] == "emt":
        from ase.calculators.emt import EMT

        neb.setup_calculator = lambda *a, **k: EMT()
    return neb


def _endpoints(ri: RunInputs, neb) -> None:
    from mliprun.core.params_io import write_endpoint_results

    p = ri.params
    ri.phase("endpoints", "optimizing the endpoints")
    results = neb.optimize_endpoints(endpoint_fmax=p["endpoint_fmax"],
                                     optimizer=p["endpoint_optimizer"],
                                     max_steps=p["endpoint_max_steps"])
    write_endpoint_results(ri.run_dir / "endpoint_optimization.txt", results)


def _band_result(out: dict, energies: list, band: list) -> dict:
    from mliprun.core.neb import summarize_neb_path

    summary = summarize_neb_path(energies) if energies else {}
    e0 = energies[0] if energies else None
    out.update({"energies_eV": energies,
                "relative_energies_eV": [e - e0 for e in energies] if energies else [],
                "forward_barrier_eV": summary.get("forward_barrier_eV"),
                "reverse_barrier_eV": summary.get("reverse_barrier_eV"),
                "reaction_energy_eV": summary.get("reaction_energy_eV"),
                "ts_image_index": summary.get("ts_image_index"),
                "ts_at_endpoint": summary.get("ts_at_endpoint"),
                "band": band})
    return out


def run_neb_job(ri: RunInputs) -> dict:
    """NEB in pair or band mode (mirrors ``neb run``)."""
    from mliprun.cli.commands.neb import NEB_OPTIMIZER_MAP
    from mliprun.webapp.schema import parse_index_list

    p = ri.params
    if "band" in ri.structures:
        frames = ri.structures["band"]
        neb = _make_custom_neb(ri, frames[0], frames[-1], num_images=len(frames) - 2)
        neb.images = [neb.initial, *frames[1:-1], neb.final]
    else:
        initial, final = ri.structures["initial"], ri.structures["final"]
        relax = (parse_index_list(p["relax_atoms"], len(initial))
                 if p.get("relax_atoms") is not None else None)
        neb = _make_custom_neb(ri, initial, final, num_images=p["n_images"], relax=relax,
                               interp_fmax=p.get("interp_fmax", 0.1),
                               interp_steps=p.get("interp_steps", 100))
        if p["optimize_endpoints"]:
            _endpoints(ri, neb)
            neb.images = neb.setup_neb()
        if p["interpolation"] == "idpp":
            if relax is not None:
                logger.warning("relax_atoms is set, so IDPP interpolation is skipped "
                               "(highly constrained mode); the band is linear.")
            ri.phase("interpolation", "IDPP interpolation")
            neb.interpolate_idpp()
        else:
            # setup_neb drops FixAtoms from the intermediate images for the
            # interpolation; interpolate_idpp puts it back. Without IDPP the
            # worker restores it the same way, so fixed atoms stay fixed.
            from ase.constraints import FixAtoms

            fix = getattr(neb, "_fix_atoms_constraint", None)
            if fix is not None and relax is None:
                for img in neb.images[1:-1]:
                    if not any(isinstance(c, FixAtoms) for c in img.constraints):
                        img.set_constraint(list(img.constraints) + [fix])
    ri.phase("run", "NEB optimization")
    neb.run_neb(optimizer=NEB_OPTIMIZER_MAP[p["neb_optimizer"]], climb=p["climb"],
                max_steps=p["neb_max_steps"], k=p["k"], dyneb=p["dyneb"],
                scale_fmax=p.get("scale_fmax", 0.0), plot=p["plot"],
                run_context=ri.run_context("neb"), observer=ri.adapter.neb)
    ri.phase("postprocess", "collecting results")
    df = neb.process_results()
    if p["plot"]:
        neb.plot_results(df)
    neb.export_poscars()
    out = record_common(ri.run_dir, "neb")
    results = out["record_results"] or {}
    energies = [float(img.get_potential_energy()) for img in neb.images]
    band = [encode_frame(img.get_positions(), img.get_cell()) for img in neb.images]
    _band_result(out, energies, band)
    out.update({"converged": out["record_status"] == "converged",
                "steps": out["_record_steps"],
                "final_fmax_eV_per_A": results.get("final_fmax_eV_per_A")})
    return out


def run_autoneb_job(ri: RunInputs) -> dict:
    """AutoNEB (mirrors ``autoneb run``; FIRE, ``n_simul=1``, prefix ``autoneb``)."""
    from ase.io import read

    p = ri.params
    neb = _make_custom_neb(ri, ri.structures["initial"], ri.structures["final"],
                           num_images=5)
    if p["optimize_endpoints"]:
        _endpoints(ri, neb)
    ri.phase("run", "AutoNEB")
    neb.run_autoneb(n_simul=1, n_max=p["n_max"], k=p["k"], climb=p["climb"],
                    optimizer=make_observed_fire(ri.adapter),
                    space_energy_ratio=p["space_energy_ratio"],
                    interpolate_method=p["interpolate_method"],
                    maxsteps=p["maxsteps"], prefix="autoneb",
                    run_context=ri.run_context("autoneb"))
    ri.phase("postprocess", "collecting the AutoNEB images")
    files = sorted(glob.glob(str(ri.run_dir / "autoneb[0-9][0-9][0-9].traj")))
    images = [read(f, index=-1) for f in files]
    energies = [float(img.get_potential_energy()) for img in images]
    band = [encode_frame(img.get_positions(), img.get_cell()) for img in images]
    if energies:
        emin = min(energies)
        with open(ri.run_dir / "autoneb_energy_profile.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "energy", "rel_energy"])
            for i, e in enumerate(energies):
                writer.writerow([i, repr(e), repr(e - emin)])
    out = record_common(ri.run_dir, "autoneb")
    _band_result(out, energies, band)
    out.update({"converged": out["record_status"] == "converged", "steps": None,
                "final_fmax_eV_per_A": None})
    return out


RUNNERS = {
    "singlepoint": run_singlepoint_job,
    "optimize": run_optimize_job,
    "freq": run_freq_job,
    "md": run_md_job,
    "neb": run_neb_job,
    "autoneb": run_autoneb_job,
}


def finalize(result: dict) -> dict:
    """Drop internal keys from a runner's result."""
    return {k: v for k, v in result.items() if not k.startswith("_")}
