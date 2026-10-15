"""The worker process end to end, with EMT (plan section 8).

Each test writes a ``job.json`` the way the Backend does, pre-writes the
server's ``job_status`` lines, and runs ``python -m mliprun.webapp.worker``
as a real subprocess.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from ase.build import add_adsorbate, bulk, fcc100
from ase.constraints import FixAtoms
from ase.io import write
from ase.mep import NEB

from mliprun.webapp.events import EventWriter, read_events
from mliprun.webapp.schema import validate_submission

SRC = str(Path(__file__).resolve().parents[2] / "src")


def _env(**extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1", **extra)
    return env


def _meta(frames):
    atoms = frames[-1]
    return {"natoms": len(atoms), "symbols": atoms.get_chemical_symbols(),
            "n_frames": len(frames), "pbc": [bool(x) for x in atoms.pbc],
            "frames_same_symbols": True}


def make_run_dir(root: Path, calc_type: str, params: dict, files: dict,
                 structure_every: int = 2) -> Path:
    """files: role -> (Atoms or [Atoms], filename, ase format)."""
    run_dir = root / calc_type
    inputs, metas = {}, {}
    for role, (obj, filename, fmt) in files.items():
        frames = obj if isinstance(obj, list) else [obj]
        target = run_dir / "inputs" / role / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        write(str(target), frames if len(frames) > 1 else frames[0], format=fmt)
        inputs[role] = {"upload_id": "0123456789ab", "filename": filename,
                        "path": f"inputs/{role}/{filename}", "format": fmt}
        metas[role] = _meta(frames if role != "band" else frames[:1])
        if role == "band":
            metas[role]["n_frames"] = len(frames)
    v = validate_submission(calc_type, {"name": "emt", "device": "auto"}, params, metas)
    job_id = f"20261015-142301-{calc_type}-a1b2c3"
    job = {"schema_version": 1, "job_id": job_id, "calc_type": calc_type,
           "label": "test", "created_at": "2026-10-15T14:23:01.123Z",
           "calculator": v["calculator"], "params": v["params"],
           "param_sources": v["param_sources"], "structure_every": structure_every,
           "inputs": inputs}
    (run_dir / "job.json").write_text(json.dumps(job))
    with EventWriter(run_dir / "events.jsonl", job_id) as w:
        w.emit("job_status", status="pending", reason=None, returncode=None)
        w.emit("job_status", status="running", reason=None, returncode=None)
    return run_dir


def run_worker(run_dir: Path, timeout=300, **env) -> int:
    with open(run_dir / "worker.log", "ab") as log:
        return subprocess.call([sys.executable, "-m", "mliprun.webapp.worker", str(run_dir)],
                               cwd=run_dir, stdout=log, stderr=log, env=_env(**env),
                               timeout=timeout)


def check_done(run_dir: Path, rc: int) -> dict:
    log = (run_dir / "worker.log").read_text(errors="replace")
    assert rc == 0, log[-3000:]
    events = read_events(run_dir / "events.jsonl")
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert all(e["job_id"] == events[0]["job_id"] for e in events)
    assert all(e["type"] != "job_status" for e in events[2:])
    results = [e for e in events if e["type"] == "result"]
    assert len(results) == 1 and events[-1]["type"] == "result"
    assert not [e for e in events if e["type"] == "error"]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "done" and summary["error"] is None
    assert summary["reproduce"]["status"] == "no"  # emt
    assert "final_frame" not in summary["result"] and "band" not in summary["result"]
    assert (run_dir / "reproduce.sh").exists() and (run_dir / "reproduce.json").exists()
    phases = [e["phase"] for e in events if e["type"] == "phase"]
    assert phases[0] == "setup" and phases[-1] == "postprocess"
    assert {"load_model", "run"} <= set(phases)
    result = results[0]
    rec = json.loads((run_dir / "mliprun_run.json").read_text())
    assert result["record_status"] == rec["status"]
    assert result["record_results"] == json.loads(json.dumps(rec["stages"][-1]["results"]))
    return {"events": events, "result": result, "summary": summary}


def _cu(rattle=True):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    if rattle:
        atoms.rattle(0.05, seed=1)
    return atoms


def _neb_pair():
    slab = fcc100("Cu", (2, 2, 3), vacuum=6.0)
    slab.set_constraint(FixAtoms(indices=[a.index for a in slab if a.tag == 3]))
    initial = slab.copy()
    add_adsorbate(initial, "Cu", 1.7, "hollow")
    final = slab.copy()
    add_adsorbate(final, "Cu", 1.7, "hollow")
    final.positions[-1, 0] += slab.cell[0, 0] / 2
    return initial, final


def test_singlepoint(tmp_path):
    rd = make_run_dir(tmp_path, "singlepoint", {}, {"structure": (_cu(), "POSCAR", "vasp")})
    out = check_done(rd, run_worker(rd))
    r = out["result"]
    from ase.calculators.emt import EMT
    from ase.io import read
    atoms = read(rd / "inputs/structure/POSCAR")
    atoms.calc = EMT()
    assert r["energy_eV"] == pytest.approx(atoms.get_potential_energy(), abs=1e-8)
    assert len(r["forces"]) == len(atoms)
    assert r["forces"][1]["fx"] == pytest.approx(atoms.get_forces()[1, 0], abs=1e-8)
    assert r["stress_GPa"] is not None and len(r["stress_GPa"]) == 6
    assert len(r["final_frame"]["positions"]) == len(atoms)


def test_optimize_frames_follow_structure_every(tmp_path):
    rd = make_run_dir(tmp_path, "optimize", {"fmax": 0.05},
                      {"structure": (_cu(), "POSCAR", "vasp")}, structure_every=2)
    out = check_done(rd, run_worker(rd))
    steps = [e for e in out["events"] if e["type"] == "opt_step"]
    rows = (rd / "opt_convergence.csv").read_text().strip().splitlines()[1:]
    assert len(steps) == len(rows) > 2
    assert [("frame" in e) for e in steps] == [i % 2 == 0 for i in range(len(steps))]
    r = out["result"]
    assert r["converged"] is True and r["final_fmax_eV_per_A"] <= 0.05
    assert r["steps"] == len(steps) - 1
    assert r["final_energy_eV"] == pytest.approx(steps[-1]["energy"], abs=1e-12)


def test_freq(tmp_path):
    rd = make_run_dir(tmp_path, "freq", {"indices": "0-1"},
                      {"structure": (_cu(rattle=False), "cu.xyz", "extxyz")})
    out = check_done(rd, run_worker(rd))
    prog = [e for e in out["events"] if e["type"] == "freq_progress"]
    assert prog[-1]["done"] == prog[-1]["total"] == 1 + 6 * 2
    r = out["result"]
    assert r["n_modes"] == len(r["modes"]) == 6
    assert r["record_results"]["n_force_calls"] == 13
    assert all(m["imaginary"] is False for m in r["modes"])


def test_md_nve(tmp_path):
    rd = make_run_dir(tmp_path, "md", {"ensemble": "nve", "steps": 20, "log_interval": 5},
                      {"structure": (_cu(), "POSCAR", "vasp")})
    out = check_done(rd, run_worker(rd))
    md = [e for e in out["events"] if e["type"] == "md_step"]
    rows = (rd / "md_energy.csv").read_text().strip().splitlines()[1:]
    assert len(md) == len(rows) == 5
    assert [e["step"] for e in md] == [0, 5, 10, 15, 20]
    assert abs(md[-1]["etot"] - md[0]["etot"]) < 1e-2
    assert "pressure_GPa" not in md[0]
    r = out["result"]
    assert r["ensemble"] == "nve" and r["steps"] == 20
    assert r["mean_total_energy_eV"] == r["record_results"]["mean_total_energy_eV"]


def test_neb_pair(tmp_path):
    initial, final = _neb_pair()
    rd = make_run_dir(tmp_path, "neb", {"n_images": 3, "fmax": 0.1},
                      {"initial": (initial, "ini.vasp", "vasp"),
                       "final": (final, "fin.vasp", "vasp")})
    out = check_done(rd, run_worker(rd))
    assert (rd / "inputs/initial.vasp").exists() and (rd / "inputs/final.vasp").exists()
    phases = [e["phase"] for e in out["events"] if e["type"] == "phase"]
    assert phases.index("endpoints") < phases.index("interpolation") < phases.index("run")
    iters = [e for e in out["events"] if e["type"] == "neb_iter"]
    assert iters and iters[0]["n_images"] == 5 and "band" in iters[0]
    r = out["result"]
    assert r["converged"] is True and r["forward_barrier_eV"] > 0
    assert r["ts_at_endpoint"] is False and len(r["band"]) == 5
    assert r["forward_barrier_eV"] == pytest.approx(r["record_results"]["forward_barrier_eV"],
                                                    abs=1e-12)
    assert (rd / "00" / "POSCAR").exists()


def test_neb_band(tmp_path):
    initial, final = _neb_pair()
    images = [initial] + [initial.copy() for _ in range(3)] + [final]
    NEB(images).interpolate(mic=True)
    rd = make_run_dir(tmp_path, "neb", {"fmax": 0.1},
                      {"band": (images, "band.traj", "traj")})
    out = check_done(rd, run_worker(rd))
    phases = [e["phase"] for e in out["events"] if e["type"] == "phase"]
    assert "endpoints" not in phases and "interpolation" not in phases
    r = out["result"]
    assert len(r["energies_eV"]) == 5 and r["forward_barrier_eV"] > 0
    assert (rd / "inputs/initial.vasp").exists()


def test_autoneb(tmp_path):
    initial, final = _neb_pair()
    rd = make_run_dir(tmp_path, "autoneb", {"n_max": 4, "fmax": 0.1},
                      {"initial": (initial, "ini.vasp", "vasp"),
                       "final": (final, "fin.vasp", "vasp")})
    out = check_done(rd, run_worker(rd))
    iters = [e for e in out["events"] if e["type"] == "neb_iter"]
    assert iters and all("sub_step" in e["autoneb"] for e in iters)
    assert [e["iteration"] for e in iters] == list(range(len(iters)))
    r = out["result"]
    assert r["steps"] is None and r["final_fmax_eV_per_A"] is None
    assert len(r["energies_eV"]) == len(r["band"]) == 4
    assert r["forward_barrier_eV"] > 0


def test_sigterm_during_md_cancels(tmp_path):
    rd = make_run_dir(tmp_path, "md", {"ensemble": "nvt", "steps": 1000000,
                                       "log_interval": 20},
                      {"structure": (_cu(), "POSCAR", "vasp")})
    with open(rd / "worker.log", "ab") as log:
        proc = subprocess.Popen([sys.executable, "-m", "mliprun.webapp.worker", str(rd)],
                                cwd=rd, stdout=log, stderr=log, env=_env(),
                                start_new_session=True)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if any(e["type"] == "md_step" for e in read_events(rd / "events.jsonl")):
                break
            time.sleep(0.1)
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert rc == 3
    assert json.loads((rd / "summary.json").read_text())["status"] == "cancelled"
    assert json.loads((rd / "mliprun_run.json").read_text())["status"] == "failed"
    events = read_events(rd / "events.jsonl")
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert any(e["type"] == "log" and e["message"] == "cancelled by user" for e in events)
    assert not any(e["type"] == "result" for e in events)


def test_crash_hook_exits_after_first_event(tmp_path):
    rd = make_run_dir(tmp_path, "singlepoint", {}, {"structure": (_cu(), "POSCAR", "vasp")})
    rc = run_worker(rd, MLIPRUN_WEBAPP_TEST_CRASH="after_first_event")
    assert rc == 7
    assert len(read_events(rd / "events.jsonl")) == 3
    assert not (rd / "summary.json").exists()


def test_unreadable_job_json_exits_2(tmp_path):
    rd = tmp_path / "20261015-142301-optimize-a1b2c3"
    rd.mkdir()
    (rd / "job.json").write_text("{not json")
    assert run_worker(rd) == 2
    events = read_events(rd / "events.jsonl")
    assert events[-1]["type"] == "error" and events[-1]["job_id"] == rd.name
    assert json.loads((rd / "summary.json").read_text())["status"] == "failed"


def test_setup_failure_exits_2(tmp_path):
    rd = make_run_dir(tmp_path, "singlepoint", {}, {"structure": (_cu(), "POSCAR", "vasp")})
    job = json.loads((rd / "job.json").read_text())
    job["inputs"]["structure"]["path"] = "inputs/structure/missing"
    (rd / "job.json").write_text(json.dumps(job))
    assert run_worker(rd) == 2
    summary = json.loads((rd / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["error"]["message"]


def test_calculation_failure_exits_1(tmp_path):
    # An unknown optimizer passes no schema check here (job.json is edited
    # directly), so the core raises inside the calculation.
    rd = make_run_dir(tmp_path, "optimize", {}, {"structure": (_cu(), "POSCAR", "vasp")})
    job = json.loads((rd / "job.json").read_text())
    job["params"]["optimizer"] = "nope"
    (rd / "job.json").write_text(json.dumps(job))
    assert run_worker(rd) == 1
    events = read_events(rd / "events.jsonl")
    assert events[-1]["type"] == "error"
    assert events[-1]["exception_type"] == "ValueError"
    assert "Traceback" in events[-1]["traceback"]
