"""Tier 1 integration tests (PLAN 12, Phase 3): real worker, ASE EMT.

Written against webapp/docs/PLAN.md only. Every job goes through the real
``create_app`` with its default ``worker_cmd`` (``python -m
mliprun.webapp.worker``), and every result is compared with a direct call of
the package's Python API on the same input file.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import ase.io  # noqa: E402
from ase.build import add_adsorbate, bulk, fcc100  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402
from ase.constraints import FixAtoms  # noqa: E402

H = {"X-Mliprun": "1"}
METALS = ["Cu", "Pt", "Au", "Ni"]
METAL_IDS = ["cu", "pt", "au", "ni"]
JOB_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z]+-[0-9a-f]{6}$")
UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{12}$")
ISO_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
TERMINAL = ("done", "failed", "cancelled")

# Every event of every job finished in this module is appended here and
# validated by ``test_event_schema_all_received``.
COLLECTED_EVENTS: list = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upload_file(client, path: Path, fmt=None, name=None):
    data = {"format": fmt} if fmt else {}
    r = client.post("/api/uploads", headers=H, data=data,
                    files=[("files", (name or path.name, path.read_bytes()))])
    assert r.status_code == 200, r.text
    info = r.json()["uploads"][0]
    assert info["ok"], info
    return info


def write_and_upload(client, tmp: Path, atoms, filename, fmt):
    """Write atoms to a file, upload it; return (UploadInfo, atoms read back)."""
    path = tmp / filename
    ase.io.write(str(path), atoms, format=fmt)
    info = upload_file(client, path, fmt=fmt)
    back = ase.io.read(str(path), index=":", format=fmt)
    return info, back


def submit_json(client, body):
    return client.post("/api/jobs", headers=H, json=body)


def submit_single(client, upload_id, calc_type, params=None, structure_every=None):
    body = {"calc_type": calc_type, "calculator": {"name": "emt", "device": "cpu"},
            "params": params or {}, "structure_ids": [upload_id]}
    if structure_every is not None:
        body["structure_every"] = structure_every
    r = submit_json(client, body)
    assert r.status_code == 201, r.text
    return r.json()["jobs"][0]["job_id"]


def wait_terminal(client, job_id, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        d = client.get(f"/api/jobs/{job_id}").json()
        if d["status"] in TERMINAL:
            return d
        time.sleep(0.1)
    raise AssertionError(f"{job_id} not terminal after {timeout} s: {d['status']}")


def all_events(client, job_id):
    r = client.get(f"/api/jobs/{job_id}/events", params={"since": 0, "limit": 5000})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["terminal"] is True
    COLLECTED_EVENTS.append((job_id, body["events"]))
    return body["events"]


def run_job(client, upload_id, calc_type, params=None, structure_every=None, timeout=300):
    jid = submit_single(client, upload_id, calc_type, params, structure_every)
    detail = wait_terminal(client, jid, timeout)
    events = all_events(client, jid)
    return detail, events


def by_type(events, t):
    return [e for e in events if e["type"] == t]


def csv_rows(path: Path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def assert_common_outputs(detail):
    run_dir = Path(detail["run_dir"])
    for name in ("job.json", "state.json", "events.jsonl", "summary.json",
                 "reproduce.sh", "reproduce.json", "worker.log", "mliprun_run.json"):
        assert (run_dir / name).is_file(), f"missing {name}"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "done"
    assert summary["job_id"] == detail["job_id"]
    rep = json.loads((run_dir / "reproduce.json").read_text())
    assert rep["status"] == "no"          # emt has no CLI equivalent
    assert detail["reproduce"]["status"] == "no"
    sh = (run_dir / "reproduce.sh").read_text()
    assert sh.startswith("#!/usr/bin/env bash")
    assert "REPLACE_WITH_AN_MLIP" in sh
    assert (run_dir / "reproduce.sh").stat().st_mode & 0o111
    # file listing endpoint mirrors the directory
    files = client_files(detail)
    assert "summary.json" in files and "reproduce.sh" in files
    return run_dir


_CLIENT = {}


def client_files(detail):
    c = _CLIENT["c"]
    r = c.get(f"/api/jobs/{detail['job_id']}/files")
    assert r.status_code == 200
    paths = [f["path"] for f in r.json()["files"]]
    assert paths == sorted(paths)
    return paths


# ---------------------------------------------------------------------------
# Event schema (PLAN 6)
# ---------------------------------------------------------------------------

def _num(x, nullable=False):
    if x is None:
        return nullable
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def check_frame(fr, natoms=None):
    assert set(fr) >= {"positions", "cell"}
    assert len(fr["cell"]) == 3 and all(len(v) == 3 for v in fr["cell"])
    if natoms is not None:
        assert len(fr["positions"]) == natoms
    for p in fr["positions"]:
        assert len(p) == 3
        for v in p:
            assert _num(v) and round(v, 4) == v       # 4-decimal rounding


RESULT_KEYS = {
    "singlepoint": {"energy_eV", "fmax_free_eV_per_A", "fmax_all_eV_per_A", "stress_GPa",
                    "stress_unavailable_reason", "forces", "final_frame"},
    "optimize": {"converged", "final_energy_eV", "final_fmax_eV_per_A", "steps", "final_frame"},
    "freq": {"modes", "n_modes", "n_imaginary", "n_imaginary_raw", "imaginary_floor_cm1",
             "zpe_eV", "fmax_at_input_free_eV_per_A", "fmax_warning", "mode_files",
             "final_frame"},
    "md": {"steps", "ensemble", "final_frame"},
    "neb": {"converged", "steps", "energies_eV", "relative_energies_eV", "forward_barrier_eV",
            "reverse_barrier_eV", "reaction_energy_eV", "ts_image_index", "ts_at_endpoint",
            "final_fmax_eV_per_A", "band"},
}
RESULT_KEYS["autoneb"] = RESULT_KEYS["neb"]


def check_event(ev, job_id=None, natoms=None):
    """Assert one event matches PLAN 6.1/6.3/6.4."""
    assert isinstance(ev["job_id"], str) and JOB_ID_RE.match(ev["job_id"])
    if job_id:
        assert ev["job_id"] == job_id
    assert _int(ev["seq"]) and ev["seq"] >= 0
    assert _num(ev["t"])
    t = ev["type"]
    if t == "job_status":
        assert ev["status"] in ("pending", "running", "done", "failed", "cancelled")
        assert ev["reason"] is None or isinstance(ev["reason"], str)
        assert ev["returncode"] is None or _int(ev["returncode"])
    elif t == "phase":
        assert ev["phase"] in ("setup", "load_model", "endpoints", "interpolation", "run",
                               "postprocess")
        assert isinstance(ev["message"], str)
    elif t == "log":
        assert ev["level"] in ("info", "warning", "error")
        assert isinstance(ev["message"], str) and isinstance(ev["logger"], str)
    elif t == "opt_step":
        assert _int(ev["step"]) and _num(ev["energy"]) and _num(ev["fmax"])
        if "frame" in ev:
            check_frame(ev["frame"], natoms)
    elif t == "md_step":
        assert _int(ev["step"])
        for k in ("time_fs", "epot", "ekin", "etot", "temperature"):
            assert _num(ev[k]), k
        for k in ("pressure_GPa", "volume_A3"):
            if k in ev:
                assert _num(ev[k], nullable=True)
        if "frame" in ev:
            check_frame(ev["frame"], natoms)
    elif t == "neb_iter":
        assert _int(ev["iteration"]) and _int(ev["highest_image"]) and _int(ev["n_images"])
        assert len(ev["energies"]) == ev["n_images"] == len(ev["rel_energies"])
        assert all(_num(e, nullable=True) for e in ev["energies"])
        assert _num(ev["fmax"]) and _num(ev["barrier"])
        if "band" in ev:
            assert len(ev["band"]) == ev["n_images"]
            for fr in ev["band"]:
                check_frame(fr, natoms)
        if "autoneb" in ev:
            assert _int(ev["autoneb"]["sub_step"])
    elif t == "freq_progress":
        assert _int(ev["done"]) and _int(ev["total"]) and 0 <= ev["done"] <= ev["total"]
    elif t == "result":
        ct = ev["calc_type"]
        assert ct in RESULT_KEYS
        assert ev["record_status"] in ("converged", "not_converged", "completed")
        assert isinstance(ev["record_results"], dict)
        assert _num(ev["walltime_s"], nullable=True)
        missing = RESULT_KEYS[ct] - set(ev)
        assert not missing, f"result {ct} missing {missing}"
    elif t == "error":
        for k in ("message", "exception_type", "traceback"):
            assert isinstance(ev[k], str)
    else:
        raise AssertionError(f"unknown event type {t!r}")


def check_event_stream(events, job_id, final_status="done", natoms=None):
    """Envelope-level invariants of a finished job's events.jsonl."""
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert events[0]["type"] == "job_status" and events[0]["status"] == "pending"
    assert events[-1]["type"] == "job_status" and events[-1]["status"] == final_status
    for e in events:
        check_event(e, job_id, natoms)
    if final_status == "done":
        assert len(by_type(events, "result")) == 1
        # result is the last worker event, just before the final job_status
        assert events[-2]["type"] == "result"
    statuses = [e["status"] for e in by_type(events, "job_status")]
    assert statuses[:2] == ["pending", "running"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient
    from mliprun.webapp.server import create_app

    runs = tmp_path_factory.mktemp("runs_tier1")
    app = create_app(runs, cancel_grace_s=3.0, poll_interval_s=0.05)
    with TestClient(app) as c:
        _CLIENT["c"] = c
        yield c
    _CLIENT.clear()


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


def rattled_bulk(sym):
    a = bulk(sym, "fcc", cubic=True).repeat(2)
    a.rattle(0.05, seed=1)
    return a


# ---------------------------------------------------------------------------
# Unit: parameter validation (schema.validate_submission, PLAN 3.3)
# ---------------------------------------------------------------------------

def _meta(atoms, n_frames=1):
    return {"natoms": len(atoms), "symbols": atoms.get_chemical_symbols(),
            "n_frames": n_frames, "pbc": [bool(x) for x in atoms.pbc]}


CU = bulk("Cu", "fcc", cubic=True)
EMT_CPU = {"name": "emt", "device": "cpu"}


def _fields(errs):
    return {e["field"] for e in errs}


def test_schema_has_every_calc_type_and_fieldspec_keys():
    from mliprun.webapp.schema import get_calc_types
    ct = get_calc_types()
    assert ct["schema_version"] == 1
    names = [c["name"] for c in ct["calc_types"]]
    assert sorted(names) == sorted(["singlepoint", "optimize", "freq", "md", "neb", "autoneb"])
    keys = {"name", "label", "type", "default", "unit", "help", "min", "max", "min_exclusive",
            "max_exclusive", "choices", "nullable", "advanced", "visible_if", "input_modes",
            "cli_flag"}
    for c in ct["calc_types"]:
        assert c["inputs"] in ("single", "pair", "pair_or_band")
        assert c["fields"]
        for f in c["fields"]:
            assert set(f) >= keys, (c["name"], f["name"], keys - set(f))


def test_validate_defaults_filled_and_sources():
    from mliprun.webapp.schema import validate_submission
    out = validate_submission("optimize", EMT_CPU, {"fmax": 0.02}, {"structure": _meta(CU)})
    assert out["params"] == {"optimizer": "bfgs", "fmax": 0.02, "max_steps": 200,
                             "relax_cell": False, "plot": False}
    assert out["param_sources"]["fmax"] == "user"
    assert out["param_sources"]["optimizer"] == "default"
    assert set(out["calculator"]) == {"name", "device", "uma_task", "mace_head",
                                      "sevennet_task"}


def test_validate_md_hidden_fields_absent_in_output():
    from mliprun.webapp.schema import validate_submission
    out = validate_submission("md", EMT_CPU, {"ensemble": "nve"}, {"structure": _meta(CU)})
    p = out["params"]
    for hidden in ("temperature", "thermostat", "barostat", "friction", "ttime", "taut",
                   "pressure", "taup", "compressibility", "pfactor", "barostat_mask"):
        assert hidden not in p
    out = validate_submission("md", EMT_CPU, {"ensemble": "npt", "barostat": "berendsen"},
                              {"structure": _meta(CU)})
    assert "compressibility" in out["params"] and "pfactor" not in out["params"]
    assert "taut" in out["params"] and "ttime" not in out["params"]


@pytest.mark.parametrize("calc_type,params,field", [
    ("optimize", {"fmax": 0}, "params.fmax"),              # exclusive lower bound
    ("optimize", {"fmax": 10.5}, "params.fmax"),
    ("optimize", {"max_steps": -1}, "params.max_steps"),
    ("optimize", {"max_steps": 100001}, "params.max_steps"),
    ("freq", {"delta": 0.2}, "params.delta"),
    ("md", {"steps": 0}, "params.steps"),
    ("md", {"timestep": 0.0}, "params.timestep"),
    ("neb", {"n_images": 31}, "params.n_images"),
    ("autoneb", {"n_max": 2}, "params.n_max"),
    ("autoneb", {"space_energy_ratio": 1.5}, "params.space_energy_ratio"),
], ids=["fmax0", "fmax_hi", "steps_neg", "steps_hi", "delta_hi", "md_steps0", "dt0",
        "nimg_hi", "nmax_lo", "ser_hi"])
def test_validate_out_of_range(calc_type, params, field):
    from mliprun.webapp.schema import ValidationFailed, validate_submission
    inputs = ({"structure": _meta(CU)} if calc_type not in ("neb", "autoneb")
              else {"initial": _meta(CU), "final": _meta(CU)})
    with pytest.raises(ValidationFailed) as ei:
        validate_submission(calc_type, EMT_CPU, params, inputs)
    assert field in _fields(ei.value.errors)


@pytest.mark.parametrize("params,field", [
    ({"fmax": "0.05"}, "params.fmax"),
    ({"fmax": True}, "params.fmax"),
    ({"max_steps": 10.5}, "params.max_steps"),
    ({"max_steps": True}, "params.max_steps"),
    ({"relax_cell": 1}, "params.relax_cell"),
    ({"relax_cell": "true"}, "params.relax_cell"),
    ({"optimizer": "BFGS"}, "params.optimizer"),           # case-sensitive choice
    ({"optimizer": None}, "params.optimizer"),             # not nullable
    ({"bogus": 1}, "params.bogus"),                        # unknown
], ids=["str_float", "bool_float", "frac_int", "bool_int", "int_bool", "str_bool",
        "choice_case", "null", "unknown"])
def test_validate_wrong_type(params, field):
    from mliprun.webapp.schema import ValidationFailed, validate_submission
    with pytest.raises(ValidationFailed) as ei:
        validate_submission("optimize", EMT_CPU, params, {"structure": _meta(CU)})
    assert field in _fields(ei.value.errors)


def test_validate_int_with_zero_fraction_accepted():
    from mliprun.webapp.schema import validate_submission
    out = validate_submission("optimize", EMT_CPU, {"max_steps": 10.0},
                              {"structure": _meta(CU)})
    assert out["params"]["max_steps"] == 10


@pytest.mark.parametrize("calc_type,params,inputs_kind,field", [
    ("md", {"ensemble": "nve", "temperature": 300}, "single", "params.temperature"),
    ("md", {"ensemble": "nvt", "thermostat": "langevin", "ttime": 25.0}, "single",
     "params.ttime"),
    ("md", {"ensemble": "npt", "barostat": "npt", "compressibility": 0.5}, "single",
     "params.compressibility"),
    ("md", {"ensemble": "nvt", "pressure": 0.0}, "single", "params.pressure"),
    ("neb", {"n_images": 5}, "band", "params.n_images"),          # input_modes pair only
    ("neb", {"optimize_endpoints": False, "endpoint_fmax": 0.01}, "pair",
     "params.endpoint_fmax"),
    ("neb", {"interpolation": "linear", "interp_steps": 10}, "pair", "params.interp_steps"),
    ("neb", {"dyneb": False, "scale_fmax": 0.5}, "pair", "params.scale_fmax"),
], ids=["nve_temp", "langevin_ttime", "nptnpt_compr", "nvt_press", "band_nimg",
        "noendp_fmax", "linear_isteps", "nodyneb_scale"])
def test_validate_hidden_field_rejected(calc_type, params, inputs_kind, field):
    from mliprun.webapp.schema import ValidationFailed, validate_submission
    band = bulk("Cu", "fcc", cubic=True)
    inputs = {"single": {"structure": _meta(CU)},
              "pair": {"initial": _meta(CU), "final": _meta(CU)},
              "band": {"band": dict(_meta(band, 5), frames_same_symbols=True)}}[inputs_kind]
    with pytest.raises(ValidationFailed) as ei:
        validate_submission(calc_type, EMT_CPU, params, inputs)
    errs = [e for e in ei.value.errors if e["field"] == field]
    assert errs, ei.value.errors
    assert "does not apply" in errs[0]["message"]


def test_validate_collects_all_errors():
    from mliprun.webapp.schema import ValidationFailed, validate_submission
    with pytest.raises(ValidationFailed) as ei:
        validate_submission("optimize", {"name": "emt", "device": "quantum"},
                            {"fmax": -1, "max_steps": "x", "optimizer": "nope"},
                            {"structure": _meta(CU)})
    f = _fields(ei.value.errors)
    assert {"params.fmax", "params.max_steps", "params.optimizer", "calculator.device"} <= f


def test_validate_indices():
    from mliprun.webapp.schema import ValidationFailed, validate_submission
    meta = {"structure": _meta(CU)}           # 4 atoms
    out = validate_submission("freq", EMT_CPU, {"indices": " 0-2, 3 "}, meta)
    assert out["params"]["indices"] == "0-2, 3"
    for bad in ("4", "0,9", "", "a,b"):
        with pytest.raises(ValidationFailed) as ei:
            validate_submission("freq", EMT_CPU, {"indices": bad}, meta)
        assert "params.indices" in _fields(ei.value.errors), bad


def test_validate_calc_type_roles_and_emt_elements():
    from mliprun.webapp.schema import ValidationFailed, validate_submission
    with pytest.raises(ValidationFailed) as ei:
        validate_submission("nope", EMT_CPU, {}, {"structure": _meta(CU)})
    assert "calc_type" in _fields(ei.value.errors)
    with pytest.raises(ValidationFailed):
        validate_submission("optimize", EMT_CPU, {}, {"initial": _meta(CU),
                                                       "final": _meta(CU)})
    fe = bulk("Fe", "bcc", cubic=True)
    with pytest.raises(ValidationFailed) as ei:
        validate_submission("singlepoint", EMT_CPU, {}, {"structure": _meta(fe)})
    errs = [e for e in ei.value.errors if e["field"] == "calculator.name"]
    assert errs and "Fe" in errs[0]["message"]


def test_validate_neb_mismatch_and_band_rules():
    from mliprun.webapp.schema import ValidationFailed, validate_submission
    a = bulk("Cu", "fcc", cubic=True)
    b = a.repeat((2, 1, 1))
    with pytest.raises(ValidationFailed) as ei:
        validate_submission("neb", EMT_CPU, {}, {"initial": _meta(a), "final": _meta(b)})
    assert "inputs.final" in _fields(ei.value.errors)
    c = a.copy()
    c[0].symbol = "Ni"
    with pytest.raises(ValidationFailed) as ei:
        validate_submission("autoneb", EMT_CPU, {}, {"initial": _meta(a), "final": _meta(c)})
    assert "inputs.final" in _fields(ei.value.errors)
    for n_frames, same in ((2, True), (53, True), (5, False)):
        with pytest.raises(ValidationFailed):
            validate_submission("neb", EMT_CPU, {}, {
                "band": dict(_meta(a, n_frames), frames_same_symbols=same)})
    validate_submission("neb", EMT_CPU, {}, {"band": dict(_meta(a, 3),
                                                          frames_same_symbols=True)})


# ---------------------------------------------------------------------------
# Job state machine via the API (PLAN 10.1)
# ---------------------------------------------------------------------------

@pytest.fixture
def paused_client(tmp_path):
    """App whose queue runner is not started: jobs stay pending."""
    from fastapi.testclient import TestClient
    from mliprun.webapp.server import create_app
    app = create_app(tmp_path / "runs", start_runner=False, poll_interval_s=0.05)
    with TestClient(app) as c:
        yield c


def test_state_pending_to_cancelled_and_no_transition_out(paused_client, tmp):
    c = paused_client
    info, _ = write_and_upload(c, tmp, CU, "POSCAR", "vasp")
    jid = submit_single(c, info["id"], "singlepoint")
    d = c.get(f"/api/jobs/{jid}").json()
    assert d["status"] == "pending" and d["queue_position"] == 1
    r = c.post(f"/api/jobs/{jid}/cancel", headers=H)
    assert r.status_code == 200
    assert r.json()["job"]["status"] == "cancelled"
    d = c.get(f"/api/jobs/{jid}").json()
    assert d["status"] == "cancelled" and d["queue_position"] is None
    # illegal: cancelled -> cancelled
    r = c.post(f"/api/jobs/{jid}/cancel", headers=H)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "conflict"
    run_dir = Path(d["run_dir"])
    state = json.loads((run_dir / "state.json").read_text())
    assert state["status"] == "cancelled" and state["pid"] is None
    assert not (run_dir / "summary.json").exists()
    evs = c.get(f"/api/jobs/{jid}/events").json()
    assert [e["status"] for e in evs["events"]] == ["pending", "cancelled"]
    assert evs["terminal"] is True


def test_state_queue_positions(paused_client, tmp):
    c = paused_client
    info, _ = write_and_upload(c, tmp, CU, "POSCAR", "vasp")
    r = submit_json(c, {"calc_type": "singlepoint", "calculator": EMT_CPU, "params": {},
                        "structure_ids": [info["id"]] * 3})
    assert r.status_code == 201
    jobs = r.json()["jobs"]
    assert [j["queue_position"] for j in jobs] == [1, 2, 3]
    c.post(f"/api/jobs/{jobs[0]['job_id']}/cancel", headers=H)
    pos = {j["job_id"]: j["queue_position"] for j in c.get("/api/jobs").json()["jobs"]}
    assert pos[jobs[1]["job_id"]] == 1 and pos[jobs[2]["job_id"]] == 2


def test_state_done_is_terminal(client, tmp):
    info, _ = write_and_upload(client, tmp, CU, "POSCAR", "vasp")
    detail, events = run_job(client, info["id"], "singlepoint")
    assert detail["status"] == "done"
    r = client.post(f"/api/jobs/{detail['job_id']}/cancel", headers=H)
    assert r.status_code == 409
    assert client.get(f"/api/jobs/{detail['job_id']}").json()["status"] == "done"
    assert [e["status"] for e in by_type(events, "job_status")] == ["pending", "running", "done"]


# ---------------------------------------------------------------------------
# API: uploads and submission
# ---------------------------------------------------------------------------

def test_upload_poscar_without_extension(client, tmp):
    info, _ = write_and_upload(client, tmp, rattled_bulk("Cu"), "POSCAR", None)
    assert UPLOAD_ID_RE.match(info["id"])
    assert info["format"] == "vasp" and info["natoms"] == 32 and info["formula"] == "Cu32"
    assert info["n_frames"] == 1 and info["frames_xyz"] is None
    assert ISO_RE.match(info["created_at"])
    lines = info["xyz"].splitlines()
    assert lines[0].strip() == "32" and "Lattice=" in lines[1] and 'pbc="T T T"' in lines[1]
    assert len(lines) == 34
    full = client.get(f"/api/uploads/{info['id']}").json()
    assert full["id"] == info["id"] and "xyz" in full
    lst = client.get("/api/uploads").json()["uploads"]
    assert lst[0]["id"] == info["id"] and "xyz" not in lst[0]


def test_upload_multiframe_extxyz(client, tmp):
    frames = [rattled_bulk("Pt") for _ in range(4)]
    for i, f in enumerate(frames):
        f.positions += 0.01 * i
    info, _ = write_and_upload(client, tmp, frames, "band.extxyz", "extxyz")
    assert info["n_frames"] == 4
    assert info["frames_xyz"] and info["frames_xyz"].count("Lattice=") == 4


def test_upload_multiple_files_and_garbage(client, tmp):
    p1 = tmp / "a.xyz"
    ase.io.write(str(p1), bulk("Au", "fcc", cubic=True), format="extxyz")
    p2 = tmp / "b.cif"
    ase.io.write(str(p2), bulk("Ni", "fcc", cubic=True), format="cif")
    files = [("files", ("a.xyz", p1.read_bytes())),
             ("files", ("junk.xyz", b"this is not\na structure file\n\x00\x01")),
             ("files", ("b.cif", p2.read_bytes())),
             ("files", ("../../evil name!.txt", b"garbage"))]
    r = client.post("/api/uploads", headers=H, files=files)
    assert r.status_code == 200, r.text
    ups = r.json()["uploads"]
    assert [u["ok"] for u in ups] == [True, False, True, False]
    assert ups[0]["formula"] == "Au4" and ups[2]["formula"] == "Ni4"
    for bad in (ups[1], ups[3]):
        assert bad["id"] is None and isinstance(bad["error"], str) and bad["error"]


def test_upload_no_files_and_header(client):
    r = client.post("/api/uploads", headers=H, data={"format": ""})
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"
    r = client.post("/api/uploads", files=[("files", ("a", b"x"))])
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    r = client.get("/api/uploads/000000000000")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_upload_forced_format(client, tmp):
    p = tmp / "weird.dat"
    ase.io.write(str(p), bulk("Cu", "fcc", cubic=True), format="vasp")
    info = upload_file(client, p, fmt="vasp")
    assert info["format"] == "vasp" and info["natoms"] == 4


def test_submit_errors(client, tmp):
    info, _ = write_and_upload(client, tmp, CU, "POSCAR", "vasp")
    base = {"calc_type": "optimize", "calculator": EMT_CPU, "params": {},
            "structure_ids": [info["id"]]}
    r = submit_json(client, dict(base, extra_key=1))
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert r.json()["error"]["details"]
    r = submit_json(client, dict(base, structure_ids=["abcdefabcdef"]))
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = submit_json(client, dict(base, initial_id=info["id"]))
    assert r.status_code == 422
    r = client.post("/api/jobs", json=base)
    assert r.status_code == 403
    r = client.post("/api/jobs", headers=dict(H, **{"Content-Type": "application/json"}),
                    content=b"{not json")
    assert r.status_code in (400, 422)
    assert "error" in r.json()
    r = submit_json(client, dict(base, params={"fmax": -1}))
    assert r.status_code == 422
    d = r.json()["error"]["details"]
    assert any(x["field"].endswith("params.fmax") for x in d)


def test_submit_all_or_nothing(client, tmp):
    good, _ = write_and_upload(client, tmp, CU, "POSCAR", "vasp")
    small, _ = write_and_upload(client, tmp, bulk("Cu", "fcc"), "one.xyz", "extxyz")
    before = len(client.get("/api/jobs").json()["jobs"])
    r = submit_json(client, {"calc_type": "freq", "calculator": EMT_CPU,
                             "params": {"indices": "0-3"},
                             "structure_ids": [good["id"], small["id"]]})
    assert r.status_code == 422, r.text
    fields = [d["field"] for d in r.json()["error"]["details"]]
    assert "structure_ids[1].params.indices" in fields
    assert len(client.get("/api/jobs").json()["jobs"]) == before


def test_jobs_listing_filters(client):
    r = client.get("/api/jobs", params={"status": "bogus"})
    assert r.status_code == 422
    r = client.get("/api/jobs", params={"status": "done,failed", "limit": 2})
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    assert len(jobs) <= 2 and all(j["status"] in ("done", "failed") for j in jobs)
    assert client.get("/api/jobs/20200101-000000-optimize-000000").status_code == 404
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "queue_length" in h and "running_job" in h


# ---------------------------------------------------------------------------
# End to end, per calc type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sym", METALS, ids=METAL_IDS)
def test_e2e_singlepoint(client, tmp, sym):
    info, back = write_and_upload(client, tmp, rattled_bulk(sym), "POSCAR", "vasp")
    detail, events = run_job(client, info["id"], "singlepoint")
    assert detail["status"] == "done", detail["error"]
    check_event_stream(events, detail["job_id"], natoms=32)
    res = detail["result"]
    ref = back[-1].copy()
    ref.calc = EMT()
    assert abs(res["energy_eV"] - ref.get_potential_energy()) < 1e-8
    F = ref.get_forces()
    assert len(res["forces"]) == len(ref)
    for row in res["forces"]:
        i = row["index"]
        assert row["symbol"] == sym
        assert np.allclose([row["fx"], row["fy"], row["fz"]], F[i], atol=1e-8, rtol=0)
    assert detail["result_brief"]["energy_eV"] == res["energy_eV"]
    run_dir = assert_common_outputs(detail)
    assert (run_dir / "singlepoint_forces.csv").is_file()


@pytest.mark.parametrize("sym", METALS, ids=METAL_IDS)
def test_e2e_optimize(client, tmp, sym):
    from mliprun.core.optimize import run_optimization
    info, back = write_and_upload(client, tmp, rattled_bulk(sym), "POSCAR", "vasp")
    detail, events = run_job(client, info["id"], "optimize", {"fmax": 0.05},
                             structure_every=5)
    assert detail["status"] == "done", detail["error"]
    check_event_stream(events, detail["job_id"], natoms=32)
    res = detail["result"]
    assert res["converged"] is True and res["final_fmax_eV_per_A"] <= 0.05
    ref = back[-1].copy()
    ref.calc = EMT()
    refdir = tmp / "ref"
    run_optimization(ref, optimizer="bfgs", fmax=0.05, max_steps=200, output_dir=refdir,
                     model_name="emt", verbose=False)
    assert abs(res["final_energy_eV"] - ref.get_potential_energy()) < 1e-6
    steps = by_type(events, "opt_step")
    run_dir = assert_common_outputs(detail)
    rows = csv_rows(run_dir / "opt_convergence.csv")
    assert len(steps) == len(rows)
    has_frame = [i for i, e in enumerate(steps) if "frame" in e]
    assert has_frame == list(range(0, len(steps), 5))
    for f in ("opt.traj", "opt.log", "opt_convergence.csv"):
        assert (run_dir / f).is_file()
    assert detail["progress"]["step"] == steps[-1]["step"]
    assert detail["progress"]["total"] == 200


@pytest.mark.parametrize("sym", METALS, ids=METAL_IDS)
def test_e2e_freq(client, tmp, sym):
    from mliprun.core.vibrations import run_frequencies
    info, back = write_and_upload(client, tmp, bulk(sym, "fcc", cubic=True), "POSCAR", "vasp")
    detail, events = run_job(client, info["id"], "freq")
    assert detail["status"] == "done", detail["error"]
    check_event_stream(events, detail["job_id"], natoms=4)
    res = detail["result"]
    N = 4
    assert res["n_modes"] == 3 * N == len(res["modes"])
    prog = by_type(events, "freq_progress")
    assert prog[-1]["done"] == prog[-1]["total"] == 1 + 6 * N
    ref = back[-1].copy()
    ref.calc = EMT()
    refdir = tmp / "ref"
    run_frequencies(ref, output_dir=refdir, prefix="freq", model_name="emt")
    rows = csv_rows(refdir / "freq_frequencies.csv")
    got = [m["frequency_cm1"] for m in res["modes"]]
    want = [float(r["frequency_cm-1"]) for r in rows]
    assert np.allclose(got, want, atol=1e-6, rtol=0)
    run_dir = assert_common_outputs(detail)
    assert (run_dir / "freq_frequencies.csv").is_file()
    for mf in res["mode_files"]:
        assert (run_dir / mf).is_file()


@pytest.mark.parametrize("sym", METALS, ids=METAL_IDS)
def test_e2e_md_nve(client, tmp, sym):
    from mliprun.core.md import run_md
    info, back = write_and_upload(client, tmp, rattled_bulk(sym), "POSCAR", "vasp")
    params = {"ensemble": "nve", "steps": 50, "timestep": 1.0, "log_interval": 5}
    detail, events = run_job(client, info["id"], "md", params)
    assert detail["status"] == "done", detail["error"]
    check_event_stream(events, detail["job_id"], natoms=32)
    run_dir = assert_common_outputs(detail)
    rows = csv_rows(run_dir / "md_energy.csv")
    md = by_type(events, "md_step")
    assert len(md) == len(rows) > 0
    etot = [e["etot"] for e in md]
    assert abs(etot[-1] - etot[0]) < 1e-2
    # NVE from rest is deterministic: compare with a direct run
    ref = back[-1].copy()
    ref.calc = EMT()
    refdir = tmp / "ref"
    run_md(ref, ensemble="nve", steps=50, timestep=1.0, log_interval=5, output_dir=refdir,
           model_name="emt")
    ref_rows = csv_rows(refdir / "md_energy.csv")
    assert abs(float(ref_rows[-1]["total_energy(eV)"]) - etot[-1]) < 1e-6
    assert detail["result"]["ensemble"] == "nve"


def test_e2e_md_nvt(client, tmp):
    info, _ = write_and_upload(client, tmp, rattled_bulk("Cu"), "POSCAR", "vasp")
    detail, events = run_job(client, info["id"], "md",
                             {"ensemble": "nvt", "steps": 50, "log_interval": 5})
    assert detail["status"] == "done", detail["error"]
    check_event_stream(events, detail["job_id"], natoms=32)
    assert math.isfinite(detail["result"]["mean_temperature_K"])
    assert detail["result_brief"]["mean_temperature_K"] == detail["result"]["mean_temperature_K"]


def test_e2e_md_npt(client, tmp):
    info, _ = write_and_upload(client, tmp, rattled_bulk("Cu"), "POSCAR", "vasp")
    detail, events = run_job(client, info["id"], "md",
                             {"ensemble": "npt", "barostat": "berendsen", "steps": 20,
                              "log_interval": 5})
    assert detail["status"] == "done", detail["error"]
    check_event_stream(events, detail["job_id"], natoms=32)
    md = by_type(events, "md_step")
    assert all("volume_A3" in e and "pressure_GPa" in e for e in md)


def neb_endpoints(sym):
    slab = fcc100(sym, size=(3, 3, 3), vacuum=6.0)
    add_adsorbate(slab, sym, 1.7, "hollow")
    slab.set_constraint(FixAtoms(indices=[a.index for a in slab if a.tag == 3]))
    initial = slab
    final = slab.copy()
    a = bulk(sym, "fcc").cell.cellpar()[0]   # nearest-neighbour distance
    final.positions[-1, 0] += a
    return initial, final


def direct_neb_barrier(initial, final, outdir, n_images=3, fmax=0.1):
    """Mirror PLAN 8.3 neb pair mode with EMT and the schema defaults."""
    from ase.optimize import FIRE
    from mliprun.core.neb import CustomNEB
    neb = CustomNEB(initial, final, num_images=n_images, interp_fmax=0.1, interp_steps=100,
                    fmax=fmax, mlip="emt", output_dir=outdir, relax_atoms=None,
                    logfile="neb.log", device="cpu")
    neb.setup_calculator = lambda *a, **k: EMT()
    neb.optimize_endpoints(endpoint_fmax=0.01, optimizer="bfgs", max_steps=200)
    neb.images = neb.setup_neb()
    neb.interpolate_idpp()
    neb.run_neb(optimizer=FIRE, climb=True, max_steps=600, k=0.1)
    E = [im.get_potential_energy() for im in neb.images]
    return max(E) - E[0]


@pytest.mark.parametrize("sym", METALS, ids=METAL_IDS)
def test_e2e_neb_pair_then_band(client, tmp, sym):
    initial, final = neb_endpoints(sym)
    pi, pf = tmp / "ini.vasp", tmp / "fin.vasp"
    ase.io.write(str(pi), initial, format="vasp", direct=True)
    ase.io.write(str(pf), final, format="vasp", direct=True)
    ui = upload_file(client, pi, fmt="vasp", name="POSCAR_initial")
    uf = upload_file(client, pf, fmt="vasp", name="POSCAR_final")
    r = submit_json(client, {"calc_type": "neb", "calculator": EMT_CPU,
                             "params": {"n_images": 3, "interpolation": "idpp", "fmax": 0.1},
                             "initial_id": ui["id"], "final_id": uf["id"]})
    assert r.status_code == 201, r.text
    jid = r.json()["jobs"][0]["job_id"]
    detail = wait_terminal(client, jid, 600)
    assert detail["status"] == "done", detail["error"]
    events = all_events(client, jid)
    natoms = len(initial)
    check_event_stream(events, jid, natoms=natoms)
    res = detail["result"]
    assert res["forward_barrier_eV"] > 0
    assert res["ts_at_endpoint"] is False
    assert len(res["band"]) == 5 and len(res["energies_eV"]) == 5
    phases = [e["phase"] for e in by_type(events, "phase")]
    assert "endpoints" in phases and "interpolation" in phases
    iters = by_type(events, "neb_iter")
    assert iters and all(e["n_images"] == 5 for e in iters)
    ref = direct_neb_barrier(ase.io.read(str(pi)), ase.io.read(str(pf)), tmp / "ref")
    assert abs(res["forward_barrier_eV"] - ref) < 1e-6
    run_dir = assert_common_outputs(detail)
    for f in ("A2B.traj", "neb.log", "inputs/initial.vasp", "inputs/final.vasp",
              "endpoint_optimization.txt", "neb_convergence.csv", "00/POSCAR", "04/POSCAR"):
        assert (run_dir / f).is_file(), f
    assert set(detail["inputs"]) == {"initial", "final"}
    assert detail["result_brief"]["forward_barrier_eV"] == res["forward_barrier_eV"]

    # band mode with the resulting A2B.traj
    band_frames = ase.io.read(str(run_dir / "A2B.traj"), index="-5:")
    bp = tmp / "band.traj"
    ase.io.write(str(bp), band_frames)
    ub = upload_file(client, bp)
    assert ub["n_frames"] == 5
    r = submit_json(client, {"calc_type": "neb", "calculator": EMT_CPU,
                             "params": {"fmax": 0.1}, "band_id": ub["id"]})
    assert r.status_code == 201, r.text
    jid2 = r.json()["jobs"][0]["job_id"]
    d2 = wait_terminal(client, jid2, 600)
    assert d2["status"] == "done", d2["error"]
    ev2 = all_events(client, jid2)
    check_event_stream(ev2, jid2, natoms=natoms)
    phases = [e["phase"] for e in by_type(ev2, "phase")]
    assert "endpoints" not in phases and "interpolation" not in phases
    assert d2["result"]["forward_barrier_eV"] > 0
    assert set(d2["inputs"]) == {"band"}
    assert_common_outputs(d2)


def test_e2e_autoneb_cu(client, tmp):
    initial, final = neb_endpoints("Cu")
    pi, pf = tmp / "ini.vasp", tmp / "fin.vasp"
    ase.io.write(str(pi), initial, format="vasp", direct=True)
    ase.io.write(str(pf), final, format="vasp", direct=True)
    ui = upload_file(client, pi, fmt="vasp")
    uf = upload_file(client, pf, fmt="vasp")
    r = submit_json(client, {"calc_type": "autoneb", "calculator": EMT_CPU,
                             "params": {"n_max": 5, "fmax": 0.1},
                             "initial_id": ui["id"], "final_id": uf["id"]})
    assert r.status_code == 201, r.text
    jid = r.json()["jobs"][0]["job_id"]
    detail = wait_terminal(client, jid, 900)
    assert detail["status"] == "done", detail["error"]
    events = all_events(client, jid)
    check_event_stream(events, jid, natoms=len(initial))
    res = detail["result"]
    assert res["forward_barrier_eV"] > 0
    assert res["steps"] is None and res["final_fmax_eV_per_A"] is None
    assert len(res["energies_eV"]) == 5
    iters = by_type(events, "neb_iter")
    assert iters and all("autoneb" in e for e in iters)
    assert [e["iteration"] for e in iters] == sorted(e["iteration"] for e in iters)
    run_dir = assert_common_outputs(detail)
    assert (run_dir / "autoneb_energy_profile.csv").is_file()
    assert sorted(p.name for p in run_dir.glob("autoneb???.traj"))[:1] == ["autoneb000.traj"]


# ---------------------------------------------------------------------------
# Live stream over the WebSocket
# ---------------------------------------------------------------------------

def test_live_stream_optimize(client, tmp):
    info, _ = write_and_upload(client, tmp, rattled_bulk("Ni"), "POSCAR", "vasp")
    jid = submit_single(client, info["id"], "optimize", {"fmax": 0.02})
    got = []
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "subscribe", "job_id": jid, "since": 0})
        deadline = time.monotonic() + 120
        acked = False
        while time.monotonic() < deadline:
            m = ws.receive_json()
            if m.get("type") == "ws_ack":
                acked = True
                assert m["replayed"] == len(got)
                continue
            if m.get("type") in ("ws_ping",):
                continue
            assert "seq" in m, m
            got.append(m)
            if m["type"] == "job_status" and m["status"] in TERMINAL:
                break
    assert acked
    assert [e["seq"] for e in got] == list(range(len(got)))   # no gap, no duplicate
    assert got[-1]["status"] == "done"
    steps = [e["step"] for e in got if e["type"] == "opt_step"]
    assert len(steps) > 3
    assert all(b > a for a, b in zip(steps, steps[1:]))
    last = [e for e in got if e["type"] == "opt_step"][-1]
    assert last["fmax"] < 0.02
    # the stream equals the file
    assert got == client.get(f"/api/jobs/{jid}/events").json()["events"]
    COLLECTED_EVENTS.append((jid, got))


def test_ws_star_subscription_is_light(client, tmp):
    info, _ = write_and_upload(client, tmp, CU, "POSCAR", "vasp")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "subscribe", "job_id": "*"})
        ack = ws.receive_json()
        assert ack["type"] == "ws_ack"
        jid = submit_single(client, info["id"], "optimize")
        seen = []
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            m = ws.receive_json()
            if m.get("type") == "ws_ping":
                continue
            if m.get("job_id") != jid:
                continue
            seen.append(m)
            if m["type"] == "job_status" and m["status"] in TERMINAL:
                break
    types = {m["type"] for m in seen}
    assert types <= {"job_status", "result", "error"}
    assert "result" in types


def test_ws_bad_subscribe(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "subscribe", "job_id": "20200101-000000-md-000000", "since": 0})
        m = ws.receive_json()
        assert m["type"] == "ws_error" and m["code"] == "not_found"
        ws.send_json({"action": "dance"})
        m = ws.receive_json()
        assert m["type"] == "ws_error" and m["code"] == "bad_request"


# ---------------------------------------------------------------------------
# Error event and schema conformance over everything received
# ---------------------------------------------------------------------------

def test_calculation_error_event(client, tmp):
    """freq on a structure whose atoms are all fixed: nothing to displace."""
    a = bulk("Cu", "fcc", cubic=True)
    a.set_constraint(FixAtoms(indices=range(len(a))))
    p = tmp / "POSCAR"
    ase.io.write(str(p), a, format="vasp")
    info = upload_file(client, p, fmt="vasp")
    assert info["has_constraints"] is True
    jid = submit_single(client, info["id"], "freq")
    d = wait_terminal(client, jid)
    events = all_events(client, jid)
    if d["status"] == "done":
        pytest.skip("core accepted an all-fixed structure; no error event to check")
    assert d["status"] == "failed"
    check_event_stream(events, jid, final_status="failed")
    errs = by_type(events, "error")
    assert errs
    assert d["error"]
    assert d["returncode"] == 1
    summary = json.loads((Path(d["run_dir"]) / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["error"]["message"]


def test_event_schema_all_received():
    """Runs last in this module: every event seen so far conforms to PLAN 6."""
    if not COLLECTED_EVENTS:
        pytest.skip("no events collected (run with the rest of the module)")
    seen = set()
    for jid, events in COLLECTED_EVENTS:
        for e in events:
            check_event(e, jid)
            seen.add(e["type"])
    assert {"job_status", "phase", "log", "opt_step", "md_step", "neb_iter",
            "freq_progress", "result"} <= seen
