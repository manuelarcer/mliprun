"""POST/GET /api/jobs: submission, validation passthrough, shapes, events endpoint."""
import json
import re

import pytest
from ase.build import bulk

JOB_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z]+-[0-9a-f]{6}$")


@pytest.fixture
def idle_client(make_app):
    """A client whose queue runner is off: submitted jobs stay pending."""
    from fastapi.testclient import TestClient

    with TestClient(make_app(start_runner=False)) as c:
        yield c


def test_submit_creates_pending_job_on_disk(idle_client, upload_atoms, submit, runs_dir):
    up = upload_atoms(idle_client, bulk("Cu", "fcc", a=3.6, cubic=True), "POSCAR", fmt="vasp")
    r = submit(idle_client, [up["id"]], params={"fmax": 0.02})
    assert r.status_code == 201, r.text
    (job,) = r.json()["jobs"]
    assert JOB_ID_RE.match(job["job_id"]) and "-optimize-" in job["job_id"]
    assert job["status"] == "pending" and job["queue_position"] == 1
    assert job["label"] == "POSCAR" and job["formula"] == "Cu4" and job["natoms"] == 4
    assert job["calculator"] == {"name": "emt", "device": "cpu"}
    assert job["progress"] is None and job["result_brief"] is None and job["error"] is None
    assert job["created_at"].endswith("Z") and job["started_at"] is None

    run_dir = runs_dir / job["job_id"]
    spec = json.loads((run_dir / "job.json").read_text())
    assert spec["schema_version"] == 1 and spec["calc_type"] == "optimize"
    assert spec["params"]["fmax"] == 0.02 and spec["params"]["max_steps"] == 200
    assert spec["param_sources"]["fmax"] == "user"
    assert spec["param_sources"]["max_steps"] == "default"
    assert set(spec["calculator"]) == {"name", "device", "uma_task", "mace_head",
                                       "sevennet_task"}
    assert spec["structure_every"] == 5
    assert spec["inputs"]["structure"] == {"upload_id": up["id"], "filename": "POSCAR",
                                           "path": "inputs/structure/POSCAR",
                                           "format": "vasp"}
    assert (run_dir / "inputs/structure/POSCAR").read_bytes() == \
        (runs_dir / "_uploads" / up["id"] / "POSCAR").read_bytes()
    state = json.loads((run_dir / "state.json").read_text())
    assert state["status"] == "pending" and state["pid"] is None
    lines = (run_dir / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    ev = json.loads(lines[0])
    assert ev["seq"] == 0 and ev["type"] == "job_status" and ev["status"] == "pending"


def test_job_detail_shape(idle_client, upload_atoms, submit, runs_dir):
    up = upload_atoms(idle_client, bulk("Cu", "fcc", a=3.6, cubic=True), "POSCAR", fmt="vasp")
    job_id = submit(idle_client, [up["id"]]).json()["jobs"][0]["job_id"]
    d = idle_client.get(f"/api/jobs/{job_id}").json()
    assert d["calculator"]["uma_task"] is None and d["calculator"]["name"] == "emt"
    assert d["params"]["optimizer"] == "bfgs"
    assert d["structure_every"] == 5
    assert d["run_dir"] == str((runs_dir / job_id).resolve())
    assert d["returncode"] is None and d["result"] is None and d["reproduce"] is None
    assert d["last_seq"] == 0
    inp = d["inputs"]["structure"]
    assert inp["upload_id"] == up["id"] and inp["natoms"] == 4
    assert inp["xyz"] == up["xyz"] and inp["frames_xyz"] is None
    assert idle_client.get("/api/jobs/20260101-000000-optimize-abcdef").status_code == 404
    assert idle_client.get("/api/jobs/nope").json()["error"]["code"] == "not_found"


def test_validation_errors_pass_through_as_422(idle_client, upload_atoms, submit):
    up = upload_atoms(idle_client, bulk("Cu"), "POSCAR", fmt="vasp")
    r = submit(idle_client, [up["id"]], params={"fmax": -1.0, "bogus": 3})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error"
    fields = {d["field"] for d in err["details"]}
    assert {"params.fmax", "params.bogus"} <= fields
    assert err["message"].startswith(f"{len(err['details'])} problem")
    assert idle_client.get("/api/jobs").json()["jobs"] == []


def test_multi_submit_is_all_or_nothing(idle_client, upload_atoms, submit, runs_dir):
    cu = upload_atoms(idle_client, bulk("Cu"), "POSCAR", fmt="vasp")
    fe = upload_atoms(idle_client, bulk("Fe"), "Fe.cif", fmt="cif")   # EMT has no Fe
    r = submit(idle_client, [cu["id"], fe["id"]])
    assert r.status_code == 422
    fields = [d["field"] for d in r.json()["error"]["details"]]
    assert fields and all(f.startswith("structure_ids[1].") for f in fields)
    assert "structure_ids[1].calculator.name" in fields
    assert idle_client.get("/api/jobs").json()["jobs"] == []
    assert sorted(p.name for p in runs_dir.iterdir()) == ["_uploads"]

    r = submit(idle_client, [cu["id"], cu["id"]], label="two")
    assert r.status_code == 201
    jobs = r.json()["jobs"]
    assert [j["queue_position"] for j in jobs] == [1, 2]
    assert [j["label"] for j in jobs] == ["two-00", "two-01"]       # PLAN 15.1
    assert [j["folder"] for j in jobs] == ["two-00", "two-01"]
    listed = idle_client.get("/api/jobs").json()["jobs"]
    assert [j["job_id"] for j in listed] == [jobs[1]["job_id"], jobs[0]["job_id"]]


def test_submission_shape_errors(idle_client, upload_atoms, submit, H):
    up = upload_atoms(idle_client, bulk("Cu"), "POSCAR", fmt="vasp")
    # unknown top-level key
    r = submit(idle_client, [up["id"]], colour="red")
    assert r.status_code == 422
    assert r.json()["error"]["details"][0]["field"] == "colour"
    # id key that does not apply
    r = submit(idle_client, [up["id"]], band_id=up["id"])
    assert r.status_code == 422
    assert [d["field"] for d in r.json()["error"]["details"]] == ["band_id"]
    # unknown calc type
    r = submit(idle_client, [up["id"]], calc_type="benchmark")
    assert r.json()["error"]["details"][0]["field"] == "calc_type"
    # structure_every out of range, label too long
    r = submit(idle_client, [up["id"]], structure_every=-1, label="x" * 81)
    assert {d["field"] for d in r.json()["error"]["details"]} == {"structure_every", "label"}
    # unknown upload id -> 404 with the field path
    r = submit(idle_client, [up["id"], "0123456789ab"])
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "not_found"
    assert err["details"][0]["field"] == "structure_ids[1]"
    # malformed JSON
    r = idle_client.post("/api/jobs", headers={**H, "Content-Type": "application/json"},
                         content=b"{not json")
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"
    # neb needs initial+final or band, not structure_ids
    r = submit(idle_client, [up["id"]], calc_type="neb")
    fields = {d["field"] for d in r.json()["error"]["details"]}
    assert {"initial_id", "final_id", "structure_ids"} <= fields
    assert idle_client.get("/api/jobs").json()["jobs"] == []


def test_neb_pair_and_band_submissions(idle_client, upload_atoms, H):
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    b = a.copy()
    b.positions[0, 0] += 0.2
    ini = upload_atoms(idle_client, a, "ini.vasp", fmt="vasp")
    fin = upload_atoms(idle_client, b, "fin.vasp", fmt="vasp")
    band = upload_atoms(idle_client, [a, a, b], "band.extxyz", fmt="extxyz")
    calc = {"name": "emt", "device": "cpu"}
    r = idle_client.post("/api/jobs", headers=H, json={
        "calc_type": "neb", "calculator": calc, "initial_id": ini["id"], "final_id": fin["id"]})
    assert r.status_code == 201, r.text
    assert r.json()["jobs"][0]["label"] == "ini.vasp -> fin.vasp"
    r = idle_client.post("/api/jobs", headers=H, json={
        "calc_type": "neb", "calculator": calc, "band_id": band["id"],
        "params": {"fmax": 0.1}})
    assert r.status_code == 201, r.text
    job_id = r.json()["jobs"][0]["job_id"]
    d = idle_client.get(f"/api/jobs/{job_id}").json()
    assert d["label"] == "band.extxyz"
    assert d["inputs"]["band"]["frames_xyz"].count("\n4\n") == 2
    r = idle_client.post("/api/jobs", headers=H, json={
        "calc_type": "neb", "calculator": calc, "band_id": band["id"],
        "initial_id": ini["id"], "final_id": fin["id"]})
    assert r.status_code == 422
    r = idle_client.post("/api/jobs", headers=H, json={
        "calc_type": "autoneb", "calculator": calc, "band_id": band["id"]})
    assert r.status_code == 422
    assert {"initial_id", "final_id", "band_id"} <= {
        d["field"] for d in r.json()["error"]["details"]}


def test_list_jobs_filters(idle_client, upload_atoms, submit):
    up = upload_atoms(idle_client, bulk("Cu"), "POSCAR", fmt="vasp")
    ids = [j["job_id"] for j in submit(idle_client, [up["id"]] * 3).json()["jobs"]]
    idle_client.post(f"/api/jobs/{ids[0]}/cancel", headers={"X-Mliprun": "1"})
    got = idle_client.get("/api/jobs?status=pending").json()["jobs"]
    assert [j["job_id"] for j in got] == [ids[2], ids[1]]
    assert [j["queue_position"] for j in got] == [2, 1]
    got = idle_client.get("/api/jobs?status=cancelled,done&limit=5").json()["jobs"]
    assert [j["job_id"] for j in got] == [ids[0]]
    assert len(idle_client.get("/api/jobs?limit=1").json()["jobs"]) == 1
    r = idle_client.get("/api/jobs?status=sleeping")
    assert r.status_code == 422 and r.json()["error"]["details"][0]["field"] == "status"
    r = idle_client.get("/api/jobs?limit=0")
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error" and err["details"][0]["field"] == "limit"


def test_events_endpoint_paging(client, cu_upload, submit, wait_for_status):
    job_id = submit(client, [cu_upload["id"]], label="steps=6").json()["jobs"][0]["job_id"]
    d = wait_for_status(client, job_id, "done")
    all_ev = client.get(f"/api/jobs/{job_id}/events").json()
    seqs = [e["seq"] for e in all_ev["events"]]
    assert seqs == list(range(len(seqs)))
    assert all_ev["terminal"] is True and all_ev["next_seq"] == len(seqs)
    assert d["last_seq"] == len(seqs) - 1
    types = [e["type"] for e in all_ev["events"]]
    assert types[:2] == ["job_status", "job_status"] and types[-1] == "job_status"
    assert [e["status"] for e in all_ev["events"] if e["type"] == "job_status"] == \
        ["pending", "running", "done"]
    assert types.count("opt_step") == 6 and types.count("result") == 1
    page = client.get(f"/api/jobs/{job_id}/events?since=3&limit=2").json()
    assert [e["seq"] for e in page["events"]] == [3, 4]
    assert page["next_seq"] == 5 and page["terminal"] is False
    empty = client.get(f"/api/jobs/{job_id}/events?since=999").json()
    assert empty == {"events": [], "next_seq": 999, "terminal": False}
    r = client.get(f"/api/jobs/{job_id}/events?since=-1")
    assert r.status_code == 422 and r.json()["error"]["details"][0]["field"] == "since"
    # summary bits derived from events
    assert d["progress"] == {"step": 5, "total": 200, "energy": pytest.approx(-1.5),
                             "fmax": pytest.approx(1 / 6), "text": "step 5/200"}
    assert d["result_brief"] == {"converged": True, "final_energy_eV": pytest.approx(-1.5),
                                 "final_fmax_eV_per_A": 0.01, "steps": 6}
    assert d["result"]["converged"] is True and "seq" not in d["result"]
    assert d["reproduce"] == {"status": "no", "notes": ["fake worker"]}
    assert d["returncode"] == 0 and d["error"] is None
    assert d["started_at"] <= d["finished_at"]


def test_health(client, runs_dir):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["queue_length"] == 0 and h["running_job"] is None
    assert h["runs_dir"] == str(runs_dir.resolve())
    assert isinstance(h["mliprun_version"], str)


def test_calc_types_and_calculators(client):
    ct = client.get("/api/calc-types").json()
    assert [c["name"] for c in ct["calc_types"]] == ["singlepoint", "optimize", "freq", "md",
                                                    "neb", "autoneb"]
    calcs = client.get("/api/calculators").json()
    assert calcs["calculators"][-1]["name"] == "emt"
    assert calcs["devices"]["choices"] == ["auto", "cpu", "cuda"]
