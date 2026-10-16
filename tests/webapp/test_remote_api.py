"""Remote machines through the REST API, no jobs running (REMOTE_PLAN sections 5, 8)."""
import json
import subprocess
import sys

import pytest
from ase.build import bulk
from remote_helpers import FAKE_SSH_CMD

H = {"X-Mliprun": "1"}
GOOD = {"name": "cos", "host": "cos-host", "default_location": "/scratch/x"}


@pytest.fixture
def cu_for(upload_atoms):
    def make(client):
        return upload_atoms(client, bulk("Cu", "fcc", a=3.6, cubic=True), "POSCAR", fmt="vasp")
    return make


def details(r, status=422):
    assert r.status_code == status, r.text
    return r.json()["error"]["details"]


# --------------------------------------------------------------------------- #
# machines
# --------------------------------------------------------------------------- #
def test_machine_crud_and_persistence(make_client, tmp_path):
    c = make_client()
    assert c.get("/api/machines").json() == {"machines": []}
    r = c.post("/api/machines", headers=H, json=GOOD)
    assert r.status_code == 201
    m = r.json()
    assert m["id"].startswith("m-") and len(m["id"]) == 8 and m["poll_interval_s"] == 2.0
    r = c.put(f"/api/machines/{m['id']}", headers=H, json={"poll_interval_s": 7, "user": "juar"})
    assert r.status_code == 200 and r.json()["poll_interval_s"] == 7.0
    on_disk = json.loads((tmp_path / "machines.json").read_text())
    assert on_disk["machines"][0]["user"] == "juar"
    assert c.get("/api/machines").json()["machines"][0]["host"] == "cos-host"
    assert c.delete(f"/api/machines/{m['id']}", headers=H).json() == {"deleted": m["id"]}
    assert c.get("/api/machines").json()["machines"] == []
    assert json.loads((tmp_path / "machines.json").read_text())["machines"] == []


def test_machine_routes_require_header_and_validate(make_client):
    c = make_client()
    assert c.post("/api/machines", json=GOOD).status_code == 403
    assert c.delete("/api/machines/m-000000").status_code == 403
    fields = {d["field"] for d in details(c.post(
        "/api/machines", headers=H, json={**GOOD, "host": "a b", "port": 70000}))}
    assert fields == {"host", "port"}
    r = c.post("/api/machines", headers=H, content=b"[1]")
    assert r.status_code == 400
    assert c.put("/api/machines/m-000000", headers=H, json={}).status_code == 404
    assert c.delete("/api/machines/m-000000", headers=H).status_code == 404
    c.post("/api/machines", headers=H, json=GOOD)
    d = details(c.post("/api/machines", headers=H, json={**GOOD, "host": "other"}))
    assert d[0]["field"] == "name" and "already exists" in d[0]["message"]


def test_machine_test_connection(make_client, tmp_path):
    c = make_client()
    mid = c.post("/api/machines", headers=H, json=GOOD).json()["id"]
    out = c.post(f"/api/machines/{mid}/test", headers=H).json()
    expected = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
    assert out["ok"] is True and out["hostname"] == expected
    assert out["gpus"] == [] and "add an environment" in out["message"]
    assert out["elapsed_s"] >= 0
    c.post(f"/api/machines/{mid}/envs", headers=H, json={"python": sys.executable})
    out = c.post(f"/api/machines/{mid}/test", headers=H).json()
    assert out["ok"] is True and isinstance(out["gpus"], list)
    down = c.post("/api/machines", headers=H,
                  json={"name": "dead", "host": "down", "default_location": "/x"}).json()["id"]
    out = c.post(f"/api/machines/{down}/test", headers=H).json()
    assert out["ok"] is False and "Connection refused" in out["message"]
    assert out["hostname"] is None and out["gpus"] == []
    assert c.post("/api/machines/m-000000/test", headers=H).status_code == 404


def test_env_add_probes_and_calculators_per_target(make_client):
    c = make_client()
    mid = c.post("/api/machines", headers=H, json=GOOD).json()["id"]
    r = c.post(f"/api/machines/{mid}/envs", headers=H, json={"python": sys.executable})
    assert r.status_code == 201
    env = r.json()
    assert env["probe"]["ok"] is True and env["probe"]["agent_version"] == 1
    assert env["name"].startswith("cos-")
    cal = c.get(f"/api/calculators?target={env['id']}")
    assert cal.status_code == 200
    names = [e["name"] for e in cal.json()["calculators"]]
    assert "emt" in names and cal.json() == env["probe"]["calculators"]
    local = [e["name"] for e in c.get("/api/calculators").json()["calculators"]]
    assert names == local                                  # same interpreter, same install
    assert c.get("/api/calculators?target=local").json()["default"] == c.get(
        "/api/calculators").json()["default"]
    assert c.get("/api/calculators?target=e-000000").status_code == 404
    # a python that cannot be run: stored, with a failed probe
    bad = c.post(f"/api/machines/{mid}/envs", headers=H, json={
        "python": "/nonexistent/bin/python", "name": "broken"})
    assert bad.status_code == 201 and bad.json()["probe"]["ok"] is False
    r = c.get(f"/api/calculators?target={bad.json()['id']}")
    assert r.status_code == 409 and "not been probed successfully" in r.json()["error"]["message"]
    # re-probe, edit, delete
    r = c.post(f"/api/machines/{mid}/envs/{env['id']}/probe", headers=H)
    assert r.status_code == 200 and r.json()["probe"]["ok"] is True
    r = c.put(f"/api/machines/{mid}/envs/{env['id']}", headers=H, json={"setup": "true"})
    assert r.status_code == 200 and r.json()["setup"] == "true"
    assert c.put(f"/api/machines/{mid}/envs/e-000000", headers=H, json={}).status_code == 404
    d = details(c.post(f"/api/machines/{mid}/envs", headers=H, json={"python": "rel/python"}))
    assert d[0]["field"] == "python"
    assert c.delete(f"/api/machines/{mid}/envs/{bad.json()['id']}", headers=H).status_code == 200
    assert c.delete(f"/api/machines/{mid}/envs/e-000000", headers=H).status_code == 404
    assert c.post(f"/api/machines/{mid}/envs/e-000000/probe", headers=H).status_code == 404


def test_env_probe_on_unreachable_machine_is_stored_as_failed(make_client):
    c = make_client()
    mid = c.post("/api/machines", headers=H,
                 json={"name": "dead", "host": "down", "default_location": "/x"}).json()["id"]
    env = c.post(f"/api/machines/{mid}/envs", headers=H, json={"python": sys.executable}).json()
    assert env["probe"]["ok"] is False and "Connection refused" in env["probe"]["error"]


def test_machines_file_survives_a_restart(make_client, remote_setup):
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"])
    machines = c.get("/api/machines").json()["machines"]
    assert [m["id"] for m in machines] == [info["machine_id"]]
    assert machines[0]["envs"][0]["id"] == info["env_id"]
    assert machines[0]["envs"][0]["probe"]["ok"] is True


# --------------------------------------------------------------------------- #
# calculator catalog
# --------------------------------------------------------------------------- #
def test_validate_calculator_with_catalog():
    from mliprun.webapp import calculators

    catalog = {"calculators": [
        {"name": "emt", "task_field": None, "task_choices": [], "task_required": False,
         "elements": ["Cu"]},
        {"name": "fancy", "task_field": "uma_task", "task_choices": ["omat", "oc20"],
         "task_required": True, "elements": None}],
        "devices": {"cuda_available": False}, "env_name": "box-mace"}
    norm, errs = calculators.validate_calculator({"name": "fancy", "uma_task": "omat"}, None,
                                                 catalog)
    assert errs == [] and norm["uma_task"] == "omat"
    _, errs = calculators.validate_calculator({"name": "mace"}, None, catalog)
    assert errs == [{"field": "calculator.name", "message":
                     "mace is not installed in environment box-mace; offered there: emt, fancy"}]
    _, errs = calculators.validate_calculator({"name": "fancy", "uma_task": "omat",
                                               "device": "cuda"}, None, catalog)
    assert [e["message"] for e in errs] == ["CUDA is not available on this machine"]
    _, errs = calculators.validate_calculator({"name": "emt", "device": "cuda"}, {"Fe"}, catalog)
    assert len(errs) == 1 and "EMT has no parameters for Fe" in errs[0]["message"]
    _, errs = calculators.validate_calculator({"name": "fancy"}, None, catalog)
    assert "required for fancy; one of omat, oc20" in errs[0]["message"]
    nameless = dict(catalog)
    nameless.pop("env_name")
    _, errs = calculators.validate_calculator({"name": "x"}, None, nameless)
    assert "not installed in environment this environment" in errs[0]["message"]
    # catalog=None is today's behaviour
    _, errs = calculators.validate_calculator({"name": "nonsense"})
    assert "unknown calculator" in errs[0]["message"]


def test_submission_checks_the_env_catalog_not_the_servers(make_client, remote_setup, cu_for,
                                                          real_probe):
    info = remote_setup()
    store_path = info["machines_file"]
    data = json.loads(store_path.read_text())
    probe = data["machines"][0]["envs"][0]["probe"]
    probe["calculators"]["calculators"] = [
        e for e in probe["calculators"]["calculators"] if e["name"] == "emt"] + [
        {"name": "chgnet", "label": "CHGNet", "family": "chgnet", "task_field": None,
         "task_choices": [], "task_required": False, "elements": None,
         "cli_reproducible": True, "note": ""}]
    probe["calculators"]["devices"]["cuda_available"] = False
    store_path.write_text(json.dumps(data))
    c = make_client(machines_file=store_path)
    up = cu_for(c)
    body = {"calc_type": "singlepoint", "structure_ids": [up["id"]], "target": info["env_id"],
            "calculator": {"name": "mace"}}
    d = details(c.post("/api/jobs", headers=H, json=body))
    assert d[0]["field"] == "calculator.name"
    assert d[0]["message"] == ("mace is not installed in environment box-env; "
                               "offered there: emt, chgnet")
    body["calculator"] = {"name": "chgnet", "device": "cuda"}
    assert details(c.post("/api/jobs", headers=H, json=body))[0]["message"].startswith("CUDA")
    assert c.get("/api/jobs").json()["jobs"] == []


# --------------------------------------------------------------------------- #
# paths/check
# --------------------------------------------------------------------------- #
def test_paths_check_remote(make_client, remote_setup, runs_dir):
    info = remote_setup()
    (info["location"] / "taken").mkdir()
    (runs_dir / "local-only").mkdir()
    c = make_client(machines_file=info["machines_file"])
    r = c.post("/api/paths/check", headers=H, json={
        "target": info["env_id"], "folders": ["taken", "local-only", "fresh one", "_uploads"]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["location"] == str(info["location"]) and out["exists"] is True
    assert out["writable"] is True and out["error"] is None
    by = {f["input"]: f for f in out["folders"]}
    assert by["taken"]["exists"] is True                        # remote
    assert by["local-only"]["exists"] is True                   # local copy location
    assert by["fresh one"]["exists"] is False and by["fresh one"]["folder"] == "fresh-one"
    assert by["fresh one"]["path"] == str(info["location"] / "fresh-one")
    assert by["fresh one"]["local_path"] == str(runs_dir / "fresh-one")
    assert by["_uploads"]["error"] is None            # not reserved on the remote
    assert out["local_location"] == {"location": str(runs_dir), "exists": True,
                                     "writable": True, "error": None}
    r = c.post("/api/paths/check", headers=H, json={
        "target": info["env_id"], "location": str(info["location"] / "new" / "deep"),
        "local_location": "relative"})
    out = r.json()
    assert out["exists"] is False and out["writable"] is None
    assert "absolute" in out["local_location"]["error"]
    assert c.post("/api/paths/check", headers=H, json={"target": "e-000000"}).status_code == 404
    assert c.post("/api/paths/check", headers=H, json={"target": 5}).status_code == 422


def test_paths_check_unreachable_is_200_with_error(make_client, remote_setup, monkeypatch,
                                                   tmp_path):
    info = remote_setup(host="down")
    c = make_client(machines_file=info["machines_file"])
    r = c.post("/api/paths/check", headers=H, json={"target": info["env_id"],
                                                    "folders": ["a b"]})
    assert r.status_code == 200
    out = r.json()
    assert "Connection refused" in out["error"] and out["exists"] is False
    assert out["folders"][0]["folder"] == "a-b" and out["folders"][0]["exists"] is False
    assert out["local_location"]["error"] is None


# --------------------------------------------------------------------------- #
# submission to a remote target (runner off: jobs stay pending)
# --------------------------------------------------------------------------- #
def remote_body(info, up_id, **extra):
    body = {"calc_type": "optimize", "structure_ids": [up_id], "target": info["env_id"],
            "calculator": {"name": "emt", "device": "cpu"}, "params": {}}
    body.update(extra)
    return body


def test_remote_submit_creates_both_copies(make_client, remote_setup, cu_for, runs_dir):
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"])
    up = cu_for(c)
    r = c.post("/api/jobs", headers=H, json=remote_body(info, up["id"], folder="my run",
                                                        label="remote one"))
    assert r.status_code == 201, r.text
    job = r.json()["jobs"][0]
    assert job["status"] == "pending" and job["queue_position"] == 1
    assert job["target"] == {"kind": "remote", "machine_name": "box", "env_name": "box-env"}
    remote_dir = info["location"].resolve() / "my-run"
    assert job["remote_run_dir"] == str(remote_dir)
    assert job["run_dir"] == str(runs_dir / "my-run")
    spec = json.loads((runs_dir / "my-run" / "job.json").read_text())
    assert spec["schema_version"] == 2 and spec["remote_run_dir"] == str(remote_dir)
    assert spec["target"]["env_id"] == info["env_id"] and spec["target"]["host"] == "box"
    assert spec["location"] == str(info["location"]) and spec["local_location"] == str(runs_dir)
    assert (remote_dir / "job.json").is_file()
    remote_spec = json.loads((remote_dir / "job.json").read_text())
    assert remote_spec["job_id"] == job["job_id"]
    assert (remote_dir / "inputs/structure/POSCAR").read_bytes() == (
        runs_dir / "my-run" / "inputs/structure/POSCAR").read_bytes()
    assert not (remote_dir / "events.jsonl").exists()      # created at launch
    detail = c.get(f"/api/jobs/{job['job_id']}").json()
    assert detail["remote_run_dir"] == str(remote_dir) and detail["remote_error"] is None


def test_remote_submit_to_other_local_copy_location(make_client, remote_setup, cu_for,
                                                    tmp_path):
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"])
    up = cu_for(c)
    other = tmp_path / "copies"
    r = c.post("/api/jobs", headers=H, json=remote_body(
        info, up["id"], folder="x1", local_location=str(other),
        location=str(info["location"] / "sub")))
    assert r.status_code == 201, r.text
    assert (other / "x1" / "job.json").is_file()
    assert (info["location"] / "sub" / "x1" / "job.json").is_file()


def test_remote_folder_exists_is_422_and_nothing_left(make_client, remote_setup, cu_for,
                                                      runs_dir):
    info = remote_setup()
    (info["location"] / "taken").mkdir()
    (info["location"] / "taken" / "keep.txt").write_text("mine")
    c = make_client(machines_file=info["machines_file"])
    up = cu_for(c)
    d = details(c.post("/api/jobs", headers=H, json=remote_body(info, up["id"],
                                                                folder="taken")))
    assert d[0]["field"] == "folder" and d[0]["code"] == "folder_exists"
    assert "box" in d[0]["message"]
    assert (info["location"] / "taken" / "keep.txt").read_text() == "mine"   # untouched
    assert not (runs_dir / "taken").exists()
    assert c.get("/api/jobs").json()["jobs"] == []


def test_remote_unreachable_submit_is_422_on_target(make_client, remote_setup, cu_for, runs_dir):
    info = remote_setup(host="down")
    c = make_client(machines_file=info["machines_file"])
    up = cu_for(c)
    d = details(c.post("/api/jobs", headers=H, json=remote_body(info, up["id"], folder="f")))
    assert d[0]["field"] == "target" and "Connection refused" in d[0]["message"]
    assert not (runs_dir / "f").exists() and c.get("/api/jobs").json()["jobs"] == []


def test_multi_submit_rolls_back_remote_and_local(make_client, remote_setup, cu_for, runs_dir):
    info = remote_setup()
    (info["location"] / "grp-01").mkdir()                    # the second job's remote folder
    c = make_client(machines_file=info["machines_file"])
    ups = [cu_for(c)["id"], cu_for(c)["id"]]
    body = remote_body(info, ups[0], folder="grp")
    body.pop("structure_ids")
    body["structure_ids"] = ups
    d = details(c.post("/api/jobs", headers=H, json=body))
    assert d[0]["field"] == "structure_ids[1].folder" and d[0]["code"] == "folder_exists"
    assert not (info["location"] / "grp-00").exists()        # first remote dir rolled back
    assert (info["location"] / "grp-01").is_dir()            # not ours: kept
    assert not (runs_dir / "grp-00").exists() and not (runs_dir / "grp-01").exists()
    assert c.get("/api/jobs").json()["jobs"] == []


@pytest.mark.parametrize("extra, field", [
    ({"target": "e-000000"}, "target"),
    ({"target": 5}, "target"),
    ({"location": "relative/dir"}, "location"),
    ({"local_location": "relative"}, "local_location"),
], ids=["unknown_target", "bad_target_type", "bad_remote_location", "bad_local_location"])
def test_remote_submit_validation(make_client, remote_setup, cu_for, extra, field):
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"])
    up = cu_for(c)
    body = remote_body(info, up["id"])
    body.update(extra)
    assert details(c.post("/api/jobs", headers=H, json=body))[0]["field"] == field


def test_unprobed_env_and_local_with_local_location_are_rejected(make_client, remote_setup,
                                                                 cu_for):
    info = remote_setup()
    data = json.loads(info["machines_file"].read_text())
    data["machines"][0]["envs"][0]["probe"] = {"ok": False, "error": "x"}
    info["machines_file"].write_text(json.dumps(data))
    c = make_client(machines_file=info["machines_file"])
    up = cu_for(c)
    d = details(c.post("/api/jobs", headers=H, json=remote_body(info, up["id"])))
    assert d[0]["field"] == "target" and "not been probed successfully" in d[0]["message"]
    body = remote_body(info, up["id"])
    body["target"] = "local"
    body["local_location"] = "/tmp"
    assert details(c.post("/api/jobs", headers=H, json=body))[0]["field"] == "local_location"


def test_delete_machine_or_env_with_a_pending_job_is_409(make_client, remote_setup, cu_for):
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"])
    up = cu_for(c)
    job = c.post("/api/jobs", headers=H, json=remote_body(info, up["id"])).json()["jobs"][0]
    r = c.delete(f"/api/machines/{info['machine_id']}", headers=H)
    assert r.status_code == 409 and "1 unfinished job" in r.json()["error"]["message"]
    r = c.delete(f"/api/machines/{info['machine_id']}/envs/{info['env_id']}", headers=H)
    assert r.status_code == 409
    assert c.get("/api/machines").json()["machines"][0]["envs"][0]["id"] == info["env_id"]
    assert c.post(f"/api/jobs/{job['job_id']}/cancel", headers=H).json()["job"]["status"] == \
        "cancelled"
    assert c.delete(f"/api/machines/{info['machine_id']}/envs/{info['env_id']}",
                    headers=H).status_code == 200
    assert c.delete(f"/api/machines/{info['machine_id']}", headers=H).status_code == 200


def test_queue_position_is_per_machine(make_client, remote_setup, cu_for):
    a = remote_setup(name="alpha", host="alpha")
    b = remote_setup(name="beta", host="beta", store_path=a["machines_file"])
    c = make_client(machines_file=a["machines_file"])
    up = cu_for(c)

    def submit(target):
        body = remote_body(a, up["id"], target=target)
        return c.post("/api/jobs", headers=H, json=body).json()["jobs"][0]

    ja1, jb1, ja2 = submit(a["env_id"]), submit(b["env_id"]), submit(a["env_id"])
    local = c.post("/api/jobs", headers=H, json={
        "calc_type": "optimize", "structure_ids": [up["id"]],
        "calculator": {"name": "emt", "device": "cpu"}}).json()["jobs"][0]
    assert [ja1["queue_position"], jb1["queue_position"], ja2["queue_position"],
            local["queue_position"]] == [1, 1, 2, 1]
    assert c.get("/api/health").json()["queue_length"] == 4
    assert c.get("/api/health").json()["running_jobs"] == []


def test_machine_removed_from_the_file_marks_pending_job(make_client, remote_setup, cu_for,
                                                        make_app):
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"])
    up = cu_for(c)
    job = c.post("/api/jobs", headers=H, json=remote_body(info, up["id"])).json()["jobs"][0]
    c.__exit__(None, None, None)
    info["machines_file"].write_text(json.dumps({"schema_version": 1, "machines": []}))
    from fastapi.testclient import TestClient

    with TestClient(make_app(ssh_command=FAKE_SSH_CMD, machines_file=info["machines_file"],
                             start_runner=True)) as c2:
        got = c2.get(f"/api/jobs/{job['job_id']}").json()
        assert got["status"] == "pending"
        assert got["remote_error"] == "machine box no longer configured"
        r = c2.post(f"/api/jobs/{job['job_id']}/cancel", headers=H)
        assert r.json()["job"]["status"] == "cancelled"


def test_cli_machines_file_option(monkeypatch, tmp_path):
    import uvicorn
    from typer.testing import CliRunner

    import mliprun.webapp.cli as wcli
    import mliprun.webapp.server as server
    from mliprun.cli.main import app

    calls = {}
    real = server.create_app
    monkeypatch.setattr(server, "create_app",
                        lambda runs, **kw: calls.setdefault("kw", kw) and real(
                            runs, start_runner=False, **kw))
    monkeypatch.setattr(uvicorn, "run", lambda app_, **kw: None)
    r = CliRunner().invoke(app, ["webapp", "--runs-dir", str(tmp_path / "runs"),
                                 "--machines-file", str(tmp_path / "m.json"), "--no-browser"])
    assert r.exit_code == 0, r.output
    assert calls["kw"]["machines_file"] == tmp_path / "m.json"
    assert wcli.DEFAULT_MACHINES_FILE == "~/.mliprun/webapp-machines.json"
