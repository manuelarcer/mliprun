"""Robustness tests (PLAN 12, Phase 3, test_robustness.py): real worker, EMT.

Serial queue, cancellation, worker crashes (test hook and external kill),
restart recovery, malformed inputs, NEB mismatches, error shapes, security
headers, path traversal and WebSocket replay.
"""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import ase.io  # noqa: E402
from ase.build import bulk  # noqa: E402

H = {"X-Mliprun": "1"}
TERMINAL = ("done", "failed", "cancelled")
EMT_CPU = {"name": "emt", "device": "cpu"}


def make_client(runs, **kw):
    from fastapi.testclient import TestClient
    from mliprun.webapp.server import create_app
    kw.setdefault("cancel_grace_s", 3.0)
    kw.setdefault("poll_interval_s", 0.05)
    return TestClient(create_app(runs, **kw))


@pytest.fixture
def runs(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture
def client(runs):
    with make_client(runs) as c:
        yield c


def upload(client, tmp, atoms, name="POSCAR", fmt="vasp"):
    p = tmp / name
    ase.io.write(str(p), atoms, format=fmt)
    r = client.post("/api/uploads", headers=H, data={"format": fmt},
                    files=[("files", (name, p.read_bytes()))])
    assert r.status_code == 200, r.text
    u = r.json()["uploads"][0]
    assert u["ok"], u
    return u


def submit(client, ids, calc_type="singlepoint", params=None):
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": calc_type, "calculator": EMT_CPU, "params": params or {},
        "structure_ids": list(ids)})
    assert r.status_code == 201, r.text
    return [j["job_id"] for j in r.json()["jobs"]]


def wait(client, jid, statuses=TERMINAL, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        d = client.get(f"/api/jobs/{jid}").json()
        if d["status"] in statuses:
            return d
        time.sleep(0.05)
    raise AssertionError(f"{jid} stuck in {d['status']}")


def wait_for_event(client, jid, etype, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        evs = client.get(f"/api/jobs/{jid}/events").json()["events"]
        if any(e["type"] == etype for e in evs):
            return evs
        time.sleep(0.05)
    raise AssertionError(f"no {etype} event for {jid}")


def cu(n=1, rattle=True):
    a = bulk("Cu", "fcc", cubic=True).repeat(n)
    if rattle:
        a.rattle(0.05, seed=1)
    return a


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # zombie counts as dead
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


def iso(s):
    from datetime import datetime
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Queue: strictly serial
# ---------------------------------------------------------------------------

def test_three_jobs_run_strictly_in_series(client, tmp_path):
    ids = [upload(client, tmp_path, cu(2), name=f"s{i}.vasp")["id"] for i in range(3)]
    jids = submit(client, ids, "md", {"ensemble": "nve", "steps": 30, "log_interval": 5})
    running_seen = set()
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        jobs = {j["job_id"]: j for j in client.get("/api/jobs").json()["jobs"]}
        running = [j for j in jids if jobs[j]["status"] == "running"]
        assert len(running) <= 1, running
        running_seen.update(running)
        h = client.get("/api/health").json()
        assert h["running_job"] in (None, *running)
        if all(jobs[j]["status"] in TERMINAL for j in jids):
            break
        time.sleep(0.02)
    details = [client.get(f"/api/jobs/{j}").json() for j in jids]
    assert [d["status"] for d in details] == ["done"] * 3
    for a, b in zip(details, details[1:]):
        assert iso(b["started_at"]) >= iso(a["finished_at"])      # FIFO and no overlap
    # same check from the worker's own event timestamps
    spans = []
    for j in jids:
        evs = client.get(f"/api/jobs/{j}/events").json()["events"]
        worker = [e["t"] for e in evs if e["type"] != "job_status"]
        spans.append((min(worker), max(worker)))
    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        assert s1 > e0


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

def test_cancel_pending_without_worker(client, tmp_path):
    u = upload(client, tmp_path, cu(2))
    first, second = submit(client, [u["id"], u["id"]], "md",
                           {"ensemble": "nve", "steps": 1000000, "log_interval": 100})
    wait(client, first, ("running",))
    r = client.post(f"/api/jobs/{second}/cancel", headers=H)
    assert r.status_code == 200 and r.json()["job"]["status"] == "cancelled"
    d = client.get(f"/api/jobs/{second}").json()
    run_dir = Path(d["run_dir"])
    assert d["started_at"] is None
    assert not (run_dir / "worker.log").exists()
    assert not (run_dir / "summary.json").exists()
    state = json.loads((run_dir / "state.json").read_text())
    assert state["pid"] is None and state["status"] == "cancelled"
    client.post(f"/api/jobs/{first}/cancel", headers=H)
    wait(client, first, timeout=30)


def test_cancel_running_md(client, tmp_path):
    u = upload(client, tmp_path, cu(2))
    (jid,) = submit(client, [u["id"]], "md",
                    {"ensemble": "nve", "steps": 1000000, "log_interval": 10})
    wait_for_event(client, jid, "md_step")
    d = client.get(f"/api/jobs/{jid}").json()
    assert d["status"] == "running"
    pid = json.loads((Path(d["run_dir"]) / "state.json").read_text())["pid"]
    assert pid and pid_alive(pid)
    t0 = time.monotonic()
    r = client.post(f"/api/jobs/{jid}/cancel", headers=H)
    assert r.status_code == 200
    assert r.json()["job"]["cancel_requested"] is True
    d = wait(client, jid, timeout=3.0 + 5)
    assert time.monotonic() - t0 <= 3.0 + 5
    assert d["status"] == "cancelled"
    assert not pid_alive(pid)
    run_dir = Path(d["run_dir"])
    record = json.loads((run_dir / "mliprun_run.json").read_text())
    assert record["status"] == "failed"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "cancelled"
    evs = client.get(f"/api/jobs/{jid}/events").json()
    assert evs["terminal"] and evs["events"][-1]["status"] == "cancelled"
    assert client.post(f"/api/jobs/{jid}/cancel", headers=H).status_code == 409


def test_cancel_sigkill_after_grace(runs, tmp_path):
    """A worker ignoring SIGTERM is SIGKILLed after cancel_grace_s."""
    script = tmp_path / "stubborn.py"
    script.write_text(
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(600)\n")
    import sys
    with make_client(runs, worker_cmd=[sys.executable, str(script)], cancel_grace_s=1.0) as c:
        u = upload(c, tmp_path, cu())
        (jid,) = submit(c, [u["id"]])
        d = wait(c, jid, ("running",))
        time.sleep(0.5)
        t0 = time.monotonic()
        c.post(f"/api/jobs/{jid}/cancel", headers=H)
        d = wait(c, jid, timeout=10)
        assert d["status"] == "cancelled"
        assert 0.9 <= time.monotonic() - t0 <= 6
        summary = json.loads((Path(d["run_dir"]) / "summary.json").read_text())
        assert summary["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Worker crashes
# ---------------------------------------------------------------------------

def test_crash_hook_marks_failed_and_server_continues(client, tmp_path, monkeypatch):
    u = upload(client, tmp_path, cu())
    monkeypatch.setenv("MLIPRUN_WEBAPP_TEST_CRASH", "after_first_event")
    (jid,) = submit(client, [u["id"]])
    d = wait(client, jid)
    monkeypatch.delenv("MLIPRUN_WEBAPP_TEST_CRASH")
    assert d["status"] == "failed"
    assert d["returncode"] == 7
    assert "7" in d["error"]
    summary = json.loads((Path(d["run_dir"]) / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["error"]["message"] == d["error"]
    evs = client.get(f"/api/jobs/{jid}/events").json()["events"]
    assert evs[-1]["type"] == "job_status" and evs[-1]["status"] == "failed"
    assert evs[-1]["returncode"] == 7
    assert [e["seq"] for e in evs] == list(range(len(evs)))
    assert client.get("/api/health").status_code == 200
    (jid2,) = submit(client, [u["id"]])
    assert wait(client, jid2)["status"] == "done"


def test_external_kill_mid_job(client, tmp_path):
    u = upload(client, tmp_path, cu(2))
    (jid,) = submit(client, [u["id"]], "md",
                    {"ensemble": "nve", "steps": 1000000, "log_interval": 10})
    (jid_next,) = submit(client, [u["id"]], "singlepoint")
    wait_for_event(client, jid, "md_step")
    d = client.get(f"/api/jobs/{jid}").json()
    pid = json.loads((Path(d["run_dir"]) / "state.json").read_text())["pid"]
    os.kill(pid, signal.SIGKILL)
    d = wait(client, jid, timeout=30)
    assert d["status"] == "failed"
    assert d["returncode"] == -9
    assert "signal 9" in d["error"]
    summary = json.loads((Path(d["run_dir"]) / "summary.json").read_text())
    assert summary["status"] == "failed"
    h = client.get("/api/health")
    assert h.status_code == 200 and h.json()["status"] == "ok"
    assert wait(client, jid_next)["status"] == "done"


# ---------------------------------------------------------------------------
# Restart recovery (D6)
# ---------------------------------------------------------------------------

def test_restart_recovery(runs, tmp_path):
    with make_client(runs, start_runner=False) as c:
        u = upload(c, tmp_path, cu())
        j_running, j_pending = submit(c, [u["id"], u["id"]])
    sp = runs / j_running / "state.json"
    state = json.loads(sp.read_text())
    state.update(status="running", pid=2 ** 22 + 12345, started_at=state["created_at"])
    sp.write_text(json.dumps(state))
    # a partial last line in events.jsonl must be tolerated (truncated)
    with open(runs / j_running / "events.jsonl", "a") as fh:
        fh.write('{"job_id": "x", "seq": 1, "ty')
    (runs / "not-a-job").mkdir()
    with make_client(runs) as c:
        d = c.get(f"/api/jobs/{j_running}").json()
        assert d["status"] == "failed"
        assert d["error"] == "server restarted while the job was running"
        evs = c.get(f"/api/jobs/{j_running}/events").json()["events"]
        assert [e["seq"] for e in evs] == list(range(len(evs)))
        assert evs[-1]["status"] == "failed"
        assert wait(c, j_pending)["status"] == "done"
        assert c.get("/api/health").status_code == 200


# ---------------------------------------------------------------------------
# Malformed uploads and NEB mismatches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,content,fmt", [
    ("POSCAR", b"Cu\n1.0\n3.6 0 0\n0 3.6 0\n0 0 3.6\nCu\n4\nDirect\n0 0 0\n", None),  # short
    ("bad.xyz", b"3\ncomment\nCu 0 0 0\nCu 1 1\n", None),
    ("empty.cif", b"", None),
    ("binary.traj", os.urandom(512), None),
    ("x.vasp", b"not a poscar at all", "vasp"),
    ("x.data", b"1 2 3", "nonexistent-format"),
], ids=["truncated_poscar", "bad_xyz", "empty", "random_bytes", "forced_vasp", "bad_format"])
def test_malformed_upload(client, name, content, fmt):
    data = {"format": fmt} if fmt else {}
    r = client.post("/api/uploads", headers=H, data=data, files=[("files", (name, content))])
    assert r.status_code == 200, r.text
    (u,) = r.json()["uploads"]
    assert u["ok"] is False and u["id"] is None and u["error"]
    assert client.get("/api/health").status_code == 200


def test_too_many_files(client):
    files = [("files", (f"f{i}.xyz", b"1\n\nCu 0 0 0\n")) for i in range(51)]
    r = client.post("/api/uploads", headers=H, files=files)
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"


def test_failed_upload_id_not_submittable(client):
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": "singlepoint", "calculator": EMT_CPU, "params": {},
        "structure_ids": ["zzzz"]})
    assert r.status_code in (404, 422)


def _neb_body(ui, uf, calc="neb"):
    return {"calc_type": calc, "calculator": EMT_CPU, "params": {},
            "initial_id": ui, "final_id": uf}


@pytest.mark.parametrize("calc", ["neb", "autoneb"])
def test_neb_mismatched_atoms(client, tmp_path, calc):
    a = upload(client, tmp_path, cu(), name="a.vasp")
    b = upload(client, tmp_path, cu().repeat((2, 1, 1)), name="b.vasp")
    r = client.post("/api/jobs", headers=H, json=_neb_body(a["id"], b["id"], calc))
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error"
    assert any(d["field"] == "inputs.final" for d in err["details"])


@pytest.mark.parametrize("calc", ["neb", "autoneb"])
def test_neb_mismatched_symbols(client, tmp_path, calc):
    a_atoms = cu()
    b_atoms = a_atoms.copy()
    b_atoms[0].symbol = "Ni"
    a = upload(client, tmp_path, a_atoms, name="a.xyz", fmt="extxyz")
    b = upload(client, tmp_path, b_atoms, name="b.xyz", fmt="extxyz")
    r = client.post("/api/jobs", headers=H, json=_neb_body(a["id"], b["id"], calc))
    assert r.status_code == 422
    assert any(d["field"] == "inputs.final" for d in r.json()["error"]["details"])
    # symbol ORDER matters too
    c_atoms = b_atoms.copy()
    c_atoms.symbols = list(reversed(c_atoms.get_chemical_symbols()))
    c = upload(client, tmp_path, c_atoms, name="c.xyz", fmt="extxyz")
    r = client.post("/api/jobs", headers=H, json=_neb_body(b["id"], c["id"], calc))
    assert r.status_code == 422


def test_neb_band_bad_frames(client, tmp_path):
    two = [cu(), cu()]
    u2 = upload(client, tmp_path, two, name="two.xyz", fmt="extxyz")
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": "neb", "calculator": EMT_CPU, "params": {}, "band_id": u2["id"]})
    assert r.status_code == 422
    mixed = [cu(), cu(), cu()]
    mixed[1][0].symbol = "Ni"
    um = upload(client, tmp_path, mixed, name="mixed.xyz", fmt="extxyz")
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": "neb", "calculator": EMT_CPU, "params": {}, "band_id": um["id"]})
    assert r.status_code == 422
    # both band_id and pair ids -> 422
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": "neb", "calculator": EMT_CPU, "params": {}, "band_id": um["id"],
        "initial_id": um["id"], "final_id": um["id"]})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Error shapes, security, traversal, replay
# ---------------------------------------------------------------------------

def _assert_error_shape(r, status, code):
    assert r.status_code == status, r.text
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert set(body) == {"error"}
    e = body["error"]
    assert e["code"] == code and isinstance(e["message"], str)
    assert isinstance(e["details"], list)
    if code == "validation_error":
        assert e["details"]
        for d in e["details"]:
            assert set(d) >= {"field", "message"}
    return e


def test_validation_422_shapes(client, tmp_path):
    u = upload(client, tmp_path, cu())
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": "md", "calculator": {"name": "emt", "device": "cpu"},
        "params": {"ensemble": "nve", "temperature": 300, "steps": -1},
        "structure_ids": [u["id"]]})
    e = _assert_error_shape(r, 422, "validation_error")
    fields = {d["field"] for d in e["details"]}
    assert {"params.temperature", "params.steps"} <= fields
    r = client.post("/api/jobs", headers=H, json={"calc_type": "md"})
    _assert_error_shape(r, 422, "validation_error")
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": "singlepoint", "calculator": EMT_CPU, "params": {},
        "structure_ids": [u["id"]], "structure_every": -1})
    _assert_error_shape(r, 422, "validation_error")
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": "singlepoint", "calculator": EMT_CPU, "params": {},
        "structure_ids": [u["id"]], "label": "x" * 81})
    _assert_error_shape(r, 422, "validation_error")
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": "singlepoint", "calculator": {"name": "no-such-model"}, "params": {},
        "structure_ids": [u["id"]]})
    e = _assert_error_shape(r, 422, "validation_error")
    assert any(d["field"] == "calculator.name" for d in e["details"])
    r = client.get("/api/jobs/x/events", params={"since": -1})
    assert r.status_code in (404, 422)
    _assert_error_shape(client.get("/api/nope"), 404, "not_found")


def test_xmliprun_header_enforced(client, tmp_path):
    u = upload(client, tmp_path, cu())
    for kw in ({}, {"headers": {"X-Mliprun": "0"}}):
        _assert_error_shape(client.post("/api/uploads",
                                        files=[("files", ("a.xyz", b"1\n\nCu 0 0 0\n"))], **kw),
                            403, "forbidden")
        _assert_error_shape(client.post("/api/jobs", json={
            "calc_type": "singlepoint", "calculator": EMT_CPU, "params": {},
            "structure_ids": [u["id"]]}, **kw), 403, "forbidden")
    assert client.get("/api/jobs").json()["jobs"] == []
    (jid,) = submit(client, [u["id"]])
    _assert_error_shape(client.post(f"/api/jobs/{jid}/cancel"), 403, "forbidden")
    r = client.post("/api/uploads", headers=H, files=[("files", ("a.xyz", b"1\n\nCu 0 0 0\n"))])
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
    wait(client, jid)


def test_host_allow_list(runs):
    with make_client(runs, allowed_hosts=["127.0.0.1:8765"]) as c:
        r = c.get("/api/health", headers={"Host": "evil.example:8765"})
        _assert_error_shape(r, 403, "forbidden")
        assert c.get("/api/health", headers={"Host": "127.0.0.1:8765"}).status_code == 200


def test_ws_foreign_origin_rejected(client):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as ei:
        with client.websocket_connect("/ws", headers={"Origin": "http://evil.example"}) as ws:
            ws.send_json({"action": "subscribe", "job_id": "*"})
            ws.receive_json()
    assert ei.value.code == 1008


def test_path_traversal(client, tmp_path):
    u = upload(client, tmp_path, cu())
    (jid,) = submit(client, [u["id"]])
    wait(client, jid)
    for bad in ("..%2F..%2F..%2Fetc%2Fpasswd", "%2E%2E/%2E%2E/state.json",
                "inputs/..%2F..%2F..%2F..%2Fetc%2Fpasswd"):
        r = client.get(f"/api/jobs/{jid}/files/{bad}")
        assert r.status_code in (400, 404), (bad, r.status_code)
        assert "root:" not in r.text
    r = client.get(f"/api/jobs/{jid}/files/..%2F..%2Fsomething")
    _assert_error_shape(r, 400, "bad_request")
    # absolute path
    r = client.get(f"/api/jobs/{jid}/files/%2Fetc%2Fpasswd")
    assert r.status_code in (400, 404) and "root:" not in r.text
    # a symlink pointing outside run_dir
    run_dir = Path(client.get(f"/api/jobs/{jid}").json()["run_dir"])
    (run_dir / "link").symlink_to("/etc/passwd")
    r = client.get(f"/api/jobs/{jid}/files/link")
    assert r.status_code == 400 and "root:" not in r.text
    # legitimate download, directory, inline
    assert client.get(f"/api/jobs/{jid}/files/summary.json").status_code == 200
    _assert_error_shape(client.get(f"/api/jobs/{jid}/files/inputs"), 404, "not_found")
    r = client.get(f"/api/jobs/{jid}/files/reproduce.sh", params={"inline": 1})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")


def test_zip_download(client, tmp_path):
    import io
    import zipfile
    u = upload(client, tmp_path, cu())
    (jid,) = submit(client, [u["id"]])
    wait(client, jid)
    r = client.get(f"/api/jobs/{jid}/zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert f'filename="{jid}.zip"' in r.headers["content-disposition"]
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert f"{jid}/summary.json" in names and all(n.startswith(f"{jid}/") for n in names)


def test_ws_replay_since0_equals_get_events(client, tmp_path):
    u = upload(client, tmp_path, cu(2))
    (jid,) = submit(client, [u["id"]], "optimize")
    wait(client, jid)
    expected = client.get(f"/api/jobs/{jid}/events").json()["events"]
    for since in (0, 5):
        got = []
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"action": "subscribe", "job_id": jid, "since": since})
            while True:
                m = ws.receive_json()
                if m.get("type") == "ws_ack":
                    assert m["replayed"] == len(got)
                    break
                got.append(m)
        assert got == expected[since:]
    page = client.get(f"/api/jobs/{jid}/events", params={"since": 3, "limit": 2}).json()
    assert [e["seq"] for e in page["events"]] == [3, 4]
    assert page["next_seq"] == 5 and page["terminal"] is False


# ---------------------------------------------------------------------------
# Misc contract details
# ---------------------------------------------------------------------------

def test_filename_sanitized_and_stored(client, runs, tmp_path):
    p = tmp_path / "x.xyz"
    ase.io.write(str(p), cu(), format="extxyz")
    r = client.post("/api/uploads", headers=H,
                    files=[("files", ("../../.my file (1).xyz", p.read_bytes()))])
    (u,) = r.json()["uploads"]
    assert u["ok"], u
    assert u["filename"] == "my_file__1_.xyz"
    stored = runs / "_uploads" / u["id"]
    assert (stored / "my_file__1_.xyz").read_bytes() == p.read_bytes()
    meta = json.loads((stored / "meta.json").read_text())
    assert meta["format"] == "extxyz" and "xyz" not in meta


def test_events_limit_bounds(client, tmp_path):
    u = upload(client, tmp_path, cu())
    (jid,) = submit(client, [u["id"]])
    wait(client, jid)
    for lim in (0, 5001):
        _assert_error_shape(client.get(f"/api/jobs/{jid}/events", params={"limit": lim}),
                            422, "validation_error")
    body = client.get(f"/api/jobs/{jid}/events", params={"since": 10 ** 6}).json()
    assert body["events"] == [] and body["next_seq"] == 10 ** 6


def test_structure_every_zero_sends_no_frames(client, tmp_path):
    u = upload(client, tmp_path, cu(2))
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": "optimize", "calculator": EMT_CPU, "params": {},
        "structure_ids": [u["id"]], "structure_every": 0})
    jid = r.json()["jobs"][0]["job_id"]
    d = wait(client, jid)
    assert d["status"] == "done" and d["structure_every"] == 0
    evs = client.get(f"/api/jobs/{jid}/events").json()["events"]
    steps = [e for e in evs if e["type"] == "opt_step"]
    assert steps and not any("frame" in e for e in steps)
    assert "final_frame" in d["result"]


def test_pair_label_and_job_json(client, tmp_path):
    from ase.build import add_adsorbate, fcc100
    slab = fcc100("Cu", (2, 2, 2), vacuum=5.0)
    add_adsorbate(slab, "Cu", 1.7, "hollow")
    fin = slab.copy()
    fin.positions[-1, 0] += 2.55
    a = upload(client, tmp_path, slab, name="ini.vasp")
    b = upload(client, tmp_path, fin, name="fin.vasp")
    r =client.post("/api/jobs", headers=H, json=_neb_body(a["id"], b["id"]))
    assert r.status_code == 201, r.text
    j = r.json()["jobs"][0]
    assert j["label"] == "ini.vasp -> fin.vasp"
    import re
    assert re.match(r"^[0-9]{8}-[0-9]{6}-neb-[0-9a-f]{6}$", j["job_id"])
    d = client.get(f"/api/jobs/{j['job_id']}").json()
    job = json.loads((Path(d["run_dir"]) / "job.json").read_text())
    assert job["schema_version"] == 1 and set(job["inputs"]) == {"initial", "final"}
    assert job["inputs"]["initial"]["path"].startswith("inputs/initial/")
    assert job["calculator"] == {"name": "emt", "device": "cpu", "uma_task": None,
                                 "mace_head": None, "sevennet_task": None}
    assert all(v == "default" for v in job["param_sources"].values())
    client.post(f"/api/jobs/{j['job_id']}/cancel", headers=H)
    wait(client, j["job_id"], timeout=30)


def test_ws_no_duplicate_with_star_and_job_subscription(client, tmp_path):
    u = upload(client, tmp_path, cu())
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "subscribe", "job_id": "*"})
        assert ws.receive_json()["type"] == "ws_ack"
        (jid,) = submit(client, [u["id"]])
        ws.send_json({"action": "subscribe", "job_id": jid, "since": 0})
        seen = []
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            m = ws.receive_json()
            if m.get("job_id") != jid or "seq" not in m:
                continue
            seen.append(m["seq"])
            if m["type"] == "job_status" and m["status"] in TERMINAL:
                break
    assert seen == list(range(len(seen))), seen


def test_submit_limits(client, tmp_path):
    u = upload(client, tmp_path, cu())
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": "singlepoint", "calculator": EMT_CPU, "params": {},
        "structure_ids": [u["id"]] * 201})
    _assert_error_shape(r, 422, "validation_error")
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": "singlepoint", "calculator": EMT_CPU, "params": {},
        "structure_ids": []})
    _assert_error_shape(r, 422, "validation_error")
    assert client.get("/api/jobs").json()["jobs"] == []
