"""Remote jobs end to end through the fake ssh (REMOTE_PLAN section 10).

The "remote machine" is a temp directory on this computer; the fake ssh runs
the agent command locally. Long-running behaviour (live events, cancel,
outages, restart, queues) uses the scripted fake worker through the agent's
``MLIPRUN_AGENT_WORKER_CMD`` hook; the one scientific check (remote EMT
energy equals a local run) uses the real worker.
"""
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pytest
from ase.build import bulk
from remote_helpers import read_jsonl, wait_until

H = {"X-Mliprun": "1"}
FAKE_WORKER = Path(__file__).resolve().parent / "fake_worker.py"


@pytest.fixture
def fake_remote_worker(monkeypatch):
    monkeypatch.setenv("MLIPRUN_AGENT_WORKER_CMD",
                       json.dumps([sys.executable, str(FAKE_WORKER)]))


@pytest.fixture
def cu(upload_atoms):
    """``cu(client) -> upload id`` of a rattled Cu cell."""
    def make(client, seed=1):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
        atoms.rattle(0.05, seed=seed)
        return upload_atoms(client, atoms, "POSCAR", fmt="vasp")["id"]
    return make


def submit(client, up_id, target=None, label="steps=3", folder=None, **extra):
    body = {"calc_type": "optimize", "structure_ids": [up_id], "label": label,
            "calculator": {"name": "emt", "device": "cpu"}, "params": {}}
    if target:
        body["target"] = target
    if folder:
        body["folder"] = folder
    body.update(extra)
    r = client.post("/api/jobs", headers=H, json=body)
    assert r.status_code == 201, r.text
    return r.json()["jobs"][0]


def events_of(client, job_id):
    return client.get(f"/api/jobs/{job_id}/events?since=0&limit=5000").json()["events"]


def n_steps(client, job_id):
    return sum(1 for e in events_of(client, job_id) if e["type"] == "opt_step")


def detail(client, job_id):
    return client.get(f"/api/jobs/{job_id}").json()


def mirror_equals_remote(job):
    """The local mirror is the remote file plus the server's one final job_status line."""
    local = (Path(job["run_dir"]) / "events.jsonl").read_bytes()
    remote = (Path(job["remote_run_dir"]) / "events.jsonl").read_bytes()
    if not local.startswith(remote):
        return False
    tail = local[len(remote):].splitlines()
    return len(tail) == 1 and json.loads(tail[0])["type"] == "job_status" and \
        json.loads(tail[0])["status"] in ("done", "failed", "cancelled")


def when_failed(client, job_id, timeout=30):
    return wait_until(lambda: (lambda x: x if x["status"] == "failed" else None)(
        detail(client, job_id)), timeout, what="status failed")


# --------------------------------------------------------------------------- #
def test_remote_emt_optimize_equals_local_run(make_client, remote_setup, cu, wait_for_status):
    from mliprun.webapp.jobs import default_worker_cmd

    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True,
                    worker_cmd=default_worker_cmd())
    up = cu(c)
    params = {"fmax": 0.01}
    local = submit(c, up, label="local", folder="loc-run", params=params)
    remote = submit(c, up, target=info["env_id"], label="remote", folder="rem-run",
                    params=params)
    ld = wait_for_status(c, local["job_id"], "done", timeout=180)
    rd = wait_for_status(c, remote["job_id"], "done", timeout=180)
    assert rd["result_brief"]["converged"] is True
    assert rd["result_brief"]["final_energy_eV"] == pytest.approx(
        ld["result_brief"]["final_energy_eV"], abs=1e-8)
    assert rd["result_brief"]["steps"] == ld["result_brief"]["steps"] > 0
    assert rd["target"]["kind"] == "remote" and rd["remote_error"] is None
    assert rd["returncode"] == 0
    # the mirror is byte-identical to the remote file; seq stayed the line index
    assert mirror_equals_remote(rd)
    events = events_of(c, remote["job_id"])
    assert [e["seq"] for e in events] == list(range(len(events)))
    statuses = [e["status"] for e in events if e["type"] == "job_status"]
    assert statuses == ["pending", "running", "done"]
    assert events[0]["type"] == "job_status" and events[-1]["status"] == "done"
    # small files were copied back; remote_files.json says so
    run_dir = Path(rd["run_dir"])
    assert (run_dir / "summary.json").is_file() and (run_dir / "worker.log").is_file()
    listing = json.loads((run_dir / "remote_files.json").read_text())
    assert listing["remote_run_dir"] == rd["remote_run_dir"] and listing["remote_host"] == "box"
    assert all(f["copied"] for f in listing["files"])
    assert any(f["path"] == "mliprun_run.json" for f in listing["files"])


def test_live_events_arrive_while_running_and_traffic_is_small(
        make_client, remote_setup, cu, fake_remote_worker, wait_for_status, monkeypatch,
        tmp_path):
    log = tmp_path / "ssh.log"
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    job = submit(c, up, target=info["env_id"], label="steps=40;sleep=0.1", folder="live")
    jid = job["job_id"]
    wait_until(lambda: n_steps(c, jid) >= 3, 30, what="live opt_step events")
    d = detail(c, jid)
    assert d["status"] == "running"                       # still going: events were live
    assert d["progress"]["step"] is not None
    assert 3 <= n_steps(c, jid) < 40
    # the WebSocket delivers them too
    with c.websocket_connect("/ws") as ws:
        ws.send_json({"action": "subscribe", "job_id": jid, "since": 0})
        types = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            msg = ws.receive_json()
            types.append(msg.get("type"))
            if msg.get("type") == "result":
                break
        assert "ws_ack" in types and "opt_step" in types and types[-1] == "result"
    wait_for_status(c, jid, "done", timeout=60)
    assert n_steps(c, jid) == 40
    # traffic: one short request per poll; every byte of events travels once
    entries = read_jsonl(log)
    polls = [e for e in entries if " poll " in e["command"]]
    total = (Path(job["run_dir"]) / "events.jsonl").stat().st_size
    out_bytes = sum(e["bytes_out"] for e in polls)
    assert len(polls) >= 5
    assert total <= out_bytes <= total + 200 * len(polls)  # events once + small JSON headers
    assert all(e["rc"] == 0 for e in polls)


def test_cancel_running_remote_job(make_client, remote_setup, cu, fake_remote_worker,
                                   wait_for_status):
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    job = submit(c, up, target=info["env_id"], label="steps=800;sleep=0.05", folder="cx")
    jid = job["job_id"]
    wait_until(lambda: n_steps(c, jid) >= 2, 30, what="events")
    r = c.post(f"/api/jobs/{jid}/cancel", headers=H)
    assert r.status_code == 200 and r.json()["job"]["cancel_requested"] is True
    d = wait_for_status(c, jid, "cancelled", timeout=60)
    remote_dir = Path(job["remote_run_dir"])
    assert json.loads((remote_dir / "worker_exit.json").read_text())["returncode"] == 3
    assert json.loads((Path(job["run_dir"]) / "summary.json").read_text())["status"] == \
        "cancelled"
    assert n_steps(c, jid) < 800
    assert mirror_equals_remote(d)
    assert c.post(f"/api/jobs/{jid}/cancel", headers=H).status_code == 409


def test_cancel_escalates_to_sigkill_when_sigterm_is_ignored(
        make_client, remote_setup, cu, fake_remote_worker, wait_for_status):
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True,
                    cancel_grace_s=0.5)
    up = cu(c)
    job = submit(c, up, target=info["env_id"],
                 label="steps=2000;sleep=0.05;ignore_term=1", folder="kx")
    jid = job["job_id"]
    wait_until(lambda: n_steps(c, jid) >= 2, 30, what="events")
    c.post(f"/api/jobs/{jid}/cancel", headers=H)
    wait_for_status(c, jid, "cancelled", timeout=60)
    exit_info = json.loads((Path(job["remote_run_dir"]) / "worker_exit.json").read_text())
    assert exit_info["returncode"] == -9


def test_unreachable_mid_run_then_recovery_does_not_fail_the_job(
        make_client, remote_setup, cu, fake_remote_worker, wait_for_status, monkeypatch,
        tmp_path):
    down = tmp_path / "ssh-down"
    monkeypatch.setenv("FAKE_SSH_DOWN", str(down))
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    job = submit(c, up, target=info["env_id"], label="steps=40;sleep=0.15", folder="flaky")
    jid = job["job_id"]
    wait_until(lambda: n_steps(c, jid) >= 2, 30, what="events")
    down.write_text("")
    d = wait_until(lambda: (lambda x: x if x["remote_error"] else None)(detail(c, jid)), 30,
                   what="remote_error to appear")
    assert d["status"] == "running"
    assert "Connection refused" in d["remote_error"] and "(since " in d["remote_error"]
    assert json.loads((Path(job["run_dir"]) / "state.json").read_text())["remote"]["error"]
    time.sleep(1.5)
    assert detail(c, jid)["status"] == "running"          # an outage never fails the job
    down.unlink()
    d = wait_for_status(c, jid, "done", timeout=90)
    assert d["remote_error"] is None
    assert n_steps(c, jid) == 40
    assert mirror_equals_remote(d)
    events = events_of(c, jid)
    assert [e["seq"] for e in events] == list(range(len(events)))


def test_cancel_during_an_outage_is_delivered_on_reconnect(
        make_client, remote_setup, cu, fake_remote_worker, wait_for_status, monkeypatch,
        tmp_path):
    down = tmp_path / "ssh-down"
    monkeypatch.setenv("FAKE_SSH_DOWN", str(down))
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    job = submit(c, up, target=info["env_id"], label="steps=3000;sleep=0.05", folder="late")
    jid = job["job_id"]
    wait_until(lambda: n_steps(c, jid) >= 2, 30, what="events")
    down.write_text("")
    wait_until(lambda: detail(c, jid)["remote_error"], 30, what="outage noticed")
    r = c.post(f"/api/jobs/{jid}/cancel", headers=H)
    assert r.status_code == 200
    time.sleep(1.0)
    assert detail(c, jid)["status"] == "running"          # remembered, not yet deliverable
    assert not (Path(job["remote_run_dir"]) / "worker_exit.json").exists()
    down.unlink()
    wait_for_status(c, jid, "cancelled", timeout=60)
    assert json.loads((Path(job["remote_run_dir"]) / "worker_exit.json").read_text()
                      )["returncode"] == 3


def test_server_restart_reattaches_and_shutdown_does_not_cancel(
        make_app, remote_setup, cu, fake_remote_worker, wait_for_status, runs_dir):
    from fastapi.testclient import TestClient
    from remote_helpers import FAKE_SSH_CMD

    info = remote_setup()
    kw = dict(machines_file=info["machines_file"], ssh_command=FAKE_SSH_CMD,
              remote_backoff_base=0.05)
    with TestClient(make_app(**kw)) as c1:
        up = cu(c1)
        job = submit(c1, up, target=info["env_id"], label="steps=40;sleep=0.15",
                     folder="survivor")
        jid = job["job_id"]
        wait_until(lambda: n_steps(c1, jid) >= 3, 30, what="events")
        before = n_steps(c1, jid)
    # app 1 is stopped: the job is still "running" on disk and the remote worker lives
    state = json.loads((runs_dir / "survivor" / "state.json").read_text())
    assert state["status"] == "running" and state["remote"]["launched"] is True
    remote_dir = Path(job["remote_run_dir"])
    worker_pid = json.loads((remote_dir / "remote_pids.json").read_text())["worker_pid"]
    os.kill(worker_pid, 0)                                 # still alive
    assert not (remote_dir / "worker_exit.json").exists()
    with TestClient(make_app(**kw)) as c2:
        d = detail(c2, jid)
        assert d["status"] == "running"                    # reattached, not failed
        assert d["error"] is None
        d = wait_for_status(c2, jid, "done", timeout=90)
        assert n_steps(c2, jid) == 40 and n_steps(c2, jid) >= before
        assert mirror_equals_remote(d)
        events = events_of(c2, jid)
        assert [e["seq"] for e in events] == list(range(len(events)))
        statuses = [e["status"] for e in events if e["type"] == "job_status"]
        assert statuses == ["pending", "running", "done"]
        assert c2.get("/api/health").json()["running_jobs"] == []


def test_local_and_remote_run_at_the_same_time_and_each_queue_is_serial(
        make_client, remote_setup, cu, fake_remote_worker, wait_for_status):
    a = remote_setup(name="alpha", host="alpha")
    b = remote_setup(name="beta", host="beta", store_path=a["machines_file"])
    c = make_client(machines_file=a["machines_file"], start_runner=True)
    up = cu(c)
    script = "steps=30;sleep=0.1"
    loc = submit(c, up, label=script, folder="l1")
    a1 = submit(c, up, target=a["env_id"], label=script, folder="a1")
    a2 = submit(c, up, target=a["env_id"], label=script, folder="a2")
    b1 = submit(c, up, target=b["env_id"], label=script, folder="b1")

    def all_three_running():
        s = {j["job_id"]: j["status"] for j in c.get("/api/jobs").json()["jobs"]}
        return (s[loc["job_id"]], s[a1["job_id"]], s[b1["job_id"]]) == ("running",) * 3

    wait_until(all_three_running, 60, what="local, alpha and beta running together")
    mid = detail(c, a2["job_id"])
    assert mid["status"] == "pending" and mid["queue_position"] == 1   # alpha is serial
    health = c.get("/api/health").json()
    assert sorted(health["running_jobs"]) == sorted([loc["job_id"], a1["job_id"], b1["job_id"]])
    assert health["running_job"] == loc["job_id"]          # compatibility: the local one
    for j in (loc, a1, a2, b1):
        wait_for_status(c, j["job_id"], "done", timeout=120)

    def span(job):
        rec = read_jsonl(Path(job["run_dir"]) / "fake_times.jsonl")[0]
        return rec["start"], rec["end"]

    (a1s, a1e), (a2s, _) = span(a1), span(a2)
    assert a2s >= a1e                                      # same machine: strictly one at a time
    (ls, le), (b1s, b1e) = span(loc), span(b1)
    assert max(ls, a1s, b1s) < min(le, a1e, b1e)           # different queues overlapped


def test_copy_back_threshold_makes_big_files_remote_only(
        make_client, remote_setup, cu, monkeypatch, tmp_path, wait_for_status):
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "import runpy, sys\n"
        "code = 0\n"
        "try:\n"
        f"    runpy.run_path({str(FAKE_WORKER)!r}, run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    code = e.code or 0\n"
        "d = sys.argv[-1]\n"
        "open(d + '/big.bin', 'wb').write(bytes(range(256)) * 20)\n"
        "open(d + '/small.txt', 'w').write('tiny')\n"
        "sys.exit(code)\n")
    monkeypatch.setenv("MLIPRUN_AGENT_WORKER_CMD", json.dumps([sys.executable, str(wrapper)]))
    info = remote_setup(copy_back_max_mb=0.002)            # 2097 bytes
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    job = submit(c, up, target=info["env_id"], label="steps=2", folder="cb")
    d = wait_for_status(c, job["job_id"], "done", timeout=60)
    run_dir, remote_dir = Path(d["run_dir"]), Path(d["remote_run_dir"])
    assert (run_dir / "small.txt").read_text() == "tiny"
    assert not (run_dir / "big.bin").exists() and (remote_dir / "big.bin").stat().st_size == 5120
    listing = json.loads((run_dir / "remote_files.json").read_text())
    by = {f["path"]: f for f in listing["files"]}
    assert by["big.bin"] == {"path": "big.bin", "size": 5120, "copied": False}
    assert by["small.txt"]["copied"] is True and by["events.jsonl"]["copied"] is True
    files = c.get(f"/api/jobs/{job['job_id']}/files").json()["files"]
    big = next(f for f in files if f["path"] == "big.bin")
    assert big == {"path": "big.bin", "size": 5120, "mtime": None, "remote_only": True}
    assert not any(f.get("remote_only") for f in files if f["path"] != "big.bin")
    assert [f["path"] for f in files] == sorted(f["path"] for f in files)
    r = c.get(f"/api/jobs/{job['job_id']}/files/big.bin")
    assert r.status_code == 200 and r.content == bytes(range(256)) * 20
    assert "attachment" in r.headers["content-disposition"]
    assert not (run_dir / "big.bin").exists()              # streamed, nothing stored
    assert c.get(f"/api/jobs/{job['job_id']}/files/small.txt").content == b"tiny"
    assert c.get(f"/api/jobs/{job['job_id']}/files/nope.bin").status_code == 404
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{job['job_id']}/zip").content))
    names = z.namelist()
    assert "cb/remote_files.json" in names and "cb/small.txt" in names
    assert "cb/big.bin" not in names


def test_copy_back_zero_leaves_everything_but_local_masters_on_the_remote(
        make_client, remote_setup, cu, fake_remote_worker, wait_for_status, monkeypatch,
        tmp_path):
    info = remote_setup(copy_back_max_mb=0)
    down = tmp_path / "ssh-down"
    monkeypatch.setenv("FAKE_SSH_DOWN", str(down))
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    job = submit(c, up, target=info["env_id"], label="steps=3", folder="zero")
    d = wait_for_status(c, job["job_id"], "done", timeout=60)   # no summary.json copied: still done
    run_dir, remote_dir = Path(d["run_dir"]), Path(d["remote_run_dir"])
    listing = json.loads((run_dir / "remote_files.json").read_text())["files"]
    remote_only = {f["path"] for f in listing if not f["copied"]}
    assert {"worker.log", "reproduce.json", "fake_times.jsonl", "worker_exit.json",
            "remote_pids.json", "supervisor.log"} <= remote_only
    assert "events.jsonl" not in remote_only and "job.json" not in remote_only
    assert not (run_dir / "worker.log").exists()
    r = c.get(f"/api/jobs/{job['job_id']}/files/worker.log")
    assert r.status_code == 200 and r.content == (remote_dir / "worker.log").read_bytes()
    r = c.get(f"/api/jobs/{job['job_id']}/files/reproduce.json?inline=1")
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text == (remote_dir / "reproduce.json").read_text()
    # the backend summary written locally is marked as such, status done
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "done" and summary["written_by"] == "server"
    # remote machine unreachable: 502, not a hang or a 500
    down.write_text("")
    r = c.get(f"/api/jobs/{job['job_id']}/files/worker.log")
    assert r.status_code == 502 and r.json()["error"]["code"] == "remote_unreachable"
    assert "Connection refused" in r.json()["error"]["message"]
    r = c.get(f"/api/jobs/{job['job_id']}/files/worker.log?inline=1")
    assert r.status_code == 502
    down.unlink()
    (remote_dir / "worker.log").unlink()
    assert c.get(f"/api/jobs/{job['job_id']}/files/worker.log").status_code == 404


def test_launch_failure_fails_the_job_with_the_message(make_client, remote_setup, cu,
                                                       monkeypatch):
    monkeypatch.setenv("MLIPRUN_AGENT_WORKER_CMD", json.dumps(["/nonexistent/worker"]))
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    job = submit(c, up, target=info["env_id"], folder="nolaunch")
    d = when_failed(c, job["job_id"])
    assert d["error"].startswith("could not start the remote worker:")
    assert "/nonexistent/worker" in d["error"]
    events = events_of(c, job["job_id"])
    assert [e["status"] for e in events if e["type"] == "job_status"] == \
        ["pending", "running", "failed"]
    assert json.loads((Path(d["run_dir"]) / "summary.json").read_text())["status"] == "failed"


def test_broken_python_path_is_rejected_at_submission(make_client, remote_setup, cu, runs_dir):
    info = remote_setup()
    data = json.loads(info["machines_file"].read_text())
    data["machines"][0]["envs"][0]["python"] = "/nonexistent/bin/python"
    info["machines_file"].write_text(json.dumps(data))
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    r = c.post("/api/jobs", headers=H, json={
        "calc_type": "optimize", "structure_ids": [up], "target": info["env_id"],
        "calculator": {"name": "emt"}, "folder": "nopy"})
    assert r.status_code == 422
    assert "/nonexistent/bin/python" in r.json()["error"]["details"][0]["message"]
    assert not (runs_dir / "nopy").exists() and not (info["location"] / "nopy").exists()


def test_remote_worker_killed_externally_is_a_failed_job(
        make_client, remote_setup, cu, fake_remote_worker):
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    job = submit(c, up, target=info["env_id"], label="steps=3000;sleep=0.05", folder="murder")
    jid = job["job_id"]
    wait_until(lambda: n_steps(c, jid) >= 2, 30, what="events")
    pids = json.loads((Path(job["remote_run_dir"]) / "remote_pids.json").read_text())
    os.killpg(pids["worker_pid"], 9)
    d = when_failed(c, jid)
    assert d["error"] == "worker killed by signal 9" and d["returncode"] == -9


def test_remote_worker_crash_without_summary_is_failed(
        make_client, remote_setup, cu, fake_remote_worker, wait_for_status):
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    job = submit(c, up, target=info["env_id"], label="steps=3;crash=1", folder="crash")
    d = wait_for_status(c, job["job_id"], "failed", timeout=60)
    assert d["returncode"] == 7 and d["error"] == "worker exited with code 7"


def test_vanished_worker_and_supervisor_is_reported(make_client, remote_setup, cu,
                                                    fake_remote_worker):
    """Supervisor and worker both gone, no exit file: 'remote worker disappeared'."""
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    job = submit(c, up, target=info["env_id"], label="steps=3000;sleep=0.05", folder="vanish")
    jid = job["job_id"]
    wait_until(lambda: n_steps(c, jid) >= 2, 30, what="events")
    remote_dir = Path(job["remote_run_dir"])
    pids = json.loads((remote_dir / "remote_pids.json").read_text())
    os.kill(pids["supervisor_pid"], 9)                     # no exit file will ever be written
    os.killpg(pids["worker_pid"], 9)
    d = when_failed(c, jid)
    assert d["error"] == "remote worker disappeared" and d["returncode"] is None
    assert not (remote_dir / "worker_exit.json").exists()
