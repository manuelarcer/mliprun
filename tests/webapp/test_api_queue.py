"""Queue runner: serial FIFO execution, cancel, worker crash, restart recovery."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

H = {"X-Mliprun": "1"}


def _rd(runs_dir, job_id):
    """A job's run_dir from the job index (PLAN 15.1: folders follow the label)."""
    entry = json.loads((runs_dir / "_jobs" / f"{job_id}.json").read_text())
    return Path(entry["run_dir"])


def _times(runs_dir, job_id):
    lines = (_rd(runs_dir, job_id) / "fake_times.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines]


def test_jobs_run_strictly_in_series_fifo(client, cu_upload, submit, wait_for_status,
                                          runs_dir):
    r = submit(client, [cu_upload["id"]] * 3, label="steps=4;sleep=0.05;")
    ids = [j["job_id"] for j in r.json()["jobs"]]
    details = [wait_for_status(client, i, "done", timeout=30) for i in ids]
    for prev, nxt in zip(details, details[1:]):
        assert nxt["started_at"] >= prev["finished_at"]
    spans = [_times(runs_dir, i)[0] for i in ids]
    for prev, nxt in zip(spans, spans[1:]):
        assert nxt["start"] >= prev["end"]          # worker processes never overlap
    assert len({s["pid"] for s in spans}) == 3      # one subprocess per job


def test_worker_env_and_cwd(make_app, cu_upload, submit, wait_for_status, runs_dir, tmp_path):
    from fastapi.testclient import TestClient

    script = tmp_path / "envdump.py"
    script.write_text(
        "import json, os, sys\n"
        "rd = sys.argv[-1]\n"
        "json.dump({'cwd': os.getcwd(), 'mpl': os.environ.get('MPLBACKEND'),\n"
        "           'unbuf': os.environ.get('PYTHONUNBUFFERED'), 'arg': rd},\n"
        "          open(os.path.join(rd, 'env.json'), 'w'))\n"
        "print('hello from worker')\n"
        "sys.exit(0)\n")
    with TestClient(make_app(worker_cmd=[sys.executable, str(script)])) as c:
        up = c.post("/api/uploads", headers=H,
                    files=[("files", ("h.xyz", b"1\n\nH 0 0 0\n"))]).json()["uploads"][0]
        job_id = submit(c, [up["id"]]).json()["jobs"][0]["job_id"]
        d = wait_for_status(c, job_id, ("done", "failed"))
    run_dir = _rd(runs_dir, job_id).resolve()
    env = json.loads((run_dir / "env.json").read_text())
    assert env == {"cwd": str(run_dir), "mpl": "Agg", "unbuf": "1", "arg": str(run_dir)}
    assert "hello from worker" in (run_dir / "worker.log").read_text()
    # rc 0 without a done summary -> failed, and the server writes the summary
    assert d["status"] == "failed" and d["returncode"] == 0
    assert "without reporting a result" in d["error"]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["error"]["message"] == d["error"]


def test_cancel_pending_job_never_starts_a_worker(client, cu_upload, submit,
                                                  wait_for_status, runs_dir):
    ids = [j["job_id"] for j in
           submit(client, [cu_upload["id"]] * 2, label="steps=200;sleep=0.02;").json()["jobs"]]
    wait_for_status(client, ids[0], "running")
    r = client.post(f"/api/jobs/{ids[1]}/cancel", headers=H)
    assert r.status_code == 200
    assert r.json()["job"]["status"] == "cancelled"
    assert not (_rd(runs_dir, ids[1]) / "worker.log").exists()
    assert client.post(f"/api/jobs/{ids[1]}/cancel", headers=H).status_code == 409
    client.post(f"/api/jobs/{ids[0]}/cancel", headers=H)
    wait_for_status(client, ids[0], "cancelled")
    time.sleep(0.3)
    assert not (_rd(runs_dir, ids[1]) / "worker.log").exists()
    ev = client.get(f"/api/jobs/{ids[1]}/events").json()["events"]
    assert [(e["status"], e["reason"]) for e in ev] == [("pending", None),
                                                       ("cancelled", "cancelled by user")]


def test_cancel_running_job_sigterm(client, cu_upload, submit, wait_for_status, runs_dir):
    job_id = submit(client, [cu_upload["id"]],
                    label="steps=100000;sleep=0.01").json()["jobs"][0]["job_id"]
    wait_for_status(client, job_id, "running")
    _wait_worker_started(client, job_id)
    t0 = time.monotonic()
    r = client.post(f"/api/jobs/{job_id}/cancel", headers=H)
    assert r.status_code == 200
    assert r.json()["job"]["cancel_requested"] is True
    d = wait_for_status(client, job_id, "cancelled", timeout=10)
    assert time.monotonic() - t0 < 2.0             # well before the 2 s SIGKILL grace
    assert d["returncode"] == 3 and d["cancel_requested"] is True
    summary = json.loads((_rd(runs_dir, job_id) / "summary.json").read_text())
    assert summary["status"] == "cancelled"        # the worker's own summary is kept
    ev = client.get(f"/api/jobs/{job_id}/events").json()
    assert ev["terminal"] and ev["events"][-1]["status"] == "cancelled"
    assert any(e["type"] == "log" and e["message"] == "cancelled by user" for e in ev["events"])
    assert client.post(f"/api/jobs/{job_id}/cancel", headers=H).json()["error"]["code"] \
        == "conflict"


def test_cancel_escalates_to_sigkill_after_grace(make_app, submit, wait_for_status, runs_dir):
    from fastapi.testclient import TestClient

    with TestClient(make_app(cancel_grace_s=0.5)) as c:
        up = c.post("/api/uploads", headers=H,
                    files=[("files", ("h.xyz", b"1\n\nH 0 0 0\n"))]).json()["uploads"][0]
        job_id = submit(c, [up["id"]], label="steps=100000;sleep=0.01;ignore_term=1") \
            .json()["jobs"][0]["job_id"]
        wait_for_status(c, job_id, "running")
        _wait_worker_started(c, job_id)
        t0 = time.monotonic()
        c.post(f"/api/jobs/{job_id}/cancel", headers=H)
        d = wait_for_status(c, job_id, "cancelled", timeout=10)
        elapsed = time.monotonic() - t0
    assert 0.5 <= elapsed < 5
    assert d["returncode"] == -signal.SIGKILL
    summary = json.loads((_rd(runs_dir, job_id) / "summary.json").read_text())
    assert summary["status"] == "cancelled" and summary["written_by"] == "server"


@pytest.mark.parametrize("label, reason, rc", [
    ("crash=1", "worker exited with code 7", 7),
    ("steps=1;error=kaboom", "kaboom", 1),
    ("steps=1;exit=2", "worker exited with code 2", 2),
], ids=["crash", "error_event", "exit_code"])
def test_failing_worker_marks_job_failed_and_server_keeps_serving(
        client, cu_upload, submit, wait_for_status, runs_dir, label, reason, rc):
    bad, good = [j["job_id"] for j in
                 (submit(client, [cu_upload["id"]], label=label).json()["jobs"]
                  + submit(client, [cu_upload["id"]]).json()["jobs"])]
    d = wait_for_status(client, bad, "failed")
    assert d["error"] == reason and d["returncode"] == rc
    assert client.get("/api/health").json()["status"] == "ok"
    assert wait_for_status(client, good, "done")["status"] == "done"
    summary = json.loads((_rd(runs_dir, bad) / "summary.json").read_text())
    assert summary["status"] == "failed"
    ev = client.get(f"/api/jobs/{bad}/events").json()["events"]
    assert [e["seq"] for e in ev] == list(range(len(ev)))
    assert ev[-1]["type"] == "job_status" and ev[-1]["status"] == "failed"
    assert ev[-1]["reason"] == reason and ev[-1]["returncode"] == rc


def test_spawn_failure_marks_failed(make_app, submit, wait_for_status, tmp_path):
    from fastapi.testclient import TestClient

    with TestClient(make_app(worker_cmd=[str(tmp_path / "does-not-exist")])) as c:
        up = c.post("/api/uploads", headers=H,
                    files=[("files", ("h.xyz", b"1\n\nH 0 0 0\n"))]).json()["uploads"][0]
        job_id = submit(c, [up["id"]]).json()["jobs"][0]["job_id"]
        d = wait_for_status(c, job_id, "failed")
        assert d["error"].startswith("could not start the worker")
        assert c.get("/api/health").json()["running_job"] is None


def test_partial_final_line_is_truncated_before_status(client, cu_upload, submit,
                                                       wait_for_status, runs_dir):
    job_id = submit(client, [cu_upload["id"]], label="steps=2;partial=1") \
        .json()["jobs"][0]["job_id"]
    wait_for_status(client, job_id, "done")
    raw = (_rd(runs_dir, job_id) / "events.jsonl").read_text()
    assert raw.endswith("\n") and "partial" not in raw
    lines = [json.loads(x) for x in raw.splitlines()]
    assert [e["seq"] for e in lines] == list(range(len(lines)))


def test_shutdown_cancels_running_job(make_app, submit, wait_for_status, runs_dir):
    from fastapi.testclient import TestClient

    with TestClient(make_app()) as c:
        up = c.post("/api/uploads", headers=H,
                    files=[("files", ("h.xyz", b"1\n\nH 0 0 0\n"))]).json()["uploads"][0]
        ids = [j["job_id"] for j in
               submit(c, [up["id"]] * 2, label="steps=100000;sleep=0.01;").json()["jobs"]]
        wait_for_status(c, ids[0], "running")
        _wait_worker_started(c, ids[0])
    state0 = json.loads((_rd(runs_dir, ids[0]) / "state.json").read_text())
    state1 = json.loads((_rd(runs_dir, ids[1]) / "state.json").read_text())
    assert state0["status"] == "cancelled" and state0["returncode"] == 3
    assert state1["status"] == "pending"           # re-queued by the next server


def _wait_worker_started(client, job_id, timeout=10):
    """Wait until the worker has emitted an opt_step (its SIGTERM handler is installed)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["progress"] is not None:
            return
        time.sleep(0.02)
    raise AssertionError("worker did not start")


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_restart_recovery(make_app, submit, wait_for_status, runs_dir):
    from fastapi.testclient import TestClient

    with TestClient(make_app(start_runner=False)) as c:
        up = c.post("/api/uploads", headers=H,
                    files=[("files", ("h.xyz", b"1\n\nH 0 0 0\n"))]).json()["uploads"][0]
        ids = [j["job_id"] for j in submit(c, [up["id"]] * 3).json()["jobs"]]
    # Fake a job that was running when the server died: dead pid, partial last line.
    run0 = runs_dir / ids[0]
    st = json.loads((run0 / "state.json").read_text())
    st.update(status="running", pid=_dead_pid(), started_at=st["created_at"])
    (run0 / "state.json").write_text(json.dumps(st))
    with open(run0 / "events.jsonl", "a") as fh:
        fh.write(json.dumps({"job_id": ids[0], "seq": 1, "t": 0, "type": "job_status",
                             "status": "running", "reason": None, "returncode": None}) + "\n")
        fh.write('{"job_id": "half a li')
    # An unreadable job dir is ignored.
    bogus = runs_dir / "20260101-000000-optimize-000000"
    bogus.mkdir()
    (bogus / "state.json").write_text("{nope")

    with TestClient(make_app()) as c:
        d0 = wait_for_status(c, ids[0], "failed")
        assert d0["error"] == "server restarted while the job was running"
        ev = c.get(f"/api/jobs/{ids[0]}/events").json()["events"]
        assert [e["seq"] for e in ev] == [0, 1, 2]
        assert ev[-1]["status"] == "failed"
        # pending jobs re-queued in creation order and run
        d1 = wait_for_status(c, ids[1], "done")
        d2 = wait_for_status(c, ids[2], "done")
        assert d2["started_at"] >= d1["finished_at"]
        assert c.get("/api/jobs/20260101-000000-optimize-000000").status_code == 404
    summary = json.loads((run0 / "summary.json").read_text())
    assert summary["status"] == "failed"


def test_restart_kills_orphan_worker(make_app, submit, runs_dir):
    from fastapi.testclient import TestClient

    with TestClient(make_app(start_runner=False)) as c:
        up = c.post("/api/uploads", headers=H,
                    files=[("files", ("h.xyz", b"1\n\nH 0 0 0\n"))]).json()["uploads"][0]
        job_id = submit(c, [up["id"]]).json()["jobs"][0]["job_id"]
    run_dir = _rd(runs_dir, job_id).resolve()
    # A process whose cmdline looks like this job's worker.
    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                               "-m", "mliprun.webapp.worker", str(run_dir)],
                              start_new_session=True)
    try:
        st = json.loads((run_dir / "state.json").read_text())
        st.update(status="running", pid=orphan.pid)
        (run_dir / "state.json").write_text(json.dumps(st))
        app = make_app(start_runner=False, cancel_grace_s=5.0)
        assert orphan.wait(timeout=5) == -signal.SIGTERM
        with TestClient(app) as c:
            assert c.get(f"/api/jobs/{job_id}").json()["status"] == "failed"
    finally:
        if orphan.poll() is None:
            orphan.kill()
            orphan.wait()


def test_unrelated_live_pid_is_not_killed(make_app, submit, runs_dir):
    from fastapi.testclient import TestClient

    with TestClient(make_app(start_runner=False)) as c:
        up = c.post("/api/uploads", headers=H,
                    files=[("files", ("h.xyz", b"1\n\nH 0 0 0\n"))]).json()["uploads"][0]
        job_id = submit(c, [up["id"]]).json()["jobs"][0]["job_id"]
    st = json.loads((_rd(runs_dir, job_id) / "state.json").read_text())
    st.update(status="running", pid=os.getpid())      # our own pid: must survive
    (_rd(runs_dir, job_id) / "state.json").write_text(json.dumps(st))
    make_app(start_runner=False)
    st = json.loads((_rd(runs_dir, job_id) / "state.json").read_text())
    assert st["status"] == "failed"
