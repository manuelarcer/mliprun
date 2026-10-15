"""Unit tests of backend helpers: errors, sanitizing, pids, broadcaster, limits."""
import asyncio
import json
import subprocess
import sys

import pytest

H = {"X-Mliprun": "1"}


def test_error_codes_and_body():
    from mliprun.webapp.errors import ApiError, code_for_status, error_body, validation_error

    assert code_for_status(404) == "not_found"
    assert code_for_status(418) == "bad_request"
    assert code_for_status(503) == "internal_error"
    assert error_body("x", "y") == {"error": {"code": "x", "message": "y", "details": []}}
    e = ApiError(409, "busy")
    assert e.code == "conflict" and e.body()["error"]["message"] == "busy"
    v = validation_error([{"field": "a", "message": "b"}])
    assert v.status == 422 and v.message == "1 problem with the submission"


@pytest.mark.parametrize("raw, clean", [
    ("POSCAR", "POSCAR"), ("CONTCAR", "CONTCAR"), ("dir/sub/a b.vasp", "a_b.vasp"),
    ("C:\\x\\y.cif", "y.cif"), ("...hidden", "hidden"), ("", "structure"), (None, "structure"),
    ("é.xyz", "_.xyz"), ("ok-1+2.traj", "ok-1+2.traj"),
])
def test_sanitize_filename(raw, clean):
    from mliprun.webapp.uploads import sanitize_filename

    assert sanitize_filename(raw) == clean


def test_iso_utc_format():
    from datetime import datetime, timezone

    from mliprun.webapp.uploads import iso_utc

    assert iso_utc(datetime(2026, 10, 15, 14, 23, 1, 123456, tzinfo=timezone.utc)) == \
        "2026-10-15T14:23:01.123Z"


def test_pid_alive_and_signal_helpers():
    from mliprun.webapp.jobs import _signal_group, pid_alive

    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert pid_alive(p.pid)
        _signal_group(p.pid, 15)                 # not a group leader: plain kill
        p.wait(timeout=5)
    finally:
        if p.poll() is None:
            p.kill()
    assert not pid_alive(p.pid)
    _signal_group(p.pid, 15)                     # dead pid: silently ignored


def test_broadcaster_drops_a_client_that_falls_far_behind(monkeypatch):
    import mliprun.webapp.jobs as jobs

    async def run():
        monkeypatch.setattr(jobs, "MAX_OUTBOX", 3)
        conn = jobs.Connection()
        b = jobs.Broadcaster()
        b.add(conn)
        conn.subs["j"] = 0
        for seq in range(5):
            b.publish("j", seq, "opt_step", f'{{"seq": {seq}}}')
        items = [conn.outbox.get_nowait() for _ in range(conn.outbox.qsize())]
        return conn, items

    conn, items = asyncio.run(run())
    assert conn.closed and items[-1] is None and len(items) == 4


def test_backend_summary_does_not_overwrite_worker_summary(tmp_path):
    from mliprun.webapp.jobs import JobRecord, write_backend_summary

    rec = JobRecord(tmp_path, {"job_id": "x", "calc_type": "md"},
                    {"status": "running", "started_at": "bad"}, 1)
    write_backend_summary(rec, "failed", "boom")
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["status"] == "failed" and s["walltime_s"] is None
    assert s["error"] == {"message": "boom", "exception_type": None}
    (tmp_path / "summary.json").write_text(json.dumps({"status": "done"}))
    write_backend_summary(rec, "failed", "boom")
    assert json.loads((tmp_path / "summary.json").read_text()) == {"status": "done"}


def test_empty_and_oversize_uploads(client, monkeypatch):
    import mliprun.webapp.uploads as uploads

    r = client.post("/api/uploads", headers=H, files=[("files", ("e.xyz", b""))])
    (info,) = r.json()["uploads"]
    assert info["ok"] is False and info["error"] == "file is empty"
    monkeypatch.setattr(uploads, "MAX_FILE_BYTES", 10)
    r = client.post("/api/uploads", headers=H,
                    files=[("files", ("big.xyz", b"1\n\nH 0 0 0\n" * 3))])
    (info,) = r.json()["uploads"]
    assert info["ok"] is False and "larger than" in info["error"]


def test_forced_format_and_bad_format(client):
    r = client.post("/api/uploads", headers=H, data={"format": "xyz"},
                    files=[("files", ("noext", b"1\n\nH 0 0 0\n"))])
    (info,) = r.json()["uploads"]
    assert info["ok"] and info["format"] == "xyz"
    r = client.post("/api/uploads", headers=H, data={"format": "no-such-format"},
                    files=[("files", ("h.xyz", b"1\n\nH 0 0 0\n"))])
    (info,) = r.json()["uploads"]
    assert info["ok"] is False


def test_index_missing_is_404(client, monkeypatch, tmp_path):
    import mliprun.webapp.server as server

    monkeypatch.setattr(server, "STATIC_DIR", tmp_path)
    r = client.get("/")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_corrupt_upload_meta_is_ignored(make_app, runs_dir):
    (runs_dir / "_uploads" / "0123456789ab").mkdir(parents=True)
    (runs_dir / "_uploads" / "0123456789ab" / "meta.json").write_text("{broken")
    from fastapi.testclient import TestClient

    with TestClient(make_app()) as c:
        assert c.get("/api/uploads").json()["uploads"] == []


def test_method_not_allowed_uses_error_format(client):
    r = client.put("/api/health", headers=H)
    assert r.status_code == 405
    assert r.json()["error"]["code"] == "bad_request"
