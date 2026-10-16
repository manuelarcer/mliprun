"""Edge paths of RemoteQueueRunner not covered by the end-to-end tests."""
import json
from pathlib import Path

import pytest
from remote_helpers import wait_until
from test_remote_e2e import H, cu, detail, events_of, fake_remote_worker, mirror_equals_remote, \
    n_steps, submit, when_failed  # noqa: F401 - fixtures re-exported for pytest


def test_chunked_polling_catches_up_without_waiting(make_client, remote_setup, cu,
                                                    fake_remote_worker, wait_for_status,
                                                    monkeypatch, tmp_path):
    from mliprun.webapp import jobs

    monkeypatch.setattr(jobs, "MAX_POLL_BYTES", 400)      # several lines per poll at most
    log = tmp_path / "ssh.log"
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    job = submit(c, up, target=info["env_id"], label="steps=25;sleep=0.0", folder="chunks")
    d = wait_for_status(c, job["job_id"], "done", timeout=60)
    assert n_steps(c, job["job_id"]) == 25 and mirror_equals_remote(d)
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    polls = [e for e in entries if " poll " in e["command"]]
    sizes = [e["bytes_out"] for e in polls]
    assert max(sizes) < 400 + 300                        # no poll returned more than ~one window
    events = events_of(c, job["job_id"])
    assert [e["seq"] for e in events] == list(range(len(events)))


@pytest.mark.parametrize("script, expected", [
    ("steps=2;error=boom", "boom"),
    ("steps=2;exit=0", "worker exited with code 0 without reporting a result"),
    ("steps=2;exit=5", "worker exited with code 5"),
], ids=["error_event", "exit0_no_result", "exit5"])
def test_failure_reasons_match_the_local_runner(make_client, remote_setup, cu,
                                                fake_remote_worker, script, expected):
    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True)
    up = cu(c)
    job = submit(c, up, target=info["env_id"], label=script, folder="reason")
    assert when_failed(c, job["job_id"])["error"] == expected


def test_copy_back_failure_is_logged_not_fatal(make_client, remote_setup, cu,
                                               fake_remote_worker, wait_for_status):
    from mliprun.webapp.remote import RemoteError

    info = remote_setup()
    c = make_client(machines_file=info["machines_file"], start_runner=True)

    def broken(*a, **k):
        raise RemoteError("pack exploded")

    c.app.state.transport.stream = broken
    up = cu(c)
    job = submit(c, up, target=info["env_id"], label="steps=2", folder="nocb")
    d = wait_for_status(c, job["job_id"], "done", timeout=60)
    state = json.loads((Path(d["run_dir"]) / "state.json").read_text())
    assert d["status"] == "done"
    assert not (Path(d["run_dir"]) / "remote_files.json").exists()
    assert state["remote"]["copy_back_error"] == "pack exploded"
    assert d["remote_error"] is None


def test_environment_removed_while_pending_fails_the_job(make_client, make_app, remote_setup,
                                                         cu):
    info = remote_setup()
    c1 = make_client(machines_file=info["machines_file"])           # runner off
    up = cu(c1)
    job = submit(c1, up, target=info["env_id"], folder="orphan-env")
    c1.__exit__(None, None, None)
    data = json.loads(info["machines_file"].read_text())
    data["machines"][0]["envs"] = []                                # env gone, machine stays
    info["machines_file"].write_text(json.dumps(data))
    from fastapi.testclient import TestClient

    with TestClient(make_app(machines_file=info["machines_file"], start_runner=True)) as c2:
        d = wait_until(lambda: (lambda x: x if x["status"] == "failed" else None)(
            detail(c2, job["job_id"])), 30, what="failed")
        assert d["error"] == "environment box-env of machine box is no longer configured"
        assert [e["status"] for e in events_of(c2, job["job_id"])
                if e["type"] == "job_status"] == ["pending", "running", "failed"]


def test_launch_while_unreachable_fails_the_job(make_client, make_app, remote_setup, cu,
                                                monkeypatch, tmp_path):
    down = tmp_path / "ssh-down"
    monkeypatch.setenv("FAKE_SSH_DOWN", str(down))
    info = remote_setup()
    c1 = make_client(machines_file=info["machines_file"])           # create works: ssh is up
    up = cu(c1)
    job = submit(c1, up, target=info["env_id"], folder="down-at-launch")
    c1.__exit__(None, None, None)
    down.write_text("")
    from fastapi.testclient import TestClient

    with TestClient(make_app(machines_file=info["machines_file"], start_runner=True,
                             ssh_command=c1.app.state.transport.ssh_command)) as c2:
        d = wait_until(lambda: (lambda x: x if x["status"] == "failed" else None)(
            detail(c2, job["job_id"])), 30, what="failed")
    assert d["error"].startswith("could not start the remote worker:")
    assert "Connection refused" in d["error"]


def test_running_job_of_a_removed_machine_can_still_be_cancelled(make_client, make_app,
                                                                 remote_setup, cu):
    info = remote_setup()
    c1 = make_client(machines_file=info["machines_file"])
    up = cu(c1)
    job = submit(c1, up, target=info["env_id"], folder="gone")
    c1.__exit__(None, None, None)
    # pretend a previous server left it running, then the machine vanished from the file
    run = Path(job["run_dir"])
    state = json.loads((run / "state.json").read_text())
    state.update(status="running", started_at="2026-10-16T10:00:00.000Z",
                 remote={"launched": True, "pids": {}})
    (run / "state.json").write_text(json.dumps(state))
    info["machines_file"].write_text(json.dumps({"schema_version": 1, "machines": []}))
    from fastapi.testclient import TestClient

    with TestClient(make_app(machines_file=info["machines_file"], start_runner=True)) as c2:
        d = detail(c2, job["job_id"])
        assert d["status"] == "running"
        assert d["remote_error"] == "machine box no longer configured"
        r = c2.post(f"/api/jobs/{job['job_id']}/cancel", headers=H)
        assert r.status_code == 200 and r.json()["job"]["status"] == "cancelled"
    final = json.loads((run / "state.json").read_text())
    assert final["status"] == "cancelled"
    assert "no longer configured" in final["reason"]
