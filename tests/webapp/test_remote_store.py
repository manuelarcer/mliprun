"""Machines file and SSH transport (REMOTE_PLAN sections 1 and 2)."""
import io
import json
import os
import re
import shlex
import stat
import sys
import tarfile
from pathlib import Path

import pytest
from remote_helpers import FAKE_SSH_CMD, make_tar, read_jsonl

from mliprun.webapp import remote
from mliprun.webapp.remote import (
    MachineNotFound,
    MachineStore,
    MachineValidationError,
    RemoteError,
    RemoteTarget,
    RemoteUnreachable,
    SSHTransport,
    backoff_delay,
    default_env_name,
)

GOOD = {"name": "cos-cluster", "host": "cos-cluster", "default_location": "/scratchb/juar/runs"}


@pytest.fixture
def store(tmp_path):
    return MachineStore(tmp_path / "machines.json")


def fields(exc_info):
    return {d["field"] for d in exc_info.value.details}


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
def test_round_trip_mode_and_ids(store, tmp_path):
    m = store.add_machine({**GOOD, "user": "juar", "port": 2222, "identity_file": "~/.ssh/id",
                           "jump_host": "me@jump:22", "poll_interval_s": 0.5,
                           "copy_back_max_mb": 0})
    assert re.fullmatch(r"m-[0-9a-f]{6}", m["id"])
    assert m["poll_interval_s"] == 0.5 and m["copy_back_max_mb"] == 0 and m["envs"] == []
    e = store.add_env(m["id"], {"python": "/x/.venv/mace/bin/python", "setup": "module load a"},
                      probe={"ok": False, "error": "nope"})
    assert re.fullmatch(r"e-[0-9a-f]{6}", e["id"]) and e["name"] == "cos-cluster-mace"
    path = tmp_path / "machines.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))
    data = json.loads(path.read_text())
    assert data["schema_version"] == 1 and data["machines"][0]["id"] == m["id"]
    reloaded = MachineStore(path)
    assert reloaded.list() == store.list()
    assert reloaded.get_env(e["id"]).machine["host"] == "cos-cluster"
    assert reloaded.get_env(e["id"]).python == "/x/.venv/mace/bin/python"


def test_defaults_applied(store):
    m = store.add_machine(GOOD)
    assert (m["user"], m["port"], m["identity_file"], m["jump_host"]) == (None,) * 4
    assert m["poll_interval_s"] == 2.0 and m["copy_back_max_mb"] == 50


@pytest.mark.parametrize("patch, field", [
    ({"name": ""}, "name"), ({"name": "a b"}, "name"), ({"name": "x" * 61}, "name"),
    ({"name": None}, "name"),
    ({"host": "-oProxyCommand=x"}, "host"), ({"host": "a b"}, "host"), ({"host": ""}, "host"),
    ({"host": ".x"}, "host"),
    ({"user": "a b"}, "user"), ({"user": "-x y"}, "user"),
    ({"port": 0}, "port"), ({"port": 65536}, "port"), ({"port": "22"}, "port"),
    ({"port": True}, "port"),
    ({"identity_file": "relative/key"}, "identity_file"),
    ({"identity_file": "/a\nb"}, "identity_file"), ({"identity_file": "/a\0b"}, "identity_file"),
    ({"jump_host": "-oX"}, "jump_host"), ({"jump_host": "a b"}, "jump_host"),
    ({"jump_host": "u@h:x"}, "jump_host"),
    ({"default_location": "relative"}, "default_location"),
    ({"default_location": None}, "default_location"),
    ({"poll_interval_s": 0.4}, "poll_interval_s"), ({"poll_interval_s": 61}, "poll_interval_s"),
    ({"poll_interval_s": "2"}, "poll_interval_s"),
    ({"copy_back_max_mb": -1}, "copy_back_max_mb"),
    ({"copy_back_max_mb": 10001}, "copy_back_max_mb"),
    ({"bogus": 1}, "bogus"),
], ids=lambda v: None)
def test_machine_validation_rejects(store, patch, field):
    with pytest.raises(MachineValidationError) as exc:
        store.add_machine({**GOOD, **patch})
    assert field in fields(exc)
    assert store.list() == []                       # nothing stored, nothing written
    assert not store.path.exists()


@pytest.mark.parametrize("patch, field", [
    ({"python": "relative/python"}, "python"), ({"python": "/a b/python"}, "python"),
    ({"python": "/a\nb"}, "python"), ({"python": None}, "python"),
    ({"setup": "x" * 4001}, "setup"), ({"setup": 5}, "setup"),
    ({"name": "bad name"}, "name"), ({"bogus": 1}, "bogus"),
], ids=lambda v: None)
def test_env_validation_rejects(store, patch, field):
    mid = store.add_machine(GOOD)["id"]
    with pytest.raises(MachineValidationError) as exc:
        store.add_env(mid, {"python": "/p/bin/python", **patch})
    assert field in fields(exc)
    assert store.get_machine(mid)["envs"] == []


def test_names_are_unique(store):
    m1 = store.add_machine(GOOD)
    with pytest.raises(MachineValidationError) as exc:
        store.add_machine({**GOOD, "host": "other"})
    assert fields(exc) == {"name"}
    m2 = store.add_machine({**GOOD, "name": "second", "host": "second"})
    store.add_env(m1["id"], {"python": "/a/b/python", "name": "same"})
    with pytest.raises(MachineValidationError) as exc:       # env names unique across machines
        store.add_env(m2["id"], {"python": "/c/d/python", "name": "same"})
    assert fields(exc) == {"name"}
    with pytest.raises(MachineValidationError):
        store.update_machine(m2["id"], {"name": "cos-cluster"})


@pytest.mark.parametrize("name, python, expected", [
    ("cos", "/x/.venv/mace/bin/python", "cos-mace"),
    ("cos", "/home/u/envs/uma/bin/python3", "cos-uma"),
    ("cos", "/opt/py/python", "cos-py"),            # no bin/ parent: the file's parent dir
    ("cos", "/usr/bin/python3", "cos-usr"),
    ("c", "/a/my env/bin/python", "c-my-env"),
], ids=["venv", "python3", "no_bin", "usr_bin", "space_sanitised"])
def test_default_env_name(name, python, expected):
    assert default_env_name(name, python) == expected


def test_update_and_delete(store):
    m = store.add_machine(GOOD)
    e = store.add_env(m["id"], {"python": "/a/venv1/bin/python"})
    store.set_probe(m["id"], e["id"], {"ok": True})
    upd = store.update_machine(m["id"], {"poll_interval_s": 5, "user": "me"})
    assert upd["poll_interval_s"] == 5.0 and upd["user"] == "me" and upd["host"] == "cos-cluster"
    e2 = store.update_env(m["id"], e["id"], {"setup": "export A=1"})
    assert e2["probe"] == {"ok": True}                       # same python: probe kept
    e3 = store.update_env(m["id"], e["id"], {"python": "/a/venv2/bin/python"})
    assert e3["probe"] is None and e3["name"] == e["name"]   # new python: probe dropped
    store.delete_env(m["id"], e["id"])
    with pytest.raises(MachineNotFound):
        store.get_env(e["id"])
    store.delete_machine(m["id"])
    assert store.list() == [] and MachineStore(store.path).list() == []
    with pytest.raises(MachineNotFound):
        store.update_machine("m-000000", {})


def test_corrupt_file_is_kept_and_store_starts_empty(tmp_path):
    path = tmp_path / "machines.json"
    path.write_text("{not json")
    s = MachineStore(path)
    assert s.list() == []
    assert (tmp_path / "machines.json.bad").read_text() == "{not json"


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #
def make_target(tmp_path, **machine):
    m = {"id": "m-aaaaaa", "name": "box", "host": "box", **machine}
    e = {"id": "e-bbbbbb", "name": "box-x", "python": sys.executable, "setup": ""}
    return RemoteTarget(m, e)


def test_ssh_argv_construction(tmp_path):
    t = SSHTransport(["ssh"], ctl_dir=tmp_path / "ctl")
    target = make_target(tmp_path, port=2222, identity_file="/k/id", jump_host="j@jh:22",
                         user="juar", host="cos.example.org")
    cmd = t.remote_command(target, "poll", ["/r", 5, 1048576])
    argv = t.ssh_argv(target.machine, cmd)
    assert argv[0] == "ssh"
    dd = argv.index("--")
    assert argv[dd + 1] == "cos.example.org" and argv[dd + 2] == cmd and len(argv) == dd + 3
    opts = argv[1:dd]
    for o in ("BatchMode=yes", "ConnectTimeout=10", "ServerAliveInterval=15",
              "ServerAliveCountMax=3", "ControlMaster=auto", "ControlPersist=300",
              f"ControlPath={tmp_path / 'ctl'}/%C"):
        assert opts[opts.index(o) - 1] == "-o"
    assert opts[opts.index("-p") + 1] == "2222"
    assert opts[opts.index("-i") + 1] == "/k/id"
    assert opts[opts.index("-J") + 1] == "j@jh:22"
    assert opts[opts.index("-l") + 1] == "juar"
    assert stat.S_IMODE((tmp_path / "ctl").stat().st_mode) == 0o700
    plain = t.ssh_argv({"host": "h"}, "true")
    assert "-p" not in plain and "-i" not in plain and "-J" not in plain and "-l" not in plain
    assert plain[-3:] == ["--", "h", "true"]


def test_remote_command_quotes_every_value(tmp_path):
    t = SSHTransport(["ssh"], ctl_dir=tmp_path / "ctl")
    target = make_target(tmp_path)
    target.env["setup"] = "module load x\nexport A='1 2'\n"
    nasty = "/p/it's a dir; rm -rf $HOME `x`"
    cmd = t.remote_command(target, "inspect", ["--", nasty, "a b"])
    outer = shlex.split(cmd)
    assert outer[:2] == ["/bin/sh", "-c"] and len(outer) == 3
    script = outer[2]
    assert script.startswith("module load x\nexport A='1 2'\n")
    last = script.splitlines()[-1]
    assert shlex.split(last) == ["exec", sys.executable, "-m", "mliprun.webapp.remote_agent",
                                 "inspect", "--", nasty, "a b"]


def test_quoted_path_round_trips_through_fake_ssh(tmp_path):
    t = SSHTransport(FAKE_SSH_CMD, ctl_dir=tmp_path / "ctl")
    target = make_target(tmp_path)
    loc = tmp_path / "it's a dir (x)"
    tar = make_tar([("job.json", b"{}")])
    out = t.run_json(target, "create", [str(loc), "run 1".replace(" ", "-")], stdin=tar)
    assert Path(out["run_dir"]) == (loc / "run-1").resolve()
    assert (loc / "run-1" / "job.json").read_bytes() == b"{}"
    info = t.run_json(target, "inspect", ["--", str(loc), "weird 'name"])
    assert info["exists"] is True
    assert info["folders"][0]["folder"] == "weird-name"
    assert info["folders"][0]["path"] == str((loc / "weird-name").resolve())


def test_setup_runs_before_the_agent(tmp_path):
    marker = tmp_path / "setup-ran"
    t = SSHTransport(FAKE_SSH_CMD, ctl_dir=tmp_path / "ctl")
    target = make_target(tmp_path)
    target.env["setup"] = f"touch {shlex.quote(str(marker))}"
    t.run_json(target, "gpus")
    assert marker.exists()


def test_unreachable_and_agent_errors(tmp_path):
    t = SSHTransport(FAKE_SSH_CMD, ctl_dir=tmp_path / "ctl")
    down = make_target(tmp_path, host="down")
    with pytest.raises(RemoteUnreachable) as exc:
        t.run(down, "gpus")
    assert exc.value.returncode == 255 and "Connection refused" in exc.value.message
    target = make_target(tmp_path)
    with pytest.raises(RemoteError) as exc:               # agent exit 4, not an ssh failure
        t.run(target, "list", [str(tmp_path / "nope")])
    assert not isinstance(exc.value, RemoteUnreachable)
    assert exc.value.returncode == 4 and "does not exist" in exc.value.message
    bad_python = RemoteTarget(target.machine, {**target.env, "python": "/nonexistent/python"})
    with pytest.raises(RemoteError) as exc:
        t.run(bad_python, "gpus")
    assert exc.value.returncode != 255


def test_password_hint_added_on_permission_denied():
    err = remote._error_from(255, b"user@host: Permission denied (publickey,password).\n", "x")
    assert isinstance(err, RemoteUnreachable)
    assert remote.PASSWORD_HINT in err.message


def test_down_file_and_traffic_log(tmp_path, monkeypatch):
    log = tmp_path / "ssh.log"
    down = tmp_path / "down"
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    monkeypatch.setenv("FAKE_SSH_DOWN", str(down))
    t = SSHTransport(FAKE_SSH_CMD, ctl_dir=tmp_path / "ctl")
    target = make_target(tmp_path)
    t.run_json(target, "gpus")
    down.write_text("")
    with pytest.raises(RemoteUnreachable):
        t.run(target, "gpus")
    down.unlink()
    t.run_json(target, "gpus")
    entries = read_jsonl(log)
    assert [e["rc"] for e in entries] == [0, 255, 0]
    assert entries[0]["bytes_out"] > 0 and entries[1]["bytes_out"] == 0
    assert "--" in entries[0]["argv"] and entries[0]["host"] == "box"


def test_timeout_is_unreachable(tmp_path):
    t = SSHTransport([sys.executable, "-c", "import time; time.sleep(30)"],
                     ctl_dir=tmp_path / "ctl")
    with pytest.raises(RemoteUnreachable) as exc:
        t.run(make_target(tmp_path), "gpus", timeout=0.3)
    assert "within 0.3 s" in exc.value.message


def test_missing_ssh_binary_is_a_remote_error(tmp_path):
    t = SSHTransport(["/definitely/not/ssh"], ctl_dir=tmp_path / "ctl")
    with pytest.raises(RemoteError) as exc:
        t.run(make_target(tmp_path), "gpus")
    assert "cannot start ssh" in exc.value.message
    with pytest.raises(RemoteError):
        t.run_raw({"host": "h"}, "true")
    with pytest.raises(RemoteError):
        t.stream(make_target(tmp_path), "cat", ["/x", "y"])


def test_poll_and_stream_cat_pack(tmp_path):
    t = SSHTransport(FAKE_SSH_CMD, ctl_dir=tmp_path / "ctl")
    target = make_target(tmp_path)
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_bytes(b'{"a":1}\n{"b":2}\n')
    (run_dir / "big.bin").write_bytes(os.urandom(300_000))
    header, chunk = t.poll(target, str(run_dir), 0, 1 << 20)
    assert chunk == b'{"a":1}\n{"b":2}\n' and header["size"] == len(chunk)
    s = t.stream(target, "cat", [str(run_dir), "big.bin"])
    try:
        data = b"".join(s.iter_chunks(65536))
        s.finish()
    finally:
        s.close()
    assert data == (run_dir / "big.bin").read_bytes()
    s = t.stream(target, "cat", [str(run_dir), "missing"])
    try:
        assert list(s.iter_chunks()) == []
        with pytest.raises(RemoteError) as exc:
            s.finish()
    finally:
        s.close()
    assert exc.value.returncode == 4
    s = t.stream(target, "pack", [str(run_dir), 1000000])
    try:
        with tarfile.open(fileobj=s.stdout, mode="r|") as tar:
            names = [m.name for m in tar]
        s.finish()
    finally:
        s.close()
    assert names == ["big.bin"]


def test_run_raw(tmp_path):
    t = SSHTransport(FAKE_SSH_CMD, ctl_dir=tmp_path / "ctl")
    assert t.run_raw({"host": "box"}, "echo hello").strip() == "hello"
    with pytest.raises(RemoteUnreachable):
        t.run_raw({"host": "down"}, "true")
    with pytest.raises(RemoteError):
        t.run_raw({"host": "box"}, "exit 3")


def test_poll_rejects_garbage(tmp_path):
    t = SSHTransport([sys.executable, "-c", "print('not json')"], ctl_dir=tmp_path / "ctl")
    with pytest.raises(RemoteError):
        t.poll(make_target(tmp_path), "/r", 0, 10)
    with pytest.raises(RemoteError):
        t.run_json(make_target(tmp_path), "gpus")


def test_probe_target_cases(tmp_path):
    t = SSHTransport(FAKE_SSH_CMD, ctl_dir=tmp_path / "ctl")
    ok = remote.probe_target(t, make_target(tmp_path))
    assert ok["ok"] is True and ok["agent_version"] == 1 and ok["worker_ok"] is True
    assert "emt" in [c["name"] for c in ok["calculators"]["calculators"]]
    down = remote.probe_target(t, make_target(tmp_path, host="down"))
    assert down["ok"] is False and "Connection refused" in down["error"]
    # an interpreter without mliprun: python -m fails with "No module named"
    bare = RemoteTarget(make_target(tmp_path).machine,
                        {"id": "e-1", "name": "n", "python": sys.executable,
                         "setup": "export PYTHONPATH=/nonexistent\n"
                                  "export PYTHONNOUSERSITE=1\ncd /"})
    # the venv's own mliprun (editable install) is still importable, so simulate by
    # a python that cannot find the module
    fake_py = tmp_path / "py"
    fake_py.write_text("#!/bin/sh\necho \"No module named mliprun.webapp.remote_agent\" >&2\n"
                       "exit 1\n")
    fake_py.chmod(0o755)
    old = RemoteTarget(bare.machine, {**bare.env, "python": str(fake_py), "setup": ""})
    r = remote.probe_target(t, old)
    assert r["ok"] is False and "too old for remote runs" in r["error"]
    assert "box" in r["error"]


def test_tar_local_inputs_and_backoff(tmp_path):
    run = tmp_path / "run"
    (run / "inputs" / "structure").mkdir(parents=True)
    (run / "job.json").write_text("{}")
    (run / "inputs" / "structure" / "POSCAR").write_text("Cu")
    (run / "events.jsonl").write_text("x\n")
    with tarfile.open(fileobj=io.BytesIO(remote.tar_local_inputs(run))) as tar:
        assert sorted(tar.getnames()) == ["inputs/structure/POSCAR", "job.json"]
    assert [backoff_delay(n) for n in (1, 2, 3, 4, 5, 6, 7, 20)] == [2, 4, 8, 16, 32, 60, 60, 60]
    assert backoff_delay(3, base=0.01) == pytest.approx(0.04)
