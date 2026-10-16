"""The remote agent (REMOTE_PLAN section 3): subcommands called in-process and as subprocesses."""
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest
from remote_helpers import agent_env, make_tar, read_jsonl, run_agent, wait_until

from mliprun.webapp import paths, remote_agent

FAKE_WORKER = Path(__file__).resolve().parent / "fake_worker.py"


def call(capsysbinary, *argv, stdin=b""):
    """Run ``main(argv)`` in-process; return (rc, stdout bytes, stderr text)."""
    sys.stdin = io.TextIOWrapper(io.BytesIO(stdin))
    try:
        rc = remote_agent.main([str(a) for a in argv])
    finally:
        sys.stdin = sys.__stdin__
    out = capsysbinary.readouterr()
    return rc, out.out, out.err.decode()


def call_json(capsysbinary, *argv, stdin=b""):
    rc, out, err = call(capsysbinary, *argv, stdin=stdin)
    assert rc == 0, err
    return json.loads(out)


def write_job(run_dir, label="steps=3;sleep=0.01", n_status=2):
    """A minimal job dir the scripted fake worker understands."""
    run_dir.mkdir(parents=True, exist_ok=True)
    job = {"job_id": "20261016-101500-optimize-abc123", "calc_type": "optimize",
           "label": label}
    (run_dir / "job.json").write_text(json.dumps(job))
    lines = [json.dumps({"job_id": job["job_id"], "seq": i, "t": 1.0, "type": "job_status",
                         "status": "pending" if i == 0 else "running"})
             for i in range(n_status)]
    (run_dir / "events.jsonl").write_text("".join(line + "\n" for line in lines))
    return job


@pytest.fixture
def fake_worker_env(monkeypatch):
    monkeypatch.setenv(remote_agent.WORKER_CMD_ENV,
                       json.dumps([sys.executable, str(FAKE_WORKER)]))


def test_agent_version_is_one():
    assert remote_agent.AGENT_VERSION == 1


def test_create_is_exclusive_and_extracts(tmp_path, capsysbinary):
    tar = make_tar([("job.json", b'{"a": 1}'), ("inputs/structure/POSCAR", b"Cu\n")])
    out = call_json(capsysbinary, "create", tmp_path / "loc", "run-1", stdin=tar)
    run_dir = Path(out["run_dir"])
    assert run_dir == (tmp_path / "loc" / "run-1").resolve()
    assert (run_dir / "job.json").read_bytes() == b'{"a": 1}'
    assert (run_dir / "inputs/structure/POSCAR").read_bytes() == b"Cu\n"
    rc, _, err = call(capsysbinary, "create", tmp_path / "loc", "run-1", stdin=tar)
    assert rc == remote_agent.EXIT_EXISTS == 3
    assert "already exists" in err
    # the existing folder is untouched
    assert (run_dir / "job.json").read_bytes() == b'{"a": 1}'


def test_create_expands_tilde_and_rejects_bad_names(tmp_path, capsysbinary, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    out = call_json(capsysbinary, "create", "~/runs", "x", stdin=make_tar([("job.json", b"{}")]))
    assert Path(out["run_dir"]) == (tmp_path / "runs" / "x").resolve()
    rc, _, err = call(capsysbinary, "create", tmp_path, "../evil")
    assert rc == remote_agent.EXIT_USAGE and "invalid folder name" in err
    rc, _, err = call(capsysbinary, "create", "relative/dir", "x")
    assert rc == remote_agent.EXIT_USAGE and "absolute" in err


@pytest.mark.parametrize("members", [
    [("/etc/passwd", b"x")],
    [("../escape.txt", b"x")],
    [("ok/../../escape.txt", b"x")],
    [("link", {"type": tarfile.SYMTYPE, "linkname": "/etc/passwd"})],
    [("hard", {"type": tarfile.LNKTYPE, "linkname": "job.json"})],
    [("dev", {"type": tarfile.CHRTYPE})],
], ids=["absolute", "dotdot", "nested_dotdot", "symlink", "hardlink", "device"])
def test_unsafe_tar_is_rejected_and_nothing_left(tmp_path, capsysbinary, members):
    tar = make_tar([("job.json", b"{}")] + members)
    rc, _, err = call(capsysbinary, "create", tmp_path / "loc", "run", stdin=tar)
    assert rc == remote_agent.EXIT_ERROR
    assert "could not extract" in err
    assert not (tmp_path / "loc" / "run").exists()      # rolled back
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "loc" / "escape.txt").exists()
    with pytest.raises(paths.UnsafeArchive):
        paths.safe_extract_tar(tar, tmp_path)


def test_safe_extract_counts_files(tmp_path):
    n = paths.safe_extract_tar(make_tar([("d/a", b"1"), ("b", b"22")]), tmp_path)
    assert n == 2 and (tmp_path / "d" / "a").read_bytes() == b"1"


def test_inspect_has_no_reserved_names(tmp_path, capsysbinary):
    (tmp_path / "exists").mkdir()
    out = call_json(capsysbinary, "inspect", "--", tmp_path, "exists", "_uploads", "new one")
    assert out["exists"] is True and out["writable"] is True and out["error"] is None
    by_input = {f["input"]: f for f in out["folders"]}
    assert by_input["exists"]["exists"] is True
    assert by_input["_uploads"]["error"] is None          # reserved only on the server
    assert by_input["new one"]["folder"] == "new-one" and by_input["new one"]["exists"] is False
    out = call_json(capsysbinary, "inspect", "relative")
    assert "absolute" in out["error"]


def test_poll_cuts_at_newline_and_resumes(tmp_path, capsysbinary):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    lines = [b'{"seq":0}\n', b'{"seq":1,"pad":"xxxxxxxx"}\n', b'{"seq":2}\n']
    (run_dir / "events.jsonl").write_bytes(b"".join(lines) + b'{"seq":3,"par')   # partial tail
    total = sum(map(len, lines))
    rc, out, _ = call(capsysbinary, "poll", run_dir, 0, len(lines[0]) + 5)
    header, _, chunk = out.partition(b"\n")
    header = json.loads(header)
    assert chunk == lines[0] and header["chunk_len"] == len(lines[0])
    assert header["size"] == total + len(b'{"seq":3,"par')
    assert header["alive"] is False and header["exit"] is None
    rc, out, _ = call(capsysbinary, "poll", run_dir, len(lines[0]), 1 << 20)
    header, _, chunk = out.partition(b"\n")
    assert chunk == lines[1] + lines[2]               # partial line not delivered
    assert json.loads(header)["chunk_len"] == len(chunk)
    rc, out, _ = call(capsysbinary, "poll", run_dir, total, 1 << 20)
    assert json.loads(out.partition(b"\n")[0])["chunk_len"] == 0


def test_poll_extends_window_for_a_line_longer_than_max(tmp_path, capsysbinary):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    long_line = b'{"pad":"' + b"y" * 5000 + b'"}\n'
    (run_dir / "events.jsonl").write_bytes(long_line)
    rc, out, _ = call(capsysbinary, "poll", run_dir, 0, 100)
    header, _, chunk = out.partition(b"\n")
    assert chunk == long_line and json.loads(header)["chunk_len"] == len(long_line)


def test_cancel_pid_reuse_guard(tmp_path, capsysbinary):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (run_dir / "remote_pids.json").write_text(json.dumps({"worker_pid": stranger.pid}))
        out = call_json(capsysbinary, "cancel", run_dir)
        assert out == {"signalled": False}
        time.sleep(0.2)
        assert stranger.poll() is None                  # not killed: cmdline lacks run_dir
    finally:
        stranger.kill()
        stranger.wait()
    mine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                             str(run_dir.resolve())])
    try:
        (run_dir / "remote_pids.json").write_text(json.dumps({"worker_pid": mine.pid}))
        assert call_json(capsysbinary, "cancel", run_dir) == {"signalled": True}
        assert mine.wait(timeout=10) == -15             # SIGTERM
    finally:
        if mine.poll() is None:
            mine.kill()
            mine.wait()
    # no pids file at all: nothing to signal
    (run_dir / "remote_pids.json").unlink()
    assert call_json(capsysbinary, "cancel", run_dir, "--kill") == {"signalled": False}


def test_cancel_kill_sends_sigkill(tmp_path, capsysbinary):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    proc = subprocess.Popen([sys.executable, "-c", code, str(run_dir.resolve())])
    try:
        (run_dir / "remote_pids.json").write_text(json.dumps({"worker_pid": proc.pid}))
        time.sleep(0.3)
        call_json(capsysbinary, "cancel", run_dir)
        time.sleep(0.3)
        assert proc.poll() is None                      # SIGTERM ignored
        assert call_json(capsysbinary, "cancel", run_dir, "--kill") == {"signalled": True}
        assert proc.wait(timeout=10) == -9
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_pack_size_limit_and_exclusions(tmp_path, capsysbinary):
    run_dir = tmp_path / "r"
    (run_dir / "inputs" / "structure").mkdir(parents=True)
    (run_dir / "sub").mkdir()
    for name, size in [("job.json", 5), ("state.json", 5), ("events.jsonl", 5),
                       ("inputs/structure/POSCAR", 5), ("small.txt", 10),
                       ("sub/edge.txt", 100), ("big.bin", 101)]:
        (run_dir / name).write_bytes(b"x" * size)
    (tmp_path / "outside.txt").write_bytes(b"secret")
    (run_dir / "link.txt").symlink_to(tmp_path / "outside.txt")
    rc, out, _ = call(capsysbinary, "pack", run_dir, 100)
    assert rc == 0
    with tarfile.open(fileobj=io.BytesIO(out)) as tar:
        names = sorted(tar.getnames())
    assert names == ["small.txt", "sub/edge.txt"]       # 100 B included, 101 B not
    rc, out, _ = call(capsysbinary, "pack", run_dir, 0)
    with tarfile.open(fileobj=io.BytesIO(out)) as tar:
        assert tar.getnames() == []


def test_list_and_cat(tmp_path, capsysbinary):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "a.txt").write_bytes(b"hello")
    (tmp_path / "outside.txt").write_bytes(b"secret")
    (run_dir / "link.txt").symlink_to(tmp_path / "outside.txt")
    listing = call_json(capsysbinary, "list", run_dir)
    assert [(f["path"], f["size"]) for f in listing] == [("a.txt", 5)]
    assert listing[0]["mtime"].endswith("Z")
    rc, out, _ = call(capsysbinary, "cat", run_dir, "a.txt")
    assert rc == 0 and out == b"hello"
    for bad in ("../outside.txt", "link.txt", "missing.txt"):
        rc, out, err = call(capsysbinary, "cat", run_dir, bad)
        assert rc == remote_agent.EXIT_NOT_FOUND and out == b""
    rc, _, err = call(capsysbinary, "list", tmp_path / "nope")
    assert rc == remote_agent.EXIT_NOT_FOUND


def test_gpus_without_nvidia_smi(capsysbinary, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    out = call_json(capsysbinary, "gpus")
    assert out["gpus"] == [] and "nvidia-smi" in out["note"]


def test_gpus_parses_csv(capsysbinary, monkeypatch):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 0, "0, NVIDIA L40S, 1024, 46068, 12\nbroken line\n1, NVIDIA L40S, 5, 46068, 0\n",
            "")
    monkeypatch.setattr(remote_agent.subprocess, "run", fake_run)
    out = call_json(capsysbinary, "gpus")
    assert [g["index"] for g in out["gpus"]] == [0, 1]
    assert out["gpus"][0]["memory_total_mb"] == 46068.0 and out["gpus"][0]["utilization_pct"] == 12.0


def test_probe_reports_calculators(capsysbinary):
    out = call_json(capsysbinary, "probe")
    assert out["agent_version"] == 1 and out["worker_ok"] is True
    assert out["python"] == sys.executable
    assert "emt" in [c["name"] for c in out["calculators"]["calculators"]]
    assert out["calculators"]["devices"]["cuda_available"] in (True, False, None)


def test_usage_errors_exit_2(capsysbinary):
    rc, _, _ = call(capsysbinary, "poll", "x", "notanumber", "1")
    assert rc == remote_agent.EXIT_USAGE


def test_discard_refuses_launched_dir(tmp_path, capsysbinary):
    fresh = tmp_path / "fresh"
    write_job(fresh)
    launched = tmp_path / "launched"
    write_job(launched)
    (launched / "remote_pids.json").write_text("{}")
    other = tmp_path / "other"
    other.mkdir()
    assert call_json(capsysbinary, "discard", fresh)["discarded"] is True
    assert not fresh.exists()
    out = call_json(capsysbinary, "discard", launched)
    assert out["discarded"] is False and launched.is_dir()
    out = call_json(capsysbinary, "discard", other)       # no job.json: not ours to delete
    assert out["discarded"] is False and other.is_dir()
    assert call_json(capsysbinary, "discard", tmp_path / "missing")["discarded"] is False


def test_launch_poll_to_completion_with_scripted_worker(tmp_path, fake_worker_env):
    run_dir = tmp_path / "r"
    write_job(run_dir, "steps=4;sleep=0.01")
    events = (run_dir / "events.jsonl").read_bytes()
    env = agent_env(**{remote_agent.WORKER_CMD_ENV: os.environ[remote_agent.WORKER_CMD_ENV]})
    proc = run_agent("launch", run_dir, stdin=events, env=env)
    assert proc.returncode == 0, proc.stderr
    pids = json.loads(proc.stdout)
    assert set(pids) >= {"supervisor_pid", "worker_pid", "hostname", "started_at"}
    # launching again does not start a second worker and returns the same pids
    again = run_agent("launch", run_dir, stdin=b"", env=env)
    assert json.loads(again.stdout) == pids

    def finished():
        p = run_agent("poll", run_dir, 0, 1 << 20, env=env)
        header = json.loads(p.stdout.split(b"\n", 1)[0])
        return header if header["exit"] is not None else None

    header = wait_until(finished, 60, what="worker exit")
    assert header["exit"]["returncode"] == 0 and header["alive"] is False
    p = run_agent("poll", run_dir, 0, 1 << 20, env=env)
    head, _, chunk = p.stdout.partition(b"\n")
    assert len(chunk) == json.loads(head)["chunk_len"] == json.loads(head)["size"]
    events = read_jsonl(run_dir / "events.jsonl")
    assert [e["seq"] for e in events] == list(range(len(events)))   # seq continues
    assert [e["type"] for e in events][:2] == ["job_status", "job_status"]
    assert events[-1]["type"] == "result" and events[-1]["steps"] == 4
    assert (run_dir / "worker.log").exists()


def test_launch_then_cancel_via_agent(tmp_path, fake_worker_env):
    run_dir = tmp_path / "r"
    write_job(run_dir, "steps=500;sleep=0.05")
    env = agent_env(**{remote_agent.WORKER_CMD_ENV: os.environ[remote_agent.WORKER_CMD_ENV]})
    events = (run_dir / "events.jsonl").read_bytes()
    assert run_agent("launch", run_dir, stdin=events, env=env).returncode == 0
    cancel = run_agent("cancel", run_dir, env=env)
    assert json.loads(cancel.stdout) == {"signalled": True}

    def finished():
        p = run_agent("poll", run_dir, 0, 1 << 20, env=env)
        header = json.loads(p.stdout.split(b"\n", 1)[0])
        return header if header["exit"] is not None else None

    header = wait_until(finished, 60, what="worker exit after cancel")
    assert header["exit"]["returncode"] == 3             # worker contract: 3 = cancelled
    assert json.loads((run_dir / "summary.json").read_text())["status"] == "cancelled"


def test_launch_real_emt_worker(tmp_path):
    from test_worker import _cu, make_run_dir

    from ase.calculators.emt import EMT
    from ase.io import read

    run_dir = make_run_dir(tmp_path, "optimize", {"fmax": 0.05},
                           {"structure": (_cu(), "POSCAR", "vasp")})
    events = (run_dir / "events.jsonl").read_bytes()
    proc = run_agent("launch", run_dir, stdin=events)
    assert proc.returncode == 0, proc.stderr

    def finished():
        p = run_agent("poll", run_dir, 0, 1 << 20)
        header = json.loads(p.stdout.split(b"\n", 1)[0])
        return header if header["exit"] is not None else None

    header = wait_until(finished, 120, what="EMT worker exit")
    assert header["exit"]["returncode"] == 0
    result = [e for e in read_jsonl(run_dir / "events.jsonl") if e["type"] == "result"][0]
    atoms = read(run_dir / "inputs/structure/POSCAR")
    atoms.calc = EMT()
    assert result["final_energy_eV"] <= atoms.get_potential_energy() + 1e-9
    assert result["converged"] is True


def test_supervise_survives_sigterm(tmp_path, fake_worker_env):
    """The detached supervisor ignores SIGTERM so only the worker is cancelled."""
    run_dir = tmp_path / "r"
    write_job(run_dir, "steps=200;sleep=0.05")
    env = agent_env(**{remote_agent.WORKER_CMD_ENV: os.environ[remote_agent.WORKER_CMD_ENV]})
    pids = json.loads(run_agent("launch", run_dir, stdin=(run_dir / "events.jsonl").read_bytes(),
                                env=env).stdout)
    os.kill(pids["supervisor_pid"], 15)
    time.sleep(0.5)
    os.kill(pids["supervisor_pid"], 0)                   # still alive: no ProcessLookupError
    run_agent("cancel", run_dir, env=env)
    wait_until(lambda: (run_dir / "worker_exit.json").exists(), 60, what="exit file")
    assert json.loads((run_dir / "worker_exit.json").read_text())["returncode"] == 3


def test_supervise_reports_a_worker_that_cannot_start(tmp_path, capsysbinary, monkeypatch):
    run_dir = tmp_path / "r"
    write_job(run_dir)
    monkeypatch.setenv(remote_agent.WORKER_CMD_ENV, json.dumps(["/nonexistent/worker"]))
    rc, _, err = call(capsysbinary, "supervise", run_dir)
    assert rc == remote_agent.EXIT_ERROR and "cannot start the worker" in err
    exit_info = json.loads((run_dir / "worker_exit.json").read_text())
    assert exit_info["returncode"] is None and "/nonexistent/worker" in exit_info["error"]
    assert not (run_dir / "remote_pids.json").exists()
    # launch sees the exit file and fails at once instead of waiting 15 s
    t0 = time.monotonic()
    rc, _, err = call(capsysbinary, "launch", run_dir)
    assert rc == remote_agent.EXIT_ERROR and "could not be started" in err
    assert time.monotonic() - t0 < 10


def test_supervise_in_process_records_pids_and_exit(tmp_path, capsysbinary, monkeypatch):
    import signal as _signal

    run_dir = tmp_path / "r"
    write_job(run_dir, "steps=2;sleep=0.01")
    monkeypatch.setenv(remote_agent.WORKER_CMD_ENV,
                       json.dumps([sys.executable, str(FAKE_WORKER)]))
    old = (_signal.getsignal(_signal.SIGTERM), _signal.getsignal(_signal.SIGHUP))
    try:
        rc, _, err = call(capsysbinary, "supervise", run_dir)
    finally:
        _signal.signal(_signal.SIGTERM, old[0])
        _signal.signal(_signal.SIGHUP, old[1])
    assert rc == 0, err
    pids = json.loads((run_dir / "remote_pids.json").read_text())
    assert pids["supervisor_pid"] == os.getpid() and pids["worker_pid"] > 0
    assert json.loads((run_dir / "worker_exit.json").read_text())["returncode"] == 0
    # the exit file is there, the worker is gone: poll reports exit, not alive
    rc, out, _ = call(capsysbinary, "poll", run_dir, 0, 0)
    header = json.loads(out.split(b"\n", 1)[0])
    assert header["alive"] is False and header["exit"]["returncode"] == 0
    assert header["chunk_len"] == 0
