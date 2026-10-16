"""Remote agent of the web app: ``python -m mliprun.webapp.remote_agent <subcmd> ...``.

Runs on the remote machine with the env's Python, started over SSH by the
server for every operation. Standard library only at module level (it starts
on every poll, so import time matters); ``probe`` imports
``mliprun.webapp.calculators`` lazily.

Output is one JSON object on stdout, except ``poll`` (a JSON header line then
raw bytes), ``pack`` (a tar stream) and ``cat`` (raw bytes). Errors: a one
line message on stderr and a non-zero exit code (3 = folder already exists).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from mliprun.webapp import paths

AGENT_VERSION = 1

EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_EXISTS = 3
EXIT_NOT_FOUND = 4
EXIT_TIMEOUT = 5

PIDS_FILE = "remote_pids.json"
EXIT_FILE = "worker_exit.json"
LAUNCH_WAIT_S = 15.0
WORKER_MODULE = "mliprun.webapp.worker"
#: Test hook: JSON list replacing the worker command prefix (run_dir appended).
WORKER_CMD_ENV = "MLIPRUN_AGENT_WORKER_CMD"


class AgentError(Exception):
    """A failure reported to the caller as a message plus an exit code."""

    def __init__(self, message: str, code: int = EXIT_ERROR):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _out(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _run_dir(raw: str) -> Path:
    path = Path(os.path.realpath(os.path.expanduser(raw)))
    if not path.is_dir():
        raise AgentError(f"run directory {raw} does not exist", EXIT_NOT_FOUND)
    return path


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001 - missing or unreadable
        return None
    return data if isinstance(data, dict) else None


def _process_info(pid: int) -> Optional[tuple]:
    """``(state, command line)`` of ``pid``, or None when it does not exist."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        state = stat.rsplit(")", 1)[1].split()[0]
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "replace")
        return state, cmd
    except (OSError, IndexError):
        pass
    try:  # no /proc (macOS): ask ps
        proc = subprocess.run(["ps", "-ww", "-o", "stat=,command=", "-p", str(pid)],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    line = proc.stdout.strip()
    if proc.returncode != 0 or not line:
        return None
    state, _, cmd = line.partition(" ")
    return state, cmd.strip()


def _worker_pid(run_dir: Path) -> Optional[int]:
    pids = _read_json(run_dir / PIDS_FILE) or {}
    pid = pids.get("worker_pid")
    return pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None


def worker_alive(run_dir: Path) -> bool:
    """Worker pid exists, is not a zombie and its command line names ``run_dir``."""
    pid = _worker_pid(run_dir)
    if pid is None:
        return False
    info = _process_info(pid)
    if info is None:
        return False
    state, cmd = info
    return not state.startswith("Z") and str(run_dir) in cmd


def _signal_group(pid: int, sig: int) -> None:
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, sig)
            return
    except OSError:
        pass
    try:
        os.kill(pid, sig)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_probe(args) -> None:
    ok = subprocess.run([sys.executable, "-c", "import mliprun.webapp.worker"],
                        capture_output=True, timeout=300).returncode == 0
    try:
        from importlib.metadata import version

        mliprun_version = version("mliprun")
    except Exception:  # noqa: BLE001
        mliprun_version = "unknown"
    from mliprun.webapp import calculators

    calculators.detect_devices()
    _out({"agent_version": AGENT_VERSION, "mliprun_version": mliprun_version,
          "python": sys.executable, "hostname": socket.gethostname(),
          "worker_ok": ok, "calculators": calculators.list_calculators()})


def cmd_gpus(args) -> None:
    query = "index,name,memory.used,memory.total,utilization.gpu"
    try:
        proc = subprocess.run(["nvidia-smi", f"--query-gpu={query}",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        _out({"gpus": [], "note": f"nvidia-smi is not available: {exc}"})
        return
    if proc.returncode != 0:
        _out({"gpus": [], "note": proc.stderr.strip() or "nvidia-smi failed"})
        return
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            gpus.append({"index": int(parts[0]), "name": parts[1],
                         "memory_used_mb": float(parts[2]), "memory_total_mb": float(parts[3]),
                         "utilization_pct": float(parts[4])})
        except ValueError:
            continue
    _out({"gpus": gpus, "note": None})


def cmd_inspect(args) -> None:
    _out(paths.check_paths(args.location, list(args.folders), Path.home(),
                           reserved_dirs=None, reserved_names=()))


def cmd_create(args) -> None:
    location = os.path.expanduser(args.location)
    if not os.path.isabs(location) or "\0" in location:
        raise AgentError("the location must be an absolute path (or start with ~)", EXIT_USAGE)
    folder = args.folder
    if not folder or folder != paths.slugify_folder(folder):
        raise AgentError(f"invalid folder name {folder!r}", EXIT_USAGE)
    try:
        os.makedirs(location, exist_ok=True)
    except OSError as exc:
        raise AgentError(f"cannot create the location {location}: {exc}")
    run_dir = Path(os.path.realpath(location)) / folder
    try:
        run_dir.mkdir()
    except FileExistsError:
        raise AgentError(f"{run_dir} already exists", EXIT_EXISTS)
    except OSError as exc:
        raise AgentError(f"cannot create {run_dir}: {exc}")
    try:
        paths.safe_extract_tar(sys.stdin.buffer, run_dir)
    except Exception as exc:  # noqa: BLE001 - tarfile raises many types
        shutil.rmtree(run_dir, ignore_errors=True)
        raise AgentError(f"could not extract the uploaded files: {exc}")
    _out({"run_dir": str(run_dir)})


def cmd_discard(args) -> None:
    run_dir = Path(os.path.realpath(os.path.expanduser(args.run_dir)))
    if not run_dir.is_dir():
        _out({"discarded": False, "reason": "no such directory"})
    elif (run_dir / PIDS_FILE).exists():
        _out({"discarded": False, "reason": "the job was launched"})
    elif not (run_dir / "job.json").is_file():
        _out({"discarded": False, "reason": "not a job directory"})
    else:
        shutil.rmtree(run_dir, ignore_errors=True)
        _out({"discarded": not run_dir.exists()})


def cmd_launch(args) -> None:
    run_dir = _run_dir(args.run_dir)
    pids_path = run_dir / PIDS_FILE
    if not pids_path.exists():
        data = sys.stdin.buffer.read()
        tmp = run_dir / "events.jsonl.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, run_dir / "events.jsonl")
        with open(run_dir / "supervisor.log", "ab") as log:
            subprocess.Popen([sys.executable, "-m", "mliprun.webapp.remote_agent",
                              "supervise", str(run_dir)],
                             cwd=str(run_dir), stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True,
                             close_fds=True)
    deadline = time.monotonic() + LAUNCH_WAIT_S
    while time.monotonic() < deadline:
        pids = _read_json(pids_path)
        if pids is not None:
            _out(pids)
            return
        time.sleep(0.05)
    raise AgentError(f"the worker did not start within {LAUNCH_WAIT_S:.0f} s "
                     f"(see {run_dir / 'supervisor.log'})", EXIT_TIMEOUT)


def _now_iso() -> str:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def cmd_supervise(args) -> None:
    run_dir = _run_dir(args.run_dir)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    override = os.environ.get(WORKER_CMD_ENV)
    cmd = json.loads(override) if override else [sys.executable, "-m", WORKER_MODULE]
    with open(run_dir / "worker.log", "ab") as log:
        proc = subprocess.Popen(cmd + [str(run_dir)], cwd=str(run_dir),
                                stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, env=env, start_new_session=True,
                                close_fds=True)
    _write_json_atomic(run_dir / PIDS_FILE, {
        "supervisor_pid": os.getpid(), "worker_pid": proc.pid,
        "hostname": socket.gethostname(), "started_at": _now_iso()})
    rc = proc.wait()
    _write_json_atomic(run_dir / EXIT_FILE, {"returncode": rc, "finished_at": _now_iso()})


def cmd_poll(args) -> None:
    run_dir = _run_dir(args.run_dir)
    alive = worker_alive(run_dir)
    exit_info = _read_json(run_dir / EXIT_FILE)
    if not alive and exit_info is None:
        # the supervisor writes the exit file just after the worker is reaped
        deadline = time.monotonic() + 1.0
        while exit_info is None and time.monotonic() < deadline:
            time.sleep(0.05)
            exit_info = _read_json(run_dir / EXIT_FILE)
    path = run_dir / "events.jsonl"
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        size = 0
    chunk = b""
    if 0 <= args.offset < size and args.max_bytes > 0:
        with open(path, "rb") as fh:
            fh.seek(args.offset)
            data = fh.read(min(args.max_bytes, size - args.offset))
            end = data.rfind(b"\n")
            while end < 0 and args.offset + len(data) < size and len(data) < (64 << 20):
                more = fh.read(1 << 20)  # one line longer than the window: extend it
                if not more:
                    break
                data += more
                end = data.rfind(b"\n")
        chunk = data[:end + 1] if end >= 0 else b""
    sys.stdout.write(json.dumps({"size": size, "alive": alive, "exit": exit_info,
                                 "chunk_len": len(chunk)}) + "\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(chunk)
    sys.stdout.buffer.flush()


def cmd_cancel(args) -> None:
    run_dir = _run_dir(args.run_dir)
    pid = _worker_pid(run_dir)
    signalled = False
    if pid is not None and worker_alive(run_dir):
        _signal_group(pid, signal.SIGKILL if args.kill else signal.SIGTERM)
        signalled = True
    _out({"signalled": signalled})


def cmd_list(args) -> None:
    _out(paths.list_files(_run_dir(args.run_dir)))


def cmd_pack(args) -> None:
    run_dir = _run_dir(args.run_dir)
    chosen = [f["path"] for f in paths.list_files(run_dir)
              if paths.copy_back_eligible(f["path"], f["size"], args.max_file_bytes)
              and (run_dir / f["path"]).is_file()]
    paths.write_tar(sys.stdout.buffer, run_dir, chosen)
    sys.stdout.buffer.flush()


def cmd_cat(args) -> None:
    run_dir = _run_dir(args.run_dir)
    target = (run_dir / args.relpath).resolve()
    if run_dir not in target.parents:
        raise AgentError("path is outside the run directory", EXIT_NOT_FOUND)
    if not target.is_file():
        raise AgentError(f"file {args.relpath!r} not found", EXIT_NOT_FOUND)
    with open(target, "rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            sys.stdout.buffer.write(block)
    sys.stdout.buffer.flush()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="remote_agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe").set_defaults(func=cmd_probe)
    sub.add_parser("gpus").set_defaults(func=cmd_gpus)
    p = sub.add_parser("inspect")
    p.add_argument("location")
    p.add_argument("folders", nargs="*")
    p.set_defaults(func=cmd_inspect)
    p = sub.add_parser("create")
    p.add_argument("location")
    p.add_argument("folder")
    p.set_defaults(func=cmd_create)
    for name, func in (("discard", cmd_discard), ("launch", cmd_launch),
                       ("supervise", cmd_supervise), ("list", cmd_list)):
        p = sub.add_parser(name)
        p.add_argument("run_dir")
        p.set_defaults(func=func)
    p = sub.add_parser("poll")
    p.add_argument("run_dir")
    p.add_argument("offset", type=int)
    p.add_argument("max_bytes", type=int)
    p.set_defaults(func=cmd_poll)
    p = sub.add_parser("cancel")
    p.add_argument("run_dir")
    p.add_argument("--kill", action="store_true")
    p.set_defaults(func=cmd_cancel)
    p = sub.add_parser("pack")
    p.add_argument("run_dir")
    p.add_argument("max_file_bytes", type=int)
    p.set_defaults(func=cmd_pack)
    p = sub.add_parser("cat")
    p.add_argument("run_dir")
    p.add_argument("relpath")
    p.set_defaults(func=cmd_cat)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run one subcommand; returns the process exit code."""
    try:
        args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:  # argparse usage error
        return EXIT_USAGE if exc.code not in (0, None) else 0
    try:
        args.func(args)
    except AgentError as exc:
        sys.stderr.write(f"{exc}\n")
        return exc.code
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return EXIT_ERROR
    return 0


if __name__ == "__main__":
    sys.exit(main())
