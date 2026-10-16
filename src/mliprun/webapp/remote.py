"""Remote machines for the web app: the machines file and the SSH transport.

``MachineStore`` keeps the user's machines and Python environments in a JSON
file (REMOTE_PLAN section 1). ``SSHTransport`` runs the remote agent
(``mliprun.webapp.remote_agent``) through the system ``ssh`` binary
(section 2). Every value that reaches an ``ssh`` argv is validated here
(rejected, never escaped) and every remote shell word goes through
``shlex.quote``.

All calls are blocking: the async server runs them in a thread.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shlex
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Sequence

from mliprun.webapp import paths

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_MACHINES_FILE = "~/.mliprun/webapp-machines.json"
AGENT_MODULE = "mliprun.webapp.remote_agent"
PASSWORD_HINT = ("SSH needs key-based login (or an open ControlMaster); passwords cannot "
                 "be typed by the web app")
TOO_OLD_HINT = ("mliprun on {machine} is too old for remote runs: update it (git pull) "
                "in that environment")

NAME_RE = re.compile(r"^[A-Za-z0-9._+-]{1,60}$")
HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
USER_RE = re.compile(r"^[A-Za-z0-9._-]+$")
JUMP_RE = re.compile(r"^([A-Za-z0-9._-]+@)?[A-Za-z0-9][A-Za-z0-9._-]*(:[0-9]+)?$")
MACHINE_FIELDS = ("name", "host", "user", "port", "identity_file", "jump_host",
                  "default_location", "poll_interval_s", "copy_back_max_mb")
ENV_FIELDS = ("name", "python", "setup")
MAX_SETUP_LEN = 4000


def _now_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class MachineValidationError(Exception):
    """Invalid machine/env settings; ``details`` is a list of ``{"field", "message"}``."""

    def __init__(self, details: List[dict]):
        super().__init__("; ".join(f"{d['field']}: {d['message']}" for d in details))
        self.details = details


class MachineNotFound(KeyError):
    """Unknown machine or environment id."""


class RemoteError(Exception):
    """A remote command failed (non-zero exit that is not an SSH failure)."""

    def __init__(self, message: str, returncode: Optional[int] = None, stderr: str = ""):
        super().__init__(message)
        self.message = message
        self.returncode = returncode
        self.stderr = stderr


class RemoteUnreachable(RemoteError):
    """ssh itself failed (exit 255) or timed out: the host cannot be reached."""


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def default_env_name(machine_name: str, python: str) -> str:
    """``<machine name>-<venv dir>``: ``/x/.venv/mace/bin/python`` -> ``<machine>-mace``.

    When the python path has no ``bin`` parent, the python file's parent
    directory name is used instead.
    """
    p = PurePosixPath(python)
    parent = p.parent
    venv = parent.parent.name if parent.name == "bin" and parent.parent.name else parent.name
    venv = re.sub(r"[^A-Za-z0-9._+-]", "-", venv) or "python"
    return f"{machine_name}-{venv}"[:60]


def _validate_machine_fields(data: dict, errors: List[dict]) -> dict:
    out: Dict[str, Any] = {}

    def err(field: str, message: str) -> None:
        errors.append({"field": field, "message": message})

    name = data.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        err("name", "required: 1-60 characters from A-Z a-z 0-9 . _ + -")
    else:
        out["name"] = name
    host = data.get("host")
    if not isinstance(host, str) or not HOST_RE.match(host):
        err("host", "required: a host name or address (letters, digits, . _ -), "
                    "not starting with - or .")
    else:
        out["host"] = host
    user = data.get("user")
    if user is None or user == "":
        out["user"] = None
    elif not isinstance(user, str) or not USER_RE.match(user):
        err("user", "letters, digits, . _ - only")
    else:
        out["user"] = user
    port = data.get("port")
    if port is None or port == "":
        out["port"] = None
    elif isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        err("port", "must be an integer from 1 to 65535")
    else:
        out["port"] = port
    ident = data.get("identity_file")
    if ident is None or ident == "":
        out["identity_file"] = None
    elif (not isinstance(ident, str) or not (ident.startswith("/") or ident.startswith("~"))
          or "\n" in ident or "\0" in ident or "\r" in ident):
        err("identity_file", "must be an absolute path (or start with ~), on one line")
    else:
        out["identity_file"] = ident
    jump = data.get("jump_host")
    if jump is None or jump == "":
        out["jump_host"] = None
    elif not isinstance(jump, str) or not JUMP_RE.match(jump):
        err("jump_host", "must look like [user@]host[:port]")
    else:
        out["jump_host"] = jump
    loc = data.get("default_location")
    if (not isinstance(loc, str) or not (loc.startswith("/") or loc.startswith("~"))
            or "\n" in loc or "\0" in loc):
        err("default_location", "required: an absolute remote path (or starting with ~)")
    else:
        out["default_location"] = loc
    poll = data.get("poll_interval_s", 2.0)
    if poll is None:
        poll = 2.0
    if isinstance(poll, bool) or not isinstance(poll, (int, float)) or not 0.5 <= poll <= 60:
        err("poll_interval_s", "must be a number from 0.5 to 60 (seconds)")
    else:
        out["poll_interval_s"] = float(poll)
    cb = data.get("copy_back_max_mb", 50)
    if cb is None:
        cb = 50
    if isinstance(cb, bool) or not isinstance(cb, (int, float)) or not 0 <= cb <= 10000:
        err("copy_back_max_mb", "must be a number from 0 to 10000 (MB)")
    else:
        out["copy_back_max_mb"] = int(cb) if float(cb).is_integer() else float(cb)
    return out


def _validate_env_fields(data: dict, machine_name: str, errors: List[dict],
                         existing_name: Optional[str] = None) -> dict:
    out: Dict[str, Any] = {}

    def err(field: str, message: str) -> None:
        errors.append({"field": field, "message": message})

    python = data.get("python")
    if (not isinstance(python, str) or not python.startswith("/")
            or re.search(r"\s", python) or "\0" in python):
        err("python", "required: an absolute path to the Python interpreter, "
                      "without spaces")
    else:
        out["python"] = python
    name = data.get("name")
    if name is None or name == "":
        name = (existing_name or (default_env_name(machine_name, python)
                                  if "python" in out else None))
    if not isinstance(name, str) or not NAME_RE.match(name):
        err("name", "1-60 characters from A-Z a-z 0-9 . _ + -")
    else:
        out["name"] = name
    setup = data.get("setup", "")
    if setup is None:
        setup = ""
    if not isinstance(setup, str) or len(setup) > MAX_SETUP_LEN or "\0" in setup:
        err("setup", f"must be text of at most {MAX_SETUP_LEN} characters")
    else:
        out["setup"] = setup.replace("\r\n", "\n")
    return out


# --------------------------------------------------------------------------- #
# Machines file
# --------------------------------------------------------------------------- #
class MachineStore:
    """The machines file: machines with their environments, loaded and saved whole.

    Thread-safe (the server mutates it from request threads). Every change is
    written atomically (temp file + ``os.replace``, mode 0600).
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path if path is not None else DEFAULT_MACHINES_FILE).expanduser()
        self._lock = threading.RLock()
        self.machines: List[dict] = []
        self.load()

    # ------------------------------------------------------------- persistence
    def load(self) -> None:
        with self._lock:
            self.machines = []
            try:
                raw = self.path.read_text()
            except FileNotFoundError:
                return
            except OSError as exc:
                logger.warning("cannot read the machines file %s: %s", self.path, exc)
                return
            try:
                data = json.loads(raw)
                machines = data["machines"]
                if not isinstance(machines, list):
                    raise ValueError("machines is not a list")
            except (ValueError, KeyError, TypeError) as exc:
                backup = self.path.with_name(self.path.name + ".bad")
                logger.warning("machines file %s is unreadable (%s); keeping a copy as %s "
                               "and starting empty", self.path, exc, backup)
                try:
                    os.replace(self.path, backup)
                except OSError:
                    pass
                return
            for m in machines:
                if isinstance(m, dict) and isinstance(m.get("id"), str):
                    m.setdefault("envs", [])
                    self.machines.append(m)

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                json.dump({"schema_version": SCHEMA_VERSION, "machines": self.machines},
                          fh, indent=1)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)

    # ------------------------------------------------------------- queries
    def list(self) -> List[dict]:
        with self._lock:
            return json.loads(json.dumps(self.machines))

    def get_machine(self, mid: str) -> dict:
        with self._lock:
            for m in self.machines:
                if m["id"] == mid:
                    return json.loads(json.dumps(m))
        raise MachineNotFound(mid)

    def get_env(self, eid: str) -> "RemoteTarget":
        with self._lock:
            for m in self.machines:
                for e in m.get("envs", []):
                    if e["id"] == eid:
                        return RemoteTarget(json.loads(json.dumps(m)),
                                            json.loads(json.dumps(e)))
        raise MachineNotFound(eid)

    # ------------------------------------------------------------- ids
    def _new_id(self, prefix: str) -> str:
        used = {m["id"] for m in self.machines}
        used |= {e["id"] for m in self.machines for e in m.get("envs", [])}
        while True:
            candidate = f"{prefix}-{secrets.token_hex(3)}"
            if candidate not in used:
                return candidate

    # ------------------------------------------------------------- machines
    def _check_machine_name(self, name: Optional[str], errors: List[dict],
                            skip_id: Optional[str] = None) -> None:
        if name is None:
            return
        for m in self.machines:
            if m["id"] != skip_id and m["name"] == name:
                errors.append({"field": "name",
                               "message": f"a machine named {name!r} already exists"})

    def add_machine(self, data: dict) -> dict:
        errors: List[dict] = []
        if not isinstance(data, dict):
            raise MachineValidationError([{"field": "request", "message": "must be an object"}])
        for key in sorted(set(data) - set(MACHINE_FIELDS)):
            errors.append({"field": key, "message": "unknown key"})
        with self._lock:
            clean = _validate_machine_fields(data, errors)
            self._check_machine_name(clean.get("name"), errors)
            if errors:
                raise MachineValidationError(errors)
            machine = {"id": self._new_id("m"), **clean, "envs": []}
            self.machines.append(machine)
            self.save()
            return json.loads(json.dumps(machine))

    def update_machine(self, mid: str, data: dict) -> dict:
        if not isinstance(data, dict):
            raise MachineValidationError([{"field": "request", "message": "must be an object"}])
        errors: List[dict] = []
        for key in sorted(set(data) - set(MACHINE_FIELDS)):
            errors.append({"field": key, "message": "unknown key"})
        with self._lock:
            machine = self._find(mid)
            merged = {k: machine.get(k) for k in MACHINE_FIELDS}
            merged.update({k: v for k, v in data.items() if k in MACHINE_FIELDS})
            clean = _validate_machine_fields(merged, errors)
            self._check_machine_name(clean.get("name"), errors, skip_id=mid)
            if errors:
                raise MachineValidationError(errors)
            machine.update(clean)
            self.save()
            return json.loads(json.dumps(machine))

    def _find(self, mid: str) -> dict:
        for m in self.machines:
            if m["id"] == mid:
                return m
        raise MachineNotFound(mid)

    def delete_machine(self, mid: str) -> None:
        with self._lock:
            machine = self._find(mid)
            self.machines.remove(machine)
            self.save()

    # ------------------------------------------------------------- envs
    def _all_env_names(self, skip_eid: Optional[str] = None) -> set:
        return {e["name"] for m in self.machines for e in m.get("envs", [])
                if e["id"] != skip_eid}

    def add_env(self, mid: str, data: dict, probe: Optional[dict] = None) -> dict:
        if not isinstance(data, dict):
            raise MachineValidationError([{"field": "request", "message": "must be an object"}])
        errors: List[dict] = []
        for key in sorted(set(data) - set(ENV_FIELDS)):
            errors.append({"field": key, "message": "unknown key"})
        with self._lock:
            machine = self._find(mid)
            clean = _validate_env_fields(data, machine["name"], errors)
            if clean.get("name") in self._all_env_names():
                errors.append({"field": "name", "message":
                               f"an environment named {clean['name']!r} already exists"})
            if errors:
                raise MachineValidationError(errors)
            env = {"id": self._new_id("e"), **clean, "probe": probe}
            machine.setdefault("envs", []).append(env)
            self.save()
            return json.loads(json.dumps(env))

    def update_env(self, mid: str, eid: str, data: dict) -> dict:
        if not isinstance(data, dict):
            raise MachineValidationError([{"field": "request", "message": "must be an object"}])
        errors: List[dict] = []
        for key in sorted(set(data) - set(ENV_FIELDS)):
            errors.append({"field": key, "message": "unknown key"})
        with self._lock:
            machine = self._find(mid)
            env = self._find_env(machine, eid)
            merged = {k: env.get(k) for k in ENV_FIELDS}
            merged.update({k: v for k, v in data.items() if k in ENV_FIELDS})
            clean = _validate_env_fields(merged, machine["name"], errors)
            if clean.get("name") in self._all_env_names(skip_eid=eid):
                errors.append({"field": "name", "message":
                               f"an environment named {clean['name']!r} already exists"})
            if errors:
                raise MachineValidationError(errors)
            if clean["python"] != env["python"]:
                env["probe"] = None                  # a different interpreter: probe is stale
            env.update(clean)
            self.save()
            return json.loads(json.dumps(env))

    @staticmethod
    def _find_env(machine: dict, eid: str) -> dict:
        for e in machine.get("envs", []):
            if e["id"] == eid:
                return e
        raise MachineNotFound(eid)

    def delete_env(self, mid: str, eid: str) -> None:
        with self._lock:
            machine = self._find(mid)
            machine["envs"].remove(self._find_env(machine, eid))
            self.save()

    def set_probe(self, mid: str, eid: str, probe: dict) -> dict:
        with self._lock:
            env = self._find_env(self._find(mid), eid)
            env["probe"] = probe
            self.save()
            return json.loads(json.dumps(env))


# --------------------------------------------------------------------------- #
# SSH transport
# --------------------------------------------------------------------------- #
@dataclass
class RemoteTarget:
    """One environment together with the machine it lives on."""

    machine: dict
    env: dict

    @property
    def python(self) -> str:
        return self.env["python"]


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _error_from(proc_rc: int, stderr: bytes, what: str) -> RemoteError:
    text = stderr.decode("utf-8", "replace")
    last = _last_line(text)
    if proc_rc == 255:
        msg = last or "ssh could not connect"
        if re.search(r"permission denied|password|authentication", text, re.I):
            msg = f"{msg}. {PASSWORD_HINT}"
        return RemoteUnreachable(msg, proc_rc, text)
    return RemoteError(last or f"{what} failed with exit code {proc_rc}", proc_rc, text)


class RemoteStream:
    """A running remote command whose stdout is read incrementally."""

    def __init__(self, proc: subprocess.Popen, errfile, what: str):
        self.proc = proc
        self._err = errfile
        self.what = what

    @property
    def stdout(self) -> BinaryIO:
        return self.proc.stdout

    def iter_chunks(self, size: int = 1 << 20) -> Iterator[bytes]:
        while True:
            block = self.proc.stdout.read(size)
            if not block:
                return
            yield block

    def finish(self) -> None:
        """Drain what is left, wait, and raise if the command failed."""
        try:
            while self.proc.stdout.read(1 << 20):
                pass
        except (OSError, ValueError):
            pass
        rc = self.proc.wait()
        self._err.seek(0)
        stderr = self._err.read()
        self._err.close()
        if rc != 0:
            raise _error_from(rc, stderr, self.what)

    def close(self) -> None:
        """Stop the command (if still running) and release the pipes; never raises."""
        try:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            self.proc.stdout.close()
            self._err.close()
        except Exception:  # noqa: BLE001
            pass


class SSHTransport:
    """Runs remote-agent subcommands over the system ``ssh``.

    Parameters
    ----------
    ssh_command : list of str, optional
        ``ssh`` binary plus any fixed leading arguments (tests pass
        ``[sys.executable, "fake_ssh.py"]``). Default ``["ssh"]``.
    ctl_dir : Path, optional
        Directory of the ControlMaster sockets. Default
        ``/tmp/mliprun-ssh-<uid>`` (short: unix socket paths are limited to
        about 104 characters on macOS).
    """

    def __init__(self, ssh_command: Optional[Sequence[str]] = None,
                 ctl_dir: Optional[Path] = None):
        self.ssh_command = list(ssh_command) if ssh_command else ["ssh"]
        self.ctl_dir = Path(ctl_dir) if ctl_dir else Path(f"/tmp/mliprun-ssh-{os.getuid()}")

    # ------------------------------------------------------------- argv
    def _ensure_ctl_dir(self) -> None:
        self.ctl_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self.ctl_dir, 0o700)
        except OSError:
            pass

    def ssh_argv(self, machine: dict, remote_command: str) -> List[str]:
        """The full local argv: options, ``--``, host, remote command."""
        self._ensure_ctl_dir()
        argv = self.ssh_command + [
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-o", "ControlMaster=auto", "-o", f"ControlPath={self.ctl_dir}/%C",
            "-o", "ControlPersist=300"]
        if machine.get("port"):
            argv += ["-p", str(machine["port"])]
        if machine.get("identity_file"):
            argv += ["-i", str(machine["identity_file"])]
        if machine.get("jump_host"):
            argv += ["-J", machine["jump_host"]]
        if machine.get("user"):
            argv += ["-l", machine["user"]]
        return argv + ["--", machine["host"], remote_command]

    @staticmethod
    def remote_command(target: RemoteTarget, subcmd: str, args: Sequence[Any] = ()) -> str:
        """``/bin/sh -c '<setup>\\nexec <python> -m <agent> <subcmd> <args>'`` (all quoted)."""
        words = " ".join(shlex.quote(str(a)) for a in args)
        script = ((target.env.get("setup") or "").rstrip("\n") + "\n"
                  + f"exec {shlex.quote(target.python)} -m {AGENT_MODULE} "
                  + shlex.quote(subcmd) + (" " + words if words else ""))
        return "/bin/sh -c " + shlex.quote(script)

    # ------------------------------------------------------------- calls
    def run(self, target: RemoteTarget, subcmd: str, args: Sequence[Any] = (),
            stdin: Optional[bytes] = None, timeout: float = 60.0) -> subprocess.CompletedProcess:
        """Run one agent subcommand; return the completed process (bytes stdout).

        Raises
        ------
        RemoteUnreachable
            ssh exit code 255 or a timeout.
        RemoteError
            Any other non-zero exit; the message is the agent's last stderr line.
        """
        argv = self.ssh_argv(target.machine, self.remote_command(target, subcmd, args))
        try:
            proc = subprocess.run(argv, input=stdin if stdin is not None else b"",
                                  capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RemoteUnreachable(f"no answer from {target.machine['host']} "
                                    f"within {timeout:g} s") from None
        except OSError as exc:
            raise RemoteError(f"cannot start ssh: {exc}") from exc
        if proc.returncode != 0:
            raise _error_from(proc.returncode, proc.stderr, subcmd)
        return proc

    def run_json(self, target: RemoteTarget, subcmd: str, args: Sequence[Any] = (),
                 stdin: Optional[bytes] = None, timeout: float = 60.0) -> Any:
        """:meth:`run`, parsing the single JSON object on stdout."""
        proc = self.run(target, subcmd, args, stdin=stdin, timeout=timeout)
        try:
            return json.loads(proc.stdout.decode("utf-8"))
        except ValueError:
            raise RemoteError(f"{subcmd}: unreadable answer from the remote agent: "
                              f"{proc.stdout[:200]!r}") from None

    def run_raw(self, machine: dict, command: str, timeout: float = 30.0) -> str:
        """Run a fixed shell command (connection test only); return stdout text."""
        argv = self.ssh_argv(machine, command)
        try:
            proc = subprocess.run(argv, input=b"", capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RemoteUnreachable(f"no answer from {machine['host']} "
                                    f"within {timeout:g} s") from None
        except OSError as exc:
            raise RemoteError(f"cannot start ssh: {exc}") from exc
        if proc.returncode != 0:
            raise _error_from(proc.returncode, proc.stderr, command)
        return proc.stdout.decode("utf-8", "replace")

    def poll(self, target: RemoteTarget, run_dir: str, offset: int, max_bytes: int,
             timeout: float = 60.0) -> tuple:
        """``agent poll``: returns ``(header dict, chunk bytes)``."""
        proc = self.run(target, "poll", [run_dir, offset, max_bytes], timeout=timeout)
        head, sep, chunk = proc.stdout.partition(b"\n")
        try:
            header = json.loads(head.decode("utf-8"))
            n = int(header["chunk_len"])
        except (ValueError, KeyError, TypeError):
            raise RemoteError("poll: unreadable answer from the remote agent") from None
        if len(chunk) != n:
            raise RemoteError(f"poll: expected {n} bytes, got {len(chunk)}")
        return header, chunk

    def stream(self, target: RemoteTarget, subcmd: str, args: Sequence[Any] = ()) -> RemoteStream:
        """Start ``subcmd`` and return its stdout as a stream (``cat`` / ``pack``)."""
        argv = self.ssh_argv(target.machine, self.remote_command(target, subcmd, args))
        errfile = tempfile.TemporaryFile()
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=errfile)
        except OSError as exc:
            errfile.close()
            raise RemoteError(f"cannot start ssh: {exc}") from exc
        return RemoteStream(proc, errfile, subcmd)


def probe_target(transport: SSHTransport, target: RemoteTarget) -> dict:
    """Probe one environment; ``{"ok": True, ...probe..., "probed_at"}`` or
    ``{"ok": False, "error", "probed_at"}`` (never raises)."""
    try:
        data = transport.run_json(target, "probe", timeout=300.0)
    except RemoteUnreachable as exc:
        return {"ok": False, "error": exc.message, "probed_at": _now_iso()}
    except RemoteError as exc:
        text = f"{exc.message}\n{exc.stderr}"
        if "No module named" in text and "mliprun" in text:
            message = TOO_OLD_HINT.format(machine=target.machine["name"])
        else:
            message = exc.message
        return {"ok": False, "error": message, "probed_at": _now_iso()}
    if not isinstance(data, dict) or "agent_version" not in data:
        return {"ok": False, "probed_at": _now_iso(),
                "error": TOO_OLD_HINT.format(machine=target.machine["name"])}
    if not data.get("worker_ok"):
        return {"ok": False, "probed_at": _now_iso(),
                "error": "mliprun is installed but the worker cannot be imported "
                         "(missing dependency, e.g. ase) in this environment"}
    return {"ok": True, "probed_at": _now_iso(), **data}


def tar_local_inputs(run_dir: Path) -> bytes:
    """Tar of ``job.json`` plus everything under ``inputs/`` of a local run dir."""
    rels = ["job.json"]
    inputs = Path(run_dir) / "inputs"
    for f in paths.list_files(inputs) if inputs.is_dir() else []:
        rels.append(f"inputs/{f['path']}")
    return paths.tar_bytes(run_dir, rels)


def backoff_delay(failures: int, base: float = 2.0, cap: float = 60.0) -> float:
    """Exponential back-off: ``base``, 2*base, 4*base, ... capped at ``cap``."""
    return min(cap, base * (2 ** max(failures - 1, 0)))
