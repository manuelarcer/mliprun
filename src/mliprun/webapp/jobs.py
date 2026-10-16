"""Job registry, state machine, serial queue runner, event tailing, broadcast.

Concurrency model
-----------------
Everything in this module that touches a job record runs on the server's
asyncio event loop thread (route handlers are ``async``; blocking file parsing
is pushed to a threadpool by the server and never mutates records). The
critical sections below contain no ``await``, so on a single-threaded loop
they are atomic: in particular "drain the events file, then register a
WebSocket subscription" cannot interleave with a broadcast, which is what
makes replay-then-live gap-free and duplicate-free.

Exactly one worker subprocess exists at a time: :class:`QueueRunner` spawns a
job's worker only after the previous one has been reaped.

``events.jsonl`` has one writer at a time (PLAN D2): the server appends
``job_status`` lines only while no worker process is alive for the job.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mliprun.webapp import events as wevents
from mliprun.webapp import paths as paths_mod
from mliprun.webapp import remote as wremote
from mliprun.webapp.paths import (  # noqa: F401 - re-exported for existing imports
    MAX_FOLDER_LEN,
    RESERVED_FOLDERS,
    _writable_dir,
    folder_problem,
    inspect_location,
    slugify_folder,
    suffix_width,
    suffixed_folder,
)
from mliprun.webapp.uploads import iso_utc

logger = logging.getLogger(__name__)

JOB_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z]+-[0-9a-f]{6}$")
STATUSES = ("pending", "running", "done", "failed", "cancelled")
TERMINAL = frozenset({"done", "failed", "cancelled"})
LIGHT_TYPES = frozenset({"job_status", "result", "error"})
PROGRESS_TYPES = frozenset({"opt_step", "md_step", "neb_iter", "freq_progress"})
RESTART_REASON = "server restarted while the job was running"
CANCEL_REASON = "cancelled by user"
SHUTDOWN_REASON = "server shut down while the job was running"
WORKER_MODULE = "mliprun.webapp.worker"
JOBS_INDEX_DIR = "_jobs"
LOCAL_KEY = "local"
MAX_POLL_BYTES = 1 << 20

# result_brief keys copied from the `result` event (PLAN 4.5).
RESULT_BRIEF_KEYS = {
    "singlepoint": ("energy_eV", "fmax_free_eV_per_A"),
    "optimize": ("converged", "final_energy_eV", "final_fmax_eV_per_A", "steps"),
    "freq": ("n_imaginary", "zpe_eV"),
    "md": ("mean_temperature_K", "mean_total_energy_eV"),
    "neb": ("forward_barrier_eV", "reverse_barrier_eV", "reaction_energy_eV"),
    "autoneb": ("forward_barrier_eV", "reverse_barrier_eV", "reaction_energy_eV"),
}

# A WebSocket client this far behind is dropped rather than buffered forever.
MAX_OUTBOX = 20000


def default_worker_cmd() -> List[str]:
    """``[sys.executable, "-m", "mliprun.webapp.worker"]``."""
    return [sys.executable, "-m", WORKER_MODULE]


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001 - missing or unreadable
        return None
    return data if isinstance(data, dict) else None


def write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON to ``path`` via a temp file and ``os.replace``."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def pid_alive(pid: int) -> bool:
    """True if ``pid`` exists and is not a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        state = stat.rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except (OSError, IndexError):
        return True


def _signal_group(pid: int, sig: int) -> None:
    """Send ``sig`` to the process group led by ``pid`` (fallback: the pid)."""
    try:
        if hasattr(os, "killpg") and os.getpgid(pid) == pid:
            os.killpg(pid, sig)
            return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


class FolderExists(Exception):
    """The run folder of a job being created already exists."""


class LocationUnusable(Exception):
    """The location of a job being created cannot be created or written."""


# --------------------------------------------------------------------------- #
# Broadcaster
# --------------------------------------------------------------------------- #
class Connection:
    """One WebSocket client: its outbox and subscriptions."""

    def __init__(self) -> None:
        self.outbox: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        # job_id -> minimum live seq delivered for that per-job subscription
        self.subs: Dict[str, int] = {}
        self.all_jobs = False
        # (job_id, seq) of events delivered through the "*" subscription
        self.light_sent: Dict[str, set] = {}
        self.closed = False

    def put(self, text: Optional[str]) -> None:
        if self.closed:
            return
        if text is not None and self.outbox.qsize() >= MAX_OUTBOX:
            self.closed = True
            self.outbox.put_nowait(None)
            return
        self.outbox.put_nowait(text)


class Broadcaster:
    """Fans events out to subscribed WebSocket connections (at most once each)."""

    def __init__(self) -> None:
        self.connections: set = set()

    def add(self, conn: Connection) -> None:
        self.connections.add(conn)

    def remove(self, conn: Connection) -> None:
        self.connections.discard(conn)

    def publish(self, job_id: str, seq: int, etype: str, text: str) -> None:
        for conn in list(self.connections):
            min_seq = conn.subs.get(job_id)
            if min_seq is not None and seq >= min_seq:
                conn.put(text)
            elif conn.all_jobs and etype in LIGHT_TYPES:
                conn.light_sent.setdefault(job_id, set()).add(seq)
                conn.put(text)

    def close_all(self) -> None:
        for conn in list(self.connections):
            conn.put(None)


# --------------------------------------------------------------------------- #
# Job record
# --------------------------------------------------------------------------- #
class JobRecord:
    """In-memory view of one job (``job.json`` + ``state.json`` + events)."""

    def __init__(self, run_dir: Path, job: dict, state: dict, order: int):
        self.run_dir = Path(run_dir)
        self.job = job
        self.state = state
        self.order = order
        self.proc: Optional[subprocess.Popen] = None
        self.log_fh = None
        # tailer
        self.offset = 0
        self.lines_seen = 0
        self.progress_event: Optional[dict] = None
        self.result: Optional[dict] = None
        self.last_error: Optional[str] = None
        # cancellation
        self.cancel_reason: Optional[str] = None
        self.kill_deadline: Optional[float] = None
        self.killed = False
        # remote jobs: a worker is active on another machine (the server must not
        # write events.jsonl); cancel delivery state
        self.remote_active = False
        self.cancel_sent = False
        # cached for JobSummary / JobDetail
        self.formula: Optional[str] = None
        self.natoms: Optional[int] = None
        self.inputs_detail: Optional[dict] = None

    @property
    def job_id(self) -> str:
        return self.job["job_id"]

    @property
    def status(self) -> str:
        return self.state["status"]

    @property
    def folder(self) -> str:
        return self.job.get("folder") or self.run_dir.name

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def target(self) -> dict:
        """``{"kind": "local"}`` or the remote target block of ``job.json``."""
        t = self.job.get("target")
        return t if isinstance(t, dict) and t.get("kind") == "remote" else {"kind": "local"}

    @property
    def queue_key(self) -> str:
        """``"local"`` or the machine id: jobs with the same key share one serial queue."""
        t = self.target
        return t.get("machine_id") or LOCAL_KEY if t["kind"] == "remote" else LOCAL_KEY

    @property
    def is_remote(self) -> bool:
        return self.queue_key != LOCAL_KEY

    # ------------------------------------------------------------- derived
    def progress(self) -> Optional[dict]:
        e = self.progress_event
        if not e:
            return None
        params = self.job.get("params") or {}
        etype = e.get("type")
        step = total = energy = fmax = None
        text = ""
        if etype == "opt_step":
            step, total = e.get("step"), params.get("max_steps")
            energy, fmax = e.get("energy"), e.get("fmax")
            text = f"step {step}/{total}" if total is not None else f"step {step}"
        elif etype == "md_step":
            step, total, energy = e.get("step"), params.get("steps"), e.get("etot")
            text = f"step {step}/{total}" if total is not None else f"step {step}"
        elif etype == "neb_iter":
            step = e.get("iteration")
            total = params.get("neb_max_steps") if self.job.get("calc_type") == "neb" else None
            energy, fmax = e.get("barrier"), e.get("fmax")
            text = f"iteration {step}/{total}" if total is not None else f"iteration {step}"
        elif etype == "freq_progress":
            step, total = e.get("done"), e.get("total")
            text = f"{step}/{total} force calls"
        return {"step": step, "total": total, "energy": energy, "fmax": fmax,
                "text": text}

    def result_brief(self) -> Optional[dict]:
        if self.result is None:
            return self.state.get("result_brief")
        keys = RESULT_BRIEF_KEYS.get(self.job.get("calc_type"), ())
        return {k: self.result[k] for k in keys if k in self.result}

    def error(self) -> Optional[str]:
        if self.status == "failed" and self.state.get("reason"):
            return self.state["reason"]
        return self.last_error

    def summary(self, queue_position: Optional[int]) -> dict:
        st = self.state
        calc = self.job.get("calculator") or {}
        return {
            "job_id": self.job_id,
            "calc_type": self.job.get("calc_type"),
            "label": self.job.get("label"),
            "folder": self.folder,
            "run_dir": str(self.run_dir),
            "status": st["status"],
            "created_at": st.get("created_at") or self.job.get("created_at"),
            "started_at": st.get("started_at"),
            "finished_at": st.get("finished_at"),
            "calculator": {"name": calc.get("name"), "device": calc.get("device")},
            "formula": self.formula,
            "natoms": self.natoms,
            "target": {"kind": self.target["kind"],
                       "machine_name": self.target.get("machine_name"),
                       "env_name": self.target.get("env_name")},
            "remote_run_dir": self.job.get("remote_run_dir"),
            "remote_error": (st.get("remote") or {}).get("error"),
            "queue_position": queue_position,
            "cancel_requested": bool(st.get("cancel_requested")),
            "progress": self.progress(),
            "result_brief": self.result_brief(),
            "error": self.error(),
        }


# --------------------------------------------------------------------------- #
# Registry (+ tailer)
# --------------------------------------------------------------------------- #
class JobRegistry:
    """All jobs of a runs directory, in memory and on disk.

    Parameters
    ----------
    runs_dir : Path
        Resolved runs directory.
    broadcaster : Broadcaster
        Receives every event read from any job's events file.
    describe_inputs : callable
        ``describe_inputs(job_dict) -> (formula, natoms)`` for JobSummary.
    cancel_grace_s : float
        SIGTERM -> SIGKILL grace, also used when killing an orphan at startup.
    """

    def __init__(self, runs_dir: Path, broadcaster: Broadcaster,
                 describe_inputs: Callable[[dict], tuple],
                 cancel_grace_s: float = 10.0):
        self.runs_dir = Path(runs_dir)
        self.broadcaster = broadcaster
        self.describe_inputs = describe_inputs
        self.cancel_grace_s = float(cancel_grace_s)
        self.jobs: Dict[str, JobRecord] = {}
        self.pending: List[str] = []           # FIFO of pending job ids (all queues)
        self.reattach: List[str] = []          # running remote jobs found at startup
        # ``machine_exists(machine_id) -> bool``; set by the server
        self.machine_exists: Optional[Callable[[str], bool]] = None
        self._order = 0
        self._last_created: Optional[datetime] = None
        self.on_enqueue: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------- queries
    def get(self, job_id: str) -> Optional[JobRecord]:
        if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
            return None
        return self.jobs.get(job_id)

    def queue_position(self, job: JobRecord) -> Optional[int]:
        if job.status != "pending":
            return None
        key = job.queue_key
        ahead = 0
        for job_id in self.pending:
            rec = self.jobs.get(job_id)
            if rec is None or rec.queue_key != key:
                continue
            ahead += 1
            if job_id == job.job_id:
                return ahead
        return None

    def summary(self, job: JobRecord) -> dict:
        out = job.summary(self.queue_position(job))
        if (job.is_remote and not job.terminal and self.machine_exists is not None
                and not self.machine_exists(job.queue_key)):
            name = job.target.get("machine_name") or job.queue_key
            out["remote_error"] = f"machine {name} no longer configured"
        return out

    def list(self, statuses: Optional[set] = None, limit: int = 500) -> List[JobRecord]:
        recs = sorted(self.jobs.values(), key=lambda r: r.order, reverse=True)
        if statuses:
            recs = [r for r in recs if r.status in statuses]
        return recs[:limit]

    def running_job(self) -> Optional[JobRecord]:
        """The running local job (compatibility); see :meth:`running_jobs`."""
        for rec in self.jobs.values():
            if rec.status == "running" and not rec.is_remote:
                return rec
        return None

    def running_jobs(self) -> List[JobRecord]:
        """Every running job, local and remote."""
        return [r for r in self.jobs.values() if r.status == "running"]

    def has_pending(self, queue_key: str) -> bool:
        return any((rec := self.jobs.get(j)) is not None and rec.status == "pending"
                   and rec.queue_key == queue_key for j in self.pending)

    def queue_keys(self) -> set:
        """Queue keys that have a pending or reattachable job."""
        keys = {self.jobs[j].queue_key for j in self.pending + self.reattach if j in self.jobs}
        return keys

    # ------------------------------------------------------------- ids/time
    def _next_created(self) -> datetime:
        """Creation time, strictly increasing (ms resolution) so FIFO survives restarts."""
        now = datetime.now(timezone.utc)
        now = now.replace(microsecond=(now.microsecond // 1000) * 1000)
        if self._last_created is not None and now <= self._last_created:
            now = self._last_created + timedelta(milliseconds=1)
        self._last_created = now
        return now

    def new_job_id(self, calc_type: str, created: datetime) -> str:
        stamp = created.strftime("%Y%m%d-%H%M%S")
        while True:
            job_id = f"{stamp}-{calc_type}-{secrets.token_hex(3)}"
            if (job_id not in self.jobs and not (self.runs_dir / job_id).exists()
                    and not self._index_path(job_id).exists()):
                return job_id

    # ------------------------------------------------------------- index
    @property
    def index_dir(self) -> Path:
        return self.runs_dir / JOBS_INDEX_DIR

    def _index_path(self, job_id: str) -> Path:
        return self.index_dir / f"{job_id}.json"

    def _write_index(self, job_id: str, run_dir: Path) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self._index_path(job_id), {"job_id": job_id, "run_dir": str(run_dir)})

    # ------------------------------------------------------------- persistence
    def write_state(self, job: JobRecord) -> None:
        write_json_atomic(job.run_dir / "state.json", job.state)

    def _emit_status(self, job: JobRecord) -> None:
        """Append a job_status line (only when no worker is alive) and broadcast it."""
        if job.proc is not None or job.remote_active:
            return
        writer = wevents.EventWriter(job.events_path, job.job_id)
        try:
            writer.emit("job_status", status=job.status, reason=job.state.get("reason"),
                        returncode=job.state.get("returncode"))
        finally:
            writer.close()
        self.drain(job)

    def transition(self, job: JobRecord, status: str, *, reason: Optional[str] = None,
                   returncode: Optional[int] = None) -> None:
        """Move ``job`` to ``status``: memory, state.json, job_status event, broadcast."""
        if job.terminal:
            raise RuntimeError(f"job {job.job_id} is already {job.status}")
        now = iso_utc()
        st = job.state
        st["status"] = status
        st["reason"] = reason
        st["returncode"] = returncode
        if status == "running":
            st["started_at"] = now
        if status in TERMINAL:
            st["finished_at"] = now
            st["result_brief"] = job.result_brief()
            if job.job_id in self.pending:
                self.pending.remove(job.job_id)
        self.write_state(job)
        self._emit_status(job)

    # ------------------------------------------------------------- tailer
    def drain(self, job: JobRecord, broadcast: bool = True) -> int:
        """Read new complete lines of ``job``'s events file; apply and broadcast.

        Returns the number of lines consumed.
        """
        path = job.events_path
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return 0
        if size <= job.offset:
            return 0
        with open(path, "rb") as fh:
            fh.seek(job.offset)
            data = fh.read(size - job.offset)
        end = data.rfind(b"\n")
        if end < 0:
            return 0
        chunk = data[:end + 1]
        job.offset += end + 1
        lines = chunk.split(b"\n")[:-1]
        for raw in lines:
            seq = job.lines_seen
            job.lines_seen += 1
            try:
                text = raw.decode("utf-8")
                event = json.loads(text)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            self._apply(job, event)
            if broadcast:
                self.broadcaster.publish(job.job_id, seq, str(event.get("type")), text)
        return len(lines)

    @staticmethod
    def _apply(job: JobRecord, event: dict) -> None:
        etype = event.get("type")
        if etype in PROGRESS_TYPES:
            job.progress_event = event
        elif etype == "result":
            job.result = event
        elif etype == "error":
            msg = event.get("message")
            job.last_error = str(msg) if msg is not None else None

    def _load_events(self, job: JobRecord) -> None:
        """Rebuild tailer state from a whole events file without parsing every line."""
        try:
            data = job.events_path.read_bytes()
        except FileNotFoundError:
            return
        end = data.rfind(b"\n")
        if end < 0:
            return
        lines = data[:end + 1].split(b"\n")[:-1]
        tail_start = max(0, len(lines) - 200)
        for i, raw in enumerate(lines):
            if i < tail_start and b'"result"' not in raw and b'"error"' not in raw:
                continue
            try:
                event = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(event, dict):
                self._apply(job, event)
        job.offset = end + 1
        job.lines_seen = len(lines)

    # ------------------------------------------------------------- creation
    def create_job(self, calc_type: str, label: str, calculator: dict, params: dict,
                   param_sources: dict, structure_every: int,
                   inputs: Dict[str, dict], folder: Optional[str] = None,
                   location: Optional[Path] = None, target: Optional[dict] = None,
                   remote_location: Optional[str] = None) -> JobRecord:
        """Create a pending job: run_dir, input copies, job.json, state.json, seq 0.

        ``inputs`` maps role -> ``{"upload_id", "filename", "format", "source"}``
        where ``source`` is the stored upload path to copy. ``run_dir`` is
        ``<location>/<folder>`` (defaults: the runs dir and the job id); it
        must not exist (:class:`FolderExists`), it is never overwritten.
        For a remote ``target`` (``{"kind": "remote", ...}``) ``run_dir`` is
        the local copy; ``remote_location`` is where the remote run dir goes
        (job.json then has schema_version 2 and the remote keys).
        Raises :class:`LocationUnusable` if the location cannot be created.
        """
        created = self._next_created()
        created_at = iso_utc(created)
        job_id = self.new_job_id(calc_type, created)
        location = Path(location) if location is not None else self.runs_dir
        folder = folder or job_id
        try:
            location.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LocationUnusable(f"cannot create the location {location}: {exc}") from exc
        if not location.is_dir() or not _writable_dir(location):
            raise LocationUnusable(f"the location {location} is not a writable directory")
        run_dir = location / folder
        try:
            run_dir.mkdir()
        except FileExistsError as exc:
            raise FolderExists(f"{run_dir} already exists") from exc
        except OSError as exc:
            raise LocationUnusable(f"cannot create {run_dir}: {exc}") from exc
        try:
            job_inputs = {}
            for role, spec in inputs.items():
                rel = f"inputs/{role}/{spec['filename']}"
                (run_dir / "inputs" / role).mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(spec["source"]), str(run_dir / rel))
                job_inputs[role] = {"upload_id": spec["upload_id"],
                                    "filename": spec["filename"],
                                    "path": rel, "format": spec.get("format")}
            remote = isinstance(target, dict) and target.get("kind") == "remote"
            job = {
                "schema_version": 2 if remote else 1,
                "job_id": job_id,
                "calc_type": calc_type,
                "label": label,
                "folder": folder,
                "location": str(location),
                "created_at": created_at,
                "calculator": calculator,
                "params": params,
                "param_sources": param_sources,
                "structure_every": structure_every,
                "inputs": job_inputs,
            }
            if remote:
                job["target"] = target
                job["location"] = remote_location
                job["local_location"] = str(location)
                job["remote_run_dir"] = None
            write_json_atomic(run_dir / "job.json", job)
            state = {"job_id": job_id, "status": "pending", "created_at": created_at,
                     "started_at": None, "finished_at": None, "pid": None,
                     "returncode": None, "cancel_requested": False, "reason": None,
                     "result_brief": None}
            self._order += 1
            rec = JobRecord(run_dir, job, state, self._order)
            rec.formula, rec.natoms = self.describe_inputs(job)
            self.write_state(rec)
            self._write_index(job_id, run_dir)
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            try:
                self._index_path(job_id).unlink()
            except OSError:
                pass
            raise
        self.jobs[job_id] = rec
        self._emit_status(rec)
        return rec

    def discard_job(self, job: JobRecord) -> None:
        """Remove a just-created job entirely (used to roll back a failed multi-submit)."""
        self.jobs.pop(job.job_id, None)
        if job.job_id in self.pending:
            self.pending.remove(job.job_id)
        shutil.rmtree(job.run_dir, ignore_errors=True)
        try:
            self._index_path(job.job_id).unlink()
        except OSError:
            pass

    def set_remote_run_dir(self, job: JobRecord, remote_run_dir: str) -> None:
        """Record where the remote run dir really is (after the agent created it)."""
        job.job["remote_run_dir"] = remote_run_dir
        write_json_atomic(job.run_dir / "job.json", job.job)

    def enqueue(self, job: JobRecord) -> None:
        self.pending.append(job.job_id)
        if self.on_enqueue is not None:
            self.on_enqueue()

    def pop_next_pending(self, queue_key: str = LOCAL_KEY) -> Optional[JobRecord]:
        """Remove and return the first pending job of ``queue_key`` (FIFO within the key)."""
        for i, job_id in enumerate(self.pending):
            rec = self.jobs.get(job_id)
            if rec is None or rec.status != "pending":
                continue
            if rec.queue_key == queue_key:
                del self.pending[i]
                return rec
        self.pending = [j for j in self.pending
                        if (r := self.jobs.get(j)) is not None and r.status == "pending"]
        return None

    def take_reattach(self, queue_key: str) -> Optional[JobRecord]:
        """A running remote job of ``queue_key`` left by a previous server, or None."""
        for i, job_id in enumerate(self.reattach):
            rec = self.jobs.get(job_id)
            if rec is not None and rec.queue_key == queue_key and rec.status == "running":
                del self.reattach[i]
                return rec
        return None

    # ------------------------------------------------------------- startup
    def load(self) -> None:
        """Load every job and apply restart recovery (PLAN 10.1, D6, 15.1).

        Jobs come from the index ``<runs-dir>/_jobs/<job_id>.json`` (their
        run_dir may be anywhere) plus old-layout ``<runs-dir>/<job_id>/`` dirs
        that have no index entry.
        """
        found = []
        seen: set = set()

        def consider(job_id: str, d: Path, strict: bool = True) -> None:
            state = _read_json(d / "state.json")
            job = _read_json(d / "job.json")
            if state is None or job is None or state.get("status") not in STATUSES:
                logger.warning("ignoring %s: missing or unreadable state.json/job.json", d)
                return
            job.setdefault("job_id", job_id)
            if not strict:
                job_id = job["job_id"]
            if job["job_id"] != job_id or job_id in seen:
                logger.warning("ignoring %s: job id mismatch or duplicate (%s)", d, job_id)
                return
            seen.add(job_id)
            found.append((state.get("created_at") or job.get("created_at") or "", job_id,
                          d, job, state))

        if self.index_dir.is_dir():
            for entry_path in sorted(self.index_dir.glob("*.json")):
                entry = _read_json(entry_path)
                job_id = (entry or {}).get("job_id") or entry_path.stem
                run_dir = (entry or {}).get("run_dir")
                if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id) \
                        or not isinstance(run_dir, str) or not run_dir:
                    logger.warning("ignoring unreadable index entry %s", entry_path)
                    continue
                d = Path(run_dir)
                if not d.is_dir() or not (d / "state.json").is_file():
                    logger.warning("skipping job %s: %s or its state.json is missing",
                                   job_id, d)
                    continue
                consider(job_id, d)
        for d in sorted(self.runs_dir.iterdir()):
            if not d.is_dir() or not JOB_ID_RE.match(d.name) or d.name in seen:
                continue
            consider(d.name, d, strict=False)
        found.sort(key=lambda t: (t[0], t[1]))
        for created_at, _, d, job, state in found:
            self._order += 1
            rec = JobRecord(d, job, state, self._order)
            try:
                rec.formula, rec.natoms = self.describe_inputs(job)
            except Exception:  # noqa: BLE001 - summary fields are optional
                rec.formula, rec.natoms = None, None
            self._load_events(rec)
            self.jobs[rec.job_id] = rec
            ts = _parse_iso(created_at)
            if ts is not None and (self._last_created is None or ts > self._last_created):
                self._last_created = ts
        for rec in sorted(self.jobs.values(), key=lambda r: r.order):
            if rec.status == "running" and rec.is_remote:
                # the remote worker keeps running; its runner resumes polling
                rec.remote_active = True
                self.reattach.append(rec.job_id)
            elif rec.status == "running":
                self._kill_orphan(rec)
                write_backend_summary(rec, "failed", RESTART_REASON)
                self.transition(rec, "failed", reason=RESTART_REASON,
                                returncode=rec.state.get("returncode"))
            elif rec.status == "pending":
                self.pending.append(rec.job_id)

    def _kill_orphan(self, rec: JobRecord) -> None:
        pid = rec.state.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or not pid_alive(pid):
            return
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace")
        except OSError:
            return
        if WORKER_MODULE not in cmdline or str(rec.run_dir) not in cmdline:
            return
        logger.warning("killing orphan worker pid %s of job %s", pid, rec.job_id)
        _signal_group(pid, signal.SIGTERM)
        deadline = time.monotonic() + self.cancel_grace_s
        while time.monotonic() < deadline and pid_alive(pid):
            time.sleep(0.05)
        if pid_alive(pid):
            _signal_group(pid, signal.SIGKILL)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and pid_alive(pid):
                time.sleep(0.05)


def write_backend_summary(job: JobRecord, status: str, reason: Optional[str]) -> None:
    """Write ``summary.json`` for a worker that died without one."""
    path = job.run_dir / "summary.json"
    if _read_json(path) is not None:
        return
    started = _parse_iso(job.state.get("started_at"))
    finished = datetime.now(timezone.utc)
    summary = {
        "job_id": job.job_id,
        "calc_type": job.job.get("calc_type"),
        "status": status,
        "started_at": job.state.get("started_at"),
        "finished_at": iso_utc(finished),
        "walltime_s": (finished - started).total_seconds() if started else None,
        "result": None,
        "error": {"message": reason, "exception_type": None} if reason else None,
        "reproduce": None,
        "written_by": "server",
    }
    try:
        write_json_atomic(path, summary)
    except OSError as exc:
        logger.warning("could not write %s: %s", path, exc)


# --------------------------------------------------------------------------- #
# Queue runner
# --------------------------------------------------------------------------- #
class QueueRunner:
    """Runs pending jobs strictly one at a time, one worker subprocess each.

    Parameters
    ----------
    registry : JobRegistry
    worker_cmd : list of str
        Command prefix; the run_dir is appended as the last argv element.
    cancel_grace_s : float
        Seconds between SIGTERM and SIGKILL on cancel.
    poll_interval_s : float
        How often the running worker's events file is tailed.
    """

    def __init__(self, registry: JobRegistry, worker_cmd: List[str],
                 cancel_grace_s: float = 10.0, poll_interval_s: float = 0.2,
                 queue_key: str = LOCAL_KEY):
        self.registry = registry
        self.queue_key = queue_key
        self.worker_cmd = list(worker_cmd)
        self.cancel_grace_s = float(cancel_grace_s)
        self.poll_interval_s = float(poll_interval_s)
        self.current: Optional[JobRecord] = None
        self.stopping = False
        self._task: Optional[asyncio.Task] = None
        self._wakeup: Optional[asyncio.Event] = None
        registry.on_enqueue = self.notify

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Start the runner task on the running event loop."""
        self._wakeup = asyncio.Event()
        self._task = asyncio.get_running_loop().create_task(self._run())

    def notify(self) -> None:
        if self._wakeup is not None:
            self._wakeup.set()

    async def stop(self) -> None:
        """Cancel the running job (SIGTERM, grace, SIGKILL), finalize it, stop."""
        self.stopping = True
        if self.current is not None:
            self.request_cancel(self.current, SHUTDOWN_REASON)
        self.notify()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._task),
                                   timeout=self.cancel_grace_s + 15.0)
        except asyncio.TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001 - the runner task failed; nothing to wait for
            logger.exception("queue runner ended with an error")

    async def _run(self) -> None:
        while not self.stopping:
            job = (self.registry.take_reattach(self.queue_key)
                   or self.registry.pop_next_pending(self.queue_key))
            if job is None:
                self._wakeup.clear()
                if self.registry.has_pending(self.queue_key):
                    continue
                await self._wakeup.wait()
                continue
            try:
                await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a bug here must not kill the queue
                logger.exception("unexpected error while running job %s", job.job_id)
                if not job.terminal:
                    job.proc = None
                    job.remote_active = False
                    self.registry.transition(job, "failed", reason="internal server error")

    # ------------------------------------------------------------- cancel
    def request_cancel(self, job: JobRecord, reason: str = CANCEL_REASON) -> None:
        """Cancel ``job``: pending -> cancelled now; running -> SIGTERM + deadline."""
        reg = self.registry
        if job.terminal:
            return
        if job.status == "pending":
            job.state["cancel_requested"] = True
            reg.transition(job, "cancelled", reason=reason)
            return
        if job.state.get("cancel_requested"):
            return
        job.state["cancel_requested"] = True
        job.cancel_reason = reason
        reg.write_state(job)
        if job.proc is not None and job.proc.poll() is None:
            _signal_group(job.proc.pid, signal.SIGTERM)
        job.kill_deadline = time.monotonic() + self.cancel_grace_s

    # ------------------------------------------------------------- one job
    def _spawn(self, job: JobRecord) -> subprocess.Popen:
        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"
        env["PYTHONUNBUFFERED"] = "1"
        job.log_fh = open(job.run_dir / "worker.log", "ab")
        try:
            return subprocess.Popen(
                self.worker_cmd + [str(job.run_dir)], cwd=str(job.run_dir),
                stdin=subprocess.DEVNULL, stdout=job.log_fh, stderr=subprocess.STDOUT,
                env=env, start_new_session=True, close_fds=True)
        except Exception:
            job.log_fh.close()
            job.log_fh = None
            raise

    async def _run_job(self, job: JobRecord) -> None:
        reg = self.registry
        self.current = job
        try:
            reg.transition(job, "running")       # no worker yet: server writes the line
            try:
                proc = self._spawn(job)
            except Exception as exc:  # noqa: BLE001
                reason = f"could not start the worker: {exc}"
                write_backend_summary(job, "failed", reason)
                reg.transition(job, "failed", reason=reason)
                return
            job.proc = proc
            job.state["pid"] = proc.pid
            reg.write_state(job)
            if job.state.get("cancel_requested"):
                _signal_group(proc.pid, signal.SIGTERM)
                if job.kill_deadline is None:
                    job.kill_deadline = time.monotonic() + self.cancel_grace_s
            try:
                while True:
                    rc = proc.poll()
                    reg.drain(job)
                    if rc is not None:
                        break
                    if (job.kill_deadline is not None and not job.killed
                            and time.monotonic() >= job.kill_deadline):
                        _signal_group(proc.pid, signal.SIGKILL)
                        job.killed = True
                    await asyncio.sleep(self.poll_interval_s)
            except asyncio.CancelledError:
                _signal_group(proc.pid, signal.SIGKILL)
                proc.wait()
                self._finish(job, proc.returncode)
                raise
            self._finish(job, rc)
        finally:
            self.current = None

    def _finish(self, job: JobRecord, rc: Optional[int]) -> None:
        reg = self.registry
        if job.log_fh is not None:
            job.log_fh.close()
            job.log_fh = None
        reg.drain(job)                # final drain after the process is reaped
        job.proc = None
        summary = _read_json(job.run_dir / "summary.json")
        if job.state.get("cancel_requested"):
            status, reason = "cancelled", job.cancel_reason or CANCEL_REASON
        elif rc == 0 and summary is not None and summary.get("status") == "done":
            status, reason = "done", None
        else:
            status = "failed"
            if job.last_error:
                reason = job.last_error
            elif rc is not None and rc < 0:
                reason = f"worker killed by signal {-rc}"
            elif rc == 0:
                reason = "worker exited with code 0 without reporting a result"
            else:
                reason = f"worker exited with code {rc}"
        if summary is None:
            write_backend_summary(job, status, reason if status != "done" else None)
        reg.transition(job, status, reason=reason, returncode=rc)


# --------------------------------------------------------------------------- #
# Remote queue runner
# --------------------------------------------------------------------------- #
class RemoteQueueRunner(QueueRunner):
    """The serial queue of one remote machine (REMOTE_PLAN section 6).

    A job's worker runs on the machine through the remote agent; this runner
    mirrors the remote ``events.jsonl`` into the local run dir by polling, so
    every reader (WebSocket, files, summary) keeps working on local files.
    An unreachable machine never fails a running job: polling backs off and
    resumes. Stopping the runner (server shutdown) leaves the remote job
    running; the next server reattaches to it.

    Parameters
    ----------
    registry : JobRegistry
    machine_id : str
        The queue key.
    store : MachineStore
    transport : SSHTransport
    cancel_grace_s : float
        Seconds between the remote SIGTERM and SIGKILL on cancel.
    backoff_base, backoff_cap : float
        Reconnect back-off: ``base``, ``2*base``, ... up to ``cap`` seconds.
    """

    def __init__(self, registry: JobRegistry, machine_id: str, store: "wremote.MachineStore",
                 transport: "wremote.SSHTransport", cancel_grace_s: float = 10.0,
                 backoff_base: float = 2.0, backoff_cap: float = 60.0):
        super().__init__(registry, [], cancel_grace_s=cancel_grace_s, queue_key=machine_id)
        self.store = store
        self.transport = transport
        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)

    async def stop(self) -> None:
        """Stop the runner task only; remote jobs are not cancelled."""
        self.stopping = True
        self.notify()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    # ------------------------------------------------------------- helpers
    def _poll_interval(self, target) -> float:
        try:
            return float(target.machine.get("poll_interval_s") or 2.0)
        except (TypeError, ValueError):
            return 2.0

    async def _sleep(self, job: JobRecord, delay: float) -> None:
        """Sleep ``delay`` s in short slices; wake early to deliver a cancel or stop."""
        end = time.monotonic() + delay
        while not self.stopping:
            now = time.monotonic()
            if now >= end:
                return
            if job.state.get("cancel_requested") and (
                    not job.cancel_sent
                    or (not job.killed and job.kill_deadline is not None
                        and now >= job.kill_deadline)):
                return
            await asyncio.sleep(min(0.1, end - now))

    def _set_error(self, job: JobRecord, message: str) -> None:
        remote = job.state.setdefault("remote", {})
        if remote.get("error") is None or remote.get("_msg") != message:
            since = remote.get("_since") or iso_utc()
            remote.update({"error": f"{message} (since {since})", "_msg": message,
                           "_since": since})
            self.registry.write_state(job)

    def _clear_error(self, job: JobRecord) -> None:
        remote = job.state.setdefault("remote", {})
        remote["last_ok_at"] = iso_utc()
        if remote.get("error") is not None:
            remote.update({"error": None, "_msg": None, "_since": None})
            self.registry.write_state(job)

    async def _deliver_cancel(self, job: JobRecord, target, run_dir: str) -> None:
        if not job.state.get("cancel_requested"):
            return
        try:
            if not job.cancel_sent:
                await asyncio.to_thread(self.transport.run_json, target, "cancel", [run_dir])
                job.cancel_sent = True
                job.kill_deadline = time.monotonic() + self.cancel_grace_s
            elif (not job.killed and job.kill_deadline is not None
                  and time.monotonic() >= job.kill_deadline):
                await asyncio.to_thread(self.transport.run_json, target, "cancel",
                                        [run_dir, "--kill"])
                job.killed = True
        except wremote.RemoteError as exc:
            self._set_error(job, exc.message)      # remembered; retried on the next pass

    # ------------------------------------------------------------- one job
    async def _run_job(self, job: JobRecord) -> None:
        reg = self.registry
        self.current = job
        resumed = job.status == "running"
        try:
            info = job.target
            try:
                target = self.store.get_env(info["env_id"])
            except (wremote.MachineNotFound, KeyError):
                reason = (f"environment {info.get('env_name')} of machine "
                          f"{info.get('machine_name')} is no longer configured")
                write_backend_summary(job, "failed", reason)
                job.remote_active = False
                if job.status == "pending":
                    reg.transition(job, "running")
                reg.transition(job, "failed", reason=reason)
                return
            run_dir = job.job.get("remote_run_dir")
            if not resumed:
                reg.transition(job, "running")       # no remote worker yet: server writes it
            if not (job.state.get("remote") or {}).get("launched"):
                if not await self._launch(job, target, run_dir, resumed):
                    return
            job.remote_active = True
            rc, reason = await self._mirror(job, target, run_dir)
            if rc is _STOPPED:
                return
            await asyncio.to_thread(self._copy_back, job, target, run_dir)
            self._finish_remote(job, rc, reason)
        finally:
            self.current = None

    def _fail_launch(self, job: JobRecord, message: str) -> None:
        reason = f"could not start the remote worker: {message}"
        job.remote_active = False
        write_backend_summary(job, "failed", reason)
        self.registry.transition(job, "failed", reason=reason)

    async def _launch(self, job: JobRecord, target, run_dir: str, resumed: bool) -> bool:
        """Start the remote worker; False when the job failed (or the runner stopped)."""
        failures = 0
        while not self.stopping:
            events = job.events_path.read_bytes()
            try:
                pids = await asyncio.to_thread(self.transport.run_json, target, "launch",
                                               [run_dir], events, 90.0)
            except wremote.RemoteUnreachable as exc:
                if not resumed:
                    self._fail_launch(job, exc.message)
                    return False
                failures += 1
                self._set_error(job, exc.message)
                await self._sleep(job, wremote.backoff_delay(failures, self.backoff_base,
                                                             self.backoff_cap))
                continue
            except wremote.RemoteError as exc:
                self._fail_launch(job, exc.message)
                return False
            remote = job.state.setdefault("remote", {})
            remote.update({"pids": pids, "launched": True, "error": None,
                           "last_ok_at": iso_utc()})
            reg.write_state(job)
            return True
        return False

    async def _mirror(self, job: JobRecord, target, run_dir: str) -> tuple:
        """Poll until the remote worker has exited and the mirror is complete.

        Returns ``(returncode or None, reason)``; ``(_STOPPED, None)`` when the
        runner was stopped first.
        """
        reg = self.registry
        failures = 0
        while True:
            if self.stopping:
                return _STOPPED, None
            await self._deliver_cancel(job, target, run_dir)
            offset = job.events_path.stat().st_size if job.events_path.exists() else 0
            try:
                header, chunk = await asyncio.to_thread(
                    self.transport.poll, target, run_dir, offset, MAX_POLL_BYTES)
            except wremote.RemoteError as exc:
                failures += 1
                self._set_error(job, exc.message)
                await self._sleep(job, wremote.backoff_delay(failures, self.backoff_base,
                                                             self.backoff_cap))
                continue
            failures = 0
            self._clear_error(job)
            if chunk:
                with open(job.events_path, "ab") as fh:
                    fh.write(chunk)
                reg.drain(job)
            caught_up = offset + len(chunk) >= header["size"]
            exit_info = header.get("exit")
            if exit_info is not None and caught_up:
                return exit_info.get("returncode"), None
            if not header.get("alive") and exit_info is None and caught_up:
                return None, "remote worker disappeared"
            if chunk and not caught_up:
                continue                               # more is waiting: poll again at once
            await self._sleep(job, self._poll_interval(target))

    def _copy_back(self, job: JobRecord, target, run_dir: str) -> None:
        """Copy small files back and write ``remote_files.json`` (never raises)."""
        max_bytes = int(float(target.machine.get("copy_back_max_mb") or 0) * 1024 * 1024)
        try:
            if max_bytes > 0:
                stream = self.transport.stream(target, "pack", [run_dir, max_bytes])
                try:
                    paths_mod.safe_extract_tar(stream.stdout, job.run_dir)
                    stream.finish()
                finally:
                    stream.close()
            listing = self.transport.run_json(target, "list", [run_dir], timeout=120.0)
            files = [{"path": f["path"], "size": f["size"],
                      "copied": (job.run_dir / f["path"]).is_file()} for f in listing]
            write_json_atomic(job.run_dir / "remote_files.json", {
                "remote_host": target.machine["host"], "remote_run_dir": run_dir,
                "files": files})
        except Exception as exc:  # noqa: BLE001 - a log line, not a job failure
            logger.warning("copy-back of job %s failed: %s", job.job_id, exc)
            job.state.setdefault("remote", {})["copy_back_error"] = str(exc)

    def _finish_remote(self, job: JobRecord, rc: Optional[int], reason: Optional[str]) -> None:
        reg = self.registry
        reg.drain(job)
        job.remote_active = False
        summary = _read_json(job.run_dir / "summary.json")
        if job.state.get("cancel_requested"):
            status, reason = "cancelled", job.cancel_reason or CANCEL_REASON
        elif rc == 0 and ((summary is not None and summary.get("status") == "done")
                          or (summary is None and job.result is not None)):
            status, reason = "done", None
        else:
            status = "failed"
            if reason:
                pass
            elif job.last_error:
                reason = job.last_error
            elif rc is not None and rc < 0:
                reason = f"worker killed by signal {-rc}"
            elif rc == 0:
                reason = "worker exited with code 0 without reporting a result"
            else:
                reason = f"worker exited with code {rc}"
        if summary is None:
            write_backend_summary(job, status, reason if status != "done" else None)
        reg.transition(job, status, reason=reason, returncode=rc)


_STOPPED = object()
