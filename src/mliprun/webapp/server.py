"""FastAPI application of the mliprun web app: REST routes, WebSocket, security.

Build it with :func:`create_app`. The server never runs a calculation itself:
each job is a separate worker subprocess launched by :class:`QueueRunner`, one
at a time. Blocking work done for a request (parsing an upload with ASE,
listing or zipping a run directory) runs in a threadpool.
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, Query, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from mliprun.webapp import calculators as wcalculators
from mliprun.webapp import events as wevents
from mliprun.webapp import paths as wpaths
from mliprun.webapp import remote as wremote
from mliprun.webapp import schema as wschema
from mliprun.webapp.errors import ApiError, code_for_status, error_body, validation_error
from mliprun.webapp.jobs import (
    JOBS_INDEX_DIR,
    TERMINAL,
    STATUSES,
    Broadcaster,
    Connection,
    FolderExists,
    JobRecord,
    JobRegistry,
    LocationUnusable,
    QueueRunner,
    RemoteQueueRunner,
    default_worker_cmd,
    folder_problem,
    inspect_location,
    slugify_folder,
    suffix_width,
    suffixed_folder,
)
from mliprun.webapp.uploads import (
    MAX_FILES_PER_REQUEST,
    UploadStore,
    describe_file,
    iso_utc,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
MAX_INLINE_BYTES = 5 * 1024 * 1024
MAX_LISTED_FILES = 10000
WS_PING_INTERVAL_S = 20.0

JOB_KEYS = frozenset({"calc_type", "calculator", "params", "structure_every", "label",
                      "location", "folder", "target", "local_location",
                      "structure_ids", "initial_id", "final_id", "band_id"})
ID_KEYS = ("structure_ids", "initial_id", "final_id", "band_id")
MAX_STRUCTURES_PER_SUBMIT = 200
MAX_LABEL_LEN = 80
MAX_PATH_LEN = 4096
MAX_CHECK_FOLDERS = 1000

_TEXT_TYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
               ".html": "text/html", ".json": "application/json", ".svg": "image/svg+xml"}


def _mliprun_version() -> str:
    try:
        from importlib.metadata import version

        return version("mliprun")
    except Exception:  # noqa: BLE001
        return "unknown"


def _json(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status)


# --------------------------------------------------------------------------- #
# Security middleware (PLAN D5)
# --------------------------------------------------------------------------- #
def _split_host(netloc: str, default_port: Optional[int]) -> tuple:
    netloc = (netloc or "").strip().lower()
    if netloc.startswith("["):
        host, _, rest = netloc[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else ""
    elif netloc.count(":") == 1:
        host, _, port = netloc.partition(":")
    else:
        host, port = netloc, ""
    try:
        port_num = int(port) if port else default_port
    except ValueError:
        port_num = -1
    return host, port_num


def origin_matches_host(origin: str, host_header: str) -> bool:
    """True if the ``Origin`` header's host:port equals the request ``Host``."""
    parts = urlsplit(origin or "")
    if not parts.netloc:
        return False
    default = {"http": 80, "ws": 80, "https": 443, "wss": 443}.get(parts.scheme.lower())
    o_host, o_port = _split_host(parts.netloc, default)
    h_host, h_port = _split_host(host_header, default)
    return bool(o_host) and o_host == h_host and o_port == h_port


class SecurityMiddleware:
    """Host allow-list (HTTP + WebSocket) and ``X-Mliprun: 1`` on mutating requests."""

    def __init__(self, app, allowed_hosts: Optional[List[str]] = None):
        self.app = app
        self.allowed = (None if allowed_hosts is None
                        else {h.strip().lower() for h in allowed_hosts})

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers") or []}
        if self.allowed is not None and headers.get("host", "").lower() not in self.allowed:
            await self._reject(scope, receive, send, "Host header not allowed")
            return
        if (scope["type"] == "http" and scope.get("method", "").upper() in MUTATING_METHODS
                and headers.get("x-mliprun") != "1"):
            await self._reject(scope, receive, send, "missing X-Mliprun: 1 header")
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(scope, receive, send, message: str) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        response = JSONResponse(error_body("forbidden", message), status_code=403)
        await response(scope, receive, send)


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(runs_dir: Path, *, start_runner: bool = True,
               allowed_hosts: Optional[List[str]] = None,
               worker_cmd: Optional[List[str]] = None,
               cancel_grace_s: float = 10.0,
               poll_interval_s: float = 0.2,
               machines_file: Optional[Path] = None,
               ssh_command: Optional[List[str]] = None,
               remote_backoff_base: float = 2.0) -> FastAPI:
    """Build the web app.

    Parameters
    ----------
    runs_dir : Path
        Where uploads and run directories live (created; ``~`` expanded).
    start_runner : bool
        Start the queue runner with the app lifespan. ``False`` keeps jobs
        pending (useful in tests).
    allowed_hosts : list of str, optional
        Accepted ``Host`` header values (``"127.0.0.1:8765"``). ``None``
        disables the Host check.
    worker_cmd : list of str, optional
        Worker command prefix; the run_dir is appended. Default
        ``[sys.executable, "-m", "mliprun.webapp.worker"]``.
    cancel_grace_s : float
        Seconds between SIGTERM and SIGKILL when cancelling.
    poll_interval_s : float
        Tail interval of the running job's ``events.jsonl``.
    machines_file : Path, optional
        JSON file with the user's SSH machines and environments (default
        ``~/.mliprun/webapp-machines.json``).
    ssh_command : list of str, optional
        ``ssh`` binary plus fixed arguments (default ``["ssh"]``); tests pass
        a fake.
    remote_backoff_base : float
        First reconnect delay (s) after an SSH failure during a remote job;
        doubles up to 60 s.

    Returns
    -------
    FastAPI
        The application. Recovery of jobs left by a previous server (D6)
        has already been applied when this returns.
    """
    runs_dir = Path(runs_dir).expanduser()
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = runs_dir.resolve()

    uploads = UploadStore(runs_dir)
    broadcaster = Broadcaster()

    def describe_inputs(job: dict) -> tuple:
        inputs = job.get("inputs") or {}
        for role in ("structure", "initial", "band"):
            spec = inputs.get(role)
            if spec:
                meta = uploads.get(spec.get("upload_id"))
                if meta:
                    return meta.get("formula"), meta.get("natoms")
        return None, None

    store = wremote.MachineStore(
        machines_file if machines_file is not None else wremote.DEFAULT_MACHINES_FILE)
    transport = wremote.SSHTransport(ssh_command)
    registry = JobRegistry(runs_dir, broadcaster, describe_inputs, cancel_grace_s)
    registry.machine_exists = lambda mid: any(m["id"] == mid for m in store.machines)
    registry.load()
    runner = QueueRunner(registry, worker_cmd or default_worker_cmd(),
                         cancel_grace_s=cancel_grace_s, poll_interval_s=poll_interval_s)
    runners: Dict[str, QueueRunner] = {"local": runner}
    loop_started = {"yes": False}

    def notify_all() -> None:
        for r in list(runners.values()):
            r.notify()

    registry.on_enqueue = notify_all

    def ensure_runner(key: str) -> Optional[QueueRunner]:
        """The queue runner of ``key`` (created on demand; None if the machine is gone)."""
        r = runners.get(key)
        if r is not None or key == "local":
            return r
        if not registry.machine_exists(key):
            return None
        r = RemoteQueueRunner(registry, key, store, transport, cancel_grace_s=cancel_grace_s,
                              backoff_base=remote_backoff_base)
        registry.on_enqueue = notify_all
        runners[key] = r
        if loop_started["yes"] and start_runner:
            r.start()
        return r

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threading.Thread(target=_probe_devices, name="mliprun-device-probe",
                         daemon=True).start()
        if start_runner:
            runner.start()
            for key in sorted(registry.queue_keys()):
                if key != "local":
                    ensure_runner(key)
            for r in runners.values():
                if r is not runner:
                    r.start()
            loop_started["yes"] = True
        try:
            yield
        finally:
            if start_runner:
                for r in list(runners.values()):
                    await r.stop()
            broadcaster.close_all()

    app = FastAPI(title="mliprun web app", version=_mliprun_version(), lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.runs_dir = runs_dir
    app.state.registry = registry
    app.state.runner = runner
    app.state.runners = runners
    app.state.machines = store
    app.state.transport = transport
    app.state.uploads = uploads
    app.state.broadcaster = broadcaster
    app.add_middleware(SecurityMiddleware, allowed_hosts=allowed_hosts)
    _install_error_handlers(app)

    def get_job(job_id: str) -> JobRecord:
        rec = registry.get(job_id)
        if rec is None:
            raise ApiError(404, f"unknown job {job_id!r}")
        return rec

    async def _remote_file_response(rec: JobRecord, entry: dict, inline: int):
        """Serve a file that stayed on the remote machine, streamed through ``cat``."""
        name = entry["path"]
        try:
            rtarget = store.get_env(rec.target["env_id"])
        except (wremote.MachineNotFound, KeyError):
            raise ApiError(409, "the machine of this job is no longer configured; "
                                f"{name!r} exists only on it")
        remote_dir = rec.job.get("remote_run_dir")
        if inline and (entry.get("size") or 0) > MAX_INLINE_BYTES:
            raise ApiError(413, f"file is larger than {MAX_INLINE_BYTES // (1024 * 1024)} MB; "
                                "download it instead")

        def _open():
            stream = transport.stream(rtarget, "cat", [remote_dir, name])
            try:
                first = stream.stdout.read(1 << 20)
                if not first:
                    stream.finish()
            except wremote.RemoteUnreachable as exc:
                stream.close()
                raise ApiError(502, exc.message, "remote_unreachable")
            except wremote.RemoteError as exc:
                stream.close()
                if exc.returncode == 4:
                    raise ApiError(404, f"file {name!r} not found on the remote machine")
                raise ApiError(502, exc.message, "remote_error")
            return stream, first

        stream, first = await run_in_threadpool(_open)
        if inline:
            def _all() -> bytes:
                try:
                    chunks = [first] + list(stream.iter_chunks())
                    stream.finish()
                finally:
                    stream.close()
                return b"".join(chunks)

            data = await run_in_threadpool(_all)
            if b"\x00" not in data:
                try:
                    return PlainTextResponse(data.decode("utf-8"),
                                             media_type="text/plain; charset=utf-8")
                except UnicodeDecodeError:
                    pass
            first_chunks = [data]
            stream = None
        else:
            first_chunks = [first]

        def _stream():
            try:
                yield from first_chunks
                if stream is not None:
                    yield from stream.iter_chunks()
            finally:
                if stream is not None:
                    stream.close()

        media = mimetypes.guess_type(name)[0] or "application/octet-stream"
        base = Path(name).name.replace('"', "")
        return StreamingResponse(_stream(), media_type=media,
                                 headers={"Content-Disposition": f'attachment; filename="{base}"'})

    def get_machine(mid: str) -> dict:
        try:
            return store.get_machine(mid)
        except wremote.MachineNotFound:
            raise ApiError(404, f"unknown machine {mid!r}")

    def unfinished(pred) -> List[JobRecord]:
        return [r for r in registry.jobs.values() if not r.terminal and pred(r)]

    def field_errors(exc: wremote.MachineValidationError) -> ApiError:
        return validation_error(exc.details)

    # ------------------------------------------------------------------ static
    @app.get("/", include_in_schema=False)
    async def index():
        path = STATIC_DIR / "index.html"
        if not path.is_file():
            raise ApiError(404, "index.html not found (static files missing)")
        return FileResponse(path, media_type="text/html",
                            headers={"Cache-Control": "no-cache"})

    @app.get("/static/{path:path}", include_in_schema=False)
    async def static_file(path: str):
        root = STATIC_DIR.resolve()
        target = (root / path).resolve()
        if root not in target.parents or not target.is_file():
            raise ApiError(404, f"static file {path!r} not found")
        media = _TEXT_TYPES.get(target.suffix.lower()) or mimetypes.guess_type(str(target))[0]
        return FileResponse(target, media_type=media or "application/octet-stream",
                            headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    async def health():
        running = registry.running_job()
        return {"status": "ok", "mliprun_version": _mliprun_version(),
                "runs_dir": str(runs_dir), "queue_length": len(registry.pending),
                "running_job": running.job_id if running else None,
                "running_jobs": [r.job_id for r in registry.running_jobs()]}

    @app.get("/api/calc-types")
    async def calc_types():
        return wschema.get_calc_types()

    @app.get("/api/calculators")
    async def calculators(target: Optional[str] = None):
        if target is None or target in ("", "local"):
            return await run_in_threadpool(wcalculators.list_calculators)
        try:
            env = store.get_env(target)
        except wremote.MachineNotFound:
            raise ApiError(404, f"unknown target {target!r}")
        probe = env.env.get("probe")
        if not isinstance(probe, dict) or not probe.get("ok"):
            detail = (probe or {}).get("error") if isinstance(probe, dict) else None
            raise ApiError(409, f"environment {env.env['name']} has not been probed "
                                f"successfully" + (f": {detail}" if detail else "")
                                + "; probe it in the Machines panel")
        return probe["calculators"]

    # ------------------------------------------------------------------ uploads
    @app.post("/api/uploads")
    async def post_uploads(request: Request):
        try:
            form = await request.form(max_files=1000, max_fields=1000)
        except Exception as exc:  # noqa: BLE001 - malformed multipart body
            raise ApiError(400, f"malformed multipart body: {exc}")
        try:
            files = [v for k, v in form.multi_items()
                     if k == "files" and hasattr(v, "filename") and hasattr(v, "file")]
            fmt_value = form.get("format")
            fmt = fmt_value.strip() if isinstance(fmt_value, str) else ""
            if not files:
                raise ApiError(400, "no files: send one or more multipart parts named 'files'")
            if len(files) > MAX_FILES_PER_REQUEST:
                raise ApiError(413, f"too many files: at most {MAX_FILES_PER_REQUEST} per request")
            results = []
            for f in files:
                results.append(await run_in_threadpool(
                    uploads.process_upload, f.filename, f.file, fmt or None))
            return {"uploads": results}
        finally:
            await form.close()

    @app.get("/api/uploads")
    async def list_uploads():
        return {"uploads": uploads.list()}

    @app.get("/api/uploads/{upload_id}")
    async def get_upload(upload_id: str):
        if uploads.get(upload_id) is None:
            raise ApiError(404, f"unknown upload {upload_id!r}")
        try:
            return await run_in_threadpool(uploads.full_info, upload_id)
        except FileNotFoundError:
            raise ApiError(404, f"file of upload {upload_id!r} is missing")

    # ------------------------------------------------------------------ jobs
    @app.post("/api/jobs", status_code=201)
    async def post_jobs(request: Request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            raise ApiError(400, "request body must be a JSON object")
        if not isinstance(body, dict):
            raise ApiError(400, "request body must be a JSON object")
        plans = _plan_submission(body, uploads, runs_dir, store)
        created: List[JobRecord] = []
        remote_created: List[tuple] = []     # (RemoteTarget, remote run dir) made by this request
        try:
            for plan in plans:
                field = plan.pop("folder_field")
                rtarget = plan.pop("remote_target", None)
                try:
                    rec = registry.create_job(**plan)
                except FolderExists as exc:
                    raise validation_error([{"field": field, "code": "folder_exists",
                                             "message": str(exc)}])
                except LocationUnusable as exc:
                    raise validation_error([{"field": "local_location" if rtarget else "location",
                                             "code": "location_not_writable",
                                             "message": str(exc)}])
                created.append(rec)
                if rtarget is not None:
                    remote_dir = await _create_remote_dir(
                        transport, rtarget, rec, plan["remote_location"], field)
                    registry.set_remote_run_dir(rec, remote_dir)
                    remote_created.append((rtarget, remote_dir))
        except BaseException:
            for rec in created:
                registry.discard_job(rec)
            for rtarget, remote_dir in remote_created:
                try:
                    await run_in_threadpool(transport.run_json, rtarget, "discard",
                                            [remote_dir])
                except Exception:  # noqa: BLE001 - best effort rollback
                    logger.warning("could not remove the remote folder %s", remote_dir)
            raise
        for rec in created:
            registry.enqueue(rec)
        return _json({"jobs": [registry.summary(r) for r in created]}, 201)

    @app.post("/api/paths/check")
    async def paths_check(request: Request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            raise ApiError(400, "request body must be a JSON object")
        if not isinstance(body, dict):
            raise ApiError(400, "request body must be a JSON object")
        errors = [{"field": k, "message": "unknown key"}
                  for k in sorted(set(body) - {"location", "folders", "target",
                                               "local_location"})]
        target_id = body.get("target", "local")
        if target_id is None:
            target_id = "local"
        local_raw = body.get("local_location")
        if not isinstance(target_id, str):
            errors.append({"field": "target", "message": "must be a string"})
            target_id = "local"
        if local_raw is not None and (not isinstance(local_raw, str)
                                      or len(local_raw) > MAX_PATH_LEN):
            errors.append({"field": "local_location",
                           "message": f"must be a string of at most {MAX_PATH_LEN} characters "
                                      "or null"})
            local_raw = None
        location = body.get("location")
        if location is not None and (not isinstance(location, str)
                                     or len(location) > MAX_PATH_LEN):
            errors.append({"field": "location",
                           "message": f"must be a string of at most {MAX_PATH_LEN} characters "
                                      "or null"})
        folders = body.get("folders", [])
        if folders is None:
            folders = []
        if (not isinstance(folders, list) or len(folders) > MAX_CHECK_FOLDERS
                or not all(isinstance(f, str) and len(f) <= MAX_PATH_LEN for f in folders)):
            errors.append({"field": "folders",
                           "message": f"must be a list of at most {MAX_CHECK_FOLDERS} strings"})
        if errors:
            raise validation_error(errors, f"{len(errors)} problem"
                                           f"{'s' if len(errors) != 1 else ''} with the request")
        if target_id == "local":
            return await run_in_threadpool(_check_paths, location, folders, runs_dir)
        try:
            rtarget = store.get_env(target_id)
        except wremote.MachineNotFound:
            raise ApiError(404, f"unknown target {target_id!r}")
        return await run_in_threadpool(_check_paths_remote, transport, rtarget, location,
                                       local_raw, folders, runs_dir)

    @app.get("/api/jobs")
    async def list_jobs(status: Optional[str] = None,
                        limit: int = Query(500, ge=1, le=5000)):
        wanted = None
        if status is not None and status.strip():
            wanted = {s.strip() for s in status.split(",") if s.strip()}
            bad = sorted(wanted - set(STATUSES))
            if bad:
                raise validation_error([{"field": "status",
                                         "message": f"unknown status {', '.join(bad)}; "
                                                    f"one of {', '.join(STATUSES)}"}])
        return {"jobs": [registry.summary(r) for r in registry.list(wanted, limit)]}

    @app.get("/api/jobs/{job_id}")
    async def job_detail(job_id: str):
        rec = get_job(job_id)
        if rec.inputs_detail is None:
            rec.inputs_detail = await run_in_threadpool(_inputs_detail, rec)
        out = registry.summary(rec)
        reproduce = None
        rj = _read_json_file(rec.run_dir / "reproduce.json")
        if rj is not None:
            reproduce = {"status": rj.get("status"), "notes": list(rj.get("notes") or [])}
        result = None
        if rec.result is not None:
            result = {k: v for k, v in rec.result.items()
                      if k not in ("job_id", "seq", "t", "type")}
        out.update({
            "params": rec.job.get("params") or {},
            "param_sources": rec.job.get("param_sources") or {},
            "structure_every": rec.job.get("structure_every"),
            "calculator": rec.job.get("calculator") or {},
            "inputs": rec.inputs_detail,
            "run_dir": str(rec.run_dir),
            "returncode": rec.state.get("returncode"),
            "result": result,
            "reproduce": reproduce,
            "last_seq": rec.lines_seen - 1,
        })
        return out

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        rec = get_job(job_id)
        if rec.terminal:
            raise ApiError(409, f"job {job_id} is already {rec.status}")
        job_runner = runners.get(rec.queue_key)
        if job_runner is not None:
            job_runner.request_cancel(rec)
        elif rec.status == "pending":
            runner.request_cancel(rec)            # pending needs no runner of its machine
        else:                                      # running on a machine that is gone
            rec.state["cancel_requested"] = True
            rec.remote_active = False
            registry.transition(
                rec, "cancelled", reason="cancelled by user; the machine is no longer "
                                         "configured, so the remote job was not stopped")
        return {"job": registry.summary(rec)}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, since: int = Query(0, ge=0),
                         limit: int = Query(1000, ge=1, le=5000)):
        rec = get_job(job_id)
        path = rec.events_path

        def _read():
            evs = wevents.read_events(path, since=since, limit=limit)
            return evs, wevents.count_complete_lines(path)

        events, n_lines = await run_in_threadpool(_read)
        next_seq = (int(events[-1].get("seq", since)) + 1) if events else since
        return {"events": events, "next_seq": next_seq,
                "terminal": rec.terminal and next_seq == n_lines}

    @app.get("/api/jobs/{job_id}/files")
    async def job_files(job_id: str):
        rec = get_job(job_id)
        files = await run_in_threadpool(_list_files, rec.run_dir)
        remote_only = await run_in_threadpool(_remote_only_files, rec.run_dir)
        if remote_only:
            have = {f["path"] for f in files}
            files += [{"path": f["path"], "size": f["size"], "mtime": None,
                       "remote_only": True}
                      for f in remote_only if f["path"] not in have]
            files.sort(key=lambda f: f["path"])
        return {"files": files}

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    async def job_file(job_id: str, path: str, inline: int = 0):
        rec = get_job(job_id)
        root = rec.run_dir.resolve()
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            raise ApiError(400, "path is outside the job directory")
        if not target.is_file():
            entry = next((f for f in await run_in_threadpool(_remote_only_files, rec.run_dir)
                          if f["path"] == path), None)
            if entry is None:
                raise ApiError(404, f"file {path!r} not found")
            return await _remote_file_response(rec, entry, inline)
        if inline:
            size = target.stat().st_size
            if size > MAX_INLINE_BYTES:
                raise ApiError(413, f"file is larger than {MAX_INLINE_BYTES // (1024 * 1024)} MB; "
                                    "download it instead")
            data = await run_in_threadpool(target.read_bytes)
            if b"\x00" not in data:
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = None
                if text is not None:
                    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")
        media = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        return FileResponse(target, media_type=media, filename=target.name)

    @app.get("/api/jobs/{job_id}/zip")
    async def job_zip(job_id: str):
        rec = get_job(job_id)
        spooled = await run_in_threadpool(_build_zip, rec.run_dir, rec.folder)

        def _stream():
            try:
                while True:
                    chunk = spooled.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
            finally:
                spooled.close()

        return StreamingResponse(
            _stream(), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{rec.folder}.zip"'})

    # ------------------------------------------------------------------ machines
    async def _json_object(request: Request) -> dict:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            raise ApiError(400, "request body must be a JSON object")
        if not isinstance(body, dict):
            raise ApiError(400, "request body must be a JSON object")
        return body

    @app.get("/api/machines")
    async def list_machines():
        return {"machines": store.list()}

    @app.post("/api/machines", status_code=201)
    async def post_machine(request: Request):
        body = await _json_object(request)
        try:
            return _json(store.add_machine(body), 201)
        except wremote.MachineValidationError as exc:
            raise field_errors(exc)

    @app.put("/api/machines/{mid}")
    async def put_machine(mid: str, request: Request):
        body = await _json_object(request)
        get_machine(mid)
        try:
            return store.update_machine(mid, body)
        except wremote.MachineValidationError as exc:
            raise field_errors(exc)

    @app.delete("/api/machines/{mid}")
    async def delete_machine(mid: str):
        machine = get_machine(mid)
        busy = unfinished(lambda r: r.queue_key == mid)
        if busy:
            raise ApiError(409, f"machine {machine['name']} still has {len(busy)} unfinished "
                                f"job{'s' if len(busy) != 1 else ''}; cancel or wait for "
                                "them first")
        store.delete_machine(mid)
        gone = runners.pop(mid, None)
        if gone is not None:
            await gone.stop()
        return {"deleted": mid}

    @app.post("/api/machines/{mid}/test")
    async def test_machine(mid: str):
        machine = get_machine(mid)

        def _test() -> dict:
            t0 = time.monotonic()
            try:
                hostname = transport.run_raw(machine, "hostname").strip()
            except wremote.RemoteError as exc:
                return {"ok": False, "message": exc.message, "hostname": None, "gpus": [],
                        "elapsed_s": round(time.monotonic() - t0, 3)}
            gpus: list = []
            message = f"connected to {hostname}"
            if machine["envs"]:
                rtarget = wremote.RemoteTarget(machine, machine["envs"][0])
                try:
                    info = transport.run_json(rtarget, "gpus", timeout=60.0)
                    gpus = info.get("gpus") or []
                    if info.get("note"):
                        message += f"; {info['note']}"
                except wremote.RemoteError as exc:
                    message += f"; could not list GPUs: {exc.message}"
            else:
                message += "; add an environment to list GPUs"
            return {"ok": True, "message": message, "hostname": hostname, "gpus": gpus,
                    "elapsed_s": round(time.monotonic() - t0, 3)}

        return await run_in_threadpool(_test)

    @app.post("/api/machines/{mid}/envs", status_code=201)
    async def post_env(mid: str, request: Request):
        body = await _json_object(request)
        get_machine(mid)
        try:
            env = store.add_env(mid, body)
        except wremote.MachineValidationError as exc:
            raise field_errors(exc)
        probe = await run_in_threadpool(
            wremote.probe_target, transport, store.get_env(env["id"]))
        return _json(store.set_probe(mid, env["id"], probe), 201)

    @app.put("/api/machines/{mid}/envs/{eid}")
    async def put_env(mid: str, eid: str, request: Request):
        body = await _json_object(request)
        get_machine(mid)
        try:
            return store.update_env(mid, eid, body)
        except wremote.MachineValidationError as exc:
            raise field_errors(exc)
        except wremote.MachineNotFound:
            raise ApiError(404, f"unknown environment {eid!r}")

    @app.delete("/api/machines/{mid}/envs/{eid}")
    async def delete_env(mid: str, eid: str):
        get_machine(mid)
        try:
            env = store.get_env(eid)
        except wremote.MachineNotFound:
            raise ApiError(404, f"unknown environment {eid!r}")
        busy = unfinished(lambda r: r.is_remote and r.target.get("env_id") == eid)
        if busy:
            raise ApiError(409, f"environment {env.env['name']} still has {len(busy)} "
                                f"unfinished job{'s' if len(busy) != 1 else ''}; cancel or "
                                "wait for them first")
        store.delete_env(mid, eid)
        return {"deleted": eid}

    @app.post("/api/machines/{mid}/envs/{eid}/probe")
    async def probe_env(mid: str, eid: str):
        get_machine(mid)
        try:
            rtarget = store.get_env(eid)
        except wremote.MachineNotFound:
            raise ApiError(404, f"unknown environment {eid!r}")
        probe = await run_in_threadpool(wremote.probe_target, transport, rtarget)
        return store.set_probe(mid, eid, probe)

    # ------------------------------------------------------------------ websocket
    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        if origin is not None and not origin_matches_host(origin,
                                                          websocket.headers.get("host", "")):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        conn = Connection()
        broadcaster.add(conn)

        async def sender():
            while True:
                item = await conn.outbox.get()
                if item is None:
                    await websocket.close(code=1000)
                    return
                await websocket.send_text(item)

        async def pinger():
            while True:
                await asyncio.sleep(WS_PING_INTERVAL_S)
                conn.put(json.dumps({"type": "ws_ping", "t": time.time()}))

        tasks = [asyncio.ensure_future(sender()), asyncio.ensure_future(pinger())]
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is None and message.get("bytes") is not None:
                    text = message["bytes"].decode("utf-8", "replace")
                _ws_handle(conn, text, registry)
                if tasks[0].done():
                    break
        except Exception:  # noqa: BLE001 - client went away mid-receive
            pass
        finally:
            conn.closed = True
            broadcaster.remove(conn)
            for t in tasks:
                t.cancel()

    return app


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _probe_devices() -> None:
    try:
        wcalculators.detect_devices()
    except Exception:  # noqa: BLE001 - informational only
        logger.exception("device detection failed")


def _read_json_file(path: Path) -> Optional[dict]:
    try:
        data = json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request, exc: ApiError):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        if exc.status_code == 404:
            message = f"not found: {request.url.path}"
        return JSONResponse(error_body(code_for_status(exc.status_code), message),
                            status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc: RequestValidationError):
        details = []
        for err in exc.errors():
            loc = [str(x) for x in err.get("loc", ())]
            if loc and loc[0] in ("query", "path", "body", "header", "cookie"):
                loc = loc[1:]
            details.append({"field": ".".join(loc) or "request",
                            "message": str(err.get("msg", "invalid value"))})
        if not details:
            details = [{"field": "request", "message": "invalid request"}]
        return JSONResponse(validation_error(details).body(), status_code=422)

    @app.exception_handler(Exception)
    async def _internal_error(request, exc: Exception):
        logger.exception("internal error on %s", request.url.path)
        return JSONResponse(error_body("internal_error", str(exc) or type(exc).__name__),
                            status_code=500)


def _as_int(value: Any) -> Optional[int]:
    """``value`` as an int if it is a JSON integer (or integral float), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        as_int = int(value)
    except (OverflowError, ValueError):
        return None
    return as_int if as_int == value else None


def _calc_type_spec(name: Any) -> Optional[dict]:
    for ct in wschema.get_calc_types().get("calc_types", []):
        if ct.get("name") == name:
            return ct
    return None


def _reserved_dirs(runs_dir: Path) -> List[Path]:
    return [runs_dir / "_uploads", runs_dir / JOBS_INDEX_DIR]


def _check_paths(location: Optional[str], folders: List[str], runs_dir: Path) -> dict:
    """``POST /api/paths/check`` (blocking; creates nothing)."""
    return wpaths.check_paths(location, folders, runs_dir, _reserved_dirs(runs_dir))


def _check_paths_remote(transport, rtarget, location: Optional[str],
                        local_location: Optional[str], folders: List[str],
                        runs_dir: Path) -> dict:
    """``POST /api/paths/check`` for a remote target (blocking; creates nothing).

    ``location``/``folders`` come from the agent on the remote machine;
    ``exists`` is true when the folder exists there **or** locally under the
    local copy location. An unreachable machine is reported in ``error``.
    """
    remote_loc = (location or "").strip() or rtarget.machine["default_location"]
    local = wpaths.check_paths(local_location, [], runs_dir, _reserved_dirs(runs_dir))
    local_path = Path(local["location"]) if local["location"] else runs_dir
    try:
        out = transport.run_json(rtarget, "inspect", ["--", remote_loc, *folders])
    except wremote.RemoteError as exc:
        out = {"location": remote_loc, "exists": False, "writable": None,
               "error": exc.message,
               "folders": [{"input": raw, "folder": slugify_folder(raw), "path": None,
                            "exists": False,
                            "error": folder_problem(slugify_folder(raw), ())}
                           for raw in folders]}
    for entry in out["folders"]:
        local_exists = False
        entry["local_path"] = None
        if entry.get("error") is None and entry.get("folder") and local["error"] is None:
            lp = local_path / entry["folder"]
            entry["local_path"] = str(lp)
            local_exists = os.path.lexists(str(lp))
        entry["exists"] = bool(entry.get("exists")) or local_exists
    out["local_location"] = {"location": local["location"], "exists": local["exists"],
                             "writable": local["writable"], "error": local["error"]}
    return out


async def _create_remote_dir(transport, rtarget, rec: JobRecord, location: str,
                             field: str) -> str:
    """Create the remote run dir of ``rec`` (exclusive) and upload job.json + inputs."""
    tar = await run_in_threadpool(wremote.tar_local_inputs, rec.run_dir)
    try:
        out = await run_in_threadpool(transport.run_json, rtarget, "create",
                                      [location, rec.folder], tar, 180.0)
    except wremote.RemoteUnreachable as exc:
        raise validation_error([{"field": "target", "code": "remote_unreachable",
                                 "message": exc.message}])
    except wremote.RemoteError as exc:
        if exc.returncode == 3:
            raise validation_error([{"field": field, "code": "folder_exists",
                                     "message": f"{exc.message} on {rtarget.machine['name']}; "
                                                "choose another name"}])
        raise validation_error([{"field": "location", "code": "location_not_writable",
                                 "message": exc.message}])
    return str(out["run_dir"])


def _remote_only_files(run_dir: Path) -> List[dict]:
    """Entries of ``remote_files.json`` that were not copied back (blocking)."""
    data = _read_json_file(Path(run_dir) / "remote_files.json")
    if data is None:
        return []
    out = []
    for f in data.get("files") or []:
        if (isinstance(f, dict) and isinstance(f.get("path"), str) and f.get("copied") is False
                and isinstance(f.get("size"), int)):
            out.append({"path": f["path"], "size": f["size"]})
    return out


def _optional_text(body: dict, key: str, max_len: int, errors: List[dict]) -> Optional[str]:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > max_len:
        errors.append({"field": key,
                       "message": f"must be a string of at most {max_len} characters"})
        return None
    return value.strip() or None


def _plan_submission(body: dict, uploads: UploadStore,
                     runs_dir: Optional[Path] = None, store=None) -> List[dict]:
    """Validate a POST /api/jobs body; return ``create_job`` kwargs per job.

    Raises :class:`ApiError` (422 / 404). All-or-nothing: nothing is created
    unless every job validates.
    """
    errors: List[dict] = []
    unknown = sorted(set(body) - JOB_KEYS)
    for key in unknown:
        errors.append({"field": key, "message": "unknown key"})

    calc_type = body.get("calc_type")
    spec = _calc_type_spec(calc_type) if isinstance(calc_type, str) else None
    if spec is None:
        names = [c.get("name") for c in wschema.get_calc_types().get("calc_types", [])]
        errors.append({"field": "calc_type",
                       "message": f"must be one of {', '.join(map(str, names))}"})

    calculator = body.get("calculator", {})
    if not isinstance(calculator, dict):
        errors.append({"field": "calculator", "message": "must be an object"})
    params = body.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        errors.append({"field": "params", "message": "must be an object"})

    structure_every = _as_int(body.get("structure_every", 5))
    if structure_every is None or not 0 <= structure_every <= 100000:
        errors.append({"field": "structure_every",
                       "message": "must be an integer from 0 to 100000"})

    label = body.get("label")
    if label is not None and (not isinstance(label, str) or len(label) > MAX_LABEL_LEN):
        errors.append({"field": "label",
                       "message": f"must be a string of at most {MAX_LABEL_LEN} characters"})
    if isinstance(label, str):
        label = label.strip() or None
    location_raw = _optional_text(body, "location", MAX_PATH_LEN, errors)
    local_raw = _optional_text(body, "local_location", MAX_PATH_LEN, errors)
    folder_raw = _optional_text(body, "folder", MAX_PATH_LEN, errors)
    runs_dir = Path(runs_dir) if runs_dir is not None else uploads.root.parent

    # Target: this computer, or one remote environment (validated against its probe).
    rtarget = None
    catalog = None
    remote_location = None
    target_id = body.get("target", "local")
    if target_id is None:
        target_id = "local"
    if not isinstance(target_id, str):
        errors.append({"field": "target", "message": "must be \"local\" or an environment id"})
    elif target_id != "local":
        try:
            if store is None:
                raise wremote.MachineNotFound(target_id)
            rtarget = store.get_env(target_id)
        except wremote.MachineNotFound:
            errors.append({"field": "target", "message": f"unknown target {target_id!r}"})
        else:
            probe = rtarget.env.get("probe")
            if not isinstance(probe, dict) or not probe.get("ok"):
                errors.append({"field": "target", "message":
                               f"environment {rtarget.env['name']} has not been probed "
                               "successfully; probe it in the Machines panel"})
                rtarget = None
            else:
                catalog = dict(probe.get("calculators") or {})
                catalog["env_name"] = rtarget.env["name"]
    if target_id == "local" and local_raw is not None:
        errors.append({"field": "local_location", "message": "only applies to a remote target"})
    if rtarget is not None:
        remote_location = location_raw or rtarget.machine["default_location"]
        if (not remote_location.startswith(("/", "~")) or "\0" in remote_location
                or "\n" in remote_location):
            errors.append({"field": "location", "code": "location_invalid", "message":
                           "the remote location must be an absolute path (or start with ~)"})
        loc_info = inspect_location(local_raw, runs_dir, _reserved_dirs(runs_dir))
        loc_field = "local_location"
    else:
        loc_info = inspect_location(location_raw, runs_dir, _reserved_dirs(runs_dir))
        loc_field = "location"
    if loc_info["code"] is not None:
        errors.append({"field": loc_field, "code": loc_info["code"],
                       "message": loc_info["error"]})
    folder_base: Optional[str] = None
    if folder_raw is not None:
        given = slugify_folder(folder_raw)
        problem = folder_problem(given)
        if problem is not None:
            errors.append({"field": "folder", "code": "folder_invalid", "message": problem})
        folder_base = given
    elif label is not None:
        # derived from the job name; an unusable slug falls back to job ids
        derived = slugify_folder(label)
        folder_base = derived if derived and folder_problem(derived) is None else None

    # Which id keys does this calc type take?
    groups: List[Dict[str, str]] = []   # one {role: id-field} per job, with field paths
    if spec is not None:
        kind = spec.get("inputs")
        present = {k for k in ID_KEYS if body.get(k) is not None}
        if kind == "single":
            allowed = {"structure_ids"}
            ids = body.get("structure_ids")
            if ids is None:
                errors.append({"field": "structure_ids", "message": "required for " + calc_type})
            elif (not isinstance(ids, list) or not ids
                  or len(ids) > MAX_STRUCTURES_PER_SUBMIT
                  or not all(isinstance(i, str) for i in ids)):
                errors.append({"field": "structure_ids",
                               "message": f"must be a list of 1 to {MAX_STRUCTURES_PER_SUBMIT} "
                                          "upload ids"})
            else:
                groups = [{"structure": (f"structure_ids[{k}]", i)} for k, i in enumerate(ids)]
        else:
            band = body.get("band_id") is not None
            pair = body.get("initial_id") is not None or body.get("final_id") is not None
            if kind == "pair_or_band" and band and pair:
                allowed = set(ID_KEYS) - {"structure_ids"}
                errors.append({"field": "band_id",
                               "message": "give either initial_id + final_id or band_id, "
                                          "not both"})
            elif kind == "pair_or_band" and band:
                allowed = {"band_id"}
                if not isinstance(body.get("band_id"), str):
                    errors.append({"field": "band_id", "message": "must be an upload id"})
                groups = [{"band": ("band_id", body.get("band_id"))}]
            else:
                allowed = {"initial_id", "final_id"}
                mode = " pair mode" if kind == "pair_or_band" else ""
                for key in ("initial_id", "final_id"):
                    if not isinstance(body.get(key), str):
                        errors.append({"field": key,
                                       "message": f"required (an upload id) for {calc_type}{mode}"})
                groups = [{"initial": ("initial_id", body.get("initial_id")),
                           "final": ("final_id", body.get("final_id"))}]
        for key in sorted(present - allowed):
            errors.append({"field": key, "message": f"does not apply to {calc_type}"})
    if errors:
        raise validation_error(errors)

    # Upload lookup (404 on unknown / failed upload ids).
    missing = []
    for group in groups:
        for role, (field, upload_id) in group.items():
            if uploads.get(upload_id) is None:
                missing.append({"field": field, "message": f"unknown upload id {upload_id!r}"})
    if missing:
        raise ApiError(404, "unknown upload id", "not_found", missing)

    multi = len(groups) > 1
    n_jobs = len(groups)
    width = suffix_width(n_jobs)
    plans: List[dict] = []
    all_errors: List[dict] = []
    for k, group in enumerate(groups):
        metas = {role: uploads.structure_meta(uid, role) for role, (_, uid) in group.items()}
        try:
            norm = wschema.validate_submission(calc_type, dict(calculator), dict(params), metas,
                                               catalog)
        except wschema.ValidationFailed as exc:
            prefix = f"structure_ids[{k}]." if multi else ""
            for e in exc.errors:
                all_errors.append({"field": prefix + str(e.get("field", "")),
                                   "message": str(e.get("message", ""))})
            continue
        inputs = {}
        for role, (_, uid) in group.items():
            meta = uploads.get(uid)
            inputs[role] = {"upload_id": uid, "filename": meta["filename"],
                            "format": meta.get("format"), "source": uploads.path_of(uid)}
        if label is not None:
            job_label = f"{label}-{k:0{width}d}" if multi else label
        elif "structure" in inputs:
            job_label = inputs["structure"]["filename"]
        elif "band" in inputs:
            job_label = inputs["band"]["filename"]
        else:
            job_label = f"{inputs['initial']['filename']} -> {inputs['final']['filename']}"
        plans.append({"calc_type": calc_type, "label": job_label,
                      "calculator": norm["calculator"], "params": norm["params"],
                      "param_sources": norm["param_sources"],
                      "structure_every": structure_every, "inputs": inputs})
    if all_errors:
        raise validation_error(all_errors)
    plan_folders(plans, loc_info["path"], folder_base, multi, width)
    if rtarget is not None:
        block = {"kind": "remote", "machine_id": rtarget.machine["id"],
                 "machine_name": rtarget.machine["name"], "env_id": rtarget.env["id"],
                 "env_name": rtarget.env["name"], "host": rtarget.machine["host"],
                 "python": rtarget.env["python"]}
        for plan in plans:
            plan.update({"target": block, "remote_location": remote_location,
                         "remote_target": rtarget})
    return plans


def plan_folders(plans: List[dict], location: Path, base: Optional[str], multi: bool,
                 width: int) -> None:
    """Add ``location``/``folder``/``folder_field`` to each plan (PLAN 15.1).

    ``base`` is the slugged folder (single job) or common base (multi), or
    None for "use the job id". Raises a 422 ``validation_error`` whose details
    carry ``code`` ``folder_exists`` / ``folder_duplicate``.
    """
    errors: List[dict] = []
    seen: Dict[str, int] = {}
    for k, plan in enumerate(plans):
        field = f"structure_ids[{k}].folder" if multi else "folder"
        if base is None:
            folder = None
        elif multi:
            folder = suffixed_folder(base, k, width)
        else:
            folder = base
        plan["location"] = location
        plan["folder"] = folder
        plan["folder_field"] = field
        if folder is None:
            continue
        path = location / folder
        if folder in seen:
            errors.append({"field": field, "code": "folder_duplicate",
                           "message": f"folder {folder!r} is used by more than one job "
                                      f"of this submission"})
            continue
        seen[folder] = k
        if os.path.lexists(str(path)):
            errors.append({"field": field, "code": "folder_exists",
                           "message": f"{path} already exists; choose another name"})
    if errors:
        raise validation_error(errors)


def _inputs_detail(rec: JobRecord) -> dict:
    """JobDetail ``inputs``: parse each input copy in the run dir (blocking)."""
    out = {}
    for role, spec in (rec.job.get("inputs") or {}).items():
        entry = {"upload_id": spec.get("upload_id"), "filename": spec.get("filename"),
                 "natoms": None, "symbols": None, "cell": None, "pbc": None,
                 "xyz": None, "frames_xyz": None}
        try:
            desc = describe_file(rec.run_dir / spec["path"], spec.get("format"))
            for key in ("natoms", "symbols", "cell", "pbc", "xyz", "frames_xyz"):
                entry[key] = desc[key]
        except Exception as exc:  # noqa: BLE001 - show what we can
            logger.warning("could not read input %s of %s: %s", role, rec.job_id, exc)
        out[role] = entry
    return out


def _inside(root: Path, p: Path) -> bool:
    """True if ``p`` resolves (symlinks followed) to a path inside ``root``."""
    return wpaths.inside(root, p)


def _list_files(run_dir: Path) -> List[dict]:
    return wpaths.list_files(run_dir)


def _build_zip(run_dir: Path, prefix: str):
    """Zip every file under ``run_dir`` as ``<prefix>/<relpath>``; files whose
    real path is outside ``run_dir`` (symlinks) are left out."""
    spooled = tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024)
    root = run_dir.resolve()
    with zipfile.ZipFile(spooled, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(run_dir):
            dirnames.sort()
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if not _inside(root, p) or not p.is_file():
                    continue
                try:
                    zf.write(p, f"{prefix}/{p.relative_to(run_dir).as_posix()}")
                except OSError:
                    continue
    spooled.seek(0)
    return spooled


def _ws_handle(conn: Connection, text: Optional[str], registry: JobRegistry) -> None:
    """Handle one client -> server WebSocket message (no awaits: atomic on the loop)."""
    def err(code: str, message: str) -> None:
        conn.put(json.dumps({"type": "ws_error", "code": code, "message": message}))

    try:
        msg = json.loads(text or "")
    except ValueError:
        err("bad_request", "message must be JSON")
        return
    if not isinstance(msg, dict):
        err("bad_request", "message must be a JSON object")
        return
    action, job_id = msg.get("action"), msg.get("job_id")
    if action not in ("subscribe", "unsubscribe"):
        err("bad_request", "action must be 'subscribe' or 'unsubscribe'")
        return
    if not isinstance(job_id, str):
        err("bad_request", "job_id must be a string")
        return
    if action == "unsubscribe":
        if job_id == "*":
            conn.all_jobs = False
        else:
            conn.subs.pop(job_id, None)
        conn.put(json.dumps({"type": "ws_ack", "action": "unsubscribe", "job_id": job_id,
                             "replayed": 0}))
        return
    if job_id == "*":
        conn.all_jobs = True
        conn.put(json.dumps({"type": "ws_ack", "action": "subscribe", "job_id": "*",
                             "replayed": 0}))
        return
    rec = registry.get(job_id)
    if rec is None:
        err("not_found", f"unknown job {job_id!r}")
        return
    since = msg.get("since")
    if since is not None and (isinstance(since, bool) or not isinstance(since, int) or since < 0):
        err("bad_request", "since must be a non-negative integer or null")
        return
    # Join replay and live under the job: everything up to lines_seen comes from
    # the file now; later lines arrive through Broadcaster.publish.
    registry.drain(rec)
    upto = rec.lines_seen
    already = conn.light_sent.get(job_id, set())
    prev_min = conn.subs.get(job_id)
    replayed = 0
    if since is not None and since < upto:
        for ev in wevents.read_events(rec.events_path, since=since):
            seq = ev.get("seq")
            if not isinstance(seq, int) or seq < since:
                continue
            if seq >= upto:
                break
            if seq in already or (prev_min is not None and seq >= prev_min):
                continue
            conn.put(json.dumps(ev, allow_nan=False, separators=(",", ":")))
            replayed += 1
    conn.subs[job_id] = max(since if since is not None else upto, upto)
    conn.put(json.dumps({"type": "ws_ack", "action": "subscribe", "job_id": job_id,
                         "replayed": replayed}))
