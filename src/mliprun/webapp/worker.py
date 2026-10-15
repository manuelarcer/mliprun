"""Worker process: run one web-app job, ``python -m mliprun.webapp.worker <run_dir>``.

Reads ``run_dir/job.json``, writes ``reproduce.sh``/``reproduce.json``, builds
the calculator, runs the core engine with live progress going to
``events.jsonl``, and finishes with ``summary.json``.

Exit codes: 0 result written; 1 the calculation raised; 2 unreadable
``job.json`` or setup failure before the calculation; 3 cancelled (SIGTERM).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

EXIT_OK = 0
EXIT_CALC_FAILED = 1
EXIT_SETUP_FAILED = 2
EXIT_CANCELLED = 3

CRASH_ENV = "MLIPRUN_WEBAPP_TEST_CRASH"
LOG_RATE_PER_S = 50


class JobCancelled(Exception):
    """Raised in the main thread by the SIGTERM handler.

    An ``Exception`` (not ``BaseException``) on purpose: the core engines'
    ``except Exception`` then marks ``mliprun_run.json`` as ``failed``
    before re-raising.
    """


def _now_iso() -> str:
    now = _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class _EventLogHandler(logging.Handler):
    """Forward log records as ``log`` events, at most ``rate`` per second."""

    def __init__(self, writer, rate: int = LOG_RATE_PER_S):
        super().__init__(level=logging.INFO)
        self.writer = writer
        self.rate = rate
        self._window = int(time.time())
        self._count = 0
        self._dropped = 0
        self._local = threading.local()

    def _flush_dropped(self) -> None:
        if self._dropped:
            n, self._dropped = self._dropped, 0
            self.writer.emit("log", level="warning",
                             message=f"{n} log messages suppressed",
                             logger="mliprun.webapp")

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        if getattr(self._local, "busy", False):
            return  # a log call made while writing an event: drop, never recurse
        self._local.busy = True
        try:
            now = int(time.time())
            if now != self._window:
                self._window, self._count = now, 0
                self._flush_dropped()
            if self._count >= self.rate:
                self._dropped += 1
                return
            self._count += 1
            level = ("error" if record.levelno >= logging.ERROR else
                     "warning" if record.levelno >= logging.WARNING else "info")
            self.writer.emit("log", level=level, message=record.getMessage(),
                             logger=record.name)
        except Exception:  # noqa: BLE001 - logging must never kill the job
            pass
        finally:
            self._local.busy = False

    def close(self) -> None:
        try:
            self._flush_dropped()
        except Exception:  # noqa: BLE001
            pass
        super().close()


class _Worker:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.job: dict = {}
        self.job_id = self.run_dir.name
        self.calc_type: Optional[str] = None
        self.writer = None
        self.started_at = _now_iso()
        self.t0 = time.time()
        self.reproduce = None
        self._cancel_seen = False
        self._crash_armed = os.environ.get(CRASH_ENV) == "after_first_event"

    # -- events ----------------------------------------------------------
    def emit(self, type_: str, **payload) -> None:
        if self.writer is None:
            return
        self.writer.emit(type_, **payload)
        if self._crash_armed:
            os._exit(7)

    def phase(self, name: str, message: str = "") -> None:
        self.emit("phase", phase=name, message=message)

    # -- signals ---------------------------------------------------------
    def _on_sigterm(self, signum, frame) -> None:
        if self._cancel_seen:
            return
        self._cancel_seen = True
        raise JobCancelled("cancelled by user")

    # -- summary ---------------------------------------------------------
    def write_summary(self, status: str, result: Optional[dict] = None,
                      error: Optional[dict] = None) -> None:
        from mliprun.webapp.runners import strip_frames

        summary = {
            "job_id": self.job_id, "calc_type": self.calc_type, "status": status,
            "started_at": self.started_at, "finished_at": _now_iso(),
            "walltime_s": round(time.time() - self.t0, 3),
            "result": strip_frames(result) if result is not None else None,
            "error": error,
            "reproduce": ({"status": self.reproduce["status"],
                           "notes": self.reproduce["notes"]}
                          if self.reproduce else None),
        }
        from mliprun.webapp.events import to_jsonable

        tmp = self.run_dir / "summary.json.tmp"
        tmp.write_text(json.dumps(to_jsonable(summary), indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
        os.replace(tmp, self.run_dir / "summary.json")

    def fail(self, exc: BaseException, code: int) -> int:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        sys.stderr.write(tb)
        message = str(exc) or type(exc).__name__
        try:
            self.emit("error", message=message, exception_type=type(exc).__name__,
                      traceback=tb)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.write_summary("failed", error={"message": message,
                                                "exception_type": type(exc).__name__})
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        return code

    def cancelled(self) -> int:
        try:
            self.emit("log", level="info", message="cancelled by user",
                      logger="mliprun.webapp")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.write_summary("cancelled", error={"message": "cancelled by user",
                                                   "exception_type": "JobCancelled"})
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        return EXIT_CANCELLED

    # -- inputs ----------------------------------------------------------
    def read_structures(self) -> tuple:
        from ase.io import read, write

        structures = {}
        band_frames = None
        inputs = self.job.get("inputs") or {}
        for role, meta in inputs.items():
            path = self.run_dir / meta["path"]
            fmt = meta.get("format") or None
            if role == "band":
                frames = read(str(path), index=":", format=fmt)
                structures["band"] = list(frames)
                band_frames = len(frames)
            else:
                structures[role] = read(str(path), index=-1, format=fmt)
        if self.calc_type in ("neb", "autoneb"):
            if "band" in structures:
                first, last = structures["band"][0], structures["band"][-1]
            else:
                first, last = structures["initial"], structures["final"]
            inputs_dir = self.run_dir / "inputs"
            inputs_dir.mkdir(exist_ok=True)
            for name, atoms in (("initial.vasp", first), ("final.vasp", last)):
                write(str(inputs_dir / name), atoms, format="vasp", direct=True,
                      vasp5=True)
        return structures, band_frames

    # -- main ------------------------------------------------------------
    def run(self) -> int:
        from mliprun.webapp.events import EventWriter

        job_path = self.run_dir / "job.json"
        try:
            self.job = json.loads(job_path.read_text(encoding="utf-8"))
            if not isinstance(self.job, dict):
                raise ValueError("job.json is not a JSON object")
            self.job_id = self.job.get("job_id") or self.job_id
            self.calc_type = self.job["calc_type"]
        except Exception as exc:  # noqa: BLE001
            try:
                self.writer = EventWriter(self.run_dir / "events.jsonl", self.job_id)
            except Exception:  # noqa: BLE001
                self.writer = None
            return self.fail(exc, EXIT_SETUP_FAILED)

        self.writer = EventWriter(self.run_dir / "events.jsonl", self.job_id)
        signal.signal(signal.SIGTERM, self._on_sigterm)
        handler = _EventLogHandler(self.writer)
        pkg_logger = logging.getLogger("mliprun")
        pkg_logger.setLevel(logging.INFO)
        pkg_logger.addHandler(handler)
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").addHandler(handler)
        try:
            return self._run_job()
        finally:
            handler.close()
            pkg_logger.removeHandler(handler)
            logging.getLogger("py.warnings").removeHandler(handler)
            self.writer.close()

    def _run_job(self) -> int:
        stage = "setup"
        try:
            self.phase("setup", "reading inputs")
            import matplotlib

            matplotlib.use("Agg")
            from mliprun.webapp import calculators, reproduce, runners

            if self.calc_type not in runners.RUNNERS:
                raise ValueError(f"unknown calc_type {self.calc_type!r}")
            spec = dict(self.job.get("calculator") or {})
            structures, band_frames = self.read_structures()
            device = calculators.resolved_device(spec)
            self.reproduce = reproduce.write_reproduce(
                self.run_dir, self.job, device_resolved=device, band_frames=band_frames)

            self.phase("load_model", f"loading {spec.get('name')} on {device}")
            adapter = runners.ObserverAdapter(self.writer,
                                              self.job.get("structure_every", 5))
            ri = runners.RunInputs(self.job, self.run_dir, spec, device, adapter,
                                   structures, self.phase)
            if self.calc_type in ("neb", "autoneb"):
                # CustomNEB builds one calculator per image itself (CLI parity);
                # here the model choice is only checked.
                calculators.check_mlip(spec)
            else:
                ri.calc = calculators.make_calculator(spec)

            stage = "run"
            if self.calc_type not in ("neb", "autoneb"):
                self.phase("run", f"running {self.calc_type}")
            result = runners.finalize(runners.RUNNERS[self.calc_type](ri))
            signal.signal(signal.SIGTERM, signal.SIG_IGN)  # past the point of no return
            self.emit("result", **result)
            self.write_summary("done", result=result)
            return EXIT_OK
        except JobCancelled:
            return self.cancelled()
        except Exception as exc:  # noqa: BLE001
            if self._cancel_seen:
                return self.cancelled()
            return self.fail(exc, EXIT_SETUP_FAILED if stage == "setup"
                             else EXIT_CALC_FAILED)


def main(argv: Optional[list] = None) -> int:
    """Entry point. ``argv`` is ``[run_dir]``."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.stderr.write("usage: python -m mliprun.webapp.worker <run_dir>\n")
        return EXIT_SETUP_FAILED
    os.environ.setdefault("MPLBACKEND", "Agg")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return _Worker(Path(argv[0])).run()


if __name__ == "__main__":
    # Run through the importable module, not ``__main__``, so that
    # ``JobCancelled`` and friends are one class wherever they are imported.
    from mliprun.webapp.worker import main as _main

    sys.exit(_main())
