"""Canonical JSON run record.

One ``mliprun_run.json`` per run directory, written from the core layer so
every caller gets one -- CLI commands and direct library callers alike. This
module is the only place that knows the file format.

Design: docs/superpowers/specs/2026-07-21-unified-run-record-design.md
Amended: docs/superpowers/specs/2026-07-29-run-record-head-provenance-design.md

The governing rule is that a record failure must never kill a run: every
public entry point swallows its own exceptions and logs a warning. A six-hour
trajectory must not be lost because a provenance file could not be written.
"""
from __future__ import annotations

import json
import logging
import math
import os
import platform
import secrets
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

RECORD_FILENAME = "mliprun_run.json"
SCHEMA_VERSION = 2

#: Model-tag prefix -> installed distribution name. Longest prefix wins, so
#: ``mace-mh-1`` resolves before the bare ``mace`` entry.
_MLIP_PACKAGES = (
    ("uma-", "fairchem-core"),
    ("mace-mh-", "mace-torch"),
    ("mace", "mace-torch"),
    ("7net", "sevenn"),
    ("chgnet", "chgnet"),
)

#: The only values `_tag()` will write to a parameter's `source` field. Kept
#: as a module constant so later code (e.g. CLI-side tagging) can reuse it
#: instead of re-typing the set.
VALID_PARAM_SOURCES = frozenset({"user", "default", "env", "prompt", "unspecified"})

#: Provenance fields compared between the run's origin and an appended
#: stage. Only these fields are meaningful to call out as "what changed" --
#: see I2 in .superpowers/sdd/task-1-fixes.md.
_PROVENANCE_DIFF_FIELDS = ("mliprun_version", "hostname", "device_resolved",
                           "mlip_model", "uma_task", "mace_head")


def _now_iso() -> str:
    """Timezone-aware local timestamp, ISO 8601."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def new_batch_id() -> str:
    """Return an identifier shared by every run of one batch."""
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def _jsonable(obj: Any) -> Any:
    """Coerce a value into something ``json.dumps`` accepts.

    NaN and Inf become ``None``: they are not valid JSON, and Python's encoder
    would otherwise emit bare ``NaN``/``Infinity`` tokens that strict parsers
    reject. A diverged run must not corrupt its own record.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    # numpy scalars and anything else array-like exposing .item()
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:  # noqa: BLE001 -- fall through to repr
            pass
    return repr(obj)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Serialize fully, then replace the target in one step.

    Serialization happens before the target is touched, so a payload that
    cannot be encoded leaves the previous record intact. ``os.replace`` is
    atomic within a directory, so a crash never leaves truncated JSON where a
    valid record used to be.
    """
    text = json.dumps(payload, indent=2, allow_nan=False)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_existing(path: Path) -> Optional[dict]:
    """Return the record at ``path``, or None if absent or unusable.

    An unparseable record, or one that parses to valid JSON that is not an
    object (e.g. a bare ``[]``), is moved aside rather than deleted: it may
    be the only evidence of what a prior run did.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("record did not parse to a JSON object")
    except (OSError, ValueError):
        backup = path.with_suffix(
            path.suffix + f".corrupt-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        )
        try:
            os.replace(path, backup)
            logger.warning("Unparseable %s moved to %s", path.name, backup.name)
        except OSError:
            logger.warning("Unparseable %s could not be backed up", path.name)
        return None
    return data


def _mlip_package(mlip_model: Any) -> dict:
    """Resolve a model tag to its installed distribution name and version.

    ``mlip_model`` may be ``None`` or otherwise non-string (a caller that
    could not determine the tag) -- that is not a package lookup failure,
    just an unresolvable one.
    """
    if not isinstance(mlip_model, str):
        return {"name": None, "version": None}
    name = None
    for prefix, dist in _MLIP_PACKAGES:
        if mlip_model.startswith(prefix):
            name = dist
            break
    if name is None:
        return {"name": None, "version": None}
    try:
        return {"name": name, "version": version(name)}
    except PackageNotFoundError:
        return {"name": name, "version": None}


def collect_provenance(*, mlip_model: Any, device_requested: str,
                        device_resolved: str, uma_task: Optional[str] = None,
                        mace_head: Optional[str] = None) -> dict:
    """Gather environment and version facts for the record.

    ``device_requested`` and ``device_resolved`` are kept apart because
    ``auto`` is what was typed and ``cuda`` is what ran. ``started_at`` and
    ``finished_at`` are filled in by :class:`RunRecord`.

    Every fallible call is individually guarded: a record with a few
    ``None`` fields is far better than no record, and this function must
    never raise -- ``socket.gethostname()`` genuinely fails on some
    HPC/container setups.

    ``uma_task`` and ``mace_head`` are gated on the model tag rather than
    trusted from the caller: CLIs pass whatever their ``--uma-task`` /
    ``--mace-head`` options resolved to, defaults included, so a MACE run
    would otherwise be recorded as carrying a UMA task it never used.
    CANON C1 makes the head its own explicit decision and C3 forbids mixing
    heads within an energy formula, so a wrong head is worse than none.
    """
    try:
        mliprun_version = version("mliprun")
    except PackageNotFoundError:
        mliprun_version = None
    try:
        from ase import __version__ as ase_version
    except Exception:  # noqa: BLE001 -- ASE absence must not break the record
        ase_version = None
    try:
        mlip_package = _mlip_package(mlip_model)
    except Exception:  # noqa: BLE001 -- unresolvable tag must not break the record
        mlip_package = {"name": None, "version": None}
    try:
        python_version = platform.python_version()
    except Exception:  # noqa: BLE001
        python_version = None
    try:
        hostname = socket.gethostname()
    except Exception:  # noqa: BLE001 -- fails on some HPC/container setups
        hostname = None
    tag = mlip_model if isinstance(mlip_model, str) else ""
    return {
        "mliprun_version": mliprun_version,
        "ase_version": ase_version,
        "mlip_package": mlip_package,
        "mlip_model": mlip_model,
        "uma_task": uma_task if tag.startswith("uma-") else None,
        "mace_head": mace_head if tag.startswith("mace-mh-") else None,
        "device_requested": device_requested,
        "device_resolved": device_resolved,
        "python_version": python_version,
        "hostname": hostname,
        "started_at": None,
        "finished_at": None,
        "walltime_s": None,
    }


@dataclass
class BatchInfo:
    """Identity of the batch a run belongs to."""

    batch_id: str
    driver: str
    argv: list[str] = field(default_factory=list)
    root: str = ""
    config_file: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "driver": self.driver,
            "argv": list(self.argv),
            "root": self.root,
            "config_file": self.config_file,
        }


@dataclass
class RunContext:
    """What core cannot determine on its own.

    ``param_sources`` maps a parameter name to one of ``user``, ``default``,
    ``env`` or ``prompt``. Any parameter absent from the map is tagged
    ``unspecified`` -- core never guesses by comparing against signature
    defaults, because a caller that explicitly passes the default value is
    indistinguishable from one that omitted it.
    """

    command: str
    mode: str = "one-off"
    batch: Optional[BatchInfo] = None
    param_sources: Optional[dict[str, str]] = None
    #: Input facts only the caller knows -- the structure filename and path.
    #: Core sees an ``Atoms`` object, which carries no provenance of its own.
    extra_inputs: Optional[dict] = None


def _tag(parameters: dict, sources: Optional[dict]) -> dict:
    sources = sources or {}

    def _source(key: str) -> str:
        tag = sources.get(key, "unspecified")
        return tag if tag in VALID_PARAM_SOURCES else "unspecified"

    return {
        str(k): {"value": _jsonable(v), "source": _source(k)}
        for k, v in parameters.items()
    }


class RunRecord:
    """A record being written. Obtain one from :meth:`begin`."""

    def __init__(self, path: Path, payload: dict, stage_index: int, t0: float):
        self.path = path
        self._payload = payload
        self._stage_index = stage_index
        self._t0 = t0

    @classmethod
    def begin(cls, output_dir, *, command: str, stage_kind: str,
              parameters: dict, inputs: dict, provenance: dict,
              run_context: Optional[RunContext] = None,
              stage_parameters: Optional[dict] = None,
              append: bool = False) -> "RunRecord":
        """Write phase one and return a handle for :meth:`complete`.

        Never raises. On failure a handle is still returned, so callers need
        no error handling of their own; the subsequent ``complete`` is simply
        a no-op.
        """
        t0 = time.perf_counter()
        path = None
        try:
            path = Path(output_dir) / RECORD_FILENAME
            sources = run_context.param_sources if run_context else None
            existing = _load_existing(path) if append else None
            stages = list(existing.get("stages", [])) if existing else []

            stage = {
                "index": len(stages),
                "kind": stage_kind,
                "status": "running",
                "started_at": _now_iso(),
                "walltime_s": None,
                "steps": None,
                "results": {},
            }
            if stage_parameters:
                stage["parameters"] = _tag(stage_parameters, sources)
            if append and existing is None:
                # Resuming a directory that predates the record, or whose
                # record was unreadable. Say so rather than implying this
                # stage is the whole story.
                stage["prior_history_unknown"] = True
            stages.append(stage)

            if existing:
                payload = existing
                payload["stages"] = stages
                payload["status"] = "running"
                # I2: the run's origin provenance never changes on append;
                # instead the stage records only what differs from it, so a
                # reader can see what changed about the environment between
                # stages without losing the original context.
                top_prov = payload.get("provenance") or {}
                prov_in = provenance or {}
                stage_delta = {
                    key: prov_in.get(key)
                    for key in _PROVENANCE_DIFF_FIELDS
                    if prov_in.get(key) != top_prov.get(key)
                }
                if stage_delta:
                    stage["stage_provenance"] = stage_delta
            else:
                prov = dict(provenance)
                prov["started_at"] = _now_iso()
                # A caller that skips collect_provenance() (as in a minimal
                # test double) may omit these; RunRecord owns them either way.
                prov.setdefault("finished_at", None)
                prov.setdefault("walltime_s", None)
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "command": command,
                    "status": "running",
                    "run": {
                        "mode": run_context.mode if run_context else "one-off",
                        "batch": (run_context.batch.as_dict()
                                  if run_context and run_context.batch else None),
                    },
                    "inputs": _jsonable({
                        **inputs,
                        **((run_context.extra_inputs or {}) if run_context else {}),
                    }),
                    "parameters": _tag(parameters, sources),
                    "provenance": _jsonable(prov),
                    "stages": stages,
                }

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            _atomic_write_json(path, payload)
            return cls(path, payload, stage["index"], t0)
        except Exception as exc:  # noqa: BLE001 -- never kill the run
            logger.warning("Could not write %s: %s", RECORD_FILENAME, exc)
            if path is None:
                # The failure happened before `path` could even be built
                # (e.g. `output_dir=None`). A dead handle's `complete()` is
                # a no-op regardless, so any placeholder path is fine.
                path = Path(".") / RECORD_FILENAME
            return cls(path, {}, -1, t0)

    def complete(self, *, status: str, steps: Optional[int] = None,
                 results: Optional[dict] = None) -> None:
        """Write phase two: terminal status, timings and results.

        Never raises.
        """
        if self._stage_index < 0:
            return
        try:
            stage = self._payload["stages"][self._stage_index]
            stage["status"] = status
            stage["walltime_s"] = round(time.perf_counter() - self._t0, 3)
            stage["steps"] = _jsonable(steps)
            stage["results"] = _jsonable(results or {})

            self._payload["status"] = status
            prov = self._payload["provenance"]
            prov["finished_at"] = _now_iso()
            # I1 (human decision): top-level walltime_s is the SUM of every
            # stage's walltime_s, not just the one that just completed --
            # `started_at` stays at the run's origin (set once, in `begin`)
            # and `finished_at` above already tracks the latest completion.
            prov["walltime_s"] = round(
                sum(
                    s["walltime_s"]
                    for s in self._payload["stages"]
                    if s.get("walltime_s") is not None
                ),
                3,
            )

            _atomic_write_json(self.path, self._payload)
        except Exception as exc:  # noqa: BLE001 -- never kill the run
            logger.warning("Could not finalize %s: %s", RECORD_FILENAME, exc)
