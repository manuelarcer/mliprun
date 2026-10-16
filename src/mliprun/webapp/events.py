"""The ``events.jsonl`` format: one JSON object per line, flushed per line.

Used by the worker (writer of everything except ``job_status``) and by the
server (writer of ``job_status`` while no worker is alive, and reader of the
whole file). ``seq`` is the 0-based line index of an event in its file, so a
reader can resume from any point by counting complete lines.

Only the standard library and numpy are imported here, so the server can use
this module without pulling in ASE or an MLIP.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

try:  # numpy is a base dependency of mliprun, but keep this module importable without it
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is always present in practice
    _np = None


def to_jsonable(obj: Any) -> Any:
    """Convert ``obj`` into plain JSON-serialisable Python, recursively.

    numpy scalars and arrays become Python numbers and lists; NaN and
    infinities become ``None`` (JSON has no spelling for them); tuples and
    sets become lists; ``Path`` becomes ``str``; dict keys become ``str``.

    Parameters
    ----------
    obj : any
        The value to convert.

    Returns
    -------
    any
        A value ``json.dumps(..., allow_nan=False)`` accepts.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return to_jsonable(obj.tolist())
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            value = float(obj)
            return value if math.isfinite(value) else None
    if hasattr(obj, "tolist"):
        return to_jsonable(obj.tolist())
    return str(obj)


def encode_frame(positions, cell=None) -> dict:
    """Build the ``Frame`` payload: positions and cell rounded to 4 decimals.

    Parameters
    ----------
    positions : array-like, shape (N, 3)
        Cartesian positions in Angstrom, input atom order.
    cell : array-like, shape (3, 3), optional
        Cell vectors as rows. ``None`` (or an undefined cell) gives zeros.

    Returns
    -------
    dict
        ``{"positions": [[x, y, z], ...], "cell": [[...], [...], [...]]}``.
    """
    pos = [[round(float(v), 4) for v in row] for row in positions]
    if cell is None:
        cell_rows = [[0.0, 0.0, 0.0] for _ in range(3)]
    else:
        if hasattr(cell, "array"):  # ase.cell.Cell
            cell = cell.array
        cell_rows = [[round(float(v), 4) for v in row] for row in cell]
        if len(cell_rows) != 3:
            cell_rows = [[0.0, 0.0, 0.0] for _ in range(3)]
    # round() can return -0.0; normalise so the JSON reads cleanly.
    pos = [[v + 0.0 for v in row] for row in pos]
    cell_rows = [[v + 0.0 for v in row] for row in cell_rows]
    return {"positions": pos, "cell": cell_rows}


def _truncate_partial_line(path: Path) -> int:
    """Drop a trailing partial line; return the number of complete lines."""
    if not path.exists():
        return 0
    with open(path, "r+b") as fh:
        data = fh.read()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1  # 0 when there is no newline at all
            fh.seek(cut)
            fh.truncate()
            data = data[:cut]
    return data.count(b"\n")


class EventWriter:
    """Append-only writer of one job's ``events.jsonl``.

    Parameters
    ----------
    path : Path
        The events file. Created when missing. A trailing partial line (a
        writer killed mid-line) is truncated on open, so the next event
        starts on a fresh line and ``seq`` stays equal to the line index.
    job_id : str
        Stamped into every event envelope.
    """

    def __init__(self, path: Path, job_id: str):
        self.path = Path(path)
        self.job_id = job_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._seq = _truncate_partial_line(self.path)
        # One os.write per event on an O_APPEND descriptor: a line is never
        # split across two writes, and no user-space buffer can hold it back.
        self._fd: Optional[int] = os.open(
            str(self.path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)

    @property
    def next_seq(self) -> int:
        """The ``seq`` the next emitted event will carry."""
        return self._seq

    def emit(self, type: str, **payload) -> dict:  # noqa: A002 - contract name
        """Write one event line and return the event dict.

        Parameters
        ----------
        type : str
            Event type (section 6.3 of the plan).
        **payload
            Event fields; converted with :func:`to_jsonable`. They may not
            override the envelope keys.

        Returns
        -------
        dict
            The event as written (envelope plus payload).
        """
        with self._lock:
            if self._fd is None:
                raise ValueError("EventWriter is closed")
            event = {"job_id": self.job_id, "seq": self._seq,
                     "t": time.time(), "type": type}
            for key, value in payload.items():
                if key in event:
                    raise ValueError(f"payload key {key!r} collides with the envelope")
                event[key] = value
            event = to_jsonable(event)
            line = json.dumps(event, allow_nan=False, ensure_ascii=False,
                              separators=(",", ":")) + "\n"
            data = line.encode("utf-8")
            written = 0
            while written < len(data):
                written += os.write(self._fd, data[written:])
            self._seq += 1
            return event

    def close(self) -> None:
        """Close the file descriptor. Further ``emit`` calls raise."""
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _complete_lines(path: Path) -> list:
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        return []
    lines = data.split(b"\n")
    return lines[:-1]  # the last element is "" or a partial line


def count_complete_lines(path: Path) -> int:
    """Number of newline-terminated lines in ``path`` (0 when missing)."""
    try:
        return Path(path).read_bytes().count(b"\n")
    except FileNotFoundError:
        return 0


def read_events(path: Path, since: int = 0, limit: Optional[int] = None) -> list:
    """Read events with ``seq >= since`` from an events file.

    Only complete (newline-terminated) lines are read. A line that fails to
    parse is skipped but still occupies its ``seq``.

    Parameters
    ----------
    path : Path
        The events file. Missing file gives an empty list.
    since : int
        First line index to return.
    limit : int, optional
        Maximum number of events to return.

    Returns
    -------
    list[dict]
        Parsed events in file order.
    """
    out = []
    since = max(int(since or 0), 0)
    for index, raw in enumerate(_complete_lines(path)):
        if index < since:
            continue
        if limit is not None and len(out) >= limit:
            break
        try:
            event = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(event, dict):
            out.append(event)
    return out
