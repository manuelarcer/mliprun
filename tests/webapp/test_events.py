"""``events.jsonl`` writer/reader (plan section 6 and 8.5)."""
from __future__ import annotations

import json
import math

import numpy as np

from mliprun.webapp.events import (
    EventWriter,
    count_complete_lines,
    encode_frame,
    read_events,
    to_jsonable,
)


def test_emit_writes_envelope_and_increments_seq(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventWriter(path, "job-1") as w:
        a = w.emit("phase", phase="setup", message="x")
        b = w.emit("opt_step", step=np.int64(3), energy=np.float64(-1.5), fmax=0.2)
    assert (a["seq"], b["seq"]) == (0, 1)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    ev = json.loads(lines[1])
    assert ev == {"job_id": "job-1", "seq": 1, "t": ev["t"], "type": "opt_step",
                  "step": 3, "energy": -1.5, "fmax": 0.2}
    assert isinstance(ev["t"], float) and ev["t"] > 1.7e9


def test_partial_line_is_truncated_and_seq_continues(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventWriter(path, "j") as w:
        w.emit("phase", phase="setup", message="")
        w.emit("phase", phase="run", message="")
    with open(path, "ab") as fh:
        fh.write(b'{"job_id":"j","seq":2,"t":1.0,"ty')  # writer killed mid-line
    assert count_complete_lines(path) == 2
    assert len(read_events(path)) == 2  # the partial line is not read
    with EventWriter(path, "j") as w:
        assert w.next_seq == 2
        ev = w.emit("result", calc_type="x")
    assert ev["seq"] == 2
    events = read_events(path)
    assert [e["seq"] for e in events] == [0, 1, 2]
    assert path.read_bytes().endswith(b"\n")


def test_partial_only_line_truncates_to_empty(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"half')
    with EventWriter(path, "j") as w:
        assert w.next_seq == 0
        w.emit("phase", phase="setup", message="")
    assert [e["seq"] for e in read_events(path)] == [0]


def test_read_events_since_limit_and_bad_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventWriter(path, "j") as w:
        for i in range(3):
            w.emit("log", level="info", message=str(i), logger="t")
    with open(path, "ab") as fh:
        fh.write(b"not json\n")
    with EventWriter(path, "j") as w:
        assert w.next_seq == 4  # the garbage line still occupies a seq
        w.emit("log", level="info", message="4", logger="t")
    assert [e["seq"] for e in read_events(path)] == [0, 1, 2, 4]
    assert [e["seq"] for e in read_events(path, since=2)] == [2, 4]
    assert [e["seq"] for e in read_events(path, since=1, limit=1)] == [1]
    assert read_events(tmp_path / "missing.jsonl") == []
    assert count_complete_lines(tmp_path / "missing.jsonl") == 0


def test_to_jsonable_nan_inf_numpy():
    obj = {"a": float("nan"), "b": [np.inf, -np.inf, 1.0], "c": np.array([[1, 2]]),
           "d": np.bool_(True), "e": (np.float32(0.5),), 3: None}
    out = to_jsonable(obj)
    assert out == {"a": None, "b": [None, None, 1.0], "c": [[1, 2]], "d": True,
                   "e": [0.5], "3": None}
    json.dumps(out, allow_nan=False)


def test_emit_converts_nan_to_null(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventWriter(path, "j") as w:
        w.emit("md_step", step=0, etot=float("nan"))
    assert json.loads(path.read_text())["etot"] is None


def test_encode_frame_rounds_and_fills_cell():
    frame = encode_frame(np.array([[0.123456, -0.00001, 1.0]]), None)
    assert frame == {"positions": [[0.1235, 0.0, 1.0]],
                     "cell": [[0.0, 0.0, 0.0]] * 3}
    assert not math.copysign(1.0, frame["positions"][0][1]) < 0  # no "-0.0"
    frame = encode_frame([[1, 2, 3]], np.eye(3) * 3.61234)
    assert frame["cell"][0] == [3.6123, 0.0, 0.0]
