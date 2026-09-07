# Committee Evaluation and Per-Configuration Uncertainty — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `mlip optimize run --structure POSCAR --committee committee.yaml` relaxes a structure under the mean force of several MLIPs living in separate Python environments, and reports a per-configuration uncertainty from their disagreement.

**Architecture:** Each committee member runs as a subprocess launched by its own MLIP env's interpreter, speaking line-delimited JSON over pipes. Driver-side, a `CommitteeCalculator` (a plain ASE `Calculator`) fans the geometry out to every member concurrently, averages energy and forces, and reduces the per-atom force disagreement to a σ. Because it is an ASE calculator, the existing `run_optimization` needs no restructuring — only output and record wiring.

**Tech Stack:** Python 3.11 (CI), ASE, NumPy, pandas, typer, PyYAML (new base dependency), stdlib `subprocess` / `select` / `json` / `concurrent.futures`.

**Spec:** `docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md` — read it before Task 1. The plan implements that spec; where this plan states a number or a rule, the spec states why.

## Global Constraints

Every task's requirements implicitly include this section.

- **`src/mliprun/core/committee/protocol.py` imports stdlib only.** It is imported inside every MLIP env, where the torch/ASE stacks differ. No numpy, no ASE, no mliprun siblings.
- **Plain arrays on the wire, never a pickled `Atoms`.** One committee held ASE 3.28.0 and 3.29.0 simultaneously.
- **Constraints stay driver-side.** Workers receive raw geometries and return raw forces; ASE applies `FixAtoms` in the driver exactly as today. Canon S8 (frozen-layer scheme) must not become invisible.
- **A failed member aborts the whole run.** Never continue with fewer members — the mean and σ would silently change definition mid-trajectory.
- **Teardown kills every remaining worker on every exit path,** including aborts. cos-cluster has no scheduler to reap orphans and a leaked worker holds a CUDA context.
- **Mixed levels of theory warn; they never refuse.** The flag is stamped into the run record *and* carried as a per-row CSV column.
- **Unit suite passes with no MLIP installed:** `pytest -m "not uma and not mace and not sevenn"`. Every new test in this plan runs under that selector.
- **Test-id trap:** `tests/conftest.py` auto-skips on `"uma" in item.keywords` (and `mace`, `sevenn`). `item.keywords` is a mapping, so the match is on an **exact key**, and a parametrize id of exactly `uma`, `mace`, or `sevenn` silently skips the whole case while reporting green. Full model tags (`uma-s-1p2`) are safe keys, but this plan uses neutral ids (`member_a`, `task_a`) with the real tags in the test body, per the spec. After adding parametrized tests, confirm they ran: `pytest <file> -q -rs` and check for `0 skipped`.
- **Never regenerate golden reference files** (`tests/goldens/*.json`). None of them carry `schema_version`, so the schema 3 → 4 bump does not touch them; if a golden test fails anyway, report the numerical delta and stop.
- **Every test asserts a numerical value or invariant.** Loosening a tolerance with the observed delta recorded is acceptable; a test that only asserts "no exception" is not.
- **CI gates:** full unit suite + diff coverage ≥ 90% on changed lines.
- **Draft PRs only; never push to main.** One concern per PR.
- **New modules start with `from __future__ import annotations`.** `pyproject.toml` claims Python 3.9 support while existing code already uses 3.10+ union syntax in signatures; do not widen that discrepancy, and do not fix it here.
- **The bridge is POSIX-only.** `select` on pipes does not work on Windows. Say so in the docstring; do not attempt a Windows path.
- Existing code style: typer CLI, lazy imports for heavy packages, NumPy-style docstrings.

---

## File Structure

**New package — `src/mliprun/core/committee/`**

| file | responsibility |
|---|---|
| `__init__.py` | empty, matching every other package `__init__` in this repo |
| `protocol.py` | message framing and construction. Stdlib only. Task 1 |
| `worker.py` | subprocess entry point run by each MLIP env's interpreter. Task 2 |
| `remote.py` | `RemoteMember`: one `Popen`, load handshake, timeouts, teardown. Task 3 |
| `calculator.py` | `committee_statistics`, `CommitteeCalculator`, trace/CSV writers. Tasks 4, 5, 8 |
| `config.py` | level-of-theory table, `committee.yaml` parsing and validation. Tasks 6, 7 |

**Modified**

| file | change |
|---|---|
| `src/mliprun/core/run_record.py` | `SCHEMA_VERSION` 3 → 4; `collect_provenance(committee=...)`. Task 9 |
| `src/mliprun/core/optimize.py` | `run_optimization(committee=..., committee_config=..., uncertainty_threshold=...)`; CSV outputs; plot panel. Tasks 10, 11 |
| `src/mliprun/cli/commands/optimize.py` | `--committee`, `--member-timeout`, `--uncertainty-threshold` on `run`. Task 12 |
| `pyproject.toml` | add `pyyaml` to base dependencies. Task 7 |
| `tests/test_run_record.py`, `tests/test_core_md.py` | schema-version assertions 3 → 4. Task 9 |
| `README.md`, `AGENTS.md`, `CONTEXT.md`, `docs/OUTPUTS.md`, `docs/PYTHON_API.md`, `docs/adr/0001-per-mlip-envs.md` | Task 13 |

**New tests**

`tests/test_committee_protocol.py`, `tests/test_committee_worker.py`, `tests/test_committee_remote.py`, `tests/test_committee_stats.py`, `tests/test_committee_calculator.py`, `tests/test_committee_theory.py`, `tests/test_committee_config.py`, `tests/test_committee_outputs.py`, `tests/test_committee_optimize.py`, `tests/test_committee_cli.py`, plus non-collected stub workers under `tests/committee_stubs/`.

**Out of scope for this plan** (from the spec): MD and NEB; `optimize batch --committee`; a `discover` helper that writes `committee.yaml`; managed environments; a project `CANON.md`.

---

## Task 1: Wire protocol

**Files:**
- Create: `src/mliprun/core/committee/__init__.py` (empty)
- Create: `src/mliprun/core/committee/protocol.py`
- Test: `tests/test_committee_protocol.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PROTOCOL_VERSION: int = 1`
  - `class ProtocolError(RuntimeError)`
  - `encode(message: dict) -> bytes` — one newline-terminated UTF-8 line
  - `decode(line: bytes | str) -> dict`
  - `load_request(mlip: str, *, uma_task=None, mace_head=None, sevennet_task=None, device="auto") -> dict`
  - `calc_request(numbers: list[int], positions: list[list[float]], cell: list[list[float]], pbc: list[bool]) -> dict`
  - `quit_request() -> dict`
  - `ok(**fields) -> dict`
  - `error(message, traceback_text: str | None = None) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_committee_protocol.py`:

```python
"""Wire-protocol framing for the committee bridge."""
import json

import pytest

from mliprun.core.committee import protocol


class TestEncodeDecode:
    def test_round_trip_preserves_a_calc_request(self):
        request = protocol.calc_request(
            numbers=[29, 29],
            positions=[[0.0, 0.0, 0.0], [1.8, 1.8, 0.0]],
            cell=[[3.6, 0.0, 0.0], [0.0, 3.6, 0.0], [0.0, 0.0, 3.6]],
            pbc=[True, True, True],
        )
        assert protocol.decode(protocol.encode(request)) == request

    def test_encoded_message_is_exactly_one_line(self):
        blob = protocol.encode(protocol.calc_request([29], [[0.0, 0.0, 0.0]],
                                                     [[1.0, 0.0, 0.0]] * 3,
                                                     [True, True, True]))
        assert blob.endswith(b"\n")
        assert blob.count(b"\n") == 1

    def test_encode_rejects_nan(self):
        """A NaN energy must fail loudly in the member's own process rather
        than poison the committee mean."""
        with pytest.raises(protocol.ProtocolError):
            protocol.encode(protocol.ok(energy=float("nan")))

    def test_encode_rejects_infinity(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.encode(protocol.ok(energy=float("inf")))

    def test_encode_rejects_unserializable_value(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.encode({"cmd": "calc", "positions": object()})

    def test_decode_accepts_str_and_bytes_identically(self):
        text = json.dumps({"ok": True, "energy": -1.5})
        assert protocol.decode(text) == protocol.decode(text.encode("utf-8"))

    def test_decode_rejects_malformed_json(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(b'{"ok": true')

    def test_decode_rejects_an_empty_line(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(b"\n")

    def test_decode_rejects_a_json_array(self):
        """A bare list is valid JSON but not a message."""
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(b"[1, 2, 3]")

    def test_decode_rejects_invalid_utf8(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(b"\xff\xfe\n")


class TestMessages:
    def test_load_request_carries_every_selector(self):
        request = protocol.load_request("uma-s-1p2", uma_task="oc20",
                                        device="cuda")
        assert request["cmd"] == "load"
        assert request["protocol"] == protocol.PROTOCOL_VERSION
        assert request["mlip"] == "uma-s-1p2"
        assert request["uma_task"] == "oc20"
        assert request["mace_head"] is None
        assert request["sevennet_task"] is None
        assert request["device"] == "cuda"

    def test_calc_request_coerces_sequences_to_lists(self):
        request = protocol.calc_request((29, 29), [[0.0] * 3] * 2,
                                        [[1.0, 0.0, 0.0]] * 3, (True,) * 3)
        assert request["numbers"] == [29, 29]
        assert request["pbc"] == [True, True, True]

    def test_ok_and_error_are_distinguishable(self):
        assert protocol.ok(energy=-1.0)["ok"] is True
        failed = protocol.error(ValueError("boom"), "Traceback ...")
        assert failed["ok"] is False
        assert failed["error"] == "boom"
        assert failed["traceback"] == "Traceback ..."

    def test_error_without_traceback_is_an_empty_string(self):
        assert protocol.error("boom")["traceback"] == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_protocol.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'mliprun.core.committee'`

- [ ] **Step 3: Write the implementation**

Create `src/mliprun/core/committee/__init__.py` as an empty file (every other package `__init__` in this repo is empty; keep it that way so importing the protocol drags nothing in).

Create `src/mliprun/core/committee/protocol.py`:

```python
"""Line-delimited JSON wire protocol for the committee bridge.

Stdlib only, deliberately: this module is imported by
:mod:`mliprun.core.committee.worker` inside every MLIP env, where the torch /
torch-geometric / e3nn / ASE stacks are mutually incompatible with the
driver's (ADR 0001). Anything imported here would have to be installable in
all of them at once. Do not add numpy, ASE, or a sibling mliprun module.

One JSON object per line, in both directions:

    -> {"cmd": "load", "mlip": "uma-s-1p2", "uma_task": "oc20", ...}
    <- {"ok": true, "t_load": 22.8, "versions": {...}}
    -> {"cmd": "calc", "numbers": [...], "positions": [[...]], ...}
    <- {"ok": true, "energy": -73.679, "forces": [[...]], "t_calc": 0.207}
    -> {"cmd": "quit"}

Design: docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md
"""
from __future__ import annotations

import json

#: Bumped when the message shape changes incompatibly. Sent with every
#: ``load`` so a driver talking to a stale worker install can say so.
PROTOCOL_VERSION = 1


class ProtocolError(RuntimeError):
    """A message could not be encoded, decoded, or understood."""


def encode(message: dict) -> bytes:
    """Serialize one message to a single newline-terminated UTF-8 line.

    ``allow_nan=False`` is load-bearing rather than cosmetic: a member that
    returns a NaN energy or force fails *here*, in its own process, with its
    own name attached, instead of propagating a silent NaN into the
    committee mean. It also keeps the stream strict-JSON, which bare
    ``NaN``/``Infinity`` tokens are not.

    Parameters
    ----------
    message : dict
        The message. Values must already be JSON-native: call ``.tolist()``
        on numpy arrays before getting here, because this module never
        imports numpy.

    Returns
    -------
    bytes
        UTF-8, exactly one trailing newline, no interior newline.

    Raises
    ------
    ProtocolError
        If the message contains a non-finite float, a value JSON cannot
        represent, or an embedded newline.
    """
    try:
        text = json.dumps(message, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"could not encode message: {exc}") from exc
    if "\n" in text:
        raise ProtocolError("encoded message contains a newline")
    return (text + "\n").encode("utf-8")


def decode(line) -> dict:
    """Parse one line into a message.

    Raises
    ------
    ProtocolError
        On invalid UTF-8, an empty line, malformed JSON, or valid JSON that
        is not an object (a bare list is not a message).
    """
    if isinstance(line, (bytes, bytearray)):
        try:
            line = bytes(line).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError(f"message was not valid UTF-8: {exc}") from exc
    line = line.strip()
    if not line:
        raise ProtocolError("empty message")
    try:
        message = json.loads(line)
    except ValueError as exc:
        raise ProtocolError(f"could not decode message: {exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolError("message did not decode to a JSON object")
    return message


def load_request(mlip: str, *, uma_task=None, mace_head=None,
                 sevennet_task=None, device: str = "auto") -> dict:
    """Build the one-time model-load request.

    Every selector is sent explicitly, ``None`` included, so the worker never
    has to guess which of them the driver meant to omit. CANON C1/C3: heads
    and tasks are independent fine-tunes with independent energy zeros.
    """
    return {
        "cmd": "load",
        "protocol": PROTOCOL_VERSION,
        "mlip": mlip,
        "uma_task": uma_task,
        "mace_head": mace_head,
        "sevennet_task": sevennet_task,
        "device": device,
    }


def calc_request(numbers, positions, cell, pbc) -> dict:
    """Build a single-point request from a plain geometry.

    ``positions`` and ``cell`` must already be nested lists of floats.
    Constraints are deliberately absent: they stay driver-side, so the worker
    always returns raw forces and ASE applies ``FixAtoms`` in the driver
    exactly as it does for a single-model run (canon S8).
    """
    return {
        "cmd": "calc",
        "numbers": [int(z) for z in numbers],
        "positions": [list(row) for row in positions],
        "cell": [list(row) for row in cell],
        "pbc": [bool(p) for p in pbc],
    }


def quit_request() -> dict:
    """Build the orderly-shutdown request."""
    return {"cmd": "quit"}


def ok(**fields) -> dict:
    """Build a success response."""
    return {"ok": True, **fields}


def error(message, traceback_text: str | None = None) -> dict:
    """Build a failure response.

    The remote traceback travels with the message because the driver has no
    other way to see inside the member's env.
    """
    return {"ok": False, "error": str(message),
            "traceback": traceback_text or ""}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_protocol.py -q -rs`
Expected: 14 passed, 0 skipped.

- [ ] **Step 5: Confirm the stdlib-only constraint holds**

Run:
```bash
python -c "
import subprocess, sys, json
code = 'import sys; import mliprun.core.committee.protocol as p; print(json.dumps(sorted(m for m in sys.modules if m.split(\".\")[0] in (\"numpy\",\"ase\",\"torch\",\"pandas\",\"matplotlib\",\"typer\")))) if False else print([m for m in sys.modules if m.split(\".\")[0] in (\"numpy\",\"ase\",\"torch\",\"pandas\",\"matplotlib\",\"typer\")])'
print(subprocess.run([sys.executable, '-c', 'import json' + chr(10) + code], capture_output=True, text=True).stdout)
"
```
Simpler equivalent, use this one:
```bash
python -c "import sys, mliprun.core.committee.protocol; print([m for m in sys.modules if m.split('.')[0] in ('numpy','ase','torch','pandas','matplotlib','typer')])"
```
Expected: `[]`

- [ ] **Step 6: Commit**

```bash
git add src/mliprun/core/committee/__init__.py src/mliprun/core/committee/protocol.py tests/test_committee_protocol.py
git commit -m "feat(committee): line-delimited JSON wire protocol

Stdlib-only framing shared verbatim by the driver and every MLIP env's
worker. allow_nan=False makes a NaN energy fail in the member's own
process instead of poisoning the committee mean.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Task 2: Worker subprocess

**Files:**
- Create: `src/mliprun/core/committee/worker.py`
- Create: `tests/committee_stubs/__init__.py` (empty)
- Create: `tests/committee_stubs/noisy_worker.py`
- Test: `tests/test_committee_worker.py`

**Interfaces:**
- Consumes: `protocol.decode`, `protocol.encode`, `protocol.ok`, `protocol.error` (Task 1).
- Produces:
  - `EMT_TAG: str = "emt"` — reserved tag building ASE's built-in EMT
  - `_protect_stdout() -> BinaryIO` — moves protocol traffic off fd 1
  - `_build_calculator(request: dict) -> Calculator`
  - `main(argv: list[str] | None = None) -> int` — the request loop
  - runnable as `python -m mliprun.core.committee.worker`

- [ ] **Step 1: Write the failing tests**

Create `tests/committee_stubs/__init__.py` as an empty file, and `tests/committee_stubs/noisy_worker.py`:

```python
"""A worker whose model load prints to stdout, like MACE/SevenNet/CHGNet do.

Not a test module: it is launched as a subprocess by
tests/test_committee_worker.py. The print happens inside
``_build_calculator``, i.e. *after* ``_protect_stdout`` has run, which is
exactly where a real library banner appears.
"""
import sys

from ase.calculators.emt import EMT

from mliprun.core.committee import worker


def _noisy_build(request):
    print("BANNER: loading model weights")
    sys.stdout.write("more chatter from the library\n")
    sys.stdout.flush()
    return EMT()


worker._build_calculator = _noisy_build

if __name__ == "__main__":
    sys.exit(worker.main())
```

Create `tests/test_committee_worker.py`:

```python
"""The committee worker: one MLIP env's side of the bridge.

Exercised through a real subprocess under the current interpreter, building
ASE's EMT. No MLIP is needed.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from mliprun.core.committee import protocol

STUB_DIR = Path(__file__).parent / "committee_stubs"


def _spawn(argv, log_path):
    return subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=open(log_path, "wb"), close_fds=True,
    )


def _exchange(proc, request):
    proc.stdin.write(protocol.encode(request))
    proc.stdin.flush()
    return protocol.decode(proc.stdout.readline())


def _geometry():
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
    atoms.positions[0] += [0.05, 0.0, 0.0]
    return atoms


@pytest.fixture
def worker_proc(tmp_path):
    proc = _spawn([sys.executable, "-m", "mliprun.core.committee.worker"],
                  tmp_path / "worker.log")
    yield proc
    try:
        proc.kill()
        proc.wait(timeout=10)
    except Exception:
        pass


class TestLoadAndCalc:
    def test_load_reports_versions(self, worker_proc):
        response = _exchange(worker_proc, protocol.load_request("emt"))
        assert response["ok"] is True
        assert response["versions"]["python"] == ".".join(
            str(n) for n in sys.version_info[:3])
        assert response["versions"]["ase"] is not None
        assert response["t_load"] >= 0.0

    def test_calc_matches_in_process_emt_exactly(self, worker_proc):
        atoms = _geometry()
        _exchange(worker_proc, protocol.load_request("emt"))
        response = _exchange(worker_proc, protocol.calc_request(
            atoms.get_atomic_numbers(), atoms.get_positions().tolist(),
            np.asarray(atoms.get_cell()).tolist(), atoms.get_pbc()))

        reference = _geometry()
        reference.calc = EMT()
        assert response["energy"] == pytest.approx(
            reference.get_potential_energy(), abs=1e-10)
        assert np.asarray(response["forces"]) == pytest.approx(
            reference.get_forces(), abs=1e-10)

    def test_calc_before_load_is_an_error_not_a_crash(self, worker_proc):
        response = _exchange(worker_proc, protocol.calc_request(
            [29], [[0.0, 0.0, 0.0]], [[3.6, 0.0, 0.0], [0.0, 3.6, 0.0],
                                      [0.0, 0.0, 3.6]], [True, True, True]))
        assert response["ok"] is False
        assert "load" in response["error"]

    def test_unknown_tag_returns_a_remote_traceback(self, worker_proc):
        response = _exchange(worker_proc,
                             protocol.load_request("not-a-real-model"))
        assert response["ok"] is False
        assert "not-a-real-model" in response["error"]
        assert "Traceback" in response["traceback"]

    def test_unknown_command_is_rejected(self, worker_proc):
        response = _exchange(worker_proc, {"cmd": "dance"})
        assert response["ok"] is False
        assert "dance" in response["error"]

    def test_garbage_line_does_not_kill_the_worker(self, worker_proc):
        worker_proc.stdin.write(b"this is not json\n")
        worker_proc.stdin.flush()
        assert protocol.decode(worker_proc.stdout.readline())["ok"] is False
        # Still alive and still usable.
        assert _exchange(worker_proc, protocol.load_request("emt"))["ok"] is True

    def test_quit_exits_zero(self, worker_proc):
        _exchange(worker_proc, protocol.load_request("emt"))
        assert _exchange(worker_proc, protocol.quit_request())["ok"] is True
        assert worker_proc.wait(timeout=30) == 0

    def test_closed_stdin_exits_zero(self, worker_proc):
        worker_proc.stdin.close()
        assert worker_proc.wait(timeout=30) == 0


class TestStdoutProtection:
    def test_library_chatter_goes_to_the_log_not_the_stream(self, tmp_path):
        """MACE, SevenNet and CHGNet print banners on import and first
        inference. Anything on fd 1 would corrupt the protocol stream."""
        log_path = tmp_path / "worker.log"
        proc = _spawn([sys.executable, str(STUB_DIR / "noisy_worker.py")],
                      log_path)
        try:
            response = _exchange(proc, protocol.load_request("emt"))
            assert response["ok"] is True
            atoms = _geometry()
            calc_response = _exchange(proc, protocol.calc_request(
                atoms.get_atomic_numbers(), atoms.get_positions().tolist(),
                np.asarray(atoms.get_cell()).tolist(), atoms.get_pbc()))
            reference = _geometry()
            reference.calc = EMT()
            assert calc_response["energy"] == pytest.approx(
                reference.get_potential_energy(), abs=1e-10)
        finally:
            proc.kill()
            proc.wait(timeout=10)

        log = log_path.read_text(encoding="utf-8", errors="replace")
        assert "BANNER: loading model weights" in log
        assert "more chatter from the library" in log
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_worker.py -q`
Expected: every test fails — the subprocess exits non-zero with `No module named mliprun.core.committee.worker`, so `readline()` returns `b""` and `decode` raises `ProtocolError: empty message`.

- [ ] **Step 3: Write the implementation**

Create `src/mliprun/core/committee/worker.py`:

```python
"""Committee member worker: one MLIP env's side of the bridge.

Launched as ``python -m mliprun.core.committee.worker`` by the *member's own*
interpreter, so the torch/ASE stack it imports is that env's, never the
driver's (ADR 0001). Reads requests on stdin, writes responses on a private
duplicate of stdout, and keeps every library banner away from the stream.

mliprun must be installed in the member env (``pip install -e .`` there);
the MLIP package it loads is that env's single MLIP.

Design: docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md
"""
from __future__ import annotations

import os
import sys
import time
import traceback

from mliprun.core.committee import protocol

#: Reserved tag: ASE's built-in EMT calculator. Not an MLIP. It exists so the
#: bridge itself can be exercised -- in CI, and by a user smoke-testing a
#: committee.yaml -- on a machine with no MLIP installed at all. EMT is a
#: toy potential for a handful of metals: never use it for science.
EMT_TAG = "emt"


def _protect_stdout():
    """Move protocol traffic off fd 1 and point fd 1 at stderr.

    MACE, SevenNet and CHGNet all print banners to stdout on import and on
    first inference. Anything they print would land mid-stream and corrupt
    the protocol, so fd 1 is redirected onto fd 2 -- which the driver
    captures into this member's worker log -- and protocol traffic goes out
    on a private duplicate that no library can reach.

    ``sys.stdout`` is deliberately left alone: it still wraps fd 1, which now
    *is* stderr, so a library's ``print`` keeps working and lands in the log.

    Returns
    -------
    BinaryIO
        Unbuffered binary stream for protocol lines.
    """
    sys.stdout.flush()
    private_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(private_fd, "wb", buffering=0)


def _build_calculator(request: dict):
    """Build this member's ASE calculator from a ``load`` request.

    Everything except the reserved EMT tag goes through the same
    :func:`mliprun.cli.utils.build_calculator` a single-model run uses, so a
    committee member is loaded exactly the way ``mlip optimize`` would load
    it -- including the missing-head and missing-task guards (CANON C1/C3).
    """
    mlip = request.get("mlip")
    if mlip == EMT_TAG:
        from ase.calculators.emt import EMT
        return EMT()

    from mliprun.cli.utils import build_calculator
    calc = build_calculator(
        mlip,
        request.get("uma_task"),
        device=request.get("device", "auto"),
        mace_head=request.get("mace_head"),
        sevennet_task=request.get("sevennet_task"),
    )
    if calc is None:
        raise ValueError(f"unknown MLIP tag: {mlip!r}")
    return calc


def _versions(mlip) -> dict:
    """Report what this env actually is, for the run record.

    Every lookup is individually guarded: an incomplete version block is far
    better than a member that refuses to load because ``torch`` is absent
    (CHGNet-only envs, CPU-only installs).
    """
    import platform
    from importlib.metadata import PackageNotFoundError, version

    info = {"python": platform.python_version(), "executable": sys.executable}
    for label, dist in (("mliprun", "mliprun"), ("ase", "ase"),
                        ("torch", "torch")):
        try:
            info[label] = version(dist)
        except PackageNotFoundError:
            info[label] = None
        except Exception:  # noqa: BLE001 -- version reporting is best-effort
            info[label] = None
    try:
        # Module-private by name, reused inside the package deliberately:
        # it is the one table mapping an MLIP tag to its distribution.
        from mliprun.core.run_record import _mlip_package
        package = _mlip_package(mlip)
    except Exception:  # noqa: BLE001
        package = {"name": None, "version": None}
    info["package"] = package["name"]
    info["package_version"] = package["version"]
    return info


def _handle_calc(calc, request: dict) -> dict:
    """Single-point the requested geometry.

    A fresh ``Atoms`` is built from plain arrays every call -- no pickled
    object crosses the pipe, because one committee can hold ASE 3.28.0 and
    3.29.0 at once. It carries no constraints, which is correct: constraints
    stay driver-side and this must return *raw* forces.
    """
    import numpy as np
    from ase import Atoms

    atoms = Atoms(
        numbers=request["numbers"],
        positions=request["positions"],
        cell=request["cell"],
        pbc=request["pbc"],
    )
    atoms.calc = calc
    t0 = time.perf_counter()
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return protocol.ok(energy=energy, forces=forces.tolist(),
                       t_calc=round(time.perf_counter() - t0, 6))


def main(argv=None) -> int:
    """Serve requests until ``quit`` or the driver closes the pipe."""
    out = _protect_stdout()
    stdin = sys.stdin.buffer
    calc = None

    while True:
        line = stdin.readline()
        if not line:
            # Driver went away. Exiting 0 keeps a clean teardown clean.
            return 0
        try:
            request = protocol.decode(line)
        except protocol.ProtocolError as exc:
            out.write(protocol.encode(protocol.error(exc)))
            continue

        command = request.get("cmd")
        try:
            if command == "load":
                t0 = time.perf_counter()
                calc = _build_calculator(request)
                response = protocol.ok(
                    t_load=round(time.perf_counter() - t0, 3),
                    versions=_versions(request.get("mlip")),
                )
            elif command == "calc":
                if calc is None:
                    raise RuntimeError("calc requested before load")
                response = _handle_calc(calc, request)
            elif command == "quit":
                out.write(protocol.encode(protocol.ok()))
                return 0
            else:
                raise ValueError(f"unknown command: {command!r}")
        except Exception as exc:  # noqa: BLE001 -- every failure is reportable
            response = protocol.error(exc, traceback.format_exc())

        try:
            out.write(protocol.encode(response))
        except protocol.ProtocolError as exc:
            # The response itself could not be encoded -- a non-finite energy
            # or force. Report that as this member's failure; the driver
            # aborts the run rather than averaging a NaN.
            out.write(protocol.encode(protocol.error(exc)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_worker.py -q -rs`
Expected: 9 passed, 0 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/committee/worker.py tests/committee_stubs tests/test_committee_worker.py
git commit -m "feat(committee): worker subprocess for one MLIP env

Runs under the member env's own interpreter, builds its calculator through
the same build_calculator a single-model run uses, and takes a private dup
of fd 1 so library banners land in the worker log instead of corrupting the
protocol stream. Reserved tag 'emt' makes the bridge testable with no MLIP.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Task 3: RemoteMember — process lifecycle, timeouts, teardown

**Files:**
- Create: `src/mliprun/core/committee/remote.py`
- Create: `tests/committee_stubs/hanging_worker.py`
- Create: `tests/committee_stubs/dying_worker.py`
- Test: `tests/test_committee_remote.py`

**Interfaces:**
- Consumes: `protocol.encode`, `protocol.decode`, `protocol.load_request`, `protocol.calc_request`, `protocol.quit_request`, `protocol.ProtocolError` (Task 1); `worker.main` via `python -m mliprun.core.committee.worker` (Task 2).
- Produces:
  - `DEFAULT_CALC_TIMEOUT_S: float = 300.0`
  - `DEFAULT_LOAD_TIMEOUT_S: float = 1800.0`
  - `class MemberError(RuntimeError)` with attributes `member: str`, `log_path: str | None`, `remote_traceback: str`
  - `class RemoteMember` with
    - `__init__(name, python_exe, *, mlip, uma_task=None, mace_head=None, sevennet_task=None, device="auto", gpu=None, log_path=None, timeout=DEFAULT_CALC_TIMEOUT_S, load_timeout=DEFAULT_LOAD_TIMEOUT_S, argv=None)`
    - `start() -> dict` — spawn + load handshake, returns the versions block
    - `calculate(numbers, positions, cell, pbc) -> tuple[float, list[list[float]]]`
    - `close(timeout: float = 5.0) -> None` — idempotent
    - attributes `name`, `pid` (kept after close), `versions`, `is_alive`

- [ ] **Step 1: Write the stub workers and the failing tests**

Create `tests/committee_stubs/hanging_worker.py`:

```python
"""A worker that loads fine and then never answers a calc request.

Not a test module: launched as a subprocess to exercise the per-request
timeout. It must also ignore SIGTERM, so the test proves the teardown
escalates to SIGKILL rather than merely asking politely.
"""
import signal
import sys
import time

from ase.calculators.emt import EMT

from mliprun.core.committee import protocol, worker


def main():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    out = worker._protect_stdout()
    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            return 0
        request = protocol.decode(line)
        if request.get("cmd") == "load":
            EMT()
            out.write(protocol.encode(protocol.ok(t_load=0.0, versions={})))
        else:
            while True:
                time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
```

Create `tests/committee_stubs/dying_worker.py`:

```python
"""A worker that loads, answers N calc requests, then exits without replying.

Not a test module: launched as a subprocess to exercise the loud abort on a
dead member, and (later, in the optimize integration) partial-results
survival. N comes from MLIPRUN_STUB_DIE_AFTER, default 0.
"""
import os
import sys

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT

from mliprun.core.committee import protocol, worker


def main():
    out = worker._protect_stdout()
    stdin = sys.stdin.buffer
    remaining = int(os.environ.get("MLIPRUN_STUB_DIE_AFTER", "0"))
    calc = None
    while True:
        line = stdin.readline()
        if not line:
            return 0
        request = protocol.decode(line)
        if request.get("cmd") == "load":
            calc = EMT()
            out.write(protocol.encode(protocol.ok(t_load=0.0, versions={})))
        elif request.get("cmd") == "calc":
            if remaining <= 0:
                # Die mid-conversation, without replying.
                os._exit(7)
            remaining -= 1
            atoms = Atoms(numbers=request["numbers"],
                          positions=request["positions"],
                          cell=request["cell"], pbc=request["pbc"])
            atoms.calc = calc
            out.write(protocol.encode(protocol.ok(
                energy=float(atoms.get_potential_energy()),
                forces=np.asarray(atoms.get_forces()).tolist(),
                t_calc=0.0)))
        else:
            out.write(protocol.encode(protocol.ok()))


if __name__ == "__main__":
    sys.exit(main())
```

Create `tests/test_committee_remote.py`:

```python
"""RemoteMember: process lifecycle, timeouts, and teardown."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from mliprun.core.committee.remote import MemberError, RemoteMember

STUB_DIR = Path(__file__).parent / "committee_stubs"


def _geometry():
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
    atoms.positions[0] += [0.05, 0.0, 0.0]
    return atoms


def _as_request(atoms):
    return (atoms.get_atomic_numbers(), atoms.get_positions().tolist(),
            np.asarray(atoms.get_cell()).tolist(), atoms.get_pbc())


def _pid_is_gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


@pytest.fixture
def member(tmp_path):
    m = RemoteMember("member_a", sys.executable, mlip="emt",
                     log_path=tmp_path / "member_a.log")
    yield m
    m.close()


class TestRoundTrip:
    def test_start_returns_the_version_block(self, member):
        versions = member.start()
        assert versions["ase"] is not None
        assert versions["executable"] == sys.executable
        assert member.versions is versions

    def test_calculate_matches_in_process_emt_exactly(self, member):
        member.start()
        atoms = _geometry()
        energy, forces = member.calculate(*_as_request(atoms))

        reference = _geometry()
        reference.calc = EMT()
        assert energy == pytest.approx(reference.get_potential_energy(),
                                       abs=1e-10)
        assert np.asarray(forces) == pytest.approx(reference.get_forces(),
                                                   abs=1e-10)

    def test_repeated_calls_reuse_one_process(self, member):
        member.start()
        pid = member.pid
        atoms = _geometry()
        for shift in (0.0, 0.01, 0.02):
            moved = _geometry()
            moved.positions[0][0] += shift
            member.calculate(*_as_request(moved))
        assert member.pid == pid
        assert member.is_alive

    def test_close_is_idempotent_and_reaps_the_process(self, member):
        member.start()
        pid = member.pid
        member.close()
        member.close()
        assert _pid_is_gone(pid)
        assert not member.is_alive


class TestFailureModes:
    def test_missing_interpreter_fails_at_start(self, tmp_path):
        member = RemoteMember("member_a", tmp_path / "no" / "such" / "python",
                              mlip="emt", log_path=tmp_path / "a.log")
        with pytest.raises(MemberError) as excinfo:
            member.start()
        assert "member_a" in str(excinfo.value)

    def test_load_failure_carries_the_remote_traceback(self, tmp_path):
        member = RemoteMember("member_a", sys.executable,
                              mlip="not-a-real-model",
                              log_path=tmp_path / "a.log")
        try:
            with pytest.raises(MemberError) as excinfo:
                member.start()
            message = str(excinfo.value)
            assert "not-a-real-model" in message
            assert "Traceback" in message
            assert str(tmp_path / "a.log") in message
        finally:
            member.close()

    def test_calc_before_start_fails_loudly(self, tmp_path):
        member = RemoteMember("member_a", sys.executable, mlip="emt",
                              log_path=tmp_path / "a.log")
        with pytest.raises(MemberError):
            member.calculate(*_as_request(_geometry()))

    def test_timeout_kills_a_hung_member(self, tmp_path):
        """A hung worker must fail rather than stall the run overnight, and
        must actually die -- the stub ignores SIGTERM, so this also proves
        the escalation to SIGKILL."""
        member = RemoteMember(
            "member_a", sys.executable, mlip="emt",
            log_path=tmp_path / "a.log", timeout=2.0,
            argv=[sys.executable, str(STUB_DIR / "hanging_worker.py")])
        member.start()
        pid = member.pid
        try:
            with pytest.raises(MemberError) as excinfo:
                member.calculate(*_as_request(_geometry()))
            assert "2 s" in str(excinfo.value)
        finally:
            member.close()
        assert _pid_is_gone(pid)

    def test_dead_member_aborts_loudly(self, tmp_path):
        member = RemoteMember(
            "member_a", sys.executable, mlip="emt",
            log_path=tmp_path / "a.log",
            argv=[sys.executable, str(STUB_DIR / "dying_worker.py")])
        member.start()
        try:
            with pytest.raises(MemberError) as excinfo:
                member.calculate(*_as_request(_geometry()))
            assert "exited" in str(excinfo.value)
            assert "7" in str(excinfo.value)
        finally:
            member.close()

    def test_worker_stderr_lands_in_the_log(self, tmp_path):
        log_path = tmp_path / "member_a.log"
        member = RemoteMember("member_a", sys.executable,
                              mlip="not-a-real-model", log_path=log_path)
        try:
            with pytest.raises(MemberError):
                member.start()
        finally:
            member.close()
        assert log_path.exists()


class TestEnvironment:
    def test_gpu_sets_cuda_visible_devices_for_that_member_only(self, tmp_path,
                                                               monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        member = RemoteMember(
            "member_a", sys.executable, mlip="emt",
            log_path=tmp_path / "a.log", gpu=2,
            argv=[sys.executable, "-c",
                  "import os, sys; sys.stderr.write("
                  "os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')); "
                  "sys.stderr.flush()"])
        try:
            with pytest.raises(MemberError):
                member.start()   # the stub never replies; it just exits
        finally:
            member.close()
        assert (tmp_path / "a.log").read_text().strip() == "2"
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_remote.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'mliprun.core.committee.remote'`

- [ ] **Step 3: Write the implementation**

Create `src/mliprun/core/committee/remote.py`:

```python
"""One committee member, running in its own MLIP env's interpreter.

Owns the subprocess: spawn, load handshake, request/response with a deadline,
and teardown. POSIX only -- ``select`` on a pipe does not work on Windows,
and the committee bridge is not supported there.

Design: docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md
"""
from __future__ import annotations

import atexit
import logging
import os
import select
import signal
import subprocess
import textwrap
import time
import weakref

from mliprun.core.committee import protocol

logger = logging.getLogger(__name__)

#: Per-request budget. A UMA single point costs ~0.2 s and a large system
#: rather more; 300 s is "something is wrong", not "this is slow".
DEFAULT_CALC_TIMEOUT_S = 300.0

#: Model load can include a first-use checkpoint download (SevenNet ships
#: ~103 MB per checkpoint into site-packages; UMA pulls from Hugging Face),
#: so the load budget is far larger than the per-call one. On an air-gapped
#: node, pre-fetch the checkpoints per the install recipe rather than
#: raising this.
DEFAULT_LOAD_TIMEOUT_S = 1800.0

_QUIT_GRACE_S = 5.0

#: Live members, for the atexit backstop. Weak so a garbage-collected member
#: is not kept alive by the registry itself.
_LIVE: "weakref.WeakSet" = weakref.WeakSet()


class MemberError(RuntimeError):
    """A committee member failed.

    Carries the member's name, the path to its worker log, and the remote
    traceback, because the driver has no other window into that env.
    """

    def __init__(self, member: str, message, log_path=None,
                 remote_traceback: str = ""):
        self.member = member
        self.log_path = str(log_path) if log_path else None
        self.remote_traceback = remote_traceback or ""
        text = f"committee member '{member}' failed: {message}"
        if self.log_path:
            text += f"\n  worker log: {self.log_path}"
        if self.remote_traceback:
            text += ("\n  remote traceback:\n"
                     + textwrap.indent(self.remote_traceback.rstrip(), "    "))
        super().__init__(text)


@atexit.register
def _kill_leftover_members() -> None:
    """Last-resort teardown for anything still alive at interpreter exit.

    cos-cluster has no scheduler to reap orphans, and a leaked worker holds a
    CUDA context that makes the GPU look busy to everyone else on the node.
    """
    for member in list(_LIVE):
        try:
            member.close(timeout=0.5)
        except Exception:  # noqa: BLE001 -- nothing useful to do at exit
            pass


class RemoteMember:
    """One MLIP env's worker process, addressed as a calculator.

    Parameters
    ----------
    name : str
        Stable identifier. Becomes a CSV column header and the label in
        every error message, so keep it short and filesystem-safe.
    python_exe : str or Path
        The member env's interpreter -- ``<env>/bin/python``.
    mlip : str
        MLIP tag for that env, or ``"emt"`` for the reserved test member.
    uma_task, mace_head, sevennet_task : str, optional
        Head/task selectors, forwarded verbatim. Independent fine-tunes with
        independent energy zeros (CANON C1/C3): never defaulted here.
    device : str
        ``"auto"``, ``"cuda"`` or ``"cpu"``, resolved inside the member env
        (the driver may have no torch at all).
    gpu : int, optional
        Sets ``CUDA_VISIBLE_DEVICES`` for this member's process only. Omitted,
        the member inherits the driver's environment.
    log_path : str or Path, optional
        File that receives the worker's stderr -- library banners, remote
        tracebacks. ``None`` inherits the driver's stderr.
    timeout : float
        Per-request deadline in seconds.
    load_timeout : float
        Deadline for the one-time model load.
    argv : list of str, optional
        Full command line for the worker. Defaults to
        ``[python_exe, "-m", "mliprun.core.committee.worker"]``. Override it
        to wrap the interpreter in a launcher (``srun``, a module-load shim),
        or to point at a stub in tests.
    """

    def __init__(self, name, python_exe, *, mlip, uma_task=None,
                 mace_head=None, sevennet_task=None, device="auto", gpu=None,
                 log_path=None, timeout=DEFAULT_CALC_TIMEOUT_S,
                 load_timeout=DEFAULT_LOAD_TIMEOUT_S, argv=None):
        self.name = str(name)
        self.python_exe = str(python_exe)
        self.mlip = mlip
        self.uma_task = uma_task
        self.mace_head = mace_head
        self.sevennet_task = sevennet_task
        self.device = device
        self.gpu = gpu
        self.log_path = str(log_path) if log_path else None
        self.timeout = float(timeout)
        self.load_timeout = float(load_timeout)
        self.argv = (list(argv) if argv else
                     [self.python_exe, "-m", "mliprun.core.committee.worker"])
        #: Kept after teardown so a caller can still say which process ran.
        self.pid = None
        self.versions: dict = {}
        self._proc = None
        self._log = None
        self._buffer = bytearray()

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> dict:
        """Spawn the worker and load its model. Returns the versions block.

        Raises
        ------
        MemberError
            If the interpreter is missing, mliprun is not installed in that
            env, the MLIP package is absent, or the model refuses to load.
            Every one of these surfaces here -- before the optimizer starts.
        """
        env = dict(os.environ)
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        # Unbuffered, so a worker that dies mid-import still leaves its
        # traceback in the log.
        env["PYTHONUNBUFFERED"] = "1"

        if self.log_path:
            try:
                self._log = open(self.log_path, "wb")
            except OSError as exc:
                raise MemberError(self.name,
                                  f"could not open worker log: {exc}") from exc

        try:
            self._proc = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log if self._log is not None else None,
                env=env,
                close_fds=True,
                # Own process group, so teardown can kill the worker and
                # anything it spawned (torch dataloader workers) in one go.
                start_new_session=True,
            )
        except OSError as exc:
            raise MemberError(
                self.name,
                f"could not start worker: {exc} (command: {' '.join(self.argv)})",
                self.log_path) from exc

        self.pid = self._proc.pid
        _LIVE.add(self)
        response = self._exchange(
            protocol.load_request(self.mlip, uma_task=self.uma_task,
                                  mace_head=self.mace_head,
                                  sevennet_task=self.sevennet_task,
                                  device=self.device),
            self.load_timeout)
        self.versions = response.get("versions", {})
        return self.versions

    def close(self, timeout: float = _QUIT_GRACE_S) -> None:
        """Shut the worker down. Idempotent, and safe on a half-started member."""
        proc = self._proc
        if proc is None:
            self._close_log()
            return
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write(protocol.encode(protocol.quit_request()))
                    proc.stdin.flush()
                except Exception:  # noqa: BLE001 -- already gone; escalate below
                    pass
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._kill()
        finally:
            for stream in (proc.stdin, proc.stdout):
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
            self._close_log()
            self._proc = None
            _LIVE.discard(self)

    def _close_log(self) -> None:
        if self._log is not None:
            try:
                self._log.close()
            except Exception:  # noqa: BLE001
                pass
            self._log = None

    def _kill(self) -> None:
        """SIGKILL the worker's whole process group, then reap it."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("committee member '%s' (pid %s) would not die",
                           self.name, proc.pid)

    # -- requests ----------------------------------------------------------

    def calculate(self, numbers, positions, cell, pbc):
        """Single-point this geometry. Returns ``(energy, forces)``.

        ``forces`` comes back as a nested list; the caller converts. Raw
        forces, with no constraints applied -- those stay driver-side.
        """
        response = self._exchange(
            protocol.calc_request(numbers, positions, cell, pbc), self.timeout)
        try:
            energy = float(response["energy"])
            forces = response["forces"]
        except (KeyError, TypeError, ValueError) as exc:
            raise MemberError(self.name, f"malformed calc response: {exc}",
                              self.log_path) from exc
        return energy, forces

    def _exchange(self, request: dict, timeout: float) -> dict:
        if not self.is_alive:
            raise MemberError(self.name, "worker is not running "
                              "(start() not called, or already closed)",
                              self.log_path)
        try:
            self._proc.stdin.write(protocol.encode(request))
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise MemberError(self.name, f"could not send request: {exc}",
                              self.log_path) from exc

        line = self._read_line(timeout)
        try:
            response = protocol.decode(line)
        except protocol.ProtocolError as exc:
            raise MemberError(self.name, str(exc), self.log_path) from exc
        if not response.get("ok"):
            raise MemberError(self.name,
                              response.get("error", "unknown error"),
                              self.log_path,
                              response.get("traceback", ""))
        return response

    def _read_line(self, timeout: float) -> bytes:
        """Read one newline-terminated line, or fail at the deadline.

        ``select`` rather than a blocking ``readline``: a hung member must
        fail, not stall the run overnight. Leftover bytes past the newline
        stay buffered for the next call, so a worker that writes two lines in
        one chunk is handled correctly.
        """
        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[:newline + 1]
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                raise MemberError(
                    self.name,
                    f"no response within {timeout:g} s (worker killed)",
                    self.log_path)
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                code = self._proc.poll()
                raise MemberError(
                    self.name,
                    f"worker exited (returncode={code}) before replying",
                    self.log_path)
            self._buffer.extend(chunk)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_remote.py -q -rs`
Expected: 11 passed, 0 skipped. `test_timeout_kills_a_hung_member` takes ~2 s; the whole file should finish well under a minute.

- [ ] **Step 5: Check for orphaned workers**

Run: `pytest tests/test_committee_remote.py -q && pgrep -f "mliprun.core.committee.worker" ; echo "exit=$?"`
Expected: `pgrep` prints nothing and `exit=1` (no matches). Any pid printed here is a teardown bug — fix it before committing.

- [ ] **Step 6: Commit**

```bash
git add src/mliprun/core/committee/remote.py tests/committee_stubs/hanging_worker.py tests/committee_stubs/dying_worker.py tests/test_committee_remote.py
git commit -m "feat(committee): RemoteMember process lifecycle and teardown

Spawns one worker per MLIP env in its own process group, times every
request out with select rather than blocking forever, and escalates
teardown to SIGKILL on the group. An atexit backstop kills leftovers: a
leaked worker holds a CUDA context that makes the GPU look busy to the
rest of the node.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Task 4: Committee arithmetic

The uncertainty metric itself, as a pure function — no subprocess, no ASE, no
committee object. Hand-computable, so it can be asserted exactly.

**Files:**
- Create: `src/mliprun/core/committee/calculator.py`
- Test: `tests/test_committee_stats.py`

**Interfaces:**
- Consumes: nothing (numpy only).
- Produces:
  - `committee_statistics(energies, forces) -> dict` with keys `energy_mean: float`, `forces_mean: np.ndarray (N, 3)`, `sigma_per_atom: np.ndarray (N,)`, `sigma_max: float`, `sigma_mean: float`, `worst_atom: int`
  - `aligned_energy_spread(energies: dict[str, float], baseline: dict[str, float]) -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_committee_stats.py`:

```python
"""Committee arithmetic: mean force field and per-atom disagreement."""
import numpy as np
import pytest

from mliprun.core.committee.calculator import (
    aligned_energy_spread,
    committee_statistics,
)


class TestCommitteeStatistics:
    def test_three_members_give_a_hand_computed_mean_and_sigma(self):
        """1.0, 3.0, 2.0 eV/A on one component: mean 2.0, std (ddof=1) 1.0."""
        forces = np.zeros((3, 1, 3))
        forces[0, 0, 0] = 1.0
        forces[1, 0, 0] = 3.0
        forces[2, 0, 0] = 2.0
        stats = committee_statistics([-1.0, -3.0, -2.0], forces)

        assert stats["energy_mean"] == pytest.approx(-2.0, abs=1e-12)
        assert stats["forces_mean"][0, 0] == pytest.approx(2.0, abs=1e-12)
        assert stats["sigma_per_atom"][0] == pytest.approx(1.0, abs=1e-12)
        assert stats["sigma_max"] == pytest.approx(1.0, abs=1e-12)
        assert stats["sigma_mean"] == pytest.approx(1.0, abs=1e-12)
        assert stats["worst_atom"] == 0

    def test_sigma_is_the_norm_of_the_per_component_std(self):
        """sigma_i = || std_across_members(F_i) ||, so 3-4-5 on the three
        components gives exactly 5."""
        forces = np.zeros((2, 1, 3))
        forces[0, 0] = [0.0, 0.0, 0.0]
        forces[1, 0] = [3.0 * np.sqrt(2), 4.0 * np.sqrt(2), 0.0]
        stats = committee_statistics([0.0, 0.0], forces)
        assert stats["sigma_per_atom"][0] == pytest.approx(5.0, abs=1e-12)

    def test_identical_members_give_zero_sigma(self):
        forces = np.tile(np.array([[[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]]),
                         (4, 1, 1))
        stats = committee_statistics([-7.5] * 4, forces)
        assert stats["sigma_max"] == pytest.approx(0.0, abs=1e-12)
        assert stats["forces_mean"] == pytest.approx(forces[0], abs=1e-12)

    def test_worst_atom_is_the_index_of_the_largest_sigma(self):
        forces = np.zeros((2, 3, 3))
        forces[1, 0, 0] = 0.1
        forces[1, 2, 0] = 0.9
        stats = committee_statistics([0.0, 0.0], forces)
        assert stats["worst_atom"] == 2
        assert stats["sigma_max"] == pytest.approx(
            stats["sigma_per_atom"][2], abs=1e-12)

    def test_sigma_mean_averages_over_atoms_not_members(self):
        forces = np.zeros((2, 4, 3))
        forces[1, 0, 0] = 2.0 * np.sqrt(2)   # sigma = 2.0 on atom 0 only
        stats = committee_statistics([0.0, 0.0], forces)
        assert stats["sigma_mean"] == pytest.approx(0.5, abs=1e-12)

    def test_one_member_is_rejected(self):
        """A committee of one has no disagreement to report."""
        with pytest.raises(ValueError, match="at least two"):
            committee_statistics([-1.0], np.zeros((1, 2, 3)))

    def test_mismatched_member_counts_are_rejected(self):
        with pytest.raises(ValueError):
            committee_statistics([-1.0, -2.0], np.zeros((3, 2, 3)))

    def test_wrong_force_rank_is_rejected(self):
        with pytest.raises(ValueError):
            committee_statistics([-1.0, -2.0], np.zeros((2, 3)))


class TestAlignedEnergySpread:
    def test_constant_offsets_cancel_exactly(self):
        """Removing each member's own step-0 energy collapses a spread that
        is pure offset to zero."""
        baseline = {"member_a": -73.68, "member_b": -82.86}
        energies = {"member_a": -73.68 - 1.5, "member_b": -82.86 - 1.5}
        assert aligned_energy_spread(energies, baseline) == pytest.approx(
            0.0, abs=1e-12)

    def test_real_disagreement_survives_alignment(self):
        baseline = {"member_a": -10.0, "member_b": -20.0}
        energies = {"member_a": -11.0, "member_b": -22.0}
        # Deltas -1.0 and -2.0; std with ddof=1 is 1/sqrt(2).
        assert aligned_energy_spread(energies, baseline) == pytest.approx(
            1.0 / np.sqrt(2.0), abs=1e-12)

    def test_missing_baseline_member_is_rejected(self):
        with pytest.raises(KeyError):
            aligned_energy_spread({"member_a": -1.0}, {"member_b": -1.0})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_stats.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'mliprun.core.committee.calculator'`

- [ ] **Step 3: Write the implementation**

Create `src/mliprun/core/committee/calculator.py`:

```python
"""Committee calculator: mean forces and per-configuration uncertainty.

Driver-side only. Nothing here runs inside an MLIP env.

Design: docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def committee_statistics(energies, forces) -> dict:
    """Reduce the members' energies and forces to a consensus and a spread.

    The uncertainty metric is the per-atom force disagreement

        sigma_i = || std_across_members( F_i ) ||

    -- the vector norm of the per-component standard deviation across
    members, ``ddof=1``. It behaves as an extrapolation detector: in the
    2026-09-04 probe it sat at 0.25-0.62 eV/A across the physical range of a
    CO height scan and rose to 3.48 eV/A at a strained geometry.

    The committee is a *proper potential*: the mean force is exactly the
    negative gradient of the mean energy, so the effective PES is well
    defined and line-search optimizers (``bfgsls``) behave correctly through
    it. Per-model constant offsets shift the mean energy by a constant
    without changing its shape.

    Parameters
    ----------
    energies : array-like, shape (M,)
        One energy per member, in the members' own order.
    forces : array-like, shape (M, N, 3)
        One force array per member, same order.

    Returns
    -------
    dict
        ``energy_mean`` (float), ``forces_mean`` (N, 3), ``sigma_per_atom``
        (N,), ``sigma_max`` (float), ``sigma_mean`` (float), ``worst_atom``
        (int).

    Raises
    ------
    ValueError
        If fewer than two members are given, or the shapes disagree. Two is
        the floor because ``ddof=1`` is undefined for one sample -- and
        because a committee of one has no disagreement to report.
    """
    energies = np.asarray(energies, dtype=float)
    forces = np.asarray(forces, dtype=float)
    if energies.ndim != 1:
        raise ValueError(f"energies must be 1-D, got shape {energies.shape}")
    if forces.ndim != 3 or forces.shape[2] != 3:
        raise ValueError(
            f"forces must have shape (n_members, n_atoms, 3), got {forces.shape}")
    if forces.shape[0] != energies.shape[0]:
        raise ValueError(
            f"{energies.shape[0]} energies but {forces.shape[0]} force arrays")
    if energies.shape[0] < 2:
        raise ValueError("a committee needs at least two members")

    sigma_components = forces.std(axis=0, ddof=1)          # (N, 3)
    sigma_per_atom = np.linalg.norm(sigma_components, axis=1)   # (N,)
    worst_atom = int(np.argmax(sigma_per_atom))
    return {
        "energy_mean": float(energies.mean()),
        "forces_mean": forces.mean(axis=0),
        "sigma_per_atom": sigma_per_atom,
        "sigma_max": float(sigma_per_atom[worst_atom]),
        "sigma_mean": float(sigma_per_atom.mean()),
        "worst_atom": worst_atom,
    }


def aligned_energy_spread(energies: dict, baseline: dict) -> float:
    """Spread of the members' energies after removing their own offsets.

    Raw energies are not comparable across packages -- the four-member probe
    saw a 4.81 eV spread at one fixed geometry, essentially all of it
    per-model constant offset. Subtracting each member's own step-0 energy
    collapsed that to 0.09 eV. Composition is fixed during a relaxation, so
    the offset cancels exactly and what remains is a real energy uncertainty.

    Returned as a standard deviation across members (``ddof=1``), matching
    the force metric -- not a max-minus-min range.

    Raises
    ------
    KeyError
        If a member present in ``energies`` has no baseline.
    """
    deltas = [energies[name] - baseline[name] for name in energies]
    return float(np.std(np.asarray(deltas, dtype=float), ddof=1))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_stats.py -q -rs`
Expected: 11 passed, 0 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/committee/calculator.py tests/test_committee_stats.py
git commit -m "feat(committee): mean forces and per-atom disagreement

sigma_i = ||std_across_members(F_i)|| with ddof=1, plus the offset-aligned
energy spread. Both are standard deviations, not ranges. Two members is a
hard floor: ddof=1 is undefined for one, and a committee of one has no
disagreement to report.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Task 5: CommitteeCalculator

**Files:**
- Modify: `src/mliprun/core/committee/calculator.py` (append)
- Test: `tests/test_committee_calculator.py`

**Interfaces:**
- Consumes: `committee_statistics` (Task 4); `RemoteMember`, `MemberError` (Task 3).
- Produces:
  - `class CommitteeError(RuntimeError)`
  - `class CommitteeCalculator(ase.calculators.calculator.Calculator)` with
    - `implemented_properties = ["energy", "free_energy", "forces"]`
    - `__init__(members, *, mixed_theory=False, levels=(), **kwargs)`
    - `start() -> dict[str, dict]` — member name to versions block
    - `preflight(atoms) -> dict` — one evaluation before the optimizer starts
    - `calculate(atoms=None, properties=("energy",), system_changes=all_changes)`
    - `close() -> None`, `__enter__`/`__exit__`
    - attributes `members`, `member_names`, `mixed_theory`, `levels`, `latest`, `n_evaluations`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_committee_calculator.py`:

```python
"""CommitteeCalculator: fan-out, abort behaviour, and use as an ASE calculator."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from mliprun.core.committee.calculator import CommitteeCalculator, CommitteeError
from mliprun.core.committee.remote import RemoteMember

STUB_DIR = Path(__file__).parent / "committee_stubs"


class FakeMember:
    """A member that answers from a fixed table. No subprocess."""

    def __init__(self, name, energy, forces):
        self.name = name
        self.energy = energy
        self.forces = forces
        self.versions = {"ase": "fake"}
        self.started = False
        self.closed = False
        self.n_calls = 0

    def start(self):
        self.started = True
        return self.versions

    def calculate(self, numbers, positions, cell, pbc):
        self.n_calls += 1
        return self.energy, self.forces

    def close(self, timeout=5.0):
        self.closed = True


def _rattled():
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 1)
    atoms.rattle(stdev=0.05, seed=7)
    return atoms


def _emt_committee(tmp_path, n=2, argv=None):
    members = [
        RemoteMember(f"member_{i}", sys.executable, mlip="emt",
                     log_path=tmp_path / f"member_{i}.log", timeout=60.0,
                     argv=argv)
        for i in range(n)
    ]
    return CommitteeCalculator(members)


class TestArithmeticWithFakeMembers:
    def test_results_are_the_committee_mean(self):
        f_a = np.zeros((2, 3)); f_a[0, 0] = 1.0
        f_b = np.zeros((2, 3)); f_b[0, 0] = 3.0
        committee = CommitteeCalculator(
            [FakeMember("member_a", -1.0, f_a.tolist()),
             FakeMember("member_b", -3.0, f_b.tolist())])
        committee.start()
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        atoms.calc = committee

        assert atoms.get_potential_energy() == pytest.approx(-2.0, abs=1e-12)
        assert atoms.get_forces()[0, 0] == pytest.approx(2.0, abs=1e-12)
        assert committee.latest["sigma_max"] == pytest.approx(
            2.0 / np.sqrt(2.0), abs=1e-12)
        assert committee.latest["energies"] == {"member_a": -1.0,
                                                "member_b": -3.0}
        committee.close()

    def test_one_member_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="at least two"):
            CommitteeCalculator([FakeMember("member_a", -1.0,
                                            np.zeros((2, 3)).tolist())])

    def test_a_wrong_force_shape_aborts_and_tears_down(self):
        good = FakeMember("member_a", -1.0, np.zeros((2, 3)).tolist())
        bad = FakeMember("member_b", -1.0, np.zeros((5, 3)).tolist())
        committee = CommitteeCalculator([good, bad])
        committee.start()
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        atoms.calc = committee
        with pytest.raises(CommitteeError, match="member_b"):
            atoms.get_potential_energy()
        assert good.closed and bad.closed

    def test_a_non_finite_force_aborts_and_names_the_member(self):
        forces = np.zeros((2, 3)); forces[1, 1] = np.inf
        committee = CommitteeCalculator(
            [FakeMember("member_a", -1.0, np.zeros((2, 3)).tolist()),
             FakeMember("member_b", -1.0, forces.tolist())])
        committee.start()
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        atoms.calc = committee
        with pytest.raises(CommitteeError, match="member_b"):
            atoms.get_potential_energy()

    def test_a_failing_member_aborts_the_whole_run(self):
        """Never continue with fewer members: the mean and sigma would
        silently change definition mid-trajectory."""
        class Broken(FakeMember):
            def calculate(self, *args):
                raise RuntimeError("[Errno 32] Broken pipe")

        good = FakeMember("member_a", -1.0, np.zeros((2, 3)).tolist())
        broken = Broken("member_b", -1.0, np.zeros((2, 3)).tolist())
        committee = CommitteeCalculator([good, broken])
        committee.start()
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        atoms.calc = committee
        with pytest.raises(CommitteeError, match="committee incomplete"):
            atoms.get_potential_energy()
        assert good.closed and broken.closed

    def test_every_member_is_asked_once_per_evaluation(self):
        members = [FakeMember("member_a", -1.0, np.zeros((2, 3)).tolist()),
                   FakeMember("member_b", -1.0, np.zeros((2, 3)).tolist())]
        committee = CommitteeCalculator(members)
        committee.start()
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        atoms.calc = committee
        atoms.get_potential_energy()
        atoms.get_forces()          # cached: same geometry, no new calls
        assert [m.n_calls for m in members] == [1, 1]
        assert committee.n_evaluations == 1
        committee.close()


class TestRealSubprocessRoundTrip:
    def test_identical_members_give_zero_sigma_and_the_single_model_energy(
            self, tmp_path):
        """N identical workers must reproduce the single-model result exactly
        and report no disagreement at all."""
        atoms = _rattled()
        committee = _emt_committee(tmp_path, n=3)
        committee.start()
        atoms.calc = committee
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
        finally:
            committee.close()

        reference = _rattled()
        reference.calc = EMT()
        assert energy == pytest.approx(reference.get_potential_energy(),
                                       abs=1e-10)
        assert forces == pytest.approx(reference.get_forces(), abs=1e-10)
        assert committee.latest["sigma_max"] == pytest.approx(0.0, abs=1e-12)

    def test_a_stock_optimizer_runs_through_the_committee_unchanged(
            self, tmp_path):
        """The whole architecture rests on this: downstream engines only ever
        touch atoms.calc, so nothing needs restructuring."""
        atoms = _rattled()
        committee = _emt_committee(tmp_path, n=2)
        committee.start()
        atoms.calc = committee
        try:
            BFGS(atoms, logfile=str(tmp_path / "opt.log")).run(fmax=0.05,
                                                               steps=50)
            positions = atoms.get_positions()
        finally:
            committee.close()

        reference = _rattled()
        reference.calc = EMT()
        BFGS(reference, logfile=str(tmp_path / "ref.log")).run(fmax=0.05,
                                                               steps=50)
        assert positions == pytest.approx(reference.get_positions(), abs=1e-8)

    def test_preflight_surfaces_a_bad_geometry_before_the_optimizer(
            self, tmp_path):
        """Finding 7: fairchem's UMA calculator rejects a pbc=(T,T,F) slab
        that MACE, SevenNet and CHGNet accept. The stand-in here is a worker
        that dies on its first calc."""
        committee = _emt_committee(
            tmp_path, n=2,
            argv=[sys.executable, str(STUB_DIR / "dying_worker.py")])
        committee.start()
        try:
            with pytest.raises(CommitteeError, match="committee incomplete"):
                committee.preflight(_rattled())
        finally:
            committee.close()

    def test_start_failure_tears_down_the_members_that_did_start(self,
                                                                 tmp_path):
        from mliprun.core.committee.remote import MemberError
        good = RemoteMember("member_a", sys.executable, mlip="emt",
                            log_path=tmp_path / "a.log")
        bad = RemoteMember("member_b", sys.executable,
                           mlip="not-a-real-model",
                           log_path=tmp_path / "b.log")
        committee = CommitteeCalculator([good, bad])
        with pytest.raises(MemberError):
            committee.start()
        assert not good.is_alive
        assert not bad.is_alive

    def test_context_manager_closes_every_member(self, tmp_path):
        with _emt_committee(tmp_path, n=2) as committee:
            committee.start()
            pids = [m.pid for m in committee.members]
            atoms = _rattled()
            atoms.calc = committee
            atoms.get_potential_energy()
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_calculator.py -q`
Expected: collection error — `ImportError: cannot import name 'CommitteeCalculator'`

- [ ] **Step 3: Write the implementation**

Append to `src/mliprun/core/committee/calculator.py` (keep the existing imports at the top and add the new ones there):

```python
# --- add to the imports at the top of the module ---
from concurrent.futures import ThreadPoolExecutor

from ase.calculators.calculator import Calculator, all_changes
```

```python
class CommitteeError(RuntimeError):
    """The committee could not produce a consensus for this geometry."""


class CommitteeCalculator(Calculator):
    """ASE calculator backed by N MLIPs in N separate environments.

    Everything downstream -- ``optimize`` today, MD and NEB later -- only
    ever touches ``atoms.calc``, so no engine needs restructuring to gain a
    committee. The 2026-09-04 probe confirmed a stock ASE ``BFGS`` relaxes
    through this unchanged, in a driver process that imports no torch,
    fairchem, mace, sevenn or chgnet.

    Cost is ``max(member)``, not ``sum(member)``: members are queried
    concurrently. The probe measured 211.7 ms for a four-member step against
    a slowest member of 206.7 ms -- 5.0 ms (2.4%) of inter-process overhead
    on a 992-byte payload, roughly constant in system size.

    Parameters
    ----------
    members : sequence
        Member handles. Anything with ``name``, ``start()``, ``calculate()``
        and ``close()`` works; in production these are
        :class:`~mliprun.core.committee.remote.RemoteMember`.
    mixed_theory : bool
        Whether the members span more than one level of theory (or any
        unrecognised one). Carried, not enforced: mixed levels warn, they do
        not refuse (Juan's call, 2026-09-04). A mixed committee's spread is a
        functional comparison, not an error bar -- the probe measured 0.363
        eV across RPBE and PBE members against 0.001-0.019 eV within a level.
    levels : sequence of str
        The distinct level-of-theory labels present, for the run record.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, members, *, mixed_theory: bool = False, levels=(),
                 **kwargs):
        super().__init__(**kwargs)
        members = list(members)
        if len(members) < 2:
            raise ValueError(
                f"a committee needs at least two members; got {len(members)}")
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate member names: {names}")
        self.members = members
        self.mixed_theory = bool(mixed_theory)
        self.levels = tuple(levels)
        #: Statistics from the most recent evaluation, plus a per-member
        #: ``energies`` dict. The caller reads this to build its trace.
        self.latest = None
        self.n_evaluations = 0
        self._pool = None

    @property
    def member_names(self) -> list:
        return [m.name for m in self.members]

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> dict:
        """Load every member's model. Returns ``{name: versions}``.

        Sequential, not concurrent: loads are one-time (22.8 s for UMA in the
        probe) and serialising them keeps peak GPU memory predictable when
        several members share a device. A member that fails to load tears
        down the ones that already started, so a failed startup leaves no
        orphans holding CUDA contexts.
        """
        started = []
        try:
            for member in self.members:
                member.start()
                started.append(member)
        except Exception:
            for member in started:
                try:
                    member.close()
                except Exception:  # noqa: BLE001 -- already failing
                    pass
            raise
        self._pool = ThreadPoolExecutor(max_workers=len(self.members),
                                        thread_name_prefix="committee")
        return {m.name: getattr(m, "versions", {}) for m in self.members}

    def close(self) -> None:
        """Tear every member down. Idempotent.

        The pool is shut down with ``wait=False`` first, so a thread blocked
        reading from a hung member does not hold teardown up: closing the
        member closes the pipe, which unblocks that read.
        """
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        for member in self.members:
            try:
                member.close()
            except Exception as exc:  # noqa: BLE001 -- keep closing the rest
                logger.warning("could not close committee member '%s': %s",
                               member.name, exc)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    # -- evaluation --------------------------------------------------------

    def preflight(self, atoms) -> dict:
        """Evaluate one geometry on every member before the optimizer starts.

        Members disagree about what input is valid -- fairchem's UMA
        calculator raises ``MixedPBCError`` on a ``pbc=(True, True, False)``
        slab that MACE, SevenNet and CHGNet all accept. Surfacing that in the
        first seconds beats discovering it on step 400 of an overnight run.
        """
        return self._evaluate(atoms)

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        stats = self._evaluate(self.atoms)
        self.results["energy"] = stats["energy_mean"]
        # ASE asks for free_energy on some paths; for these potentials it is
        # the same number.
        self.results["free_energy"] = stats["energy_mean"]
        self.results["forces"] = stats["forces_mean"]

    def _evaluate(self, atoms) -> dict:
        if self._pool is None:
            raise CommitteeError(
                "committee has not been started: call start() before use")

        numbers = [int(z) for z in atoms.get_atomic_numbers()]
        positions = atoms.get_positions().tolist()
        cell = np.asarray(atoms.get_cell()).tolist()
        pbc = [bool(p) for p in atoms.get_pbc()]

        futures = {m.name: self._pool.submit(m.calculate, numbers, positions,
                                             cell, pbc)
                   for m in self.members}
        energies, forces, failures = {}, {}, []
        for name, future in futures.items():
            try:
                energies[name], forces[name] = future.result()
            except Exception as exc:  # noqa: BLE001 -- collect them all
                failures.append(exc)

        if failures:
            detail = "; ".join(str(exc) for exc in failures)
            self.close()
            raise CommitteeError(f"committee incomplete: {detail}")

        try:
            stacked = self._validate(energies, forces, len(numbers))
        except CommitteeError:
            self.close()
            raise

        ordered = [energies[m.name] for m in self.members]
        stats = committee_statistics(ordered, stacked)
        stats["energies"] = dict(energies)
        self.latest = stats
        self.n_evaluations += 1
        return stats

    def _validate(self, energies: dict, forces: dict, n_atoms: int):
        """Check every member's reply before it enters the mean.

        A NaN or a wrong-shaped array must be caught and *named* here: once
        averaged, a non-finite value is anonymous and poisons every number
        downstream.
        """
        stacked = []
        for member in self.members:
            array = np.asarray(forces[member.name], dtype=float)
            if array.shape != (n_atoms, 3):
                raise CommitteeError(
                    f"member '{member.name}' returned forces of shape "
                    f"{array.shape}, expected {(n_atoms, 3)}")
            if not np.isfinite(array).all():
                raise CommitteeError(
                    f"member '{member.name}' returned a non-finite force")
            if not np.isfinite(energies[member.name]):
                raise CommitteeError(
                    f"member '{member.name}' returned a non-finite energy")
            stacked.append(array)
        return np.stack(stacked)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_calculator.py -q -rs`
Expected: 11 passed, 0 skipped.

- [ ] **Step 5: Check for orphaned workers**

Run: `pytest tests/test_committee_calculator.py -q && pgrep -f "committee" ; echo "exit=$?"`
Expected: nothing printed, `exit=1`.

- [ ] **Step 6: Commit**

```bash
git add src/mliprun/core/committee/calculator.py tests/test_committee_calculator.py
git commit -m "feat(committee): CommitteeCalculator with concurrent fan-out

An ASE calculator, so a stock BFGS drives it unchanged and no engine needs
restructuring. Members are queried concurrently, so cost is max(member) not
sum(member). Any member failure, wrong shape or non-finite value aborts the
whole run with the member named: continuing with fewer members would change
the definition of the mean and sigma mid-trajectory.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Task 6: Level-of-theory resolution

> **Review point before implementing:** the table below is a scientific
> classification, and it is deliberately incomplete. Every entry omitted
> resolves to `unknown`, which *warns* — the conservative direction. Confirm
> the entries with Juan before this task lands; adding a row is a deliberate
> act, never a guess.

**Files:**
- Create: `src/mliprun/core/committee/config.py`
- Test: `tests/test_committee_theory.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `UNKNOWN_LEVEL: str = "unknown"`
  - `resolve_level_of_theory(mlip: str, *, uma_task=None, mace_head=None, sevennet_task=None) -> str`
  - `is_mixed_theory(levels) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_committee_theory.py`. Note the neutral parametrize ids — an
id of exactly `uma`, `mace`, or `sevenn` would silently skip the case.

```python
"""Level-of-theory resolution for committee members.

CANON C3 is load-bearing here: heads and tasks are independent fine-tunes
with independent energy zeros, so a committee spanning two levels reports
the level difference, not model error. The 2026-09-04 probe measured 0.363
eV across RPBE and PBE members against 0.001-0.019 eV within a level.
"""
import pytest

from mliprun.core.committee.config import (
    UNKNOWN_LEVEL,
    is_mixed_theory,
    resolve_level_of_theory,
)


class TestKnownLevels:
    @pytest.mark.parametrize(
        "mlip,kwargs,expected",
        [
            ("uma-s-1p2", {"uma_task": "oc20"}, "RPBE/OC20"),
            ("uma-s-1p2", {"uma_task": "oc22"}, "PBE+U/OC22"),
            ("uma-s-1p2", {"uma_task": "omat"}, "PBE/OMat24"),
            ("7net-omni", {"sevennet_task": "oc20"}, "RPBE/OC20"),
            ("7net-omni", {"sevennet_task": "omat24"}, "PBE/OMat24"),
            ("7net-mf-ompa", {"sevennet_task": "omat24"}, "PBE/OMat24"),
            ("7net-omat", {}, "PBE/OMat24"),
            ("mace-mh-1", {"mace_head": "oc20_usemppbe"}, "RPBE/OC20"),
            ("mace-mh-1", {"mace_head": "omat_pbe"}, "PBE/OMat24"),
            ("mace", {}, "PBE/MPtrj"),
            ("chgnet", {}, "PBE+U/MPtrj"),
        ],
        ids=["member_a", "member_b", "member_c", "member_d", "member_e",
             "member_f", "member_g", "member_h", "member_i", "member_j",
             "member_k"],
    )
    def test_table_lookups(self, mlip, kwargs, expected):
        assert resolve_level_of_theory(mlip, **kwargs) == expected

    def test_three_packages_agree_on_the_oc20_label(self):
        """The whole point: a genuinely cross-package same-level committee."""
        levels = {
            resolve_level_of_theory("uma-s-1p2", uma_task="oc20"),
            resolve_level_of_theory("7net-omni", sevennet_task="oc20"),
            resolve_level_of_theory("mace-mh-1", mace_head="oc20_usemppbe"),
        }
        assert levels == {"RPBE/OC20"}
        assert is_mixed_theory(levels) is False


class TestUnknownLevels:
    def test_an_unregistered_tag_is_unknown(self):
        """cli/utils deliberately forwards unknown uma-*/7net-* tags to their
        packages, so this path is reachable in production."""
        assert resolve_level_of_theory("uma-x-9p9",
                                       uma_task="oc20") == "RPBE/OC20"
        assert resolve_level_of_theory("not-a-model") == UNKNOWN_LEVEL

    def test_an_unregistered_task_is_unknown(self):
        assert resolve_level_of_theory(
            "uma-s-1p2", uma_task="brand-new-head") == UNKNOWN_LEVEL

    def test_a_missing_task_is_unknown(self):
        assert resolve_level_of_theory("7net-omni") == UNKNOWN_LEVEL

    def test_the_reserved_emt_tag_is_unknown(self):
        assert resolve_level_of_theory("emt") == UNKNOWN_LEVEL

    def test_task_matching_is_case_sensitive(self):
        """7net-mf-0 names its tasks in UPPERCASE where every other model uses
        lowercase, so case-folding here would resolve a task no checkpoint
        has."""
        assert resolve_level_of_theory(
            "uma-s-1p2", uma_task="OC20") == UNKNOWN_LEVEL


class TestMixedDetection:
    def test_one_level_is_not_mixed(self):
        assert is_mixed_theory(["RPBE/OC20", "RPBE/OC20"]) is False

    def test_two_levels_are_mixed(self):
        assert is_mixed_theory(["RPBE/OC20", "PBE/MPtrj"]) is True

    def test_any_unknown_counts_as_mixed(self):
        """Unknown is *possibly* mixed: it warns rather than passing
        silently."""
        assert is_mixed_theory(["RPBE/OC20", UNKNOWN_LEVEL]) is True
        assert is_mixed_theory([UNKNOWN_LEVEL, UNKNOWN_LEVEL]) is True

    def test_an_empty_set_is_not_mixed(self):
        assert is_mixed_theory([]) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_theory.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'mliprun.core.committee.config'`

- [ ] **Step 3: Write the implementation**

Create `src/mliprun/core/committee/config.py`:

```python
"""Committee declaration: the level-of-theory table and the YAML file.

Lives in core and imports nothing from the CLI layer. The per-tag task and
head *requirements* (which tags need a task, which reject one) are not
duplicated here: they are enforced by ``build_calculator`` inside each
worker, so a missing task surfaces as a member start failure before the
optimizer runs, with the same message a single-model run would print.

Design: docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md
"""
from __future__ import annotations

#: A tag/task combination this table does not recognise. Treated as
#: *possibly* mixed: it warns rather than passing silently. Being wrong in
#: this direction costs a spurious warning; being wrong the other way would
#: label an RPBE-vs-PBE comparison as an error bar.
UNKNOWN_LEVEL = "unknown"

# ---------------------------------------------------------------------------
# Level-of-theory table
#
# DELIBERATELY INCOMPLETE. A combination is listed only where the underlying
# dataset's level of theory is unambiguous; everything else resolves to
# UNKNOWN_LEVEL and warns. Adding a row is a deliberate act that changes
# whether a committee is reported as same-level, so it needs the same care as
# a CANON entry -- do not add one to silence a warning.
#
# Labels are compared for equality, nothing more. Their exact spelling is a
# display detail; what matters is that two members at the same level produce
# the same string.
# ---------------------------------------------------------------------------

#: UMA task heads (fairchem-core).
_UMA_LEVELS = {
    "oc20": "RPBE/OC20",
    "oc22": "PBE+U/OC22",
    "omat": "PBE/OMat24",
}

#: SevenNet inference tasks ("modals").
_SEVENNET_LEVELS = {
    "oc20": "RPBE/OC20",
    "oc22": "PBE+U/OC22",
    "omat24": "PBE/OMat24",
    "matpes_pbe": "PBE/MatPES",
    "matpes_r2scan": "r2SCAN/MatPES",
}

#: Heads of the multi-head MACE foundation checkpoints (mace-mh-*).
_MACE_MH_LEVELS = {
    "oc20_usemppbe": "RPBE/OC20",
    "omat_pbe": "PBE/OMat24",
    "matpes_r2scan": "r2SCAN/MatPES",
}

#: Single-task tags whose level is fixed by the tag alone.
_FIXED_LEVELS = {
    "mace": "PBE/MPtrj",
    "chgnet": "PBE+U/MPtrj",
    "7net-omat": "PBE/OMat24",
}


def resolve_level_of_theory(mlip: str, *, uma_task=None, mace_head=None,
                            sevennet_task=None) -> str:
    """Resolve one member's ``(tag, task-or-head)`` to a level-of-theory label.

    Matching is exact and case-sensitive: ``7net-mf-0`` names its tasks in
    uppercase (``PBE``, ``R2SCAN``) where every other model uses lowercase,
    so case-folding would resolve a task no checkpoint has.

    Returns
    -------
    str
        A label, or :data:`UNKNOWN_LEVEL` for anything this table does not
        recognise -- including an unregistered tag, which is reachable in
        production because ``cli/utils`` deliberately forwards unknown
        ``uma-*`` and ``7net-*`` tags through to their packages unchanged.
    """
    if not isinstance(mlip, str):
        return UNKNOWN_LEVEL
    if mlip in _FIXED_LEVELS:
        return _FIXED_LEVELS[mlip]
    if mlip.startswith("uma-"):
        return _UMA_LEVELS.get(uma_task or "", UNKNOWN_LEVEL)
    if mlip.startswith("mace-mh-"):
        return _MACE_MH_LEVELS.get(mace_head or "", UNKNOWN_LEVEL)
    if mlip.startswith("7net"):
        return _SEVENNET_LEVELS.get(sevennet_task or "", UNKNOWN_LEVEL)
    return UNKNOWN_LEVEL


def is_mixed_theory(levels) -> bool:
    """Whether these members span more than one level of theory.

    Any :data:`UNKNOWN_LEVEL` makes the committee mixed, because an
    unrecognised member *might* sit at a different level and there is no way
    to tell. Mixed does not refuse -- it warns (Juan's call, 2026-09-04) --
    but the flag is stamped into the run record and every CSV row so
    downstream analysis can filter on it without anyone having read the
    terminal.
    """
    levels = set(levels)
    if not levels:
        return False
    return len(levels) > 1 or UNKNOWN_LEVEL in levels
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_theory.py -q -rs`
Expected: 21 passed, 0 skipped. If any case reports `skipped`, a parametrize id collided with an MLIP marker name — fix the id, do not touch conftest.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/committee/config.py tests/test_committee_theory.py
git commit -m "feat(committee): level-of-theory resolution

Maps (tag, task-or-head) to a level label so a mixed-level committee can be
flagged. The table is deliberately incomplete: anything unrecognised
resolves to 'unknown' and counts as possibly mixed, so the failure
direction is a spurious warning rather than an RPBE-vs-PBE comparison
reported as an error bar (CANON C3).

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Task 7: committee.yaml parsing and validation

**Files:**
- Modify: `src/mliprun/core/committee/config.py` (append)
- Modify: `pyproject.toml` (add `pyyaml` to base dependencies)
- Test: `tests/test_committee_config.py`

**Interfaces:**
- Consumes: `resolve_level_of_theory`, `is_mixed_theory`, `UNKNOWN_LEVEL` (Task 6).
- Produces:
  - `class CommitteeConfigError(ValueError)`
  - `MemberSpec` dataclass (frozen): `name: str`, `env: str`, `mlip: str`, `uma_task: str | None`, `mace_head: str | None`, `sevennet_task: str | None`, `device: str`, `gpu: int | None`, `level_of_theory: str`, `python_exe: str`
  - `CommitteeConfig` dataclass (frozen): `members: tuple[MemberSpec, ...]`, `levels: tuple[str, ...]`, `mixed_theory: bool`, `sha256: str`, `source_path: str`
  - `CommitteeConfig.as_provenance() -> dict`
  - `load_committee(path) -> CommitteeConfig`
  - `python_for_env(env) -> Path`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_committee_config.py`:

```python
"""Parsing and validating committee.yaml."""
import hashlib
import sys
import textwrap
from pathlib import Path

import pytest

from mliprun.core.committee.config import (
    CommitteeConfigError,
    load_committee,
    python_for_env,
)


@pytest.fixture
def fake_env(tmp_path):
    """A directory that looks like a venv: <env>/bin/python exists."""
    def _make(name):
        env = tmp_path / name
        (env / "bin").mkdir(parents=True)
        (env / "bin" / "python").write_text("#!/bin/sh\n")
        return env
    return _make


def _write(tmp_path, text):
    path = tmp_path / "committee.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


class TestValidFile:
    def test_a_three_member_committee_parses(self, tmp_path, fake_env):
        a, b, c = fake_env("uma"), fake_env("sevenn"), fake_env("mace")
        path = _write(tmp_path, f"""
            members:
              - env:  {a}
                mlip: uma-s-1p2
                uma_task: oc20
                gpu: 0
              - env:  {b}
                mlip: 7net-omni
                sevennet_task: oc20
                gpu: 1
              - env:  {c}
                mlip: mace-mh-1
                mace_head: oc20_usemppbe
                gpu: 2
        """)
        config = load_committee(path)

        assert len(config.members) == 3
        assert [m.mlip for m in config.members] == ["uma-s-1p2", "7net-omni",
                                                    "mace-mh-1"]
        assert [m.gpu for m in config.members] == [0, 1, 2]
        assert config.levels == ("RPBE/OC20",)
        assert config.mixed_theory is False
        assert config.members[0].python_exe == str(a / "bin" / "python")

    def test_default_names_pair_the_tag_with_its_task(self, tmp_path,
                                                      fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: uma-s-1p2, uma_task: oc20}}
              - {{env: {b}, mlip: chgnet}}
        """)
        config = load_committee(path)
        assert [m.name for m in config.members] == ["uma-s-1p2@oc20", "chgnet"]

    def test_an_explicit_name_wins(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: uma-s-1p2, uma_task: oc20, name: reference}}
              - {{env: {b}, mlip: chgnet}}
        """)
        assert load_committee(path).members[0].name == "reference"

    def test_repeated_tag_and_task_get_distinct_names(self, tmp_path,
                                                      fake_env):
        """Two members can legitimately differ only by env (two builds of the
        same model), and the CSV needs one column per member."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet}}
              - {{env: {b}, mlip: chgnet}}
        """)
        names = [m.name for m in load_committee(path).members]
        assert len(set(names)) == 2
        assert names[0] == "chgnet"

    def test_sha256_matches_the_file_bytes(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
        """)
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert load_committee(path).sha256 == expected

    def test_env_paths_expand_user_and_variables(self, tmp_path, fake_env,
                                                 monkeypatch):
        a, b = fake_env("a"), fake_env("b")
        monkeypatch.setenv("MLIPRUN_TEST_ROOT", str(a))
        path = _write(tmp_path, f"""
            members:
              - {{env: "$MLIPRUN_TEST_ROOT", mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
        """)
        assert load_committee(path).members[0].env == str(a)

    def test_provenance_block_carries_every_member(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
        """)
        provenance = load_committee(path).as_provenance()
        assert len(provenance["members"]) == 2
        assert provenance["members"][0]["mlip"] == "chgnet"
        assert provenance["members"][0]["level_of_theory"] == "PBE+U/MPtrj"
        assert provenance["mixed_theory"] is True   # PBE+U/MPtrj vs PBE/MPtrj
        assert sorted(provenance["levels"]) == ["PBE+U/MPtrj", "PBE/MPtrj"]
        assert provenance["config_sha256"] == load_committee(path).sha256


class TestMixedTheoryFlag:
    def test_two_levels_set_the_flag(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: uma-s-1p2, uma_task: oc20}}
              - {{env: {b}, mlip: mace}}
        """)
        config = load_committee(path)
        assert config.mixed_theory is True
        assert sorted(config.levels) == ["PBE/MPtrj", "RPBE/OC20"]

    def test_an_unknown_task_sets_the_flag(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: uma-s-1p2, uma_task: oc20}}
              - {{env: {b}, mlip: 7net-omni, sevennet_task: pet_mad}}
        """)
        assert load_committee(path).mixed_theory is True


class TestRejections:
    def test_a_missing_file(self, tmp_path):
        with pytest.raises(CommitteeConfigError, match="not found"):
            load_committee(tmp_path / "absent.yaml")

    def test_malformed_yaml(self, tmp_path):
        path = _write(tmp_path, "members: [unclosed\n")
        with pytest.raises(CommitteeConfigError, match="could not parse"):
            load_committee(path)

    def test_a_top_level_list(self, tmp_path):
        path = _write(tmp_path, "- one\n- two\n")
        with pytest.raises(CommitteeConfigError, match="mapping"):
            load_committee(path)

    def test_a_missing_members_key(self, tmp_path):
        path = _write(tmp_path, "committee: []\n")
        with pytest.raises(CommitteeConfigError, match="members"):
            load_committee(path)

    def test_a_single_member(self, tmp_path, fake_env):
        a = fake_env("a")
        path = _write(tmp_path, f"members:\n  - {{env: {a}, mlip: chgnet}}\n")
        with pytest.raises(CommitteeConfigError, match="at least two"):
            load_committee(path)

    def test_an_unknown_member_key(self, tmp_path, fake_env):
        """Typo protection: a silently ignored 'gpus:' would send every
        member to the same device."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet, gpus: 0}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match="gpus"):
            load_committee(path)

    def test_a_missing_env(self, tmp_path, fake_env):
        b = fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match="env"):
            load_committee(path)

    def test_a_missing_mlip(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match="mlip"):
            load_committee(path)

    def test_an_env_without_an_interpreter(self, tmp_path, fake_env):
        b = fake_env("b")
        empty = tmp_path / "empty"
        empty.mkdir()
        path = _write(tmp_path, f"""
            members:
              - {{env: {empty}, mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match="interpreter"):
            load_committee(path)

    def test_duplicate_explicit_names(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet, name: same}}
              - {{env: {b}, mlip: mace, name: same}}
        """)
        with pytest.raises(CommitteeConfigError, match="duplicate"):
            load_committee(path)

    def test_a_name_with_unsafe_characters(self, tmp_path, fake_env):
        """Names become CSV headers and log filenames."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet, name: "../escape"}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match="name"):
            load_committee(path)

    def test_a_non_integer_gpu(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet, gpu: "first"}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match="gpu"):
            load_committee(path)


class TestPythonForEnv:
    def test_the_current_environment_resolves(self):
        env = Path(sys.executable).parents[1]
        assert python_for_env(env).exists()

    def test_python3_is_accepted_when_python_is_absent(self, tmp_path):
        env = tmp_path / "env"
        (env / "bin").mkdir(parents=True)
        (env / "bin" / "python3").write_text("#!/bin/sh\n")
        assert python_for_env(env).name == "python3"

    def test_a_missing_env_is_rejected(self, tmp_path):
        with pytest.raises(CommitteeConfigError, match="interpreter"):
            python_for_env(tmp_path / "nope")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_config.py -q`
Expected: collection error — `ImportError: cannot import name 'load_committee'`

- [ ] **Step 3: Add the YAML dependency**

In `pyproject.toml`, add `pyyaml` to `[project] dependencies`, after `"scipy",`:

```toml
    "scipy",
    # Committee declarations (committee.yaml). Pure-Python fallback exists,
    # so this adds no build burden on a login node.
    "pyyaml",
]
```

Then reinstall so the metadata updates: `pip install -e ".[dev,neb]"`

- [ ] **Step 4: Write the implementation**

Append to `src/mliprun/core/committee/config.py`:

```python
# --- add to the imports at the top of the module ---
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
```

```python
class CommitteeConfigError(ValueError):
    """The committee file is missing, malformed, or internally inconsistent."""


#: Keys a member may carry. Anything else is a typo and is rejected: a
#: silently ignored ``gpus:`` would send every member to the same device.
_MEMBER_KEYS = frozenset({
    "name", "env", "mlip", "uma_task", "mace_head", "sevennet_task",
    "device", "gpu",
})

#: Member names become CSV column headers and worker-log filenames.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+-]*$")


@dataclass(frozen=True)
class MemberSpec:
    """One declared committee member."""

    name: str
    env: str
    python_exe: str
    mlip: str
    uma_task: str | None
    mace_head: str | None
    sevennet_task: str | None
    device: str
    gpu: int | None
    level_of_theory: str

    def as_provenance(self) -> dict:
        return {
            "name": self.name,
            "mlip": self.mlip,
            "uma_task": self.uma_task,
            "mace_head": self.mace_head,
            "sevennet_task": self.sevennet_task,
            "env": self.env,
            "python": self.python_exe,
            "device": self.device,
            "gpu": self.gpu,
            "level_of_theory": self.level_of_theory,
        }


@dataclass(frozen=True)
class CommitteeConfig:
    """A parsed, validated committee declaration."""

    members: tuple
    levels: tuple
    mixed_theory: bool
    sha256: str
    source_path: str

    def as_provenance(self) -> dict:
        """The block the run record stores under ``provenance.committee``."""
        return {
            "members": [m.as_provenance() for m in self.members],
            "levels": list(self.levels),
            "mixed_theory": self.mixed_theory,
            "config_sha256": self.sha256,
            "config_path": self.source_path,
        }

    def mixed_theory_warning(self) -> str:
        """The warn-don't-refuse message. Empty when the levels agree."""
        if not self.mixed_theory:
            return ""
        rows = "\n".join(f"    {m.name}: {m.level_of_theory}"
                         for m in self.members)
        return (
            "This committee spans more than one level of theory:\n"
            f"{rows}\n"
            "  The spread it reports is a comparison BETWEEN levels of "
            "theory, not an error bar. In the 2026-09-04 probe the "
            "across-level difference was 0.363 eV against 0.001-0.019 eV "
            "within a level. The 'unknown' label means this tag/task is not "
            "in mliprun's level-of-theory table, not that the members "
            "disagree. Every CSV row carries mixed_theory=True so downstream "
            "analysis can filter on it."
        )


def python_for_env(env) -> Path:
    """Return the interpreter inside a member env.

    Existence only -- not executability. A path that exists but cannot be
    executed fails at member start with the exec error attached, which is a
    better message than anything guessed here.
    """
    root = Path(os.path.expandvars(str(env))).expanduser()
    for candidate in (root / "bin" / "python", root / "bin" / "python3"):
        if candidate.exists():
            return candidate
    raise CommitteeConfigError(
        f"no interpreter found in env '{env}': expected "
        f"{root / 'bin' / 'python'} or {root / 'bin' / 'python3'}. "
        f"Each committee member needs its own MLIP env with mliprun "
        f"installed in it -- see docs/install/README.md.")


def _member_name(entry: dict, mlip: str) -> str:
    task = (entry.get("uma_task") or entry.get("mace_head")
            or entry.get("sevennet_task"))
    return f"{mlip}@{task}" if task else mlip


def _parse_member(entry, index: int) -> dict:
    if not isinstance(entry, dict):
        raise CommitteeConfigError(
            f"member {index} is not a mapping (got {type(entry).__name__})")
    unknown = sorted(set(entry) - _MEMBER_KEYS)
    if unknown:
        raise CommitteeConfigError(
            f"unknown key(s) {unknown} in member {index}; valid keys are "
            f"{sorted(_MEMBER_KEYS)}")
    for required in ("env", "mlip"):
        if not entry.get(required):
            raise CommitteeConfigError(
                f"member {index} is missing required key '{required}'")
    gpu = entry.get("gpu")
    if gpu is not None and not isinstance(gpu, int):
        raise CommitteeConfigError(
            f"member {index}: gpu must be an integer device index, got "
            f"{gpu!r}")
    return entry


def load_committee(path) -> CommitteeConfig:
    """Parse and validate a ``committee.yaml``.

    Structural validation only. Whether a tag *requires* a task or head is
    not re-checked here: ``build_calculator`` enforces that inside the
    member's own env, so a missing ``--uma-task`` equivalent fails at member
    start with exactly the message a single-model run would print (CANON C1).

    Raises
    ------
    CommitteeConfigError
        With a message naming the offending member by index.
    """
    import yaml

    path = Path(path)
    if not path.is_file():
        raise CommitteeConfigError(f"committee file not found: {path}")
    raw = path.read_bytes()
    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise CommitteeConfigError(
            f"could not parse {path}: {exc}") from exc

    if not isinstance(document, dict):
        raise CommitteeConfigError(
            f"{path}: top level must be a mapping with a 'members' key")
    entries = document.get("members")
    if not isinstance(entries, list):
        raise CommitteeConfigError(
            f"{path}: 'members' must be a list, got "
            f"{type(entries).__name__}")
    if len(entries) < 2:
        raise CommitteeConfigError(
            f"{path}: a committee needs at least two members; got "
            f"{len(entries)}. A committee of one has no disagreement to "
            f"report.")

    members = []
    used_names = set()
    for index, entry in enumerate(entries):
        entry = _parse_member(entry, index)
        mlip = entry["mlip"]
        explicit = entry.get("name")
        name = explicit or _member_name(entry, mlip)
        if not _NAME_RE.match(str(name)):
            raise CommitteeConfigError(
                f"member {index}: name {name!r} must match "
                f"{_NAME_RE.pattern} -- it becomes a CSV column header and a "
                f"log filename")
        if name in used_names:
            if explicit:
                raise CommitteeConfigError(
                    f"member {index}: duplicate member name {name!r}")
            suffix = 2
            while f"{name}-{suffix}" in used_names:
                suffix += 1
            name = f"{name}-{suffix}"
        used_names.add(name)

        env = str(Path(os.path.expandvars(str(entry["env"]))).expanduser())
        members.append(MemberSpec(
            name=name,
            env=env,
            python_exe=str(python_for_env(env)),
            mlip=mlip,
            uma_task=entry.get("uma_task"),
            mace_head=entry.get("mace_head"),
            sevennet_task=entry.get("sevennet_task"),
            device=entry.get("device", "auto"),
            gpu=entry.get("gpu"),
            level_of_theory=resolve_level_of_theory(
                mlip,
                uma_task=entry.get("uma_task"),
                mace_head=entry.get("mace_head"),
                sevennet_task=entry.get("sevennet_task"),
            ),
        ))

    levels = tuple(sorted({m.level_of_theory for m in members}))
    return CommitteeConfig(
        members=tuple(members),
        levels=levels,
        mixed_theory=is_mixed_theory(levels),
        sha256=hashlib.sha256(raw).hexdigest(),
        source_path=str(path.resolve()),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_committee_config.py -q -rs`
Expected: 24 passed, 0 skipped.

- [ ] **Step 6: Commit**

```bash
git add src/mliprun/core/committee/config.py tests/test_committee_config.py pyproject.toml
git commit -m "feat(committee): committee.yaml parsing and validation

The committee is a version-controllable artifact that hashes into the run
record. Structural validation only -- whether a tag requires a task or head
stays enforced by build_calculator inside the member's own env, so the
message is the same one a single-model run prints. Unknown keys are
rejected: a silently ignored 'gpus:' would send every member to one device.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Task 8: Trace CSV, per-atom CSV, and the flagging rule

**Files:**
- Modify: `src/mliprun/core/committee/calculator.py` (append)
- Test: `tests/test_committee_outputs.py`

**Interfaces:**
- Consumes: `aligned_energy_spread` (Task 4); the `latest` dict produced by `CommitteeCalculator._evaluate` (Task 5).
- Produces:
  - `class CommitteeTraceWriter` with `__init__(path, member_names, mixed_theory)`, `write_step(step: int, latest: dict, fmax_value: float) -> dict`, `close() -> None`, attribute `rows: list[dict]`
  - `write_peratom_sigma(path, symbols, sigma_per_atom) -> None`
  - `uncertainty_summary(rows, latest, *, threshold, threshold_source, symbols=None) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_committee_outputs.py`:

```python
"""Committee outputs: the step trace, the per-atom file, and the flag."""
import csv

import numpy as np
import pytest

from mliprun.core.committee.calculator import (
    CommitteeTraceWriter,
    uncertainty_summary,
    write_peratom_sigma,
)


def _latest(energies, sigma_per_atom, energy_mean=None):
    sigma = np.asarray(sigma_per_atom, dtype=float)
    worst = int(np.argmax(sigma))
    values = list(energies.values())
    return {
        "energies": dict(energies),
        "energy_mean": (energy_mean if energy_mean is not None
                        else float(np.mean(values))),
        "sigma_per_atom": sigma,
        "sigma_max": float(sigma[worst]),
        "sigma_mean": float(sigma.mean()),
        "worst_atom": worst,
    }


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestTraceWriter:
    def test_header_carries_one_column_per_member(self, tmp_path):
        path = tmp_path / "opt_committee.csv"
        writer = CommitteeTraceWriter(path, ["member_a", "member_b"], False)
        writer.write_step(0, _latest({"member_a": -1.0, "member_b": -3.0},
                                     [0.1, 0.2]), 0.5)
        writer.close()

        with open(path, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert header == [
            "step", "energy_mean_eV", "energy_spread_aligned_eV",
            "E_member_a_eV", "E_member_b_eV", "fmax_eV_per_A",
            "sigma_max_eV_per_A", "sigma_mean_eV_per_A", "worst_atom",
            "mixed_theory",
        ]

    def test_values_land_in_the_right_columns(self, tmp_path):
        path = tmp_path / "opt_committee.csv"
        writer = CommitteeTraceWriter(path, ["member_a", "member_b"], True)
        writer.write_step(3, _latest({"member_a": -1.0, "member_b": -3.0},
                                     [0.1, 0.4]), 0.25)
        writer.close()

        row = _read(path)[0]
        assert int(row["step"]) == 3
        assert float(row["energy_mean_eV"]) == pytest.approx(-2.0, abs=1e-12)
        assert float(row["E_member_a_eV"]) == pytest.approx(-1.0, abs=1e-12)
        assert float(row["E_member_b_eV"]) == pytest.approx(-3.0, abs=1e-12)
        assert float(row["fmax_eV_per_A"]) == pytest.approx(0.25, abs=1e-12)
        assert float(row["sigma_max_eV_per_A"]) == pytest.approx(0.4, abs=1e-12)
        assert float(row["sigma_mean_eV_per_A"]) == pytest.approx(0.25,
                                                                  abs=1e-12)
        assert int(row["worst_atom"]) == 1
        assert row["mixed_theory"] == "True"

    def test_first_step_has_zero_aligned_spread_by_construction(self, tmp_path):
        """Step 0 defines each member's baseline, so its aligned spread is
        exactly zero -- not a measurement."""
        writer = CommitteeTraceWriter(tmp_path / "c.csv",
                                      ["member_a", "member_b"], False)
        row = writer.write_step(0, _latest({"member_a": -73.68,
                                            "member_b": -82.86}, [0.1]), 1.0)
        writer.close()
        assert row["energy_spread_aligned_eV"] == pytest.approx(0.0, abs=1e-12)

    def test_constant_offsets_cancel_on_later_steps(self, tmp_path):
        """A 9.18 eV raw spread that is pure per-model offset must report as
        zero uncertainty once both members move by the same amount."""
        writer = CommitteeTraceWriter(tmp_path / "c.csv",
                                      ["member_a", "member_b"], False)
        writer.write_step(0, _latest({"member_a": -73.68,
                                      "member_b": -82.86}, [0.1]), 1.0)
        row = writer.write_step(1, _latest({"member_a": -74.68,
                                            "member_b": -83.86}, [0.1]), 0.9)
        writer.close()
        assert row["energy_spread_aligned_eV"] == pytest.approx(0.0, abs=1e-12)

    def test_real_disagreement_survives_alignment(self, tmp_path):
        writer = CommitteeTraceWriter(tmp_path / "c.csv",
                                      ["member_a", "member_b"], False)
        writer.write_step(0, _latest({"member_a": -10.0, "member_b": -20.0},
                                     [0.1]), 1.0)
        row = writer.write_step(1, _latest({"member_a": -11.0,
                                            "member_b": -22.0}, [0.1]), 0.9)
        writer.close()
        assert row["energy_spread_aligned_eV"] == pytest.approx(
            1.0 / np.sqrt(2.0), abs=1e-12)

    def test_rows_are_flushed_as_they_are_written(self, tmp_path):
        """A run that dies at step 300 keeps its first 300 steps."""
        path = tmp_path / "opt_committee.csv"
        writer = CommitteeTraceWriter(path, ["member_a", "member_b"], False)
        for step in range(3):
            writer.write_step(step, _latest({"member_a": -1.0,
                                             "member_b": -3.0}, [0.1]), 0.5)
            assert len(_read(path)) == step + 1     # readable before close
        writer.close()
        assert len(writer.rows) == 3

    def test_a_member_name_with_a_comma_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            CommitteeTraceWriter(tmp_path / "c.csv", ["a,b", "c"], False)


class TestPerAtomFile:
    def test_one_row_per_atom_with_symbol_and_sigma(self, tmp_path):
        path = tmp_path / "opt_committee_peratom.csv"
        write_peratom_sigma(path, ["Cu", "Cu", "C", "O"],
                            [0.01, 0.02, 0.9, 0.7])
        rows = _read(path)
        assert [r["symbol"] for r in rows] == ["Cu", "Cu", "C", "O"]
        assert [int(r["atom_index"]) for r in rows] == [0, 1, 2, 3]
        assert float(rows[2]["sigma_eV_per_A"]) == pytest.approx(0.9,
                                                                 abs=1e-12)

    def test_length_mismatch_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            write_peratom_sigma(tmp_path / "p.csv", ["Cu"], [0.1, 0.2])


class TestFlaggingRule:
    def _rows(self, sigmas):
        return [{"step": i, "sigma_max_eV_per_A": s,
                 "energy_spread_aligned_eV": 0.0} for i, s in enumerate(sigmas)]

    def test_sigma_above_the_threshold_flags_the_configuration(self):
        summary = uncertainty_summary(
            self._rows([0.30, 0.12, 0.08]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.02, 0.08]),
            threshold=0.05, threshold_source="fmax", symbols=["Cu", "C"])
        assert summary["flagged"] is True
        assert summary["sigma_max_final_eV_per_A"] == pytest.approx(0.08,
                                                                    abs=1e-12)
        assert summary["threshold_eV_per_A"] == pytest.approx(0.05, abs=1e-12)
        assert summary["threshold_source"] == "fmax"

    def test_sigma_below_the_threshold_does_not_flag(self):
        summary = uncertainty_summary(
            self._rows([0.30, 0.02]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.01, 0.02]),
            threshold=0.05, threshold_source="fmax")
        assert summary["flagged"] is False

    def test_the_flag_uses_the_final_geometry_not_the_peak(self):
        """A path that passed through a strained geometry but converged to a
        well-constrained minimum is not flagged."""
        summary = uncertainty_summary(
            self._rows([3.476, 0.01]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.005, 0.01]),
            threshold=0.05, threshold_source="fmax")
        assert summary["flagged"] is False
        assert summary["sigma_max_peak_eV_per_A"] == pytest.approx(3.476,
                                                                   abs=1e-12)
        assert summary["peak_step"] == 0

    def test_exactly_at_the_threshold_does_not_flag(self):
        summary = uncertainty_summary(
            self._rows([0.05]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.05]),
            threshold=0.05, threshold_source="explicit")
        assert summary["flagged"] is False

    def test_the_worst_atom_is_reported_with_its_symbol(self):
        summary = uncertainty_summary(
            self._rows([0.5]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.01, 0.5]),
            threshold=0.05, threshold_source="fmax", symbols=["Cu", "C"])
        assert summary["worst_atom"] == 1
        assert summary["worst_atom_symbol"] == "C"

    def test_an_empty_trace_reports_no_flag_rather_than_crashing(self):
        """A run that died before its first optimizer step still has to
        finish its record."""
        summary = uncertainty_summary([], None, threshold=0.05,
                                      threshold_source="fmax")
        assert summary["flagged"] is False
        assert summary["sigma_max_final_eV_per_A"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_outputs.py -q`
Expected: collection error — `ImportError: cannot import name 'CommitteeTraceWriter'`

- [ ] **Step 3: Write the implementation**

Append to `src/mliprun/core/committee/calculator.py`:

```python
# --- add to the imports at the top of the module ---
import csv
```

```python
class CommitteeTraceWriter:
    """Append-as-you-go writer for ``<stem>_committee.csv``.

    One row per optimizer step, flushed as it is written: a committee run
    that dies at step 300 keeps its first 300 steps of disagreement data.

    Column meanings
    ---------------
    energy_mean_eV
        Mean of the members' raw energies -- the optimizer's objective. Its
        absolute value is meaningless across packages (the four-member probe
        saw a 4.81 eV spread at one fixed geometry, almost all of it
        per-model offset); differences along a trajectory are not.
    energy_spread_aligned_eV
        Standard deviation (``ddof=1``, not a range) of the members' energies
        after each member's own step-0 energy is removed. Composition is
        fixed during a relaxation, so the offset cancels exactly and this is
        a real energy uncertainty. Zero by construction on the first row.
    E_<member>_eV
        Each member's raw energy.
    sigma_max_eV_per_A, sigma_mean_eV_per_A, worst_atom
        Per-atom force disagreement, reduced.
    mixed_theory
        The warn-don't-refuse flag, repeated on every row so downstream
        analysis can filter on it without having read the terminal.
    """

    def __init__(self, path, member_names, mixed_theory: bool):
        member_names = list(member_names)
        for name in member_names:
            if "," in name or any(c.isspace() for c in name):
                raise ValueError(
                    f"member name {name!r} is not usable as a CSV column "
                    f"header")
        self.path = str(path)
        self.member_names = member_names
        self.mixed_theory = bool(mixed_theory)
        self.rows: list = []
        self._baseline: dict = {}
        self._fieldnames = (
            ["step", "energy_mean_eV", "energy_spread_aligned_eV"]
            + [f"E_{name}_eV" for name in member_names]
            + ["fmax_eV_per_A", "sigma_max_eV_per_A", "sigma_mean_eV_per_A",
               "worst_atom", "mixed_theory"]
        )
        self._handle = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle,
                                      fieldnames=self._fieldnames)
        self._writer.writeheader()
        self._handle.flush()

    def write_step(self, step: int, latest: dict, fmax_value: float) -> dict:
        """Append one optimizer step. Returns the row it wrote."""
        energies = latest["energies"]
        if not self._baseline:
            self._baseline = dict(energies)
        row = {
            "step": int(step),
            "energy_mean_eV": float(latest["energy_mean"]),
            "energy_spread_aligned_eV": aligned_energy_spread(
                energies, self._baseline),
            "fmax_eV_per_A": float(fmax_value),
            "sigma_max_eV_per_A": float(latest["sigma_max"]),
            "sigma_mean_eV_per_A": float(latest["sigma_mean"]),
            "worst_atom": int(latest["worst_atom"]),
            "mixed_theory": self.mixed_theory,
        }
        for name in self.member_names:
            row[f"E_{name}_eV"] = float(energies[name])
        self._writer.writerow(row)
        self._handle.flush()
        self.rows.append(row)
        return row

    def close(self) -> None:
        """Close the file. Idempotent."""
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None


def write_peratom_sigma(path, symbols, sigma_per_atom) -> None:
    """Write the final geometry's per-atom force disagreement.

    Final geometry only: a per-atom field at every step would be a large file
    for little gain, and ``worst_atom`` already traces where the disagreement
    lives during the run. This file is the most diagnostically useful output
    -- it says *which* atoms the models disagree about, which is usually the
    adsorbate or the reacting bond.
    """
    symbols = list(symbols)
    sigma_per_atom = np.asarray(sigma_per_atom, dtype=float).reshape(-1)
    if len(symbols) != sigma_per_atom.size:
        raise ValueError(
            f"{len(symbols)} symbols but {sigma_per_atom.size} sigma values")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["atom_index", "symbol", "sigma_eV_per_A"])
        for index, (symbol, sigma) in enumerate(zip(symbols, sigma_per_atom)):
            writer.writerow([index, symbol, float(sigma)])


def uncertainty_summary(rows, latest, *, threshold: float,
                        threshold_source: str, symbols=None) -> dict:
    """Reduce a committee run to the block the run record stores.

    The flagging rule: a configuration is flagged when ``sigma_max`` at the
    **final** geometry exceeds ``threshold``. Self-scaling and physically
    motivated -- if the models disagree about the forces by more than the
    convergence tolerance, the located minimum sits inside the committee's
    own noise and the geometry is not resolved. A path that passed through a
    strained geometry but converged to a well-constrained minimum is not
    flagged, which is why the peak is reported separately.

    **The default threshold is uncalibrated.** The 2026-09-04 probe measured
    sigma_F only across four mixed-level members, so there is no same-level
    number yet. Calibrating it is a natural first use of the feature; until
    then the threshold and its source travel with the flag so a later reader
    knows what was applied.

    Parameters
    ----------
    rows : list of dict
        The trace rows, as written by :class:`CommitteeTraceWriter`.
    latest : dict or None
        The final evaluation's statistics. ``None`` when the run died before
        evaluating anything.
    threshold : float
        The sigma_max above which the configuration is flagged.
    threshold_source : {"fmax", "explicit"}
        Where the threshold came from.
    symbols : sequence of str, optional
        Chemical symbols, used to name the worst atom.
    """
    summary = {
        "n_steps": len(rows),
        "threshold_eV_per_A": float(threshold),
        "threshold_source": threshold_source,
        "sigma_max_final_eV_per_A": None,
        "sigma_mean_final_eV_per_A": None,
        "sigma_max_peak_eV_per_A": None,
        "peak_step": None,
        "worst_atom": None,
        "worst_atom_symbol": None,
        "energy_spread_aligned_final_eV": None,
        "flagged": False,
    }
    if rows:
        peak = max(rows, key=lambda r: r["sigma_max_eV_per_A"])
        summary["sigma_max_peak_eV_per_A"] = float(peak["sigma_max_eV_per_A"])
        summary["peak_step"] = int(peak["step"])
        summary["energy_spread_aligned_final_eV"] = float(
            rows[-1]["energy_spread_aligned_eV"])
    if latest is not None:
        sigma_max = float(latest["sigma_max"])
        worst = int(latest["worst_atom"])
        summary["sigma_max_final_eV_per_A"] = sigma_max
        summary["sigma_mean_final_eV_per_A"] = float(latest["sigma_mean"])
        summary["worst_atom"] = worst
        if symbols is not None and worst < len(symbols):
            summary["worst_atom_symbol"] = symbols[worst]
        summary["flagged"] = bool(sigma_max > threshold)
    return summary
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_outputs.py -q -rs`
Expected: 15 passed, 0 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/committee/calculator.py tests/test_committee_outputs.py
git commit -m "feat(committee): step trace, per-atom sigma, and the flagging rule

Rows are flushed as they are written, so a run that dies at step 300 keeps
its first 300 steps. The energy spread is aligned against each member's own
step-0 energy, which cancels the per-model offset exactly. A configuration
is flagged when final sigma_max exceeds the threshold -- default fmax, and
uncalibrated: the threshold and its source travel with the flag.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Task 9: Run record, schema 3 → 4

**Files:**
- Modify: `src/mliprun/core/run_record.py:33` (`SCHEMA_VERSION`), `:53-55` (`_PROVENANCE_DIFF_FIELDS`), `:166-237` (`collect_provenance`)
- Modify: `tests/test_run_record.py:390-399`, `tests/test_core_md.py:151`
- Test: `tests/test_run_record.py` (append a `TestCommitteeProvenance` class)

**Interfaces:**
- Consumes: the provenance block from `CommitteeConfig.as_provenance()` (Task 7).
- Produces:
  - `SCHEMA_VERSION = 4`
  - `collect_provenance(*, mlip_model, device_requested, device_resolved, uma_task=None, mace_head=None, sevennet_task=None, committee=None) -> dict` — adds `provenance["committee"]` and `provenance["committee_config_sha256"]` **only when `committee` is not None**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run_record.py`:

```python
class TestCommitteeProvenance:
    """Schema 4: a committee run records a list of members, not one model."""

    def _committee(self):
        return {
            "members": [
                {"name": "uma-s-1p2@oc20", "mlip": "uma-s-1p2",
                 "uma_task": "oc20", "mace_head": None, "sevennet_task": None,
                 "env": "/scratchb/juar/.venv/uma",
                 "python": "/scratchb/juar/.venv/uma/bin/python",
                 "device": "auto", "gpu": 0,
                 "level_of_theory": "RPBE/OC20"},
                {"name": "7net-omni@oc20", "mlip": "7net-omni",
                 "uma_task": None, "mace_head": None,
                 "sevennet_task": "oc20",
                 "env": "/scratchb/juar/.venv/sevenn",
                 "python": "/scratchb/juar/.venv/sevenn/bin/python",
                 "device": "auto", "gpu": 1,
                 "level_of_theory": "RPBE/OC20"},
            ],
            "levels": ["RPBE/OC20"],
            "mixed_theory": False,
            "config_sha256": "a" * 64,
            "config_path": "/work/committee.yaml",
        }

    def test_the_committee_block_is_recorded_verbatim(self):
        prov = collect_provenance(mlip_model="committee",
                                  device_requested="auto",
                                  device_resolved="cuda",
                                  committee=self._committee())
        assert len(prov["committee"]["members"]) == 2
        assert prov["committee"]["members"][0]["level_of_theory"] == "RPBE/OC20"
        assert prov["committee"]["mixed_theory"] is False
        assert prov["committee_config_sha256"] == "a" * 64

    def test_the_config_hash_is_promoted_to_a_flat_field(self):
        """Flat, so an appended stage can report 'the committee changed' as
        one string rather than diffing a nested member list."""
        prov = collect_provenance(mlip_model="committee",
                                  device_requested="auto",
                                  device_resolved="cuda",
                                  committee=self._committee())
        assert prov["committee_config_sha256"] == prov["committee"][
            "config_sha256"]

    def test_a_single_model_run_gains_no_committee_keys(self):
        """Single-model runs keep writing exactly what they write today."""
        prov = collect_provenance(mlip_model="uma-s-1p2",
                                  device_requested="auto",
                                  device_resolved="cuda", uma_task="oc20")
        assert "committee" not in prov
        assert "committee_config_sha256" not in prov

    def test_head_gating_still_applies_to_a_committee_run(self):
        """model tag 'committee' is not a UMA/MACE/SevenNet tag, so the
        top-level head fields stay null -- the heads live per member."""
        prov = collect_provenance(mlip_model="committee",
                                  device_requested="auto",
                                  device_resolved="cpu", uma_task="oc20",
                                  committee=self._committee())
        assert prov["uma_task"] is None
        assert prov["mace_head"] is None
        assert prov["sevennet_task"] is None

    def test_a_malformed_committee_block_does_not_raise(self):
        """collect_provenance must be total: it runs outside RunRecord.begin's
        try, so anything it raises kills the run."""
        prov = collect_provenance(mlip_model="committee",
                                  device_requested="auto",
                                  device_resolved="cpu",
                                  committee={"members": {1, 2, 3}})
        assert "committee" in prov

    def test_a_changed_committee_shows_up_in_an_appended_stage(self, tmp_path):
        RunRecord.begin(tmp_path, command="optimize", stage_kind="optimize",
                        parameters={}, inputs={},
                        provenance=collect_provenance(
                            mlip_model="committee", device_requested="auto",
                            device_resolved="cpu",
                            committee=self._committee())).complete(
                                status="converged")
        changed = self._committee()
        changed["config_sha256"] = "b" * 64
        RunRecord.begin(tmp_path, command="optimize", stage_kind="optimize",
                        parameters={}, inputs={},
                        provenance=collect_provenance(
                            mlip_model="committee", device_requested="auto",
                            device_resolved="cpu", committee=changed),
                        append=True).complete(status="converged")

        stage = _read(tmp_path)["stages"][1]
        assert stage["stage_provenance"]["committee_config_sha256"] == "b" * 64
```

Modify the two existing schema assertions:

`tests/test_run_record.py:390-399` becomes

```python
class TestSchemaVersion:
    def test_schema_version_is_four(self):
        """2 added uma_task/mace_head; 3 adds sevennet_task; 4 adds the
        committee block, because collect_provenance took a single
        mlip_model and a committee needs a list."""
        assert SCHEMA_VERSION == 4

    def test_written_record_carries_the_new_version(self, tmp_path):
        _begin(tmp_path)
        assert _read(tmp_path)["schema_version"] == 4
```

`tests/test_core_md.py:151` becomes `assert data["schema_version"] == 4`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_run_record.py tests/test_core_md.py -q -m "not uma and not mace and not sevenn"`
Expected: `TestCommitteeProvenance` fails with `TypeError: collect_provenance() got an unexpected keyword argument 'committee'`, and the two schema-version tests fail with `assert 3 == 4`.

- [ ] **Step 3: Write the implementation**

In `src/mliprun/core/run_record.py`, change line 33:

```python
SCHEMA_VERSION = 4
```

Extend `_PROVENANCE_DIFF_FIELDS` (lines 53-55):

```python
#: Provenance fields compared between the run's origin and an appended
#: stage. Only these fields are meaningful to call out as "what changed" --
#: see I2 in .superpowers/sdd/task-1-fixes.md. The committee is represented
#: here by its config hash, not the nested member list: a one-string diff
#: says "a different committee ran" without copying every member into the
#: stage.
_PROVENANCE_DIFF_FIELDS = ("mliprun_version", "hostname", "device_resolved",
                           "mlip_model", "uma_task", "mace_head",
                           "sevennet_task", "committee_config_sha256")
```

Add the `committee` keyword to `collect_provenance`. Change the signature (lines 166-169) to:

```python
def collect_provenance(*, mlip_model: Any, device_requested: str,
                        device_resolved: str, uma_task: Optional[str] = None,
                        mace_head: Optional[str] = None,
                        sevennet_task: Optional[str] = None,
                        committee: Optional[dict] = None) -> dict:
```

Append to its docstring, before the closing `"""`:

```
    ``committee`` is the block from
    :meth:`~mliprun.core.committee.config.CommitteeConfig.as_provenance` --
    one entry per member with its tag, task or head, env, resolved level of
    theory and GPU, plus the levels present, the ``mixed_theory`` flag and a
    SHA-256 of the ``committee.yaml``. It is added only when a committee
    actually ran, so a single-model record is byte-identical to what schema 3
    wrote apart from the version number. Like everything else here it is
    individually guarded: a committee block that cannot be serialized must
    not cost the run its record.
```

Then, in the return statement (lines 222-237), build the dict and add the committee keys conditionally:

```python
    tag = mlip_model if isinstance(mlip_model, str) else ""
    provenance = {
        "mliprun_version": mliprun_version,
        "ase_version": ase_version,
        "mlip_package": mlip_package,
        "mlip_model": mlip_model,
        "uma_task": uma_task if tag.startswith("uma-") else None,
        "mace_head": mace_head if tag.startswith("mace-mh-") else None,
        "sevennet_task": sevennet_task if tag.startswith("7net") else None,
        "device_requested": device_requested,
        "device_resolved": device_resolved,
        "python_version": python_version,
        "hostname": hostname,
        "started_at": None,
        "finished_at": None,
        "walltime_s": None,
    }
    if committee is not None:
        try:
            provenance["committee"] = _jsonable(committee)
            provenance["committee_config_sha256"] = committee.get(
                "config_sha256")
        except Exception:  # noqa: BLE001 -- never cost the run its record
            provenance["committee"] = {"error": "could not record committee"}
            provenance["committee_config_sha256"] = None
    return provenance
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_run_record.py tests/test_core_md.py tests/test_run_record_integration.py -q -m "not uma and not mace and not sevenn" -rs`
Expected: all pass, 0 unexpected skips.

- [ ] **Step 5: Confirm no golden reference file moved**

Run: `pytest tests/test_characterization_golden.py -q -m "not uma and not mace and not sevenn"`
Expected: pass. The goldens carry no `schema_version`; if one fails here, **stop** and report the numerical delta rather than regenerating it.

- [ ] **Step 6: Commit**

```bash
git add src/mliprun/core/run_record.py tests/test_run_record.py tests/test_core_md.py
git commit -m "feat(record)!: schema 4 with committee provenance

collect_provenance took a single mlip_model; a committee needs a list, which
forces the bump. The block lands only when a committee ran, so single-model
records are unchanged apart from the version number, and the config hash is
promoted to a flat field so an appended stage can report 'a different
committee ran' as one string.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Task 10: Wire the committee into `run_optimization`

**Files:**
- Modify: `src/mliprun/core/optimize.py:41-59` (signature), `:60-122` (docstring), `:126-135` (output paths), `:146-173` (record begin), `:181-191` (`log_convergence`), `:193-223` (run and complete)
- Test: `tests/test_committee_optimize.py`

**Interfaces:**
- Consumes: `CommitteeCalculator` (Task 5), `CommitteeTraceWriter`, `write_peratom_sigma`, `uncertainty_summary` (Task 8), `CommitteeConfig` (Task 7), `collect_provenance(committee=...)` (Task 9).
- Produces: `run_optimization(..., committee=None, committee_config=None, uncertainty_threshold=None) -> bool`, writing `<stem>_committee.csv` and `<stem>_committee_peratom.csv` next to the existing `<stem>_convergence.csv`.

**Ownership note:** `run_optimization` never calls `committee.close()`. It did not create the committee, and an API caller may want to reuse a loaded committee across structures exactly as `optimize batch` reuses one calculator. Teardown belongs to whoever built it — the CLI's `finally` in Task 12, plus the `atexit` backstop from Task 3.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_committee_optimize.py`:

```python
"""run_optimization driven by a committee."""
import csv
import json
import sys
from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from mliprun.core.committee.calculator import CommitteeCalculator, CommitteeError
from mliprun.core.committee.remote import RemoteMember
from mliprun.core.optimize import run_optimization

STUB_DIR = Path(__file__).parent / "committee_stubs"


def _rattled():
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 1)
    atoms.rattle(stdev=0.05, seed=7)
    return atoms


def _committee(tmp_path, n=2, argv=None):
    members = [
        RemoteMember(f"member_{i}", sys.executable, mlip="emt",
                     log_path=tmp_path / f"committee_member_{i}.log",
                     timeout=60.0, argv=argv)
        for i in range(n)
    ]
    return CommitteeCalculator(members, mixed_theory=True,
                               levels=("unknown",))


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _record(tmp_path):
    return json.loads((tmp_path / "mliprun_run.json").read_text())


class TestCommitteeRun:
    def test_it_writes_both_committee_files(self, tmp_path):
        atoms = _rattled()
        committee = _committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=30,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee)
        finally:
            committee.close()

        trace = _read_csv(tmp_path / "opt_committee.csv")
        peratom = _read_csv(tmp_path / "opt_committee_peratom.csv")
        assert len(trace) >= 1
        assert len(peratom) == len(atoms)
        assert [int(r["atom_index"]) for r in peratom] == list(
            range(len(atoms)))

    def test_identical_members_reproduce_the_single_model_relaxation(
            self, tmp_path):
        """The committee must not perturb the trajectory when its members
        agree: same final positions, same energy, sigma identically zero."""
        atoms = _rattled()
        committee = _committee(tmp_path, n=3)
        committee.start()
        atoms.calc = committee
        try:
            converged = run_optimization(atoms, fmax=0.05, max_steps=50,
                                         output_dir=tmp_path,
                                         model_name="committee",
                                         verbose=False, committee=committee)
        finally:
            committee.close()

        reference = _rattled()
        reference.calc = EMT()
        BFGS(reference, logfile=str(tmp_path / "ref.log")).run(fmax=0.05,
                                                               steps=50)
        assert converged is True
        assert atoms.get_positions() == pytest.approx(
            reference.get_positions(), abs=1e-8)

        trace = _read_csv(tmp_path / "opt_committee.csv")
        assert all(float(r["sigma_max_eV_per_A"]) == pytest.approx(0.0,
                                                                   abs=1e-12)
                   for r in trace)
        assert all(float(r["energy_spread_aligned_eV"]) == pytest.approx(
            0.0, abs=1e-12) for r in trace)

    def test_the_trace_has_one_row_per_optimizer_step(self, tmp_path):
        atoms = _rattled()
        committee = _committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=30,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee)
        finally:
            committee.close()

        trace = _read_csv(tmp_path / "opt_committee.csv")
        convergence = _read_csv(tmp_path / "opt_convergence.csv")
        assert len(trace) == len(convergence)
        assert [int(r["step"]) for r in trace] == [int(r["step"])
                                                   for r in convergence]
        for committee_row, convergence_row in zip(trace, convergence):
            assert float(committee_row["fmax_eV_per_A"]) == pytest.approx(
                float(convergence_row["fmax(eV/A)"]), abs=1e-12)
            assert float(committee_row["energy_mean_eV"]) == pytest.approx(
                float(convergence_row["energy(eV)"]), abs=1e-10)

    def test_the_mixed_theory_flag_reaches_every_row(self, tmp_path):
        atoms = _rattled()
        committee = _committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=10,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee)
        finally:
            committee.close()
        assert all(r["mixed_theory"] == "True"
                   for r in _read_csv(tmp_path / "opt_committee.csv"))


class TestRunRecord:
    def test_the_record_carries_members_and_uncertainty(self, tmp_path,
                                                        fake_committee_file):
        path, config = fake_committee_file
        atoms = _rattled()
        committee = _committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=30,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee,
                             committee_config=config)
        finally:
            committee.close()

        record = _record(tmp_path)
        assert record["schema_version"] == 4
        assert len(record["provenance"]["committee"]["members"]) == 2
        assert record["provenance"]["committee_config_sha256"] == config.sha256

        uncertainty = record["stages"][0]["results"]["committee_uncertainty"]
        assert uncertainty["threshold_source"] == "fmax"
        assert uncertainty["threshold_eV_per_A"] == pytest.approx(0.05,
                                                                  abs=1e-12)
        assert uncertainty["sigma_max_final_eV_per_A"] == pytest.approx(
            0.0, abs=1e-12)
        assert uncertainty["flagged"] is False

    def test_an_explicit_threshold_is_recorded_as_explicit(self, tmp_path):
        atoms = _rattled()
        committee = _committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=10,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee,
                             uncertainty_threshold=0.2)
        finally:
            committee.close()
        uncertainty = _record(tmp_path)["stages"][0]["results"][
            "committee_uncertainty"]
        assert uncertainty["threshold_source"] == "explicit"
        assert uncertainty["threshold_eV_per_A"] == pytest.approx(0.2,
                                                                  abs=1e-12)


class TestPartialResults:
    def test_a_member_dying_mid_run_keeps_the_steps_already_written(
            self, tmp_path, monkeypatch):
        """A committee run that dies at step N keeps its first N steps."""
        monkeypatch.setenv("MLIPRUN_STUB_DIE_AFTER", "4")
        atoms = _rattled()
        committee = _committee(
            tmp_path, argv=[sys.executable, str(STUB_DIR / "dying_worker.py")])
        committee.start()
        atoms.calc = committee
        try:
            with pytest.raises(CommitteeError):
                run_optimization(atoms, fmax=0.001, max_steps=100,
                                 output_dir=tmp_path, model_name="committee",
                                 verbose=False, committee=committee)
        finally:
            committee.close()

        trace = _read_csv(tmp_path / "opt_committee.csv")
        assert 1 <= len(trace) <= 4
        assert _record(tmp_path)["status"] == "failed"
        assert "committee incomplete" in (
            _record(tmp_path)["stages"][0]["results"]["error"])


class TestSingleModelUnchanged:
    def test_no_committee_files_without_a_committee(self, tmp_path):
        atoms = _rattled()
        atoms.calc = EMT()
        run_optimization(atoms, fmax=0.05, max_steps=30, output_dir=tmp_path,
                         model_name="emt", verbose=False)
        assert not (tmp_path / "opt_committee.csv").exists()
        assert not (tmp_path / "opt_committee_peratom.csv").exists()
        assert "committee" not in _record(tmp_path)["provenance"]
```

Add this fixture to `tests/conftest.py` (it is reused by the CLI tests in Task 12):

```python
@pytest.fixture
def fake_committee_file(tmp_path):
    """A two-member committee.yaml pointing at the current interpreter.

    Both members are the reserved ``emt`` tag, so the file is usable end to
    end with no MLIP installed. Returns ``(path, CommitteeConfig)``.
    """
    from mliprun.core.committee.config import load_committee

    env = Path(sys.executable).parents[1]
    path = tmp_path / "committee.yaml"
    path.write_text(
        "members:\n"
        f"  - {{env: {env}, mlip: emt, name: member_a}}\n"
        f"  - {{env: {env}, mlip: emt, name: member_b}}\n",
        encoding="utf-8")
    return path, load_committee(path)
```

with `import sys` and `from pathlib import Path` already present at the top of `conftest.py` (`Path` is; add `import sys`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_optimize.py -q`
Expected: failures with `TypeError: run_optimization() got an unexpected keyword argument 'committee'`.

- [ ] **Step 3: Extend the signature and docstring**

In `src/mliprun/core/optimize.py`, add three parameters at the end of the signature (after `sevennet_task`, line 58):

```python
    sevennet_task: Optional[str] = None,
    committee=None,
    committee_config=None,
    uncertainty_threshold: Optional[float] = None,
) -> bool:
```

Add to the docstring's parameter list, after the `mace_head` entry:

```
    committee : CommitteeCalculator, optional
        The committee driving this relaxation, when there is one. It must
        already be started and attached as ``atoms.calc``; this function only
        reads its per-step statistics and writes the trace. Teardown belongs
        to whoever built it -- an API caller may reuse one loaded committee
        across many structures, exactly as ``optimize batch`` reuses one
        calculator.
    committee_config : CommitteeConfig, optional
        The parsed ``committee.yaml``, for the run record: member list, envs,
        resolved levels of theory, and the file's SHA-256.
    uncertainty_threshold : float, optional
        sigma_max above which the final configuration is flagged as
        high-disagreement. Defaults to ``fmax``: if the models disagree about
        the forces by more than the convergence tolerance, the located
        minimum sits inside the committee's own noise and the geometry is not
        resolved. The value applied and where it came from are both recorded.
```

- [ ] **Step 4: Add the output paths and the threshold resolution**

After line 135 (`contcar_file = output_path / "CONTCAR"`), add:

```python
    committee_csv = output_path / f"{logfile_stem}_committee.csv"
    committee_peratom_csv = output_path / f"{logfile_stem}_committee_peratom.csv"

    # Default threshold is fmax itself -- self-scaling, and physically
    # motivated. UNCALIBRATED: no same-level sigma_F measurement exists yet
    # (see the design's "Flagging rule"), so both the value and its origin
    # travel with the flag.
    threshold = (float(uncertainty_threshold)
                 if uncertainty_threshold is not None else float(fmax))
    threshold_source = "explicit" if uncertainty_threshold is not None else "fmax"
```

- [ ] **Step 5: Extend the record's parameters and provenance**

Add this helper above `run_optimization`. It is a helper rather than an inline
expression because `committee` and `committee_config` are independently
optional: an API caller may pass a committee it built by hand, with no YAML
behind it.

```python
def _committee_parameters(committee, committee_config, threshold) -> dict:
    """Committee-only entries for the record's parameter block."""
    if committee is None:
        return {}
    parameters = {"uncertainty_threshold": threshold,
                  "n_members": len(committee.members)}
    if committee_config is not None:
        parameters["committee_file"] = committee_config.source_path
    return parameters
```

Then replace the `parameters=` and `provenance=` arguments of `RunRecord.begin`
(lines 150-171) with:

```python
        parameters={
            "optimizer": optimizer_name,
            "fmax": fmax,
            "max_steps": max_steps,
            "relax_cell": relax_cell,
            "trajectory": trajectory,
            "logfile": logfile,
            "plot": plot,
            "verbose": verbose,
            **_committee_parameters(committee, committee_config, threshold),
        },
        inputs={
            "n_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
        },
        provenance=collect_provenance(
            mlip_model=model_name,
            device_requested=device_requested,
            device_resolved=device_resolved,
            uma_task=uma_task,
            mace_head=mace_head,
            sevennet_task=sevennet_task,
            committee=(committee_config.as_provenance()
                       if committee_config is not None else None),
        ),
```

- [ ] **Step 6: Write the trace as the optimization runs**

Add the imports at the top of `src/mliprun/core/optimize.py`:

```python
from mliprun.core.committee.calculator import (
    CommitteeTraceWriter,
    uncertainty_summary,
    write_peratom_sigma,
)
```

Replace the block from `log_data = {...}` through the end of the `except` handler (lines 181-210) with:

```python
    log_data = {"step": [], "energy(eV)": [], "fmax(eV/A)": []}

    trace_writer = (
        CommitteeTraceWriter(committee_csv, committee.member_names,
                             committee.mixed_theory)
        if committee is not None else None
    )

    def log_convergence():
        step = opt.nsteps
        energy = atoms.get_potential_energy()
        # opt_target.get_forces() includes cell virials when relax_cell is on,
        # matching what the optimizer's fmax convergence is checking against.
        fmax_val = calc_fmax(opt_target.get_forces())
        log_data["step"].append(step)
        log_data["energy(eV)"].append(energy)
        log_data["fmax(eV/A)"].append(fmax_val)
        if trace_writer is not None and committee.latest is not None:
            # Both calls above hit the (cached) committee at this geometry, so
            # `latest` is this step's evaluation. Flushed per row, so a run
            # that dies at step 300 keeps its first 300 steps.
            trace_writer.write_step(step, committee.latest, fmax_val)

    try:
        if verbose:
            opt = OptimizerClass(opt_target, trajectory=str(traj_file), logfile=str(log_file))
            opt.attach(log_convergence, interval=1)
            logger.info("Starting optimization with %s (fmax=%.4f, max_steps=%d, relax_cell=%s)",
                        optimizer.upper(), fmax, max_steps, relax_cell)
            converged = opt.run(fmax=fmax, steps=max_steps)
        else:
            with open(log_file, "w") as lf:
                opt = OptimizerClass(opt_target, trajectory=str(traj_file), logfile=lf)
                opt.attach(log_convergence, interval=1)
                converged = opt.run(fmax=fmax, steps=max_steps)

        final_energy = atoms.get_potential_energy()
        final_fmax = calc_fmax(opt_target.get_forces())
    except Exception as exc:
        # Partial results survive: the trace is already on disk, and the
        # record says what the committee had seen when the run died.
        results = {"error": str(exc)}
        if trace_writer is not None:
            trace_writer.close()
            results["committee_uncertainty"] = uncertainty_summary(
                trace_writer.rows, committee.latest, threshold=threshold,
                threshold_source=threshold_source,
                symbols=atoms.get_chemical_symbols())
        record.complete(status="failed", results=results)
        raise
```

- [ ] **Step 7: Finish the committee outputs on the success path**

Replace the `record.complete(...)` call on the success path (lines 215-223) with:

```python
    results = {
        "converged": bool(converged),
        "final_energy_eV": float(final_energy),
        "final_fmax_eV_per_A": float(final_fmax),
    }
    if trace_writer is not None:
        trace_writer.close()
        write_peratom_sigma(committee_peratom_csv,
                            atoms.get_chemical_symbols(),
                            committee.latest["sigma_per_atom"])
        summary = uncertainty_summary(
            trace_writer.rows, committee.latest, threshold=threshold,
            threshold_source=threshold_source,
            symbols=atoms.get_chemical_symbols())
        results["committee_uncertainty"] = summary
        if summary["flagged"]:
            logger.warning(
                "High committee disagreement at the final geometry: "
                "sigma_max = %.4f eV/Ang > %.4f (%s). The located minimum "
                "sits inside the committee's own noise; this configuration "
                "deserves a DFT check. Worst atom: %s (%s).",
                summary["sigma_max_final_eV_per_A"], threshold,
                threshold_source, summary["worst_atom"],
                summary["worst_atom_symbol"])

    record.complete(
        status="converged" if converged else "not_converged",
        steps=int(opt.nsteps),
        results=results,
    )
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `pytest tests/test_committee_optimize.py tests/test_core_optimize.py tests/test_optimize_batch.py -q -m "not uma and not mace and not sevenn" -rs`
Expected: all pass, 0 unexpected skips.

- [ ] **Step 9: Commit**

```bash
git add src/mliprun/core/optimize.py tests/test_committee_optimize.py tests/conftest.py
git commit -m "feat(optimize): committee-driven relaxation with per-step uncertainty

run_optimization gains committee/committee_config/uncertainty_threshold and
writes opt_committee.csv (one row per optimizer step, flushed as it goes)
plus opt_committee_peratom.csv at the final geometry. Teardown stays with
whoever built the committee, so one loaded committee can serve many
structures.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Task 11: σ trace on the convergence figure

**Files:**
- Modify: `src/mliprun/core/optimize.py:231-253` (the `if plot:` block)
- Test: `tests/test_committee_plot.py`

**Interfaces:**
- Consumes: `trace_writer.rows` (Task 8/10).
- Produces: `_plot_convergence(df, fmax, optimizer, committee_rows=None) -> matplotlib.figure.Figure`

- [ ] **Step 1: Write the failing test**

Create `tests/test_committee_plot.py`:

```python
"""The optional sigma panel on the convergence figure."""
import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt          # noqa: E402 -- after use("Agg")

from mliprun.core.optimize import _plot_convergence  # noqa: E402


@pytest.fixture
def convergence_frame():
    return pd.DataFrame({"step": [0, 1, 2],
                         "energy(eV)": [-1.0, -1.2, -1.3],
                         "fmax(eV/A)": [0.9, 0.3, 0.02]})


def _rows(sigmas):
    return [{"step": i, "sigma_max_eV_per_A": s,
             "sigma_mean_eV_per_A": s / 2.0}
            for i, s in enumerate(sigmas)]


def test_without_a_committee_the_figure_has_two_panels(convergence_frame):
    figure = _plot_convergence(convergence_frame, 0.05, "bfgs")
    assert len(figure.axes) == 2
    plt.close(figure)


def test_with_a_committee_a_third_panel_carries_sigma(convergence_frame):
    figure = _plot_convergence(convergence_frame, 0.05, "bfgs",
                               committee_rows=_rows([0.9, 0.3, 0.02]))
    assert len(figure.axes) == 3
    sigma_axis = figure.axes[2]
    plotted = sigma_axis.lines[0].get_ydata()
    assert list(plotted) == pytest.approx([0.9, 0.3, 0.02], abs=1e-12)
    assert sigma_axis.get_yscale() == "log"
    plt.close(figure)


def test_an_empty_committee_trace_falls_back_to_two_panels(convergence_frame):
    figure = _plot_convergence(convergence_frame, 0.05, "bfgs",
                               committee_rows=[])
    assert len(figure.axes) == 2
    plt.close(figure)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_committee_plot.py -q`
Expected: `ImportError: cannot import name '_plot_convergence'`

- [ ] **Step 3: Extract the plotting helper and add the σ panel**

In `src/mliprun/core/optimize.py`, replace the whole `if plot:` block (lines 231-253) with a call:

```python
    # Plot convergence (skippable: the figure + savefig is per-structure IO that
    # dominates short relaxations; the CSV above retains the same data).
    if plot:
        figure = _plot_convergence(
            df, fmax, optimizer,
            committee_rows=(trace_writer.rows if trace_writer is not None
                            else None))
        figure.savefig(convergence_plot, dpi=150)
        plt.close(figure)
```

and add the helper above `run_optimization`:

```python
def _plot_convergence(df, fmax: float, optimizer: str, committee_rows=None):
    """Build the convergence figure.

    Two panels normally -- energy and max force. A committee run gets a third
    carrying the per-atom force disagreement, on the same log scale as the
    force panel so the two are read against each other: where sigma_max
    approaches fmax, the minimum sits inside the committee's own noise.

    Returns
    -------
    matplotlib.figure.Figure
        The caller saves and closes it.
    """
    n_panels = 3 if committee_rows else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4 * n_panels))
    ax1, ax2 = axes[0], axes[1]

    ax1.plot(df["step"], df["energy(eV)"], marker="o", markersize=4,
             linewidth=1.5)
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title(f"Energy Convergence ({optimizer.upper()})")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["step"], df["fmax(eV/A)"], marker="o", markersize=4,
             linewidth=1.5, color="orange")
    ax2.axhline(y=fmax, color="r", linestyle="--",
                label=f"fmax target = {fmax}")
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Max Force (eV/Ang)")
    ax2.set_title("Force Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    if committee_rows:
        ax3 = axes[2]
        steps = [row["step"] for row in committee_rows]
        ax3.plot(steps, [row["sigma_max_eV_per_A"] for row in committee_rows],
                 marker="o", markersize=4, linewidth=1.5,
                 label="sigma_max")
        ax3.plot(steps, [row["sigma_mean_eV_per_A"] for row in committee_rows],
                 marker="s", markersize=3, linewidth=1.0, label="sigma_mean")
        ax3.axhline(y=fmax, color="r", linestyle="--",
                    label=f"fmax target = {fmax}")
        ax3.set_xlabel("Optimization Step")
        ax3.set_ylabel("Force disagreement (eV/Ang)")
        ax3.set_title("Committee Disagreement")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale("log")

    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_plot.py tests/test_cli_run_plot.py -q -m "not uma and not mace and not sevenn" -rs`
Expected: all pass, 0 unexpected skips.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/optimize.py tests/test_committee_plot.py
git commit -m "feat(optimize): sigma panel on the convergence figure

Extracts the plotting into _plot_convergence and adds a third panel for a
committee run, on the same log scale as the force panel so the two read
against each other.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Task 12: `optimize run --committee`

**Files:**
- Modify: `src/mliprun/cli/commands/optimize.py:1-23` (imports), `:51-73` (options), `:79-155` (body), `:362-383` (`_write_params`)
- Test: `tests/test_committee_cli.py`

**Interfaces:**
- Consumes: `load_committee`, `CommitteeConfigError` (Task 7); `RemoteMember`, `MemberError`, `DEFAULT_CALC_TIMEOUT_S` (Task 3); `CommitteeCalculator`, `CommitteeError` (Task 5); `run_optimization(committee=...)` (Task 10).
- Produces: `mlip optimize run --committee PATH [--member-timeout S] [--uncertainty-threshold X]`; helper `_build_committee(config, output_dir, member_timeout) -> CommitteeCalculator`.

**Scope:** `optimize run` only. `optimize batch` keeps its single-model path — the spec's first deliverable is `optimize`, and batch's "load the model once" contract needs its own thinking about committee reuse across structures.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_committee_cli.py`:

```python
"""CLI surface for committee runs."""
import json

import pytest
from ase.build import bulk
from ase.io import write
from typer.testing import CliRunner

from mliprun.cli.commands.optimize import app

runner = CliRunner()


@pytest.fixture
def structure(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 1)
    atoms.rattle(stdev=0.05, seed=7)
    path = tmp_path / "POSCAR"
    write(str(path), atoms, format="vasp")
    return path


class TestMutualExclusion:
    def test_committee_with_an_explicit_mlip_is_rejected(
            self, structure, fake_committee_file):
        path, _ = fake_committee_file
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path),
                                     "--mlip", "uma-s-1p2"])
        assert result.exit_code == 1
        assert "--mlip" in result.output
        assert "--committee" in result.output

    def test_committee_with_an_explicit_task_is_rejected(
            self, structure, fake_committee_file):
        """The committee file owns model selection; a stray --uma-task would
        silently do nothing."""
        path, _ = fake_committee_file
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path),
                                     "--uma-task", "oc20"])
        assert result.exit_code == 1
        assert "uma-task" in result.output

    def test_the_default_mlip_value_does_not_trip_the_check(
            self, structure, fake_committee_file):
        """--mlip defaults to 'auto'; only an explicitly typed one conflicts."""
        path, _ = fake_committee_file
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path),
                                     "--max-steps", "5", "--no-verbose"])
        assert result.exit_code == 0


class TestConfigErrors:
    def test_a_missing_committee_file_exits_one(self, structure, tmp_path):
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee",
                                     str(tmp_path / "absent.yaml")])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_a_single_member_file_exits_one(self, structure, tmp_path):
        import sys
        from pathlib import Path
        env = Path(sys.executable).parents[1]
        path = tmp_path / "one.yaml"
        path.write_text(f"members:\n  - {{env: {env}, mlip: emt}}\n",
                        encoding="utf-8")
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path)])
        assert result.exit_code == 1
        assert "at least two" in result.output


class TestEndToEnd:
    def test_a_two_member_run_writes_every_output(self, structure,
                                                  fake_committee_file):
        path, config = fake_committee_file
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path),
                                     "--fmax", "0.05", "--max-steps", "50",
                                     "--no-verbose"])
        assert result.exit_code == 0, result.output

        out = structure.parent
        assert (out / "opt_committee.csv").exists()
        assert (out / "opt_committee_peratom.csv").exists()
        assert (out / "opt_convergence.csv").exists()
        assert (out / "CONTCAR").exists()
        assert (out / "committee_member_a.log").exists()
        assert (out / "committee_member_b.log").exists()

        record = json.loads((out / "mliprun_run.json").read_text())
        assert record["schema_version"] == 4
        assert record["provenance"]["committee"]["config_sha256"] == \
            config.sha256
        uncertainty = record["stages"][0]["results"]["committee_uncertainty"]
        assert uncertainty["sigma_max_final_eV_per_A"] == pytest.approx(
            0.0, abs=1e-12)
        assert uncertainty["flagged"] is False

    def test_the_mixed_theory_warning_is_printed(self, structure,
                                                 fake_committee_file):
        """Both members are the reserved 'emt' tag, which resolves to
        'unknown' -- unknown counts as possibly mixed, so the warning fires."""
        path, _ = fake_committee_file
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path),
                                     "--max-steps", "5", "--no-verbose"])
        assert result.exit_code == 0
        assert "more than one level of theory" in result.output
        assert "not an error bar" in result.output

    def test_params_file_lists_the_members(self, structure,
                                           fake_committee_file):
        path, _ = fake_committee_file
        runner.invoke(app, ["run", "--structure", str(structure),
                            "--committee", str(path), "--max-steps", "5",
                            "--no-verbose"])
        params = (structure.parent / "opt_params.txt").read_text()
        assert "Committee:" in params
        assert "member_a" in params and "member_b" in params
        assert "unknown" in params

    def test_an_explicit_threshold_reaches_the_record(self, structure,
                                                      fake_committee_file):
        path, _ = fake_committee_file
        runner.invoke(app, ["run", "--structure", str(structure),
                            "--committee", str(path), "--max-steps", "5",
                            "--no-verbose",
                            "--uncertainty-threshold", "0.001"])
        record = json.loads(
            (structure.parent / "mliprun_run.json").read_text())
        uncertainty = record["stages"][0]["results"]["committee_uncertainty"]
        assert uncertainty["threshold_source"] == "explicit"
        assert uncertainty["threshold_eV_per_A"] == pytest.approx(0.001,
                                                                  abs=1e-12)


class TestSingleModelPathUnchanged:
    def test_no_committee_flag_takes_the_single_model_path(self, structure,
                                                           monkeypatch):
        """Without --committee nothing about the existing path changes: the
        run still goes through detect_mlip, and no committee code is
        reached."""
        import mliprun.cli.commands.optimize as optimize_cli

        calls = []

        def _fake_detect():
            calls.append("detect")
            raise RuntimeError("no MLIP installed")

        monkeypatch.setattr(optimize_cli, "detect_mlip", _fake_detect)
        result = runner.invoke(app, ["run", "--structure", str(structure)])
        assert calls == ["detect"]
        assert "committee" not in result.output.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_cli.py -q`
Expected: failures — `Error: No such option: --committee`, exit code 2.

- [ ] **Step 3: Add the imports and the options**

In `src/mliprun/cli/commands/optimize.py`, extend the imports:

```python
from mliprun.core.committee.calculator import CommitteeCalculator, CommitteeError
from mliprun.core.committee.config import CommitteeConfigError, load_committee
from mliprun.core.committee.remote import (
    DEFAULT_CALC_TIMEOUT_S,
    MemberError,
    RemoteMember,
)
```

Add three options to `run`, after `sevennet_task` (line 59):

```python
    committee: Path = typer.Option(
        None, "--committee",
        help="Path to a committee.yaml declaring two or more MLIP members, "
             "each in its own env. Drives the relaxation with their mean "
             "force and reports their disagreement as a per-configuration "
             "uncertainty. Mutually exclusive with --mlip and the head/task "
             "options: the file owns model selection."),
    member_timeout: float = typer.Option(
        DEFAULT_CALC_TIMEOUT_S, "--member-timeout",
        help="Seconds a single committee member may take per single-point "
             "before it is killed and the run aborts. Model loading has its "
             "own, much longer budget."),
    uncertainty_threshold: float = typer.Option(
        None, "--uncertainty-threshold",
        help="Flag the final configuration when the committee's per-atom "
             "force disagreement exceeds this (eV/Å). Defaults to --fmax: if "
             "the models disagree by more than the convergence tolerance, "
             "the minimum sits inside the committee's own noise. "
             "UNCALIBRATED default -- see docs/OUTPUTS.md."),
```

- [ ] **Step 4: Add the committee helper**

Add above `run` in the same module:

```python
#: Model-selection options the committee file owns. Passing any of them with
#: --committee is an error rather than a silent precedence rule.
_COMMITTEE_CONFLICTS = ("mlip", "uma_task", "mace_head", "sevennet_task")


def _reject_conflicting_options(ctx) -> None:
    """Fail if a model-selection option was typed alongside --committee.

    Checked against the *parameter source*, not the value: --mlip defaults to
    'auto', so comparing values would either miss a typed 'auto' or reject
    every committee run.
    """
    sources = param_sources_from_ctx(ctx)
    typed = [name for name in _COMMITTEE_CONFLICTS
             if sources.get(name) == "user"]
    if typed:
        flags = ", ".join("--" + name.replace("_", "-") for name in typed)
        typer.echo(
            f"❌ {flags} cannot be combined with --committee.\n"
            f"   The committee file declares the model, task and head for "
            f"every member; a flag here would silently do nothing.")
        raise typer.Exit(1)


def _build_committee(config, output_dir: Path, member_timeout: float):
    """Turn a parsed committee.yaml into a started CommitteeCalculator.

    Each member gets its own worker log next to the run's other outputs --
    that is where library banners and remote tracebacks land, and every error
    message points at it.
    """
    members = [
        RemoteMember(
            spec.name,
            spec.python_exe,
            mlip=spec.mlip,
            uma_task=spec.uma_task,
            mace_head=spec.mace_head,
            sevennet_task=spec.sevennet_task,
            device=spec.device,
            gpu=spec.gpu,
            log_path=output_dir / f"committee_{spec.name}.log",
            timeout=member_timeout,
        )
        for spec in config.members
    ]
    return CommitteeCalculator(members, mixed_theory=config.mixed_theory,
                               levels=config.levels)
```

- [ ] **Step 5: Branch the command body**

Replace the MLIP-selection and calculator-attachment block (lines 84-110) with:

```python
    committee_config = None
    committee_calc = None
    if committee is not None:
        _reject_conflicting_options(ctx)
        try:
            committee_config = load_committee(committee)
        except CommitteeConfigError as exc:
            typer.echo(f"❌ {exc}")
            raise typer.Exit(1)
        mlip = "committee"
        typer.echo(f"🧠 Committee of {len(committee_config.members)} members "
                   f"from {committee}")
        for spec in committee_config.members:
            typer.echo(f"   {spec.name}: {spec.mlip} "
                       f"[{spec.level_of_theory}] gpu={spec.gpu} "
                       f"env={spec.env}")
        if committee_config.mixed_theory:
            typer.echo(f"\n⚠️  {committee_config.mixed_theory_warning()}\n")
    else:
        # Detect or validate MLIP
        if mlip == "auto":
            mlip = detect_mlip()
            typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
            # An auto-detected tag still has to satisfy its own task rules.
            validate_mlip(mlip, sevennet_task, uma_task, mace_head)
        else:
            validate_mlip(mlip, sevennet_task, uma_task, mace_head)
            typer.echo(f"🧠 Using MLIP: {mlip}")

    # Validate optimizer
    if optimizer.lower() not in OPTIMIZER_MAP:
        typer.echo(f"❌ Unknown optimizer: {optimizer}")
        typer.echo(f"   Available: {', '.join(OPTIMIZER_MAP.keys())}")
        raise typer.Exit(1)

    # Output directory
    output_dir = structure.parent

    if committee_config is not None:
        committee_calc = _build_committee(committee_config, output_dir,
                                          member_timeout)
        typer.echo("⚙️  Starting committee members (one model load each)...")
        try:
            committee_calc.start()
            # Every member evaluates the input geometry once, up front:
            # members disagree about what input is valid (fairchem's UMA
            # calculator rejects a pbc=(T,T,F) slab that the others accept),
            # and that must surface now, not on step 400 tonight.
            committee_calc.preflight(atoms)
        except (MemberError, CommitteeError) as exc:
            committee_calc.close()
            typer.echo(f"❌ {exc}")
            raise typer.Exit(1)
        atoms.calc = committee_calc
    else:
        # Assign calculator
        typer.echo(f"⚙️  Attaching {mlip} calculator (device={device})...")
        if mlip.startswith("uma-"):
            typer.echo(f"   UMA task: {uma_task}")
        if mlip.startswith("mace-mh-"):
            typer.echo(f"   MACE head: {mace_head}")
        if mlip.startswith("7net"):
            typer.echo(f"   SevenNet task: {sevennet_task}")
        atoms = setup_calculator(atoms, mlip, uma_task, device=device,
                                  mace_head=mace_head,
                                  sevennet_task=sevennet_task)
```

Delete the now-duplicated `# Output directory` / `output_dir = structure.parent`
lines that followed (old lines 112-113).

Then wrap the call to `run_optimization` (lines 131-149) so teardown happens on
every exit path:

```python
    try:
        converged = run_optimization(
            atoms=atoms,
            optimizer=optimizer,
            fmax=fmax,
            max_steps=max_steps,
            trajectory=trajectory,
            logfile=logfile,
            output_dir=output_dir,
            model_name=mlip,
            verbose=verbose,
            relax_cell=relax_cell,
            plot=plot,
            run_context=run_context,
            device_requested=device,
            device_resolved=_resolve_device(device),
            uma_task=uma_task,
            mace_head=mace_head,
            sevennet_task=sevennet_task,
            committee=committee_calc,
            committee_config=committee_config,
            uncertainty_threshold=uncertainty_threshold,
        )
    finally:
        # A leaked worker holds a CUDA context that makes the GPU look busy
        # to everyone else on the node, and cos-cluster has no scheduler to
        # reap orphans.
        if committee_calc is not None:
            committee_calc.close()
```

- [ ] **Step 6: Extend the params file and the output summary**

Change the `_write_params` call (lines 152-155) to pass the config:

```python
    _write_params(output_dir / "opt_params.txt", mlip, uma_task, mace_head,
                  device, relax_cell, structure.name, optimizer, fmax,
                  max_steps, converged, output_dir,
                  sevennet_task=sevennet_task,
                  committee_config=committee_config,
                  uncertainty_threshold=uncertainty_threshold)
```

Extend `_write_params` (lines 362-383):

```python
def _write_params(param_file, mlip, uma_task, mace_head, device, relax_cell,
                  structure_name, optimizer, fmax, max_steps, converged,
                  output_dir, sevennet_task=None, committee_config=None,
                  uncertainty_threshold=None):
    """Write the per-structure opt_params.txt (matches ``optimize run``)."""
    with open(param_file, "w", encoding="utf-8") as f:
        f.write("Geometry Optimization Parameters\n")
        f.write("=================================\n")
        f.write(f"MLIP model:        {mlip}\n")
        if committee_config is not None:
            f.write(f"Committee:         {committee_config.source_path}\n")
            f.write(f"  sha256:          {committee_config.sha256}\n")
            f.write(f"  mixed theory:    {committee_config.mixed_theory}\n")
            for spec in committee_config.members:
                f.write(f"  - {spec.name}: {spec.mlip} "
                        f"[{spec.level_of_theory}] gpu={spec.gpu} "
                        f"env={spec.env}\n")
            threshold = (uncertainty_threshold
                         if uncertainty_threshold is not None else fmax)
            source = ("explicit" if uncertainty_threshold is not None
                      else "fmax")
            f.write(f"Uncertainty thr.:  {threshold} ({source})\n")
        if mlip.startswith("uma-"):
            f.write(f"UMA task:          {uma_task}\n")
        if mlip.startswith("mace-mh-"):
            f.write(f"MACE head:         {mace_head}\n")
        if mlip.startswith("7net"):
            f.write(f"SevenNet task:     {sevennet_task}\n")
        f.write(f"Device:            {device}\n")
        f.write(f"Relax cell:        {relax_cell}\n")
        f.write(f"Structure:         {structure_name}\n")
        f.write(f"Optimizer:         {optimizer.upper()}\n")
        f.write(f"fmax (eV/Å):       {fmax}\n")
        f.write(f"Max steps:         {max_steps}\n")
        f.write(f"Converged:         {converged}\n")
        f.write(f"Output dir:        {output_dir.resolve()}\n")
```

Extend the output-file summary (lines 162-171) so the committee files are listed:

```python
    output_files = [
        trajectory,
        logfile,
        f"{logfile_stem}_convergence.csv",
        f"{logfile_stem}_final.vasp",
        "CONTCAR",
        "opt_params.txt"
    ]
    if committee_config is not None:
        output_files.insert(3, f"{logfile_stem}_committee.csv")
        output_files.insert(4, f"{logfile_stem}_committee_peratom.csv")
    if plot:
        output_files.insert(3, f"{logfile_stem}_convergence.png")
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest tests/test_committee_cli.py tests/test_cli_commands.py tests/test_cli.py -q -m "not uma and not mace and not sevenn" -rs`
Expected: all pass, 0 unexpected skips.

- [ ] **Step 8: Run the whole unit suite and the coverage gate**

Run:
```bash
pytest -m "not uma and not mace and not sevenn" -q --cov=mliprun --cov-report=xml
diff-cover coverage.xml --compare-branch=origin/main --fail-under=90 --exclude setup.py --exclude "scripts/*"
```
Expected: full suite green; diff coverage ≥ 90%. Then confirm no orphans:
`pgrep -f "mliprun.core.committee" ; echo "exit=$?"` → nothing, `exit=1`.

- [ ] **Step 9: Commit**

```bash
git add src/mliprun/cli/commands/optimize.py tests/test_committee_cli.py
git commit -m "feat(cli): optimize run --committee

Loads the declared members, warns loudly on mixed levels of theory,
evaluates the input geometry on every member before the optimizer starts,
and tears every worker down in a finally. --mlip and the head/task options
are rejected alongside --committee rather than silently ignored, checked
against the parameter source so the 'auto' default does not trip it.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Task 13: Documentation and the example committee file

**Files:**
- Create: `examples/committee.yaml`
- Modify: `docs/OUTPUTS.md`, `docs/PYTHON_API.md`, `README.md`, `AGENTS.md`, `CONTEXT.md`, `docs/adr/0001-per-mlip-envs.md`

- [ ] **Step 1: Write the example committee file**

Create `examples/committee.yaml`:

```yaml
# A committee: several MLIPs, each in its own env, evaluated together.
#
#   mlip optimize run --structure POSCAR --committee committee.yaml --fmax 0.05
#
# The relaxation is driven by the members' MEAN force; their disagreement is
# reported as a per-configuration uncertainty. Every member needs its own
# environment with mliprun installed in it (pip install -e /path/to/mliprun)
# plus exactly one MLIP package -- see docs/install/README.md.
#
# CANON C3: heads and tasks are independent fine-tunes with independent
# energy zeros. A committee spanning two levels of theory reports the LEVEL
# difference, not model error, and mliprun warns loudly when it sees one.
# The example below is deliberately same-level: every member is RPBE/OC20.

members:
  - env:  /scratchb/juar/envs/uma
    mlip: uma-s-1p2
    uma_task: oc20
    gpu: 0

  - env:  /scratchb/juar/envs/sevenn
    mlip: 7net-omni
    sevennet_task: oc20
    gpu: 1

  - env:  /scratchb/juar/envs/mace
    mlip: mace-mh-1
    mace_head: oc20_usemppbe
    gpu: 2

# Optional per member:
#   name:   column header in opt_committee.csv and the worker log filename.
#           Defaults to "<mlip>@<task-or-head>".
#   device: auto (default) | cuda | cpu, resolved inside that member's env.
#   gpu:    sets CUDA_VISIBLE_DEVICES for that member's process only.
#           Omitted, the member inherits the driver's environment.
#
# The reserved tag `emt` builds ASE's built-in EMT calculator and needs no
# MLIP at all. Use it to smoke-test the bridge on a new machine; never for
# science.
```

- [ ] **Step 2: Document the outputs**

Add a section to `docs/OUTPUTS.md`, following the format of the existing
`optimize` section, covering:

- `opt_committee.csv` — the column table from the spec, verbatim, plus these
  three notes: `energy_mean_eV` is meaningless in absolute terms across
  packages but its differences are not; `energy_spread_aligned_eV` is a
  standard deviation (`ddof=1`), not a range, and is zero by construction on
  step 0; `mixed_theory` repeats per row so a downstream filter needs no
  terminal output.
- `opt_committee_peratom.csv` — `atom_index`, `symbol`, `sigma_eV_per_A` at
  the final geometry only, and why it is the most diagnostically useful
  output (it names the atoms the models disagree about, usually the adsorbate
  or the reacting bond).
- `committee_<member>.log` — one per member, carrying that env's library
  banners and any remote traceback.
- The flagging rule and that **the default threshold is uncalibrated**: the
  probe measured σ_F only across mixed-level members, so there is no
  same-level number yet, and calibrating it is a natural first use.
- `mliprun_run.json` schema 4: `provenance.committee` and
  `results.committee_uncertainty`, with the field lists.

- [ ] **Step 3: Document the Python API**

Add to `docs/PYTHON_API.md` a worked example that mirrors the CLI path:

```python
from ase.io import read

from mliprun.core.committee.calculator import CommitteeCalculator
from mliprun.core.committee.config import load_committee
from mliprun.core.committee.remote import RemoteMember
from mliprun.core.optimize import run_optimization

config = load_committee("committee.yaml")
members = [
    RemoteMember(spec.name, spec.python_exe, mlip=spec.mlip,
                 uma_task=spec.uma_task, mace_head=spec.mace_head,
                 sevennet_task=spec.sevennet_task, gpu=spec.gpu,
                 log_path=f"committee_{spec.name}.log")
    for spec in config.members
]
atoms = read("POSCAR")
with CommitteeCalculator(members, mixed_theory=config.mixed_theory,
                         levels=config.levels) as committee:
    committee.start()
    committee.preflight(atoms)
    atoms.calc = committee
    run_optimization(atoms, fmax=0.05, output_dir=".",
                     model_name="committee", committee=committee,
                     committee_config=config)
```

State plainly that the context manager is the teardown contract: a leaked
worker holds a CUDA context.

- [ ] **Step 4: Update README, AGENTS, CONTEXT and ADR 0001**

- `README.md`: a "Committee evaluation" subsection under usage — the one
  command, the one YAML, and the one-sentence statement of what the numbers
  mean and what mixing levels of theory does to them.
- `AGENTS.md`: in "Running", note that `--committee` needs mliprun installed
  in every member env, and that the bridge is POSIX-only. In "Testing", note
  that the committee suite runs with no MLIP installed because the reserved
  `emt` tag exists, and repeat the parametrize-id trap.
- `CONTEXT.md`: add three terms to the Language section — **Committee** (two
  or more MLIP members evaluated together, driving one trajectory with their
  mean force), **Member** (one MLIP tag in one MLIP env, addressed through a
  worker subprocess), **Level of theory** (the label a tag+task resolves to;
  members sharing one are same-level, and `unknown` counts as possibly
  mixed). Add the relationship: a **Committee** contains two or more
  **Members**, each of which is one **MLIP tag** in one **MLIP env**.
- `docs/adr/0001-per-mlip-envs.md`: append a dated note recording that the
  ADR's own revisit trigger fired ("users start asking for 'compare MACE and
  UMA in one run' workflows"), that the calculator-in-subprocess bridge was
  built for `optimize` at a fraction of the multi-month estimate because
  managed environments stayed out of scope, and that the one-MLIP-per-env
  rule is unchanged — the bridge relies on it rather than escaping it. Link
  the design doc and this plan. Do not rewrite the ADR's decision.

- [ ] **Step 5: Verify the example file parses**

Run:
```bash
python - <<'EOF'
from mliprun.core.committee.config import load_committee, CommitteeConfigError
try:
    load_committee("examples/committee.yaml")
except CommitteeConfigError as exc:
    print("expected (the example env paths do not exist here):", exc)
EOF
```
Expected: the "no interpreter found in env" message — the file's *structure*
is valid and only its placeholder paths are absent. Any other error means the
example is malformed; fix it.

- [ ] **Step 6: Full suite, then commit**

```bash
pytest -m "not uma and not mace and not sevenn" -q -rs
git add examples/committee.yaml docs README.md AGENTS.md CONTEXT.md
git commit -m "docs(committee): outputs, Python API, example file, ADR note

Records what the two CSVs mean, that the default flagging threshold is
uncalibrated, and that ADR 0001's own revisit trigger fired -- the bridge
relies on one-MLIP-per-env rather than escaping it.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014Hv6msBZ267YZCMFHfNafe"
```

---

## Final verification before the PR

- [ ] Full unit suite with no MLIP: `pytest -m "not uma and not mace and not sevenn" -q -rs` — green, and no unexpected `skipped` lines among the new tests.
- [ ] Diff coverage: `pytest -q --cov=mliprun --cov-report=xml -m "not uma and not mace and not sevenn"` then `diff-cover coverage.xml --compare-branch=origin/main --fail-under=90 --exclude setup.py --exclude "scripts/*"`.
- [ ] No orphaned workers after the suite: `pgrep -f "mliprun.core.committee"` prints nothing.
- [ ] Goldens untouched: `git status --short tests/goldens/` is empty.
- [ ] `mlip doctor` still exits 0 on an env with one MLIP, and its output is unchanged (this plan does not touch it).
- [ ] Draft PR only, against `main`, one concern.

## Manual verification on cos-cluster (after the unit suite is green)

The unit suite proves the bridge; it cannot prove that four real packages
coexist. Repeat the shape the SevenNet work used:

- [ ] A same-level committee — `uma-s-1p2@oc20` + `7net-omni@oc20` +
  `mace-mh-1@oc20_usemppbe` — relaxing a real adsorbate/slab system.
  Confirm: no mixed-theory warning; `opt_committee.csv` written every step;
  the driver process imports no torch (`py-spy dump` or a `sys.modules` check
  in the same env).
- [ ] Record the **same-level σ_F** it produces. This is the number the
  default flagging threshold needs and the probe could not supply — it is the
  first calibration datapoint, and it belongs in the prov ledger (canon K2)
  with the rest of the run.
- [ ] A deliberately mixed committee (add `chgnet`) to confirm the warning
  fires, the flag reaches every CSV row, and the spread jumps as the probe
  predicts.
- [ ] Kill one member mid-run (`kill -9` on its worker pid) and confirm: the
  run aborts naming that member, the partial CSV survives, the record says
  `failed`, and no worker survives (`nvidia-smi` shows no leftover context).

---

## Self-review

**Spec coverage.** Every section of the design maps to a task: architecture
and the five modules (Tasks 1-8); wire protocol and its three constraints
(Tasks 1, 2, and the driver-side constraint note in Task 10); committee file
and `--committee`/`--mlip` exclusion (Tasks 7, 12); level of theory including
the unknown-warns rule (Task 6); uncertainty metric (Task 4); outputs, both
CSVs and the plot (Tasks 8, 11); flagging rule with its recorded threshold
source (Tasks 8, 10); run record 3 → 4 (Task 9); every listed error-handling
case — failed member, per-request timeout, non-finite values, startup
failure, geometry rejection, teardown on every exit path, partial results
(Tasks 3, 5, 10, 12); every listed test including the parametrize-id trap
(throughout). The spec's five "not in scope" items stay out, and are listed as
such in the File Structure section.

**Open items the plan deliberately leaves to Juan.** The level-of-theory table
in Task 6 is a scientific classification with a review gate on it. The
same-level σ_F calibration is a manual-verification step, not a code task —
the spec names it an open question and the plan carries the threshold's
provenance so a later recalibration can find what was applied.
