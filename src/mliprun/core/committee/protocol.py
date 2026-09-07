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
