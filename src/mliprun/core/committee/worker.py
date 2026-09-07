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
import threading
import time
import traceback

from mliprun.core.committee import protocol

#: Reserved tag: ASE's built-in EMT calculator. Not an MLIP. It exists so the
#: bridge itself can be exercised -- in CI, and by a user smoke-testing a
#: committee.yaml -- on a machine with no MLIP installed at all. EMT is a
#: toy potential for a handful of metals: never use it for science.
EMT_TAG = "emt"

#: How often the worker checks that its driver is still alive. Cheap enough
#: to be invisible next to a model forward pass.
ORPHAN_POLL_S = 2.0

#: Exit status for a worker that outlived its driver, so a worker log tells
#: this apart from a model that crashed.
ORPHAN_EXIT_CODE = 3


def _exit_when_orphaned(poll_s: float = ORPHAN_POLL_S) -> threading.Thread:
    """Quit if the driver that spawned this worker disappears.

    The driver's death normally arrives as EOF on stdin, in ``main()``. That
    covers only an *idle* worker: inside a calculation -- where a committee
    member spends nearly all of a relaxation's wall clock -- this process is
    not reading stdin and cannot see the pipe close at all. Without this
    poll, a driver that dies with no chance to clean up (SIGKILL, an OOM
    kill, a failed node) leaves the worker running and holding its CUDA
    context, which makes the GPU look busy to everyone else on a shared node.

    A driver killed with SIGTERM tears its workers down itself; this is the
    case where it never gets to run any code.

    ``os._exit`` rather than an exception: this runs on a thread, the main
    thread is deep inside a model's forward pass, and the point is to release
    the GPU now rather than to unwind tidily.
    """
    original_parent = os.getppid()

    def _watch() -> None:
        while True:
            time.sleep(poll_s)
            # Compared against the pid recorded at start, not against 1: on
            # a system with a subreaper the orphan is inherited by that,
            # not by init.
            if os.getppid() != original_parent:
                sys.stderr.write(
                    f"committee worker {os.getpid()}: driver "
                    f"{original_parent} is gone, exiting\n")
                sys.stderr.flush()
                os._exit(ORPHAN_EXIT_CODE)

    thread = threading.Thread(target=_watch, name="orphan-watchdog",
                              daemon=True)
    thread.start()
    return thread


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
    """Serve requests until ``quit``, or the driver goes away."""
    _exit_when_orphaned()
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
