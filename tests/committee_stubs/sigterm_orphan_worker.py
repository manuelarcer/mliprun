"""A worker that reports its pid, ignores SIGTERM, and never finishes a calc.

Not a test module: launched as a subprocess through a fake member env whose
``bin/python`` execs this file, so the CLI reaches it without knowing it is a
stub.

It stands in for a real MLIP member that is *inside* a model forward pass
when the driver is killed. That is the state the driver's death is invisible
in: the worker is not at ``stdin.readline()``, so it never observes the
closed pipe, and on cos-cluster (2026-09-07) two such workers outlived their
SIGTERMed driver while still holding GPU memory. Ignoring SIGTERM as well
proves the driver's teardown escalates to SIGKILL rather than merely asking
politely.

The pid goes to stderr, which the driver captures into
``committee_<name>.log``, so a test can detect an orphan without a process
listing -- ``ps`` and ``pgrep`` are blocked in some sandboxes and fail there
in a way that looks like "no orphans".
"""
import os
import signal
import sys
import time

from ase.calculators.emt import EMT

from mliprun.core.committee import protocol, worker


def main():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sys.stderr.write(f"WORKER_PID {os.getpid()}\n")
    sys.stderr.flush()
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
