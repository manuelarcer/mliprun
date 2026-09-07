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
