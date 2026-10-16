#!/usr/bin/env python3
"""Stand-in for ``ssh`` in the remote-machine tests (REMOTE_PLAN section 10).

Understands ``ssh [-o X] [-p X] [-i X] [-J X] [-l X] -- host command``, runs
``command`` locally with ``/bin/sh -c`` (stdin and stdout pass through) and
exits with its code. Tests start it as ``[sys.executable, <this file>]``.

- host ``down``, or the file named by ``FAKE_SSH_DOWN`` existing: exit 255 with
  an ssh-like message, the command is not run;
- ``FAKE_SSH_LOG``: one JSON line appended per call:
  ``{"argv": [...], "host": ..., "command": ..., "rc": ..., "bytes_out": N}``.
"""
import json
import os
import subprocess
import sys

OPTS_WITH_VALUE = {"-o", "-p", "-i", "-J", "-l"}


def log(entry):
    path = os.environ.get("FAKE_SSH_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")


def main(argv):
    i = 0
    while i < len(argv) and argv[i] != "--":
        if argv[i] in OPTS_WITH_VALUE:
            i += 2
        else:
            i += 1
    if i >= len(argv) or len(argv) - i < 3:
        sys.stderr.write("fake ssh: usage: ssh [options] -- host command\n")
        return 2
    host, command = argv[i + 1], " ".join(argv[i + 2:])
    entry = {"argv": argv, "host": host, "command": command, "rc": None, "bytes_out": 0}
    down_file = os.environ.get("FAKE_SSH_DOWN")
    if host == "down" or (down_file and os.path.exists(down_file)):
        sys.stderr.write(f"ssh: connect to host {host} port 22: Connection refused\n")
        entry["rc"] = 255
        log(entry)
        return 255
    proc = subprocess.Popen(["/bin/sh", "-c", command], stdout=subprocess.PIPE)
    out = sys.stdout.buffer
    try:
        while True:
            block = proc.stdout.read1(1 << 16)
            if not block:
                break
            entry["bytes_out"] += len(block)
            out.write(block)
            out.flush()
    except BrokenPipeError:
        proc.kill()
    rc = proc.wait()
    entry["rc"] = rc
    log(entry)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
