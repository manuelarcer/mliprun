"""Helpers shared by the remote-machine tests (plain module, not a fixture file)."""
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

FAKE_SSH = Path(__file__).resolve().parent / "fake_ssh.py"
FAKE_SSH_CMD = [sys.executable, str(FAKE_SSH)]
SRC = str(Path(__file__).resolve().parents[2] / "src")


def agent_env(**extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    env.update(extra)
    return env


def run_agent(*args, stdin=b"", env=None, timeout=120):
    """Run ``python -m mliprun.webapp.remote_agent ...`` as a subprocess."""
    return subprocess.run([sys.executable, "-m", "mliprun.webapp.remote_agent", *map(str, args)],
                          input=stdin, capture_output=True, env=env or agent_env(),
                          timeout=timeout)


def make_tar(members):
    """``members``: list of (name, bytes) or (name, TarInfo-kwargs dict) -> tar bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, spec in members:
            if isinstance(spec, bytes):
                info = tarfile.TarInfo(name)
                info.size = len(spec)
                tar.addfile(info, io.BytesIO(spec))
            else:
                info = tarfile.TarInfo(name)
                for k, v in spec.items():
                    setattr(info, k, v)
                tar.addfile(info)
    return buf.getvalue()


def wait_until(predicate, timeout=60.0, interval=0.05, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout} s waiting for {what}")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
