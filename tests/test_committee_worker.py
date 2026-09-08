"""The committee worker: one MLIP env's side of the bridge.

Exercised through a real subprocess under the current interpreter, building
ASE's EMT. No MLIP is needed.
"""
import os
import signal
import subprocess
import sys
import time
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


#: A driver that starts one real worker, holds it inside a calculation, and
#: then waits to be killed. Written as a script rather than driven in-process
#: because the whole point is that the *parent* dies.
_ORPHANING_DRIVER = """
import os, sys
os.environ["PYTHONPATH"] = {patch_dir!r} + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["MLIPRUN_TEST_EMT_DELAY"] = "60"
sys.path.insert(0, {src!r})
from mliprun.core.committee.remote import RemoteMember

member = RemoteMember("held", sys.executable, mlip="emt",
                      log_path={log!r}, timeout=600.0, load_timeout=120.0)
member.start()
print(member.pid, flush=True)
member.calculate([29], [[0.0, 0.0, 0.0]],
                 [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
                 [True, True, True])
"""

#: Makes every EMT single point take MLIPRUN_TEST_EMT_DELAY seconds, so the
#: worker is reliably *inside* a calculation when its driver is killed.
_SLOW_EMT_SITECUSTOMIZE = """
import os
import time

_delay = os.environ.get("MLIPRUN_TEST_EMT_DELAY")
if _delay:
    from ase.calculators.emt import EMT
    _original = EMT.calculate

    def calculate(self, *args, **kwargs):
        time.sleep(float(_delay))
        return _original(self, *args, **kwargs)

    EMT.calculate = calculate
"""


def _pid_is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TestOrphanWatchdog:
    """A worker must not outlive a driver that died without cleaning up.

    Closing the driver's end of the pipe is the documented backstop, but the
    worker only reaches ``stdin.readline()`` *between* calculations. Inside
    one -- where a committee member spends nearly all of its wall clock --
    the driver's death is invisible to it, so a SIGKILLed, OOM-killed or
    node-failed driver leaves the worker running and holding its GPU memory.
    SIGTERM is handled driver-side; this is the case where the driver never
    gets to run any code at all.
    """

    def test_the_watchdog_exits_with_its_own_status_once_the_parent_changes(
            self, monkeypatch):
        """The decision, checked directly.

        The end-to-end test below proves the worker really goes away, but a
        process that calls ``os._exit`` can report nothing back -- not even
        its coverage -- so the status code and the comparison against the
        *recorded* parent pid are asserted here instead. ``os._exit`` is
        replaced by a ``SystemExit``, which ends the watchdog thread rather
        than the test session.
        """
        from mliprun.core.committee import worker as worker_mod

        exits = []

        def _fake_exit(code):
            exits.append(code)
            raise SystemExit(code)

        # A subreaper, not init: the watchdog must compare against the pid it
        # recorded at startup, not against 1.
        parents = iter([4321, 4321, 99])
        monkeypatch.setattr(worker_mod.os, "_exit", _fake_exit)
        monkeypatch.setattr(worker_mod.os, "getppid",
                            lambda: next(parents, 99))

        thread = worker_mod._exit_when_orphaned(poll_s=0.02)
        thread.join(timeout=10)

        assert exits == [worker_mod.ORPHAN_EXIT_CODE]
        assert worker_mod.ORPHAN_EXIT_CODE == 3

    def test_a_worker_inside_a_calculation_exits_when_its_driver_is_killed(
            self, tmp_path):
        patch_dir = tmp_path / "slow_emt"
        patch_dir.mkdir()
        (patch_dir / "sitecustomize.py").write_text(_SLOW_EMT_SITECUSTOMIZE,
                                                    encoding="utf-8")
        src = str(Path(__file__).resolve().parents[1] / "src")
        script = tmp_path / "driver.py"
        script.write_text(
            _ORPHANING_DRIVER.format(patch_dir=str(patch_dir), src=src,
                                     log=str(tmp_path / "worker.log")),
            encoding="utf-8")

        driver = subprocess.Popen([sys.executable, str(script)],
                                  stdout=subprocess.PIPE, text=True,
                                  start_new_session=True)
        worker_pid = None
        try:
            worker_pid = int(driver.stdout.readline().strip())
            # Well inside the 60 s calculation, so the worker is not at
            # stdin.readline() and cannot see the pipe close.
            time.sleep(3.0)
            assert _pid_is_alive(worker_pid)

            driver.kill()          # SIGKILL: no handler, no teardown, no atexit
            driver.wait(timeout=10)

            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and _pid_is_alive(worker_pid):
                time.sleep(0.2)
            assert not _pid_is_alive(worker_pid), (
                f"worker {worker_pid} outlived its SIGKILLed driver")
        finally:
            if driver.poll() is None:
                driver.kill()
                driver.wait(timeout=10)
            if worker_pid is not None and _pid_is_alive(worker_pid):
                os.kill(worker_pid, signal.SIGKILL)
