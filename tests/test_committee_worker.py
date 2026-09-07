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
