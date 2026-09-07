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
        # The log was opened (log_path's parent exists) before Popen failed;
        # it must be released deterministically, not left for GC to close.
        assert member._log is None

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
