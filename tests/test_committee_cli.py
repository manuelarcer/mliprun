"""CLI surface for committee runs."""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from ase.build import bulk
from ase.io import write
from typer.testing import CliRunner

from mliprun.cli.commands.optimize import app

runner = CliRunner()

STUB_DIR = Path(__file__).parent / "committee_stubs"


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

    def test_committee_with_an_explicit_device_is_rejected(
            self, structure, fake_committee_file):
        """Each member's device comes from committee.yaml (spec.device); a
        stray --device here would silently do nothing rather than apply."""
        path, _ = fake_committee_file
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path),
                                     "--device", "cuda"])
        assert result.exit_code == 1
        assert "--device" in result.output
        assert "--committee" in result.output

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
        from pathlib import Path as _Path
        env = _Path(sys.executable).parents[1]
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
        assert record["schema_version"] == 5
        assert record["provenance"]["committee"]["config_sha256"] == \
            config.sha256
        uncertainty = record["stages"][0]["results"]["committee_uncertainty"]
        assert uncertainty["sigma_max_final_eV_per_A"] == pytest.approx(
            0.0, abs=1e-12)
        # No threshold was passed on this invocation, so no verdict is
        # reached -- see "flagged semantics" in the design note. This is not
        # in Task 5's brief; it broke the moment the fmax default was
        # removed, since this test never passes --uncertainty-threshold.
        assert uncertainty["flagged"] is None

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

    def test_params_file_records_no_threshold_when_none_is_given(
            self, structure, fake_committee_file):
        """opt_params.txt must not claim a verdict nobody asked for.

        This mirrors ``threshold_source`` in the run record: with no
        --uncertainty-threshold, the old code wrote the removed fmax
        default (e.g. "0.05 (fmax)") into this artifact even though the
        run record correctly recorded no verdict -- a false claim in a
        run artifact that survived every earlier task in this plan because
        no test read this specific line.
        """
        path, _ = fake_committee_file
        runner.invoke(app, ["run", "--structure", str(structure),
                            "--committee", str(path), "--max-steps", "5",
                            "--no-verbose"])
        params = (structure.parent / "opt_params.txt").read_text()
        assert "Uncertainty thr.:  none\n" in params
        assert "fmax)" not in params

    def test_params_file_records_the_explicit_threshold(
            self, structure, fake_committee_file):
        path, _ = fake_committee_file
        runner.invoke(app, ["run", "--structure", str(structure),
                            "--committee", str(path), "--max-steps", "5",
                            "--no-verbose",
                            "--uncertainty-threshold", "0.001"])
        params = (structure.parent / "opt_params.txt").read_text()
        assert "Uncertainty thr.:  0.001 (explicit)\n" in params


class TestFlaggedPath:
    """The headline claim -- "flags high-disagreement configurations" --
    exercised through the CLI with a committee that genuinely disagrees.

    Every other end-to-end test here uses two identical EMT members, so
    sigma is ~0 and the flag branch never executes outside a direct unit
    test of ``uncertainty_summary``. ``member_b`` is rerouted (via a
    monkeypatched ``RemoteMember``) to ``biased_worker.py``, which wraps EMT
    and adds a fixed, large force offset -- deterministic disagreement, no
    randomness to make the test flaky.
    """

    def test_a_genuinely_disagreeing_committee_trips_the_threshold(
            self, structure, fake_committee_file, monkeypatch):
        path, config = fake_committee_file
        import mliprun.cli.commands.optimize as optimize_cli
        from mliprun.core.committee.remote import RemoteMember as _RealRemoteMember

        stub = STUB_DIR / "biased_worker.py"

        def _mixed_remote_member(name, python_exe, **kwargs):
            argv = [python_exe, str(stub)] if name == "member_b" else None
            return _RealRemoteMember(name, python_exe, argv=argv, **kwargs)

        monkeypatch.setattr(optimize_cli, "RemoteMember", _mixed_remote_member)

        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path),
                                     "--max-steps", "3", "--no-verbose",
                                     "--uncertainty-threshold", "0.01"])
        assert result.exit_code == 0, result.output
        # Exactly once: `run_optimization`'s own log call is INFO-level and
        # silent at the default logging configuration, so the CLI echo here
        # is the *only* channel -- not a duplicate of a WARNING-level record
        # reaching the terminal via `logging.lastResort`.
        assert result.output.count("deserves a DFT check") == 1

        record = json.loads(
            (structure.parent / "mliprun_run.json").read_text())
        uncertainty = record["stages"][0]["results"]["committee_uncertainty"]
        assert uncertainty["flagged"] is True
        assert uncertainty["sigma_max_final_eV_per_A"] > 0.01


class TestCommitteeUncertaintyEcho:
    def _invoke(self, structure, path, *extra):
        return runner.invoke(app, ["run", "--structure", str(structure),
                                   "--committee", str(path),
                                   "--max-steps", "3", "--no-verbose",
                                   *extra])

    def test_the_numbers_are_printed_without_a_threshold(
            self, structure, fake_committee_file):
        path, _ = fake_committee_file
        result = self._invoke(structure, path)
        assert result.exit_code == 0, result.output
        assert "Committee disagreement at the final geometry" in result.output
        assert "free atoms" in result.output

    def test_no_warning_is_printed_without_a_threshold(
            self, structure, fake_committee_file):
        """A verdict nobody asked for is what this change removes."""
        path, _ = fake_committee_file
        result = self._invoke(structure, path)
        assert "deserves a DFT check" not in result.output

    def test_a_tripped_explicit_threshold_warns(
            self, structure, fake_committee_file):
        """Two identical EMT members give sigma exactly 0, so -1 is the only
        threshold this harness can exceed."""
        path, _ = fake_committee_file
        result = self._invoke(structure, path,
                              "--uncertainty-threshold", "-1")
        assert result.exit_code == 0, result.output
        assert "deserves a DFT check" in result.output

    def test_an_untripped_explicit_threshold_does_not_warn(
            self, structure, fake_committee_file):
        path, _ = fake_committee_file
        result = self._invoke(structure, path,
                              "--uncertainty-threshold", "1e9")
        assert "deserves a DFT check" not in result.output
        assert "Committee disagreement at the final geometry" in result.output


class TestTeardownOnFailure:
    """The CLI owns committee teardown, and must run it on every exit path.

    A leaked worker holds a CUDA context that makes the GPU look busy to
    everyone else on a shared node, so these are the highest-stakes tests in
    this file even though they were not in the original brief.
    """

    def test_a_startup_failure_still_closes_every_member(self, structure,
                                                          tmp_path,
                                                          monkeypatch):
        """One member has an unloadable tag: start() fails, and the CLI's
        own except-block close() must run (CommitteeCalculator.start()
        already closes itself first -- close() is idempotent, and both
        calls must land without error)."""
        import sys
        from mliprun.core.committee.calculator import CommitteeCalculator
        import mliprun.core.committee.remote as remote_mod

        env = Path(sys.executable).parents[1]
        path = tmp_path / "committee.yaml"
        path.write_text(
            "members:\n"
            f"  - {{env: {env}, mlip: emt, name: member_a}}\n"
            f"  - {{env: {env}, mlip: not-a-real-mlip-tag, "
            f"name: member_b}}\n",
            encoding="utf-8")

        close_calls = []
        original_close = CommitteeCalculator.close

        def _spy_close(self):
            close_calls.append(self)
            return original_close(self)

        monkeypatch.setattr(CommitteeCalculator, "close", _spy_close)

        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path)])
        assert result.exit_code == 1
        assert "member_b" in result.output
        assert len(close_calls) >= 1
        live_names = {m.name for m in remote_mod._LIVE}
        assert "member_a" not in live_names
        assert "member_b" not in live_names

    def test_a_keyboard_interrupt_mid_run_still_closes_every_member(
            self, structure, fake_committee_file, monkeypatch):
        """run_optimization raising KeyboardInterrupt must still hit the
        CLI's `finally`: teardown does not depend on how the run ended."""
        path, _ = fake_committee_file
        import mliprun.cli.commands.optimize as optimize_cli
        import mliprun.core.committee.remote as remote_mod

        def _interrupt(*args, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(optimize_cli, "run_optimization", _interrupt)

        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path),
                                     "--no-verbose"])
        assert result.exit_code == 130
        live_names = {m.name for m in remote_mod._LIVE}
        assert "member_a" not in live_names
        assert "member_b" not in live_names


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


def _pid_is_alive(pid: int) -> bool:
    """Signal 0 rather than a process listing.

    ``ps``/``pgrep`` are blocked in some sandboxes and fail there in a way
    that reads as "no such process", which would turn this whole class into
    a test that always passes.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TestSigtermTeardown:
    """SIGTERM must tear the committee down, not orphan it.

    SIGTERM's default disposition kills the process without unwinding the
    stack, so the CLI's teardown and the ``atexit`` backstop in ``remote``
    are both skipped. The worker's own defence -- noticing the driver's
    closed pipe -- is unreachable while it is inside a calculation, because
    it is not reading stdin then. On cos-cluster (2026-09-07) that left two
    workers running with 2948 MiB of GPU memory held after their driver was
    killed. Ctrl+C was never affected and must stay that way.
    """

    def test_the_guard_installs_and_restores_the_sigterm_disposition(self):
        """``run_optimization`` is a Python API entry point too, so the
        handler must not outlive the committee window."""
        from mliprun.cli.commands.optimize import _sigterm_as_interrupt

        before = signal.getsignal(signal.SIGTERM)
        with _sigterm_as_interrupt():
            during = signal.getsignal(signal.SIGTERM)
            assert callable(during)
            assert during is not before
        assert signal.getsignal(signal.SIGTERM) is before

    def test_the_installed_handler_unwinds_instead_of_terminating(self):
        """The whole fix is this: SIGTERM raises, so ``finally`` runs."""
        from mliprun.cli.commands.optimize import _sigterm_as_interrupt

        with _sigterm_as_interrupt():
            handler = signal.getsignal(signal.SIGTERM)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGTERM, None)

    def test_a_caller_on_a_worker_thread_keeps_its_own_disposition(self):
        """Only the main thread may install a handler, and only the main
        thread ever runs one. A caller driving this command from another
        thread must not have the process's disposition changed underneath
        them."""
        from mliprun.cli.commands.optimize import _sigterm_as_interrupt

        observed = {}

        def _inside():
            with _sigterm_as_interrupt():
                observed["during"] = signal.getsignal(signal.SIGTERM)

        before = signal.getsignal(signal.SIGTERM)
        thread = threading.Thread(target=_inside)
        thread.start()
        thread.join(timeout=10)
        assert observed["during"] is before
        assert signal.getsignal(signal.SIGTERM) is before

    def test_a_committee_run_is_armed_while_it_holds_workers(
            self, structure, fake_committee_file, monkeypatch):
        """Armed for the whole window in which workers exist, and disarmed
        again once they are gone."""
        path, _ = fake_committee_file
        import mliprun.cli.commands.optimize as optimize_cli

        observed = {}

        def _capture(*args, **kwargs):
            observed["during"] = signal.getsignal(signal.SIGTERM)
            return True

        monkeypatch.setattr(optimize_cli, "run_optimization", _capture)

        before = signal.getsignal(signal.SIGTERM)
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path),
                                     "--no-verbose"])
        assert result.exit_code == 0
        assert observed["during"] is not before
        assert observed["during"] is not signal.SIG_DFL
        assert signal.getsignal(signal.SIGTERM) is before

    def test_a_sigterm_mid_run_closes_every_member_and_exits_143(
            self, structure, fake_committee_file, monkeypatch):
        """The handler is fired from inside the run, which is exactly what a
        real SIGTERM does at the next bytecode boundary. A real signal is not
        sent here because an unarmed process would take the default
        disposition and kill the whole pytest session; the end-to-end proof
        with a real signal is the subprocess test below.

        143 is 128 + SIGTERM, so a shell still reads the run as terminated
        rather than as a clean exit.
        """
        path, _ = fake_committee_file
        import mliprun.cli.commands.optimize as optimize_cli
        import mliprun.core.committee.remote as remote_mod

        def _terminate(*args, **kwargs):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            raise AssertionError("the SIGTERM handler did not raise")

        monkeypatch.setattr(optimize_cli, "run_optimization", _terminate)

        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path),
                                     "--no-verbose"])
        assert result.exit_code == 143
        live_names = {m.name for m in remote_mod._LIVE}
        assert "member_a" not in live_names
        assert "member_b" not in live_names

    def test_the_single_model_path_leaves_sigterm_alone(self, structure,
                                                        monkeypatch):
        """Scoped to committee runs: a single-model run keeps whatever
        disposition the process already had."""
        import mliprun.cli.commands.optimize as optimize_cli

        observed = {}

        def _capture(*args, **kwargs):
            observed["during"] = signal.getsignal(signal.SIGTERM)
            return True

        # No MLIP is installed in this env, and this test is about the
        # signal disposition, not about model availability.
        monkeypatch.setattr(optimize_cli, "validate_mlip",
                            lambda *a, **k: None)
        monkeypatch.setattr(optimize_cli, "setup_calculator",
                            lambda atoms, *a, **k: atoms)
        monkeypatch.setattr(optimize_cli, "run_optimization", _capture)

        before = signal.getsignal(signal.SIGTERM)
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--mlip", "uma-s-1p2",
                                     "--uma-task", "oc20",
                                     "--no-verbose"])
        assert result.exit_code == 0
        assert observed["during"] is before

    def test_sigterm_to_a_real_cli_subprocess_leaves_no_worker(
            self, structure, tmp_path):
        """The end-to-end guard, with a real signal to a real driver.

        One member is held inside a calculation and ignores SIGTERM itself,
        which is the cluster's situation: teardown has to escalate to
        SIGKILL on the worker's process group. Unlike the in-process tests
        above, this one can send the real signal, because the driver is a
        separate process.
        """
        member_env = tmp_path / "hanging_env"
        (member_env / "bin").mkdir(parents=True)
        launcher = member_env / "bin" / "python"
        launcher.write_text(
            "#!/bin/sh\n"
            f'exec "{sys.executable}" '
            f'"{STUB_DIR / "sigterm_orphan_worker.py"}"\n')
        launcher.chmod(0o755)

        config = tmp_path / "committee.yaml"
        config.write_text(
            "members:\n"
            f"  - {{env: {member_env}, mlip: emt, name: held}}\n"
            f"  - {{env: {Path(sys.executable).parents[1]}, mlip: emt, "
            f"name: normal}}\n",
            encoding="utf-8")

        driver = subprocess.Popen(
            [sys.executable, "-m", "mliprun.cli.main", "optimize", "run",
             "--structure", str(structure), "--committee", str(config),
             "--no-verbose"],
            cwd=str(tmp_path), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
            start_new_session=True)

        worker_pid = None
        try:
            member_log = tmp_path / "committee_held.log"
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if member_log.exists():
                    match = re.search(r"WORKER_PID (\d+)",
                                      member_log.read_text())
                    if match:
                        worker_pid = int(match.group(1))
                        break
                assert driver.poll() is None, (
                    f"driver exited early: {driver.communicate()[0]}")
                time.sleep(0.1)
            assert worker_pid is not None, "the held member never started"

            # Let the driver get past the loads and into the evaluation the
            # held member never answers.
            time.sleep(2.0)
            assert _pid_is_alive(worker_pid)

            os.kill(driver.pid, signal.SIGTERM)
            driver.communicate(timeout=60)

            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and _pid_is_alive(worker_pid):
                time.sleep(0.2)
            assert not _pid_is_alive(worker_pid), (
                f"worker {worker_pid} outlived its SIGTERMed driver")
        finally:
            if driver.poll() is None:
                driver.kill()
                driver.wait(timeout=10)
            if worker_pid is not None and _pid_is_alive(worker_pid):
                os.kill(worker_pid, signal.SIGKILL)
