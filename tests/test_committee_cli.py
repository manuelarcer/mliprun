"""CLI surface for committee runs."""
import json
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
        assert record["schema_version"] == 4
        assert record["provenance"]["committee"]["config_sha256"] == \
            config.sha256
        uncertainty = record["stages"][0]["results"]["committee_uncertainty"]
        assert uncertainty["sigma_max_final_eV_per_A"] == pytest.approx(
            0.0, abs=1e-12)
        assert uncertainty["flagged"] is False

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
        assert "High committee disagreement" in result.output

        record = json.loads(
            (structure.parent / "mliprun_run.json").read_text())
        uncertainty = record["stages"][0]["results"]["committee_uncertainty"]
        assert uncertainty["flagged"] is True
        assert uncertainty["sigma_max_final_eV_per_A"] > 0.01


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
