"""The head/task must survive every CLI hop, not just validation.

These cover the wiring the other suites do not reach: the per-command echo
lines, the params-file entries, the restart display, the benchmark skip
note, and the calculator branches. All run with no MLIP installed -- the
calculator classes are patched.

Deliberately not named after any MLIP: `tests/conftest.py` auto-skips items
whose keywords include `uma`, `mace` or `sevenn`, and the module name lands
in `item.keywords`.
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from mliprun.cli.commands.autoneb import app as autoneb_app
from mliprun.cli.commands.benchmark import app as benchmark_app
from mliprun.cli.commands.md import app as md_app
from mliprun.cli.commands.neb import app as neb_app
from mliprun.cli.commands.optimize import app as optimize_app

runner = CliRunner()


def _poscar(tmp_path, name="POSCAR"):
    from ase.build import bulk
    from ase.io import write
    path = tmp_path / name
    write(str(path), bulk("Cu", "fcc", a=3.6))
    return path


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------

class TestOptimizeEchoesAndRecordsTheTask:
    def test_run_echoes_and_writes_the_task(self, tmp_path, monkeypatch):
        structure = _poscar(tmp_path)
        monkeypatch.setattr("mliprun.cli.commands.optimize.setup_calculator",
                            lambda atoms, *a, **k: atoms)
        monkeypatch.setattr("mliprun.cli.commands.optimize.run_optimization",
                            lambda **kw: True)
        with patch("mliprun.cli.utils.SEVENN_AVAILABLE", True):
            result = runner.invoke(optimize_app, [
                "run", "--structure", str(structure), "--mlip", "7net-omni",
                "--sevennet-task", "mpa"])

        assert result.exit_code == 0, result.output
        assert "SevenNet task: mpa" in result.output
        params = (tmp_path / "opt_params.txt").read_text()
        assert "SevenNet task:     mpa" in params
        assert "UMA task" not in params

    def test_batch_echoes_the_task(self, tmp_path, monkeypatch):
        sub = tmp_path / "s1"
        sub.mkdir()
        _poscar(sub, "POSCAR")
        monkeypatch.setattr("mliprun.cli.commands.optimize.build_calculator",
                            lambda *a, **k: MagicMock())
        monkeypatch.setattr("mliprun.cli.commands.optimize.run_optimization",
                            lambda **kw: True)
        with patch("mliprun.cli.utils.SEVENN_AVAILABLE", True):
            result = runner.invoke(optimize_app, [
                "batch", "--parent", str(tmp_path), "--input-name", "POSCAR",
                "--mlip", "7net-omni", "--sevennet-task", "oc20"])

        assert "SevenNet task: oc20" in result.output


# ---------------------------------------------------------------------------
# md
# ---------------------------------------------------------------------------

class TestMdValidatesTheAutoDetectedTag:
    def test_auto_detected_tag_is_validated(self, tmp_path):
        # md.py's auto branch must validate too; without it an auto-detected
        # multi-task tag reaches SevenNet with no modal.
        structure = _poscar(tmp_path)
        with patch("mliprun.cli.commands.md.detect_mlip",
                   return_value="7net-omni"), \
             patch("mliprun.cli.utils.SEVENN_AVAILABLE", True):
            result = runner.invoke(md_app, [
                "--structure", str(structure), "--mlip", "auto", "--steps", "1"])

        assert result.exit_code != 0
        assert "--sevennet-task is required" in result.output

    def test_auto_detected_tag_passes_with_a_task(self, tmp_path, monkeypatch):
        structure = _poscar(tmp_path)
        captured = {}
        monkeypatch.setattr("mliprun.cli.commands.md.setup_calculator",
                            lambda atoms, *a, **k: atoms)
        monkeypatch.setattr("mliprun.cli.commands.md.run_md",
                            lambda **kw: captured.update(kw))
        with patch("mliprun.cli.commands.md.detect_mlip",
                   return_value="7net-omni"), \
             patch("mliprun.cli.utils.SEVENN_AVAILABLE", True):
            result = runner.invoke(md_app, [
                "--structure", str(structure), "--mlip", "auto",
                "--sevennet-task", "mpa", "--steps", "1"])

        assert result.exit_code == 0, result.output
        assert captured["sevennet_task"] == "mpa"
        assert "SevenNet task: mpa" in result.output


# ---------------------------------------------------------------------------
# neb / autoneb
# ---------------------------------------------------------------------------

class TestNebEchoesTheTask:
    def test_new_run_echoes_the_task(self, tmp_path, monkeypatch):
        initial, final = _poscar(tmp_path, "i.vasp"), _poscar(tmp_path, "f.vasp")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("mliprun.cli.commands.neb.CustomNEB", MagicMock())
        with patch("mliprun.cli.utils.SEVENN_AVAILABLE", True):
            result = runner.invoke(neb_app, [
                "--initial", str(initial), "--final", str(final),
                "--mlip", "7net-omni", "--sevennet-task", "mpa"])

        assert "SevenNet task: mpa" in result.output
        params = (tmp_path / "neb_parameters.txt").read_text()
        assert "SevenNet task:" in params and "mpa" in params

    def test_restart_displays_the_stored_task(self, tmp_path, monkeypatch, capsys):
        # Driven through _handle_restart directly: the full CLI restart path
        # needs a real trajectory and a backup dance that add nothing here.
        from ase.build import bulk
        from ase.io import write as ase_write
        from mliprun.cli.commands.neb import _handle_restart

        ase_write(str(tmp_path / "A2B_full.traj"), [bulk("Cu", "fcc", a=3.6)] * 7)
        loaded = {
            "mlip": "7net-omni", "sevennet_task": "oc20", "num_images": 5,
            "total_images": 7, "fmax": 0.05, "k": 0.1, "climb": True,
            "dyneb": False, "scale_fmax": 0.0, "neb_optimizer": "fire",
            "neb_max_steps": 600, "log": "neb.log", "device": "cpu",
        }
        monkeypatch.setattr(
            "mliprun.cli.commands.neb.CustomNEB",
            MagicMock(load_from_restart=MagicMock(
                return_value=(MagicMock(), loaded))))

        _, params = _handle_restart(
            tmp_path, mlip=None, uma_task=None, mace_head=None,
            sevennet_task=None, fmax=None, log=None, k=None, climb=None,
            dyneb=None, scale_fmax=None, neb_optimizer=None,
            neb_max_steps=None, device=None)

        out = capsys.readouterr().out
        assert "SevenNet task:       oc20" in out
        # The stored task must survive into the resolved parameters, or the
        # restarted band runs on a different energy zero (CANON C3).
        assert params["sevennet_task"] == "oc20"
        assert "SevenNet task:" in (tmp_path / "neb_parameters.txt").read_text()


class TestAutonebEchoesTheTask:
    def test_echoes_and_records_the_task(self, tmp_path, monkeypatch):
        initial, final = _poscar(tmp_path, "i.vasp"), _poscar(tmp_path, "f.vasp")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("mliprun.cli.commands.autoneb.CustomNEB", MagicMock())
        with patch("mliprun.cli.utils.SEVENN_AVAILABLE", True):
            result = runner.invoke(autoneb_app, [
                "--initial", str(initial), "--final", str(final),
                "--mlip", "7net-omni", "--sevennet-task", "mpa"])

        assert "SevenNet task: mpa" in result.output
        params = (tmp_path / "autoneb_parameters.txt").read_text()
        assert "SevenNet task:" in params and "mpa" in params


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

class TestBenchmarkSkipNote:
    def test_prints_why_sevennet_was_skipped(self, tmp_path):
        structure = _poscar(tmp_path)
        with patch("mliprun.cli.commands.benchmark.SEVENN_AVAILABLE", True), \
             patch("mliprun.cli.commands.benchmark.FAIRCHEM_AVAILABLE", False), \
             patch("mliprun.cli.commands.benchmark.MACE_AVAILABLE", False), \
             patch("mliprun.cli.commands.benchmark.CHGNET_AVAILABLE", False):
            result = runner.invoke(benchmark_app, ["--structure", str(structure)])

        assert "SevenNet is installed but skipped" in result.output

    def test_forwards_the_task_to_the_calculator(self, tmp_path, monkeypatch):
        structure = _poscar(tmp_path)
        seen = {}

        def fake_setup(atoms, model, uma_task, **kwargs):
            seen["model"] = model
            seen["sevennet_task"] = kwargs.get("sevennet_task")
            atoms.calc = MagicMock(**{"get_potential_energy.return_value": -1.0})
            return atoms

        monkeypatch.setattr("mliprun.cli.commands.benchmark.setup_calculator",
                            fake_setup)
        with patch("mliprun.cli.commands.benchmark.SEVENN_AVAILABLE", True), \
             patch("mliprun.cli.commands.benchmark.FAIRCHEM_AVAILABLE", False), \
             patch("mliprun.cli.commands.benchmark.MACE_AVAILABLE", False), \
             patch("mliprun.cli.commands.benchmark.CHGNET_AVAILABLE", False):
            result = runner.invoke(benchmark_app, [
                "--structure", str(structure), "--sevennet-task", "mpa"])

        assert result.exit_code == 0, result.output
        assert seen["model"] == "7net-omni"
        assert seen["sevennet_task"] == "mpa"


# ---------------------------------------------------------------------------
# helpers and the CustomNEB calculator branch
# ---------------------------------------------------------------------------

class TestRecipeForUnknownSevenNetTag:
    def test_unknown_7net_tag_still_points_at_the_recipe(self):
        from mliprun.cli.utils import _recipe_for_tag
        assert _recipe_for_tag("7net-future-model") == "sevenn.md"
        assert _recipe_for_tag("7net-omni") == "sevenn.md#7net-omni"
        assert _recipe_for_tag("totally-unknown") == ""


class TestCustomNEBBuildsTheSevenNetCalculator:
    """CustomNEB wires its own calculators, so the 7net branch there is
    separate code from cli/utils.build_calculator and needs its own cover."""

    def _neb(self, task):
        from mliprun.core.neb import CustomNEB
        neb = CustomNEB.__new__(CustomNEB)
        neb.mlip, neb.uma_task, neb.mace_head = "7net-omni", "omat", "omat_pbe"
        neb.sevennet_task, neb.device = task, "cpu"
        return neb

    def _fake_sevenn(self, monkeypatch):
        fake_cls = MagicMock()
        module = types.ModuleType("sevenn.calculator")
        module.SevenNetCalculator = fake_cls
        pkg = types.ModuleType("sevenn")
        pkg.calculator = module
        monkeypatch.setitem(sys.modules, "sevenn", pkg)
        monkeypatch.setitem(sys.modules, "sevenn.calculator", module)
        return fake_cls

    def test_multi_task_model_passes_the_modal(self, monkeypatch):
        fake_cls = self._fake_sevenn(monkeypatch)
        self._neb("oc20").setup_calculator()
        fake_cls.assert_called_once_with("7net-omni", modal="oc20", device="cpu")

    def test_single_task_model_omits_the_modal(self, monkeypatch):
        fake_cls = self._fake_sevenn(monkeypatch)
        neb = self._neb(None)
        neb.mlip = "7net-0"
        neb.setup_calculator()
        fake_cls.assert_called_once_with("7net-0", device="cpu")
