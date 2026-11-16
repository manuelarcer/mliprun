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


class TestCustomNEBGuardsTheHead:
    """CustomNEB wires its own calculators, so the C1 guards live here as
    well as in build_calculator. These are the branches that stop a NEB band
    from being built on a head nobody chose."""

    def _neb(self, model, **attrs):
        from mliprun.core.neb import CustomNEB
        neb = CustomNEB.__new__(CustomNEB)
        neb.mlip = model
        neb.uma_task = attrs.get("uma_task")
        neb.mace_head = attrs.get("mace_head")
        neb.sevennet_task = attrs.get("sevennet_task")
        neb.device = "cpu"
        return neb

    def test_uma_without_a_task_raises(self):
        with pytest.raises(ValueError) as exc:
            self._neb("uma-s-1p2").setup_calculator()
        assert "uma_task" in str(exc.value)
        assert "--uma-task" in str(exc.value)

    def test_mace_mh_without_a_head_raises(self):
        with pytest.raises(ValueError) as exc:
            self._neb("mace-mh-1").setup_calculator()
        assert "mace_head" in str(exc.value)
        assert "--mace-head" in str(exc.value)


class TestCoreRecordsTheAqcatOptions:
    """EMT stands in for the model; only the record and the guards are under
    test."""

    def _atoms(self):
        from ase.build import bulk
        from ase.calculators.emt import EMT
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        atoms.rattle(0.01, seed=1)
        atoms.calc = EMT()
        return atoms

    def _record(self, path):
        import json
        return json.loads((path / "mliprun_run.json").read_text())

    def test_run_optimization_records_spin_and_fidelity(self, tmp_path):
        from mliprun.core.optimize import run_optimization
        run_optimization(self._atoms(), max_steps=1, output_dir=tmp_path,
                         model_name="aqcat25-ev2-midfilm", verbose=False,
                         aqcat_spin="on", aqcat_fidelity="low")
        prov = self._record(tmp_path)["provenance"]
        assert prov["aqcat_spin"] == "on"
        assert prov["aqcat_fidelity"] == "low"

    def test_run_optimization_refuses_relax_cell_before_writing(self, tmp_path):
        from mliprun.core.optimize import run_optimization
        with pytest.raises(ValueError) as exc:
            run_optimization(self._atoms(), output_dir=tmp_path,
                             model_name="aqcat25-ev2-midfilm",
                             relax_cell=True, aqcat_spin="on")
        assert "stress" in str(exc.value)
        assert not (tmp_path / "mliprun_run.json").exists()

    def test_run_optimization_refuses_gpmin(self, tmp_path):
        from mliprun.core.optimize import run_optimization
        with pytest.raises(ValueError) as exc:
            run_optimization(self._atoms(), optimizer="gpmin",
                             output_dir=tmp_path,
                             model_name="aqcat25-ev2-infilm", aqcat_spin="on")
        assert "gradient" in str(exc.value)

    def test_run_singlepoint_records_spin_and_fidelity(self, tmp_path):
        from mliprun.core.singlepoint import run_singlepoint
        run_singlepoint(self._atoms(), output_dir=tmp_path,
                        model_name="aqcat25-ev2-infilm", aqcat_spin="off")
        prov = self._record(tmp_path)["provenance"]
        assert prov["aqcat_spin"] == "off"
        assert prov["aqcat_fidelity"] == "high"


# ---------------------------------------------------------------------------
# AQCat25-EV2 CLI wiring
# ---------------------------------------------------------------------------

_AQ = "aqcat25-ev2-midfilm"


class TestOptimizeRunWithAqcat:
    def _invoke(self, tmp_path, monkeypatch, *extra):
        structure = _poscar(tmp_path)
        seen = {}

        def fake_setup(atoms, *a, **k):
            seen["setup"] = k
            return atoms

        monkeypatch.setattr("mliprun.cli.commands.optimize.setup_calculator",
                            fake_setup)
        monkeypatch.setattr("mliprun.cli.commands.optimize.run_optimization",
                            lambda **kw: seen.update(run=kw) or True)
        with patch("mliprun.cli.utils.AQCAT_AVAILABLE", True):
            result = runner.invoke(optimize_app, [
                "run", "--structure", str(structure), "--mlip", _AQ, *extra])
        return result, seen

    def test_echoes_forwards_and_writes_the_options(self, tmp_path, monkeypatch):
        result, seen = self._invoke(tmp_path, monkeypatch, "--aqcat-spin", "off")
        assert result.exit_code == 0, result.output
        assert "AQCat spin: off" in result.output
        assert "AQCat fidelity: high" in result.output
        assert "adsorption energy" in result.output
        assert seen["setup"]["aqcat_spin"] == "off"
        assert seen["setup"]["aqcat_fidelity"] is None
        assert seen["run"]["aqcat_spin"] == "off"
        params = (tmp_path / "opt_params.txt").read_text()
        assert "AQCat spin:        off" in params
        assert "AQCat fidelity:    high" in params
        assert "SevenNet task" not in params

    def test_missing_spin_stops_the_run(self, tmp_path, monkeypatch):
        result, seen = self._invoke(tmp_path, monkeypatch)
        assert result.exit_code != 0
        assert "--aqcat-spin" in result.output
        assert "setup" not in seen

    def test_relax_cell_is_refused_before_loading(self, tmp_path, monkeypatch):
        result, seen = self._invoke(tmp_path, monkeypatch, "--aqcat-spin",
                                    "on", "--relax-cell")
        assert result.exit_code != 0
        assert "stress" in result.output
        assert "setup" not in seen

    def test_line_search_optimizer_is_refused(self, tmp_path, monkeypatch):
        result, seen = self._invoke(tmp_path, monkeypatch, "--aqcat-spin",
                                    "on", "--optimizer", "bfgsls")
        assert result.exit_code != 0
        assert "gradient" in result.output
        assert "setup" not in seen

    def test_spin_next_to_a_committee_is_rejected(self, tmp_path,
                                                  fake_committee_file):
        path, _ = fake_committee_file
        structure = _poscar(tmp_path)
        result = runner.invoke(optimize_app, [
            "run", "--structure", str(structure), "--committee", str(path),
            "--aqcat-spin", "on"])
        assert result.exit_code != 0
        assert "--aqcat-spin" in result.output


class TestOptimizeBatchWithAqcat:
    def _invoke(self, tmp_path, monkeypatch, *extra):
        sub = tmp_path / "s1"
        sub.mkdir()
        _poscar(sub, "POSCAR")
        seen = {}

        def fake_build(*a, **k):
            seen["build"] = k
            return MagicMock(**{"get_potential_energy.return_value": -1.0})

        monkeypatch.setattr("mliprun.cli.commands.optimize.build_calculator",
                            fake_build)
        monkeypatch.setattr("mliprun.cli.commands.optimize.run_optimization",
                            lambda **kw: seen.update(run=kw) or True)
        with patch("mliprun.cli.utils.AQCAT_AVAILABLE", True):
            result = runner.invoke(optimize_app, [
                "batch", "--parent", str(tmp_path), "--input-name", "POSCAR",
                "--mlip", _AQ, *extra])
        return result, seen

    def test_echoes_and_forwards_the_options(self, tmp_path, monkeypatch):
        result, seen = self._invoke(tmp_path, monkeypatch, "--aqcat-spin",
                                    "on", "--aqcat-fidelity", "low")
        assert result.exit_code == 0, result.output
        assert "AQCat spin: on" in result.output
        assert "AQCat fidelity: low" in result.output
        assert seen["build"]["aqcat_spin"] == "on"
        assert seen["build"]["aqcat_fidelity"] == "low"
        assert seen["run"]["aqcat_fidelity"] == "low"
        params = (tmp_path / "s1" / "opt_params.txt").read_text()
        assert "AQCat fidelity:    low" in params

    def test_relax_cell_is_refused(self, tmp_path, monkeypatch):
        result, seen = self._invoke(tmp_path, monkeypatch, "--aqcat-spin",
                                    "on", "--relax-cell")
        assert result.exit_code != 0
        assert "stress" in result.output
        assert "build" not in seen

    def test_gpmin_is_refused(self, tmp_path, monkeypatch):
        result, seen = self._invoke(tmp_path, monkeypatch, "--aqcat-spin",
                                    "on", "--optimizer", "gpmin")
        assert result.exit_code != 0
        assert "gradient" in result.output
        assert "build" not in seen


class TestSinglepointWithAqcat:
    def test_echoes_and_forwards_the_options(self, tmp_path, monkeypatch):
        from mliprun.cli.commands.singlepoint import app as sp_app
        structure = _poscar(tmp_path)
        seen = {}

        def fake_setup(atoms, *a, **k):
            seen["setup"] = k
            return atoms

        def fake_run(**kw):
            seen["run"] = kw
            return {"energy_eV": -1.0, "fmax_free_eV_per_A": 0.0,
                    "fmax_all_eV_per_A": 0.0, "n_free_atoms": 1,
                    "worst_force_atom_free": 0,
                    "worst_force_atom_free_symbol": "Cu",
                    "worst_force_atom_all": 0,
                    "worst_force_atom_all_symbol": "Cu",
                    "stress_GPa": None, "stress_unavailable_reason": "none",
                    "unhandled_constraints": []}

        monkeypatch.setattr("mliprun.cli.commands.singlepoint.setup_calculator",
                            fake_setup)
        monkeypatch.setattr("mliprun.cli.commands.singlepoint.run_singlepoint",
                            fake_run)
        with patch("mliprun.cli.utils.AQCAT_AVAILABLE", True):
            result = runner.invoke(sp_app, [
                "run", "--structure", str(structure), "--mlip", _AQ,
                "--aqcat-spin", "on", "--aqcat-fidelity", "low"])

        assert result.exit_code == 0, result.output
        assert "AQCat spin: on" in result.output
        assert "AQCat fidelity: low" in result.output
        assert "direct model output" in result.output
        assert seen["setup"]["aqcat_spin"] == "on"
        assert seen["setup"]["aqcat_fidelity"] == "low"
        assert seen["run"]["aqcat_spin"] == "on"
        assert seen["run"]["aqcat_fidelity"] == "low"

    def test_missing_spin_stops_the_run(self, tmp_path):
        from mliprun.cli.commands.singlepoint import app as sp_app
        structure = _poscar(tmp_path)
        with patch("mliprun.cli.utils.AQCAT_AVAILABLE", True):
            result = runner.invoke(sp_app, [
                "run", "--structure", str(structure), "--mlip", _AQ])
        assert result.exit_code != 0
        assert "--aqcat-spin" in result.output


class TestNebWithAqcat:
    def test_new_run_echoes_writes_and_forwards(self, tmp_path, monkeypatch):
        initial, final = _poscar(tmp_path, "i.vasp"), _poscar(tmp_path, "f.vasp")
        monkeypatch.chdir(tmp_path)
        fake_neb = MagicMock()
        monkeypatch.setattr("mliprun.cli.commands.neb.CustomNEB", fake_neb)
        with patch("mliprun.cli.utils.AQCAT_AVAILABLE", True):
            result = runner.invoke(neb_app, [
                "--initial", str(initial), "--final", str(final),
                "--mlip", _AQ, "--aqcat-spin", "off",
                "--no-optimize-endpoints"])

        assert result.exit_code == 0, result.output
        assert "AQCat spin: off" in result.output
        assert "adsorption energy" in result.output
        kwargs = fake_neb.call_args.kwargs
        assert kwargs["aqcat_spin"] == "off"
        assert kwargs["aqcat_fidelity"] is None
        params = (tmp_path / "neb_parameters.txt").read_text()
        assert "AQCat spin:" in params and "off" in params
        assert "AQCat fidelity:" in params and "high" in params

    def test_new_run_without_spin_stops(self, tmp_path, monkeypatch):
        initial, final = _poscar(tmp_path, "i.vasp"), _poscar(tmp_path, "f.vasp")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("mliprun.cli.commands.neb.CustomNEB", MagicMock())
        with patch("mliprun.cli.utils.AQCAT_AVAILABLE", True):
            result = runner.invoke(neb_app, [
                "--initial", str(initial), "--final", str(final),
                "--mlip", _AQ])
        assert result.exit_code != 0
        assert "--aqcat-spin" in result.output

    def _restart(self, tmp_path, monkeypatch, loaded, **overrides):
        from ase.build import bulk
        from ase.io import write as ase_write
        from mliprun.cli.commands.neb import _handle_restart

        ase_write(str(tmp_path / "A2B_full.traj"), [bulk("Cu", "fcc", a=3.6)] * 3)
        monkeypatch.setattr(
            "mliprun.cli.commands.neb.CustomNEB",
            MagicMock(load_from_restart=MagicMock(
                return_value=(MagicMock(), loaded))))
        kwargs = dict(mlip=None, uma_task=None, mace_head=None,
                      sevennet_task=None, fmax=None, log=None, k=None,
                      climb=None, dyneb=None, scale_fmax=None,
                      neb_optimizer=None, neb_max_steps=None, device=None)
        kwargs.update(overrides)
        return _handle_restart(tmp_path, **kwargs)

    def _loaded(self):
        return {
            "mlip": _AQ, "aqcat_spin": "off", "aqcat_fidelity": "low",
            "num_images": 1, "total_images": 3, "fmax": 0.05, "k": 0.1,
            "climb": True, "dyneb": False, "scale_fmax": 0.0,
            "neb_optimizer": "fire", "neb_max_steps": 600, "log": "neb.log",
            "device": "cpu",
        }

    def test_restart_keeps_and_displays_the_stored_options(
            self, tmp_path, monkeypatch, capsys):
        _, params = self._restart(tmp_path, monkeypatch, self._loaded())
        out = capsys.readouterr().out
        assert "AQCat spin:          off" in out
        assert "AQCat fidelity:      low" in out
        assert "adsorption energy" in out
        assert params["aqcat_spin"] == "off"
        assert params["aqcat_fidelity"] == "low"
        text = (tmp_path / "neb_parameters.txt").read_text()
        assert "AQCat spin:" in text and "AQCat fidelity:" in text

    def test_restart_override_wins_over_the_stored_value(
            self, tmp_path, monkeypatch):
        _, params = self._restart(tmp_path, monkeypatch, self._loaded(),
                                  aqcat_spin="on", aqcat_fidelity="high")
        assert params["aqcat_spin"] == "on"
        assert params["aqcat_fidelity"] == "high"

    def test_restart_cli_forwards_the_options(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        seen = {}

        def fake_restart(output_dir, **kwargs):
            seen.update(kwargs)
            return MagicMock(), {"total_images": 3, "neb_optimizer": "fire"}

        monkeypatch.setattr("mliprun.cli.commands.neb._handle_restart",
                            fake_restart)
        result = runner.invoke(neb_app, ["--restart", "--aqcat-spin", "on",
                                         "--aqcat-fidelity", "low"])
        assert result.exit_code == 0, result.output
        assert seen["aqcat_spin"] == "on"
        assert seen["aqcat_fidelity"] == "low"


class TestOutOfScopeCommandsRejectAqcat:
    _MSG = "AQCat25-EV2 models are supported in singlepoint, optimize and neb only"

    def test_md(self, tmp_path):
        result = runner.invoke(md_app, [
            "--structure", str(_poscar(tmp_path)), "--mlip", _AQ,
            "--steps", "1"])
        assert result.exit_code != 0
        assert self._MSG in result.output

    def test_freq(self, tmp_path):
        from mliprun.cli.commands.freq import app as freq_app
        result = runner.invoke(freq_app, [
            "run", "--structure", str(_poscar(tmp_path)), "--mlip", _AQ])
        assert result.exit_code != 0
        assert self._MSG in result.output

    def test_autoneb(self, tmp_path, monkeypatch):
        initial, final = _poscar(tmp_path, "i.vasp"), _poscar(tmp_path, "f.vasp")
        monkeypatch.chdir(tmp_path)
        neb_cls = MagicMock()
        monkeypatch.setattr("mliprun.cli.commands.autoneb.CustomNEB", neb_cls)
        result = runner.invoke(autoneb_app, [
            "--initial", str(initial), "--final", str(final), "--mlip", _AQ])
        assert result.exit_code != 0
        assert self._MSG in result.output
        neb_cls.assert_not_called()

    def test_benchmark_explicit_models(self, tmp_path, monkeypatch):
        setup = MagicMock()
        monkeypatch.setattr("mliprun.cli.commands.benchmark.setup_calculator",
                            setup)
        result = runner.invoke(benchmark_app, [
            "--structure", str(_poscar(tmp_path)), "--models", f"mace,{_AQ}"])
        assert result.exit_code != 0
        assert self._MSG in result.output
        setup.assert_not_called()

    def test_benchmark_never_lists_the_tags(self):
        from mliprun.cli.commands import benchmark
        with patch("mliprun.cli.utils.AQCAT_AVAILABLE", True), \
             patch.object(benchmark, "FAIRCHEM_AVAILABLE", True), \
             patch.object(benchmark, "SEVENN_AVAILABLE", True), \
             patch.object(benchmark, "MACE_AVAILABLE", True), \
             patch.object(benchmark, "CHGNET_AVAILABLE", True):
            models = benchmark._available_models("mpa", "omat")
        assert not any(m.startswith("aqcat25-") for m in models)


class TestBenchmarkSkipsUmaWithoutATask:
    def test_prints_why_uma_was_skipped(self, tmp_path):
        structure = _poscar(tmp_path)
        with patch("mliprun.cli.commands.benchmark.FAIRCHEM_AVAILABLE", True), \
             patch("mliprun.cli.commands.benchmark.SEVENN_AVAILABLE", False), \
             patch("mliprun.cli.commands.benchmark.MACE_AVAILABLE", False), \
             patch("mliprun.cli.commands.benchmark.CHGNET_AVAILABLE", False):
            result = runner.invoke(benchmark_app, ["--structure", str(structure)])

        assert "UMA is installed but skipped" in result.output
