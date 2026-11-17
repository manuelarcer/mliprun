"""``--relax-cell all|volume|shape``: what the cell relaxation optimizes.

`all` relaxes volume and shape (VASP ISIF=3), `volume` keeps the shape fixed
(ISIF=8-comparable), `shape` keeps the volume fixed (ISIF=4). A bare
``--relax-cell`` (and ``relax_cell=True`` in Python) means `all`. The physics
tests use ASE's EMT on a deliberately non-cubic, off-volume copper cell, so no
MLIP is needed.
"""
import json
import re

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from typer.testing import CliRunner

from mliprun.cli.commands import optimize as opt_cmd
from mliprun.core.optimize import resolve_cell_mode, run_optimization
from mliprun.core.run_record import RECORD_FILENAME

runner = CliRunner()


def _bad(value, name="--relax-cell"):
    return f"{name} must be one of all, volume, shape; got '{value}'."


def _strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _flat(text):
    return " ".join(_strip_ansi(text).split())


def _start():
    """Cubic Cu, then non-cubic and off-volume."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.set_cell(atoms.cell * [1.03, 1.03, 1.08], scale_atoms=True)
    atoms.calc = EMT()
    return atoms


def _shape(atoms):
    lengths = atoms.cell.lengths()
    return lengths[1] / lengths[0], lengths[2] / lengths[0], atoms.cell.angles()


def _stress(atoms):
    fresh = atoms.copy()
    fresh.calc = EMT()
    return fresh.get_stress(voigt=False)


def _run(atoms, tmp_path, **kw):
    return run_optimization(atoms, optimizer="bfgs", fmax=0.005,
                            max_steps=500, output_dir=tmp_path,
                            verbose=False, **kw)


def _report(label, v0, c0, atoms):
    s = _stress(atoms)
    pressure = -np.trace(s) / 3.0
    aniso = s[2, 2] - s[0, 0]
    print(f"[{label}] V {v0:.4f}->{atoms.get_volume():.4f} "
          f"c/a {c0:.5f}->{_shape(atoms)[1]:.5f} "
          f"P={pressure:.5f} szz-sxx={aniso:.5f}")
    return pressure, aniso


class TestResolveCellMode:
    @pytest.mark.parametrize("relax_cell, expected", [
        (False, None),
        (None, None),
        (True, "all"),
        ("all", "all"),
        ("volume", "volume"),
        ("shape", "shape"),
        (np.bool_(True), "all"),
        (np.bool_(False), None),
    ])
    def test_truth_table(self, relax_cell, expected):
        assert resolve_cell_mode(relax_cell) == expected

    @pytest.mark.parametrize("value", ["cubic", "", "Volume", "true"])
    def test_invalid_value_raises(self, value):
        with pytest.raises(ValueError) as exc:
            resolve_cell_mode(value)
        assert str(exc.value) == _bad(value, name="relax_cell")

    def test_the_message_names_the_parameter_as_asked(self):
        with pytest.raises(ValueError) as exc:
            resolve_cell_mode("cubic", name="--relax-cell")
        assert str(exc.value) == _bad("cubic")


class TestFillBareRelaxCell:
    @pytest.mark.parametrize("args, expected", [
        (["--structure", "P"], ["--structure", "P"]),
        (["--relax-cell"], ["--relax-cell", "all"]),
        (["--structure", "P", "--relax-cell"],
         ["--structure", "P", "--relax-cell", "all"]),
        (["--relax-cell", "--fmax", "0.01"],
         ["--relax-cell", "all", "--fmax", "0.01"]),
        (["--relax-cell", "volume", "--fmax", "0.01"],
         ["--relax-cell", "volume", "--fmax", "0.01"]),
        (["--relax-cell", "shape"], ["--relax-cell", "shape"]),
        (["--relax-cell=volume"], ["--relax-cell=volume"]),
        (["--relax-cell", "cubic"], ["--relax-cell", "cubic"]),
    ])
    def test_only_a_bare_flag_gets_a_value(self, args, expected):
        assert opt_cmd.fill_bare_relax_cell(args) == expected


class TestCoreCellModes:
    def test_volume_mode_keeps_the_shape(self, tmp_path):
        atoms = _start()
        v0 = atoms.get_volume()
        b0, c0, ang0 = _shape(atoms)
        _run(atoms, tmp_path, relax_cell="volume")
        b1, c1, ang1 = _shape(atoms)
        assert b1 == pytest.approx(b0, abs=1e-6)
        assert c1 == pytest.approx(c0, abs=1e-6)
        assert np.allclose(ang1, ang0, atol=1e-6)
        assert abs(atoms.get_volume() / v0 - 1.0) > 0.01
        pressure, aniso = _report("volume", v0, c0, atoms)
        assert abs(pressure) < 0.01
        assert abs(aniso) > 0.01

    def test_shape_mode_keeps_the_volume(self, tmp_path):
        atoms = _start()
        v0 = atoms.get_volume()
        _, c0, _ = _shape(atoms)
        _run(atoms, tmp_path, relax_cell="shape")
        _, c1, _ = _shape(atoms)
        assert atoms.get_volume() == pytest.approx(v0, rel=1e-6)
        assert abs(c1 / c0 - 1.0) > 0.01
        assert abs(c1 - 1.0) < abs(c0 - 1.0)
        pressure, aniso = _report("shape", v0, c0, atoms)
        assert abs(aniso) < 0.01
        assert abs(pressure) > 0.01

    def test_all_mode_equals_relax_cell_true(self, tmp_path):
        v0 = _start().get_volume()
        _, c0, _ = _shape(_start())
        a = _start()
        b = _start()
        _run(a, tmp_path / "a", relax_cell="all")
        _run(b, tmp_path / "b", relax_cell=True)
        assert np.allclose(a.cell.array, b.cell.array, atol=1e-6)
        assert abs(a.get_volume() / v0 - 1.0) > 0.01
        assert abs(_shape(a)[1] / c0 - 1.0) > 0.01
        _report("all", v0, c0, a)

    @pytest.mark.parametrize("value, in_record", [
        (True, None), ("all", None), ("volume", "volume"), ("shape", "shape")])
    def test_the_record_keeps_relax_cell_a_boolean(self, tmp_path, value,
                                                   in_record):
        _run(_start(), tmp_path, relax_cell=value)
        params = json.loads((tmp_path / RECORD_FILENAME).read_text())["parameters"]
        assert params["relax_cell"]["value"] is True
        if in_record is None:
            assert "cell_mode" not in params
        else:
            assert params["cell_mode"]["value"] == in_record

    def test_no_cell_relaxation_keeps_the_cell(self, tmp_path):
        atoms = _start()
        cell0 = atoms.cell.array.copy()
        _run(atoms, tmp_path)
        assert np.allclose(atoms.cell.array, cell0, atol=1e-12)

    def test_invalid_mode_raises_before_writing(self, tmp_path):
        with pytest.raises(ValueError) as exc:
            _run(_start(), tmp_path, relax_cell="cubic")
        assert str(exc.value) == _bad("cubic", name="relax_cell")
        assert not (tmp_path / RECORD_FILENAME).exists()


def _attach_emt(atoms, *args, **kwargs):
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def emt_cli(monkeypatch):
    monkeypatch.setattr(opt_cmd, "validate_mlip", lambda *a, **k: None)
    monkeypatch.setattr(opt_cmd, "setup_calculator", _attach_emt)


def _structure(tmp_path):
    path = tmp_path / "POSCAR"
    write(str(path), bulk("Cu", "fcc", a=3.7), format="vasp")
    return path


def _record(tmp_path):
    return json.loads((tmp_path / RECORD_FILENAME).read_text())


def _run_cli(tmp_path, *extra):
    return runner.invoke(opt_cmd.app, [
        "run", "--structure", str(_structure(tmp_path)), "--mlip", "emt-test",
        "--fmax", "0.1", "--max-steps", "30", *extra])


class TestCliCellMode:
    def test_volume_mode(self, tmp_path, emt_cli):
        r = _run_cli(tmp_path, "--relax-cell", "volume")
        assert r.exit_code == 0, r.output
        lines = (tmp_path / "opt_params.txt").read_text().splitlines()
        i = lines.index("Relax cell:        True")
        assert lines[i + 1] == "Cell mode:         volume"
        params = _record(tmp_path)["parameters"]
        assert params["cell_mode"]["value"] == "volume"
        assert params["relax_cell"]["value"] is True

    def test_bare_relax_cell_output_is_unchanged(self, tmp_path, emt_cli):
        r = _run_cli(tmp_path, "--relax-cell")
        assert r.exit_code == 0, r.output
        text = (tmp_path / "opt_params.txt").read_text()
        assert "Relax cell:        True" in text
        assert not any(l.startswith("Cell mode") for l in text.splitlines())
        params = _record(tmp_path)["parameters"]
        assert "cell_mode" not in params
        assert params["relax_cell"]["value"] is True

    def test_bare_relax_cell_before_another_option(self, tmp_path, emt_cli):
        r = runner.invoke(opt_cmd.app, [
            "run", "--relax-cell", "--structure", str(_structure(tmp_path)),
            "--mlip", "emt-test", "--fmax", "0.1", "--max-steps", "30"])
        assert r.exit_code == 0, r.output
        assert "Relax cell:        True" in (
            tmp_path / "opt_params.txt").read_text()

    def test_the_equals_form(self, tmp_path, emt_cli):
        r = _run_cli(tmp_path, "--relax-cell=shape")
        assert r.exit_code == 0, r.output
        assert "Cell mode:         shape" in (
            tmp_path / "opt_params.txt").read_text()

    def test_bare_relax_cell_through_the_mlip_entry_point(self, tmp_path,
                                                          emt_cli):
        from mliprun.cli.main import app as main_app
        r = runner.invoke(main_app, [
            "optimize", "run", "--structure", str(_structure(tmp_path)),
            "--mlip", "emt-test", "--fmax", "0.1", "--max-steps", "30",
            "--relax-cell"])
        assert r.exit_code == 0, r.output
        assert _record(tmp_path)["parameters"]["relax_cell"]["value"] is True

    def test_all_mode_writes_no_cell_mode_line(self, tmp_path, emt_cli):
        r = _run_cli(tmp_path, "--relax-cell", "all")
        assert r.exit_code == 0, r.output
        text = (tmp_path / "opt_params.txt").read_text()
        assert "Relax cell:        True" in text
        assert not any(l.startswith("Cell mode") for l in text.splitlines())
        assert "cell_mode" not in _record(tmp_path)["parameters"]

    def test_no_option_output_is_unchanged(self, tmp_path, emt_cli):
        r = _run_cli(tmp_path)
        assert r.exit_code == 0, r.output
        text = (tmp_path / "opt_params.txt").read_text()
        assert "Relax cell:        False" in text
        assert not any(l.startswith("Cell mode") for l in text.splitlines())
        assert "cell_mode" not in _record(tmp_path)["parameters"]

    def test_invalid_mode_exits_before_setup(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(opt_cmd, "validate_mlip", lambda *a, **k: None)
        monkeypatch.setattr(opt_cmd, "setup_calculator",
                            lambda *a, **k: seen.update(setup=True))
        r = _run_cli(tmp_path, "--relax-cell", "cubic")
        assert r.exit_code == 1
        assert "❌ " + _bad("cubic") in _flat(r.output)
        assert "setup" not in seen

    def test_batch_invalid_mode_exits_before_build(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(opt_cmd, "validate_mlip", lambda *a, **k: None)
        monkeypatch.setattr(opt_cmd, "build_calculator",
                            lambda *a, **k: seen.update(build=True))
        sub = tmp_path / "s1"
        sub.mkdir()
        _structure(sub)
        r = runner.invoke(opt_cmd.app, [
            "batch", "--parent", str(tmp_path), "--input-name", "POSCAR",
            "--mlip", "emt-test", "--relax-cell", "cubic"])
        assert r.exit_code == 1
        assert _bad("cubic") in _flat(r.output)
        assert "build" not in seen

    def test_batch_forwards_the_mode(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(opt_cmd, "validate_mlip", lambda *a, **k: None)
        monkeypatch.setattr(opt_cmd, "build_calculator",
                            lambda *a, **k: EMT())
        monkeypatch.setattr(opt_cmd, "run_optimization",
                            lambda **kw: seen.update(run=kw) or True)
        sub = tmp_path / "s1"
        sub.mkdir()
        _structure(sub)
        r = runner.invoke(opt_cmd.app, [
            "batch", "--parent", str(tmp_path), "--input-name", "POSCAR",
            "--mlip", "emt-test", "--relax-cell", "shape"])
        assert r.exit_code == 0, r.output
        assert seen["run"]["relax_cell"] == "shape"
        assert "cell_mode" not in seen["run"]
        params = (sub / "opt_params.txt").read_text()
        assert "Relax cell:        True" in params
        assert "Cell mode:         shape" in params


_AQ = "aqcat25-ev2-midfilm"


class TestAqcatRefusesCellMode:
    def _invoke(self, tmp_path, monkeypatch, command):
        from unittest.mock import patch
        seen = {}
        monkeypatch.setattr(opt_cmd, "setup_calculator",
                            lambda atoms, *a, **k: seen.update(setup=True))
        monkeypatch.setattr(opt_cmd, "build_calculator",
                            lambda *a, **k: seen.update(build=True))
        if command == "run":
            args = ["run", "--structure", str(_structure(tmp_path))]
        else:
            sub = tmp_path / "s1"
            sub.mkdir()
            _structure(sub)
            args = ["batch", "--parent", str(tmp_path), "--input-name",
                    "POSCAR"]
        with patch("mliprun.cli.utils.AQCAT_AVAILABLE", True):
            r = runner.invoke(opt_cmd.app, [
                *args, "--mlip", _AQ, "--aqcat-spin", "on",
                "--relax-cell", "volume"])
        return r, seen

    @pytest.mark.parametrize("command", ["run", "batch"])
    def test_refused_before_loading(self, tmp_path, monkeypatch, command):
        r, seen = self._invoke(tmp_path, monkeypatch, command)
        assert r.exit_code != 0
        assert "stress" in r.output
        assert seen == {}

    def test_core_refuses_before_writing(self, tmp_path):
        with pytest.raises(ValueError, match="stress"):
            run_optimization(_start(), output_dir=tmp_path,
                             model_name=_AQ, relax_cell="volume",
                             aqcat_spin="on")
        assert not (tmp_path / RECORD_FILENAME).exists()
