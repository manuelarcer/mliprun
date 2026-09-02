"""CLI-level tests for the `run` subcommands, focused on opt-in plotting.

These drive `optimize run`, `md run`, and `neb run` through Typer's CliRunner
with the MLIP calculator mocked to EMT, so the full command body (including the
output-file listing) executes without a real model. The behavioural contract:
plotting is opt-in -- PNGs appear only with `--plot`; the CSV data always does.
"""
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from typer.testing import CliRunner

from mliprun.cli.commands import optimize as opt_cmd
from mliprun.cli.commands import md as md_cmd
from mliprun.cli.commands import neb as neb_cmd
from mliprun.core.neb import CustomNEB

runner = CliRunner()


def _attach_emt(atoms, *args, **kwargs):
    atoms.calc = EMT()
    return atoms


def _write_structure(path, atoms):
    write(str(path), atoms, format="vasp")


class TestOptimizeRunPlot:
    def _patch(self, monkeypatch):
        monkeypatch.setattr(opt_cmd, "validate_mlip", lambda *a, **k: None)
        monkeypatch.setattr(opt_cmd, "setup_calculator", _attach_emt)

    def test_no_png_by_default(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        s = tmp_path / "POSCAR"
        _write_structure(s, bulk("Cu", "fcc", a=3.7))
        r = runner.invoke(opt_cmd.app, [
            "run", "--structure", str(s), "--mlip", "mace",
            "--fmax", "0.1", "--max-steps", "30",
        ])
        assert r.exit_code == 0, r.output
        assert (tmp_path / "opt_convergence.csv").exists()
        assert not (tmp_path / "opt_convergence.png").exists()

    def test_png_with_plot(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        s = tmp_path / "POSCAR"
        _write_structure(s, bulk("Cu", "fcc", a=3.7))
        r = runner.invoke(opt_cmd.app, [
            "run", "--structure", str(s), "--mlip", "mace",
            "--fmax", "0.1", "--max-steps", "30", "--plot",
        ])
        assert r.exit_code == 0, r.output
        assert (tmp_path / "opt_convergence.png").exists()


class TestMdRunPlot:
    def _patch(self, monkeypatch):
        monkeypatch.setattr(md_cmd, "validate_mlip", lambda *a, **k: None)
        monkeypatch.setattr(md_cmd, "setup_calculator", _attach_emt)

    def test_no_png_by_default(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        s = tmp_path / "POSCAR"
        _write_structure(s, bulk("Cu", "fcc", a=3.6) * (2, 2, 2))
        r = runner.invoke(md_cmd.app, [
            "--structure", str(s), "--mlip", "mace",
            "--ensemble", "nve", "--steps", "3",
            "--log-interval", "1", "--traj-interval", "1",
        ])
        assert r.exit_code == 0, r.output
        assert (tmp_path / "md_energy.csv").exists()
        assert not (tmp_path / "md_energy.png").exists()

    def test_png_with_plot(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        s = tmp_path / "POSCAR"
        _write_structure(s, bulk("Cu", "fcc", a=3.6) * (2, 2, 2))
        r = runner.invoke(md_cmd.app, [
            "--structure", str(s), "--mlip", "mace",
            "--ensemble", "nve", "--steps", "3",
            "--log-interval", "1", "--traj-interval", "1", "--plot",
        ])
        assert r.exit_code == 0, r.output
        assert (tmp_path / "md_energy.png").exists()
        assert (tmp_path / "md_temperature.png").exists()

    def test_npt_png_with_plot(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        s = tmp_path / "POSCAR"
        _write_structure(s, bulk("Cu", "fcc", a=3.6) * (2, 2, 2))
        r = runner.invoke(md_cmd.app, [
            "--structure", str(s), "--mlip", "mace",
            "--ensemble", "npt", "--temperature", "300", "--pressure", "0.0",
            "--barostat", "berendsen", "--steps", "3",
            "--log-interval", "1", "--traj-interval", "1", "--plot",
        ])
        assert r.exit_code == 0, r.output
        assert (tmp_path / "md_pressure.png").exists()
        assert (tmp_path / "md_volume.png").exists()


class TestNebRunPlot:
    def _patch(self, monkeypatch):
        # NEB builds a calculator per image via CustomNEB.setup_calculator(),
        # and resolves the MLIP name up front; mock both so no model is needed.
        monkeypatch.setattr(CustomNEB, "setup_calculator", lambda self, *a, **k: EMT())
        monkeypatch.setattr(neb_cmd, "resolve_mlip", lambda m=None, *a, **k: "mace")

    def _pair(self, tmp_path):
        initial = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        final = initial.copy()
        pos = final.get_positions()
        pos[0] += np.array([0.3, 0.3, 0.0])
        final.set_positions(pos)
        _write_structure(tmp_path / "initial.vasp", initial)
        _write_structure(tmp_path / "final.vasp", final)
        return tmp_path / "initial.vasp", tmp_path / "final.vasp"

    def _args(self, ini, fin, extra):
        return [
            "--initial", str(ini), "--final", str(fin),
            "--mlip", "mace", "--num-images", "3", "--fmax", "0.5",
            "--neb-max-steps", "2", "--no-optimize-endpoints",
        ] + extra

    def test_no_png_by_default(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        monkeypatch.chdir(tmp_path)  # neb writes to CWD
        ini, fin = self._pair(tmp_path)
        r = runner.invoke(neb_cmd.app, self._args(ini, fin, []))
        assert r.exit_code == 0, r.output
        assert (tmp_path / "neb_convergence.csv").exists()
        assert not (tmp_path / "neb_convergence.png").exists()
        assert not (tmp_path / "neb_energy.png").exists()

    def test_png_with_plot(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        monkeypatch.chdir(tmp_path)
        ini, fin = self._pair(tmp_path)
        r = runner.invoke(neb_cmd.app, self._args(ini, fin, ["--plot"]))
        assert r.exit_code == 0, r.output
        assert (tmp_path / "neb_convergence.png").exists()
        assert (tmp_path / "neb_energy.png").exists()
