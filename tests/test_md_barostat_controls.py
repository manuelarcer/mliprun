"""Tests for the barostat response controls: compressibility and pfactor.

Both numbers set how fast the cell answers a pressure difference. A wrong
unit there is invisible in the energies: the run finishes, the cell barely
moves, and nothing reports an error. The tests therefore assert on how far
the cell actually moves, and on the values written to md_params.txt and the
run record.
"""
import json
import re

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from typer.testing import CliRunner

from mliprun.cli.commands.md import app as md_app
from mliprun.core.md import (
    DEFAULT_COMPRESSIBILITY_PER_GPA,
    PFACTOR_GPA_FS2,
    default_pfactor,
    run_md,
    setup_dynamics,
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Strip rich's ANSI styling (CI colorizes usage errors, local runs do not)."""
    return _ANSI.sub("", output)


def _strained_cu():
    """2x2x2 cubic Cu at a = 3.9 A, far from EMT's 3.59 A minimum.

    The cell starts at -19.25 GPa (tension), so a working barostat has to
    contract it within a few steps.
    """
    atoms = bulk("Cu", "fcc", a=3.9, cubic=True) * (2, 2, 2)
    atoms.calc = EMT()
    return atoms


def _seed(atoms, temperature=300):
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,
                                 rng=np.random.default_rng(0))


# -- compressibility unit -------------------------------------------------

class TestCompressibilityDefault:
    def test_default_is_water_in_inverse_gpa(self):
        """4.57e-5 1/bar is 0.457 1/GPa; the old default kept the 1/bar
        number but the code converts it as 1/GPa."""
        assert DEFAULT_COMPRESSIBILITY_PER_GPA == pytest.approx(4.57e-5 * 1e4)

    def test_setup_dynamics_default_is_the_constant(self):
        atoms = _strained_cu()
        dyn = setup_dynamics(atoms, ensemble="npt", barostat="berendsen",
                             temperature=300)
        assert dyn.compressibility == pytest.approx(
            DEFAULT_COMPRESSIBILITY_PER_GPA / units.GPa)

    def test_default_berendsen_changes_the_volume_measurably(self):
        """The handoff's acceptance test.

        Same cell, default taup = 1000 fs, 20 steps. Measured
        dV/V = -0.132 with the corrected default and -1.7e-5 with the old
        4.57e-5, a factor of 1e4. The threshold sits between the two with
        an order of magnitude of margin on each side.
        """
        atoms = _strained_cu()
        _seed(atoms)
        v0 = atoms.get_volume()
        dyn = setup_dynamics(atoms, ensemble="npt", barostat="berendsen",
                             temperature=300, pressure=0.0, timestep=1.0,
                             set_velocities=False)
        dyn.run(20)
        rel = (atoms.get_volume() - v0) / v0
        assert rel < -1e-2, f"cell barely moved: dV/V = {rel:.3e}"

    def test_explicit_compressibility_scales_the_response(self):
        """Ten times smaller compressibility, roughly ten times smaller
        first-steps response: proves the value is used, not ignored."""
        responses = []
        for beta in (0.457, 0.0457):
            atoms = _strained_cu()
            _seed(atoms)
            v0 = atoms.get_volume()
            dyn = setup_dynamics(atoms, ensemble="npt", barostat="berendsen",
                                 temperature=300, pressure=0.0, timestep=1.0,
                                 compressibility=beta, set_velocities=False)
            dyn.run(2)
            responses.append((atoms.get_volume() - v0) / v0)
        ratio = responses[0] / responses[1]
        assert 8.0 < ratio < 12.0, responses

    @pytest.mark.parametrize("bad", [0.0, -0.457])
    def test_non_positive_compressibility_raises(self, bad):
        with pytest.raises(ValueError, match="compressibility"):
            setup_dynamics(_strained_cu(), ensemble="npt",
                           barostat="berendsen", temperature=300,
                           compressibility=bad)


# -- pfactor ----------------------------------------------------------------

class TestPfactor:
    def test_unit_constant(self):
        assert PFACTOR_GPA_FS2 == pytest.approx(units.GPa * units.fs ** 2)

    def test_default_pfactor_is_unchanged(self):
        """The auto value must stay bit-identical to the historical formula;
        changing it is a separate, scientific decision."""
        assert default_pfactor(25.0) == (25.0 * 75 * units.GPa) ** 2
        # 25 fs -> about 2.28e6 GPa fs^2, inside ASE's suggested range.
        assert default_pfactor(25.0) / PFACTOR_GPA_FS2 == pytest.approx(
            2.2742e6, rel=1e-4)

    def test_explicit_pfactor_reaches_npt(self):
        atoms = _strained_cu()
        dyn = setup_dynamics(atoms, ensemble="npt", barostat="npt",
                             temperature=300, pfactor=50.0)
        assert dyn.pfactor_given == 50.0

    def test_omitted_pfactor_uses_the_default(self):
        atoms = _strained_cu()
        dyn = setup_dynamics(atoms, ensemble="npt", barostat="npt",
                             temperature=300, ttime=25.0)
        assert dyn.pfactor_given == default_pfactor(25.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_pfactor_raises(self, bad):
        with pytest.raises(ValueError, match="pfactor"):
            setup_dynamics(_strained_cu(), ensemble="npt", barostat="npt",
                           temperature=300, pfactor=bad)


# -- run record -------------------------------------------------------------

class TestRecord:
    def _atoms(self):
        atoms = bulk("Cu", "fcc", a=3.7, cubic=True)
        atoms.calc = EMT()
        return atoms

    def test_berendsen_record_carries_compressibility(self, tmp_path):
        run_md(self._atoms(), ensemble="npt", barostat="berendsen",
               temperature=300, steps=2, log_interval=1, traj_interval=1,
               output_dir=tmp_path, compressibility=0.1)
        params = json.loads((tmp_path / "mliprun_run.json").read_text())["parameters"]
        assert params["compressibility_per_GPa"]["value"] == 0.1

    def test_npt_record_carries_the_resolved_pfactor(self, tmp_path):
        """An omitted pfactor is recorded as the value actually used."""
        run_md(self._atoms(), ensemble="npt", barostat="npt",
               temperature=300, steps=2, log_interval=1, traj_interval=1,
               output_dir=tmp_path, ttime=25.0)
        params = json.loads((tmp_path / "mliprun_run.json").read_text())["parameters"]
        assert params["pfactor_GPa_fs2"]["value"] == pytest.approx(
            default_pfactor(25.0) / PFACTOR_GPA_FS2)


# -- CLI ----------------------------------------------------------------------

def _structure(tmp_path):
    from ase.io import write
    path = tmp_path / "POSCAR"
    write(str(path), bulk("Cu", "fcc", a=3.6, cubic=True))
    return path


def _stub_run(tmp_path, monkeypatch, extra_args):
    structure = _structure(tmp_path)
    captured = {}
    monkeypatch.setattr("mliprun.cli.commands.md.setup_calculator",
                        lambda atoms, *a, **k: atoms)
    monkeypatch.setattr("mliprun.cli.commands.md.validate_mlip",
                        lambda *a, **k: None)
    monkeypatch.setattr("mliprun.cli.commands.md.run_md",
                        lambda **kwargs: captured.update(kwargs))
    result = CliRunner().invoke(md_app, [
        "--structure", str(structure), "--mlip", "mace-mp-0", "--steps", "1",
        "--temperature", "300",
    ] + extra_args)
    return result, captured, structure.parent / "md_params.txt"


class TestCLI:
    def test_compressibility_reaches_run_md_and_params(self, tmp_path, monkeypatch):
        result, captured, params = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "npt", "--barostat", "berendsen",
            "--compressibility", "0.0071",
        ])
        assert result.exit_code == 0, result.output
        assert captured["compressibility"] == 0.0071
        text = params.read_text()
        assert "Compressibility (1/GPa): 0.0071" in text
        assert "0.0071" in _plain(result.output)

    def test_default_compressibility_is_recorded(self, tmp_path, monkeypatch):
        result, captured, params = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "npt", "--barostat", "berendsen",
        ])
        assert result.exit_code == 0, result.output
        assert captured["compressibility"] == DEFAULT_COMPRESSIBILITY_PER_GPA
        assert "Compressibility (1/GPa): 0.457" in params.read_text()

    def test_pfactor_is_converted_from_gpa_fs2(self, tmp_path, monkeypatch):
        result, captured, params = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "npt", "--barostat", "npt", "--pfactor", "2e6",
        ])
        assert result.exit_code == 0, result.output
        assert captured["pfactor"] == pytest.approx(2e6 * PFACTOR_GPA_FS2)
        text = params.read_text()
        assert "pfactor (GPa fs^2):" in text
        assert "2e+06" in text or "2000000" in text

    def test_omitted_pfactor_records_the_auto_value(self, tmp_path, monkeypatch):
        result, captured, params = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "npt", "--barostat", "npt", "--ttime", "25",
        ])
        assert result.exit_code == 0, result.output
        assert captured["pfactor"] is None
        text = params.read_text()
        assert "pfactor (GPa fs^2):" in text
        assert "auto" in text
        assert "2.274e+06" in text

    @pytest.mark.parametrize("args, flag", [
        (["--ensemble", "npt", "--barostat", "npt", "--compressibility", "0.1"],
         "--compressibility"),
        (["--ensemble", "nvt", "--compressibility", "0.1"], "--compressibility"),
        (["--ensemble", "npt", "--barostat", "berendsen", "--pfactor", "1e6"],
         "--pfactor"),
        (["--ensemble", "nvt", "--pfactor", "1e6"], "--pfactor"),
    ])
    def test_a_flag_its_barostat_ignores_is_rejected(self, tmp_path, monkeypatch,
                                                      args, flag):
        """Silently ignoring it would let the user believe they tuned the run."""
        result, captured, _ = _stub_run(tmp_path, monkeypatch, args)
        assert result.exit_code == 2, result.output
        assert flag in _plain(result.output)
        assert captured == {}

    @pytest.mark.parametrize("args, flag", [
        (["--ensemble", "npt", "--barostat", "berendsen", "--compressibility", "0"],
         "--compressibility"),
        (["--ensemble", "npt", "--barostat", "npt", "--pfactor", "-5"],
         "--pfactor"),
    ])
    def test_non_positive_values_are_rejected(self, tmp_path, monkeypatch, args, flag):
        result, captured, _ = _stub_run(tmp_path, monkeypatch, args)
        assert result.exit_code == 2, result.output
        assert flag in _plain(result.output)
        assert captured == {}

    def test_nvt_params_file_omits_both(self, tmp_path, monkeypatch):
        result, _, params = _stub_run(tmp_path, monkeypatch, ["--ensemble", "nvt"])
        assert result.exit_code == 0, result.output
        text = params.read_text()
        assert "Compressibility" not in text
        assert "pfactor" not in text


class TestCLIProvenance:
    """Record keys differ from the CLI names (units in the key), so the
    source tag has to be carried over explicitly; asserted, not assumed."""

    def _run(self, tmp_path, monkeypatch, extra_args):
        structure = _structure(tmp_path)

        def _attach_emt(atoms, *a, **k):
            atoms.calc = EMT()
            return atoms

        monkeypatch.setattr("mliprun.cli.commands.md.setup_calculator", _attach_emt)
        monkeypatch.setattr("mliprun.cli.commands.md.validate_mlip",
                            lambda *a, **k: None)
        result = CliRunner().invoke(md_app, [
            "--structure", str(structure), "--mlip", "mace-mp-0",
            "--steps", "2", "--log-interval", "1", "--traj-interval", "1",
            "--ensemble", "npt", "--temperature", "300",
        ] + extra_args)
        data = json.loads((structure.parent / "mliprun_run.json").read_text())
        return result, data["parameters"]

    def test_explicit_compressibility_is_tagged_user(self, tmp_path, monkeypatch):
        result, params = self._run(tmp_path, monkeypatch, [
            "--barostat", "berendsen", "--compressibility", "0.2"])
        assert result.exit_code == 0, result.output
        assert params["compressibility_per_GPa"] == {"value": 0.2, "source": "user"}

    def test_explicit_pfactor_is_tagged_user(self, tmp_path, monkeypatch):
        result, params = self._run(tmp_path, monkeypatch, [
            "--barostat", "npt", "--pfactor", "1e6"])
        assert result.exit_code == 0, result.output
        entry = params["pfactor_GPa_fs2"]
        assert entry["value"] == pytest.approx(1e6)
        assert entry["source"] == "user"

    def test_omitted_pfactor_is_tagged_default(self, tmp_path, monkeypatch):
        result, params = self._run(tmp_path, monkeypatch, ["--barostat", "npt"])
        assert result.exit_code == 0, result.output
        assert params["pfactor_GPa_fs2"]["source"] == "default"
