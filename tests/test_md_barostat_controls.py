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

    def test_default_pfactor_is_ptime_squared_times_b(self):
        """ASE's documented form: (75 fs)^2 * 100 GPa = 5.625e5 GPa fs^2."""
        assert default_pfactor() == pytest.approx(
            (75 * units.fs) ** 2 * 100 * units.GPa, rel=1e-12)
        assert default_pfactor() / PFACTOR_GPA_FS2 == pytest.approx(5.625e5)

    def test_default_npt_contracts_a_strained_cell_and_stays_bounded(self):
        """Measured on this cell (EMT minimum is dV/V = -0.22): dV/V reaches
        -0.11 by 50 steps and turns around near -0.33 by ~130 steps. The old
        default had only reached -0.03 at 50 steps."""
        atoms = _strained_cu()
        _seed(atoms)
        v0 = atoms.get_volume()
        dyn = setup_dynamics(atoms, ensemble="npt", barostat="npt",
                             temperature=300, pressure=0.0, timestep=1.0,
                             set_velocities=False)
        history = []
        for _ in range(20):
            dyn.run(10)
            history.append(atoms.get_volume() / v0 - 1)
        assert history[4] < -0.07, history
        assert -0.5 < min(history) and max(history) < 0.05, history

    def test_explicit_pfactor_reaches_npt(self):
        atoms = _strained_cu()
        dyn = setup_dynamics(atoms, ensemble="npt", barostat="npt",
                             temperature=300, pfactor=50.0)
        assert dyn.pfactor_given == 50.0

    def test_omitted_pfactor_uses_the_default(self):
        atoms = _strained_cu()
        dyn = setup_dynamics(atoms, ensemble="npt", barostat="npt",
                             temperature=300, ttime=25.0)
        assert dyn.pfactor_given == default_pfactor()

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
        assert params["pfactor_GPa_fs2"]["value"] == pytest.approx(5.625e5)


# -- logged pressure ---------------------------------------------------------

class TestLoggedPressure:
    """md_energy.csv must log the pressure the barostat acts on: kinetic
    term included, with the diagonal components a masked run controls."""

    def _run(self, tmp_path, **kw):
        import pandas as pd
        atoms = bulk("Cu", "fcc", a=3.7, cubic=True) * (2, 2, 2)
        atoms.calc = EMT()
        run_md(atoms, ensemble="npt", barostat="berendsen", temperature=300,
               steps=3, log_interval=1, traj_interval=1, output_dir=tmp_path,
               **kw)
        return pd.read_csv(tmp_path / "md_energy.csv"), atoms

    def test_columns(self, tmp_path):
        df, _ = self._run(tmp_path)
        assert list(df.columns[-5:]) == [
            "pressure(GPa)", "pressure_xx(GPa)", "pressure_yy(GPa)",
            "pressure_zz(GPa)", "volume(A^3)"]

    def test_mean_is_the_average_of_the_diagonal(self, tmp_path):
        df, _ = self._run(tmp_path)
        diag = df[["pressure_xx(GPa)", "pressure_yy(GPa)",
                   "pressure_zz(GPa)"]].mean(axis=1)
        assert np.allclose(df["pressure(GPa)"], diag, rtol=0, atol=1e-12)

    def test_last_row_includes_the_kinetic_term(self, tmp_path):
        """Recompute the final row from the final state: virial pressure +
        N k_B T / V. The ideal-gas term is the part the old log dropped."""
        df, atoms = self._run(tmp_path)
        virial = -np.trace(atoms.get_stress(voigt=False)) / 3
        ideal = len(atoms) * units.kB * atoms.get_temperature() / atoms.get_volume()
        assert ideal / units.GPa > 0.05  # large enough to be seen below
        # abs, not rel: N k_B T / V via get_temperature differs from ASE's
        # momentum-tensor ideal-gas term by ~2e-7 GPa (measured), far below
        # the >0.05 GPa term being tested.
        assert df["pressure(GPa)"].iloc[-1] == pytest.approx(
            (virial + ideal) / units.GPa, abs=1e-5)

    def test_record_carries_the_per_axis_means(self, tmp_path):
        df, _ = self._run(tmp_path)
        results = json.loads(
            (tmp_path / "mliprun_run.json").read_text())["stages"][0]["results"]
        for axis in "xyz":
            assert results[f"mean_pressure_{axis}{axis}_GPa"] == pytest.approx(
                df[f"pressure_{axis}{axis}(GPa)"].mean())

    def test_resume_from_a_csv_without_the_columns_is_refused(self, tmp_path):
        import pandas as pd
        self._run(tmp_path)
        csv = tmp_path / "md_energy.csv"
        pd.read_csv(csv).drop(columns=["pressure_zz(GPa)"]).to_csv(csv, index=False)
        atoms = bulk("Cu", "fcc", a=3.7, cubic=True) * (2, 2, 2)
        atoms.calc = EMT()
        with pytest.raises(ValueError, match="pressure_zz"):
            run_md(atoms, ensemble="npt", barostat="berendsen",
                   temperature=300, steps=1, output_dir=tmp_path, resume=True)

    def test_npt_plot_draws_the_components(self, tmp_path):
        self._run(tmp_path, plot=True)
        assert (tmp_path / "md_pressure.png").stat().st_size > 0


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
        assert "5.625e+05" in text

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
