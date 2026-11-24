"""Tests for NEB restart logic."""
import pytest
from pathlib import Path

from mliprun.core.neb import CustomNEB


class TestParseParametersFile:
    def test_parses_standard_file(self, tmp_path):
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            uma-s-1p1\n"
            "UMA task:              omat\n"
            "Initial:               initial.vasp\n"
            "Final:                 final.vasp\n"
            "Intermediate images:   5\n"
            "Total images:          7\n"
            "IDPP fmax:             0.1\n"
            "IDPP steps:            100\n"
            "Final fmax:            0.05\n"
            "Spring constant (k):   0.1\n"
            "Climb:                 True\n"
            "NEB optimizer:         fire\n"
            "NEB max steps:         600\n"
            "Optimize endpoints:    True\n"
            "Log file:              neb.log\n"
            "Output dir:            /tmp/test\n"
        )

        params = CustomNEB._parse_parameters_file(params_file)
        assert params["mlip"] == "uma-s-1p1"
        assert params["uma_task"] == "omat"
        assert params["num_images"] == 5
        assert params["total_images"] == 7
        assert params["fmax"] == 0.05
        assert params["k"] == 0.1
        assert params["climb"] is True
        assert params["neb_optimizer"] == "fire"
        assert params["neb_max_steps"] == 600

    def test_parses_none_values(self, tmp_path):
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            mace\n"
            "Intermediate images:   3\n"
            "Total images:          5\n"
            "Final fmax:            0.05\n"
            "Spring constant (k):   None\n"
            "NEB max steps:         None\n"
        )
        params = CustomNEB._parse_parameters_file(params_file)
        assert params["k"] is None
        assert params["neb_max_steps"] is None

    def test_parses_relax_atoms(self, tmp_path):
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            mace\n"
            "Intermediate images:   3\n"
            "Total images:          5\n"
            "Final fmax:            0.05\n"
            "Relax atoms:           [0, 1, 2]\n"
        )
        params = CustomNEB._parse_parameters_file(params_file)
        assert params["relax_atoms"] == [0, 1, 2]

    def test_missing_required_fields_raises(self, tmp_path):
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            mace\n"
        )
        with pytest.raises(ValueError, match="missing required fields"):
            CustomNEB._parse_parameters_file(params_file)


class TestLoadFromRestart:
    def test_missing_trajectory_raises(self, tmp_path):
        # Create only the params file, not the trajectory
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            mace\n"
            "Intermediate images:   3\n"
            "Total images:          5\n"
            "Final fmax:            0.05\n"
        )
        with pytest.raises(FileNotFoundError, match="A2B_full.traj"):
            CustomNEB.load_from_restart(output_dir=tmp_path)

    def test_missing_params_raises(self, tmp_path):
        # Create a dummy trajectory but no params file
        (tmp_path / "A2B_full.traj").touch()
        with pytest.raises(FileNotFoundError, match="neb_parameters.txt"):
            CustomNEB.load_from_restart(output_dir=tmp_path)


class TestCreateBackupFolder:
    def test_creates_backup(self, tmp_path):
        from mliprun.cli.commands.neb import create_backup_folder

        # Create some dummy files
        (tmp_path / "A2B.traj").touch()
        (tmp_path / "neb.log").touch()
        (tmp_path / "neb_parameters.txt").write_text("test")
        poscar_dir = tmp_path / "00"
        poscar_dir.mkdir()
        (poscar_dir / "POSCAR").touch()

        backup_dir, moved = create_backup_folder(tmp_path)

        assert backup_dir.exists()
        assert backup_dir.name.startswith("bkup_")
        assert "A2B.traj" in moved
        assert "neb.log" in moved
        assert "00/" in moved
        assert "neb_parameters.txt (copied)" in moved

        # Original neb_parameters.txt should still exist (copied, not moved)
        assert (tmp_path / "neb_parameters.txt").exists()
        # But A2B.traj should be moved
        assert not (tmp_path / "A2B.traj").exists()


class TestSevenNetTaskRoundTrip:
    """A restart that dropped the task would resume the band on a different
    energy zero -- the discontinuity CANON C3 forbids, and the reason the
    task has to survive neb_parameters.txt."""

    def _params_text(self, extra=""):
        return (
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            7net-omni\n"
            "SevenNet task:         oc20\n"
            "Initial:               initial.vasp\n"
            "Final:                 final.vasp\n"
            "Intermediate images:   5\n"
            "Total images:          7\n"
            "IDPP fmax:             0.1\n"
            "IDPP steps:            100\n"
            "Final fmax:            0.05\n"
            "Spring constant (k):   0.1\n"
            "Climb:                 True\n"
            + extra
        )

    def test_task_survives_the_params_round_trip(self, tmp_path):
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(self._params_text())
        params = CustomNEB._parse_parameters_file(params_file)
        assert params["mlip"] == "7net-omni"
        assert params["sevennet_task"] == "oc20"

    def test_uppercase_task_survives_verbatim(self, tmp_path):
        # 7net-mf-0's tasks are uppercase; the parser must not normalise.
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            7net-mf-0\n"
            "SevenNet task:         R2SCAN\n"
            "Intermediate images:   3\n"
            "Total images:          5\n"
            "Final fmax:            0.05\n"
        )
        params = CustomNEB._parse_parameters_file(params_file)
        assert params["sevennet_task"] == "R2SCAN"

    def test_absent_task_parses_as_missing_not_empty(self, tmp_path):
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            uma-s-1p2\n"
            "UMA task:              omat\n"
            "Intermediate images:   3\n"
            "Total images:          5\n"
            "Final fmax:            0.05\n"
        )
        params = CustomNEB._parse_parameters_file(params_file)
        assert params.get("sevennet_task") is None


class TestCustomNEBSevenNetCalculator:
    def test_multi_task_without_a_task_raises_a_clear_error(self):
        # CustomNEB wires its own calculators, so the guard has to exist here
        # too -- otherwise a direct API caller (or the stale library default
        # mlip="7net-mf-ompa") reaches SevenNet with no modal and fails deep
        # inside it instead of at the boundary.
        neb = CustomNEB.__new__(CustomNEB)
        neb.mlip = "7net-omni"
        neb.uma_task = "omat"
        neb.mace_head = "omat_pbe"
        neb.sevennet_task = None
        neb.device = "cpu"
        with pytest.raises(ValueError) as exc:
            neb.setup_calculator()
        assert "oc20" in str(exc.value)
        assert "7net-omni" in str(exc.value)


_AQCAT_TAG = "aqcat25-ev2-midfilm"


def _aqcat_params_text(spin="off", fidelity="low"):
    lines = [
        "NEB Run Parameters",
        "===================",
        f"MLIP model:            {_AQCAT_TAG}",
    ]
    if spin is not None:
        lines.append(f"AQCat spin:            {spin}")
    if fidelity is not None:
        lines.append(f"AQCat fidelity:        {fidelity}")
    lines += [
        "Device:                cpu",
        "Intermediate images:   1",
        "Total images:          3",
        "Final fmax:            0.05",
    ]
    return "\n".join(lines) + "\n"


class TestAqcatRoundTrip:
    """Spin on versus off moves the energy by up to several eV, so a restart
    that dropped the spin would resume the band on another energy surface."""

    def test_spin_and_fidelity_survive_the_params_file(self, tmp_path):
        path = tmp_path / "neb_parameters.txt"
        path.write_text(_aqcat_params_text())
        params = CustomNEB._parse_parameters_file(path)
        assert params["aqcat_spin"] == "off"
        assert params["aqcat_fidelity"] == "low"

    def test_absent_options_parse_as_missing(self, tmp_path):
        path = tmp_path / "neb_parameters.txt"
        path.write_text(_aqcat_params_text(spin=None, fidelity=None))
        params = CustomNEB._parse_parameters_file(path)
        assert params.get("aqcat_spin") is None
        assert params.get("aqcat_fidelity") is None

    def _restart_dir(self, tmp_path, **kwargs):
        from ase.build import bulk
        from ase.io import write
        (tmp_path / "neb_parameters.txt").write_text(_aqcat_params_text(**kwargs))
        write(str(tmp_path / "A2B_full.traj"), [bulk("Cu", "fcc", a=3.6)] * 3)
        return tmp_path

    def test_load_from_restart_keeps_the_stored_options(self, tmp_path):
        neb, params = CustomNEB.load_from_restart(self._restart_dir(tmp_path))
        assert neb.aqcat_spin == "off"
        assert neb.aqcat_fidelity == "low"
        assert params["aqcat_spin"] == "off"

    def test_load_from_restart_override_wins(self, tmp_path):
        neb, _ = CustomNEB.load_from_restart(
            self._restart_dir(tmp_path), aqcat_spin="on",
            aqcat_fidelity="high")
        assert neb.aqcat_spin == "on"
        assert neb.aqcat_fidelity == "high"


class TestCustomNEBAqcatCalculator:
    def _neb(self, spin, fidelity=None):
        neb = CustomNEB.__new__(CustomNEB)
        neb.mlip = _AQCAT_TAG
        neb.uma_task = neb.mace_head = neb.sevennet_task = None
        neb.aqcat_spin, neb.aqcat_fidelity = spin, fidelity
        neb.device = "cpu"
        return neb

    def test_goes_through_the_shared_builder(self, monkeypatch):
        from unittest.mock import MagicMock
        helper = MagicMock(return_value="calc")
        monkeypatch.setattr("mliprun.cli.utils._build_aqcat_calculator", helper)
        assert self._neb("on", "low").setup_calculator() == "calc"
        helper.assert_called_once_with(_AQCAT_TAG, "on", "low", "cpu")

    def test_builds_with_explicit_flags(self, monkeypatch):
        from unittest.mock import MagicMock
        fake = MagicMock()
        monkeypatch.setattr("mliprun.cli.utils._load_aqcat_calc",
                            lambda: fake)
        monkeypatch.setattr("mliprun.cli.utils._ensure_aqcat_checkpoint",
                            lambda tag: "/c.pt")
        self._neb("off").setup_calculator()
        fake.assert_called_once_with("/c.pt", is_spin_off=True,
                                     is_low_fi=False, cpu=True)

    def test_missing_spin_raises_before_loading(self, monkeypatch):
        from unittest.mock import MagicMock
        loader = MagicMock()
        monkeypatch.setattr("mliprun.cli.utils._load_aqcat_calc", loader)
        with pytest.raises(ValueError) as exc:
            self._neb(None).setup_calculator()
        assert "aqcat_spin" in str(exc.value)
        loader.assert_not_called()


class TestAqcatNebRecord:
    def _neb(self, tmp_path, monkeypatch):
        from ase.build import bulk
        from ase.calculators.emt import EMT
        initial = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        final = initial.copy()
        final.positions[0] += [0.2, 0.0, 0.0]
        neb = CustomNEB(initial, final, num_images=1, mlip=_AQCAT_TAG,
                        aqcat_spin="off", output_dir=tmp_path)
        # EMT stands in for the model: only the record is under test.
        monkeypatch.setattr(neb, "setup_calculator", lambda: EMT())
        return neb

    def test_run_neb_records_spin_and_resolved_fidelity(self, tmp_path,
                                                        monkeypatch):
        import json
        neb = self._neb(tmp_path, monkeypatch)
        neb.run_neb(max_steps=1)
        record = json.loads((tmp_path / "mliprun_run.json").read_text())
        assert record["provenance"]["aqcat_spin"] == "off"
        assert record["provenance"]["aqcat_fidelity"] == "high"
        assert record["parameters"]["aqcat_spin"]["value"] == "off"
        assert record["parameters"]["aqcat_fidelity"]["value"] == "high"

    def test_run_autoneb_refuses_the_tag(self, tmp_path, monkeypatch):
        neb = self._neb(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as exc:
            neb.run_autoneb()
        assert "singlepoint, optimize and neb only" in str(exc.value)
        assert not (tmp_path / "mliprun_run.json").exists()


# ---------------------------------------------------------------------------
# Second restart in the same folder
# ---------------------------------------------------------------------------
# A restart used to write "IDPP fmax: (from restart)" into neb_parameters.txt,
# and the next restart's parser ran float() on it and crashed.

_FIRST_RESTART_PARAMS = (
    "NEB Run Parameters (RESTART)\n"
    "=============================\n"
    "Restarted from:        previous_1\n"
    "MLIP model:            mace\n"
    "Initial:               (from restart)\n"
    "Final:                 (from restart)\n"
    "Intermediate images:   3\n"
    "Total images:          5\n"
    "IDPP fmax:             (from restart)\n"
    "IDPP steps:            (from restart)\n"
    "Final fmax:            0.05\n"
    "Optimize endpoints:    False (restart)\n"
)


class TestSecondRestart:
    def test_a_file_written_by_an_older_restart_parses(self, tmp_path):
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(_FIRST_RESTART_PARAMS)
        params = CustomNEB._parse_parameters_file(params_file)
        assert params["num_images"] == 3
        assert params["fmax"] == 0.05
        assert "interp_fmax" not in params
        assert "interp_steps" not in params

    def test_two_restarts_in_one_folder_run_and_keep_the_idpp_values(
            self, tmp_path, monkeypatch):
        from ase.build import add_adsorbate, fcc100
        from ase.calculators.emt import EMT
        from ase.constraints import FixAtoms
        from ase.io import write
        from typer.testing import CliRunner

        from mliprun.cli.commands.neb import app

        def slab(shift):
            s = fcc100("Al", size=(2, 2, 2), vacuum=5.0)
            add_adsorbate(s, "Al", 1.5, "hollow")
            s.positions[-1] += shift * s.cell[0]
            z = s.positions[:, 2]
            s.set_constraint(FixAtoms(mask=z < z.min() + 0.1))
            return s

        monkeypatch.chdir(tmp_path)
        write("initial.vasp", slab(0.0), format="vasp")
        write("final.vasp", slab(0.5), format="vasp")
        monkeypatch.setattr("mliprun.cli.commands.neb.resolve_mlip",
                            lambda mlip, *a, **k: "emt")
        monkeypatch.setattr(CustomNEB, "setup_calculator",
                            lambda self, *a, **k: EMT())
        runner = CliRunner()

        new = runner.invoke(app, ["--initial", "initial.vasp", "--final",
                                  "final.vasp", "--num-images", "3",
                                  "--interp-fmax", "0.2", "--interp-steps", "50",
                                  "--neb-max-steps", "5"])
        assert new.exit_code == 0, new.stdout
        for fmax in ("0.04", "0.03"):
            restarted = runner.invoke(app, ["--restart", "--fmax", fmax,
                                            "--neb-max-steps", "5"])
            assert restarted.exit_code == 0, restarted.stdout

        params = CustomNEB._parse_parameters_file(tmp_path / "neb_parameters.txt")
        assert params["fmax"] == 0.03
        assert params["interp_fmax"] == 0.2
        assert params["interp_steps"] == 50
