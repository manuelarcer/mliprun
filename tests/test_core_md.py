"""Tests for mliprun.core.md."""
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen

from mliprun.core.md import setup_dynamics, run_md


class TestSetupDynamics:
    def _make_atoms(self):
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms.calc = EMT()
        return atoms

    def test_nve(self):
        atoms = self._make_atoms()
        dyn = setup_dynamics(atoms, ensemble="nve")
        assert isinstance(dyn, VelocityVerlet)

    def test_nvt_langevin(self):
        atoms = self._make_atoms()
        dyn = setup_dynamics(atoms, ensemble="nvt", thermostat="langevin", temperature=300)
        assert isinstance(dyn, Langevin)

    def test_nvt_berendsen(self):
        atoms = self._make_atoms()
        dyn = setup_dynamics(atoms, ensemble="nvt", thermostat="berendsen", temperature=300)
        assert isinstance(dyn, NVTBerendsen)

    def test_invalid_ensemble_raises(self):
        atoms = self._make_atoms()
        with pytest.raises(ValueError, match="Unknown ensemble"):
            setup_dynamics(atoms, ensemble="invalid")

    def test_invalid_thermostat_raises(self):
        atoms = self._make_atoms()
        with pytest.raises(ValueError, match="Unknown thermostat"):
            setup_dynamics(atoms, ensemble="nvt", thermostat="invalid")

    def test_invalid_barostat_raises(self):
        atoms = self._make_atoms()
        with pytest.raises(ValueError, match="Unknown barostat"):
            setup_dynamics(atoms, ensemble="npt", barostat="invalid")


class TestRunMd:
    def test_short_nve(self, tmp_workdir):
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms.calc = EMT()

        run_md(
            atoms, ensemble="nve", steps=5, log_interval=1, traj_interval=1,
            output_dir=tmp_workdir,
        )

        assert (tmp_workdir / "md.traj").exists()
        assert (tmp_workdir / "md_energy.csv").exists()
        # Plotting is opt-in: no PNGs unless plot=True.
        assert not (tmp_workdir / "md_energy.png").exists()
        assert not (tmp_workdir / "md_temperature.png").exists()

    def test_short_nve_plot_opt_in(self, tmp_workdir):
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms.calc = EMT()

        run_md(
            atoms, ensemble="nve", steps=5, log_interval=1, traj_interval=1,
            output_dir=tmp_workdir, plot=True,
        )

        assert (tmp_workdir / "md_energy.png").exists()
        assert (tmp_workdir / "md_temperature.png").exists()

    def test_short_nvt_langevin(self, tmp_workdir):
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms.calc = EMT()

        run_md(
            atoms, ensemble="nvt", thermostat="langevin",
            temperature=300, steps=5, log_interval=1, traj_interval=1,
            output_dir=tmp_workdir,
        )

        import pandas as pd
        df = pd.read_csv(tmp_workdir / "md_energy.csv")
        assert len(df) > 0
        assert "temperature(K)" in df.columns

    def test_resume_extends_trajectory(self, tmp_workdir):
        from ase.io import read, iread

        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms.calc = EMT()
        run_md(
            atoms, ensemble="nvt", thermostat="langevin",
            temperature=100, steps=10, log_interval=2, traj_interval=2,
            output_dir=tmp_workdir, friction=0.05,
        )
        import pandas as pd
        df1 = pd.read_csv(tmp_workdir / "md_energy.csv")
        last_step_1 = int(df1["step"].iloc[-1])
        n_frames_1 = sum(1 for _ in iread(str(tmp_workdir / "md.traj")))

        atoms2 = read(tmp_workdir / "md.traj", index=-1)
        atoms2.calc = EMT()
        run_md(
            atoms2, ensemble="nvt", thermostat="langevin",
            temperature=100, steps=10, log_interval=2, traj_interval=2,
            output_dir=tmp_workdir, friction=0.05, resume=True,
        )

        df2 = pd.read_csv(tmp_workdir / "md_energy.csv")
        n_frames_2 = sum(1 for _ in iread(str(tmp_workdir / "md.traj")))

        assert len(df2) > len(df1), "resume should extend the CSV"
        assert n_frames_2 > n_frames_1, "resume should append trajectory frames"
        assert int(df2["step"].iloc[-1]) > last_step_1, "step counter should advance"
        assert (df2["step"].diff().dropna() >= 0).all(), "step values must be monotonic"
        assert (df2["time(fs)"].diff().dropna() >= 0).all(), "time values must be monotonic"

    def test_resume_without_existing_files_raises(self, tmp_workdir):
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms.calc = EMT()
        with pytest.raises(FileNotFoundError, match="Cannot resume"):
            run_md(
                atoms, ensemble="nvt", steps=5, log_interval=1, traj_interval=1,
                output_dir=tmp_workdir, resume=True,
            )


class TestMDRecordsHead:
    def test_run_md_writes_the_uma_task_into_the_record(self, tmp_path):
        import json
        from ase.build import bulk
        from ase.calculators.emt import EMT
        from mliprun.core.md import run_md

        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms.calc = EMT()
        run_md(atoms, ensemble="nve", steps=2, log_interval=1,
               traj_interval=1, output_dir=tmp_path, model_name="uma-s-1p2",
               uma_task="oc25")

        data = json.loads((tmp_path / "mliprun_run.json").read_text())
        assert data["provenance"]["uma_task"] == "oc25"
        assert data["provenance"]["mace_head"] is None
        assert data["schema_version"] == 3
