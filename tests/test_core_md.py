"""Tests for mliprun.core.md."""
import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
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
        assert data["schema_version"] == 5


class TestBarostatMask:
    """`barostat_mask` restricts which cell axes the barostat may change.

    The motivating case is a slab-liquid interface: the in-plane lattice is
    fixed by the relaxed bulk and must not be strained, while the z length
    must be free so the liquid reaches its own density at the set pressure.
    """

    def _cubic_atoms(self, a=3.9):
        """A cubic Cu cell, deliberately strained away from EMT's minimum.

        EMT's fcc Cu minimum is near a = 3.59 A, so a = 3.9 leaves a large
        tensile stress and the barostat has something to do within a handful
        of steps. Cubic (not primitive) because ASE's MTK ``NPT`` requires an
        upper-triangular cell.
        """
        atoms = bulk("Cu", "fcc", a=a, cubic=True) * (2, 2, 2)
        atoms.calc = EMT()
        return atoms

    def _seeded_velocities(self, atoms, temperature=300):
        """Deterministic velocities, so the cell deltas are reproducible."""
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        rng = np.random.default_rng(0)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)

    # -- class selection -------------------------------------------------

    def test_default_mask_keeps_the_plain_berendsen_class(self):
        """The default must take today's code path, not a new one."""
        atoms = self._cubic_atoms()
        dyn = setup_dynamics(atoms, ensemble="npt", barostat="berendsen",
                             temperature=300)
        # `type(...) is` and not isinstance: Inhomogeneous_NPTBerendsen is a
        # subclass, so isinstance would pass for both classes and prove nothing.
        assert type(dyn) is NPTBerendsen

    def test_masked_berendsen_selects_the_inhomogeneous_class(self):
        atoms = self._cubic_atoms()
        dyn = setup_dynamics(atoms, ensemble="npt", barostat="berendsen",
                             temperature=300, barostat_mask=(0, 0, 1))
        assert isinstance(dyn, Inhomogeneous_NPTBerendsen)
        assert tuple(dyn.mask) == (0, 0, 1)

    def test_mtk_npt_carries_the_mask(self):
        atoms = self._cubic_atoms()
        dyn = setup_dynamics(atoms, ensemble="npt", barostat="npt",
                             temperature=300, barostat_mask=(0, 0, 1))
        assert isinstance(dyn, NPT)
        # ASE's NPT.set_mask stores the outer product of the 3-vector, so a
        # (0, 0, 1) mask leaves exactly the zz strain component free: no
        # in-plane strain and no xz/yz shear either.
        expected = np.outer([0, 0, 1], [0, 0, 1]).astype(bool)
        assert np.array_equal(np.asarray(dyn.mask, dtype=bool), expected)

    def test_mtk_npt_default_mask_leaves_every_component_free(self):
        atoms = self._cubic_atoms()
        dyn = setup_dynamics(atoms, ensemble="npt", barostat="npt",
                             temperature=300)
        assert np.asarray(dyn.mask, dtype=bool).all()

    # -- the test that matters: are the axes actually held? --------------

    def test_masked_berendsen_moves_z_only(self):
        atoms = self._cubic_atoms()
        self._seeded_velocities(atoms)
        cell_before = atoms.get_cell().array.copy()

        dyn = setup_dynamics(atoms, ensemble="npt", barostat="berendsen",
                             temperature=300, pressure=0.0, timestep=1.0,
                             taup=10.0, barostat_mask=(0, 0, 1),
                             set_velocities=False)
        dyn.run(10)

        cell_after = atoms.get_cell().array
        # In-plane rows are multiplied by exactly 1.0 when their mask entry is
        # zero, so this is an equality check with room only for set_cell's
        # round trip.
        assert np.allclose(cell_after[0], cell_before[0], atol=1e-12)
        assert np.allclose(cell_after[1], cell_before[1], atol=1e-12)
        # The strained cell is under tension at 0 GPa, so z must contract.
        # Measured dz = -2.261e-3 A over these 10 steps; the threshold keeps
        # a factor of ~2 of margin.
        dz = cell_after[2, 2] - cell_before[2, 2]
        assert dz < -1e-3, f"z length barely moved: dz = {dz:.3e} A"

    def test_unmasked_berendsen_moves_every_axis(self):
        """Control for the test above: without a mask, x and y move too.

        Without this, the masked test would also pass if the barostat were
        broken and no axis moved at all. Measured with the same setup:
        dx = dy = dz = -2.253e-3 A, i.e. isotropic.
        """
        atoms = self._cubic_atoms()
        self._seeded_velocities(atoms)
        cell_before = atoms.get_cell().array.copy()

        dyn = setup_dynamics(atoms, ensemble="npt", barostat="berendsen",
                             temperature=300, pressure=0.0, timestep=1.0,
                             taup=10.0, set_velocities=False)
        dyn.run(10)

        cell_after = atoms.get_cell().array
        assert abs(cell_after[0, 0] - cell_before[0, 0]) > 1e-3
        assert abs(cell_after[1, 1] - cell_before[1, 1]) > 1e-3

    def test_masked_mtk_npt_moves_z_only(self):
        atoms = self._cubic_atoms()
        self._seeded_velocities(atoms)
        cell_before = atoms.get_cell().array.copy()

        dyn = setup_dynamics(atoms, ensemble="npt", barostat="npt",
                             temperature=300, pressure=0.0, timestep=1.0,
                             ttime=25.0, barostat_mask=(0, 0, 1),
                             set_velocities=False)
        dyn.run(20)

        cell_after = atoms.get_cell().array
        assert np.allclose(cell_after[0], cell_before[0], atol=1e-10)
        assert np.allclose(cell_after[1], cell_before[1], atol=1e-10)
        # Measured dz = -1.302e-2 A over these 20 steps. Unmasked, the same
        # run also shears (max off-diagonal 2.8e-5 A); the outer-product mask
        # removes that, so the in-plane rows above are held whole, not just
        # their lengths.
        dz = cell_after[2, 2] - cell_before[2, 2]
        assert dz < -1e-3, f"z length barely moved: dz = {dz:.3e} A"

    # -- rejection paths -------------------------------------------------

    def test_non_default_mask_outside_npt_raises(self):
        atoms = self._cubic_atoms()
        with pytest.raises(ValueError, match="barostat_mask"):
            setup_dynamics(atoms, ensemble="nvt", temperature=300,
                           barostat_mask=(0, 0, 1))

    def test_default_mask_outside_npt_is_accepted(self):
        """The default must stay silent everywhere, or it is not a default."""
        atoms = self._cubic_atoms()
        dyn = setup_dynamics(atoms, ensemble="nve", barostat_mask=(1, 1, 1))
        assert isinstance(dyn, VelocityVerlet)

    @pytest.mark.parametrize("bad_mask", [5, None, 1.0])
    def test_non_sequence_mask_raises_value_error(self, bad_mask):
        """A scalar is the obvious Python-API slip; it must not surface as a
        bare TypeError from inside `list()`."""
        atoms = self._cubic_atoms()
        with pytest.raises(ValueError, match="barostat_mask"):
            setup_dynamics(atoms, ensemble="npt", barostat="berendsen",
                           temperature=300, barostat_mask=bad_mask)

    @pytest.mark.parametrize("bad_mask", [(1, 1), (1, 1, 1, 1), (0, 0, 2),
                                          (-1, 0, 1), ("a", "b", "c")])
    def test_malformed_mask_raises(self, bad_mask):
        atoms = self._cubic_atoms()
        with pytest.raises(ValueError, match="barostat_mask"):
            setup_dynamics(atoms, ensemble="npt", barostat="berendsen",
                           temperature=300, barostat_mask=bad_mask)


class TestBarostatMaskProvenance:
    """A masked run must be distinguishable from an isotropic one by reading
    the record alone, without opening the trajectory."""

    def _atoms(self):
        atoms = bulk("Cu", "fcc", a=3.7, cubic=True)
        atoms.calc = EMT()
        return atoms

    def test_run_md_records_the_mask(self, tmp_path):
        import json

        run_md(self._atoms(), ensemble="npt", barostat="berendsen",
               temperature=300, pressure=0.0, steps=2, log_interval=1,
               traj_interval=1, output_dir=tmp_path,
               barostat_mask=(0, 0, 1))

        data = json.loads((tmp_path / "mliprun_run.json").read_text())
        entry = data["parameters"]["barostat_mask"]
        assert entry["value"] == [0, 0, 1]
        # No RunContext was passed, so the source is unknown, not guessed.
        assert entry["source"] == "unspecified"

    def test_run_md_records_the_default_mask(self, tmp_path):
        import json

        run_md(self._atoms(), ensemble="npt", barostat="berendsen",
               temperature=300, pressure=0.0, steps=2, log_interval=1,
               traj_interval=1, output_dir=tmp_path)

        data = json.loads((tmp_path / "mliprun_run.json").read_text())
        assert data["parameters"]["barostat_mask"]["value"] == [1, 1, 1]
