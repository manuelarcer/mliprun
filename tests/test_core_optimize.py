"""Tests for mliprun.core.optimize."""
import pytest
import numpy as np
from pathlib import Path

from ase.build import bulk
from ase.calculators.emt import EMT

from mliprun.core.optimize import run_optimization, OPTIMIZER_MAP


class TestOptimizationConverges:
    def test_cu_bulk_converges(self, tmp_workdir):
        atoms = bulk("Cu", "fcc", a=3.7)  # Slightly off equilibrium
        atoms.calc = EMT()
        converged = run_optimization(
            atoms, optimizer="bfgs", fmax=0.05, max_steps=200,
            output_dir=tmp_workdir, verbose=False,
        )
        assert converged

    def test_optimization_outputs_created(self, tmp_workdir):
        atoms = bulk("Cu", "fcc", a=3.7)
        atoms.calc = EMT()
        run_optimization(
            atoms, optimizer="bfgs", fmax=0.1, max_steps=50,
            output_dir=tmp_workdir, verbose=False,
        )
        assert (tmp_workdir / "opt.traj").exists()
        assert (tmp_workdir / "opt.log").exists()
        assert (tmp_workdir / "opt_convergence.csv").exists()
        assert (tmp_workdir / "opt_final.vasp").exists()
        assert (tmp_workdir / "CONTCAR").exists()
        # Plotting is opt-in: no PNG unless plot=True.
        assert not (tmp_workdir / "opt_convergence.png").exists()

    def test_plot_opt_in_writes_png(self, tmp_workdir):
        atoms = bulk("Cu", "fcc", a=3.7)
        atoms.calc = EMT()
        run_optimization(
            atoms, optimizer="bfgs", fmax=0.1, max_steps=50,
            output_dir=tmp_workdir, verbose=False, plot=True,
        )
        assert (tmp_workdir / "opt_convergence.png").exists()

    def test_convergence_csv_has_data(self, tmp_workdir):
        import pandas as pd
        atoms = bulk("Cu", "fcc", a=3.7)
        atoms.calc = EMT()
        run_optimization(
            atoms, optimizer="fire", fmax=0.1, max_steps=50,
            output_dir=tmp_workdir, verbose=False,
        )
        df = pd.read_csv(tmp_workdir / "opt_convergence.csv")
        assert "step" in df.columns
        assert "energy(eV)" in df.columns
        assert "fmax(eV/A)" in df.columns
        assert len(df) > 0


class TestOptimizerSelection:
    def test_unknown_optimizer_raises(self):
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.calc = EMT()
        with pytest.raises(ValueError, match="Unknown optimizer"):
            run_optimization(atoms, optimizer="nonexistent")

    def test_all_optimizers_recognized(self):
        expected = {"fire", "bfgs", "lbfgs", "bfgsls", "gpmin", "mdmin"}
        assert set(OPTIMIZER_MAP.keys()) == expected

    @pytest.mark.parametrize("opt_name", ["fire", "bfgs", "lbfgs"])
    def test_optimizer_runs(self, tmp_workdir, opt_name):
        atoms = bulk("Cu", "fcc", a=3.7)
        atoms.calc = EMT()
        converged = run_optimization(
            atoms, optimizer=opt_name, fmax=0.5, max_steps=10,
            output_dir=tmp_workdir, verbose=False,
        )
        assert converged is True or converged is False or isinstance(converged, (bool, np.bool_))


class TestOptimizeRecordsHead:
    def test_run_optimization_writes_the_mace_head_into_the_record(self, tmp_path):
        import json
        from ase.build import bulk
        from ase.calculators.emt import EMT
        from mliprun.core.optimize import run_optimization

        atoms = bulk("Cu", "fcc", a=3.7) * (2, 2, 2)
        atoms.rattle(stdev=0.05, seed=42)
        atoms.calc = EMT()
        run_optimization(atoms, optimizer="bfgs", fmax=0.05, max_steps=20,
                         output_dir=tmp_path, verbose=False,
                         model_name="mace-mh-1", mace_head="omat_pbe")

        data = json.loads((tmp_path / "mliprun_run.json").read_text())
        assert data["provenance"]["mace_head"] == "omat_pbe"
        assert data["provenance"]["uma_task"] is None
