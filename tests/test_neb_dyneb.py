"""Tests for DyNEB (dynamic NEB) support using the EMT calculator.

DyNEB is ASE's ``ase.mep.dyneb.DyNEB``: a drop-in NEB variant that freezes
images already converged below fmax (dynamic relaxation) and can loosen the
convergence criterion for images far from the barrier top (scale_fmax).
mliprun exposes it as an opt-in flag on ``run_neb`` / ``mlip neb run``;
the default path must remain plain NEB, numerically untouched.
"""
import json

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.mep.dyneb import DyNEB

from mliprun.core.neb import CustomNEB


def _make_neb_pair():
    """Create a simple initial/final pair for NEB testing."""
    initial = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    final = initial.copy()
    pos = final.get_positions()
    pos[0] += np.array([0.3, 0.3, 0.0])
    final.set_positions(pos)
    return initial, final


def _emt_neb(output_dir, monkeypatch, fmax=0.1):
    """CustomNEB on the toy Cu path with EMT swapped in for the MLIP."""
    initial, final = _make_neb_pair()
    neb = CustomNEB(
        initial=initial, final=final, num_images=3, fmax=fmax,
        mlip="test", output_dir=output_dir,
    )
    monkeypatch.setattr(neb, "setup_calculator", lambda: EMT())
    return neb


class TestDynebEngineSelection:
    def test_dyneb_true_builds_dyneb_with_consistent_fmax(self, tmp_workdir, monkeypatch):
        """dyneb=True must construct ase DyNEB with dynamic relaxation on,
        the NEB fmax (DyNEB requires it to equal the optimizer's fmax),
        the forwarded scale_fmax, and the same tangent method as plain NEB."""
        captured = {}

        class RecordingDyNEB(DyNEB):
            def __init__(self, images, **kwargs):
                captured.update(kwargs)
                super().__init__(images, **kwargs)

        monkeypatch.setattr("mliprun.core.neb.DyNEB", RecordingDyNEB)

        neb = _emt_neb(tmp_workdir, monkeypatch)
        neb.run_neb(dyneb=True, scale_fmax=0.5, max_steps=2)

        assert captured["dynamic_relaxation"] is True
        assert captured["fmax"] == neb.fmax
        assert captured["scale_fmax"] == 0.5
        assert captured["method"] == "improvedtangent"

    def test_default_run_does_not_use_dyneb(self, tmp_workdir, monkeypatch):
        """Without the flag the engine must stay plain NEB (frozen-goldens
        policy: default numerics are untouched)."""

        class MustNotBeUsed:
            def __init__(self, *args, **kwargs):
                raise AssertionError("DyNEB must not be instantiated by default")

        monkeypatch.setattr("mliprun.core.neb.DyNEB", MustNotBeUsed)

        neb = _emt_neb(tmp_workdir, monkeypatch)
        neb.run_neb(max_steps=2)
        assert (tmp_workdir / "neb_convergence.csv").exists()


class TestDynebOutputsAndRecord:
    def test_dyneb_writes_standard_outputs_and_stage_parameters(self, tmp_workdir, monkeypatch):
        neb = _emt_neb(tmp_workdir, monkeypatch)
        neb.run_neb(dyneb=True, scale_fmax=0.5, max_steps=2)

        assert (tmp_workdir / "neb_convergence.csv").exists()
        assert (tmp_workdir / "A2B.traj").exists()
        assert (tmp_workdir / "A2B_full.traj").exists()

        record = json.loads((tmp_workdir / "mliprun_run.json").read_text())
        stage_params = record["stages"][0]["parameters"]
        assert stage_params["dyneb"]["value"] is True
        assert stage_params["scale_fmax"]["value"] == 0.5


class TestDynebNumerics:
    def test_dyneb_matches_plain_neb_barrier_with_fewer_force_calls(
            self, tmp_workdir, monkeypatch):
        """Converged to the same fmax, DyNEB must reproduce the plain-NEB
        forward barrier AND spend fewer force evaluations (its whole point:
        converged images are frozen). Observed on ASE 3.26.0 / EMT toy path:
        68 calls plain vs 40 DyNEB, barrier delta < 1e-6 eV; the 0.02 eV
        tolerance is margin for future ASE changes, not observed spread."""
        fmax = 0.1
        barriers = {}
        calls = {}
        for name, dyneb in [("plain", False), ("dyneb", True)]:
            counter = {"n": 0}

            class CountingEMT(EMT):
                def calculate(self, *args, **kwargs):
                    counter["n"] += 1
                    super().calculate(*args, **kwargs)

            outdir = tmp_workdir / name
            initial, final = _make_neb_pair()
            neb = CustomNEB(
                initial=initial, final=final, num_images=3, fmax=fmax,
                mlip="test", output_dir=outdir,
            )
            monkeypatch.setattr(neb, "setup_calculator", lambda: CountingEMT())
            images = neb.run_neb(dyneb=dyneb, max_steps=600)

            record = json.loads((outdir / "mliprun_run.json").read_text())
            assert record["stages"][0]["status"] == "converged", name

            energies = [img.get_potential_energy() for img in images]
            barriers[name] = max(energies) - energies[0]
            calls[name] = counter["n"]

        assert barriers["dyneb"] == pytest.approx(barriers["plain"], abs=0.02)
        assert calls["dyneb"] < calls["plain"]


class TestDynebParametersFile:
    def test_parse_parameters_file_reads_dyneb_keys(self, tmp_path):
        """Restart round-trip: the two DyNEB keys parse back; a legacy file
        without them must not grow the keys (restart then defaults to off)."""
        legacy = tmp_path / "legacy_parameters.txt"
        legacy.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            mace\n"
            "Intermediate images:   3\n"
            "Total images:          5\n"
            "Final fmax:            0.05\n"
        )
        params = CustomNEB._parse_parameters_file(legacy)
        assert "dyneb" not in params
        assert "scale_fmax" not in params

        with_dyneb = tmp_path / "neb_parameters.txt"
        with_dyneb.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            mace\n"
            "Intermediate images:   3\n"
            "Total images:          5\n"
            "Final fmax:            0.05\n"
            "Dynamic NEB:           True\n"
            "Scale fmax:            0.5\n"
        )
        params = CustomNEB._parse_parameters_file(with_dyneb)
        assert params["dyneb"] is True
        assert params["scale_fmax"] == 0.5


class TestCLIForwardsDyneb:
    def test_neb_run_forwards_dyneb_flags(self, tmp_path, monkeypatch):
        """--dyneb / --scale-fmax must reach run_neb and be persisted in
        neb_parameters.txt so a restart reproduces them."""
        import pandas as pd
        from ase.io import write as ase_write
        from typer.testing import CliRunner

        from mliprun.cli.commands.neb import app as neb_app

        atoms_initial = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms_final = atoms_initial.copy()
        pos = atoms_final.get_positions()
        pos[0] += np.array([0.3, 0.3, 0.0])
        atoms_final.set_positions(pos)

        initial_path = tmp_path / "initial.vasp"
        final_path = tmp_path / "final.vasp"
        ase_write(str(initial_path), atoms_initial, format="vasp")
        ase_write(str(final_path), atoms_final, format="vasp")

        captured = {}

        class FakeNEB:
            def __init__(self, **kwargs):
                captured["init"] = kwargs
                self.images = []

            def interpolate_idpp(self):
                pass

            def run_neb(self, **kwargs):
                captured["run_neb"] = kwargs
                return []

            def process_results(self):
                return pd.DataFrame()

            def export_poscars(self):
                pass

        monkeypatch.setattr("mliprun.cli.commands.neb.CustomNEB", FakeNEB)
        monkeypatch.setattr("mliprun.cli.commands.neb.resolve_mlip", lambda m, *a, **k: "mace")
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(neb_app, [
            "--initial", str(initial_path), "--final", str(final_path),
            "--num-images", "3", "--no-optimize-endpoints",
            "--dyneb", "--scale-fmax", "0.5",
        ])

        assert result.exit_code == 0, result.output
        assert captured["run_neb"]["dyneb"] is True
        assert captured["run_neb"]["scale_fmax"] == 0.5

        params_text = (tmp_path / "neb_parameters.txt").read_text()
        assert "Dynamic NEB:" in params_text
        assert "Scale fmax:" in params_text


def _make_restart_dir(tmp_path, dyneb="True", scale_fmax="0.5"):
    """Build a minimal restartable NEB directory: 5-frame trajectory plus a
    parameters file carrying the DyNEB keys."""
    from ase.io.trajectory import Trajectory

    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    with Trajectory(str(tmp_path / "A2B_full.traj"), "w") as traj:
        for _ in range(5):
            traj.write(atoms)

    (tmp_path / "neb_parameters.txt").write_text(
        "NEB Run Parameters\n"
        "===================\n"
        "MLIP model:            mace\n"
        "Device:                cpu\n"
        "Intermediate images:   3\n"
        "Total images:          5\n"
        "IDPP fmax:             0.1\n"
        "IDPP steps:            100\n"
        "Final fmax:            0.05\n"
        "Spring constant (k):   0.1\n"
        "Climb:                 True\n"
        f"Dynamic NEB:           {dyneb}\n"
        f"Scale fmax:            {scale_fmax}\n"
        "NEB optimizer:         fire\n"
        "NEB max steps:         600\n"
        "Optimize endpoints:    False\n"
        "Log file:              neb.log\n"
        f"Output dir:            {tmp_path}\n"
    )


class TestCLIRestartDyneb:
    def _invoke_restart(self, tmp_path, monkeypatch, extra_args=()):
        """Run `neb --restart` with the heavy methods captured/stubbed;
        loading and parameter resolution stay real."""
        import pandas as pd
        from typer.testing import CliRunner

        from mliprun.cli.commands.neb import app as neb_app

        captured = {}
        monkeypatch.setattr(
            CustomNEB, "run_neb",
            lambda self, **kwargs: captured.update(kwargs) or [])
        monkeypatch.setattr(
            CustomNEB, "process_results", lambda self: pd.DataFrame())
        monkeypatch.setattr(CustomNEB, "export_poscars", lambda self: None)
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(neb_app, ["--restart", *extra_args])
        assert result.exit_code == 0, result.output
        return captured

    def test_restart_roundtrips_dyneb_from_params_file(self, tmp_path, monkeypatch):
        """A restart with no flags must reproduce the recorded DyNEB
        settings, and rewrite them into the new parameters file."""
        _make_restart_dir(tmp_path)
        captured = self._invoke_restart(tmp_path, monkeypatch)

        assert captured["dyneb"] is True
        assert captured["scale_fmax"] == 0.5

        rewritten = (tmp_path / "neb_parameters.txt").read_text()
        assert "Dynamic NEB:" in rewritten
        assert "Scale fmax:" in rewritten

    def test_restart_cli_flags_override_params_file(self, tmp_path, monkeypatch):
        """--no-dyneb on restart must beat the True recorded in the file
        (same override semantics as climb/k)."""
        _make_restart_dir(tmp_path)
        captured = self._invoke_restart(
            tmp_path, monkeypatch, extra_args=["--no-dyneb", "--scale-fmax", "0.0"])

        assert captured["dyneb"] is False
        assert captured["scale_fmax"] == 0.0
