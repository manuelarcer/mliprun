"""Frequencies, ZPE, imaginary modes and restart, on EMT."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import fcc111, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.vibrations import VibrationsData

from mliprun.core.vibrations import run_frequencies


@pytest.fixture
def n2():
    """N2 at EMT's own minimum, so the only real mode is the stretch."""
    from ase.optimize import BFGS
    atoms = molecule("N2")
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-6)
    return atoms


def test_the_frequency_count_is_three_times_the_displaced_atoms(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path)
    assert results["n_modes"] == 6
    assert results["n_displaced_atoms"] == 2
    assert len(results["frequencies_cm-1"]) == 6


def test_the_force_call_count_is_one_plus_six_per_displaced_atom(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path)
    assert results["n_force_calls"] == 1 + 6 * 2


def test_nfree_four_doubles_the_displacements(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path, nfree=4)
    assert results["n_force_calls"] == 1 + 12 * 2


def test_frequencies_match_ases_own_for_the_same_settings(n2, tmp_path):
    from ase.vibrations import Vibrations
    # Vibrations.__init__ captures `self.calc = atoms.calc` at construction
    # time, and Atoms.copy() never copies an attached calculator -- so the
    # calc must be attached before the Vibrations object is built, not after.
    reference_atoms = n2.copy()
    reference_atoms.calc = EMT()
    reference = Vibrations(reference_atoms, name=str(tmp_path / "ref"))
    reference.run()
    expected = np.sort(np.abs(reference.get_frequencies()))
    results = run_frequencies(n2, output_dir=tmp_path / "ours")
    assert np.sort(results["frequencies_cm-1"]) == pytest.approx(
        expected, abs=1e-8)


def test_the_zero_point_energy_is_half_the_summed_real_energies(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path)
    data = VibrationsData.read(str(tmp_path / "freq_vibrations.json"))
    assert results["zpe_eV"] == pytest.approx(
        0.5 * np.asarray(data.get_energies()).real.sum(), abs=1e-12)


def test_a_stretched_bond_produces_an_imaginary_mode(tmp_path):
    """A geometry at a maximum along the bond, not a minimum."""
    atoms = molecule("N2")
    atoms.positions[1][2] += 1.6          # far past EMT's minimum
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path)
    assert results["n_imaginary"] >= 1
    assert any(results["imaginary_mask"])


def test_an_imaginary_frequency_is_written_as_a_positive_magnitude(tmp_path):
    atoms = molecule("N2")
    atoms.positions[1][2] += 1.6
    atoms.calc = EMT()
    run_frequencies(atoms, output_dir=tmp_path)
    rows = list(csv.DictReader((tmp_path / "freq_frequencies.csv").open()))
    imaginary = [r for r in rows if r["imaginary"] == "True"]
    assert imaginary
    for row in imaginary:
        assert float(row["frequency_cm-1"]) > 0.0


def test_an_imaginary_mode_contributes_nothing_to_the_zpe(tmp_path):
    atoms = molecule("N2")
    atoms.positions[1][2] += 1.6
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path)
    data = VibrationsData.read(str(tmp_path / "freq_vibrations.json"))
    energies = np.asarray(data.get_energies())
    real_only = 0.5 * energies.real.sum()
    assert results["zpe_eV"] == pytest.approx(real_only, abs=1e-12)


def test_only_the_free_atoms_are_displaced_on_a_constrained_slab(tmp_path):
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path)
    assert results["n_displaced_atoms"] == 8
    assert results["n_force_calls"] == 1 + 6 * 8
    assert results["n_modes"] == 24


def test_the_summary_file_is_written_once_not_appended(n2, tmp_path):
    """vib.summary()'s log argument opens a PATH in append mode. Passing a
    path twice would write two tables into one file."""
    run_frequencies(n2, output_dir=tmp_path)
    first = (tmp_path / "freq_summary.txt").read_text()
    run_frequencies(n2, output_dir=tmp_path)
    second = (tmp_path / "freq_summary.txt").read_text()
    assert first == second


def test_a_restart_makes_only_the_remaining_calls(n2, tmp_path):
    results_one = run_frequencies(n2, output_dir=tmp_path)
    assert results_one["n_force_calls"] == 13
    results_two = run_frequencies(n2, output_dir=tmp_path)
    assert results_two["n_force_calls"] == 0
    assert results_two["frequencies_cm-1"] == pytest.approx(
        results_one["frequencies_cm-1"], abs=1e-12)


def test_the_fmax_at_the_input_geometry_is_recorded(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path)
    assert results["fmax_at_input_free_eV_per_A"] == pytest.approx(
        0.0, abs=1e-5)
    assert results["fmax_expectation_source"] == "none"
    assert results["fmax_warning"] is None


def test_an_exceeded_expectation_warns_but_still_runs(tmp_path):
    atoms = molecule("N2")
    atoms.positions[1][2] += 0.3
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path, expect_fmax=1e-6)
    assert results["fmax_warning"] is True
    assert results["n_modes"] == 6          # it still ran


def test_the_run_record_carries_the_freq_stage(n2, tmp_path):
    run_frequencies(n2, output_dir=tmp_path, model_name="emt")
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    stage = record["stages"][0]
    assert stage["kind"] == "freq"
    assert stage["status"] == "completed"
    # Without a RunContext, `_tag` stores bare values rather than
    # {"value":, "source":} dicts. Unwrap either shape rather than asserting
    # one and discovering the other in CI.
    delta = stage["parameters"]["delta"]
    assert (delta["value"] if isinstance(delta, dict) else delta) == (
        pytest.approx(0.01))


def test_the_recorded_indices_match_what_was_displaced(tmp_path):
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path)
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    recorded = record["stages"][0]["parameters"]["indices"]
    recorded = recorded["value"] if isinstance(recorded, dict) else recorded
    assert len(recorded) == results["n_displaced_atoms"]
    assert set(recorded).isdisjoint(bottom)


def test_only_imaginary_modes_get_a_trajectory_by_default(tmp_path):
    atoms = molecule("N2")
    atoms.positions[1][2] += 1.6
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path)
    written = sorted(p.name for p in tmp_path.glob("freq.*.traj"))
    assert len(written) == results["n_imaginary"]


def test_write_modes_none_writes_no_trajectory(tmp_path):
    atoms = molecule("N2")
    atoms.positions[1][2] += 1.6
    atoms.calc = EMT()
    run_frequencies(atoms, output_dir=tmp_path, write_modes="none")
    assert list(tmp_path.glob("freq.*.traj")) == []


def test_write_modes_all_writes_every_mode(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path, write_modes="all")
    assert len(list(tmp_path.glob("freq.*.traj"))) == results["n_modes"]
