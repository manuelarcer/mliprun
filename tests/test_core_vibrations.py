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


def test_the_energy_column_is_a_magnitude_for_an_imaginary_mode(tmp_path):
    """An imaginary mode's energy is purely imaginary, so ``.real`` of it is
    exactly 0.0 -- and the row then reported a nonzero frequency beside a
    zero energy for the same mode.

    The consistency check is the point: every row's ``energy_meV`` must be
    its ``frequency_cm-1`` in energy units (``ase.units.invcm`` eV per
    cm^-1), imaginary rows included.
    """
    from ase import units

    atoms = molecule("N2")
    atoms.positions[1][2] += 1.6
    atoms.calc = EMT()
    run_frequencies(atoms, output_dir=tmp_path)
    rows = list(csv.DictReader((tmp_path / "freq_frequencies.csv").open()))
    imaginary = [r for r in rows if r["imaginary"] == "True"]
    assert imaginary                       # the case is not vacuous
    for row in imaginary:
        assert float(row["energy_meV"]) > 0.0
        assert float(row["energy_meV"]) == pytest.approx(
            float(row["frequency_cm-1"]) * units.invcm * 1000.0, rel=1e-9)
    for row in rows:
        assert float(row["energy_meV"]) == pytest.approx(
            float(row["frequency_cm-1"]) * units.invcm * 1000.0,
            rel=1e-9, abs=1e-12)


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


# -- the displacement cache is checked against this run's calculator ----

class _OffsetEMT(EMT):
    """EMT with a fixed offset added to every force component.

    Stands in for a real MLIP on a GPU, which is not bit-reproducible
    between runs: the cache guard must accept a re-evaluation that differs
    from the cached forces by less than its tolerance, and refuse one that
    differs by more. A constant offset cancels in the central differences,
    so it changes the guard's input without changing the physics under test.
    """

    def __init__(self, offset):
        super().__init__()
        self._offset = float(offset)

    def calculate(self, *args, **kwargs):
        super().calculate(*args, **kwargs)
        self.results["forces"] = self.results["forces"] + self._offset


def test_a_second_run_with_a_different_calculator_is_refused(n2, tmp_path):
    """The cache is named `<output_dir>/<prefix>` and `prefix` defaults to
    `freq` whatever the model is, so run 2 used to reuse run 1's forces and
    report them under its own provenance: 0 force calls, EMT's frequencies,
    `provenance.mlip_model: "lj"`."""
    from ase.calculators.lj import LennardJones

    from mliprun.core.vibrations import FrequencyCacheError

    first = run_frequencies(n2, output_dir=tmp_path, model_name="emt")
    assert first["n_force_calls"] == 13          # a real sweep happened

    other = n2.copy()
    other.calc = LennardJones()
    with pytest.raises(FrequencyCacheError) as caught:
        run_frequencies(other, output_dir=tmp_path, model_name="lj")

    message = str(caught.value)
    assert str(tmp_path / "freq") in message     # names the cache directory
    assert "--prefix" in message                 # names the second remedy


def test_the_refused_run_leaves_a_failed_record_not_a_running_one(
        n2, tmp_path):
    """`status: "running"` means "the job died without reporting back"
    (docs/OUTPUTS.md). A guard that stops the run must still complete the
    record."""
    from ase.calculators.lj import LennardJones

    from mliprun.core.vibrations import FrequencyCacheError

    run_frequencies(n2, output_dir=tmp_path, model_name="emt")
    other = n2.copy()
    other.calc = LennardJones()
    with pytest.raises(FrequencyCacheError):
        run_frequencies(other, output_dir=tmp_path, model_name="lj")

    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    assert record["status"] == "failed"
    assert record["stages"][-1]["status"] == "failed"
    assert "displacement cache" in record["stages"][-1]["results"]["error"]


def test_a_restart_whose_forces_moved_less_than_the_tolerance_is_accepted(
        n2, tmp_path):
    """Exact equality would be the wrong test: a real MLIP on a GPU is not
    bit-reproducible between runs. 1e-9 eV/A is such a re-evaluation."""
    from mliprun.core.vibrations import CACHE_IDENTITY_ATOL

    first = run_frequencies(n2, output_dir=tmp_path, model_name="emt")
    jittered = n2.copy()
    jittered.calc = _OffsetEMT(1e-9)
    assert 1e-9 < CACHE_IDENTITY_ATOL
    second = run_frequencies(jittered, output_dir=tmp_path, model_name="emt")

    assert second["n_force_calls"] == 0          # the cache was reused
    assert second["frequencies_cm-1"] == pytest.approx(
        first["frequencies_cm-1"], abs=1e-12)


def test_a_restart_whose_forces_moved_more_than_the_tolerance_is_refused(
        n2, tmp_path):
    from mliprun.core.vibrations import (
        CACHE_IDENTITY_ATOL,
        FrequencyCacheError,
    )

    run_frequencies(n2, output_dir=tmp_path, model_name="emt")
    shifted = n2.copy()
    shifted.calc = _OffsetEMT(1e-3)
    assert 1e-3 > CACHE_IDENTITY_ATOL
    with pytest.raises(FrequencyCacheError):
        run_frequencies(shifted, output_dir=tmp_path, model_name="emt")


def test_a_different_prefix_keeps_the_two_runs_apart(n2, tmp_path):
    """The remedy the error message names has to actually work."""
    from ase.calculators.lj import LennardJones

    emt_results = run_frequencies(n2, output_dir=tmp_path, model_name="emt")
    other = n2.copy()
    other.calc = LennardJones()
    lj_results = run_frequencies(other, output_dir=tmp_path,
                                 prefix="freq_lj", model_name="lj")

    assert lj_results["n_force_calls"] == 13     # its own sweep, not a reuse
    assert max(lj_results["frequencies_cm-1"]) != pytest.approx(
        max(emt_results["frequencies_cm-1"]), abs=1e-6)


def test_the_fmax_at_the_input_geometry_is_recorded(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path)
    assert results["fmax_at_input_free_eV_per_A"] == pytest.approx(
        0.0, abs=1e-5)
    assert results["fmax_expectation_source"] == "none"
    assert results["fmax_warning"] is None


@pytest.fixture
def relaxed_slab():
    """Pt(111) 2x2x4 + H, bottom two layers held, relaxed to fmax 0.02.

    The dominant real use case, and the one that makes the two fmax
    populations differ: a frozen layer carries a large force the optimizer
    never had to remove.
    """
    from ase.build import add_adsorbate
    from ase.optimize import BFGS

    atoms = fcc111("Pt", size=(2, 2, 4), vacuum=7.0)
    add_adsorbate(atoms, "H", 1.5, "ontop")
    atoms.set_constraint(
        FixAtoms(indices=[a.index for a in atoms if a.tag in (3, 4)]))
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=0.02)
    return atoms


def test_the_two_input_fmax_populations_differ_on_a_constrained_slab(
        relaxed_slab, tmp_path):
    """The free value is the constrained one, and it is NOT the all-atom one.

    ASE fills its displacement cache from ``calc.get_forces(atoms)``, which
    bypasses constraints. Reporting that raw number under a ``_free`` name
    made the stationary-point warning fire on every correctly relaxed slab:
    0.38 eV/A reported against the 0.02 eV/A the optimizer converged to.
    """
    from mliprun.core.utils import calc_fmax

    results = run_frequencies(relaxed_slab, output_dir=tmp_path)
    free = results["fmax_at_input_free_eV_per_A"]
    everything = results["fmax_at_input_all_eV_per_A"]

    # The constrained criterion the optimizer actually met.
    assert free == pytest.approx(calc_fmax(relaxed_slab.get_forces()),
                                 abs=1e-9)
    assert free < 0.02 + 1e-9
    # The two are genuinely different numbers here, by more than an order of
    # magnitude -- which is why the same value cannot serve both names.
    assert everything > 10 * free
    assert everything == pytest.approx(
        calc_fmax(relaxed_slab.calc.get_forces(relaxed_slab)), abs=1e-9)


def test_a_relaxed_constrained_slab_does_not_trip_the_fmax_warning(
        relaxed_slab, tmp_path):
    """The regression in one line: the warning must stay silent here.

    Measured against the raw all-atom forces it fired every time, on a slab
    that had converged to exactly the expectation it is handed.
    """
    results = run_frequencies(relaxed_slab, output_dir=tmp_path,
                              expect_fmax=0.02)
    assert results["fmax_warning"] is False
    assert results["fmax_expectation"] == pytest.approx(0.02)


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
    # RunRecord.begin() only writes a per-stage `parameters` block when the
    # caller passes `stage_parameters=`, which run_frequencies does not (it
    # matches optimize/singlepoint/md's convention: one parameters block, at
    # the top level, never duplicated per stage). So this reads the record's
    # top-level `parameters`, not the stage's.
    # Without a RunContext, `_tag` stores bare values rather than
    # {"value":, "source":} dicts. Unwrap either shape rather than asserting
    # one and discovering the other in CI.
    delta = record["parameters"]["delta"]
    assert (delta["value"] if isinstance(delta, dict) else delta) == (
        pytest.approx(0.01))


def test_the_recorded_indices_match_what_was_displaced(tmp_path):
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path)
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    # Top-level `parameters`, not the stage's -- see the comment in
    # test_the_run_record_carries_the_freq_stage.
    recorded = record["parameters"]["indices"]
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


def test_no_imaginary_modes_writes_no_trajectory_by_default(n2, tmp_path):
    """write_modes defaults to 'imaginary'; with none present, nothing is
    written -- the selection logic is right by inspection, but nothing
    pinned it."""
    results = run_frequencies(n2, output_dir=tmp_path)
    assert results["n_imaginary"] == 0
    assert list(tmp_path.glob("freq.*.traj")) == []


def test_the_frequency_csv_and_summary_agree_on_which_modes_are_imaginary(
        tmp_path):
    """ASE's own summary table classifies a mode by ``abs(energy.imag) >
    1e-8`` on the mode ENERGY in eV (VibrationsData._tabulate_from_energies),
    not by the sign of the frequency in cm^-1. H2O relaxed under EMT with
    nfree=4 reproducibly leaves one mode's energy imaginary part at ~6e-9 eV
    -- below that threshold -- while its frequency in cm^-1 (~5e-5) is still
    nonzero to floating point. Classifying on the frequency instead of the
    energy would call that mode imaginary in the CSV while the summary table
    calls it real: exactly the disagreement this guards against."""
    from ase.optimize import BFGS
    atoms = molecule("H2O")
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-6)
    results = run_frequencies(atoms, output_dir=tmp_path, nfree=4)
    summary = (tmp_path / "freq_summary.txt").read_text()
    n_imaginary_in_summary = sum(
        1 for line in summary.splitlines() if line.rstrip().endswith("i"))
    assert results["n_imaginary"] > 0          # the case is not vacuous
    assert results["n_imaginary"] == n_imaginary_in_summary
