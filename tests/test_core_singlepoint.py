"""Single-point evaluation: energy, forces, fmax, stress."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk, fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from mliprun.core.singlepoint import run_singlepoint


@pytest.fixture
def slab():
    """Pt(111) 2x2x3 with the bottom layer fixed, one atom nudged off site.

    The nudge guarantees a non-zero force so fmax is a real number rather
    than numerical noise, and puts the largest force on a FIXED atom so
    fmax_free and fmax_all cannot coincide by accident.
    """
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    atoms.positions[bottom[0]] += (0.0, 0.0, 0.35)
    atoms.calc = EMT()
    return atoms


def test_energy_matches_a_direct_emt_evaluation(slab, tmp_path):
    expected = slab.get_potential_energy()
    results = run_singlepoint(slab, output_dir=tmp_path)
    assert results["energy_eV"] == pytest.approx(expected, abs=1e-10)


def test_free_and_all_fmax_differ_when_the_worst_atom_is_fixed(slab, tmp_path):
    results = run_singlepoint(slab, output_dir=tmp_path)
    constrained = np.sqrt((slab.get_forces() ** 2).sum(axis=1).max())
    raw = np.sqrt((slab.calc.get_forces(slab) ** 2).sum(axis=1).max())
    assert results["fmax_free_eV_per_A"] == pytest.approx(constrained, abs=1e-10)
    assert results["fmax_all_eV_per_A"] == pytest.approx(raw, abs=1e-10)
    assert results["fmax_all_eV_per_A"] > results["fmax_free_eV_per_A"]


def test_the_free_atom_count_excludes_the_fixed_layer(slab, tmp_path):
    results = run_singlepoint(slab, output_dir=tmp_path)
    assert results["n_free_atoms"] == 8      # 12 atoms, 4 fixed in layer 3


def test_the_worst_force_atom_is_named_for_each_population(slab, tmp_path):
    results = run_singlepoint(slab, output_dir=tmp_path)
    raw = slab.calc.get_forces(slab)
    assert results["worst_force_atom_all"] == int(
        np.argmax((raw ** 2).sum(axis=1)))
    assert results["worst_force_atom_all_symbol"] == "Pt"
    constrained = slab.get_forces()
    assert results["worst_force_atom_free"] == int(
        np.argmax((constrained ** 2).sum(axis=1)))


def test_a_fully_fixed_structure_reports_no_worst_free_atom(tmp_path):
    """`argmax` over an all-zero array returns 0, which named atom 0 as the
    worst offender among the free atoms when there are no free atoms at all.
    An explicit null says "there is no such atom"; a plausible index does
    not, and it is the kind of number that reaches a table."""
    from ase.build import fcc111
    from ase.calculators.emt import EMT
    from ase.constraints import FixAtoms

    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    atoms.set_constraint(FixAtoms(indices=list(range(len(atoms)))))
    atoms.calc = EMT()
    results = run_singlepoint(atoms, output_dir=tmp_path)

    assert results["n_free_atoms"] == 0
    assert results["worst_force_atom_free"] is None
    assert results["worst_force_atom_free_symbol"] is None
    assert results["fmax_free_eV_per_A"] == pytest.approx(0.0, abs=1e-12)
    # The all-atom population still has a worst atom -- it is a real one.
    assert results["worst_force_atom_all"] is not None
    assert results["worst_force_atom_all_symbol"] == "Pt"


def test_the_forces_csv_has_one_row_per_atom_with_the_free_mask(slab, tmp_path):
    run_singlepoint(slab, output_dir=tmp_path)
    rows = list(csv.DictReader(
        (tmp_path / "singlepoint_forces.csv").open()))
    assert len(rows) == len(slab)
    assert rows[0].keys() == {
        "index", "symbol", "fx", "fy", "fz", "f_norm",
        "free_x", "free_y", "free_z"}
    fixed = [a.index for a in slab if a.tag == 3]
    # fcc111's tag-3 (bottom) layer occupies indices 0-3 in this build, so
    # index 0 is itself fixed -- checking a hardcoded "row 0" for True would
    # contradict the row-fixed[0]-is-False assertion above on the same row.
    # Derive a genuinely free row from `fixed` instead of assuming which
    # index that is.
    free_index = next(i for i in range(len(slab)) if i not in fixed)
    assert rows[fixed[0]]["free_z"] == "False"
    assert rows[free_index]["free_z"] == "True"


def test_the_csv_carries_raw_forces_not_constrained_ones(slab, tmp_path):
    run_singlepoint(slab, output_dir=tmp_path)
    rows = list(csv.DictReader(
        (tmp_path / "singlepoint_forces.csv").open()))
    raw = slab.calc.get_forces(slab)
    fixed = [a.index for a in slab if a.tag == 3][0]
    # The constrained array would report exactly zero here.
    assert float(rows[fixed]["f_norm"]) == pytest.approx(
        float(np.linalg.norm(raw[fixed])), abs=1e-10)
    assert float(rows[fixed]["f_norm"]) > 1e-6


def test_stress_is_recorded_for_a_fully_periodic_cell(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.calc = EMT()
    results = run_singlepoint(atoms, output_dir=tmp_path)
    assert results["stress_eV_per_A3"] is not None
    assert len(results["stress_eV_per_A3"]) == 6
    assert results["stress_unavailable_reason"] is None
    expected_gpa = np.asarray(atoms.get_stress()) / 0.006241509
    assert results["stress_GPa"] == pytest.approx(expected_gpa, abs=1e-6)


def test_a_slab_skips_stress_and_says_why(slab, tmp_path):
    results = run_singlepoint(slab, output_dir=tmp_path)
    assert results["stress_eV_per_A3"] is None
    assert "periodic" in results["stress_unavailable_reason"]


def test_a_calculator_without_stress_records_why_and_still_reports_energy(
        tmp_path):
    """A missing stress must never cost the energy."""
    from ase.calculators.calculator import PropertyNotImplementedError

    class NoStress(EMT):
        def get_stress(self, atoms=None):
            raise PropertyNotImplementedError("no stress here")

    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.calc = NoStress()
    results = run_singlepoint(atoms, output_dir=tmp_path)
    assert results["stress_eV_per_A3"] is None
    assert results["stress_GPa"] is None
    assert "PropertyNotImplementedError" in results["stress_unavailable_reason"]
    assert results["energy_eV"] == pytest.approx(
        atoms.get_potential_energy(), abs=1e-10)


def test_stress_false_never_attempts_it(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.calc = EMT()
    results = run_singlepoint(atoms, output_dir=tmp_path, stress=False)
    assert results["stress_eV_per_A3"] is None
    assert results["stress_unavailable_reason"] == "not requested"


def test_stress_true_forces_the_attempt_on_a_slab(slab, tmp_path):
    """The default skips a slab because its vacuum-direction stress means
    nothing; an explicit --stress is the caller overriding that."""
    results = run_singlepoint(slab, output_dir=tmp_path, stress=True)
    assert results["stress_eV_per_A3"] is not None
    assert results["stress_unavailable_reason"] is None


def test_the_run_record_carries_the_singlepoint_stage(slab, tmp_path):
    run_singlepoint(slab, output_dir=tmp_path, model_name="emt")
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    stage = record["stages"][0]
    assert stage["kind"] == "singlepoint"
    assert stage["status"] == "completed"
    assert stage["results"]["energy_eV"] == pytest.approx(
        slab.get_potential_energy(), abs=1e-10)


def test_nothing_is_written_back_as_a_structure(slab, tmp_path):
    run_singlepoint(slab, output_dir=tmp_path)
    written = {p.name for p in tmp_path.iterdir()}
    assert "CONTCAR" not in written
    assert not any(name.endswith(".traj") for name in written)
    assert not any(name.endswith("_final.vasp") for name in written)
