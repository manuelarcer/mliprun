"""verify_ts on a one-atom double well, where every answer is known analytically."""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.emt import EMT
from ase.build import molecule
from ase.constraints import FixAtoms
from ase.io import read as ase_read

from mliprun.core.ts_verify import verify_ts
from mliprun.core.vibrations import run_frequencies
from ts_helpers import CENTRE, DoubleWell, single_atom

X = np.array([1.0, 0.0, 0.0])


def _structures(ts_shift=0.0):
    ts = single_atom(CENTRE + ts_shift * X)
    initial = single_atom(CENTRE - X)
    final = single_atom(CENTRE + X)
    return ts, initial, final


def _verify(tmp_path, ts_shift=0.0, factory=DoubleWell, **kwargs):
    ts, initial, final = _structures(ts_shift)
    return verify_ts(ts, initial, final, calculator_factory=factory,
                     output_dir=tmp_path, **kwargs)


def test_a_true_saddle_passes_the_overlap_check(tmp_path):
    r = _verify(tmp_path)
    assert r["verdict"] == "verified"
    assert r["n_imaginary"] == 1
    assert r["overlap"]["value"] == pytest.approx(1.0, abs=1e-4)
    assert r["overlap"]["reference"] == "is_to_fs"
    assert r["overlap"]["status"] == "pass"


def test_a_mode_orthogonal_to_the_reference_fails(tmp_path):
    ref = np.array([[0.0, 1.0, 0.0]])
    r = _verify(tmp_path, reference_direction=ref, reference_label="neb_tangent")
    assert r["overlap"]["value"] == pytest.approx(0.0, abs=1e-4)
    assert r["verdict"] == "not_verified"
    assert r["overlap"]["reference"] == "neb_tangent"


def test_the_overlap_threshold_is_adjustable(tmp_path):
    ref = np.array([[1.0, 1.0, 0.0]])  # overlap 0.7071
    assert _verify(tmp_path / "a", reference_direction=ref,
                   reference_label="x")["verdict"] == "verified"
    assert _verify(tmp_path / "b", reference_direction=ref, reference_label="x",
                   overlap_threshold=0.8)["verdict"] == "not_verified"


def test_two_imaginary_modes_at_a_converged_geometry_is_not_verified(tmp_path):
    r = _verify(tmp_path, factory=lambda: DoubleWell(by=-2.0))
    assert r["n_imaginary"] == 2 and len(r["imaginary_frequencies_cm-1"]) == 2
    assert r["verdict"] == "not_verified"
    assert r["overlap"]["status"] == "not_run"


def test_two_imaginary_modes_at_a_loose_geometry_is_inconclusive(tmp_path):
    # TS displaced 0.05 A along x: |F| = 4*0.05*(1-0.0025) = 0.1995 eV/A > 0.03
    r = _verify(tmp_path, ts_shift=0.05, factory=lambda: DoubleWell(by=-2.0))
    assert r["ts_fmax_free_eV_per_A"] == pytest.approx(0.1995, abs=1e-3)
    assert r["ts_fmax_above_limit"] is True
    assert r["verdict"] == "inconclusive" and "not converged enough" in r["reason"]


def test_a_minimum_has_no_imaginary_mode(tmp_path):
    r = _verify(tmp_path, ts_shift=1.0)  # "TS" sits at the minimum centre + (1, 0, 0)
    assert r["n_imaginary"] == 0
    assert r["verdict"] == "not_verified"
    assert r["overlap"]["status"] == "not_run"


def test_outputs_and_record(tmp_path):
    _verify(tmp_path)
    for name in ("ts_structure.vasp", "ts_freq_frequencies.csv", "ts_freq_vibrations.json",
                 "ts_verify.txt", "mliprun_run.json"):
        assert (tmp_path / name).exists()
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    assert record["command"] == "ts-verify"
    assert [s["kind"] for s in record["stages"]] == ["freq", "ts-verify"]
    assert record["stages"][1]["status"] == "completed"
    assert record["stages"][1]["results"]["verdict"] == "verified"
    assert (tmp_path / "ts_verify.txt").read_text().splitlines()[0].endswith("verified")


def test_extra_inputs_are_kept_in_the_stage_results(tmp_path):
    _verify(tmp_path, extra_inputs={"ts_source": "neb.traj"})
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    assert record["stages"][1]["results"]["inputs"] == {"ts_source": "neb.traj"}


def test_the_callers_atoms_are_not_modified(tmp_path):
    ts, initial, final = _structures()
    verify_ts(ts, initial, final, calculator_factory=DoubleWell, output_dir=tmp_path)
    assert ts.calc is None
    assert np.allclose(ts.get_positions(), [CENTRE])


def test_mismatched_structures_stop_before_any_force_call(tmp_path):
    calls = []

    def factory():
        calls.append(1)
        return DoubleWell()

    ts, initial, final = _structures()
    final.set_chemical_symbols(["He"])
    with pytest.raises(ValueError, match="final"):
        verify_ts(ts, initial, final, calculator_factory=factory, output_dir=tmp_path)
    assert calls == []
    assert not (tmp_path / "mliprun_run.json").exists()


def test_identical_endpoints_are_an_error_not_nan(tmp_path):
    ts, initial, _ = _structures()
    with pytest.raises(ValueError, match="zero"):
        verify_ts(ts, initial, initial, calculator_factory=DoubleWell, output_dir=tmp_path)


def test_a_failure_after_the_stage_began_is_recorded_as_failed(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("overlap exploded")

    monkeypatch.setattr("mliprun.core.ts_verify.mode_overlap", broken)
    with pytest.raises(RuntimeError, match="overlap exploded"):
        _verify(tmp_path)
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    assert [s["kind"] for s in record["stages"]] == ["freq", "ts-verify"]
    assert record["stages"][-1]["status"] == "failed"
    assert record["stages"][-1]["results"]["error"] == "overlap exploded"


def test_a_reference_of_the_wrong_shape_is_refused_before_any_force_call(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        _verify(tmp_path, reference_direction=np.ones((3, 3)), reference_label="bad")
    assert not (tmp_path / "mliprun_run.json").exists()


def test_an_empty_checks_is_refused_before_any_force_call(tmp_path):
    calls = []

    def factory():
        calls.append(1)
        return DoubleWell()

    with pytest.raises(ValueError, match="checks"):
        _verify(tmp_path, factory=factory, checks=())
    assert calls == []
    assert not (tmp_path / "mliprun_run.json").exists()


def test_a_reference_label_without_a_direction_is_refused(tmp_path):
    with pytest.raises(ValueError, match="reference_label"):
        _verify(tmp_path, reference_label="neb_tangent")
    assert not (tmp_path / "mliprun_run.json").exists()


def test_a_reference_direction_without_a_label_is_refused(tmp_path):
    with pytest.raises(ValueError, match="reference_label"):
        _verify(tmp_path, reference_direction=np.array([[1.0, 0.0, 0.0]]))
    assert not (tmp_path / "mliprun_run.json").exists()


def test_a_true_saddle_connects_both_minima(tmp_path):
    r = _verify(tmp_path, checks=("overlap", "relax"))
    assert r["relax"]["result"] == "connects" and r["relax"]["status"] == "pass"
    assert r["verdict"] == "verified"
    sides = {r["relax"]["plus"]["matched"], r["relax"]["minus"]["matched"]}
    assert sides == {"initial", "final"}
    for side in ("plus", "minus"):
        assert r["relax"][side]["converged"] is True
        assert r["relax"][side]["returned_to_ts"] is False
        assert r["relax"][side]["distance_to_ts_A"] == pytest.approx(1.0, abs=1e-2)
        assert r["relax"][side]["energy_difference_eV"] == pytest.approx(0.0, abs=1e-4)
        assert r["relax"][side]["distance_to_initial_A"] + r["relax"][side]["distance_to_final_A"] == pytest.approx(2.0, abs=1e-2)
    assert r["relax"]["endpoints_relaxed"] is True
    assert "note" not in r["relax"]
    assert r["relax"]["initial_fmax_eV_per_A"] == pytest.approx(0.0, abs=1e-8)
    assert r["relax"]["optimizer"] == "bfgs" and r["relax"]["max_steps"] == 200
    assert r["relax"]["displacement_A"] == 0.1 and r["relax"]["match_tolerance_A"] == 0.3
    assert r["relax"]["fmax_eV_per_A"] == 0.01
    for name in ("ts_relax_plus.traj", "ts_relax_minus.traj", "ts_relax_plus.log",
                 "ts_relax_minus.log", "ts_relax_plus.vasp", "ts_relax_minus.vasp"):
        assert (tmp_path / name).exists()
    report = (tmp_path / "ts_verify.txt").read_text()
    assert "Displace-and-relax check: connects" in report
    assert "fmax of IS 0.0000 eV/A, of FS 0.0000 eV/A" in report
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    assert record["stages"][1]["results"]["relax"]["result"] == "connects"


def test_relax_only_does_not_report_an_overlap(tmp_path):
    r = _verify(tmp_path, checks=("relax",))
    assert "overlap" not in r and r["verdict"] == "verified"


def test_a_wrong_final_state_is_one_side(tmp_path):
    ts, initial, _ = _structures()
    final = single_atom(CENTRE + np.array([0.0, 3.0, 0.0]))  # not a minimum
    r = verify_ts(ts, initial, final, calculator_factory=DoubleWell,
                  checks=("relax",), output_dir=tmp_path)
    assert r["relax"]["result"] == "one_side" and r["verdict"] == "not_verified"
    assert r["relax"]["endpoints_relaxed"] is False
    assert r["relax"]["final_fmax_eV_per_A"] == pytest.approx(12.0, abs=1e-6)
    assert "Unrelaxed IS or FS" in (tmp_path / "ts_verify.txt").read_text()


def test_an_unconverged_relaxation_is_inconclusive(tmp_path):
    r = _verify(tmp_path, checks=("relax",), relax_max_steps=1)
    assert r["relax"]["result"] == "inconclusive" and r["relax"]["status"] == "inconclusive"
    assert r["relax"]["plus"]["converged"] is False
    assert r["verdict"] == "inconclusive"


def test_the_match_tolerance_decides_the_match(tmp_path):
    ts = single_atom(CENTRE)
    initial = single_atom(CENTRE - X)
    final = single_atom(CENTRE + 1.2 * X)  # 0.2 A from the true minimum
    loose = verify_ts(ts, initial, final, calculator_factory=DoubleWell,
                      checks=("relax",), output_dir=tmp_path / "a")
    tight = verify_ts(ts, initial, final, calculator_factory=DoubleWell,
                      checks=("relax",), match_tolerance=0.1, output_dir=tmp_path / "b")
    assert loose["relax"]["result"] == "connects"
    assert tight["relax"]["result"] == "one_side"


def test_the_displacement_size_is_applied(tmp_path):
    for size in (0.1, 0.25):
        out = tmp_path / str(size)
        _verify(out, checks=("relax",), displacement=size)
        first = ase_read(str(out / "ts_relax_plus.traj"), index=0)
        assert np.linalg.norm(first.get_positions()[0] - CENTRE) == pytest.approx(size, abs=1e-6)


def test_an_unknown_optimizer_stops_before_any_force_call(tmp_path):
    calls = []

    def factory():
        calls.append(1)
        return DoubleWell()

    with pytest.raises(ValueError, match="optimizer"):
        _verify(tmp_path, factory=factory, checks=("relax",), relax_optimizer="newton")
    assert calls == []
    assert not (tmp_path / "mliprun_run.json").exists()


def test_the_optimizer_name_is_case_insensitive(tmp_path):
    r = _verify(tmp_path, checks=("relax",), relax_optimizer="Fire", relax_max_steps=400)
    assert r["relax"]["optimizer"] == "fire"
    assert r["relax"]["result"] == "connects"


def _assert_returned_to_ts(r):
    relax = r["relax"]
    assert relax["result"] == "inconclusive" and relax["status"] == "inconclusive"
    assert r["verdict"] == "inconclusive"
    assert any(relax[s]["returned_to_ts"] is True and relax[s]["distance_to_ts_A"] < 0.3
               for s in ("plus", "minus"))
    assert relax["note"] == "a relaxation returned to the TS; use a larger displacement or another optimizer"


def test_lbfgs_converging_back_onto_the_saddle_is_inconclusive(tmp_path):
    # On this surface LBFGS overshoots across the flat saddle and converges on it.
    r = _verify(tmp_path, checks=("relax",), relax_optimizer="lbfgs")
    _assert_returned_to_ts(r)
    assert "a relaxation returned to the TS" in (tmp_path / "ts_verify.txt").read_text()


def test_a_relaxation_that_starts_converged_on_the_ts_is_inconclusive(tmp_path):
    # At the exact saddle the force is zero, so a tiny displacement converges at once.
    r = _verify(tmp_path, checks=("relax",), displacement=1e-6)
    _assert_returned_to_ts(r)
    assert r["relax"]["plus"]["steps"] == 0 and r["relax"]["plus"]["matched"] is None


def test_two_sides_matching_the_same_endpoint_is_same_side(tmp_path):
    ts = single_atom(CENTRE)
    initial = single_atom(CENTRE - X)
    final = single_atom(CENTRE + np.array([0.0, 5.0, 0.0]))
    r = verify_ts(ts, initial, final, calculator_factory=DoubleWell, checks=("relax",),
                  match_tolerance=2.5, output_dir=tmp_path)
    assert r["relax"]["result"] == "same_side" and r["verdict"] == "not_verified"
    assert r["relax"]["plus"]["matched"] == r["relax"]["minus"]["matched"] == "initial"


def test_the_relax_check_is_not_run_without_exactly_one_imaginary_mode(tmp_path):
    calls = []

    def factory():
        calls.append(1)
        return DoubleWell(by=-2.0)

    r = _verify(tmp_path, factory=factory, checks=("relax",))
    assert r["n_imaginary"] == 2
    assert r["relax"]["status"] == "not_run"
    assert r["relax"]["optimizer"] == "bfgs" and r["relax"]["max_steps"] == 200
    for key in ("plus", "minus", "endpoints_relaxed", "initial_fmax_eV_per_A"):
        assert key not in r["relax"]
    assert len(calls) == 1  # the frequency run only: no relaxation, no endpoint force call
    assert not (tmp_path / "ts_relax_plus.traj").exists()


class _WithSpectator(DoubleWell):
    """Adds a harmonic pull on atom 1 towards the origin."""

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.positions[1]
        self.results["energy"] += float(pos @ pos)
        self.results["forces"][1] = -2.0 * pos


def test_constraints_are_carried_into_the_relaxations(tmp_path):
    spectator = np.array([8.0, 8.0, 8.0])

    def pair(position):
        atoms = Atoms("H2", positions=[position, spectator], cell=[10.0, 10.0, 10.0], pbc=False)
        atoms.set_constraint(FixAtoms(indices=[1]))
        return atoms

    r = verify_ts(pair(CENTRE), pair(CENTRE - X), pair(CENTRE + X),
                  calculator_factory=_WithSpectator, checks=("relax",), output_dir=tmp_path)
    assert r["relax"]["result"] == "connects"
    for name in ("ts_relax_plus.vasp", "ts_relax_minus.vasp"):
        relaxed = ase_read(str(tmp_path / name))
        assert np.allclose(relaxed.get_positions()[1], spectator, atol=1e-6)



def _pair_with_fixed_spectator(position):
    atoms = Atoms("H2", positions=[position, [8.0, 8.0, 8.0]], cell=[10.0, 10.0, 10.0], pbc=False)
    atoms.set_constraint(FixAtoms(indices=[1]))
    return atoms


def test_explicit_indices_that_include_a_fixed_atom_still_displace_the_free_atom_fully(tmp_path):
    r = verify_ts(_pair_with_fixed_spectator(CENTRE), _pair_with_fixed_spectator(CENTRE - X),
                  _pair_with_fixed_spectator(CENTRE + X), calculator_factory=_WithSpectator,
                  checks=("relax",), indices=[0, 1], output_dir=tmp_path)
    assert r["displaced_indices"] == [0, 1]
    first = ase_read(str(tmp_path / "ts_relax_plus.traj"), index=0)
    assert np.linalg.norm(first.get_positions()[0] - CENTRE) == pytest.approx(0.1, abs=1e-6)


def test_a_mode_component_on_a_fixed_atom_does_not_shrink_the_displacement(tmp_path, monkeypatch):
    # The fixed atom 1 carries the largest component: before the fix the free atom moved only 0.05 A.
    mode = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    monkeypatch.setattr("mliprun.core.ts_verify._read_imaginary_mode", lambda *a, **k: mode)
    verify_ts(_pair_with_fixed_spectator(CENTRE), _pair_with_fixed_spectator(CENTRE - X),
              _pair_with_fixed_spectator(CENTRE + X), calculator_factory=_WithSpectator,
              checks=("relax",), indices=[0, 1], output_dir=tmp_path)
    first = ase_read(str(tmp_path / "ts_relax_plus.traj"), index=0)
    assert np.linalg.norm(first.get_positions()[0] - CENTRE) == pytest.approx(0.1, abs=1e-6)


def test_a_mode_with_no_free_component_makes_the_relax_check_inconclusive(tmp_path, monkeypatch):
    mode = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    monkeypatch.setattr("mliprun.core.ts_verify._read_imaginary_mode", lambda *a, **k: mode)
    r = verify_ts(_pair_with_fixed_spectator(CENTRE), _pair_with_fixed_spectator(CENTRE - X),
                  _pair_with_fixed_spectator(CENTRE + X), calculator_factory=_WithSpectator,
                  checks=("relax",), indices=[0, 1], output_dir=tmp_path)
    assert r["relax"]["status"] == "inconclusive" and r["relax"]["result"] == "inconclusive"
    assert r["relax"]["note"] == "the imaginary mode has no component on the free atoms"
    assert r["verdict"] == "inconclusive"
    assert not (tmp_path / "ts_relax_plus.traj").exists()
    assert "Note: the imaginary mode has no component on the free atoms" in (tmp_path / "ts_verify.txt").read_text()


CONSTRAINT_WARNING = "constraints differ between the TS and the initial/final structures"


def test_constraints_that_differ_between_ts_and_endpoints_are_a_warning(tmp_path):
    def free_pair(position):
        return Atoms("H2", positions=[position, [8.0, 8.0, 8.0]], cell=[10.0, 10.0, 10.0], pbc=False)

    r = verify_ts(_pair_with_fixed_spectator(CENTRE), free_pair(CENTRE - X), free_pair(CENTRE + X),
                  calculator_factory=_WithSpectator, output_dir=tmp_path)
    assert r["warnings"] == [CONSTRAINT_WARNING]
    assert f"Warning: {CONSTRAINT_WARNING}" in (tmp_path / "ts_verify.txt").read_text()
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    assert record["stages"][-1]["results"]["warnings"] == [CONSTRAINT_WARNING]


def test_matching_constraints_give_no_warning_key(tmp_path):
    r = verify_ts(_pair_with_fixed_spectator(CENTRE), _pair_with_fixed_spectator(CENTRE - X),
                  _pair_with_fixed_spectator(CENTRE + X), calculator_factory=_WithSpectator,
                  output_dir=tmp_path)
    assert "warnings" not in r
    assert "Warning:" not in (tmp_path / "ts_verify.txt").read_text()


def test_freq_run_record_command_is_unchanged_by_default(tmp_path):
    atoms = molecule("N2")
    atoms.calc = EMT()
    run_frequencies(atoms, output_dir=tmp_path)
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    assert record["command"] == "freq"


def test_freq_run_record_command_can_be_overridden(tmp_path):
    atoms = molecule("N2")
    atoms.calc = EMT()
    run_frequencies(atoms, output_dir=tmp_path, record_command="ts-verify")
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    assert record["command"] == "ts-verify"


def test_an_unknown_check_is_refused_before_any_force_call(tmp_path):
    with pytest.raises(ValueError, match="unknown check"):
        _verify(tmp_path, checks=("overlap", "bogus"))
    assert not (tmp_path / "mliprun_run.json").exists()


FMAX_NOTE = "Note: the TS fmax is above the limit; the geometry is loosely converged."
BAND_NOTE = "Note: the NEB band was not a converged climbing-image run."


def test_a_loose_ts_is_flagged_in_the_report_even_when_verified(tmp_path):
    # Displaced 0.05 A: one imaginary mode, overlap passes, but fmax 0.17 > 0.03 eV/A.
    r = _verify(tmp_path, ts_shift=0.05)
    assert r["verdict"] == "verified" and r["ts_fmax_above_limit"] is True
    assert FMAX_NOTE in (tmp_path / "ts_verify.txt").read_text()


def test_a_tight_ts_has_no_fmax_note(tmp_path):
    _verify(tmp_path)
    assert "loosely converged" not in (tmp_path / "ts_verify.txt").read_text()


def test_the_report_lists_the_inputs(tmp_path):
    _verify(tmp_path, extra_inputs={"neb_dir": "/some/neb", "n_images": 5})
    report = (tmp_path / "ts_verify.txt").read_text()
    assert "Inputs:\n  neb_dir: /some/neb\n  n_images: 5\n" in report
    assert "not a converged climbing-image run" not in report


@pytest.mark.parametrize("inputs", [
    {"neb_last_stage_status": "not_converged", "neb_last_stage_climb": True},
    {"neb_last_stage_status": "converged", "neb_last_stage_climb": False},
])
def test_an_unconverged_or_non_climbing_band_is_noted_in_the_report(tmp_path, inputs):
    _verify(tmp_path, extra_inputs=inputs)
    assert BAND_NOTE in (tmp_path / "ts_verify.txt").read_text()


def test_a_converged_climbing_band_and_unknown_status_have_no_band_note(tmp_path):
    _verify(tmp_path / "a", extra_inputs={"neb_last_stage_status": "converged", "neb_last_stage_climb": True})
    _verify(tmp_path / "b", extra_inputs={"neb_last_stage_status": None, "neb_last_stage_climb": None})
    for name in ("a", "b"):
        assert BAND_NOTE not in (tmp_path / name / "ts_verify.txt").read_text()


def test_a_changed_ts_geometry_archives_the_previous_verification(tmp_path):
    first = _verify(tmp_path)
    assert "archived_previous" not in first
    first_text = (tmp_path / "ts_verify.txt").read_text()

    second = _verify(tmp_path, ts_shift=0.01)
    assert second["archived_previous"] == "previous_1"
    assert (tmp_path / "previous_1" / "ts_verify.txt").read_text() == first_text
    assert (tmp_path / "previous_1" / "ts_structure.vasp").exists()
    assert (tmp_path / "previous_1" / "ts_freq_vibrations.json").exists()
    report = (tmp_path / "ts_verify.txt").read_text()
    assert "A verification of a different TS geometry was found here and moved to previous_1/." in report
    new_ts = ase_read(str(tmp_path / "ts_structure.vasp"))
    assert new_ts.get_positions()[0][0] == pytest.approx(CENTRE[0] + 0.01, abs=1e-5)

    third = _verify(tmp_path, ts_shift=0.02)
    assert third["archived_previous"] == "previous_2"
    assert (tmp_path / "previous_1").is_dir()  # earlier archives are never moved into the next one
    assert not (tmp_path / "previous_2" / "previous_1").exists()
    assert "A verification of a different TS geometry" in (tmp_path / "ts_verify.txt").read_text()


def test_archiving_moves_only_the_verifications_own_files(tmp_path):
    _verify(tmp_path)
    (tmp_path / "user_notes.txt").write_text("keep me")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "x.txt").write_text("input")

    second = _verify(tmp_path, ts_shift=0.01)
    assert second["archived_previous"] == "previous_1"
    assert (tmp_path / "user_notes.txt").read_text() == "keep me"
    assert (tmp_path / "inputs" / "x.txt").read_text() == "input"
    archived = {p.name for p in (tmp_path / "previous_1").iterdir()}
    assert {"ts_verify.txt", "ts_structure.vasp", "mliprun_run.json", "ts_freq",
            "ts_freq_vibrations.json"} <= archived
    assert "user_notes.txt" not in archived and "inputs" not in archived


def test_a_foreign_run_record_is_not_archived(tmp_path):
    from mliprun.core.ts_verify import _prepare_output_dir

    stale = _structures(0.5)[0]
    from ase.io import write as ase_write
    ase_write(str(tmp_path / "ts_structure.vasp"), stale, format="vasp")
    foreign = '{"command": "neb", "stages": []}'
    (tmp_path / "mliprun_run.json").write_text(foreign)
    (tmp_path / "A2B.traj").write_text("band")

    assert _prepare_output_dir(tmp_path, _structures()[0]) == "previous_1"
    assert (tmp_path / "mliprun_run.json").read_text() == foreign
    assert (tmp_path / "A2B.traj").read_text() == "band"
    assert not (tmp_path / "previous_1" / "mliprun_run.json").exists()
    assert (tmp_path / "previous_1" / "ts_structure.vasp").exists()


def test_an_unreadable_run_record_is_not_archived(tmp_path):
    from ase.io import write as ase_write

    from mliprun.core.ts_verify import _prepare_output_dir

    ase_write(str(tmp_path / "ts_structure.vasp"), _structures(0.5)[0], format="vasp")
    (tmp_path / "mliprun_run.json").write_text("{ not json")
    assert _prepare_output_dir(tmp_path, _structures()[0]) == "previous_1"
    assert (tmp_path / "mliprun_run.json").read_text() == "{ not json"


def test_a_stale_report_is_removed_even_without_a_stored_structure(tmp_path):
    (tmp_path / "ts_verify.txt").write_text("TS verification: verified\n")
    (tmp_path / "ts_relax_plus.log").write_text("old")
    from mliprun.core.ts_verify import _prepare_output_dir

    assert _prepare_output_dir(tmp_path, _structures()[0]) is None
    assert not (tmp_path / "ts_verify.txt").exists()
    assert not (tmp_path / "ts_relax_plus.log").exists()


def test_the_foreign_record_after_a_run_is_replaced_by_the_frequency_run(tmp_path):
    from ase.io import write as ase_write

    ase_write(str(tmp_path / "ts_structure.vasp"), _structures(0.5)[0], format="vasp")
    (tmp_path / "mliprun_run.json").write_text('{"command": "neb", "stages": []}')
    _verify(tmp_path)
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    # Measured: the left-in-place foreign record is replaced by the frequency run,
    # as `freq run --output-dir` does; it is not appended to.
    assert record["command"] == "ts-verify"
    assert [s["kind"] for s in record["stages"]] == ["freq", "ts-verify"]


@pytest.mark.parametrize("change", ["count", "symbols", "cell", "unreadable"])
def test_same_geometry_is_false_for_a_different_system(tmp_path, change):
    from ase.io import write as ase_write

    from mliprun.core.ts_verify import _same_geometry

    ts = _structures()[0]
    stored = ts.copy()
    if change == "count":
        stored = Atoms("H2", positions=[CENTRE, CENTRE + 1], cell=ts.cell, pbc=False)
    elif change == "symbols":
        stored.set_chemical_symbols(["He"])
    elif change == "cell":
        stored.set_cell([11.0, 10.0, 10.0])
    path = tmp_path / "ts_structure.vasp"
    if change == "unreadable":
        path.write_text("garbage\n")
    else:
        ase_write(str(path), stored, format="vasp")
    assert _same_geometry(path, ts) is False


def test_the_same_geometry_keeps_the_directory_and_reuses_the_cache(tmp_path):
    first = _verify(tmp_path, checks=("overlap", "relax"))
    assert first["n_force_calls_frequencies"] > 0
    assert (tmp_path / "ts_relax_plus.traj").exists()

    second = _verify(tmp_path, checks=("overlap",))
    assert "archived_previous" not in second
    assert not list(tmp_path.glob("previous_*"))
    # The displacement cache was valid, so the sweep made no force call of its own.
    assert second["n_force_calls_frequencies"] == 0
    assert not list(tmp_path.glob("ts_relax_plus.*"))
    assert not list(tmp_path.glob("ts_relax_minus.*"))
    assert "Displace-and-relax" not in (tmp_path / "ts_verify.txt").read_text()


def test_a_failed_rerun_on_the_same_geometry_leaves_no_old_verdict(tmp_path):
    _verify(tmp_path, checks=("overlap", "relax"))
    assert (tmp_path / "ts_verify.txt").exists()
    with pytest.raises(RuntimeError, match="different settings"):
        _verify(tmp_path, delta=0.02)
    assert not (tmp_path / "ts_verify.txt").exists()
    assert not list(tmp_path.glob("ts_relax_*"))
