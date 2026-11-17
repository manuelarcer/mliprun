"""verify_ts on a one-atom double well, where every answer is known analytically."""

import json

import numpy as np
import pytest
from ase.calculators.emt import EMT
from ase.build import molecule

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


def test_the_relax_check_is_not_available_yet(tmp_path):
    calls = []

    def factory():
        calls.append(1)
        return DoubleWell()

    ts, initial, final = _structures()
    with pytest.raises(NotImplementedError):
        verify_ts(ts, initial, final, calculator_factory=factory,
                  checks=("overlap", "relax"), output_dir=tmp_path)
    assert calls == []


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
