"""The optional ``observer`` hooks on the four core engines (plan section 9).

Each hook must report exactly what the engine already logs (same count as
the CSV rows, same values), and an exception raised by the observer must
stop the run with the record marked ``failed``. EMT only; no fastapi.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from ase.build import add_adsorbate, bulk, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from mliprun.core.md import run_md
from mliprun.core.neb import CustomNEB
from mliprun.core.optimize import run_optimization
from mliprun.core.vibrations import run_frequencies


class Boom(RuntimeError):
    pass


def _raise_at(n):
    calls = []

    def observer(payload):
        calls.append(payload)
        if len(calls) == n:
            raise Boom(f"observer stop at call {n}")
    return observer


def _record_status(path):
    return json.loads((path / "mliprun_run.json").read_text())["status"]


def _rattled_cu():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 1, 1)
    atoms.rattle(0.05, seed=1)
    atoms.calc = EMT()
    return atoms


def _neb_pair():
    slab = fcc100("Cu", (2, 2, 3), vacuum=6.0)
    slab.set_constraint(FixAtoms(indices=[a.index for a in slab if a.tag == 3]))
    initial = slab.copy()
    add_adsorbate(initial, "Cu", 1.7, "hollow")
    final = slab.copy()
    add_adsorbate(final, "Cu", 1.7, "hollow")
    final.positions[-1, 0] += slab.cell[0, 0] / 2
    return initial, final


def _neb(tmp_path):
    initial, final = _neb_pair()
    neb = CustomNEB(initial, final, num_images=3, fmax=0.1, mlip="emt",
                    output_dir=tmp_path, interp_steps=50)
    neb.setup_calculator = lambda *a, **k: EMT()
    neb.interpolate_idpp()
    return neb


# --- (a) the observer reports what the engine logs -------------------------

def test_optimize_observer_matches_convergence_csv(tmp_path):
    seen = []
    atoms = _rattled_cu()
    converged = run_optimization(atoms, fmax=0.05, output_dir=tmp_path,
                                 observer=seen.append)
    assert converged
    df = pd.read_csv(tmp_path / "opt_convergence.csv")
    assert len(seen) == len(df) > 2
    np.testing.assert_allclose([s["step"] for s in seen], df["step"], atol=0)
    np.testing.assert_allclose([s["energy"] for s in seen], df["energy(eV)"],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose([s["fmax"] for s in seen], df["fmax(eV/A)"],
                               rtol=0, atol=1e-10)
    assert seen[0]["positions"].shape == (len(atoms), 3)
    assert seen[0]["cell"].shape == (3, 3)
    # Copies, not views: the first frame differs from the final geometry.
    assert not np.allclose(seen[0]["positions"], atoms.get_positions())
    np.testing.assert_allclose(seen[-1]["positions"], atoms.get_positions(), atol=0)


def test_optimize_observer_does_not_change_numerics(tmp_path):
    a1, a2 = _rattled_cu(), _rattled_cu()
    run_optimization(a1, fmax=0.02, output_dir=tmp_path / "a")
    run_optimization(a2, fmax=0.02, output_dir=tmp_path / "b",
                     observer=lambda p: None)
    np.testing.assert_array_equal(a1.get_positions(), a2.get_positions())
    assert a1.get_potential_energy() == a2.get_potential_energy()


def test_md_observer_matches_energy_csv(tmp_path):
    seen = []
    atoms = _rattled_cu()
    run_md(atoms, ensemble="nve", steps=20, log_interval=5, traj_interval=10,
           output_dir=tmp_path, observer=seen.append)
    df = pd.read_csv(tmp_path / "md_energy.csv")
    assert len(seen) == len(df) == 5
    for column in df.columns:
        np.testing.assert_allclose([s["row"][column] for s in seen], df[column],
                                   rtol=0, atol=1e-10)
    assert set(seen[0]["row"]) == set(df.columns)
    assert seen[-1]["positions"].shape == (len(atoms), 3)


def test_md_observer_npt_row_has_pressure_and_volume(tmp_path):
    seen = []
    run_md(_rattled_cu(), ensemble="npt", barostat="berendsen", steps=4,
           log_interval=2, output_dir=tmp_path, observer=seen.append)
    df = pd.read_csv(tmp_path / "md_energy.csv")
    assert len(seen) == len(df) == 3
    np.testing.assert_allclose([s["row"]["volume(A^3)"] for s in seen],
                               df["volume(A^3)"], rtol=0, atol=1e-10)
    np.testing.assert_allclose([s["row"]["pressure(GPa)"] for s in seen],
                               df["pressure(GPa)"], rtol=0, atol=1e-10)


def test_freq_observer_counts_every_force_call(tmp_path):
    seen = []
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = EMT()
    res = run_frequencies(atoms, output_dir=tmp_path, indices=[0, 1],
                          observer=seen.append)
    assert [s["done"] for s in seen] == list(range(1, len(seen) + 1))
    assert seen[-1]["done"] == res["n_force_calls"] == seen[-1]["total"] == 1 + 6 * 2


def test_freq_observer_nfree4_total(tmp_path):
    seen = []
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = EMT()
    res = run_frequencies(atoms, output_dir=tmp_path, indices=[0], nfree=4,
                          observer=seen.append)
    assert seen[-1]["done"] == res["n_force_calls"] == seen[-1]["total"] == 1 + 12


def test_freq_cached_rerun_reports_nothing(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = EMT()
    run_frequencies(atoms, output_dir=tmp_path, indices=[0])
    seen = []
    res = run_frequencies(atoms, output_dir=tmp_path, indices=[0], observer=seen.append)
    assert res["n_force_calls"] == 0
    assert seen == []


def test_neb_observer_matches_convergence_csv(tmp_path):
    seen = []
    neb = _neb(tmp_path)
    neb.run_neb(climb=True, max_steps=200, observer=seen.append)
    df = pd.read_csv(tmp_path / "neb_convergence.csv")
    assert len(seen) == len(df) > 1
    np.testing.assert_allclose([s["step"] for s in seen], df["step"], atol=0)
    np.testing.assert_allclose([s["fmax"] for s in seen], df["fmax(eV/A)"],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose([s["barrier"] for s in seen], df["barrier(eV)"],
                               rtol=0, atol=1e-10)
    last = seen[-1]
    assert len(last["energies"]) == len(last["positions"]) == 5
    assert last["barrier"] == pytest.approx(max(last["energies"]) - last["energies"][0],
                                            abs=1e-12)
    final = [img.get_potential_energy() for img in neb.images]
    np.testing.assert_allclose(last["energies"], final, rtol=0, atol=1e-10)


# --- (b) an observer exception fails the run -------------------------------

def test_optimize_observer_exception_marks_record_failed(tmp_path):
    with pytest.raises(Boom):
        run_optimization(_rattled_cu(), fmax=0.001, output_dir=tmp_path,
                         observer=_raise_at(3))
    assert _record_status(tmp_path) == "failed"


def test_md_observer_exception_marks_record_failed(tmp_path):
    with pytest.raises(Boom):
        run_md(_rattled_cu(), ensemble="nve", steps=50, log_interval=1,
               output_dir=tmp_path, observer=_raise_at(3))
    assert _record_status(tmp_path) == "failed"


def test_freq_observer_exception_marks_record_failed(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = EMT()
    with pytest.raises(Boom):
        run_frequencies(atoms, output_dir=tmp_path, indices=[0], observer=_raise_at(3))
    assert _record_status(tmp_path) == "failed"


def test_neb_observer_exception_marks_record_failed(tmp_path):
    neb = _neb(tmp_path)
    with pytest.raises(Boom):
        neb.run_neb(climb=True, max_steps=200, observer=_raise_at(3))
    assert _record_status(tmp_path) == "failed"
