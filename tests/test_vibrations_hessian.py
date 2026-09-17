"""Our Hessian assembly against ASE's own, by exact equality.

ASE builds the Hessian inside Vibrations.read from its displacement cache.
A committee needs one Hessian per member, which that path cannot express, so
the assembly is reimplemented here -- and pinned to ASE's by exact equality
rather than a tolerance, because the two run the same arithmetic on the same
numbers and any difference at all is a transcription error.
"""
import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations

from mliprun.core.vibrations import assemble_hessian


def _run_ase(tmp_path, nfree=2, delta=0.01):
    atoms = molecule("N2")
    atoms.calc = EMT()
    vib = Vibrations(atoms, name=str(tmp_path / "vib"),
                     delta=delta, nfree=nfree)
    vib.run()
    return atoms, vib


def _cached_forces(vib, nfree):
    """Pull every displacement's forces back out of ASE's cache.

    ``vib._disp`` and ``vib._eq_disp`` are ASE-private. They are used
    deliberately: a restarted run skips displacements already on disk, so a
    calculator-side hook would miss them, and the cache is the only complete
    record. Pinned by this test file against ase>=3.23.
    """
    forces = {"eq": np.asarray(vib._eq_disp().forces())}
    steps = [-1, 1] if nfree == 2 else [-2, -1, 1, 2]
    for a in vib.indices:
        for i in range(3):
            for n in steps:
                forces[(int(a), i, n)] = np.asarray(
                    vib._disp(a, i, n).forces())
    return forces


@pytest.mark.parametrize("nfree", [2, 4], ids=["nfree_2", "nfree_4"])
def test_central_differences_match_ase_exactly(tmp_path, nfree):
    atoms, vib = _run_ase(tmp_path, nfree=nfree)
    vib.read(method="standard", direction="central")
    ours = assemble_hessian(
        _cached_forces(vib, nfree), vib.indices, vib.delta,
        nfree=nfree, direction="central", method="standard")
    assert np.array_equal(ours, vib.H)


@pytest.mark.parametrize("direction", ["forward", "backward"],
                         ids=["dir_forward", "dir_backward"])
def test_one_sided_differences_match_ase_exactly(tmp_path, direction):
    atoms, vib = _run_ase(tmp_path)
    vib.read(method="standard", direction=direction)
    ours = assemble_hessian(
        _cached_forces(vib, 2), vib.indices, vib.delta,
        nfree=2, direction=direction, method="standard")
    assert np.array_equal(ours, vib.H)


def test_frederiksen_matches_ase_exactly(tmp_path):
    atoms, vib = _run_ase(tmp_path)
    vib.read(method="frederiksen", direction="central")
    ours = assemble_hessian(
        _cached_forces(vib, 2), vib.indices, vib.delta,
        nfree=2, direction="central", method="frederiksen")
    assert np.array_equal(ours, vib.H)


def test_the_hessian_is_symmetric(tmp_path):
    atoms, vib = _run_ase(tmp_path)
    vib.read()
    ours = assemble_hessian(_cached_forces(vib, 2), vib.indices, vib.delta)
    assert np.array_equal(ours, ours.T)


def test_the_hessian_is_square_in_three_times_the_displaced_atoms(tmp_path):
    atoms, vib = _run_ase(tmp_path)
    vib.read()
    ours = assemble_hessian(_cached_forces(vib, 2), vib.indices, vib.delta)
    assert ours.shape == (3 * len(vib.indices), 3 * len(vib.indices))


def test_an_unknown_nfree_is_rejected(tmp_path):
    atoms, vib = _run_ase(tmp_path)
    vib.read()
    with pytest.raises(ValueError, match="nfree"):
        assemble_hessian(_cached_forces(vib, 2), vib.indices, vib.delta,
                         nfree=3)
