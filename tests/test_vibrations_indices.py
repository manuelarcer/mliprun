"""Which atoms get displaced, and what we say about constraints we cannot honour."""
import pytest
from ase.build import fcc111, molecule
from ase.constraints import FixAtoms, FixBondLengths, FixCartesian

from mliprun.core.vibrations import parse_indices, select_indices


def test_a_comma_list_parses():
    assert parse_indices("0,1,5", n_atoms=10) == [0, 1, 5]


def test_a_range_parses_inclusive():
    assert parse_indices("2-5", n_atoms=10) == [2, 3, 4, 5]


def test_ranges_and_singles_mix_and_sort_unique():
    assert parse_indices("7,2-4,2", n_atoms=10) == [2, 3, 4, 7]


def test_none_passes_through():
    assert parse_indices(None, n_atoms=10) is None


def test_an_out_of_range_index_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        parse_indices("0,12", n_atoms=10)


def test_a_malformed_range_is_rejected():
    with pytest.raises(ValueError, match="could not parse"):
        parse_indices("2-", n_atoms=10)


def test_fixatoms_drives_the_default_selection():
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    indices, unhandled = select_indices(atoms)
    assert set(indices) == set(range(12)) - set(bottom)
    assert len(indices) == 8
    assert unhandled == []


def test_an_unconstrained_structure_displaces_every_atom():
    atoms = molecule("H2O")
    indices, unhandled = select_indices(atoms)
    assert indices == [0, 1, 2]
    assert unhandled == []


def test_fixcartesian_is_named_as_unhandled_and_the_atom_still_moves():
    """ASE's indices select whole atoms, so a partial Hessian cannot be
    expressed. The atom is displaced in full and we say so (D6)."""
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    atoms.set_constraint(FixCartesian(0, mask=(True, True, False)))
    indices, unhandled = select_indices(atoms)
    assert 0 in indices
    assert unhandled == ["FixCartesian"]


def test_other_constraint_types_are_named_too():
    """FixBondLengths, not the deprecated singular FixBondLength factory
    function, which returns a FixBondLengths instance as of ase 3.29.0 --
    see tests/test_committee_stats.py for the same gotcha."""
    atoms = molecule("H2O")
    atoms.set_constraint(FixBondLengths([(0, 1)]))
    indices, unhandled = select_indices(atoms)
    assert unhandled == ["FixBondLengths"]


def test_an_explicit_selection_overrides_the_constraints():
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    indices, unhandled = select_indices(atoms, explicit=[0, 1])
    assert indices == [0, 1]


def test_an_explicit_selection_of_a_fixed_atom_is_honoured():
    """The user asked for it explicitly. Vibrations bypasses the constraint
    machinery when it collects forces, so the row is real, not zeros."""
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    indices, unhandled = select_indices(atoms, explicit=[bottom[0]])
    assert indices == [bottom[0]]
