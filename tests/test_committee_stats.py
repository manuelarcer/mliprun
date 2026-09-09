"""Committee arithmetic: mean force field and per-atom disagreement."""
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian

from mliprun.core.committee.calculator import (
    aligned_energy_spread,
    committee_statistics,
    free_component_mask,
)


class TestCommitteeStatistics:
    def test_three_members_give_a_hand_computed_mean_and_sigma(self):
        """1.0, 3.0, 2.0 eV/A on one component: mean 2.0, std (ddof=1) 1.0."""
        forces = np.zeros((3, 1, 3))
        forces[0, 0, 0] = 1.0
        forces[1, 0, 0] = 3.0
        forces[2, 0, 0] = 2.0
        stats = committee_statistics([-1.0, -3.0, -2.0], forces)

        assert stats["energy_mean"] == pytest.approx(-2.0, abs=1e-12)
        assert stats["forces_mean"][0, 0] == pytest.approx(2.0, abs=1e-12)
        assert stats["sigma_per_atom_all"][0] == pytest.approx(1.0, abs=1e-12)
        assert stats["sigma_max_free"] == pytest.approx(1.0, abs=1e-12)
        assert stats["sigma_mean_free"] == pytest.approx(1.0, abs=1e-12)
        assert stats["worst_atom_free"] == 0

    def test_sigma_is_the_norm_of_the_per_component_std(self):
        """sigma_i = || std_across_members(F_i) ||, so 3-4-5 on the three
        components gives exactly 5."""
        forces = np.zeros((2, 1, 3))
        forces[0, 0] = [0.0, 0.0, 0.0]
        forces[1, 0] = [3.0 * np.sqrt(2), 4.0 * np.sqrt(2), 0.0]
        stats = committee_statistics([0.0, 0.0], forces)
        assert stats["sigma_per_atom_all"][0] == pytest.approx(5.0, abs=1e-12)

    def test_identical_members_give_zero_sigma(self):
        forces = np.tile(np.array([[[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]]),
                         (4, 1, 1))
        stats = committee_statistics([-7.5] * 4, forces)
        assert stats["sigma_max_free"] == pytest.approx(0.0, abs=1e-12)
        assert stats["forces_mean"] == pytest.approx(forces[0], abs=1e-12)

    def test_worst_atom_is_the_index_of_the_largest_sigma(self):
        forces = np.zeros((2, 3, 3))
        forces[1, 0, 0] = 0.1
        forces[1, 2, 0] = 0.9
        stats = committee_statistics([0.0, 0.0], forces)
        assert stats["worst_atom_free"] == 2
        assert stats["sigma_max_free"] == pytest.approx(
            stats["sigma_per_atom_all"][2], abs=1e-12)

    def test_sigma_mean_averages_over_atoms_not_members(self):
        forces = np.zeros((2, 4, 3))
        forces[1, 0, 0] = 2.0 * np.sqrt(2)   # sigma = 2.0 on atom 0 only
        stats = committee_statistics([0.0, 0.0], forces)
        assert stats["sigma_mean_free"] == pytest.approx(0.5, abs=1e-12)

    def test_one_member_is_rejected(self):
        """A committee of one has no disagreement to report."""
        with pytest.raises(ValueError, match="at least two"):
            committee_statistics([-1.0], np.zeros((1, 2, 3)))

    def test_mismatched_member_counts_are_rejected(self):
        with pytest.raises(ValueError):
            committee_statistics([-1.0, -2.0], np.zeros((3, 2, 3)))

    def test_wrong_force_rank_is_rejected(self):
        with pytest.raises(ValueError):
            committee_statistics([-1.0, -2.0], np.zeros((2, 3)))


class TestAlignedEnergySpread:
    def test_constant_offsets_cancel_exactly(self):
        """Removing each member's own step-0 energy collapses a spread that
        is pure offset to zero."""
        baseline = {"member_a": -73.68, "member_b": -82.86}
        energies = {"member_a": -73.68 - 1.5, "member_b": -82.86 - 1.5}
        assert aligned_energy_spread(energies, baseline) == pytest.approx(
            0.0, abs=1e-12)

    def test_real_disagreement_survives_alignment(self):
        baseline = {"member_a": -10.0, "member_b": -20.0}
        energies = {"member_a": -11.0, "member_b": -22.0}
        # Deltas -1.0 and -2.0; std with ddof=1 is 1/sqrt(2).
        assert aligned_energy_spread(energies, baseline) == pytest.approx(
            1.0 / np.sqrt(2.0), abs=1e-12)

    def test_missing_baseline_member_is_rejected(self):
        with pytest.raises(KeyError):
            aligned_energy_spread({"member_a": -1.0, "member_b": -2.0}, {"member_b": -1.0})

    def test_one_member_is_rejected(self):
        """A committee of one has no disagreement to report."""
        with pytest.raises(ValueError, match="at least two"):
            aligned_energy_spread({"member_a": -1.0}, {"member_a": -1.0})


def _two_atoms_sigma_on_x(sigma_0, sigma_1):
    """Two members, disagreement on x only.

    For two samples, std(ddof=1) of {0, d} is d/sqrt(2), so a component set
    to sigma*sqrt(2) gives exactly sigma -- the same construction the
    existing 3-4-5 test uses.
    """
    forces = np.zeros((2, 2, 3))
    forces[1, 0, 0] = sigma_0 * np.sqrt(2)
    forces[1, 1, 0] = sigma_1 * np.sqrt(2)
    return forces


class TestFreeComponentMask:
    def test_fixatoms_frees_no_component_of_its_atoms(self):
        atoms = Atoms("H3", positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        atoms.set_constraint(FixAtoms(indices=[0, 2]))
        mask, unhandled = free_component_mask(atoms)
        assert mask[0].tolist() == [False, False, False]
        assert mask[1].tolist() == [True, True, True]
        assert mask[2].tolist() == [False, False, False]
        assert unhandled == []

    def test_fixcartesian_frees_the_directions_it_does_not_hold(self):
        atoms = Atoms("H2", positions=[(0, 0, 0), (1, 0, 0)])
        atoms.set_constraint(FixCartesian([0], mask=(False, False, True)))
        mask, unhandled = free_component_mask(atoms)
        assert mask[0].tolist() == [True, True, False]
        assert mask[1].tolist() == [True, True, True]
        assert unhandled == []

    def test_no_constraints_leaves_everything_free(self):
        atoms = Atoms("H2", positions=[(0, 0, 0), (1, 0, 0)])
        mask, unhandled = free_component_mask(atoms)
        assert mask.all()
        assert unhandled == []

    def test_an_unhandled_constraint_leaves_atoms_free_and_is_named(self):
        """Over-reporting sigma is the safe direction; the type name travels
        so the fallback is visible rather than silent.

        ``FixBondLengths`` (not the deprecated singular ``FixBondLength``
        factory function, which returns a ``FixBondLengths`` instance as of
        ase 3.29.0) is used directly so this test's expected class name does
        not depend on which ASE version resolves the alias."""
        from ase.constraints import FixBondLengths
        atoms = Atoms("H2", positions=[(0, 0, 0), (1, 0, 0)])
        atoms.set_constraint(FixBondLengths([(0, 1)]))
        mask, unhandled = free_component_mask(atoms)
        assert mask.all()
        assert unhandled == ["FixBondLengths"]


class TestConstrainedComponentsAreMasked:
    def test_a_fixed_atom_is_excluded_from_sigma_max(self):
        """Atom 0 disagrees ten times more, but cannot move."""
        mask = np.array([[False, False, False], [True, True, True]])
        stats = committee_statistics([0.0, 0.0],
                                     _two_atoms_sigma_on_x(0.9, 0.1),
                                     free_mask=mask)
        assert stats["sigma_max_free"] == pytest.approx(0.1, abs=1e-12)
        assert stats["worst_atom_free"] == 1
        assert stats["n_free_atoms"] == 1

    def test_the_unmasked_numbers_are_kept_alongside(self):
        mask = np.array([[False, False, False], [True, True, True]])
        stats = committee_statistics([0.0, 0.0],
                                     _two_atoms_sigma_on_x(0.9, 0.1),
                                     free_mask=mask)
        assert stats["sigma_max_all"] == pytest.approx(0.9, abs=1e-12)
        assert stats["worst_atom_all"] == 0
        assert stats["sigma_mean_all"] == pytest.approx(0.5, abs=1e-12)

    def test_a_fixed_atom_does_not_dilute_sigma_mean(self):
        """The mean is over free atoms, not over free atoms plus zeros."""
        mask = np.array([[False, False, False], [True, True, True]])
        stats = committee_statistics([0.0, 0.0],
                                     _two_atoms_sigma_on_x(0.9, 0.1),
                                     free_mask=mask)
        assert stats["sigma_mean_free"] == pytest.approx(0.1, abs=1e-12)

    def test_a_partly_fixed_atom_keeps_its_free_components(self):
        """3 on x and 4 on z, z held: sigma is 3, not 5."""
        forces = np.zeros((2, 1, 3))
        forces[1, 0] = [3.0 * np.sqrt(2), 0.0, 4.0 * np.sqrt(2)]
        mask = np.array([[True, True, False]])
        stats = committee_statistics([0.0, 0.0], forces, free_mask=mask)
        assert stats["sigma_max_free"] == pytest.approx(3.0, abs=1e-12)
        assert stats["sigma_max_all"] == pytest.approx(5.0, abs=1e-12)

    def test_no_mask_reproduces_the_previous_numbers(self):
        stats = committee_statistics([0.0, 0.0],
                                     _two_atoms_sigma_on_x(0.9, 0.1))
        assert stats["sigma_max_free"] == pytest.approx(0.9, abs=1e-12)
        assert stats["sigma_max_free"] == pytest.approx(stats["sigma_max_all"])
        assert stats["worst_atom_free"] == 0
        assert stats["n_free_atoms"] == 2

    def test_every_component_constrained_falls_back_to_the_unmasked_numbers(self):
        """Possible for a constrained single point, not for a relaxation.
        Reporting nothing would be less useful than reporting the unmasked
        numbers and saying so."""
        mask = np.zeros((2, 3), dtype=bool)
        stats = committee_statistics([0.0, 0.0],
                                     _two_atoms_sigma_on_x(0.9, 0.1),
                                     free_mask=mask)
        assert stats["all_constrained"] is True
        assert stats["n_free_atoms"] == 0
        assert stats["sigma_max_free"] == pytest.approx(0.9, abs=1e-12)

    def test_a_mask_of_the_wrong_shape_is_rejected(self):
        with pytest.raises(ValueError, match="free_mask"):
            committee_statistics([0.0, 0.0], _two_atoms_sigma_on_x(0.9, 0.1),
                                 free_mask=np.ones((3, 3), dtype=bool))
