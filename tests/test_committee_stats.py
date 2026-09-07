"""Committee arithmetic: mean force field and per-atom disagreement."""
import numpy as np
import pytest

from mliprun.core.committee.calculator import (
    aligned_energy_spread,
    committee_statistics,
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
        assert stats["sigma_per_atom"][0] == pytest.approx(1.0, abs=1e-12)
        assert stats["sigma_max"] == pytest.approx(1.0, abs=1e-12)
        assert stats["sigma_mean"] == pytest.approx(1.0, abs=1e-12)
        assert stats["worst_atom"] == 0

    def test_sigma_is_the_norm_of_the_per_component_std(self):
        """sigma_i = || std_across_members(F_i) ||, so 3-4-5 on the three
        components gives exactly 5."""
        forces = np.zeros((2, 1, 3))
        forces[0, 0] = [0.0, 0.0, 0.0]
        forces[1, 0] = [3.0 * np.sqrt(2), 4.0 * np.sqrt(2), 0.0]
        stats = committee_statistics([0.0, 0.0], forces)
        assert stats["sigma_per_atom"][0] == pytest.approx(5.0, abs=1e-12)

    def test_identical_members_give_zero_sigma(self):
        forces = np.tile(np.array([[[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]]),
                         (4, 1, 1))
        stats = committee_statistics([-7.5] * 4, forces)
        assert stats["sigma_max"] == pytest.approx(0.0, abs=1e-12)
        assert stats["forces_mean"] == pytest.approx(forces[0], abs=1e-12)

    def test_worst_atom_is_the_index_of_the_largest_sigma(self):
        forces = np.zeros((2, 3, 3))
        forces[1, 0, 0] = 0.1
        forces[1, 2, 0] = 0.9
        stats = committee_statistics([0.0, 0.0], forces)
        assert stats["worst_atom"] == 2
        assert stats["sigma_max"] == pytest.approx(
            stats["sigma_per_atom"][2], abs=1e-12)

    def test_sigma_mean_averages_over_atoms_not_members(self):
        forces = np.zeros((2, 4, 3))
        forces[1, 0, 0] = 2.0 * np.sqrt(2)   # sigma = 2.0 on atom 0 only
        stats = committee_statistics([0.0, 0.0], forces)
        assert stats["sigma_mean"] == pytest.approx(0.5, abs=1e-12)

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
            aligned_energy_spread({"member_a": -1.0}, {"member_b": -1.0})
