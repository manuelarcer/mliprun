"""Unit tests for the pure geometry and verdict helpers of TS verification."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mliprun.core.ts_verify import (
    check_same_system,
    classify_connection,
    decide_verdict,
    endpoint_distance,
    is_to_fs_direction,
    mode_overlap,
    neb_tangent,
    scale_mode,
)


def _image(x, energy, cell=None, pbc=False):
    atoms = Atoms("H", positions=[[x, 0.0, 0.0]], cell=cell, pbc=pbc)
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    return atoms


def test_overlap_of_parallel_and_antiparallel_vectors_is_one():
    v = np.array([[1.0, 0, 0], [0, 2.0, 0]])
    assert mode_overlap(v, -3 * v, [0, 1]) == pytest.approx(1.0)


def test_overlap_of_orthogonal_vectors_is_zero():
    assert mode_overlap(np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]), [0]) == pytest.approx(0.0)


def test_overlap_ignores_atoms_outside_the_indices():
    mode = np.array([[1.0, 0, 0], [0, 5.0, 0]])
    ref = np.array([[1.0, 0, 0], [0, 0, 9.0]])
    assert mode_overlap(mode, ref, [0]) == pytest.approx(1.0)


def test_a_zero_reference_is_an_error_not_nan():
    with pytest.raises(ValueError, match="zero"):
        mode_overlap(np.ones((1, 3)), np.zeros((1, 3)), [0])


def test_scale_mode_sets_the_largest_atom_displacement():
    scaled = scale_mode(np.array([[3.0, 4.0, 0], [1.0, 0, 0]]), 0.1)
    assert np.linalg.norm(scaled, axis=1).max() == pytest.approx(0.1)
    assert np.linalg.norm(scaled[1]) == pytest.approx(0.02)


def test_endpoint_distance_uses_the_minimum_image():
    # one atom at x=0.1 and one at x=9.9 in a 10 A periodic cell are 0.2 A apart
    a = Atoms("H", positions=[[0.1, 0, 0]], cell=[10, 10, 10], pbc=True)
    b = Atoms("H", positions=[[9.9, 0, 0]], cell=[10, 10, 10], pbc=True)
    assert endpoint_distance(a, b) == pytest.approx(0.2)


def test_is_to_fs_direction_uses_the_minimum_image():
    a = Atoms("H", positions=[[0.1, 0, 0]], cell=[10, 10, 10], pbc=True)
    b = Atoms("H", positions=[[9.9, 0, 0]], cell=[10, 10, 10], pbc=True)
    assert is_to_fs_direction(a, b) == pytest.approx(np.array([[-0.2, 0, 0]]))


def test_tangent_at_a_symmetric_maximum_is_the_sum_of_both_segments():
    images = [_image(0.0, 0.0), _image(1.0, 1.0), _image(2.0, 0.0)]
    tangent = neb_tangent(images, 1)
    assert tangent / np.linalg.norm(tangent) == pytest.approx(np.array([[1.0, 0, 0]]))


def test_tangent_on_an_uphill_segment_is_the_forward_segment():
    images = [_image(0.0, 0.0), _image(1.0, 1.0), _image(3.0, 2.0)]
    assert neb_tangent(images, 1) == pytest.approx(np.array([[2.0, 0, 0]]))


def test_tangent_on_a_downhill_segment_is_the_backward_segment():
    images = [_image(0.0, 2.0), _image(1.0, 1.0), _image(3.0, 0.0)]
    assert neb_tangent(images, 1) == pytest.approx(np.array([[1.0, 0, 0]]))


def test_tangent_at_an_asymmetric_maximum_is_weighted_toward_the_lower_neighbour():
    # E(i+1) = 0.5 > E(i-1) = 0 : tangent = t_plus * dE_min + t_minus * dE_max
    images = [_image(0.0, 0.0), _image(1.0, 1.0), _image(3.0, 0.5)]
    # t_plus = 2, t_minus = 1, dE_max = 1.0, dE_min = 0.5
    assert neb_tangent(images, 1) == pytest.approx(np.array([[2 * 0.5 + 1 * 1.0, 0, 0]]))


def test_tangent_at_an_asymmetric_maximum_with_higher_backward_neighbour():
    images = [_image(0.0, 0.5), _image(1.0, 1.0), _image(3.0, 0.0)]
    # E(i+1) = 0 < E(i-1) = 0.5 : tangent = t_plus * dE_max + t_minus * dE_min
    # t_plus = 2, t_minus = 1, dE_max = 1.0, dE_min = 0.5
    assert neb_tangent(images, 1) == pytest.approx(np.array([[2 * 1.0 + 1 * 0.5, 0, 0]]))


def test_tangent_uses_the_minimum_image():
    cell = [10, 10, 10]
    images = [
        _image(9.5, 0.0, cell, True),
        _image(0.5, 1.0, cell, True),
        _image(1.5, 0.0, cell, True),
    ]
    tangent = neb_tangent(images, 1)
    assert tangent / np.linalg.norm(tangent) == pytest.approx(np.array([[1.0, 0, 0]]))


@pytest.mark.parametrize("index", [0, 2])
def test_tangent_at_an_endpoint_is_an_error(index):
    images = [_image(0.0, 0.0), _image(1.0, 1.0), _image(2.0, 0.0)]
    with pytest.raises(ValueError):
        neb_tangent(images, index)


@pytest.mark.parametrize(
    "plus, minus, expected",
    [
        (("initial", True), ("final", True), "connects"),
        (("final", True), ("initial", True), "connects"),
        (("initial", True), ("initial", True), "same_side"),
        (("initial", True), (None, True), "one_side"),
        ((None, True), (None, True), "neither"),
        (("initial", True), ("final", False), "inconclusive"),
    ],
)
def test_classify_connection(plus, minus, expected):
    def as_result(pair):
        return {"matched": pair[0], "converged": pair[1]}

    assert classify_connection(as_result(plus), as_result(minus)) == expected


@pytest.mark.parametrize(
    "n_imag, fmax, statuses, expected",
    [
        (1, 0.01, {"overlap": "pass"}, "verified"),
        (1, 0.10, {"overlap": "pass"}, "verified"),  # one mode: fmax does not block
        (1, 0.01, {"overlap": "pass", "relax": "fail"}, "not_verified"),
        (1, 0.01, {"overlap": "pass", "relax": "inconclusive"}, "inconclusive"),
        (0, 0.01, {"overlap": "not_run"}, "not_verified"),
        (2, 0.01, {"overlap": "not_run"}, "not_verified"),
        (2, 0.031, {"overlap": "not_run"}, "inconclusive"),  # D9, limit 0.03
        (0, 0.031, {"overlap": "not_run"}, "inconclusive"),
    ],
)
def test_decide_verdict(n_imag, fmax, statuses, expected):
    assert decide_verdict(n_imag, fmax, 0.03, statuses)[0] == expected


def test_the_d9_reason_names_convergence():
    assert "not converged enough" in decide_verdict(2, 0.05, 0.03, {"overlap": "not_run"})[1]


def _ts(symbols="HO", cell=(10, 10, 10)):
    return Atoms(symbols, positions=[[0, 0, 0], [0, 0, 1]][: len(symbols)], cell=cell)


def test_check_same_system_accepts_matching_structures():
    check_same_system(_ts(), _ts(), _ts())


def test_check_same_system_names_a_different_atom_count():
    with pytest.raises(ValueError, match=r"initial.*(2|1)"):
        check_same_system(_ts(), _ts("H"), _ts())


def test_check_same_system_names_a_different_symbol_order():
    with pytest.raises(ValueError, match=r"final.*symbol"):
        check_same_system(_ts("HO"), _ts("HO"), _ts("OH"))


def test_check_same_system_names_a_different_cell():
    with pytest.raises(ValueError, match=r"initial.*cell"):
        check_same_system(_ts(), _ts(cell=(10, 10, 11)), _ts())
