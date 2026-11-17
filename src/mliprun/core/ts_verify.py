"""Pure geometry and verdict helpers for transition-state (TS) verification.

Nothing here runs a calculation. The functions take structures, vectors and
check results and return numbers, vectors or verdict strings.
"""

from __future__ import annotations

import numpy as np
from ase.geometry import find_mic

IMAGINARY_FLOOR_CM1 = 10.0

_ZERO_NORM = 1e-12


def _mic_displacement(from_atoms, to_atoms) -> np.ndarray:
    """Minimum-image displacement (N, 3) from one structure to another."""
    disp = to_atoms.get_positions() - from_atoms.get_positions()
    mic_disp, _ = find_mic(disp, from_atoms.cell, pbc=from_atoms.pbc)
    return mic_disp


def neb_tangent(images: list, index: int) -> np.ndarray:
    """Improved tangent (Henkelman and Jonsson, J. Chem. Phys. 113, 9978 (2000)).

    Returns an (N, 3) array that is not normalised. Raises ValueError for the
    first and last image, which have only one neighbour.
    """
    if index <= 0 or index >= len(images) - 1:
        raise ValueError(f"NEB tangent needs an interior image; index {index} of {len(images)} images has only one neighbour")

    t_plus = _mic_displacement(images[index], images[index + 1])
    t_minus = _mic_displacement(images[index - 1], images[index])

    e_prev = images[index - 1].get_potential_energy()
    e_here = images[index].get_potential_energy()
    e_next = images[index + 1].get_potential_energy()

    if e_next > e_here > e_prev:
        return t_plus
    if e_next < e_here < e_prev:
        return t_minus

    d_next = abs(e_next - e_here)
    d_prev = abs(e_prev - e_here)
    d_max, d_min = max(d_next, d_prev), min(d_next, d_prev)
    if e_next > e_prev:
        return t_plus * d_min + t_minus * d_max
    return t_plus * d_max + t_minus * d_min


def is_to_fs_direction(initial, final) -> np.ndarray:
    """Minimum-image displacement (N, 3) from the initial to the final state."""
    return _mic_displacement(initial, final)


def mode_overlap(mode, reference, indices) -> float:
    """Absolute cosine between two (N, 3) vectors restricted to ``indices``."""
    idx = list(indices)
    a = np.asarray(mode, dtype=float)[idx].ravel()
    b = np.asarray(reference, dtype=float)[idx].ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < _ZERO_NORM:
        raise ValueError("mode vector is zero on the selected atoms; overlap is undefined")
    if norm_b < _ZERO_NORM:
        raise ValueError("reference vector is zero on the selected atoms; overlap is undefined")
    return float(abs(np.dot(a / norm_a, b / norm_b)))


def scale_mode(mode, displacement: float) -> np.ndarray:
    """Scale an (N, 3) mode so the largest single-atom displacement equals ``displacement``."""
    mode = np.asarray(mode, dtype=float)
    largest = np.linalg.norm(mode, axis=1).max()
    if largest < _ZERO_NORM:
        raise ValueError("mode vector is zero; it cannot be scaled")
    return mode * (displacement / largest)


def endpoint_distance(atoms, endpoint) -> float:
    """Largest single-atom minimum-image distance (Angstrom) between two structures."""
    disp = endpoint.get_positions() - atoms.get_positions()
    _, dist = find_mic(disp, atoms.cell, pbc=atoms.pbc)
    return float(np.max(dist))


def classify_connection(plus: dict, minus: dict) -> str:
    """Combine the two relaxations from a TS into one result string.

    Each dict carries ``converged`` (bool) and ``matched`` ("initial", "final"
    or None).
    """
    if not (plus["converged"] and minus["converged"]):
        return "inconclusive"
    matched = {plus["matched"], minus["matched"]}
    if matched == {"initial", "final"}:
        return "connects"
    if plus["matched"] is not None and plus["matched"] == minus["matched"]:
        return "same_side"
    if None in matched:
        return "neither" if matched == {None} else "one_side"
    return "neither"  # unreachable with valid labels


def decide_verdict(n_imaginary, ts_fmax, ts_fmax_limit, check_statuses: dict) -> tuple[str, str]:
    """Return ``(verdict, reason)`` from the mode count, TS force and requested checks."""
    if n_imaginary != 1:
        if ts_fmax > ts_fmax_limit:
            return (
                "inconclusive",
                f"{n_imaginary} imaginary modes and the TS is not converged enough "
                f"(fmax {ts_fmax:.3f} > {ts_fmax_limit:.3f} eV/A); relax it further and repeat",
            )
        return (
            "not_verified",
            f"{n_imaginary} imaginary modes on a converged TS (fmax {ts_fmax:.3f} <= {ts_fmax_limit:.3f} eV/A); expected exactly 1",
        )

    failed = sorted(k for k, v in check_statuses.items() if v == "fail")
    if failed:
        return "not_verified", "one imaginary mode, but check failed: " + ", ".join(failed)
    unclear = sorted(k for k, v in check_statuses.items() if v == "inconclusive")
    if unclear:
        return "inconclusive", "one imaginary mode, but check inconclusive: " + ", ".join(unclear)
    return "verified", "one imaginary mode and every requested check passed"


def check_same_system(ts_atoms, initial, final) -> None:
    """Raise ValueError naming which of initial/final differs from the TS, and how."""
    ts_symbols = ts_atoms.get_chemical_symbols()
    for label, other in (("initial", initial), ("final", final)):
        if len(other) != len(ts_atoms):
            raise ValueError(f"{label} has {len(other)} atoms, the TS has {len(ts_atoms)}")
        symbols = other.get_chemical_symbols()
        if symbols != ts_symbols:
            first = next(i for i, (a, b) in enumerate(zip(symbols, ts_symbols)) if a != b)
            raise ValueError(
                f"{label} chemical symbols differ from the TS (first difference at atom {first}: {symbols[first]} vs {ts_symbols[first]})"
            )
        if not np.allclose(np.asarray(other.cell), np.asarray(ts_atoms.cell), rtol=0.0, atol=1e-6):
            raise ValueError(f"{label} cell differs from the TS cell by more than 1e-6 Angstrom")
