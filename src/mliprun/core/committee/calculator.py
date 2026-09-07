"""Committee calculator: mean forces and per-configuration uncertainty.

Driver-side only. Nothing here runs inside an MLIP env.

Design: docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def committee_statistics(energies, forces) -> dict:
    """Reduce the members' energies and forces to a consensus and a spread.

    The uncertainty metric is the per-atom force disagreement

        sigma_i = || std_across_members( F_i ) ||

    -- the vector norm of the per-component standard deviation across
    members, ``ddof=1``. It behaves as an extrapolation detector: in the
    2026-09-04 probe it sat at 0.25-0.62 eV/A across the physical range of a
    CO height scan and rose to 3.48 eV/A at a strained geometry.

    The committee is a *proper potential*: the mean force is exactly the
    negative gradient of the mean energy, so the effective PES is well
    defined and line-search optimizers (``bfgsls``) behave correctly through
    it. Per-model constant offsets shift the mean energy by a constant
    without changing its shape.

    Parameters
    ----------
    energies : array-like, shape (M,)
        One energy per member, in the members' own order.
    forces : array-like, shape (M, N, 3)
        One force array per member, same order.

    Returns
    -------
    dict
        ``energy_mean`` (float), ``forces_mean`` (N, 3), ``sigma_per_atom``
        (N,), ``sigma_max`` (float), ``sigma_mean`` (float), ``worst_atom``
        (int).

    Raises
    ------
    ValueError
        If fewer than two members are given, or the shapes disagree. Two is
        the floor because ``ddof=1`` is undefined for one sample -- and
        because a committee of one has no disagreement to report.
    """
    energies = np.asarray(energies, dtype=float)
    forces = np.asarray(forces, dtype=float)
    if energies.ndim != 1:
        raise ValueError(f"energies must be 1-D, got shape {energies.shape}")
    if forces.ndim != 3 or forces.shape[2] != 3:
        raise ValueError(
            f"forces must have shape (n_members, n_atoms, 3), got {forces.shape}")
    if forces.shape[0] != energies.shape[0]:
        raise ValueError(
            f"{energies.shape[0]} energies but {forces.shape[0]} force arrays")
    if energies.shape[0] < 2:
        raise ValueError("a committee needs at least two members")

    sigma_components = forces.std(axis=0, ddof=1)          # (N, 3)
    sigma_per_atom = np.linalg.norm(sigma_components, axis=1)   # (N,)
    worst_atom = int(np.argmax(sigma_per_atom))
    return {
        "energy_mean": float(energies.mean()),
        "forces_mean": forces.mean(axis=0),
        "sigma_per_atom": sigma_per_atom,
        "sigma_max": float(sigma_per_atom[worst_atom]),
        "sigma_mean": float(sigma_per_atom.mean()),
        "worst_atom": worst_atom,
    }


def aligned_energy_spread(energies: dict, baseline: dict) -> float:
    """Spread of the members' energies after removing their own offsets.

    Raw energies are not comparable across packages -- the four-member probe
    saw a 4.81 eV spread at one fixed geometry, essentially all of it
    per-model constant offset. Subtracting each member's own step-0 energy
    collapsed that to 0.09 eV. Composition is fixed during a relaxation, so
    the offset cancels exactly and what remains is a real energy uncertainty.

    Returned as a standard deviation across members (``ddof=1``), matching
    the force metric -- not a max-minus-min range.

    Raises
    ------
    KeyError
        If a member present in ``energies`` has no baseline.
    """
    deltas = [energies[name] - baseline[name] for name in energies]
    return float(np.std(np.asarray(deltas, dtype=float), ddof=1))
