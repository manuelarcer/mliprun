"""Vibrational frequencies by finite differences of forces.

Wraps ``ase.vibrations.Vibrations`` rather than reimplementing the
displacement sweep. The one piece that is reimplemented is the Hessian
assembly, because a committee needs one Hessian per member and ASE's
``Vibrations.read`` can only build the single Hessian implied by whatever
``atoms.calc`` returned. ``assemble_hessian`` is pinned to ASE's arithmetic
by exact equality in tests/test_vibrations_hessian.py.

Design: docs/superpowers/specs/2026-09-17-singlepoint-and-frequencies-design.md
"""
import numpy as np

VALID_NFREE = (2, 4)
VALID_DIRECTIONS = ("central", "forward", "backward")
VALID_METHODS = ("standard", "frederiksen")


def assemble_hessian(forces, indices, delta, nfree=2,
                     direction="central", method="standard"):
    """Build the Hessian from one displacement sweep's forces.

    Mirrors ``ase.vibrations.Vibrations.read`` exactly. Kept separate so it
    can be applied to any set of force arrays -- in particular to one
    committee member's own forces, which never reach ``atoms.calc``.

    Parameters
    ----------
    forces : dict
        ``(atom_index, cartesian_index, ndisp) -> (N, 3) array``, with
        ``ndisp`` in ``-2, -1, 1, 2``, plus the key ``"eq"`` holding the
        undisplaced geometry's forces. ``"eq"`` is read only for the
        one-sided directions.
    indices : sequence of int
        The displaced atoms, in the order the Hessian rows follow.
    delta : float
        Displacement in Angstrom.
    nfree : int
        2 (three-point) or 4 (five-point stencil).
    direction : str
        ``'central'``, ``'forward'`` or ``'backward'``.
    method : str
        ``'standard'`` or ``'frederiksen'`` (acoustic sum-rule correction).

    Returns
    -------
    numpy.ndarray
        The symmetrized ``(3n, 3n)`` Hessian, n being ``len(indices)``.

    Raises
    ------
    ValueError
        On an unknown ``nfree``, ``direction`` or ``method``.
    """
    if nfree not in VALID_NFREE:
        raise ValueError(f"nfree must be one of {VALID_NFREE}, got {nfree}")
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {VALID_DIRECTIONS}, got {direction!r}")
    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {VALID_METHODS}, got {method!r}")

    indices = np.asarray(indices, dtype=int)
    n = 3 * len(indices)
    hessian = np.empty((n, n))
    row = 0

    if direction != "central":
        feq = np.asarray(forces["eq"], dtype=float)

    for atom in indices:
        for cartesian in range(3):
            # np.array copies: the Frederiksen correction below mutates, and
            # the caller's cached arrays must survive being read twice.
            fminus = np.array(forces[(int(atom), cartesian, -1)], dtype=float)
            fplus = np.array(forces[(int(atom), cartesian, 1)], dtype=float)
            if method == "frederiksen":
                fminus[atom] -= fminus.sum(0)
                fplus[atom] -= fplus.sum(0)
            if nfree == 4:
                fmm = np.array(forces[(int(atom), cartesian, -2)], dtype=float)
                fpp = np.array(forces[(int(atom), cartesian, 2)], dtype=float)
                if method == "frederiksen":
                    fmm[atom] -= fmm.sum(0)
                    fpp[atom] -= fpp.sum(0)

            if direction == "central":
                if nfree == 2:
                    hessian[row] = 0.5 * (fminus - fplus)[indices].ravel()
                else:
                    hessian[row] = (
                        -fmm + 8 * fminus - 8 * fplus + fpp
                    )[indices].ravel() / 12.0
            elif direction == "forward":
                hessian[row] = (feq - fplus)[indices].ravel()
            else:
                hessian[row] = (fminus - feq)[indices].ravel()

            hessian[row] /= 2 * delta
            row += 1

    hessian += hessian.copy().T
    return hessian
