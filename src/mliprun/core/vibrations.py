"""Vibrational frequencies by finite differences of forces.

Wraps ``ase.vibrations.Vibrations`` rather than reimplementing the
displacement sweep. The one piece that is reimplemented is the Hessian
assembly, because a committee needs one Hessian per member and ASE's
``Vibrations.read`` can only build the single Hessian implied by whatever
``atoms.calc`` returned. ``assemble_hessian`` is pinned to ASE's arithmetic
by exact equality in tests/test_vibrations_hessian.py.

Design: docs/superpowers/specs/2026-09-17-singlepoint-and-frequencies-design.md
"""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

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


def parse_indices(text, n_atoms):
    """Turn ``'0,1,5'`` / ``'12-30'`` / a mix of both into a sorted list.

    Ranges are inclusive at both ends, which is what a user writing
    ``12-30`` for "layers 12 through 30" means. Duplicates collapse.

    Parameters
    ----------
    text : str or None
        The option value. None returns None, meaning "no explicit
        selection", which :func:`select_indices` reads as "use the
        constraints".
    n_atoms : int
        For the range check.

    Returns
    -------
    list of int, or None

    Raises
    ------
    ValueError
        On an unparsable token or an index outside ``0..n_atoms-1``.
    """
    if text is None:
        return None
    chosen = set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token.lstrip("-"):
            lo_text, _, hi_text = token.partition("-")
            try:
                lo, hi = int(lo_text), int(hi_text)
            except ValueError:
                raise ValueError(
                    f"could not parse '{token}' as an index range") from None
            if hi < lo:
                raise ValueError(
                    f"could not parse '{token}': range ends below its start")
            chosen.update(range(lo, hi + 1))
        else:
            try:
                chosen.add(int(token))
            except ValueError:
                raise ValueError(
                    f"could not parse '{token}' as an atom index") from None
    out_of_range = sorted(i for i in chosen if not 0 <= i < n_atoms)
    if out_of_range:
        raise ValueError(
            f"atom index out of range for a {n_atoms}-atom structure: "
            f"{out_of_range}")
    return sorted(chosen)


def select_indices(atoms, explicit=None):
    """Which atoms to displace, and which constraints we could not honour.

    The default comes from the structure's constraints, which is also ASE's
    own default: every atom not held by ``FixAtoms``. An explicit selection
    overrides it entirely, including selecting a fixed atom -- ASE's
    ``Vibrations`` collects forces through ``calc.get_forces(atoms)``, which
    bypasses the constraint machinery, so such a row carries real forces
    rather than zeros.

    Any constraint type other than ``FixAtoms`` cannot be honoured here:
    ASE's ``indices`` selects whole atoms, so a partially held atom has no
    partial Hessian to express. Those atoms are displaced in full and the
    type names are returned, so the caller can say so (D6 in the design
    note). They are never a refusal.

    This deliberately does *not* reuse
    :func:`mliprun.core.committee.calculator.free_component_mask`'s
    ``unhandled`` list, even though that function also scans ``atoms``'
    constraints: that function additionally masks ``FixCartesian``, because a
    per-component mask is exactly what its force-averaging statistic needs,
    and its own tests pin ``unhandled == []`` for a ``FixCartesian``-only
    structure (``tests/test_committee_stats.py::
    TestFreeComponentMask::test_fixcartesian_frees_the_directions_it_does_not_hold``).
    Atom *selection* has no such partial option -- a half-held atom still
    gets displaced in full -- so ``FixCartesian`` belongs in *this*
    function's ``unhandled`` even though ``free_component_mask`` correctly
    leaves it out of its own. The two functions answer different questions
    about the same constraint and are expected to disagree about it.

    Returns
    -------
    (list of int, list of str)
        The indices to displace, and the sorted names of the constraint
        types that were not honoured.
    """
    fixed = set()
    unhandled = set()
    for constraint in getattr(atoms, "constraints", ()) or ():
        kind = type(constraint).__name__
        if kind == "FixAtoms":
            fixed.update(int(i) for i in constraint.get_indices())
        else:
            unhandled.add(kind)
    unhandled = sorted(unhandled)

    if explicit is not None:
        return list(explicit), unhandled

    return [i for i in range(len(atoms)) if i not in fixed], unhandled


def _param_value(raw):
    """Unwrap a run-record parameter's ``{"value": ..., "source": ...}``
    shape, or pass a bare value through unchanged.

    ``run_record._tag`` wraps every parameter it writes this way
    unconditionally, so a value written through it is never bare. This
    still accepts a bare value anyway: nothing here can guarantee an
    on-disk record went through ``_tag`` at all (a hand-edited file, or a
    schema variant predating it), and :func:`resolve_fmax_expectation`
    must not raise on one.
    """
    if isinstance(raw, dict):
        raw = raw.get("value")
    return None if raw is None else float(raw)


def resolve_fmax_expectation(explicit, structure_dir):
    """What counts as "relaxed enough" for this structure, and where it came from.

    A frequency analysis assumes a stationary point. When the geometry is not
    one, the measured curvature is not the curvature of a minimum and the
    failure shows up as spurious imaginary modes. This never refuses
    (design note D5) -- it supplies the number a warning is measured against.

    Order: an explicit value, else the fmax a *converged* ``optimize`` stage
    in this directory's run record actually met, else nothing. The record is
    not an invented constant: it is the criterion already applied to this
    structure, with its provenance attached. A not-converged stage supplies
    nothing, because the fmax it was aiming at is not one it met.

    Where that fmax is read from within the record is not uniform.
    ``src/mliprun/core/optimize.py`` never passes ``stage_parameters`` to
    ``RunRecord.begin`` for an "optimize" stage, so its fmax lives only in
    the record's top-level ``parameters`` -- the block ``RunRecord.begin``
    writes once, for whichever command first creates the record file, and
    never rewrites on append (only NEB's restart stages carry their own
    per-stage ``parameters``, for a different, already-fixed reason -- see
    ``tests/test_run_record_integration.py::
    test_restart_records_the_new_fmax_per_stage``). So a stage's own
    ``parameters`` is preferred when present, and the top-level block is
    used only as a fallback, and only when the record's origin command
    (``payload["command"]``) was itself "optimize" -- otherwise that block
    belongs to a different command and must not be misread as this stage's
    fmax.

    Never raises: a missing, unreadable or unexpected record yields
    ``(None, "none")``. A corrupt record must not cost the calculation.

    Parameters
    ----------
    explicit : float or None
        A caller-supplied ``--expect-fmax`` value. Wins over everything
        else when given.
    structure_dir : str or pathlib.Path or None
        The structure's own directory, expected to hold
        ``mliprun_run.json``.

    Returns
    -------
    (float or None, str)
        The expectation and its source: ``"explicit"``, ``"run_record"`` or
        ``"none"``.
    """
    from mliprun.core.run_record import RECORD_FILENAME

    if explicit is not None:
        return float(explicit), "explicit"
    if structure_dir is None:
        return None, "none"

    try:
        path = Path(structure_dir) / RECORD_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))

        top_level_fmax = None
        if payload.get("command") == "optimize":
            top_level_fmax = _param_value(
                (payload.get("parameters") or {}).get("fmax"))

        found = None
        for stage in payload.get("stages", []):
            if stage.get("kind") != "optimize":
                continue
            if stage.get("status") != "converged":
                continue
            stage_parameters = stage.get("parameters")
            if stage_parameters is not None:
                raw = _param_value(stage_parameters.get("fmax"))
            else:
                raw = top_level_fmax
            if raw is not None:
                found = raw
        if found is not None:
            return found, "run_record"
    except Exception as exc:  # noqa: BLE001 -- provenance is never fatal
        logger.debug("no fmax expectation from a run record: %s", exc)
    return None, "none"
