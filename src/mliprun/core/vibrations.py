"""Vibrational frequencies by finite differences of forces.

Wraps ``ase.vibrations.Vibrations`` rather than reimplementing the
displacement sweep. The one piece that is reimplemented is the Hessian
assembly, because a committee needs one Hessian per member and ASE's
``Vibrations.read`` can only build the single Hessian implied by whatever
``atoms.calc`` returned. ``assemble_hessian`` is pinned to ASE's arithmetic
by exact equality in tests/test_vibrations_hessian.py.

Design: docs/superpowers/specs/2026-09-17-singlepoint-and-frequencies-design.md
"""
import csv
import json
import logging
from pathlib import Path

import numpy as np
from ase.vibrations import Vibrations

from mliprun.core.committee.calculator import (
    free_component_mask,
    uncertainty_summary,
)
from mliprun.core.run_record import RunRecord, collect_provenance
from mliprun.core.utils import calc_fmax

logger = logging.getLogger(__name__)

VALID_NFREE = (2, 4)
VALID_DIRECTIONS = ("central", "forward", "backward")
VALID_METHODS = ("standard", "frederiksen")

#: Imaginary-mode threshold, matching ASE's own `im_tol` in
#: `VibrationsData._tabulate_from_energies`. Applied to the mode ENERGY in
#: eV, not to the frequency in cm^-1, so that <prefix>_frequencies.csv and
#: <prefix>_summary.txt can never classify the same mode differently.
IMAGINARY_ENERGY_TOL_EV = 1e-8

#: Mode-overlap below which index pairing is reported as suspect. A
#: diagnostic trigger for a warning, not a scientific verdict -- the overlaps
#: themselves are in the CSV for anyone who disagrees with the number.
MODE_OVERLAP_WARN = 0.9

#: Tolerance, in eV/A, for deciding that a reused displacement cache was
#: written by THIS run's calculator.
#:
#: Not exact equality, deliberately. A real MLIP on a GPU is not
#: bit-reproducible between runs -- the reduction order inside the kernels is
#: not fixed -- so an exact comparison would reject the legitimate restart
#: this cache exists to make cheap. A DIFFERENT model, on the other hand,
#: disagrees by orders of magnitude: the EMT/Lennard-Jones pair that exposed
#: the bug differs by ~1 eV/A on N2, six orders above this. 1e-6 eV/A
#: therefore separates "same calculator, re-evaluated" from "someone else's
#: cache" cleanly, and is itself far below any force anyone reports.
CACHE_IDENTITY_ATOL = 1e-6


class FrequencyCacheError(RuntimeError):
    """A displacement cache in this output directory cannot be trusted.

    Raised rather than worked around: the alternative is reporting another
    calculator's forces under this run's provenance, which is a wrong number
    carrying a false attribution. Both remedies -- delete the cache, or give
    this run its own ``--prefix`` -- are in the message.
    """


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

    try:
        if explicit is not None:
            return float(explicit), "explicit"
        if structure_dir is None:
            return None, "none"

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


class CountingVibrations(Vibrations):
    """``Vibrations`` that counts the force calls it actually makes.

    ``run()`` skips any displacement already in the cache, so a restarted
    sweep makes fewer calls than its geometry implies. Counting from
    ``len(indices)`` would report work that never happened.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_force_calls = 0

    def calculate(self, atoms, disp):
        results = super().calculate(atoms, disp)
        self.n_force_calls += 1
        return results


class CommitteeVibrations(CountingVibrations):
    """``Vibrations`` that also caches every member's own forces.

    ASE stores whatever ``calculate`` returns in its JSON cache, so the
    ``(M, N, 3)`` array survives the round trip and a restarted sweep keeps
    its spread. A calculator-side hook would not: a resumed run skips the
    displacements already on disk, and ``calculate`` is never called for
    them.
    """

    def calculate(self, atoms, disp):
        results = super().calculate(atoms, disp)
        results["forces_per_member"] = np.asarray(
            self.calc.latest["forces_per_member"], dtype=float)
        return results


#: Human labels for the cache-identity fields, used in the refusal message.
#: Keys are the sidecar's own field names.
_CACHE_FIELD_LABELS = {
    "delta": "displacement --delta",
    "nfree": "stencil --nfree",
    "model": "model",
    "per_member_forces": "per-member forces cached (--committee)",
}

#: Sentinel for "this field is absent from the recorded identity", so that a
#: field whose real value is None is not confused with a missing one.
_FIELD_ABSENT = object()


def _cache_identity_path(cache_dir):
    """Where the cache-identity sidecar lives: *beside* the cache directory.

    Not inside it. ASE owns every ``cache.*`` file under
    ``<output_dir>/<prefix>/`` and iterates them by that prefix, so keeping
    our file out of that directory means nothing here depends on how ASE
    globs its own.
    """
    return Path(str(cache_dir) + "_cache.json")


def _cache_identity(delta, nfree, model_name, committee):
    """What a displacement cache must have been built under to be reusable.

    ``delta`` and ``nfree`` are the two that make a reused cache numerically
    *wrong* rather than merely mislabelled. ASE names its cache entries by
    atom, axis and sign only -- there is no displacement size in the name --
    so a sweep at a different delta silently reuses the old forces and
    divides them by the new one. Measured on N2 under EMT: ``--delta 0.01``
    then ``--delta 0.05`` in one directory reported the top mode at 415.08
    cm^-1 where a clean run at 0.05 gives 930.86 cm^-1. Wrong by a factor of
    2.24, ``status: completed``, no warning.

    ``model`` and ``per_member_forces`` are cheap to record alongside them,
    and turn two failures that were previously caught only *after* the sweep
    into a refusal before anything has been written.

    ``model`` is the name the caller declared, not a measurement of it: a
    library caller who leaves ``model_name`` at its default gets the same
    string for two different potentials. :func:`_reject_a_foreign_cache`
    stays behind this for exactly that case.
    """
    return {
        "delta": float(delta),
        "nfree": int(nfree),
        "model": str(model_name),
        "per_member_forces": committee is not None,
    }


def _cache_has_entries(cache_dir):
    """Whether ASE has written any displacement into this cache directory.

    ``MultiFileJSONCache._filename`` composes ``cache.<key><extension>``, so
    one glob answers it without depending on the extension. A directory that
    exists but holds nothing (``Vibrations.__init__`` creates it) is not a
    cache to reuse.
    """
    return any(Path(cache_dir).glob("cache.*"))


def _check_the_cache_identity(cache_dir, identity):
    """Refuse, or claim, a displacement cache -- **before** the sweep runs.

    Running before ``vib.run()`` is the point, not an implementation detail.
    A refusal issued after the sweep has already written its own entries
    into the shared directory leaves a mixture of two identities on disk,
    and a later run of either identity finds a cache that partly matches it.
    Refusing before anything is written makes that state unreachable.

    A cache with no identity recorded beside it is refused rather than
    accepted on trust: the fields that matter most (``delta``, ``nfree``)
    leave no trace in the cached forces themselves, so there is no
    measurement that could recover them. :func:`_reject_a_foreign_cache`
    cannot help here -- the undisplaced geometry's forces do not depend on
    the displacement size.

    Raises
    ------
    FrequencyCacheError
        When entries exist and the recorded identity is missing, unreadable,
        or differs from this run's in any field.
    """
    sidecar = _cache_identity_path(cache_dir)
    if not _cache_has_entries(cache_dir):
        # Nothing to reuse, so this run owns the directory. Any sidecar
        # still sitting here describes a cache that is gone.
        sidecar.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return

    recorded = None
    if sidecar.exists():
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 -- treated as "no identity"
            logger.debug("unreadable cache identity %s: %s", sidecar, exc)
        else:
            if isinstance(loaded, dict):
                recorded = loaded

    if recorded is None:
        raise FrequencyCacheError(
            f"the displacement cache in {cache_dir} has no readable "
            f"{sidecar.name} beside it recording what it was swept under, "
            f"so this run cannot tell whether it used the same --delta and "
            f"--nfree. Those leave no trace in the cached forces, so there "
            f"is nothing to measure instead: reusing the cache on the "
            f"chance that they match would report frequencies wrong by the "
            f"ratio of the two displacements while the run reports success. "
            f"Delete {cache_dir} to recompute the sweep, or pass a "
            f"different --prefix to give this run its own cache.")

    differences = [
        (field, recorded.get(field, _FIELD_ABSENT), value)
        for field, value in identity.items()
        if recorded.get(field, _FIELD_ABSENT) != value
    ]
    if not differences:
        return

    detail = "; ".join(
        f"{_CACHE_FIELD_LABELS[field]}: cache "
        f"{'not recorded' if was is _FIELD_ABSENT else repr(was)}, "
        f"this run {now!r}"
        for field, was, now in differences)
    delta_note = ""
    if any(field == "delta" for field, _, _ in differences):
        delta_note = (
            " ASE names its cache entries by atom, axis and sign only, with "
            "no displacement size in the name, so those forces would be "
            "divided by this run's --delta and the frequencies would come "
            "out wrong by the ratio of the two.")
    raise FrequencyCacheError(
        f"the displacement cache in {cache_dir} was swept under different "
        f"settings -- {detail}.{delta_note} Delete {cache_dir} to recompute "
        f"the sweep, or pass a different --prefix to give this run its own "
        f"cache.")


def _displacement_keys(vib, nfree):
    """``(cache name, forces-dict key)`` for every entry this sweep uses.

    One enumeration, used both to check a cache before trusting it and to
    read it back per member, so the two can never drift apart.

    ``_disp`` and ``_eq_disp`` are ASE-private, used deliberately: the cache
    is the only complete record once a restart has skipped displacements.
    Pinned against ase>=3.23 by tests/test_vibrations_hessian.py.
    """
    pairs = [(vib._eq_disp().name, "eq")]
    steps = [-1, 1] if nfree == 2 else [-2, -1, 1, 2]
    for a in vib.indices:
        for i in range(3):
            for n in steps:
                pairs.append((vib._disp(a, i, n).name, (int(a), i, n)))
    return pairs


def _member_forces(vib, nfree):
    """Every displacement's per-member forces, back out of ASE's cache."""
    out = {}
    for name, key in _displacement_keys(vib, nfree):
        entry = vib.cache[name]
        if "forces_per_member" not in entry:
            # Reachable for a direct caller of `member_frequencies`; a run
            # through `run_frequencies` is stopped earlier, by
            # `_reject_a_cache_without_member_forces`, before any output
            # file has been rewritten.
            raise FrequencyCacheError(
                f"the displacement cache entry {name!r} holds no per-member "
                f"forces: it was written by a single-model run, which stores "
                f"only the consensus forces.")
        out[key] = np.asarray(entry["forces_per_member"], dtype=float)
    return out


def _expected_force_calls(n_displaced, nfree):
    """What a sweep from an empty cache would cost, in force calls.

    ``1`` for the undisplaced geometry, plus ``6`` per displaced atom at
    ``nfree=2`` (three-point: two signs x three Cartesian directions) or
    ``12`` at ``nfree=4``. Pinned by
    tests/test_core_vibrations.py::test_the_force_call_count_is_one_plus_six_
    per_displaced_atom and its ``nfree=4`` sibling.
    """
    return 1 + (6 if nfree == 2 else 12) * int(n_displaced)


def _check_the_displacement_cache(vib, atoms, cache_dir, nfree, committee):
    """The post-sweep cache guards, run only when something was reused.

    A sweep that computed every displacement itself wrote every cache entry
    itself, so there is nothing to verify and nothing to pay for.

    These sit *behind* :func:`_check_the_cache_identity`, which has already
    refused a cache swept at a different ``--delta`` or ``--nfree`` before
    this run wrote anything. What is left for these to catch is what a
    recorded identity cannot see: a calculator whose declared model name is
    unchanged but whose weights, head or task differ, and a cache whose
    per-member forces are absent despite the sidecar saying otherwise.
    """
    if vib.n_force_calls >= _expected_force_calls(len(vib.indices), nfree):
        return
    _reject_a_foreign_cache(vib, atoms, cache_dir)
    if committee is not None:
        _reject_a_cache_without_member_forces(vib, cache_dir, nfree)


def _reject_a_cache_without_member_forces(vib, cache_dir, nfree):
    """Refuse a single-model cache to a committee run.

    A single-model sweep writes entries whose only force key is ``forces``:
    the consensus is all there is. A committee needs ``forces_per_member`` to
    build one Hessian per member, so reading a single-model cache used to
    fail on a bare ``KeyError: 'forces_per_member'`` -- raised after the run
    record had been opened and before it was completed, which left the record
    saying ``status: "running"``, i.e. (docs/OUTPUTS.md) "the job died
    without reporting back".

    `freq` then `freq --committee` in one directory is the obvious way to
    reach this: both default to the same directory and the same prefix.

    Raises
    ------
    FrequencyCacheError
        When any entry this sweep will read carries no per-member forces.
    """
    for name, _ in _displacement_keys(vib, nfree):
        if "forces_per_member" in vib.cache[name]:
            continue
        raise FrequencyCacheError(
            f"the displacement cache in {cache_dir} was written by a "
            f"single-model run: entry {name!r} holds the consensus forces "
            f"only, with no per-member forces for a committee to build one "
            f"Hessian per member from. Delete {cache_dir} to recompute the "
            f"sweep with the committee, or pass a different --prefix to "
            f"give this run its own cache.")


def _reject_a_foreign_cache(vib, atoms, cache_dir):
    """Refuse a displacement cache that a different calculator wrote.

    ASE names its cache ``<output_dir>/<prefix>``, and ``prefix`` defaults to
    ``freq`` whatever the model is. So a second run with a DIFFERENT MLIP in
    the same directory silently reuses the first model's forces and reports
    them under its own provenance. Measured before this guard existed, EMT
    then Lennard-Jones on N2 in one directory: run 2 made 0 force calls,
    reported EMT's 928.1448 cm^-1 top frequency, and wrote
    ``provenance.mlip_model: "lj"``. Comparing two potentials on one
    structure is an obvious workflow and both runs default to the same
    directory and the same prefix.

    Called only when something was actually reused (see
    :func:`_check_the_displacement_cache`), and costs exactly one force
    evaluation against the ``6n`` a restart saves. It is not counted in
    ``n_force_calls``, which reports the sweep's own cost.

    Raises
    ------
    FrequencyCacheError
        When the current calculator's forces at the undisplaced geometry
        differ from the cached ones by more than
        :data:`CACHE_IDENTITY_ATOL`.
    """
    cached = np.asarray(vib._eq_disp().forces(), dtype=float)
    # `vib.calc.get_forces(atoms)` is exactly the call ASE's own
    # `Vibrations.calculate` makes (`results['forces'] =
    # self.calc.get_forces(atoms)`), so this is like-for-like rather than a
    # near-equivalent. `atoms` is back at the undisplaced geometry here --
    # `run()` restores it after every displacement.
    current = np.asarray(vib.calc.get_forces(atoms), dtype=float)
    if (current.shape == cached.shape
            and np.allclose(current, cached, rtol=0,
                            atol=CACHE_IDENTITY_ATOL)):
        return
    deviation = (float(np.abs(current - cached).max())
                 if current.shape == cached.shape else float("nan"))
    raise FrequencyCacheError(
        f"the displacement cache in {cache_dir} was not written by this "
        f"run's calculator: re-evaluating the undisplaced geometry gives "
        f"forces differing from the cached ones by {deviation:.3g} eV/A, "
        f"above the {CACHE_IDENTITY_ATOL:g} eV/A tolerance that separates a "
        f"re-evaluation of the same model from a different one. Reusing it "
        f"would report that calculator's frequencies under this run's "
        f"provenance. Delete {cache_dir} to recompute the sweep, or pass a "
        f"different --prefix to give this run its own cache.")


def member_frequencies(vib, atoms, indices, delta, nfree, direction, method,
                       member_names, committee_modes):
    """Per-member frequencies, ZPE, per-mode spread and mode overlaps.

    Each member's Hessian is diagonalized independently and its eigenvalues
    come back sorted ascending, so for near-degenerate modes member A's mode
    7 and member B's mode 7 need not be the same physical mode. Pairing is by
    index and the risk is made visible rather than corrected: ``overlaps``
    carries ``|<u_member,i | u_committee,i>|`` per member per mode, which is
    close to 1 for a clean match.

    ``VibrationsData.get_modes()`` returns Cartesian mode vectors that are
    unit-normalized in the MASS-WEIGHTED basis, not in plain Cartesian space
    -- ``modes = eigh_vectors * masses ** -0.5`` -- so their raw Cartesian
    L2 norm is ``1/sqrt(mass)``, not 1 (exactly reproduced for two identical
    N2 members: every row norm came out ``1/sqrt(14.007) = 0.267``, and the
    raw dot product of two identical rows was its square, ``0.071``, not
    ``1.0``). Both mode arrays are renormalized to unit Cartesian L2 norm
    per row before the dot product so that a clean match reads as 1.0
    regardless of atomic mass.

    Frequency magnitude uses the complex modulus (``np.abs``), exactly like
    the headline column in :func:`run_frequencies`: a mode's ASE frequency is
    the complex square root of a real eigenvalue, so exactly one of
    ``.real``/``.imag`` is nonzero and ``np.abs`` always recovers it with no
    separate classification step. A mode ENERGY threshold
    (``IMAGINARY_ENERGY_TOL_EV``, the rule the headline ``imaginary`` column
    uses) only matters for *labelling* a mode real or imaginary for display;
    it changes nothing about its magnitude. This path does not label modes
    per member -- ``_write_committee_frequency_csv`` writes one ``imaginary``
    column, from the committee's own (headline) classification, and every
    member's magnitude is comparable to it because both use the same modulus
    expression.

    Returns
    -------
    dict
        ``frequencies`` ({name: (3n,) magnitudes}), ``zpe`` ({name: float}),
        ``std`` ((3n,) across members, ddof=1), ``overlaps``
        ({name: (3n,) floats}).
    """
    from ase.vibrations import VibrationsData

    def unit_rows(array):
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / norms

    committee_unit = unit_rows(committee_modes)
    per_displacement = _member_forces(vib, nfree)
    frequencies, zpe, overlaps = {}, {}, {}

    for position, name in enumerate(member_names):
        forces = {key: value[position]
                  for key, value in per_displacement.items()}
        hessian = assemble_hessian(forces, indices, delta, nfree=nfree,
                                   direction=direction, method=method)
        data = VibrationsData.from_2d(atoms, hessian, indices)
        raw = np.asarray(data.get_frequencies())
        frequencies[name] = np.abs(raw)
        zpe[name] = float(data.get_zero_point_energy())
        modes = unit_rows(np.asarray(data.get_modes()).reshape(len(raw), -1))
        overlaps[name] = np.abs(
            np.einsum("ij,ij->i", modes, committee_unit))

    stacked = np.stack([frequencies[name] for name in member_names])
    return {
        "frequencies": frequencies,
        "zpe": zpe,
        "std": stacked.std(axis=0, ddof=1),
        "overlaps": overlaps,
    }


def _write_committee_frequency_csv(path, magnitudes, imaginary, member_names,
                                   block):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["mode_index", "frequency_committee_cm-1", "imaginary"]
        header += [f"{name}_cm-1" for name in member_names]
        header += ["frequency_member_std_cm-1"]
        header += [f"{name}_overlap" for name in member_names]
        writer.writerow(header)
        for index in range(len(magnitudes)):
            row = [index, float(magnitudes[index]), bool(imaginary[index])]
            row += [float(block["frequencies"][name][index])
                    for name in member_names]
            row += [float(block["std"][index])]
            row += [float(block["overlaps"][name][index])
                    for name in member_names]
            writer.writerow(row)


def _write_frequency_csv(path, frequencies, energies_eV, imaginary):
    """Magnitudes plus a boolean, never a signed number.

    Writing an imaginary frequency as a negative one is the widespread
    convention and a silent trap for anything that sums or sorts the column.

    Both numeric columns are magnitudes, and for the same reason: an
    imaginary mode's energy is purely imaginary, so taking ``.real`` of it
    wrote 0.0 meV next to a nonzero frequency on the same row. Caller passes
    ``np.abs(energies)``, matching ``np.abs(frequencies)``.
    """
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode_index", "frequency_cm-1", "energy_meV",
                         "imaginary"])
        for index, (freq, energy, imag) in enumerate(
                zip(frequencies, energies_eV, imaginary)):
            writer.writerow([index, float(freq), float(energy) * 1000.0,
                             bool(imag)])


def run_frequencies(
    atoms,
    output_dir=".",
    prefix: str = "freq",
    model_name: str = "mlip",
    indices=None,
    delta: float = 0.01,
    nfree: int = 2,
    direction: str = "central",
    method: str = "standard",
    write_modes: str = "imaginary",
    expect_fmax=None,
    structure_dir=None,
    run_context=None,
    device_requested: str = "auto",
    device_resolved: str = "auto",
    uma_task=None,
    mace_head=None,
    sevennet_task=None,
    committee=None,
    committee_config=None,
    uncertainty_threshold=None,
) -> dict:
    """Vibrational frequencies by finite differences.

    Parameters
    ----------
    atoms : ase.Atoms
        With a calculator attached.
    indices : sequence of int, optional
        Atoms to displace. None derives them from the structure's
        ``FixAtoms`` constraints; see :func:`select_indices`.
    delta : float
        Displacement in Angstrom.
    nfree : int
        2 or 4.
    direction, method : str
        Passed to :func:`assemble_hessian` and to ASE's own reader.
    write_modes : str
        ``'none'``, ``'imaginary'`` or ``'all'``.
    expect_fmax : float, optional
        Warn when fmax at the input geometry, over the free force components,
        exceeds this. Never refuses.
    structure_dir : str or Path, optional
        Where to look for a run record supplying the expectation when
        ``expect_fmax`` is None. Defaults to ``output_dir``.

    Returns
    -------
    dict
        The results block, as written to the run record.
    """
    if write_modes not in ("none", "imaginary", "all"):
        raise ValueError(
            f"write_modes must be 'none', 'imaginary' or 'all', "
            f"got {write_modes!r}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frequencies_csv = output_path / f"{prefix}_frequencies.csv"
    summary_txt = output_path / f"{prefix}_summary.txt"
    vibrations_json = output_path / f"{prefix}_vibrations.json"

    chosen, unhandled = select_indices(atoms, explicit=indices)
    expectation, expectation_source = resolve_fmax_expectation(
        expect_fmax,
        structure_dir if structure_dir is not None else output_path)

    if committee is not None:
        committee.latest = None
        committee.latest_uncertainty_summary = None
        device_requested = "committee"
        device_resolved = "committee"

    parameters = {
        "prefix": prefix,
        "delta": delta,
        "nfree": nfree,
        "direction": direction,
        "method": method,
        "write_modes": write_modes,
        "indices": list(chosen),
        "expect_fmax": expectation,
        **({} if committee is None else {
            "uncertainty_threshold": uncertainty_threshold,
            "n_members": len(committee.members),
        }),
    }
    record = RunRecord.begin(
        output_path,
        command="freq",
        stage_kind="freq",
        parameters=parameters,
        inputs={
            "n_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
        },
        provenance=collect_provenance(
            mlip_model=model_name,
            device_requested=device_requested,
            device_resolved=device_resolved,
            uma_task=uma_task,
            mace_head=mace_head,
            sevennet_task=sevennet_task,
            committee=(None if committee_config is None
                       else committee_config.as_provenance(
                           measured_versions=getattr(
                               committee, "member_versions", None))),
        ),
        run_context=run_context,
    )

    # `name` sets both the cache directory and the mode filenames: ASE's
    # write_mode composes f"{vib.name}.{n}.traj". One name, two artefacts.
    vibration_name = str(output_path / prefix)

    # One guard over everything from the sweep to the last write, rather
    # than the two statements it used to cover. Every step below can raise --
    # the cache guards raise deliberately -- and an exception escaping before
    # the record is completed leaves it saying `status: "running"`, which
    # docs/OUTPUTS.md defines as "the job died without reporting back". A
    # wrong number must stop the run loudly; it must not also leave a record
    # claiming the job is still going.
    try:
        # BEFORE the sweep, deliberately: a refusal issued afterwards would
        # already have written this run's own displacements into the shared
        # cache directory, leaving a mixture of two identities that a later
        # run of either one would partly match.
        _check_the_cache_identity(
            vibration_name,
            _cache_identity(delta, nfree, model_name, committee))

        vib = _vibrations_class(committee)(
            atoms, indices=list(chosen), name=vibration_name,
            delta=delta, nfree=nfree)
        vib.run()
        vib.read(method=method, direction=direction)
        # The backstop to the identity check above: it catches a calculator
        # whose declared `model` name is unchanged but whose weights, head or
        # task differ, which a recorded name cannot see.
        _check_the_displacement_cache(vib, atoms, vibration_name, nfree,
                                      committee)

        results_uncertainty = None
        if committee is not None:
            # ASE's Vibrations.run() restores atoms.positions after every
            # displacement (ase.vibrations.vibrations.Vibrations.iterdisplace:
            # `if inplace: atoms.positions[disp.a, disp.i] = pos0`, which
            # fires for every displacement including the last, whether or not
            # it was actually recomputed this call) -- confirmed here rather
            # than assumed: `atoms.get_positions()` after `vib.run()` matches
            # the pre-run geometry bit-for-bit, on both a fresh sweep and a
            # fully cached restart. So `atoms` is back at the INPUT geometry
            # now, not the last-displaced one -- reading `committee.latest`
            # at this point (set by whichever displacement's `calculate()`
            # ran last, or not set at all after a fully cached restart) would
            # silently describe the wrong geometry, or none. One explicit
            # evaluation here is the only way to be sure: `preflight` (==
            # `_evaluate`) is cheap (one call against 1 + 6*n_displaced for
            # the sweep) and, unlike the cache, gives the same answer on a
            # restart as on a fresh run.
            committee.preflight(atoms)
            threshold = (float(uncertainty_threshold)
                         if uncertainty_threshold is not None else None)
            threshold_source = ("explicit"
                                if uncertainty_threshold is not None
                                else "none")
            results_uncertainty = uncertainty_summary(
                [], committee.latest, threshold=threshold,
                threshold_source=threshold_source,
                symbols=atoms.get_chemical_symbols())
            committee.latest_uncertainty_summary = results_uncertainty

        # The undisplaced geometry is already in the cache -- run() evaluates
        # it first -- so this costs nothing.
        eq_forces = np.asarray(vib._eq_disp().forces(), dtype=float)
        # What ASE put in that cache came from `calc.get_forces(atoms)`,
        # which BYPASSES the constraint machinery: it is an all-atom number.
        # The expectation it is compared against comes from an `optimize`
        # record -- the CONSTRAINED criterion the optimizer actually
        # converged against -- so measuring the comparison against the raw
        # number would fire on every correctly relaxed slab with frozen
        # layers (measured on a Pt(111) 2x2x4 + H slab relaxed to fmax 0.02
        # with the bottom two layers held: raw 0.3809 eV/A against
        # constrained 0.0198 eV/A). Both are reported, each named for its
        # population, exactly as `run_singlepoint` does.
        #
        # Masking the cached forces reproduces `atoms.get_forces()` exactly
        # for the constraints `free_component_mask` handles (FixAtoms,
        # FixCartesian): on that same slab both routes give 0.019832 eV/A.
        # This deliberately uses `free_component_mask`'s notion of "free",
        # not `select_indices`' -- the two answer different questions about
        # the same constraints, as `select_indices`' own docstring sets out.
        # A projecting constraint (FixedPlane, FixedLine, ...) is left
        # unmasked there, so the free value over-reports rather than
        # under-reports for those atoms; `unhandled_constraints` in the
        # results says when that applies.
        free_mask, _ = free_component_mask(atoms)
        fmax_at_input_free = calc_fmax(eq_forces * free_mask)
        fmax_at_input_all = calc_fmax(eq_forces)
        fmax_warning = (None if expectation is None
                        else bool(fmax_at_input_free > expectation))

        data = vib.get_vibrations(method=method, direction=direction)
        energies = np.asarray(data.get_energies())
        frequencies = np.asarray(data.get_frequencies())
        # Classify exactly as ASE's own summary table does (data.py's
        # _tabulate_from_energies): on the mode ENERGY in eV against im_tol,
        # never on the frequency in cm^-1. A near-zero frustrated
        # translation/rotation on a slab can pick up an arbitrary tiny sign
        # from finite differences; a `> 0` threshold on the frequency would
        # call that noise imaginary here while the summary table -- and any
        # transition-state "exactly one imaginary mode" check -- called it
        # real.
        imaginary = np.abs(energies.imag) > IMAGINARY_ENERGY_TOL_EV
        # ASE's frequency for each mode is the complex square root of a real
        # eigenvalue: non-negative gives a purely real, non-negative result;
        # negative gives a purely imaginary result with a non-negative
        # imaginary part. Exactly one of .real/.imag is nonzero, so np.abs()
        # (the complex modulus) always recovers that value -- unlike
        # selecting .imag or .real by the `imaginary` flag above, which now
        # uses a threshold on a DIFFERENT quantity (the energy) and can
        # therefore pick the wrong, exactly-zero component for a mode sitting
        # right at that threshold (see test_the_frequency_csv_and_summary_
        # agree_on_which_modes_are_imaginary and the regression it caught in
        # test_frequencies_match_ases_own_for_the_same_settings).
        magnitudes = np.abs(frequencies)
        # The same modulus, for the same reason, applied to the mode
        # energies: an imaginary mode's energy is purely imaginary, so
        # `.real` of it is exactly 0.0 and the CSV row reported a nonzero
        # frequency beside a zero energy (654.41 cm-1 written as 0.0 meV,
        # where the honest value is 81.1 meV).
        energy_magnitudes = np.abs(energies)

        with vibrations_json.open("w") as handle:
            data.write(handle)
        # A handle in write mode, not a path: summary()'s log argument opens
        # a path with mode 'a', so a restart would write a second table into
        # the same file and the result would read as twice as many modes.
        with summary_txt.open("w") as handle:
            vib.summary(method=method, direction=direction, log=handle)
        _write_frequency_csv(frequencies_csv, magnitudes, energy_magnitudes,
                             imaginary)

        results_committee = None
        if committee is not None:
            committee_modes = np.asarray(data.get_modes()).reshape(
                len(frequencies), -1)
            block = member_frequencies(
                vib, atoms, chosen, delta, nfree, direction, method,
                committee.member_names, committee_modes)
            _write_committee_frequency_csv(
                output_path / f"{prefix}_committee_frequencies.csv",
                magnitudes, imaginary, committee.member_names, block)
            zpe_values = [block["zpe"][name]
                          for name in committee.member_names]
            worst_overlap = min(
                float(block["overlaps"][name].min())
                for name in committee.member_names)
            results_committee = {
                "zpe_eV_per_member": block["zpe"],
                "zpe_mean_eV": float(np.mean(zpe_values)),
                "zpe_std_eV": float(np.std(zpe_values, ddof=1)),
                "frequency_member_std_cm-1": [float(v) for v in block["std"]],
                "worst_mode_overlap": worst_overlap,
                "mode_pairing_suspect": bool(
                    worst_overlap < MODE_OVERLAP_WARN),
            }
            if worst_overlap < MODE_OVERLAP_WARN:
                logger.warning(
                    "committee frequencies: lowest mode overlap is %.3f, "
                    "below %.2f. Modes are paired by index, so a "
                    "near-degenerate pair whose order differs between members "
                    "is compared like-for-unlike and its spread is not "
                    "disagreement.",
                    worst_overlap, MODE_OVERLAP_WARN)

        if write_modes == "all":
            for index in range(len(frequencies)):
                vib.write_mode(index)
        elif write_modes == "imaginary":
            for index in np.flatnonzero(imaginary):
                vib.write_mode(int(index))

        results = {
            "n_modes": int(len(frequencies)),
            "n_imaginary": int(imaginary.sum()),
            "frequencies_cm-1": [float(v) for v in magnitudes],
            "imaginary_mask": [bool(v) for v in imaginary],
            "zpe_eV": float(data.get_zero_point_energy()),
            "fmax_at_input_free_eV_per_A": float(fmax_at_input_free),
            "fmax_at_input_all_eV_per_A": float(fmax_at_input_all),
            "fmax_expectation": expectation,
            "fmax_expectation_source": expectation_source,
            "fmax_warning": fmax_warning,
            "n_displaced_atoms": int(len(chosen)),
            "n_force_calls": int(vib.n_force_calls),
            "unhandled_constraints": unhandled,
        }
        if results_committee is not None:
            results["committee_frequencies"] = results_committee
        if results_uncertainty is not None:
            results["committee_uncertainty"] = results_uncertainty
    except Exception as exc:
        record.complete(status="failed", results={"error": str(exc)})
        raise

    record.complete(status="completed", results=results)
    return results


def _vibrations_class(committee):
    """The committee-aware subclass when there is a committee to capture."""
    return CommitteeVibrations if committee is not None else CountingVibrations
