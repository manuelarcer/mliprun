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

from mliprun.core.run_record import RunRecord, collect_provenance
from mliprun.core.utils import calc_fmax

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


def _write_frequency_csv(path, frequencies, energies_eV, imaginary):
    """Magnitudes plus a boolean, never a signed number.

    Writing an imaginary frequency as a negative one is the widespread
    convention and a silent trap for anything that sums or sorts the column.
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
        Warn when fmax at the input geometry exceeds this. Never refuses.
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

    stage_parameters = {
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
        parameters=stage_parameters,
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
        stage_parameters=stage_parameters,
    )

    # `name` sets both the cache directory and the mode filenames: ASE's
    # write_mode composes f"{vib.name}.{n}.traj". One name, two artefacts.
    vibration_name = str(output_path / prefix)
    vib = _vibrations_class(committee)(
        atoms, indices=list(chosen), name=vibration_name,
        delta=delta, nfree=nfree)

    try:
        vib.run()
        vib.read(method=method, direction=direction)
    except Exception as exc:
        record.complete(status="failed", results={"error": str(exc)})
        raise

    # The undisplaced geometry is already in the cache -- run() evaluates it
    # first -- so this costs nothing.
    eq_forces = np.asarray(vib._eq_disp().forces(), dtype=float)
    fmax_at_input = calc_fmax(eq_forces)
    fmax_warning = (None if expectation is None
                    else bool(fmax_at_input > expectation))

    data = vib.get_vibrations(method=method, direction=direction)
    energies = np.asarray(data.get_energies())
    frequencies = np.asarray(data.get_frequencies())
    imaginary = np.abs(frequencies.imag) > 0
    magnitudes = np.where(imaginary, np.abs(frequencies.imag),
                          frequencies.real)

    with vibrations_json.open("w") as handle:
        data.write(handle)
    # A handle in write mode, not a path: summary()'s log argument opens a
    # path with mode 'a', so a restart would write a second table into the
    # same file and the result would read as twice as many modes.
    with summary_txt.open("w") as handle:
        vib.summary(method=method, direction=direction, log=handle)
    _write_frequency_csv(frequencies_csv, magnitudes, energies.real,
                         imaginary)

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
        "fmax_at_input_free_eV_per_A": float(fmax_at_input),
        "fmax_expectation": expectation,
        "fmax_expectation_source": expectation_source,
        "fmax_warning": fmax_warning,
        "n_displaced_atoms": int(len(chosen)),
        "n_force_calls": int(vib.n_force_calls),
        "unhandled_constraints": unhandled,
    }

    record.complete(status="completed", results=results)
    return results


def _vibrations_class(committee):
    """``CountingVibrations``, or the committee-aware subclass from Task 12."""
    return CountingVibrations
