"""Transition-state (TS) verification: geometry and verdict helpers, and ``verify_ts``.

The helpers take structures, vectors and check results and return numbers,
vectors or verdict strings. ``verify_ts`` runs the calculations (frequencies,
and the displace-and-relax check when requested).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import write as ase_write
from ase.vibrations import VibrationsData

from ase.optimize import BFGS, FIRE, LBFGS

from mliprun.core.committee.calculator import free_component_mask
from mliprun.core.run_record import RunRecord, collect_provenance
from mliprun.core.utils import calc_fmax
from mliprun.core.vibrations import IMAGINARY_FLOOR_CM1, run_frequencies, select_indices

_KNOWN_CHECKS = ("overlap", "relax")
_OPTIMIZERS = {"bfgs": BFGS, "lbfgs": LBFGS, "fire": FIRE}  # same map as CustomNEB.optimize_endpoints
_FREQ_PROVENANCE_KEYS = ("device_requested", "device_resolved", "uma_task", "mace_head", "sevennet_task")

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
    # Henkelman and Jonsson, Eq. 11: the segment toward the higher neighbour gets the larger weight.
    if e_next > e_prev:
        return t_plus * d_max + t_minus * d_min
    return t_plus * d_min + t_minus * d_max


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
    n_matched = sum(m is not None for m in (plus["matched"], minus["matched"]))
    return "one_side" if n_matched == 1 else "neither"


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


def _read_imaginary_mode(vibrations_json: Path, mode_index: int) -> np.ndarray:
    """Cartesian displacement pattern (N, 3) of one mode, zeros on atoms not displaced.

    ASE's ``VibrationsData._energies_and_modes`` (ase/vibrations/data.py,
    line 373-374) multiplies the eigenvectors of the mass-weighted Hessian by
    ``masses**-0.5``, so ``get_modes`` returns Cartesian displacements, not
    mass-weighted ones. They are compared with a Cartesian reference, so no
    conversion is applied.
    """
    data = VibrationsData.read(str(vibrations_json))
    return np.asarray(data.get_modes(all_atoms=True)[mode_index], dtype=float)


def _endpoint_state(endpoint, free_mask, calculator_factory) -> tuple[float, float]:
    """Energy and free-component fmax of a structure, from one force call."""
    probe = endpoint.copy()
    probe.calc = calculator_factory()
    energy = float(probe.get_potential_energy())
    fmax = calc_fmax(probe.get_forces(apply_constraint=False) * free_mask)
    return energy, fmax


def _displace_and_relax(
    work, initial, final, vector, *, calculator_factory, output_path, optimizer_class, fmax, max_steps, match_tolerance
) -> dict:
    """Relax TS plus and minus ``vector``; compare both with the endpoints."""
    free_mask, _ = free_component_mask(work)
    endpoints = {}
    for label, endpoint in (("initial", initial), ("final", final)):
        energy, endpoint_fmax = _endpoint_state(endpoint, free_mask, calculator_factory)
        endpoints[label] = {"structure": endpoint, "energy": energy, "fmax": endpoint_fmax}

    sides = {}
    for side, sign in (("plus", 1.0), ("minus", -1.0)):
        atoms = work.copy()
        atoms.set_positions(work.get_positions() + sign * vector)
        atoms.calc = calculator_factory()
        prefix = output_path / f"ts_relax_{side}"
        optimizer = optimizer_class(atoms, trajectory=str(prefix) + ".traj", logfile=str(prefix) + ".log")
        converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
        relaxed_energy = float(atoms.get_potential_energy())
        ase_write(str(prefix) + ".vasp", atoms, format="vasp")

        distances = {label: endpoint_distance(atoms, info["structure"]) for label, info in endpoints.items()}
        nearest = min(distances, key=distances.get)
        matched = nearest if distances[nearest] <= match_tolerance else None
        distance_to_ts = endpoint_distance(atoms, work)
        sides[side] = {
            "converged": converged,
            "steps": int(optimizer.nsteps),
            "distance_to_initial_A": distances["initial"],
            "distance_to_final_A": distances["final"],
            "distance_to_ts_A": distance_to_ts,
            "matched": matched,
            "returned_to_ts": bool(matched is None and distance_to_ts <= match_tolerance),
            "energy_difference_eV": None if matched is None else relaxed_energy - endpoints[matched]["energy"],
        }

    result = classify_connection(sides["plus"], sides["minus"])
    return {
        "status": {"connects": "pass", "inconclusive": "inconclusive"}.get(result, "fail"),
        "result": result,
        "initial_fmax_eV_per_A": endpoints["initial"]["fmax"],
        "final_fmax_eV_per_A": endpoints["final"]["fmax"],
        "endpoints_relaxed": bool(max(endpoints["initial"]["fmax"], endpoints["final"]["fmax"]) <= fmax),
        "plus": sides["plus"],
        "minus": sides["minus"],
    }


def _format_report(results: dict, settings: dict) -> str:
    lines = [
        f"TS verification: {results['verdict']}",
        f"Reason: {results['reason']}",
        "",
        f"Imaginary modes: {results['n_imaginary']}",
    ]
    for freq in results["imaginary_frequencies_cm-1"]:
        lines.append(f"  {freq:.1f}i cm^-1")
    lines.append(
        f"TS fmax (free components): {results['ts_fmax_free_eV_per_A']:.4f} eV/A "
        f"(limit {results['ts_fmax_limit_eV_per_A']:.4f} eV/A)"
    )
    overlap = results.get("overlap")
    if overlap is not None:
        value = "not computed" if overlap["value"] is None else f"{overlap['value']:.4f}"
        lines.append(f"Overlap check: {overlap['status']}")
        lines.append(f"  reference: {overlap['reference']}")
        lines.append(f"  overlap: {value} (threshold {overlap['threshold']:.2f})")
    relax = results.get("relax")
    if relax is not None:
        lines.append(f"Displace-and-relax check: {relax['result'] or relax['status']}")
        lines.append(
            f"  displacement {relax['displacement_A']} A, match tolerance {relax['match_tolerance_A']} A, "
            f"{relax['optimizer']} to fmax {relax['fmax_eV_per_A']} eV/A in at most {relax['max_steps']} steps"
        )
        if relax["status"] == "not_run":
            lines.append("  not run: it needs exactly one imaginary mode")
        else:
            for side in ("plus", "minus"):
                row = relax[side]
                energy = "n/a" if row["energy_difference_eV"] is None else f"{row['energy_difference_eV']:.4f} eV"
                lines.append(
                    f"  {side:5s}: matched {row['matched']}, distance to IS {row['distance_to_initial_A']:.3f} A, "
                    f"to FS {row['distance_to_final_A']:.3f} A, to TS {row['distance_to_ts_A']:.3f} A, "
                    f"energy difference {energy}, "
                    f"{'converged' if row['converged'] else 'not converged'} in {row['steps']} steps"
                )
            if "note" in relax:
                lines.append(f"  Note: {relax['note']}")
            lines.append(
                f"  fmax of IS{relax['initial_fmax_eV_per_A']:.4f} eV/A, of FS {relax['final_fmax_eV_per_A']:.4f} eV/A"
            )
            if not relax["endpoints_relaxed"]:
                lines.append(
                    f"  Note: Unrelaxed IS or FS (fmax above {relax['fmax_eV_per_A']} eV/A); "
                    "a failed match can then reflect the endpoints, not the TS"
                )
    lines.append("")
    lines.append("Settings:")
    lines.extend(f"  {key}: {value}" for key, value in settings.items())
    return "\n".join(lines) + "\n"


def verify_ts(
    ts_atoms,
    initial,
    final,
    *,
    calculator_factory,
    checks=("overlap",),
    reference_direction=None,
    reference_label=None,
    output_dir="ts_verify",
    model_name="mlip",
    indices=None,
    delta=0.01,
    nfree=2,
    imaginary_floor=IMAGINARY_FLOOR_CM1,
    ts_fmax_limit=0.03,
    overlap_threshold=0.7,
    displacement=0.1,
    match_tolerance=0.3,
    relax_fmax=0.01,
    relax_optimizer="bfgs",
    relax_max_steps=200,
    provenance=None,
    extra_inputs=None,
    run_context=None,
) -> dict:
    """Verify a transition state by vibrational analysis.

    Runs a frequency calculation on a copy of ``ts_atoms``, counts the
    imaginary modes and, when exactly one is found, compares it with a
    reference direction (default: the initial-to-final displacement).

    ``calculator_factory()`` must return a fresh ASE calculator.
    ``provenance`` holds keyword arguments for ``collect_provenance`` (model,
    device, task and head). Returns the results dict that is also written as
    the ``results`` of the ``ts-verify`` stage in ``mliprun_run.json``.
    """
    checks = tuple(checks)
    if not checks:
        raise ValueError(f"checks is empty; choose at least one of {list(_KNOWN_CHECKS)}")
    unknown = [c for c in checks if c not in _KNOWN_CHECKS]
    if unknown:
        raise ValueError(f"unknown check(s) {unknown}; choose from {list(_KNOWN_CHECKS)}")
    if (reference_direction is None) != (reference_label is None):
        raise ValueError(
            "reference_direction and reference_label go together: a supplied direction must be named, "
            "and a label needs the direction it names (the default is the initial-to-final direction)"
        )
    optimizer_class = _OPTIMIZERS.get(str(relax_optimizer).lower())
    if optimizer_class is None:
        raise ValueError(f"unknown relax optimizer {relax_optimizer!r}; choose from {sorted(_OPTIMIZERS)}")

    check_same_system(ts_atoms, initial, final)

    work = ts_atoms.copy()  # keeps the constraints; the caller's object is never touched
    chosen, _ = select_indices(work, explicit=indices)

    reference = None
    reference_name = None
    if "overlap" in checks:
        if reference_direction is None:
            reference = is_to_fs_direction(initial, final)
            reference_name = "is_to_fs"
        else:
            reference = np.asarray(reference_direction, dtype=float)
            reference_name = reference_label
        if reference.shape != (len(work), 3):
            raise ValueError(f"reference direction has shape {reference.shape}; expected {(len(work), 3)}")
        if np.linalg.norm(reference[list(chosen)]) < _ZERO_NORM:
            raise ValueError(
                f"reference direction ({reference_name}) is zero on the displaced atoms; "
                "overlap is undefined (are the initial and final structures identical?)"
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ase_write(str(output_path / "ts_structure.vasp"), work, format="vasp")

    prov_kwargs = {"mlip_model": model_name, "device_requested": "auto", "device_resolved": "auto"}
    prov_kwargs.update(provenance or {})
    freq_provenance = {k: prov_kwargs[k] for k in _FREQ_PROVENANCE_KEYS if prov_kwargs.get(k) is not None}

    work.calc = calculator_factory()
    freq = run_frequencies(
        work,
        output_dir=output_path,
        prefix="ts_freq",
        model_name=model_name,
        indices=indices,
        delta=delta,
        nfree=nfree,
        write_modes="imaginary",
        imaginary_floor=imaginary_floor,
        run_context=run_context,
        record_command="ts-verify",
        **freq_provenance,
    )

    settings = {
        "checks": list(checks),
        "delta": delta,
        "nfree": nfree,
        "imaginary_floor_cm-1": imaginary_floor,
        "ts_fmax_limit_eV_per_A": ts_fmax_limit,
        "overlap_threshold": overlap_threshold,
        "displacement_A": displacement,
        "match_tolerance_A": match_tolerance,
        "relax_fmax": relax_fmax,
        "relax_optimizer": relax_optimizer,
        "relax_max_steps": relax_max_steps,
    }
    record = RunRecord.begin(
        output_path,
        command="ts-verify",
        stage_kind="ts-verify",
        parameters={},
        inputs={},
        provenance=collect_provenance(**prov_kwargs),
        run_context=run_context,
        stage_parameters=settings,
        append=True,
    )

    try:
        mask = np.asarray(freq["imaginary_mask"], dtype=bool)
        signed = np.asarray(freq["frequencies_cm-1"], dtype=float)
        n_imaginary = int(freq["n_imaginary"])
        ts_fmax = float(freq["fmax_at_input_free_eV_per_A"])

        statuses = {}
        overlap = None
        if "overlap" in checks:
            overlap = {"status": "not_run", "value": None, "threshold": overlap_threshold, "reference": reference_name}
            if n_imaginary == 1:
                mode = _read_imaginary_mode(output_path / "ts_freq_vibrations.json", int(np.flatnonzero(mask)[0]))
                value = mode_overlap(mode, reference, chosen)
                overlap["value"] = value
                overlap["status"] = "pass" if value >= overlap_threshold else "fail"
            statuses["overlap"] = overlap["status"]

        relax = None
        if "relax" in checks:
            relax = {
                "status": "not_run",
                "result": None,
                "displacement_A": displacement,
                "match_tolerance_A": match_tolerance,
                "fmax_eV_per_A": relax_fmax,
                "optimizer": str(relax_optimizer).lower(),
                "max_steps": relax_max_steps,
            }
            if n_imaginary == 1:
                mode = _read_imaginary_mode(output_path / "ts_freq_vibrations.json", int(np.flatnonzero(mask)[0]))
                relax.update(
                    _displace_and_relax(
                        work, initial, final, scale_mode(mode, displacement),
                        calculator_factory=calculator_factory,
                        output_path=output_path,
                        optimizer_class=optimizer_class,
                        fmax=relax_fmax,
                        max_steps=relax_max_steps,
                        match_tolerance=match_tolerance,
                    )
                )
                if relax["plus"]["returned_to_ts"] or relax["minus"]["returned_to_ts"]:
                    # A relaxation that ends on the TS did not test the connection.
                    relax["result"] = "inconclusive"
                    relax["status"] = "inconclusive"
                    relax["note"] = "a relaxation returned to the TS; use a larger displacement or another optimizer"
            statuses["relax"] = relax["status"]

        verdict, reason = decide_verdict(n_imaginary, ts_fmax, ts_fmax_limit, statuses)

        results = {
            "verdict": verdict,
            "reason": reason,
            "checks_requested": list(checks),
            "n_imaginary": n_imaginary,
            "imaginary_frequencies_cm-1": [float(abs(v)) for v in signed[mask]],
            "ts_fmax_free_eV_per_A": ts_fmax,
            "ts_fmax_limit_eV_per_A": ts_fmax_limit,
            "ts_fmax_above_limit": bool(ts_fmax > ts_fmax_limit),
            "displaced_indices": [int(i) for i in chosen],
            "n_force_calls_frequencies": int(freq["n_force_calls"]),
        }
        if overlap is not None:
            results["overlap"] = overlap
        if relax is not None:
            results["relax"] = relax
        if extra_inputs:
            results["inputs"] = dict(extra_inputs)

        (output_path / "ts_verify.txt").write_text(_format_report(results, settings))
    except Exception as exc:
        record.complete(status="failed", results={"error": str(exc)})
        raise

    record.complete(status="completed", results=results)
    return results
