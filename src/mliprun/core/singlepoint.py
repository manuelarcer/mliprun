"""Single-point evaluation: one structure, one calculation, no motion.

Distinct from ``optimize run --max-steps 0``, which performs the same single
evaluation but reports it as a failed relaxation: status ``not_converged``,
the "increase max_steps" advice block, a trajectory and a CONTCAR for a
geometry that never moved, and no per-atom forces anywhere.
"""
import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from mliprun.core.committee.calculator import (
    free_component_mask,
    uncertainty_summary,
    write_peratom_sigma,
)
from mliprun.core.run_record import RunContext, RunRecord, collect_provenance
from mliprun.core.utils import GPA_TO_EV_PER_ANG3, calc_fmax

logger = logging.getLogger(__name__)


def _stress_block(atoms, stress: Optional[bool]) -> dict:
    """The three stress fields, whatever happened.

    ``stress`` of None means "attempt it when the cell is periodic in all
    three directions". A slab at pbc=(True, True, False) has a stress
    component along the vacuum that means nothing, and reporting it invites
    it to be used; True forces the attempt anyway.

    A calculator without stress raises PropertyNotImplementedError. That is
    recorded and never raised: a missing stress must not cost the energy.
    """
    if stress is False:
        return {"stress_eV_per_A3": None, "stress_GPa": None,
                "stress_unavailable_reason": "not requested"}
    if stress is None and not all(atoms.get_pbc()):
        return {"stress_eV_per_A3": None, "stress_GPa": None,
                "stress_unavailable_reason":
                    f"cell is not periodic in all three directions "
                    f"(pbc={tuple(bool(p) for p in atoms.get_pbc())})"}
    try:
        voigt = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception as exc:  # noqa: BLE001 -- never fail the run for stress
        return {"stress_eV_per_A3": None, "stress_GPa": None,
                "stress_unavailable_reason": f"{type(exc).__name__}: {exc}"}
    return {
        "stress_eV_per_A3": [float(v) for v in voigt],
        "stress_GPa": [float(v / GPA_TO_EV_PER_ANG3) for v in voigt],
        "stress_unavailable_reason": None,
    }


def _write_forces_csv(path: Path, symbols, raw_forces, free_mask) -> None:
    """One row per atom: the raw forces the model predicts, plus the mask.

    Raw, not constrained: ``atoms.get_forces()`` zeroes held components, and
    a CSV of zeros says nothing about what the model thinks. The mask columns
    are what tell a reader which rows the free statistics cover.
    """
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "symbol", "fx", "fy", "fz", "f_norm",
                         "free_x", "free_y", "free_z"])
        for index, (symbol, force, mask) in enumerate(
                zip(symbols, raw_forces, free_mask)):
            writer.writerow([
                index, symbol,
                float(force[0]), float(force[1]), float(force[2]),
                float(np.linalg.norm(force)),
                bool(mask[0]), bool(mask[1]), bool(mask[2]),
            ])


def run_singlepoint(
    atoms,
    output_dir=".",
    prefix: str = "singlepoint",
    model_name: str = "mlip",
    stress: Optional[bool] = None,
    run_context: Optional[RunContext] = None,
    device_requested: str = "auto",
    device_resolved: str = "auto",
    uma_task: Optional[str] = None,
    mace_head: Optional[str] = None,
    sevennet_task: Optional[str] = None,
    committee=None,
    committee_config=None,
    uncertainty_threshold: Optional[float] = None,
) -> dict:
    """Evaluate ``atoms`` once and write the results.

    Parameters
    ----------
    atoms : ase.Atoms
        With a calculator already attached.
    output_dir : str or Path
        Where the CSV and the run record go.
    prefix : str
        Stem for this command's output files.
    stress : bool, optional
        None attempts the stress only when the cell is periodic in all three
        directions; True always attempts it; False never does.
    committee : CommitteeCalculator, optional
        When given, it must already be started and attached as ``atoms.calc``.
        Teardown belongs to whoever built it.
    uncertainty_threshold : float, optional
        No default, matching ``run_optimization``: without one, sigma is
        reported and no verdict asserted.

    Returns
    -------
    dict
        The results block, exactly as written to the run record.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    forces_csv = output_path / f"{prefix}_forces.csv"
    peratom_csv = output_path / f"{prefix}_committee_peratom.csv"

    threshold = (float(uncertainty_threshold)
                 if uncertainty_threshold is not None else None)
    threshold_source = ("explicit" if uncertainty_threshold is not None
                        else "none")

    if committee is not None:
        committee.latest = None
        committee.latest_uncertainty_summary = None
        # The driver process has no torch at all (ADR 0001), so resolving a
        # device here would be a false claim about what hardware ran this.
        device_requested = "committee"
        device_resolved = "committee"

    record = RunRecord.begin(
        output_path,
        command="singlepoint",
        stage_kind="singlepoint",
        parameters={
            "prefix": prefix,
            "stress": stress,
            **({} if committee is None else {
                "uncertainty_threshold": threshold,
                "n_members": len(committee.members),
            }),
        },
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

    try:
        energy = float(atoms.get_potential_energy())
        constrained_forces = np.asarray(atoms.get_forces(), dtype=float)
        # Bypasses the constraint machinery on the Atoms object, so these are
        # what the model actually predicts rather than what an optimizer is
        # allowed to act on.
        raw_forces = np.asarray(atoms.calc.get_forces(atoms), dtype=float)
    except Exception as exc:
        record.complete(status="failed", results={"error": str(exc)})
        raise

    free_mask, unhandled = free_component_mask(atoms)
    symbols = atoms.get_chemical_symbols()

    results = {
        "energy_eV": energy,
        "fmax_free_eV_per_A": calc_fmax(constrained_forces),
        "fmax_all_eV_per_A": calc_fmax(raw_forces),
        "n_free_atoms": int(free_mask.any(axis=1).sum()),
        "worst_force_atom_all": int(
            np.argmax((raw_forces ** 2).sum(axis=1))),
        "worst_force_atom_free": int(
            np.argmax((constrained_forces ** 2).sum(axis=1))),
        "unhandled_constraints": unhandled,
        **_stress_block(atoms, stress),
    }
    results["worst_force_atom_all_symbol"] = symbols[
        results["worst_force_atom_all"]]
    results["worst_force_atom_free_symbol"] = symbols[
        results["worst_force_atom_free"]]

    if committee is not None and committee.latest is not None:
        summary = uncertainty_summary(
            [], committee.latest, threshold=threshold,
            threshold_source=threshold_source, symbols=symbols)
        results["committee_uncertainty"] = summary
        committee.latest_uncertainty_summary = summary
        write_peratom_sigma(
            peratom_csv, symbols,
            committee.latest["sigma_per_atom_all"],
            sigma_free=committee.latest["sigma_per_atom_free"],
            free_mask=committee.latest["free_mask"])

    _write_forces_csv(forces_csv, symbols, raw_forces, free_mask)
    record.complete(status="completed", results=results)

    logger.info("Single point: E = %.6f eV, fmax_free = %.6f, fmax_all = %.6f",
                energy, results["fmax_free_eV_per_A"],
                results["fmax_all_eV_per_A"])
    return results
