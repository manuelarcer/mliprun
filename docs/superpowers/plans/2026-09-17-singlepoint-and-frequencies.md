# Single-point and frequency analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `mlip singlepoint` (evaluate one structure: energy, forces, fmax, stress) and `mlip freq` (finite-difference vibrational frequencies and ZPE), both with committee support, where `freq` additionally reports per-member frequencies and their per-mode spread.

**Architecture:** Two new core engines (`core/singlepoint.py`, `core/vibrations.py`) with thin typer commands over them, following the existing `optimize` shape exactly. Frequencies wrap `ase.vibrations.Vibrations` rather than reimplementing finite differences; committee frequencies come from one displacement sweep by capturing each member's own force array at every displacement and assembling one Hessian per member. Shared committee-session handling is extracted out of `optimize.py` first, so the SIGTERM teardown lives in one place.

**Tech Stack:** Python 3.9+, typer (`>=0.20,<0.28`), ASE `>=3.23` (developed against 3.29.0), numpy, pandas, pytest. No MLIP package is needed to run any test in this plan.

**Spec:** `docs/superpowers/specs/2026-09-17-singlepoint-and-frequencies-design.md` — read it before Task 1. The plan argues from the spec; where they disagree, the spec is wrong and must be corrected in the same commit.

## Global Constraints

- **Draft PRs only. Never push to `main`.** One concern per PR. This plan is three PRs plus a verification pass: Task 1 (PR A), Tasks 2–6 (PR B), Tasks 7–14 (PR C), and Task 15 on cos-cluster before either PR B or PR C is marked ready for review.
- **Branch:** `feat/singlepoint-and-frequencies` already exists and carries the spec commit. PR B and PR C branch from the merged predecessor, not from each other.
- **Every test asserts a numerical value or an invariant.** Loosening a tolerance is acceptable only by recording the observed delta. Silently changed numerics are not.
- **Every test in this plan runs with no MLIP installed.** Use `ase.calculators.emt.EMT` directly, or the reserved `emt` tag through the committee worker path. Never import `mace`, `fairchem`, `sevenn` or `chgnet`.
- **Never regenerate `tests/goldens/*.json`.** If a golden test fails, report the numerical delta and stop.
- **Parametrize-id trap:** `tests/conftest.py` auto-skips any test whose keywords contain `uma`, `mace` or `sevenn`, and pytest adds a parametrize id as a keyword. Never use those bare strings as an id — use `member_a`, `case_a`. After adding parametrized tests, confirm `0 skipped` among the new cases, not merely a green run.
- **CI gate:** full unit suite plus diff coverage ≥ 90% on changed lines.
- **Unit suite command:** `pytest -m "not uma and not mace and not sevenn"`.
- **Style:** typer CLI, lazy imports for heavy packages, NumPy-style docstrings. Match the surrounding code.
- **Commit trailer:** every commit ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- **Python version, a discrepancy to leave alone:** `pyproject.toml` declares `requires-python = ">=3.9"`, but `src/mliprun/core/optimize.py` already annotates `output_dir: str | Path` — a PEP 604 union evaluated at import time, which raises `TypeError` on 3.9. The effective floor is therefore 3.10. Do not "fix" this here; it is out of scope and touching it would widen three PRs. Write new annotations that work on both, or omit the annotation as the code in this plan does.
- **Naming rule, non-negotiable (settled 2026-09-08):** any statistic over a subset of force components carries `_free` or `_all` on its key. A bare `fmax` or `sigma_max` is a defect.
- **Record failures must never kill a run.** `RunRecord.begin`/`complete` already swallow their own exceptions; never add a code path where a bad record aborts a calculation.

---

# PR A — extract the committee session helpers

No behaviour changes. This exists so the SIGTERM teardown from PR #47 lives in one module instead of three copies.

### Task 1: Move committee-session handling into `cli/committee_session.py`

**Files:**
- Create: `src/mliprun/cli/committee_session.py`
- Modify: `src/mliprun/cli/commands/optimize.py` — delete lines defining the six helpers, import them instead
- Test: no new test file; `tests/test_committee_cli.py` is the regression guard

**Interfaces:**
- Consumes: nothing.
- Produces, all public (they now cross a module boundary, so the leading underscore goes):
  - `COMMITTEE_CONFLICTS: tuple[str, ...]`
  - `reject_conflicting_options(ctx) -> None`
  - `build_committee(config, output_dir: Path, member_timeout: float) -> CommitteeCalculator`
  - `class Terminated(KeyboardInterrupt)`
  - `sigterm_as_interrupt()` — context manager
  - `started_committee(config, output_dir: Path, member_timeout: float, atoms)` — context manager yielding a started `CommitteeCalculator`
  - `report_committee_uncertainty(committee_calc) -> None`

- [ ] **Step 1: Record the current suite state**

Run: `pytest -m "not uma and not mace and not sevenn" -q 2>&1 | tail -3`

Write down the exact passed count. It must be identical after this task; a refactor that changes a test count changed behaviour.

- [ ] **Step 2: Create the new module by moving the code verbatim**

Create `src/mliprun/cli/committee_session.py`. Move these from `src/mliprun/cli/commands/optimize.py` **without editing their bodies**, renaming only the definitions listed in Interfaces above: `_COMMITTEE_CONFLICTS`, `_reject_conflicting_options`, `_build_committee`, `_Terminated`, `_sigterm_as_interrupt`, `_started_committee`, `_report_committee_uncertainty`.

Keep every docstring and every inline comment exactly as written. Those comments record why the SIGTERM path exists and what it cost to find; losing them in a move is the real risk of this task.

The module header:

```python
"""Committee lifecycle for CLI commands: startup, teardown, and reporting.

Extracted from ``cli/commands/optimize.py`` so that ``optimize``,
``singlepoint`` and ``freq`` share one copy. The SIGTERM path here is the fix
from PR #47: a plain ``kill`` on a driver used to leave every worker running
and holding a CUDA context on a node with no scheduler to reap them. Three
copies of that fix would be three chances for it to drift.
"""
```

Required imports in the new module: `contextlib`, `signal`, `threading`, `typer`, `Path`, and from `mliprun.core.committee.calculator` `CommitteeCalculator` and `CommitteeError`; from `mliprun.core.committee.remote` `MemberError` and `RemoteMember`; from `mliprun.cli.utils` `param_sources_from_ctx`.

- [ ] **Step 3: Point `optimize.py` at the new module**

Delete the moved definitions from `src/mliprun/cli/commands/optimize.py` and add:

```python
from mliprun.cli.committee_session import (
    build_committee,
    reject_conflicting_options,
    report_committee_uncertainty,
    sigterm_as_interrupt,
    started_committee,
)
```

Update the three call sites in `run()` and `batch()`: `_reject_conflicting_options(ctx)` → `reject_conflicting_options(ctx)`, `_started_committee(...)` → `started_committee(...)`, `_report_committee_uncertainty(...)` → `report_committee_uncertainty(...)`. Remove any now-unused imports from `optimize.py` (`signal`, `threading`, `CommitteeCalculator`, `CommitteeError`, `MemberError`, `RemoteMember` — check each with `grep` before deleting, some are still used).

- [ ] **Step 4: Verify nothing moved but the code**

Run: `pytest -m "not uma and not mace and not sevenn" -q 2>&1 | tail -3`

Expected: the identical passed count from Step 1, zero failures.

Then run the SIGTERM and teardown tests explicitly, because they are the ones this task can break silently:

Run: `pytest tests/test_committee_cli.py -q -k "sigterm or closes_every_member or keyboard_interrupt" -v`

Expected: PASS for `test_the_guard_installs_and_restores_the_sigterm_disposition`, `test_the_installed_handler_unwinds_instead_of_terminating`, `test_a_caller_on_a_worker_thread_keeps_its_own_disposition`, `test_a_committee_run_is_armed_while_it_holds_workers`, `test_a_sigterm_mid_run_closes_every_member_and_exits_143`, `test_the_single_model_path_leaves_sigterm_alone`, `test_a_startup_failure_still_closes_every_member`, `test_a_keyboard_interrupt_mid_run_still_closes_every_member`.

If any of these reference `optimize._sigterm_as_interrupt` by its old private name, update the test to the new import path. That is the only test edit this task may make.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/cli/committee_session.py src/mliprun/cli/commands/optimize.py tests/test_committee_cli.py
git commit -m "refactor(cli): extract committee session handling from optimize

singlepoint and freq both need committee startup, SIGTERM teardown and
uncertainty reporting. Three copies of the PR #47 teardown fix would be
three chances for it to drift on the exact path whose failure orphaned
GPU workers. No behaviour change: same suite, same count.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Open PR A as a draft**

```bash
git push -u origin feat/committee-session-extraction
gh pr create --draft --title "refactor(cli): extract committee session handling from optimize" --body "$(cat <<'EOF'
Pure move, no behaviour change. `optimize`, `singlepoint` and `freq` all need
committee startup, SIGTERM teardown and uncertainty reporting; this puts the
PR #47 teardown fix in one module instead of three.

Suite count identical before and after.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# PR B — `singlepoint`

### Task 2: `run_singlepoint` core, single model

**Files:**
- Create: `src/mliprun/core/singlepoint.py`
- Test: `tests/test_core_singlepoint.py`

**Interfaces:**
- Consumes: `mliprun.core.utils.calc_fmax`, `mliprun.core.utils.GPA_TO_EV_PER_ANG3`, `mliprun.core.committee.calculator.free_component_mask`, `mliprun.core.run_record.RunRecord`, `collect_provenance`, `RunContext`.
- Produces:

```python
def run_singlepoint(
    atoms,
    output_dir: str | Path = ".",
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
    """Evaluate one structure once. Returns the results dict."""
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core_singlepoint.py`:

```python
"""Single-point evaluation: energy, forces, fmax, stress."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk, fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from mliprun.core.singlepoint import run_singlepoint


@pytest.fixture
def slab():
    """Pt(111) 2x2x3 with the bottom layer fixed, one atom nudged off site.

    The nudge guarantees a non-zero force so fmax is a real number rather
    than numerical noise, and puts the largest force on a FIXED atom so
    fmax_free and fmax_all cannot coincide by accident.
    """
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    atoms.positions[bottom[0]] += (0.0, 0.0, 0.35)
    atoms.calc = EMT()
    return atoms


def test_energy_matches_a_direct_emt_evaluation(slab, tmp_path):
    expected = slab.get_potential_energy()
    results = run_singlepoint(slab, output_dir=tmp_path)
    assert results["energy_eV"] == pytest.approx(expected, abs=1e-10)


def test_free_and_all_fmax_differ_when_the_worst_atom_is_fixed(slab, tmp_path):
    results = run_singlepoint(slab, output_dir=tmp_path)
    constrained = np.sqrt((slab.get_forces() ** 2).sum(axis=1).max())
    raw = np.sqrt((slab.calc.get_forces(slab) ** 2).sum(axis=1).max())
    assert results["fmax_free_eV_per_A"] == pytest.approx(constrained, abs=1e-10)
    assert results["fmax_all_eV_per_A"] == pytest.approx(raw, abs=1e-10)
    assert results["fmax_all_eV_per_A"] > results["fmax_free_eV_per_A"]


def test_the_free_atom_count_excludes_the_fixed_layer(slab, tmp_path):
    results = run_singlepoint(slab, output_dir=tmp_path)
    assert results["n_free_atoms"] == 8      # 12 atoms, 4 fixed in layer 3


def test_the_worst_force_atom_is_named_for_each_population(slab, tmp_path):
    results = run_singlepoint(slab, output_dir=tmp_path)
    raw = slab.calc.get_forces(slab)
    assert results["worst_force_atom_all"] == int(
        np.argmax((raw ** 2).sum(axis=1)))
    assert results["worst_force_atom_all_symbol"] == "Pt"
    constrained = slab.get_forces()
    assert results["worst_force_atom_free"] == int(
        np.argmax((constrained ** 2).sum(axis=1)))


def test_the_forces_csv_has_one_row_per_atom_with_the_free_mask(slab, tmp_path):
    run_singlepoint(slab, output_dir=tmp_path)
    rows = list(csv.DictReader(
        (tmp_path / "singlepoint_forces.csv").open()))
    assert len(rows) == len(slab)
    assert rows[0].keys() == {
        "index", "symbol", "fx", "fy", "fz", "f_norm",
        "free_x", "free_y", "free_z"}
    fixed = [a.index for a in slab if a.tag == 3]
    assert rows[fixed[0]]["free_z"] == "False"
    assert rows[0]["free_z"] == "True"


def test_the_csv_carries_raw_forces_not_constrained_ones(slab, tmp_path):
    run_singlepoint(slab, output_dir=tmp_path)
    rows = list(csv.DictReader(
        (tmp_path / "singlepoint_forces.csv").open()))
    raw = slab.calc.get_forces(slab)
    fixed = [a.index for a in slab if a.tag == 3][0]
    # The constrained array would report exactly zero here.
    assert float(rows[fixed]["f_norm"]) == pytest.approx(
        float(np.linalg.norm(raw[fixed])), abs=1e-10)
    assert float(rows[fixed]["f_norm"]) > 1e-6


def test_stress_is_recorded_for_a_fully_periodic_cell(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.calc = EMT()
    results = run_singlepoint(atoms, output_dir=tmp_path)
    assert results["stress_eV_per_A3"] is not None
    assert len(results["stress_eV_per_A3"]) == 6
    assert results["stress_unavailable_reason"] is None
    expected_gpa = np.asarray(atoms.get_stress()) / 0.006241509
    assert results["stress_GPa"] == pytest.approx(expected_gpa, abs=1e-6)


def test_a_slab_skips_stress_and_says_why(slab, tmp_path):
    results = run_singlepoint(slab, output_dir=tmp_path)
    assert results["stress_eV_per_A3"] is None
    assert "periodic" in results["stress_unavailable_reason"]


def test_a_calculator_without_stress_records_why_and_still_reports_energy(
        tmp_path):
    """A missing stress must never cost the energy."""
    from ase.calculators.calculator import PropertyNotImplementedError

    class NoStress(EMT):
        def get_stress(self, atoms=None):
            raise PropertyNotImplementedError("no stress here")

    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.calc = NoStress()
    results = run_singlepoint(atoms, output_dir=tmp_path)
    assert results["stress_eV_per_A3"] is None
    assert results["stress_GPa"] is None
    assert "PropertyNotImplementedError" in results["stress_unavailable_reason"]
    assert results["energy_eV"] == pytest.approx(
        atoms.get_potential_energy(), abs=1e-10)


def test_stress_false_never_attempts_it(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.calc = EMT()
    results = run_singlepoint(atoms, output_dir=tmp_path, stress=False)
    assert results["stress_eV_per_A3"] is None
    assert results["stress_unavailable_reason"] == "not requested"


def test_stress_true_forces_the_attempt_on_a_slab(slab, tmp_path):
    """The default skips a slab because its vacuum-direction stress means
    nothing; an explicit --stress is the caller overriding that."""
    results = run_singlepoint(slab, output_dir=tmp_path, stress=True)
    assert results["stress_eV_per_A3"] is not None
    assert results["stress_unavailable_reason"] is None


def test_the_run_record_carries_the_singlepoint_stage(slab, tmp_path):
    run_singlepoint(slab, output_dir=tmp_path, model_name="emt")
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    stage = record["stages"][0]
    assert stage["kind"] == "singlepoint"
    assert stage["status"] == "completed"
    assert stage["results"]["energy_eV"] == pytest.approx(
        slab.get_potential_energy(), abs=1e-10)


def test_nothing_is_written_back_as_a_structure(slab, tmp_path):
    run_singlepoint(slab, output_dir=tmp_path)
    written = {p.name for p in tmp_path.iterdir()}
    assert "CONTCAR" not in written
    assert not any(name.endswith(".traj") for name in written)
    assert not any(name.endswith("_final.vasp") for name in written)
```

- [ ] **Step 2: Run the tests and watch every one fail**

Run: `pytest tests/test_core_singlepoint.py -v`
Expected: every test FAILS with `ModuleNotFoundError: No module named 'mliprun.core.singlepoint'`.

Do not proceed until you have seen them fail. A test that has never been observed failing is not evidence of anything.

- [ ] **Step 3: Write `src/mliprun/core/singlepoint.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_core_singlepoint.py -v`
Expected: all PASS.

If `test_the_run_record_carries_the_singlepoint_stage` fails on `stage["status"]`, check the literal `RunRecord.complete` uses elsewhere — `run_optimization` passes `"converged"`/`"not_converged"`, and `"completed"` may need to be added to whatever validates it. Read `src/mliprun/core/run_record.py` before changing anything; if the status is a free string there, no change is needed.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/singlepoint.py tests/test_core_singlepoint.py
git commit -m "feat(singlepoint): core single-point evaluation

Energy, per-atom forces, fmax over free and over all components as two
separately named numbers, and stress where the cell is fully periodic.
Nothing is written back as a structure, because nothing moved.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 3: The `singlepoint` CLI command, single model

**Files:**
- Create: `src/mliprun/cli/commands/singlepoint.py`
- Modify: `src/mliprun/cli/main.py`, `pyproject.toml`
- Test: `tests/test_cli_singlepoint.py`

**Interfaces:**
- Consumes: `run_singlepoint` from Task 2; `mliprun.cli.utils` helpers `DEVICE_HELP`, `MACE_HEAD_HELP`, `MLIP_HELP`, `UMA_TASK_HELP`, `SEVENNET_TASK_HELP`, `_resolve_device`, `detect_mlip`, `param_sources_from_ctx`, `setup_calculator`, `validate_mlip`.
- Produces: `app` (a `typer.Typer`) with one command `run`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_singlepoint.py`:

```python
"""CLI surface of `mlip singlepoint run`."""
import csv
from pathlib import Path

import pytest
from ase.build import fcc111
from ase.io import write
from typer.testing import CliRunner

from mliprun.cli.commands.singlepoint import app

runner = CliRunner()


@pytest.fixture
def structure(tmp_path):
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    path = tmp_path / "POSCAR"
    write(path, atoms, format="vasp")
    return path


def test_help_lists_the_run_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout


def test_a_run_writes_the_forces_csv_and_the_record(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    assert result.exit_code == 0, result.stdout
    out = structure.parent
    assert (out / "singlepoint_forces.csv").exists()
    assert (out / "mliprun_run.json").exists()
    rows = list(csv.DictReader((out / "singlepoint_forces.csv").open()))
    assert len(rows) == 12


def test_the_echo_names_both_fmax_populations(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    assert "fmax_free" in result.stdout
    assert "fmax_all" in result.stdout


def test_committee_with_an_explicit_mlip_is_rejected(structure, tmp_path):
    committee_file = tmp_path / "committee.yaml"
    committee_file.write_text("members: []\n")
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(committee_file), "--mlip", "mace"])
    assert result.exit_code == 1
    assert "--mlip" in result.stdout
    assert "--committee" in result.stdout


def test_a_typed_auto_mlip_is_still_rejected(structure, tmp_path):
    """Checked by parameter source, not by value: --mlip defaults to 'auto',
    so comparing values would miss a typed one."""
    committee_file = tmp_path / "committee.yaml"
    committee_file.write_text("members: []\n")
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(committee_file), "--mlip", "auto"])
    assert result.exit_code == 1
    assert "--mlip" in result.stdout


def _use_emt(monkeypatch):
    """Attach ASE's EMT wherever the command would attach an MLIP.

    No MLIP package is installed in the unit environment, so the calculator
    setup is the one thing that cannot run for real.
    """
    from ase.calculators.emt import EMT

    def fake_setup(atoms, *args, **kwargs):
        atoms.calc = EMT()
        return atoms

    monkeypatch.setattr(
        "mliprun.cli.commands.singlepoint.setup_calculator", fake_setup)
    monkeypatch.setattr(
        "mliprun.cli.commands.singlepoint.detect_mlip", lambda: "emt")
    monkeypatch.setattr(
        "mliprun.cli.commands.singlepoint.validate_mlip",
        lambda *args, **kwargs: None)
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `pytest tests/test_cli_singlepoint.py -v`
Expected: collection error, `No module named 'mliprun.cli.commands.singlepoint'`.

- [ ] **Step 3: Write the command**

Create `src/mliprun/cli/commands/singlepoint.py`. Model it on `optimize.py`'s `run()` and keep the echo style identical (the emoji prefixes are the existing convention, not decoration).

```python
"""Single-point CLI command: evaluate one structure and stop."""
import contextlib
from pathlib import Path

import typer
from ase.io import read

from mliprun.cli.committee_session import (
    reject_conflicting_options,
    report_committee_uncertainty,
    started_committee,
)
from mliprun.cli.utils import (
    DEVICE_HELP,
    MACE_HEAD_HELP,
    MLIP_HELP,
    SEVENNET_TASK_HELP,
    UMA_TASK_HELP,
    _resolve_device,
    detect_mlip,
    param_sources_from_ctx,
    setup_calculator,
    validate_mlip,
)
from mliprun.core.committee.config import CommitteeConfigError, load_committee
from mliprun.core.committee.remote import DEFAULT_CALC_TIMEOUT_S
from mliprun.core.run_record import RunContext
from mliprun.core.singlepoint import run_singlepoint

app = typer.Typer(help="Evaluate a structure once: energy, forces, stress.")


@app.command()
def run(
    ctx: typer.Context,
    structure: Path = typer.Option(..., prompt=True,
                                   help="Structure file (.vasp)"),
    mlip: str = typer.Option("auto", help=MLIP_HELP),
    uma_task: str = typer.Option(None, help=UMA_TASK_HELP),
    device: str = typer.Option("auto", help=DEVICE_HELP),
    mace_head: str = typer.Option(None, help=MACE_HEAD_HELP),
    sevennet_task: str = typer.Option(None, help=SEVENNET_TASK_HELP),
    committee: Path = typer.Option(
        None, "--committee",
        help="Path to a committee.yaml. Reports a consensus energy and the "
             "members' disagreement for this one configuration, with no "
             "relaxation. Mutually exclusive with --mlip and the head/task "
             "options."),
    member_timeout: float = typer.Option(
        DEFAULT_CALC_TIMEOUT_S, "--member-timeout",
        help="Seconds a single committee member may take before it is "
             "killed and the run aborts."),
    uncertainty_threshold: float = typer.Option(
        None, "--uncertainty-threshold",
        help="Flag this configuration when the committee's force "
             "disagreement over the free atoms exceeds this (eV/Å). NO "
             "DEFAULT: without it sigma is reported and no verdict "
             "asserted."),
    stress: bool = typer.Option(
        None, "--stress/--no-stress",
        help="Ask the calculator for the stress tensor. Default: attempted "
             "only when the cell is periodic in all three directions, "
             "because a slab's stress along the vacuum means nothing."),
    prefix: str = typer.Option("singlepoint",
                               help="Stem for this command's output files"),
):
    """Evaluate a structure once and report energy, forces and stress."""
    atoms = read(structure)
    typer.echo(f"📂 Loaded structure: {structure.name}")
    typer.echo(f"   Atoms: {len(atoms)}, "
               f"Formula: {atoms.get_chemical_formula()}")

    committee_config = None
    committee_calc = None
    if committee is not None:
        reject_conflicting_options(ctx)
        try:
            committee_config = load_committee(committee)
        except CommitteeConfigError as exc:
            typer.echo(f"❌ {exc}")
            raise typer.Exit(1)
        mlip = "committee"
        typer.echo(f"🧠 Committee of {len(committee_config.members)} members "
                   f"from {committee}")
        for spec in committee_config.members:
            typer.echo(f"   {spec.name}: {spec.mlip} "
                       f"[{spec.level_of_theory}] gpu={spec.gpu} "
                       f"env={spec.env}")
        if committee_config.mixed_theory:
            typer.echo(f"\n⚠️  {committee_config.mixed_theory_warning()}\n")
    else:
        if mlip == "auto":
            mlip = detect_mlip()
            typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
        else:
            typer.echo(f"🧠 Using MLIP: {mlip}")
        validate_mlip(mlip, sevennet_task, uma_task, mace_head)

    output_dir = structure.parent

    run_context = RunContext(
        command="singlepoint",
        mode="one-off",
        param_sources=param_sources_from_ctx(ctx),
    )
    run_context.extra_inputs = {
        "structure": structure.name,
        "structure_abspath": str(structure.resolve()),
    }

    with contextlib.ExitStack() as session:
        if committee_config is not None:
            committee_calc = session.enter_context(
                started_committee(committee_config, output_dir,
                                  member_timeout, atoms))
            atoms.calc = committee_calc
        else:
            typer.echo(f"⚙️  Attaching {mlip} calculator (device={device})...")
            atoms = setup_calculator(atoms, mlip, uma_task, device=device,
                                     mace_head=mace_head,
                                     sevennet_task=sevennet_task)

        results = run_singlepoint(
            atoms=atoms,
            output_dir=output_dir,
            prefix=prefix,
            model_name=mlip,
            stress=stress,
            run_context=run_context,
            device_requested=device,
            device_resolved=_resolve_device(device),
            uma_task=uma_task,
            mace_head=mace_head,
            sevennet_task=sevennet_task,
            committee=committee_calc,
            committee_config=committee_config,
            uncertainty_threshold=uncertainty_threshold,
        )

    typer.echo(f"\n⚡ Energy: {results['energy_eV']:.6f} eV")
    typer.echo(f"   fmax_free = {results['fmax_free_eV_per_A']:.6f} eV/Å "
               f"over {results['n_free_atoms']} free atoms "
               f"(worst: {results['worst_force_atom_free_symbol']} "
               f"#{results['worst_force_atom_free']})")
    typer.echo(f"   fmax_all  = {results['fmax_all_eV_per_A']:.6f} eV/Å "
               f"(worst: {results['worst_force_atom_all_symbol']} "
               f"#{results['worst_force_atom_all']})")
    if results["stress_GPa"] is None:
        typer.echo(f"   stress: not reported "
                   f"({results['stress_unavailable_reason']})")
    else:
        typer.echo(f"   stress (GPa, Voigt): "
                   f"{[round(v, 4) for v in results['stress_GPa']]}")
    if results["unhandled_constraints"]:
        typer.echo(
            f"   Note: constraint type(s) "
            f"{', '.join(results['unhandled_constraints'])} are not masked, "
            f"so the free statistics over-report for their atoms.")

    typer.echo("\n✅ Single point complete. Output files:")
    written = [f"{prefix}_forces.csv", "mliprun_run.json"]
    if committee_config is not None:
        written.insert(1, f"{prefix}_committee_peratom.csv")
    for name in written:
        typer.echo(f"   📄 {(output_dir / name).resolve()}")

    report_committee_uncertainty(committee_calc)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Register the command**

In `src/mliprun/cli/main.py`, add `singlepoint` to the import list and:

```python
app.add_typer(singlepoint.app, name="singlepoint",
              help="Evaluate a structure once: energy, forces, stress")
```

In `pyproject.toml`, under `[project.scripts]`:

```toml
singlepoint = "mliprun.cli.commands.singlepoint:app"
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_cli_singlepoint.py -v`
Expected: all PASS.

Then reinstall so the new console script exists, and check the command is reachable both ways:

Run: `pip install -e . -q && mlip singlepoint run --help && singlepoint run --help`
Expected: both print the help for `run`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add src/mliprun/cli/commands/singlepoint.py src/mliprun/cli/main.py pyproject.toml tests/test_cli_singlepoint.py
git commit -m "feat(singlepoint): mlip singlepoint run

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 4: Committee support on `singlepoint`

**Files:**
- Test: `tests/test_committee_singlepoint.py`
- Modify: only if a test fails — Tasks 2 and 3 already wired the committee path; this task proves it end to end against real worker subprocesses.

**Interfaces:**
- Consumes: everything from Tasks 2 and 3, plus the reserved `emt` tag in `mliprun.core.committee.worker`.
- Produces: nothing new.

- [ ] **Step 1: Read how the existing committee CLI tests build a committee file**

Read `tests/test_committee_cli.py` lines 1–140, in particular the fixture that writes a two-member `committee.yaml` pointing at `sys.executable` with `mlip: emt`. Reuse that fixture shape verbatim; do not invent a second convention for the same file.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_committee_singlepoint.py`:

```python
"""Committee evaluation of one configuration, no relaxation."""
import csv
import json
import sys
from pathlib import Path

import pytest
from ase.build import fcc111
from ase.io import write
from typer.testing import CliRunner

from mliprun.cli.commands.singlepoint import app

runner = CliRunner()


@pytest.fixture
def structure(tmp_path):
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    path = tmp_path / "POSCAR"
    write(path, atoms, format="vasp")
    return path


def _committee_file(tmp_path, *, biased=False):
    """Two EMT members in this interpreter. `biased` makes them disagree.

    The biased stub scales one member's forces, so sigma is non-zero by
    construction and the spread is an assertable number rather than a
    hopeful one.
    """
    stub = ("tests.committee_stubs.biased_worker" if biased
            else "mliprun.core.committee.worker")
    path = tmp_path / "committee.yaml"
    path.write_text(
        "members:\n"
        f"  - name: member_a\n    mlip: emt\n    env: {sys.executable}\n"
        f"  - name: member_b\n    mlip: emt\n    env: {sys.executable}\n"
    )
    return path


def test_two_identical_members_agree_exactly(structure, tmp_path):
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    assert result.exit_code == 0, result.stdout
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    summary = record["stages"][0]["results"]["committee_uncertainty"]
    assert summary["sigma_max_free_final_eV_per_A"] == pytest.approx(
        0.0, abs=1e-12)


def test_the_peratom_sigma_csv_is_written(structure, tmp_path):
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    path = structure.parent / "singlepoint_committee_peratom.csv"
    assert path.exists()
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 12


def test_no_threshold_means_no_verdict(structure, tmp_path):
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    summary = record["stages"][0]["results"]["committee_uncertainty"]
    assert summary["flagged"] is None
    assert summary["threshold_source"] == "none"


def test_an_explicit_threshold_reaches_the_record(structure, tmp_path):
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path)),
        "--uncertainty-threshold", "0.05"])
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    summary = record["stages"][0]["results"]["committee_uncertainty"]
    assert summary["threshold_eV_per_A"] == pytest.approx(0.05)
    assert summary["threshold_source"] == "explicit"
    assert summary["flagged"] is False     # identical members, sigma is 0


def test_the_record_names_both_members(structure, tmp_path):
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    members = record["provenance"]["committee"]["members"]
    assert [m["name"] for m in members] == ["member_a", "member_b"]
```

- [ ] **Step 3: Run the tests**

Run: `pytest tests/test_committee_singlepoint.py -v`
Expected: PASS. If `test_two_identical_members_agree_exactly` fails with a `KeyError` on `committee_uncertainty`, the cause is `uncertainty_summary([], committee.latest, ...)` in Task 2 — read its signature in `src/mliprun/core/committee/calculator.py` and check whether an empty `rows` list is handled. If it is not, fix `uncertainty_summary` rather than passing a fake row: a fabricated trace row would corrupt every mean it computes.

- [ ] **Step 4: Confirm nothing was skipped**

Run: `pytest tests/test_committee_singlepoint.py -q 2>&1 | tail -2`
Expected: `5 passed`, and **`0 skipped`**. A skip here means a member name collided with the `uma`/`mace`/`sevenn` keyword trap.

- [ ] **Step 5: Commit**

```bash
git add tests/test_committee_singlepoint.py
git commit -m "test(singlepoint): committee evaluation of one configuration

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 5: Python API entry point and documentation for `singlepoint`

**Files:**
- Modify: `docs/OUTPUTS.md`, `docs/PYTHON_API.md`, `README.md`, `AGENTS.md`, `CHANGELOG.md`
- Test: `tests/test_cli_singlepoint.py` (one added test)

**Interfaces:**
- Consumes: `run_singlepoint` from Task 2.
- Produces: nothing in code.

- [ ] **Step 1: Write the failing test for the documented output list**

Append to `tests/test_cli_singlepoint.py`:

```python
def test_every_listed_output_file_actually_exists(structure, monkeypatch):
    """The terminal lists what it wrote. A listing that names a missing file
    is worse than a short listing."""
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    listed = [line.split("📄")[1].strip()
              for line in result.stdout.splitlines() if "📄" in line]
    assert listed
    for path in listed:
        assert Path(path).exists(), f"listed but missing: {path}"
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_cli_singlepoint.py::test_every_listed_output_file_actually_exists -v`
Expected: PASS if Task 3 was written correctly. If it FAILS, the command lists a file it did not write — fix the command, not the test.

- [ ] **Step 3: Document the outputs**

In `docs/OUTPUTS.md`:
- Add `singlepoint` to the list of stage kinds.
- Add a `singlepoint` section documenting `<prefix>_forces.csv` (the column table from the spec) and the `results` block, copying the field list from the spec verbatim.
- State that the CSV carries **raw** forces while `fmax_free` is computed from the constrained ones, and why.
- State that `worst_force_atom_free` is the atom of greatest force while `committee_uncertainty.worst_atom_free` is the atom of greatest disagreement — two different atoms, two different questions.
- Under `schema_version`, add one sentence: the `singlepoint` stage kind is additive and does not bump the schema, because `provenance` is untouched and no existing field changes meaning.

In `docs/PYTHON_API.md`, add a `run_singlepoint` section with a runnable example following the shape of the existing `run_optimization` one.

In `README.md`, add `singlepoint` to the command list with a one-line example.

In `AGENTS.md`, add `singlepoint` to the entry-points line under "Running".

In `CHANGELOG.md` under `## [Unreleased]` → `### Added`, describe the command: what it does, that it replaces the `optimize run --max-steps 0` workaround which mislabelled the run as not-converged, the two separately named fmax values and why, the stress default, and committee support.

- [ ] **Step 4: Verify the docs match the code**

Run: `grep -n "singlepoint" docs/OUTPUTS.md docs/PYTHON_API.md README.md AGENTS.md CHANGELOG.md`
Expected: hits in all five files.

Run: `pytest -m "not uma and not mace and not sevenn" -q 2>&1 | tail -3`
Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add docs/ README.md AGENTS.md CHANGELOG.md tests/test_cli_singlepoint.py
git commit -m "docs(singlepoint): outputs, python API, changelog

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 6: Open PR B

- [ ] **Step 1: Check diff coverage before opening**

Run: `coverage run -m pytest -m "not uma and not mace and not sevenn" -q && coverage combine 2>/dev/null; coverage xml -q && diff-cover coverage.xml --compare-branch=main --fail-under=90`

Expected: ≥ 90% on changed lines. Below that, the red X on the PR will be the diff-coverage gate rather than a broken test — add tests for the uncovered lines rather than guessing at what failed.

- [ ] **Step 2: Open the draft PR**

```bash
git push -u origin feat/singlepoint
gh pr create --draft --title "feat(singlepoint): evaluate one structure, energy and forces" --body "$(cat <<'EOF'
`mlip singlepoint run` evaluates a structure once and reports energy,
per-atom forces, fmax and stress, with committee support.

Reachable before only as `optimize run --max-steps 0`, which does the same
single evaluation but records `status: not_converged`, prints the
"increase max_steps" advice, writes a trajectory and a CONTCAR for a
geometry that never moved, and writes no per-atom forces anywhere.

Two fmax values, separately named: `fmax_free` over the unconstrained
components (what an optimizer converges against) and `fmax_all` over
everything. The per-atom CSV carries the raw forces the model predicts,
not the constrained array, which would be zeros on the fixed layer.

Stress is attempted only when the cell is periodic in all three
directions; a slab's stress along the vacuum means nothing.

All tests run with no MLIP installed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Verify the PR body actually landed**

Run: `gh pr view --json body --jq '.body' | head -5`

Expected: the body text above. `gh pr edit` fails silently against Projects-classic with a GraphQL error and leaves the body unchanged, so if the body is empty, patch it with `gh api` and check again rather than assuming the create succeeded.

---

# PR C — `freq`

### Task 7: `assemble_hessian`, a pure function checked against ASE

**Files:**
- Create: `src/mliprun/core/vibrations.py`
- Test: `tests/test_vibrations_hessian.py`

**Interfaces:**
- Consumes: numpy only.
- Produces:

```python
def assemble_hessian(forces, indices, delta, nfree=2,
                     direction="central", method="standard"):
    """forces maps (atom, cartesian, ndisp) -> (N, 3); "eq" -> (N, 3)."""
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vibrations_hessian.py`:

```python
"""Our Hessian assembly against ASE's own, by exact equality.

ASE builds the Hessian inside Vibrations._read from its displacement cache.
A committee needs one Hessian per member, which that path cannot express, so
the assembly is reimplemented here -- and pinned to ASE's by exact equality
rather than a tolerance, because the two run the same arithmetic on the same
numbers and any difference at all is a transcription error.
"""
import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations

from mliprun.core.vibrations import assemble_hessian


def _run_ase(tmp_path, nfree=2, delta=0.01):
    atoms = molecule("N2")
    atoms.calc = EMT()
    vib = Vibrations(atoms, name=str(tmp_path / "vib"),
                     delta=delta, nfree=nfree)
    vib.run()
    return atoms, vib


def _cached_forces(vib, nfree):
    """Pull every displacement's forces back out of ASE's cache.

    ``vib._disp`` and ``vib._eq_disp`` are ASE-private. They are used
    deliberately: a restarted run skips displacements already on disk, so a
    calculator-side hook would miss them, and the cache is the only complete
    record. Pinned by this test file against ase>=3.23.
    """
    forces = {"eq": np.asarray(vib._eq_disp().forces())}
    steps = [-1, 1] if nfree == 2 else [-2, -1, 1, 2]
    for a in vib.indices:
        for i in range(3):
            for n in steps:
                forces[(int(a), i, n)] = np.asarray(
                    vib._disp(a, i, n).forces())
    return forces


@pytest.mark.parametrize("nfree", [2, 4], ids=["nfree_2", "nfree_4"])
def test_central_differences_match_ase_exactly(tmp_path, nfree):
    atoms, vib = _run_ase(tmp_path, nfree=nfree)
    vib.read(method="standard", direction="central")
    ours = assemble_hessian(
        _cached_forces(vib, nfree), vib.indices, vib.delta,
        nfree=nfree, direction="central", method="standard")
    assert np.array_equal(ours, vib.H)


@pytest.mark.parametrize("direction", ["forward", "backward"],
                         ids=["dir_forward", "dir_backward"])
def test_one_sided_differences_match_ase_exactly(tmp_path, direction):
    atoms, vib = _run_ase(tmp_path)
    vib.read(method="standard", direction=direction)
    ours = assemble_hessian(
        _cached_forces(vib, 2), vib.indices, vib.delta,
        nfree=2, direction=direction, method="standard")
    assert np.array_equal(ours, vib.H)


def test_frederiksen_matches_ase_exactly(tmp_path):
    atoms, vib = _run_ase(tmp_path)
    vib.read(method="frederiksen", direction="central")
    ours = assemble_hessian(
        _cached_forces(vib, 2), vib.indices, vib.delta,
        nfree=2, direction="central", method="frederiksen")
    assert np.array_equal(ours, vib.H)


def test_the_hessian_is_symmetric(tmp_path):
    atoms, vib = _run_ase(tmp_path)
    vib.read()
    ours = assemble_hessian(_cached_forces(vib, 2), vib.indices, vib.delta)
    assert np.array_equal(ours, ours.T)


def test_the_hessian_is_square_in_three_times_the_displaced_atoms(tmp_path):
    atoms, vib = _run_ase(tmp_path)
    vib.read()
    ours = assemble_hessian(_cached_forces(vib, 2), vib.indices, vib.delta)
    assert ours.shape == (3 * len(vib.indices), 3 * len(vib.indices))


def test_an_unknown_nfree_is_rejected(tmp_path):
    atoms, vib = _run_ase(tmp_path)
    vib.read()
    with pytest.raises(ValueError, match="nfree"):
        assemble_hessian(_cached_forces(vib, 2), vib.indices, vib.delta,
                         nfree=3)
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `pytest tests/test_vibrations_hessian.py -v`
Expected: collection error, `No module named 'mliprun.core.vibrations'`.

- [ ] **Step 3: Write `assemble_hessian`**

Create `src/mliprun/core/vibrations.py` with the module docstring and this function:

```python
"""Vibrational frequencies by finite differences of forces.

Wraps ``ase.vibrations.Vibrations`` rather than reimplementing the
displacement sweep. The one piece that is reimplemented is the Hessian
assembly, because a committee needs one Hessian per member and ASE's
``Vibrations._read`` can only build the single Hessian implied by whatever
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

    Mirrors ``ase.vibrations.Vibrations._read`` exactly. Kept separate so it
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
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_vibrations_hessian.py -v`
Expected: all PASS, with `np.array_equal` true — exact equality, not `approx`.

Run: `pytest tests/test_vibrations_hessian.py -q 2>&1 | tail -2`
Expected: `8 passed`, **`0 skipped`**.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/vibrations.py tests/test_vibrations_hessian.py
git commit -m "feat(freq): Hessian assembly pinned to ASE by exact equality

A committee needs one Hessian per member, which Vibrations._read cannot
express. The assembly is reimplemented and tested against ase's own vib.H
with np.array_equal, for central, forward, backward, nfree 2 and 4, and
Frederiksen.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 8: Index selection and the constraint warning

**Files:**
- Modify: `src/mliprun/core/vibrations.py`
- Test: `tests/test_vibrations_indices.py`

**Interfaces:**
- Consumes: `mliprun.core.committee.calculator.free_component_mask`.
- Produces:

```python
def parse_indices(text: Optional[str], n_atoms: int) -> Optional[list[int]]:
    """'0,1,5' and '12-30', mixed. None passes through as None."""

def select_indices(atoms, explicit: Optional[list[int]] = None) -> tuple[list[int], list[str]]:
    """Returns (indices to displace, unmasked constraint type names)."""
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vibrations_indices.py`:

```python
"""Which atoms get displaced, and what we say about constraints we cannot honour."""
import pytest
from ase.build import fcc111, molecule
from ase.constraints import FixAtoms, FixBondLength, FixCartesian

from mliprun.core.vibrations import parse_indices, select_indices


def test_a_comma_list_parses():
    assert parse_indices("0,1,5", n_atoms=10) == [0, 1, 5]


def test_a_range_parses_inclusive():
    assert parse_indices("2-5", n_atoms=10) == [2, 3, 4, 5]


def test_ranges_and_singles_mix_and_sort_unique():
    assert parse_indices("7,2-4,2", n_atoms=10) == [2, 3, 4, 7]


def test_none_passes_through():
    assert parse_indices(None, n_atoms=10) is None


def test_an_out_of_range_index_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        parse_indices("0,12", n_atoms=10)


def test_a_malformed_range_is_rejected():
    with pytest.raises(ValueError, match="could not parse"):
        parse_indices("2-", n_atoms=10)


def test_fixatoms_drives_the_default_selection():
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    indices, unhandled = select_indices(atoms)
    assert set(indices) == set(range(12)) - set(bottom)
    assert len(indices) == 8
    assert unhandled == []


def test_an_unconstrained_structure_displaces_every_atom():
    atoms = molecule("H2O")
    indices, unhandled = select_indices(atoms)
    assert indices == [0, 1, 2]
    assert unhandled == []


def test_fixcartesian_is_named_as_unhandled_and_the_atom_still_moves():
    """ASE's indices select whole atoms, so a partial Hessian cannot be
    expressed. The atom is displaced in full and we say so (D6)."""
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    atoms.set_constraint(FixCartesian(0, mask=(True, True, False)))
    indices, unhandled = select_indices(atoms)
    assert 0 in indices
    assert unhandled == ["FixCartesian"]


def test_other_constraint_types_are_named_too():
    atoms = molecule("H2O")
    atoms.set_constraint(FixBondLength(0, 1))
    indices, unhandled = select_indices(atoms)
    assert unhandled == ["FixBondLength"]


def test_an_explicit_selection_overrides_the_constraints():
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    indices, unhandled = select_indices(atoms, explicit=[0, 1])
    assert indices == [0, 1]


def test_an_explicit_selection_of_a_fixed_atom_is_honoured():
    """The user asked for it explicitly. Vibrations bypasses the constraint
    machinery when it collects forces, so the row is real, not zeros."""
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    indices, unhandled = select_indices(atoms, explicit=[bottom[0]])
    assert indices == [bottom[0]]
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `pytest tests/test_vibrations_indices.py -v`
Expected: every test FAILS with `ImportError: cannot import name 'parse_indices'`.

- [ ] **Step 3: Add both functions to `src/mliprun/core/vibrations.py`**

```python
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

    Returns
    -------
    (list of int, list of str)
        The indices to displace, and the sorted names of the constraint
        types that were not honoured.
    """
    from mliprun.core.committee.calculator import free_component_mask

    _, unhandled = free_component_mask(atoms)

    if explicit is not None:
        return list(explicit), unhandled

    fixed = set()
    for constraint in getattr(atoms, "constraints", ()) or ():
        if type(constraint).__name__ == "FixAtoms":
            fixed.update(int(i) for i in constraint.get_indices())
    return [i for i in range(len(atoms)) if i not in fixed], unhandled
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_vibrations_indices.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/vibrations.py tests/test_vibrations_indices.py
git commit -m "feat(freq): constraint-driven index selection, warn on what we cannot honour

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 9: The stationary-point expectation

**Files:**
- Modify: `src/mliprun/core/vibrations.py`
- Test: `tests/test_vibrations_fmax_expectation.py`

**Interfaces:**
- Consumes: `mliprun.core.run_record.RECORD_FILENAME`.
- Produces:

```python
def resolve_fmax_expectation(explicit, structure_dir) -> tuple[Optional[float], str]:
    """Returns (expectation or None, source in {"explicit","run_record","none"})."""
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vibrations_fmax_expectation.py`:

```python
"""Where the stationary-point expectation comes from (D5: warn, never refuse)."""
import json

import pytest

from mliprun.core.vibrations import resolve_fmax_expectation


def _record(tmp_path, stages):
    (tmp_path / "mliprun_run.json").write_text(
        json.dumps({"schema_version": 5, "stages": stages}))
    return tmp_path


def test_an_explicit_value_wins():
    value, source = resolve_fmax_expectation(0.02, structure_dir=None)
    assert value == pytest.approx(0.02)
    assert source == "explicit"


def test_an_explicit_value_wins_even_over_a_record(tmp_path):
    directory = _record(tmp_path, [
        {"kind": "optimize", "status": "converged",
         "parameters": {"fmax": {"value": 0.05, "source": "user"}}}])
    value, source = resolve_fmax_expectation(0.01, structure_dir=directory)
    assert value == pytest.approx(0.01)
    assert source == "explicit"


def test_a_converged_optimize_stage_supplies_the_expectation(tmp_path):
    directory = _record(tmp_path, [
        {"kind": "optimize", "status": "converged",
         "parameters": {"fmax": {"value": 0.05, "source": "user"}}}])
    value, source = resolve_fmax_expectation(None, structure_dir=directory)
    assert value == pytest.approx(0.05)
    assert source == "run_record"


def test_the_latest_converged_optimize_stage_wins(tmp_path):
    directory = _record(tmp_path, [
        {"kind": "optimize", "status": "converged",
         "parameters": {"fmax": {"value": 0.05, "source": "user"}}},
        {"kind": "optimize", "status": "converged",
         "parameters": {"fmax": {"value": 0.01, "source": "user"}}}])
    value, source = resolve_fmax_expectation(None, structure_dir=directory)
    assert value == pytest.approx(0.01)


def test_a_not_converged_stage_supplies_nothing(tmp_path):
    """The fmax it was aiming at is not a criterion it met."""
    directory = _record(tmp_path, [
        {"kind": "optimize", "status": "not_converged",
         "parameters": {"fmax": {"value": 0.05, "source": "user"}}}])
    value, source = resolve_fmax_expectation(None, structure_dir=directory)
    assert value is None
    assert source == "none"


def test_a_non_optimize_stage_supplies_nothing(tmp_path):
    directory = _record(tmp_path, [
        {"kind": "md", "status": "completed",
         "parameters": {"fmax": {"value": 0.05, "source": "user"}}}])
    value, source = resolve_fmax_expectation(None, structure_dir=directory)
    assert value is None
    assert source == "none"


def test_no_record_at_all_supplies_nothing(tmp_path):
    value, source = resolve_fmax_expectation(None, structure_dir=tmp_path)
    assert value is None
    assert source == "none"


def test_an_unreadable_record_supplies_nothing_and_does_not_raise(tmp_path):
    """A corrupt record must never cost the calculation."""
    (tmp_path / "mliprun_run.json").write_text("{not json")
    value, source = resolve_fmax_expectation(None, structure_dir=tmp_path)
    assert value is None
    assert source == "none"


def test_an_untagged_parameter_value_is_read_too(tmp_path):
    """Records written without a RunContext carry bare values, not
    {"value":, "source":} dicts."""
    directory = _record(tmp_path, [
        {"kind": "optimize", "status": "converged",
         "parameters": {"fmax": 0.03}}])
    value, source = resolve_fmax_expectation(None, structure_dir=directory)
    assert value == pytest.approx(0.03)
    assert source == "run_record"
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `pytest tests/test_vibrations_fmax_expectation.py -v`
Expected: `ImportError: cannot import name 'resolve_fmax_expectation'`.

- [ ] **Step 3: Confirm the record's parameter shape before implementing**

Run: `grep -n -A20 "def _tag" src/mliprun/core/run_record.py`

Read how a parameter value is stored. If `_tag` wraps values as `{"value": ..., "source": ...}` only when `sources` is given, the implementation below must handle both shapes — which is why `test_an_untagged_parameter_value_is_read_too` exists. If the shape differs from what that test assumes, fix the test to match the real format and note the correction in the commit message.

- [ ] **Step 4: Implement**

Add to `src/mliprun/core/vibrations.py`:

```python
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

    Never raises: a missing, unreadable or unexpected record yields
    ``(None, "none")``. A corrupt record must not cost the calculation.

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
        found = None
        for stage in payload.get("stages", []):
            if stage.get("kind") != "optimize":
                continue
            if stage.get("status") != "converged":
                continue
            raw = (stage.get("parameters") or {}).get("fmax")
            if isinstance(raw, dict):
                raw = raw.get("value")
            if raw is not None:
                found = float(raw)
        if found is not None:
            return found, "run_record"
    except Exception as exc:  # noqa: BLE001 -- provenance is never fatal
        logger.debug("no fmax expectation from a run record: %s", exc)
    return None, "none"
```

Add `import json`, `import logging`, `from pathlib import Path` and `logger = logging.getLogger(__name__)` to the module header if Task 7 did not already.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_vibrations_fmax_expectation.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mliprun/core/vibrations.py tests/test_vibrations_fmax_expectation.py
git commit -m "feat(freq): stationary-point expectation from the input's run record

Warns, never refuses. With no explicit --expect-fmax, the comparison
value is the fmax a converged optimize stage in the structure's own
directory actually met -- the criterion already chosen for this
structure, not an invented constant.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 10: `run_frequencies`, single model

**Files:**
- Modify: `src/mliprun/core/vibrations.py`
- Test: `tests/test_core_vibrations.py`

**Interfaces:**
- Consumes: `assemble_hessian`, `select_indices`, `resolve_fmax_expectation` from Tasks 7–9; `calc_fmax`; `RunRecord`.
- Produces:

```python
def run_frequencies(
    atoms, output_dir=".", prefix="freq", model_name="mlip",
    indices=None, delta=0.01, nfree=2, direction="central",
    method="standard", write_modes="imaginary", expect_fmax=None,
    structure_dir=None, run_context=None, device_requested="auto",
    device_resolved="auto", uma_task=None, mace_head=None,
    sevennet_task=None, committee=None, committee_config=None,
    uncertainty_threshold=None,
) -> dict:
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core_vibrations.py`:

```python
"""Frequencies, ZPE, imaginary modes and restart, on EMT."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import fcc111, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.vibrations import VibrationsData

from mliprun.core.vibrations import run_frequencies


@pytest.fixture
def n2():
    """N2 at EMT's own minimum, so the only real mode is the stretch."""
    from ase.optimize import BFGS
    atoms = molecule("N2")
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-6)
    return atoms


def test_the_frequency_count_is_three_times_the_displaced_atoms(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path)
    assert results["n_modes"] == 6
    assert results["n_displaced_atoms"] == 2
    assert len(results["frequencies_cm-1"]) == 6


def test_the_force_call_count_is_one_plus_six_per_displaced_atom(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path)
    assert results["n_force_calls"] == 1 + 6 * 2


def test_nfree_four_doubles_the_displacements(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path, nfree=4)
    assert results["n_force_calls"] == 1 + 12 * 2


def test_frequencies_match_ases_own_for_the_same_settings(n2, tmp_path):
    from ase.vibrations import Vibrations
    reference = Vibrations(n2.copy(), name=str(tmp_path / "ref"))
    reference.atoms.calc = EMT()
    reference.run()
    expected = np.sort(np.abs(reference.get_frequencies()))
    results = run_frequencies(n2, output_dir=tmp_path / "ours")
    assert np.sort(results["frequencies_cm-1"]) == pytest.approx(
        expected, abs=1e-8)


def test_the_zero_point_energy_is_half_the_summed_real_energies(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path)
    data = VibrationsData.read(str(tmp_path / "freq_vibrations.json"))
    assert results["zpe_eV"] == pytest.approx(
        0.5 * np.asarray(data.get_energies()).real.sum(), abs=1e-12)


def test_a_stretched_bond_produces_an_imaginary_mode(tmp_path):
    """A geometry at a maximum along the bond, not a minimum."""
    atoms = molecule("N2")
    atoms.positions[1][2] += 1.6          # far past EMT's minimum
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path)
    assert results["n_imaginary"] >= 1
    assert any(results["imaginary_mask"])


def test_an_imaginary_frequency_is_written_as_a_positive_magnitude(tmp_path):
    atoms = molecule("N2")
    atoms.positions[1][2] += 1.6
    atoms.calc = EMT()
    run_frequencies(atoms, output_dir=tmp_path)
    rows = list(csv.DictReader((tmp_path / "freq_frequencies.csv").open()))
    imaginary = [r for r in rows if r["imaginary"] == "True"]
    assert imaginary
    for row in imaginary:
        assert float(row["frequency_cm-1"]) > 0.0


def test_an_imaginary_mode_contributes_nothing_to_the_zpe(tmp_path):
    atoms = molecule("N2")
    atoms.positions[1][2] += 1.6
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path)
    data = VibrationsData.read(str(tmp_path / "freq_vibrations.json"))
    energies = np.asarray(data.get_energies())
    real_only = 0.5 * energies.real.sum()
    assert results["zpe_eV"] == pytest.approx(real_only, abs=1e-12)


def test_only_the_free_atoms_are_displaced_on_a_constrained_slab(tmp_path):
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path)
    assert results["n_displaced_atoms"] == 8
    assert results["n_force_calls"] == 1 + 6 * 8
    assert results["n_modes"] == 24


def test_the_summary_file_is_written_once_not_appended(n2, tmp_path):
    """vib.summary()'s log argument opens a PATH in append mode. Passing a
    path twice would write two tables into one file."""
    run_frequencies(n2, output_dir=tmp_path)
    first = (tmp_path / "freq_summary.txt").read_text()
    run_frequencies(n2, output_dir=tmp_path)
    second = (tmp_path / "freq_summary.txt").read_text()
    assert first == second


def test_a_restart_makes_only_the_remaining_calls(n2, tmp_path):
    results_one = run_frequencies(n2, output_dir=tmp_path)
    assert results_one["n_force_calls"] == 13
    results_two = run_frequencies(n2, output_dir=tmp_path)
    assert results_two["n_force_calls"] == 0
    assert results_two["frequencies_cm-1"] == pytest.approx(
        results_one["frequencies_cm-1"], abs=1e-12)


def test_the_fmax_at_the_input_geometry_is_recorded(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path)
    assert results["fmax_at_input_free_eV_per_A"] == pytest.approx(
        0.0, abs=1e-5)
    assert results["fmax_expectation_source"] == "none"
    assert results["fmax_warning"] is None


def test_an_exceeded_expectation_warns_but_still_runs(tmp_path):
    atoms = molecule("N2")
    atoms.positions[1][2] += 0.3
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path, expect_fmax=1e-6)
    assert results["fmax_warning"] is True
    assert results["n_modes"] == 6          # it still ran


def test_the_run_record_carries_the_freq_stage(n2, tmp_path):
    run_frequencies(n2, output_dir=tmp_path, model_name="emt")
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    stage = record["stages"][0]
    assert stage["kind"] == "freq"
    assert stage["status"] == "completed"
    # Without a RunContext, `_tag` stores bare values rather than
    # {"value":, "source":} dicts. Unwrap either shape rather than asserting
    # one and discovering the other in CI.
    delta = stage["parameters"]["delta"]
    assert (delta["value"] if isinstance(delta, dict) else delta) == (
        pytest.approx(0.01))


def test_the_recorded_indices_match_what_was_displaced(tmp_path):
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    bottom = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom))
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path)
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    recorded = record["stages"][0]["parameters"]["indices"]
    recorded = recorded["value"] if isinstance(recorded, dict) else recorded
    assert len(recorded) == results["n_displaced_atoms"]
    assert set(recorded).isdisjoint(bottom)


def test_only_imaginary_modes_get_a_trajectory_by_default(tmp_path):
    atoms = molecule("N2")
    atoms.positions[1][2] += 1.6
    atoms.calc = EMT()
    results = run_frequencies(atoms, output_dir=tmp_path)
    written = sorted(p.name for p in tmp_path.glob("freq.*.traj"))
    assert len(written) == results["n_imaginary"]


def test_write_modes_none_writes_no_trajectory(tmp_path):
    atoms = molecule("N2")
    atoms.positions[1][2] += 1.6
    atoms.calc = EMT()
    run_frequencies(atoms, output_dir=tmp_path, write_modes="none")
    assert list(tmp_path.glob("freq.*.traj")) == []


def test_write_modes_all_writes_every_mode(n2, tmp_path):
    results = run_frequencies(n2, output_dir=tmp_path, write_modes="all")
    assert len(list(tmp_path.glob("freq.*.traj"))) == results["n_modes"]
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `pytest tests/test_core_vibrations.py -v`
Expected: every test FAILS with `ImportError: cannot import name 'run_frequencies'`.

- [ ] **Step 3: Implement `run_frequencies`**

Add to `src/mliprun/core/vibrations.py`. The force-call counter is the one piece that needs care: `Vibrations.run()` skips displacements already cached, so counting must happen inside `calculate`, not from `len(indices)`.

```python
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

    record = RunRecord.begin(
        output_path,
        command="freq",
        stage_kind="freq",
        parameters={
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
```

Add to the module header: `import csv`, `from ase.vibrations import Vibrations`, `from mliprun.core.run_record import RunRecord, collect_provenance`, `from mliprun.core.utils import calc_fmax`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_core_vibrations.py -v`
Expected: all PASS.

If `test_frequencies_match_ases_own_for_the_same_settings` fails, do **not** loosen the tolerance first. The two paths run identical arithmetic; a difference means the settings diverged (check `nfree`, `delta`, and that the reference got its own cache directory). Only if they genuinely differ, record the observed delta in the test and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/vibrations.py tests/test_core_vibrations.py
git commit -m "feat(freq): finite-difference frequencies, ZPE and mode trajectories

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 11: Per-member forces on the committee calculator

**Files:**
- Modify: `src/mliprun/core/committee/calculator.py`
- Test: `tests/test_committee_stats.py` (added tests)

**Interfaces:**
- Consumes: nothing new.
- Produces: `CommitteeCalculator.latest["forces_per_member"]` — an `(M, N, 3)` `numpy.ndarray` in `self.members` order.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_committee_stats.py`:

```python
def test_latest_carries_every_members_own_forces(started_committee_double):
    """Per-member Hessians need each member's own forces at every
    displacement. The mean alone cannot produce a spread."""
    from ase.build import molecule
    atoms = molecule("H2O")
    committee = started_committee_double
    committee.preflight(atoms)
    per_member = committee.latest["forces_per_member"]
    assert per_member.shape == (len(committee.members), len(atoms), 3)


def test_the_member_order_matches_member_names(started_committee_double):
    from ase.build import molecule
    import numpy as np
    atoms = molecule("H2O")
    committee = started_committee_double
    committee.preflight(atoms)
    per_member = committee.latest["forces_per_member"]
    mean = committee.latest["forces_mean"]
    assert np.allclose(per_member.mean(axis=0), mean, atol=1e-12)


def test_the_mean_forces_are_unchanged_by_the_new_key(started_committee_double):
    """Regression guard: adding a key must not perturb what optimize reads."""
    from ase.build import molecule
    import numpy as np
    atoms = molecule("H2O")
    committee = started_committee_double
    committee.preflight(atoms)
    stats = committee.latest
    assert np.allclose(
        stats["forces_mean"],
        stats["forces_per_member"].mean(axis=0), atol=1e-12)
```

Read the top of `tests/test_committee_stats.py` first and reuse whatever fixture it already has for a started two-member committee. If none exists there, take the one from `tests/test_committee_calculator.py`. Do not write a third fixture for the same object.

- [ ] **Step 2: Run the tests and watch them fail**

Run: `pytest tests/test_committee_stats.py -v -k "per_member or member_order or unchanged_by_the_new_key"`
Expected: FAIL with `KeyError: 'forces_per_member'`.

- [ ] **Step 3: Add the key**

In `src/mliprun/core/committee/calculator.py`, inside `_evaluate`, after `stats["energies"] = dict(energies)`:

```python
        # Every member's own forces, in `self.members` order, kept so a
        # caller can build one Hessian per member from a single displacement
        # sweep (freq --committee). `stacked` is already (M, N, 3) in that
        # order. Negligible: 72 kB for six members on 500 atoms, and
        # `latest` is overwritten every evaluation.
        stats["forces_per_member"] = stacked
```

Confirm by reading `_validate` that `stacked` is ordered by `self.members` and not by dict insertion; if it is not, order it explicitly here rather than relying on the dict.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_committee_stats.py -v`
Expected: all PASS, including every pre-existing test in the file.

Run: `pytest tests/test_committee_optimize.py tests/test_committee_cli.py -q 2>&1 | tail -3`
Expected: green — this is the regression guard that a new key changed nothing for `optimize`.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/committee/calculator.py tests/test_committee_stats.py
git commit -m "feat(committee): keep each member's own forces on latest

Per-member Hessians need them at every displacement; the mean alone has
no spread to report. One key, (M, N, 3), member order.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 12: Committee frequencies

**Files:**
- Modify: `src/mliprun/core/vibrations.py`
- Test: `tests/test_committee_vibrations.py`

**Interfaces:**
- Consumes: `assemble_hessian` (Task 7), `CountingVibrations` (Task 10), `latest["forces_per_member"]` (Task 11).
- Produces:

```python
class CommitteeVibrations(CountingVibrations):
    """Captures each member's forces into ASE's displacement cache."""

def member_frequencies(vib, atoms, indices, delta, nfree, direction, method,
                       member_names, committee_modes) -> dict:
    """Per-member frequencies, ZPE, per-mode spread and mode overlaps.

    committee_modes is the (3n, 3n) array of the committee's own mode
    vectors, against which each member's are overlapped.
    Returns {"frequencies", "zpe", "std", "overlaps"}.
    """

MODE_OVERLAP_WARN = 0.9   # module constant, diagnostic trigger only
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_committee_vibrations.py`:

```python
"""Per-member frequencies from one displacement sweep."""
import csv
import json
import sys

import numpy as np
import pytest
from ase.build import molecule
from ase.io import write
from typer.testing import CliRunner

runner = CliRunner()


def _committee_file(tmp_path):
    path = tmp_path / "committee.yaml"
    path.write_text(
        "members:\n"
        f"  - name: member_a\n    mlip: emt\n    env: {sys.executable}\n"
        f"  - name: member_b\n    mlip: emt\n    env: {sys.executable}\n"
    )
    return path


@pytest.fixture
def structure(tmp_path):
    atoms = molecule("N2")
    atoms.center(vacuum=5.0)
    path = tmp_path / "POSCAR"
    write(path, atoms, format="vasp")
    return path


def test_identical_members_give_exactly_zero_spread(structure, tmp_path):
    from mliprun.cli.commands.freq import app
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    assert result.exit_code == 0, result.stdout
    rows = list(csv.DictReader(
        (structure.parent / "freq_committee_frequencies.csv").open()))
    assert rows
    for row in rows:
        assert float(row["frequency_member_std_cm-1"]) == pytest.approx(
            0.0, abs=1e-9)


def test_identical_members_give_overlaps_of_exactly_one(structure, tmp_path):
    from mliprun.cli.commands.freq import app
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    rows = list(csv.DictReader(
        (structure.parent / "freq_committee_frequencies.csv").open()))
    for row in rows:
        assert float(row["member_a_overlap"]) == pytest.approx(1.0, abs=1e-9)
        assert float(row["member_b_overlap"]) == pytest.approx(1.0, abs=1e-9)


def test_the_csv_has_one_column_per_member(structure, tmp_path):
    from mliprun.cli.commands.freq import app
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    rows = list(csv.DictReader(
        (structure.parent / "freq_committee_frequencies.csv").open()))
    assert "member_a_cm-1" in rows[0]
    assert "member_b_cm-1" in rows[0]
    assert "frequency_committee_cm-1" in rows[0]


def test_the_headline_frequency_is_the_mean_potentials(structure, tmp_path):
    """Not the mean of the members' frequencies -- different numbers (D8).
    With identical members they coincide, which is what makes this a
    meaningful check of the plumbing rather than of the arithmetic."""
    from mliprun.cli.commands.freq import app
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    main = list(csv.DictReader(
        (structure.parent / "freq_frequencies.csv").open()))
    committee = list(csv.DictReader(
        (structure.parent / "freq_committee_frequencies.csv").open()))
    for a, b in zip(main, committee):
        assert float(a["frequency_cm-1"]) == pytest.approx(
            float(b["frequency_committee_cm-1"]), abs=1e-9)


def test_per_member_zpe_reaches_the_run_record(structure, tmp_path):
    from mliprun.cli.commands.freq import app
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    block = record["stages"][0]["results"]["committee_frequencies"]
    assert set(block["zpe_eV_per_member"]) == {"member_a", "member_b"}
    assert block["zpe_std_eV"] == pytest.approx(0.0, abs=1e-12)


def test_a_restart_reproduces_the_same_per_member_frequencies(
        structure, tmp_path):
    """Per-member forces must survive ASE's JSON cache, or a resumed
    committee run silently loses its spread."""
    from mliprun.cli.commands.freq import app
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    first = (structure.parent / "freq_committee_frequencies.csv").read_text()
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    second = (structure.parent / "freq_committee_frequencies.csv").read_text()
    assert first == second


def test_a_disagreeing_committee_gives_a_non_zero_spread(tmp_path):
    """Driven directly, not through the CLI: the point is the arithmetic."""
    from mliprun.core.vibrations import assemble_hessian
    from ase.calculators.emt import EMT
    from ase.vibrations import Vibrations, VibrationsData

    atoms = molecule("N2")
    atoms.calc = EMT()
    vib = Vibrations(atoms, name=str(tmp_path / "vib"))
    vib.run()
    vib.read()

    forces = {"eq": np.asarray(vib._eq_disp().forces())}
    for a in vib.indices:
        for i in range(3):
            for n in (-1, 1):
                forces[(int(a), i, n)] = np.asarray(vib._disp(a, i, n).forces())

    scaled = {k: v * 1.05 for k, v in forces.items()}
    h_ref = assemble_hessian(forces, vib.indices, vib.delta)
    h_scaled = assemble_hessian(scaled, vib.indices, vib.delta)
    f_ref = VibrationsData.from_2d(
        atoms, h_ref, vib.indices).get_frequencies()
    f_scaled = VibrationsData.from_2d(
        atoms, h_scaled, vib.indices).get_frequencies()
    spread = np.std([np.abs(f_ref), np.abs(f_scaled)], axis=0, ddof=1)
    assert spread.max() > 1.0        # cm-1, well above numerical noise


def test_swapped_modes_are_caught_by_the_overlap_not_by_the_spread(tmp_path):
    """The failure the overlap column exists to make visible.

    Two members whose mode 0 and mode 1 are the same two physical modes in
    the opposite order. Paired by index they look like a large
    disagreement; the overlap says the pairing is what is wrong.
    """
    from mliprun.core.vibrations import MODE_OVERLAP_WARN

    committee_modes = np.array([[1.0, 0.0], [0.0, 1.0]])
    member_modes = np.array([[0.0, 1.0], [1.0, 0.0]])     # order swapped
    overlaps = np.abs(np.einsum("ij,ij->i", member_modes, committee_modes))
    assert overlaps.max() == pytest.approx(0.0, abs=1e-12)
    assert overlaps.min() < MODE_OVERLAP_WARN


def test_the_run_record_flags_suspect_mode_pairing(structure, tmp_path,
                                                   monkeypatch):
    """With identical members the pairing is clean, so the flag is False.
    Forcing the threshold above 1.0 makes every clean pairing 'suspect',
    which is how the flag's wiring is exercised without faking an overlap."""
    import mliprun.core.vibrations as vibrations_module
    from mliprun.cli.commands.freq import app

    monkeypatch.setattr(vibrations_module, "MODE_OVERLAP_WARN", 1.5)
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    block = record["stages"][0]["results"]["committee_frequencies"]
    assert block["mode_pairing_suspect"] is True
    assert block["worst_mode_overlap"] == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `pytest tests/test_committee_vibrations.py -v`
Expected: FAIL — `No module named 'mliprun.cli.commands.freq'` for most, and the last one passes only after Task 7 (it exercises `assemble_hessian` alone).

- [ ] **Step 3: Implement the committee path**

Add to `src/mliprun/core/vibrations.py`:

```python
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


def _member_forces(vib, nfree):
    """Every displacement's per-member forces, back out of ASE's cache.

    ``_disp`` and ``_eq_disp`` are ASE-private, used deliberately: the cache
    is the only complete record once a restart has skipped displacements.
    Pinned against ase>=3.23 by tests/test_vibrations_hessian.py.
    """
    def cached(disp):
        return np.asarray(vib.cache[disp.name]["forces_per_member"],
                          dtype=float)

    out = {"eq": cached(vib._eq_disp())}
    steps = [-1, 1] if nfree == 2 else [-2, -1, 1, 2]
    for a in vib.indices:
        for i in range(3):
            for n in steps:
                out[(int(a), i, n)] = cached(vib._disp(a, i, n))
    return out


def member_frequencies(vib, atoms, indices, delta, nfree, direction, method,
                       member_names, committee_modes):
    """Per-member frequencies, ZPE, per-mode spread and mode overlaps.

    Each member's Hessian is diagonalized independently and its eigenvalues
    come back sorted ascending, so for near-degenerate modes member A's mode
    7 and member B's mode 7 need not be the same physical mode. Pairing is by
    index and the risk is made visible rather than corrected: ``overlaps``
    carries ``|<u_member,i | u_committee,i>|`` per member per mode, which is
    close to 1 for a clean match.

    Returns
    -------
    dict
        ``frequencies`` ({name: (3n,) magnitudes}), ``zpe`` ({name: float}),
        ``std`` ((3n,) across members, ddof=1), ``overlaps``
        ({name: (3n,) floats}).
    """
    from ase.vibrations import VibrationsData

    per_displacement = _member_forces(vib, nfree)
    frequencies, zpe, overlaps = {}, {}, {}

    for position, name in enumerate(member_names):
        forces = {key: value[position]
                  for key, value in per_displacement.items()}
        hessian = assemble_hessian(forces, indices, delta, nfree=nfree,
                                   direction=direction, method=method)
        data = VibrationsData.from_2d(atoms, hessian, indices)
        raw = np.asarray(data.get_frequencies())
        frequencies[name] = np.where(np.abs(raw.imag) > 0,
                                     np.abs(raw.imag), raw.real)
        zpe[name] = float(data.get_zero_point_energy())
        modes = np.asarray(data.get_modes()).reshape(len(raw), -1)
        overlaps[name] = np.abs(
            np.einsum("ij,ij->i", modes, committee_modes))

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
```

Change `_vibrations_class` from Task 10:

```python
def _vibrations_class(committee):
    """The committee-aware subclass when there is a committee to capture."""
    return CommitteeVibrations if committee is not None else CountingVibrations
```

In `run_frequencies`, after `_write_frequency_csv(...)`, add:

```python
    if committee is not None:
        committee_modes = np.asarray(data.get_modes()).reshape(
            len(frequencies), -1)
        block = member_frequencies(
            vib, atoms, chosen, delta, nfree, direction, method,
            committee.member_names, committee_modes)
        _write_committee_frequency_csv(
            output_path / f"{prefix}_committee_frequencies.csv",
            magnitudes, imaginary, committee.member_names, block)
        zpe_values = [block["zpe"][name] for name in committee.member_names]
        worst_overlap = min(
            float(block["overlaps"][name].min())
            for name in committee.member_names)
        results_committee = {
            "zpe_eV_per_member": block["zpe"],
            "zpe_mean_eV": float(np.mean(zpe_values)),
            "zpe_std_eV": float(np.std(zpe_values, ddof=1)),
            "frequency_member_std_cm-1": [float(v) for v in block["std"]],
            "worst_mode_overlap": worst_overlap,
            "mode_pairing_suspect": bool(worst_overlap < MODE_OVERLAP_WARN),
        }
        if worst_overlap < MODE_OVERLAP_WARN:
            logger.warning(
                "committee frequencies: lowest mode overlap is %.3f, below "
                "%.2f. Modes are paired by index, so a near-degenerate pair "
                "whose order differs between members is compared "
                "like-for-unlike and its spread is not disagreement.",
                worst_overlap, MODE_OVERLAP_WARN)
```

and merge `results_committee` into `results` as `results["committee_frequencies"]` before `record.complete`. Add the constant near the top of the module:

```python
#: Mode-overlap below which index pairing is reported as suspect. A
#: diagnostic trigger for a warning, not a scientific verdict -- the overlaps
#: themselves are in the CSV for anyone who disagrees with the number.
MODE_OVERLAP_WARN = 0.9
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_committee_vibrations.py -v`
Expected: all PASS once Task 13 has created `cli/commands/freq.py`. Run Task 13 first if the CLI-driven tests here cannot import it — the two tasks are written in this order for reading, not for execution.

Run: `pytest tests/test_committee_vibrations.py -q 2>&1 | tail -2`
Expected: `9 passed`, **`0 skipped`**.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/vibrations.py tests/test_committee_vibrations.py
git commit -m "feat(freq): per-member frequencies and per-mode spread

One displacement sweep, one Hessian per member. Members are queried
concurrently, so the cost is max(member), not sum(member). Mode pairing
is by index with a per-mode overlap column, so an ordering swap between
near-degenerate modes is visible rather than absorbed into the spread.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 13: The `freq` CLI command

**Files:**
- Create: `src/mliprun/cli/commands/freq.py`
- Modify: `src/mliprun/cli/main.py`, `pyproject.toml`
- Test: `tests/test_cli_freq.py`

**Interfaces:**
- Consumes: `run_frequencies`, `parse_indices` from `core/vibrations`; the committee session helpers from Task 1.
- Produces: `app` (a `typer.Typer`) with one command `run`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_freq.py`:

```python
"""CLI surface of `mlip freq run`."""
from pathlib import Path

import pytest
from ase.build import molecule
from ase.io import write
from typer.testing import CliRunner

from mliprun.cli.commands.freq import app

runner = CliRunner()


@pytest.fixture
def structure(tmp_path):
    atoms = molecule("N2")
    atoms.center(vacuum=5.0)
    path = tmp_path / "POSCAR"
    write(path, atoms, format="vasp")
    return path


def _use_emt(monkeypatch):
    from ase.calculators.emt import EMT

    def fake_setup(atoms, *args, **kwargs):
        atoms.calc = EMT()
        return atoms

    monkeypatch.setattr("mliprun.cli.commands.freq.setup_calculator",
                        fake_setup)
    monkeypatch.setattr("mliprun.cli.commands.freq.detect_mlip",
                        lambda: "emt")
    monkeypatch.setattr("mliprun.cli.commands.freq.validate_mlip",
                        lambda *a, **k: None)


def test_help_lists_the_run_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout


def test_a_run_writes_every_expected_output(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    assert result.exit_code == 0, result.stdout
    out = structure.parent
    for name in ("freq_frequencies.csv", "freq_summary.txt",
                 "freq_vibrations.json", "mliprun_run.json"):
        assert (out / name).exists(), name
    assert (out / "freq").is_dir()          # ASE's displacement cache


def test_every_listed_output_file_actually_exists(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    listed = [line.split("📄")[1].strip()
              for line in result.stdout.splitlines() if "📄" in line]
    assert listed
    for path in listed:
        assert Path(path).exists(), f"listed but missing: {path}"


@pytest.mark.parametrize("value", ["3", "0", "-1"],
                         ids=["nfree_3", "nfree_0", "nfree_negative"])
def test_an_invalid_nfree_is_rejected_with_the_allowed_values(
        structure, monkeypatch, value):
    _use_emt(monkeypatch)
    result = runner.invoke(app, [
        "run", "--structure", str(structure), "--nfree", value])
    assert result.exit_code == 1
    assert "2" in result.stdout and "4" in result.stdout


def test_an_unknown_direction_is_rejected(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, [
        "run", "--structure", str(structure), "--direction", "sideways"])
    assert result.exit_code == 1
    assert "central" in result.stdout


def test_an_unknown_write_modes_is_rejected(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, [
        "run", "--structure", str(structure), "--write-modes", "some"])
    assert result.exit_code == 1
    assert "imaginary" in result.stdout


def test_a_bad_indices_string_is_rejected_before_any_force_call(
        structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, [
        "run", "--structure", str(structure), "--indices", "0,99"])
    assert result.exit_code == 1
    assert "out of range" in result.stdout
    assert not (structure.parent / "freq_frequencies.csv").exists()


def test_the_echo_reports_fmax_at_the_input_geometry(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    assert "fmax at the input geometry" in result.stdout


def test_committee_with_an_explicit_mlip_is_rejected(structure, tmp_path):
    committee_file = tmp_path / "committee.yaml"
    committee_file.write_text("members: []\n")
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(committee_file), "--mlip", "mace"])
    assert result.exit_code == 1
    assert "--committee" in result.stdout
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `pytest tests/test_cli_freq.py -v`
Expected: collection error, `No module named 'mliprun.cli.commands.freq'`.

- [ ] **Step 3: Write the command**

Create `src/mliprun/cli/commands/freq.py`, following the shape of `singlepoint.py` from Task 3. The parts that differ:

Imports: the same set as `singlepoint.py`, plus `parse_indices`,
`run_frequencies`, `VALID_DIRECTIONS`, `VALID_METHODS` and `VALID_NFREE`
from `mliprun.core.vibrations`.

```python
app = typer.Typer(
    help="Vibrational frequencies by finite differences of forces.")


@app.command()
def run(
    ctx: typer.Context,
    structure: Path = typer.Option(..., prompt=True,
                                   help="Structure file (.vasp)"),
    mlip: str = typer.Option("auto", help=MLIP_HELP),
    uma_task: str = typer.Option(None, help=UMA_TASK_HELP),
    device: str = typer.Option("auto", help=DEVICE_HELP),
    mace_head: str = typer.Option(None, help=MACE_HEAD_HELP),
    sevennet_task: str = typer.Option(None, help=SEVENNET_TASK_HELP),
    committee: Path = typer.Option(
        None, "--committee",
        help="Path to a committee.yaml. Builds one Hessian per member from "
             "a single displacement sweep and reports each member's "
             "frequencies plus the per-mode spread. Mutually exclusive with "
             "--mlip and the head/task options."),
    member_timeout: float = typer.Option(
        DEFAULT_CALC_TIMEOUT_S, "--member-timeout",
        help="Seconds a single committee member may take per single-point "
             "before it is killed and the run aborts."),
    uncertainty_threshold: float = typer.Option(
        None, "--uncertainty-threshold",
        help="Flag the input geometry when the committee's force "
             "disagreement over the free atoms exceeds this (eV/Å). NO "
             "DEFAULT: without it sigma is reported and no verdict "
             "asserted."),
    indices: str = typer.Option(
        None, "--indices",
        help="Atoms to displace, e.g. '0,1,5' or '12-30' (inclusive), or a "
             "mix. Default: every atom not held by a FixAtoms constraint, "
             "which is the structure's own answer."),
    delta: float = typer.Option(0.01, help="Displacement in Å"),
    nfree: int = typer.Option(
        2, help="2 (three-point) or 4 (five-point stencil, twice the cost)"),
    direction: str = typer.Option(
        "central", help="central, forward or backward"),
    method: str = typer.Option(
        "standard",
        help="standard, or frederiksen for the acoustic sum-rule correction"),
    write_modes: str = typer.Option(
        "imaginary", "--write-modes",
        help="Which modes get an animated trajectory: none, imaginary "
             "(default) or all."),
    expect_fmax: float = typer.Option(
        None, "--expect-fmax",
        help="Warn when fmax at the input geometry exceeds this (eV/Å). "
             "Never stops the run. Default: the fmax a converged optimize "
             "stage in the structure's directory actually met, if there is "
             "one."),
    prefix: str = typer.Option("freq",
                               help="Stem for this command's output files"),
):
    """Compute vibrational frequencies and the zero-point energy."""
    atoms = read(structure)
    typer.echo(f"📂 Loaded structure: {structure.name}")
    typer.echo(f"   Atoms: {len(atoms)}, "
               f"Formula: {atoms.get_chemical_formula()}")

    # Validated here rather than deep in the engine, so a typo costs nothing
    # instead of failing after the first displacement.
    if nfree not in VALID_NFREE:
        typer.echo(f"❌ --nfree must be 2 or 4, got {nfree}.")
        raise typer.Exit(1)
    if direction not in VALID_DIRECTIONS:
        typer.echo(f"❌ --direction must be one of "
                   f"{', '.join(VALID_DIRECTIONS)}; got {direction!r}.")
        raise typer.Exit(1)
    if method not in VALID_METHODS:
        typer.echo(f"❌ --method must be one of "
                   f"{', '.join(VALID_METHODS)}; got {method!r}.")
        raise typer.Exit(1)
    if write_modes not in ("none", "imaginary", "all"):
        typer.echo("❌ --write-modes must be none, imaginary or all; "
                   f"got {write_modes!r}.")
        raise typer.Exit(1)
    try:
        chosen = parse_indices(indices, n_atoms=len(atoms))
    except ValueError as exc:
        typer.echo(f"❌ {exc}")
        raise typer.Exit(1)

    committee_config = None
    committee_calc = None
    if committee is not None:
        reject_conflicting_options(ctx)
        try:
            committee_config = load_committee(committee)
        except CommitteeConfigError as exc:
            typer.echo(f"❌ {exc}")
            raise typer.Exit(1)
        mlip = "committee"
        typer.echo(f"🧠 Committee of {len(committee_config.members)} members "
                   f"from {committee}")
        for spec in committee_config.members:
            typer.echo(f"   {spec.name}: {spec.mlip} "
                       f"[{spec.level_of_theory}] gpu={spec.gpu} "
                       f"env={spec.env}")
        if committee_config.mixed_theory:
            typer.echo(f"\n⚠️  {committee_config.mixed_theory_warning()}\n")
    else:
        if mlip == "auto":
            mlip = detect_mlip()
            typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
        else:
            typer.echo(f"🧠 Using MLIP: {mlip}")
        validate_mlip(mlip, sevennet_task, uma_task, mace_head)

    output_dir = structure.parent

    run_context = RunContext(
        command="freq",
        mode="one-off",
        param_sources=param_sources_from_ctx(ctx),
    )
    run_context.extra_inputs = {
        "structure": structure.name,
        "structure_abspath": str(structure.resolve()),
    }

    with contextlib.ExitStack() as session:
        if committee_config is not None:
            committee_calc = session.enter_context(
                started_committee(committee_config, output_dir,
                                  member_timeout, atoms))
            atoms.calc = committee_calc
        else:
            typer.echo(f"⚙️  Attaching {mlip} calculator (device={device})...")
            atoms = setup_calculator(atoms, mlip, uma_task, device=device,
                                     mace_head=mace_head,
                                     sevennet_task=sevennet_task)

        typer.echo(f"\n〰️  Displacing "
                   f"{len(chosen) if chosen is not None else 'the free'} "
                   f"atoms, delta = {delta} Å, nfree = {nfree}\n")

        results = run_frequencies(
            atoms=atoms,
            output_dir=output_dir,
            prefix=prefix,
            model_name=mlip,
            indices=chosen,
            delta=delta,
            nfree=nfree,
            direction=direction,
            method=method,
            write_modes=write_modes,
            expect_fmax=expect_fmax,
            structure_dir=structure.parent,
            run_context=run_context,
            device_requested=device,
            device_resolved=_resolve_device(device),
            uma_task=uma_task,
            mace_head=mace_head,
            sevennet_task=sevennet_task,
            committee=committee_calc,
            committee_config=committee_config,
            uncertainty_threshold=uncertainty_threshold,
        )

    typer.echo(f"\n〰️  {results['n_modes']} modes over "
               f"{results['n_displaced_atoms']} displaced atoms "
               f"({results['n_force_calls']} force calls)")
    typer.echo(f"   ZPE: {results['zpe_eV']:.6f} eV")
    typer.echo(f"   fmax at the input geometry: "
               f"{results['fmax_at_input_free_eV_per_A']:.6f} eV/Å")
    if results["fmax_expectation_source"] == "none":
        typer.echo("   (no relaxation provenance next to this structure, so "
                   "nothing to compare it against)")
    elif results["fmax_warning"]:
        typer.echo(
            f"\n⚠️  That is above the {results['fmax_expectation']:.6f} eV/Å "
            f"expected from {results['fmax_expectation_source']}. A geometry "
            f"that is not a stationary point produces spurious imaginary "
            f"modes; the run continued.")
    if results["n_imaginary"]:
        typer.echo(f"\n⚠️  {results['n_imaginary']} imaginary mode(s). ZPE "
                   f"above counts the real modes only.")
    if results["unhandled_constraints"]:
        typer.echo(
            f"\n⚠️  Constraint type(s) "
            f"{', '.join(results['unhandled_constraints'])} could not be "
            f"honoured: ASE selects whole atoms, so those atoms were "
            f"displaced in full and their held components are in the "
            f"Hessian as if free. Use --indices to exclude them.")
```

Then the output listing and the committee echo:

```python
    typer.echo("\n✅ Frequencies complete. Output files:")
    written = [f"{prefix}_frequencies.csv", f"{prefix}_summary.txt",
               f"{prefix}_vibrations.json", "mliprun_run.json"]
    if committee_config is not None:
        written.insert(1, f"{prefix}_committee_frequencies.csv")
    for name in written:
        typer.echo(f"   📄 {(output_dir / name).resolve()}")
    # Globbed, not predicted: --write-modes imaginary on a clean minimum
    # writes nothing, and a listing that names a file which is not there is
    # worse than a short listing.
    for path in sorted(output_dir.glob(f"{prefix}.*.traj")):
        typer.echo(f"   📄 {path.resolve()}")
    typer.echo(f"   📁 {(output_dir / prefix).resolve()}  "
               f"(displacement cache — delete to force a full recompute)")

    report_committee_uncertainty(committee_calc)
```

Register in `main.py`:

```python
app.add_typer(freq.app, name="freq",
              help="Vibrational frequencies by finite differences")
```

and in `pyproject.toml`:

```toml
freq = "mliprun.cli.commands.freq:app"
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_cli_freq.py -v`
Expected: all PASS.

Run: `pytest tests/test_cli_freq.py -q 2>&1 | tail -2`
Expected: `11 passed`, **`0 skipped`** — the `nfree_*` parametrize ids are neutral by construction, but check anyway.

Run: `pip install -e . -q && mlip freq run --help && freq run --help`
Expected: both print help, exit 0.

- [ ] **Step 5: Run the committee frequency tests that needed this module**

Run: `pytest tests/test_committee_vibrations.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mliprun/cli/commands/freq.py src/mliprun/cli/main.py pyproject.toml tests/test_cli_freq.py
git commit -m "feat(freq): mlip freq run

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 14: Documentation and PR C

**Files:**
- Modify: `docs/OUTPUTS.md`, `docs/PYTHON_API.md`, `README.md`, `AGENTS.md`, `CHANGELOG.md`

- [ ] **Step 1: Document the outputs**

In `docs/OUTPUTS.md`, add a `freq` section covering:
- Every output file: `<prefix>_frequencies.csv` (column table), `<prefix>_summary.txt`, `<prefix>_vibrations.json`, `<prefix>/` (the displacement cache), `<prefix>.<n>.traj`, and `<prefix>_committee_frequencies.csv`.
- That the frequency column is a **magnitude plus a boolean**, never a signed number, and why.
- That **ZPE counts the real modes only**: ASE sums the real parts, so an imaginary mode contributes zero, and a structure with imaginary modes has no well-defined zero-point energy.
- The stationary-point warning and its three sources, verbatim from the spec.
- That any constraint other than `FixAtoms` warns and the atoms are displaced in full.
- The cost formula: `1 + 6 × n_displaced` at `nfree=2`, `1 + 12 × n_displaced` at `nfree=4`.
- Restart: the displacement cache makes it free, for committee runs too.
- Add `freq` to the stage-kind list.

Under committee outputs, add the **two caveats verbatim from the spec**, because they are the difference between a number and a misleading number:
- Mode pairing is by index; the `<member>_overlap` columns are how an ordering swap becomes visible, and below 0.9 the run warns.
- At `delta = 0.01 Å` a merely noisy member contributes to the spread alongside genuine model disagreement, and one sweep cannot separate them. Running at two values of `--delta` distinguishes them.

In `docs/PYTHON_API.md`, add `run_frequencies` with a runnable example, and a short note that `<prefix>_vibrations.json` reloads through `VibrationsData.read` and carries the full Hessian — which is what makes free energies later cost no forces.

In `README.md` and `AGENTS.md`, add `freq` to the command lists.

In `CHANGELOG.md` under `## [Unreleased]` → `### Added`, describe the command including: the constraint-driven selection and the warning for what cannot be honoured, the stationary-point warning and that it never refuses, committee frequencies and the one-sweep cost, the mode-overlap diagnostic, and that thermochemistry is deliberately absent but reachable from the written `VibrationsData`.

- [ ] **Step 2: Verify**

Run: `grep -n "freq" docs/OUTPUTS.md docs/PYTHON_API.md README.md AGENTS.md CHANGELOG.md | head -20`
Expected: hits in all five.

Run: `pytest -m "not uma and not mace and not sevenn" -q 2>&1 | tail -3`
Expected: full suite green. Record the passed count; it should exceed the PR B count by the number of tests added in Tasks 7–13.

- [ ] **Step 3: Check diff coverage**

Run: `coverage run -m pytest -m "not uma and not mace and not sevenn" -q && coverage combine 2>/dev/null; coverage xml -q && diff-cover coverage.xml --compare-branch=main --fail-under=90`
Expected: ≥ 90%.

- [ ] **Step 4: Commit and open PR C**

```bash
git add docs/ README.md AGENTS.md CHANGELOG.md
git commit -m "docs(freq): outputs, python API, changelog

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push -u origin feat/freq
gh pr create --draft --title "feat(freq): vibrational frequencies with committee spread" --body "$(cat <<'EOF'
`mlip freq run` computes vibrational frequencies by finite differences of
forces, reporting frequencies, the imaginary-mode count and the zero-point
energy. Which atoms are displaced comes from the structure's own FixAtoms
constraints; `--indices` overrides.

Two things warn and never refuse: a constraint type ASE's whole-atom
`indices` cannot express (the atoms are displaced in full and their held
components enter the Hessian as if free), and an input geometry that is
not a stationary point (compared against `--expect-fmax`, or the fmax a
converged optimize stage in the same directory actually met).

With `--committee`, one displacement sweep yields one Hessian per member:
per-member frequencies, per-member ZPE and the per-mode spread. Members
are queried concurrently, so the cost is max(member), not sum(member).
The Hessian assembly is pinned to ASE's own by exact equality.

Mode pairing is by index, with a per-mode overlap column per member so an
ordering swap between near-degenerate modes is visible rather than
absorbed into the spread.

Thermochemistry is deliberately out of scope. The command writes
VibrationsData JSON carrying the full Hessian, so free energies later cost
no forces.

All tests run with no MLIP installed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Verify the PR body landed**

Run: `gh pr view --json body --jq '.body' | head -5`
Expected: the body above. If empty, patch with `gh api` and check again — `gh pr edit` fails silently against Projects-classic.

---

---

# Verification on cos-cluster

### Task 15: Verify both commands against real MLIPs on cos-cluster

Runs with PR C's branch checked out, which contains PR B's work. Do this **before** either PR is marked ready for review: a green unit suite on EMT is not evidence that a frequency is a frequency.

**Files:** none in the repo. The deliverable is a written verification report appended to `CHANGELOG.md`'s entry and quoted in the PR bodies, plus the numbers recorded in the prov ledger (canon K2).

**Model and head, which canon C1 says are never guessed:**
- Slab and adsorbate legs: `uma-s-1p2` with `--uma-task oc20`. Surface + adsorbate work, per the cos-cluster skill's lesson 1.
- Gas-molecule leg: `uma-s-1p2` with `--uma-task omol`.
- These two legs are **separate comparisons**. No energy difference is ever taken between them — canon C3, one head per energy formula. A frequency is a curvature, not an energy difference, so using two heads across two independent legs is legitimate; combining their numbers into one table would not be.
- Committee members: `uma-s-1p2` (oc20), `mace`, `chgnet`. Mixed level of theory by construction, which the run will say out loud. That is the point: it exercises the mixed-theory path, and the spread is a method comparison, not an error bar.

**If Juan has not confirmed those choices, stop and ask before running anything.** They are his call, and a verification run reported under a head nobody chose is worse than no verification.

- [ ] **Step 1: Reset the cos-cluster test clone, which is in a known mixed state**

`.claude/handoffs/committee-followups.md` item 4 records it: `/scratchb/juar/committee-test/mliprun` sits on commit `321013b` with two files hand-copied over it during the PR #47 session. It is neither `main` nor a clean branch. Reusing it would test a chimera.

```bash
mkdir -p /tmp/ssh_mux && ssh -O check cos-cluster
```

If no master is running, do **not** authenticate non-interactively — ask Juan to run `! ssh -fN cos-cluster` and type the password, then re-check.

```bash
ssh cos-cluster 'cd /scratchb/juar/committee-test/mliprun && git status --short && git rev-parse HEAD'
```

Expected: the two modified files and `321013b`. Then reset it:

```bash
ssh cos-cluster 'cd /scratchb/juar/committee-test/mliprun && git fetch --all --prune && git checkout -- . && git checkout feat/freq && git rev-parse HEAD && git status --short'
```

Expected: the branch tip, and **empty** `git status --short`. A non-empty status here means the reset did not take; stop and report rather than running on a dirty tree.

- [ ] **Step 2: Confirm the PYTHONPATH injection actually wins**

Never switch branches on the shared editable install at `/app1-cos/mlip-platform/mlip-platform` — it is the tool every venv and every running job on the node is using.

```bash
ssh cos-cluster 'PYTHONPATH=/scratchb/juar/committee-test/mliprun/src /scratchb/juar/EWaste2GreenCat/explicit_solvation/.venv/uma/bin/python -c "import mliprun, mliprun.core.vibrations as v; print(mliprun.__file__); print(v.__file__)"'
```

Expected: **both** paths under `/scratchb/juar/committee-test/mliprun/src`. If either points at `/app1-cos`, the venv's editable install has changed from the legacy `.pth` style to the modern import-hook finder, `PYTHONPATH` has silently lost, and every number below would describe the wrong code. Stop and report.

Also confirm the shared install was not disturbed:

```bash
ssh cos-cluster 'cd /app1-cos/mlip-platform/mlip-platform && git branch --show-current && git status --short'
```

Expected: `main`, empty status.

- [ ] **Step 3: Pre-flight the shared node**

It has no scheduler and everyone shares it. Overcommitting can take the node down for other users.

```bash
ssh cos-cluster '/app1-cos/checkNodeUserStatus.sh; nvidia-smi; top -bn1 | head -5; df -h /scratchb'
```

Record which GPU is free. If all three are busy with someone else's work, stop and report rather than adding load.

- [ ] **Step 4: `singlepoint` against `optimize --max-steps 0`, the exact-equality cross-check**

Both do one evaluation of the same geometry with the same model. Their energies must agree to the last bit; anything else means one of them is not doing what it claims.

Stage a relaxed O/Pt(111) slab (the structure PR #50's README figure used) into `/scratchb/juar/freq-verify/sp/`, then:

```bash
ssh cos-cluster 'cd /scratchb/juar/freq-verify/sp && PYTHONPATH=/scratchb/juar/committee-test/mliprun/src /scratchb/juar/EWaste2GreenCat/explicit_solvation/.venv/uma/bin/mlip singlepoint run --structure POSCAR --mlip uma-s-1p2 --uma-task oc20 --device cuda 2>&1 | tail -20'
```

Then the same structure through the old path, in a separate directory so the records do not collide, and compare:

```bash
ssh cos-cluster 'cd /scratchb/juar/freq-verify/sp_opt && PYTHONPATH=... mlip optimize run --structure POSCAR --mlip uma-s-1p2 --uma-task oc20 --device cuda --max-steps 0 2>&1 | tail -5'
```

Record both energies to full precision from the two `mliprun_run.json` files. Expected: **identical**. Record the delta whatever it is; a non-zero delta is a finding, not a tolerance to widen.

Also check by hand: `singlepoint_forces.csv` has one row per atom, the `free_*` columns are False exactly on the fixed layers, and `fmax_all` exceeds `fmax_free`.

- [ ] **Step 5: `freq` on a gas molecule against a known number**

The first real physics check. A CO molecule's stretch is ~2143 cm⁻¹ experimentally.

```bash
ssh cos-cluster 'cd /scratchb/juar/freq-verify/co && PYTHONPATH=... mlip freq run --structure CO.vasp --mlip uma-s-1p2 --uma-task omol --device cuda 2>&1 | tail -20'
```

Relax it first with the same model and head, or the stationary-point warning will fire and the number will be wrong for a reason the command already told you about.

Expected: one large real mode, the rest near zero (translations and rotations), no imaginary modes. **Record the stretch and its delta from 2143 cm⁻¹. Do not assert a verdict on whether the delta is acceptable** — that is a question about the model, not about this code, and it is Juan's to judge.

- [ ] **Step 6: `freq` on a constrained slab — the constraint path against a real structure**

The EMT tests prove `select_indices` reads `FixAtoms`. This proves it survives a real VASP POSCAR's selective dynamics, which is how constraints actually reach this tool (canon S8: the constraint is invisible on the command line).

```bash
ssh cos-cluster 'cd /scratchb/juar/freq-verify/slab && PYTHONPATH=... mlip freq run --structure CONTCAR --mlip uma-s-1p2 --uma-task oc20 --device cuda 2>&1 | tail -25'
```

Check: `n_displaced_atoms` equals the number of atoms free in the POSCAR's selective dynamics; `n_force_calls` equals `1 + 6 × n_displaced`; `n_modes` equals `3 × n_displaced`. Record the wall clock.

Check the stationary-point line: the directory holds the `mliprun_run.json` from the relaxation that produced this CONTCAR, so `fmax_expectation_source` should read `run_record` and name the fmax that run converged to. **This is the only place the run-record lookup is exercised against a record it did not write itself.**

- [ ] **Step 7: The transition-state case, opportunistic**

If a converged NEB saddle from an earlier project is available, run `freq` on it. Expected: **exactly one imaginary mode**, along the reaction coordinate, and `freq.<n>.traj` written for it. This is the use case the imaginary-mode default exists for.

If no saddle is to hand, say so in the report rather than skipping it silently. Do not construct one for this purpose — that is a calculation, not a verification.

- [ ] **Step 8: Committee frequencies with three real MLIPs in three real environments**

The thing no EMT test can reach: three packages with mutually incompatible torch pins, driven as subprocess workers from a driver with no torch at all.

Build the driver venv if it is not there (ADR 0001 — it must have no MLIP):

```bash
ssh cos-cluster 'cd /scratchb/juar/freq-verify && python -m venv .venv/driver && .venv/driver/bin/pip install -q -e /scratchb/juar/committee-test/mliprun && .venv/driver/bin/python -c "import torch" 2>&1 | tail -1'
```

Expected: `ModuleNotFoundError: No module named 'torch'` — that is the architecture working, not a failure.

Write `committee.yaml` with the three members named above, each pointing at its own venv's python, then run `freq --committee` on a **small** adsorbate-only selection — use `--indices` to displace just the adsorbate. A full slab at three members is a long run and this step is about correctness, not endurance.

Check every one of these:
- `freq_committee_frequencies.csv` exists, one `<member>_cm-1` column per member, one `<member>_overlap` column per member.
- The mixed-theory warning was printed. These three members are deliberately not same-level.
- `frequency_committee_cm-1` matches `freq_frequencies.csv` row for row — the headline is the mean potential's, not the mean of members (D8).
- `worst_mode_overlap` and `mode_pairing_suspect` in the run record. **Record the worst overlap.** If it is below 0.9 the run warned; that is the diagnostic doing its job on real near-degenerate modes, and the number matters more than the flag.
- Per-member ZPE and `zpe_std_eV`. Record them.

- [ ] **Step 9: Measure the cost claim instead of asserting it**

The design says cost is `max(member)`, not `sum(member)`. Measure it.

Time the three-member committee sweep from Step 8. Time the same sweep with one member alone, for each member. Expected: the committee wall clock is close to the slowest single member's, not to their sum. **Record all four numbers and the ratio.** If the committee time approaches the sum, the concurrency claim in the design note and in `docs/OUTPUTS.md` is wrong and must be corrected there before the PR merges.

- [ ] **Step 10: Measure whether `delta = 0.01 Å` is above the model's force noise**

The docs currently say a merely noisy member contributes to the spread alongside genuine disagreement, and that two values of `--delta` distinguish them. That is reasoning. Turn it into a number.

Re-run Step 8's committee selection at `--delta 0.02`, and re-run it at `--delta 0.01` into a fresh directory (delete the `freq/` cache, or it will reuse the first sweep and prove nothing — **this is the easiest way to get a meaningless result in this whole task**).

Compare the per-mode spread at the two deltas. Record both. If the spread is roughly unchanged, it is model disagreement. If it shrinks markedly at the larger delta, part of what the smaller delta reported was numerical noise, and the docs' caveat needs the measured number written into it.

- [ ] **Step 11: Teardown on the committee path, unsandboxed**

`freq --committee` is a new caller of the committee session extracted in Task 1. The SIGTERM path is the one whose failure left workers holding CUDA contexts on this exact node.

Start a committee `freq` run, note the driver pid, `kill` it, then check:

```bash
ssh cos-cluster 'pgrep -af "committee.worker"; nvidia-smi'
```

Expected: no worker process, no CUDA context left by this run, exit code 143.

**Run this check unsandboxed.** `ps` and `pgrep` are blocked inside the Claude Bash sandbox and fail in a way that reads as empty output — a check that only tests for an empty result would report success on a blocked syscall. Pass `dangerouslyDisableSandbox: true` for this step specifically, or have Juan run it with `!`.

- [ ] **Step 12: Leave the node as you found it, and record the run**

```bash
ssh cos-cluster 'cd /app1-cos/mlip-platform/mlip-platform && git branch --show-current && git status --short'
ssh cos-cluster 'nvidia-smi'
```

Expected: `main`, empty status, and no GPU memory held by any process of this verification.

Then record the calculations in the prov ledger (canon K2): every MLIP calculation outside catplat's management is recorded before its numbers are reported or used. Use the project root the ledger already knows for this work; a deeper root silently doubles the ledger under a second id namespace.

- [ ] **Step 13: Write the verification report**

Into the PR bodies of both PRs and into the `CHANGELOG.md` entries. It states, with numbers:

- The energy delta between `singlepoint` and `optimize --max-steps 0` (expected exactly zero).
- The CO stretch and its delta from 2143 cm⁻¹, with **no verdict** on whether the delta is acceptable.
- The slab run's displaced-atom count, force-call count, and wall clock.
- Whether the run-record fmax lookup fired, and against what value.
- The transition state's imaginary-mode count, or an explicit statement that no saddle was available.
- The committee's three timings, the ratio, and whether `max(member)` held.
- The worst mode overlap and the per-member ZPE spread.
- The per-mode spread at both deltas and what that says about force noise.
- The teardown result.

Anything that did not run says so by name. A verification report with a silent gap is worse than a short one.

## Verification before claiming done

Run all of these and read the output before reporting anything complete:

```bash
pytest -m "not uma and not mace and not sevenn" -q 2>&1 | tail -3
mlip singlepoint run --help >/dev/null && echo "singlepoint OK"
mlip freq run --help >/dev/null && echo "freq OK"
gh pr list --draft --json number,title
```

The goal's `done-when` is met only when: `mlip singlepoint` reports energy, forces and both fmax values; `mlip freq` reports frequencies and ZPE; both work with a committee, `freq` reporting per-member frequencies and the per-mode spread; the unit suite passes with no MLIP installed; **and Task 15's cos-cluster verification has run against real MLIPs with its report written.**

**Still unratified by Juan, carried forward for review at PR time:** the `--expect-fmax` default reading a converged `optimize` stage out of the input directory's run record, and `MODE_OVERLAP_WARN = 0.9`.

## What EMT cannot prove

Every test in Tasks 1–14 runs on ASE's EMT in one process. That proves the plumbing and says nothing about the physics, about MLIP packages in genuinely separate environments, or about cost at real repetition rates. Task 15 closes that gap on cos-cluster and is part of this plan, not a follow-up.
