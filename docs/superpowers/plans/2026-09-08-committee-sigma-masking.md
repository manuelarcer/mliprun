# Committee sigma masking + opt-in threshold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the committee's force-disagreement statistic exclude constrained force components, and stop asserting a pass/fail verdict about it by default.

**Architecture:** Two changes to one subsystem. (1) `committee_statistics` gains an optional `(N, 3)` free-component mask, built driver-side from `atoms.constraints`, so `sigma_max` and `sigma_mean` are taken over the same atoms ASE's `fmax` uses; the unmasked numbers are kept alongside under `*_all` names. (2) `--uncertainty-threshold` stops defaulting to `fmax`; with no threshold the run reports the numbers and `flagged` is `null` rather than `false`. Nothing touches the forces the optimizer sees — the committee stays a proper potential.

**Tech Stack:** Python 3, numpy, ASE (`ase.constraints.FixAtoms` / `FixCartesian`), typer CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-committee-sigma-masking-design.md`
(amends `docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md`)

## Global Constraints

- **No MLIP may be required by any test.** Every test in this plan runs on a laptop with no UMA/MACE/SevenNet/CHGNet installed. ASE is a hard dependency and is fine to import.
- **Test-first.** Every test is watched failing before its implementation is written. A test that passes on first run has not tested anything and must be corrected.
- **Draft PR only.** Branch off `main`, open the PR as a draft. Juan merges.
- **Frozen goldens.** No file in `tests/goldens/` may change. Verified 2026-09-08: none of the four contains `sigma`, so a golden diff means something went wrong.
- **Diff coverage ≥ 90 % on changed lines.** Arm subprocess coverage locally or the gate under-reports:
  ```bash
  python -c "import pathlib,sysconfig; p=pathlib.Path(sysconfig.get_paths()['purelib'])/'mliprun-subprocess-coverage.pth'; p.write_text('import coverage; coverage.process_startup()\n')"
  COVERAGE_PROCESS_START=$PWD/pyproject.toml pytest --cov=mliprun --cov-report=xml
  diff-cover coverage.xml --compare-branch=origin/main --fail-under=90 --exclude setup.py --exclude "scripts/*"
  ```
- **No `pragma: no cover`.** The repo uses none; do not introduce the convention.
- **Run record `SCHEMA_VERSION` goes 4 → 5** (Task 5). Five test sites hardcode `4` and are named in that task.
- **Naming, fixed across tasks.** `sigma_max` / `sigma_mean` / `worst_atom` are the free-atom (headline) values. `sigma_max_all` / `sigma_mean_all` / `worst_atom_all` are the pre-change unmasked values. `sigma_per_atom` stays all-component; `sigma_per_atom_free` is the masked one.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/mliprun/core/committee/calculator.py` | `free_component_mask`, masked `committee_statistics`, `CommitteeCalculator._evaluate`, `CommitteeTraceWriter`, `write_peratom_sigma`, `uncertainty_summary` | 1–4 |
| `src/mliprun/core/optimize.py` | Threshold resolution, record parameters, terminal-independent logging | 5 |
| `src/mliprun/core/run_record.py` | `SCHEMA_VERSION` | 5 |
| `src/mliprun/cli/commands/optimize.py` | `--uncertainty-threshold` help, terminal reporting | 6 |
| `tests/test_committee_stats.py` | Statistics arithmetic and the mask helper | 1 |
| `tests/test_committee_calculator.py` | Mask reaching the statistic through the calculator | 2 |
| `tests/test_committee_outputs.py` | CSV columns, `uncertainty_summary` | 3, 4 |
| `tests/test_committee_optimize.py` | `run_optimization` threshold resolution, record | 5 |
| `tests/test_committee_cli.py` | CLI flag and echo | 6 |
| `docs/OUTPUTS.md`, `docs/PYTHON_API.md`, `CHANGELOG.md` | Documentation | 7 |

Task 7 keeps documentation together rather than folding it into Tasks 3–6, because `docs/OUTPUTS.md` describes the output set as one coherent document and four partial edits to it would conflict.

---

### Task 1: Free-component mask and masked statistics

**Files:**
- Modify: `src/mliprun/core/committee/calculator.py:19-82` (`committee_statistics`)
- Test: `tests/test_committee_stats.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `free_component_mask(atoms) -> tuple[numpy.ndarray, list[str]]` — an `(N, 3)` bool array (True = free) and the sorted names of unhandled constraint types.
  - `committee_statistics(energies, forces, free_mask=None) -> dict` — the existing keys plus `sigma_per_atom_free` (N,), `sigma_max_all` (float), `sigma_mean_all` (float), `worst_atom_all` (int), `n_free_atoms` (int), `all_constrained` (bool).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_committee_stats.py`:

```python
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian

from mliprun.core.committee.calculator import free_component_mask


def _two_atoms_sigma_on_x(sigma_0, sigma_1):
    """Two members, disagreement on x only.

    For two samples, std(ddof=1) of {0, d} is d/sqrt(2), so a component set
    to sigma*sqrt(2) gives exactly sigma -- the same construction the
    existing 3-4-5 test uses.
    """
    forces = np.zeros((2, 2, 3))
    forces[1, 0, 0] = sigma_0 * np.sqrt(2)
    forces[1, 1, 0] = sigma_1 * np.sqrt(2)
    return forces


class TestFreeComponentMask:
    def test_fixatoms_frees_no_component_of_its_atoms(self):
        atoms = Atoms("H3", positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        atoms.set_constraint(FixAtoms(indices=[0, 2]))
        mask, unhandled = free_component_mask(atoms)
        assert mask[0].tolist() == [False, False, False]
        assert mask[1].tolist() == [True, True, True]
        assert mask[2].tolist() == [False, False, False]
        assert unhandled == []

    def test_fixcartesian_frees_the_directions_it_does_not_hold(self):
        atoms = Atoms("H2", positions=[(0, 0, 0), (1, 0, 0)])
        atoms.set_constraint(FixCartesian([0], mask=(False, False, True)))
        mask, unhandled = free_component_mask(atoms)
        assert mask[0].tolist() == [True, True, False]
        assert mask[1].tolist() == [True, True, True]
        assert unhandled == []

    def test_no_constraints_leaves_everything_free(self):
        atoms = Atoms("H2", positions=[(0, 0, 0), (1, 0, 0)])
        mask, unhandled = free_component_mask(atoms)
        assert mask.all()
        assert unhandled == []

    def test_an_unhandled_constraint_leaves_atoms_free_and_is_named(self):
        """Over-reporting sigma is the safe direction; the type name travels
        so the fallback is visible rather than silent."""
        from ase.constraints import FixBondLength
        atoms = Atoms("H2", positions=[(0, 0, 0), (1, 0, 0)])
        atoms.set_constraint(FixBondLength(0, 1))
        mask, unhandled = free_component_mask(atoms)
        assert mask.all()
        assert unhandled == ["FixBondLength"]


class TestConstrainedComponentsAreMasked:
    def test_a_fixed_atom_is_excluded_from_sigma_max(self):
        """Atom 0 disagrees ten times more, but cannot move."""
        mask = np.array([[False, False, False], [True, True, True]])
        stats = committee_statistics([0.0, 0.0],
                                     _two_atoms_sigma_on_x(0.9, 0.1),
                                     free_mask=mask)
        assert stats["sigma_max"] == pytest.approx(0.1, abs=1e-12)
        assert stats["worst_atom"] == 1
        assert stats["n_free_atoms"] == 1

    def test_the_unmasked_numbers_are_kept_alongside(self):
        mask = np.array([[False, False, False], [True, True, True]])
        stats = committee_statistics([0.0, 0.0],
                                     _two_atoms_sigma_on_x(0.9, 0.1),
                                     free_mask=mask)
        assert stats["sigma_max_all"] == pytest.approx(0.9, abs=1e-12)
        assert stats["worst_atom_all"] == 0
        assert stats["sigma_mean_all"] == pytest.approx(0.5, abs=1e-12)

    def test_a_fixed_atom_does_not_dilute_sigma_mean(self):
        """The mean is over free atoms, not over free atoms plus zeros."""
        mask = np.array([[False, False, False], [True, True, True]])
        stats = committee_statistics([0.0, 0.0],
                                     _two_atoms_sigma_on_x(0.9, 0.1),
                                     free_mask=mask)
        assert stats["sigma_mean"] == pytest.approx(0.1, abs=1e-12)

    def test_a_partly_fixed_atom_keeps_its_free_components(self):
        """3 on x and 4 on z, z held: sigma is 3, not 5."""
        forces = np.zeros((2, 1, 3))
        forces[1, 0] = [3.0 * np.sqrt(2), 0.0, 4.0 * np.sqrt(2)]
        mask = np.array([[True, True, False]])
        stats = committee_statistics([0.0, 0.0], forces, free_mask=mask)
        assert stats["sigma_max"] == pytest.approx(3.0, abs=1e-12)
        assert stats["sigma_max_all"] == pytest.approx(5.0, abs=1e-12)

    def test_no_mask_reproduces_the_previous_numbers(self):
        stats = committee_statistics([0.0, 0.0],
                                     _two_atoms_sigma_on_x(0.9, 0.1))
        assert stats["sigma_max"] == pytest.approx(0.9, abs=1e-12)
        assert stats["sigma_max"] == pytest.approx(stats["sigma_max_all"])
        assert stats["worst_atom"] == 0
        assert stats["n_free_atoms"] == 2

    def test_every_component_constrained_falls_back_to_the_unmasked_numbers(self):
        """Possible for a constrained single point, not for a relaxation.
        Reporting nothing would be less useful than reporting the unmasked
        numbers and saying so."""
        mask = np.zeros((2, 3), dtype=bool)
        stats = committee_statistics([0.0, 0.0],
                                     _two_atoms_sigma_on_x(0.9, 0.1),
                                     free_mask=mask)
        assert stats["all_constrained"] is True
        assert stats["n_free_atoms"] == 0
        assert stats["sigma_max"] == pytest.approx(0.9, abs=1e-12)

    def test_a_mask_of_the_wrong_shape_is_rejected(self):
        with pytest.raises(ValueError, match="free_mask"):
            committee_statistics([0.0, 0.0], _two_atoms_sigma_on_x(0.9, 0.1),
                                 free_mask=np.ones((3, 3), dtype=bool))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_stats.py -v`
Expected: FAIL — `ImportError: cannot import name 'free_component_mask'`.

- [ ] **Step 3: Write the implementation**

In `src/mliprun/core/committee/calculator.py`, add above `committee_statistics`:

```python
def free_component_mask(atoms):
    """Which force components are free to move, as an ``(N, 3)`` bool array.

    Only ``FixAtoms`` and ``FixCartesian`` are masked. They are the stock ASE
    constraints whose ``adjust_forces`` is a pure component mask -- verified
    against ase 3.26.0, ``forces[index] = 0.0`` and
    ``forces[index] *= ~mask[None, :]`` respectively -- so the component they
    hold is exactly zero and dropping it from the statistic is unambiguous.

    Everything else in ``ase.constraints`` (``FixScaled``, ``FixedPlane``,
    ``FixedLine``, ``FixBondLength``) projects rather than masks, and a
    projection does not carry over to a standard deviation, which is not a
    vector. Those atoms stay free: sigma is over-reported rather than
    under-reported, and the type names are returned so the fallback reaches
    the run record instead of being silent.

    Returns
    -------
    (numpy.ndarray, list of str)
        The ``(N, 3)`` mask -- True where the component is free -- and the
        sorted names of the constraint types that were not masked.
    """
    mask = np.ones((len(atoms), 3), dtype=bool)
    unhandled = set()
    for constraint in getattr(atoms, "constraints", ()) or ():
        kind = type(constraint).__name__
        if kind == "FixAtoms":
            mask[np.asarray(constraint.index, dtype=int)] = False
        elif kind == "FixCartesian":
            mask[np.asarray(constraint.index, dtype=int)] &= ~np.asarray(
                constraint.mask, dtype=bool)
        else:
            unhandled.add(kind)
    return mask, sorted(unhandled)
```

Change the signature to `def committee_statistics(energies, forces, free_mask=None) -> dict:` and replace the reduction block (currently `calculator.py:71-82`) with:

```python
    sigma_components = forces.std(axis=0, ddof=1)                 # (N, 3)
    sigma_per_atom = np.linalg.norm(sigma_components, axis=1)     # (N,)
    worst_all = int(np.argmax(sigma_per_atom))

    if free_mask is None:
        free_mask = np.ones(sigma_components.shape, dtype=bool)
    else:
        free_mask = np.asarray(free_mask, dtype=bool)
        if free_mask.shape != sigma_components.shape:
            raise ValueError(
                f"free_mask must have shape {sigma_components.shape}, got "
                f"{free_mask.shape}")

    sigma_per_atom_free = np.linalg.norm(sigma_components * free_mask, axis=1)
    movable = np.flatnonzero(free_mask.any(axis=1))

    if movable.size:
        worst = int(movable[np.argmax(sigma_per_atom_free[movable])])
        sigma_max = float(sigma_per_atom_free[worst])
        sigma_mean = float(sigma_per_atom_free[movable].mean())
        all_constrained = False
    else:
        worst = worst_all
        sigma_max = float(sigma_per_atom[worst_all])
        sigma_mean = float(sigma_per_atom.mean())
        all_constrained = True

    return {
        "energy_mean": float(energies.mean()),
        "forces_mean": forces.mean(axis=0),
        "sigma_per_atom": sigma_per_atom,
        "sigma_per_atom_free": sigma_per_atom_free,
        "sigma_max": sigma_max,
        "sigma_mean": sigma_mean,
        "worst_atom": worst,
        "sigma_max_all": float(sigma_per_atom[worst_all]),
        "sigma_mean_all": float(sigma_per_atom.mean()),
        "worst_atom_all": worst_all,
        "n_free_atoms": int(movable.size),
        "all_constrained": all_constrained,
    }
```

Update the docstring's `Returns` block to list the new keys, and add a paragraph stating that `sigma_max`/`sigma_mean` are taken over free components only, pointing at the design note.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_stats.py -v`
Expected: PASS, including the pre-existing tests in the file (they pass no mask, so they exercise the `free_mask is None` path unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/committee/calculator.py tests/test_committee_stats.py
git commit -m "feat(committee): sigma excludes constrained force components"
```

---

### Task 2: Wire the mask through the calculator

**Files:**
- Modify: `src/mliprun/core/committee/calculator.py` (`CommitteeCalculator._evaluate`)
- Test: `tests/test_committee_calculator.py`

**Interfaces:**
- Consumes: `free_component_mask`, `committee_statistics(..., free_mask=)` from Task 1.
- Produces: `CommitteeCalculator.latest` carries the masked statistics plus `free_mask` (the `(N, 3)` bool array actually used) and `unhandled_constraints` (list of str). Task 3 reads `latest["free_mask"]` rather than recomputing it, so the CSV can never disagree with the statistic.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_committee_calculator.py`. It already defines `FakeMember(name, energy, forces)` at line 18 — reuse it, do not add a second stub. Add `from ase import Atoms` and `from ase.constraints import FixAtoms` to the imports.

```python
class TestConstraintsReachTheStatistic:
    """FakeMember answers from a fixed table, so the disagreement is exact
    and independent of geometry."""

    def _members(self, sigma_0, sigma_1):
        return [
            FakeMember("a", -1.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            FakeMember("b", -1.0, [[sigma_0 * np.sqrt(2), 0.0, 0.0],
                                   [sigma_1 * np.sqrt(2), 0.0, 0.0]]),
        ]

    def _atoms(self, fixed=()):
        atoms = Atoms("H2", positions=[(0, 0, 0), (1.0, 0, 0)])
        if len(fixed):
            atoms.set_constraint(FixAtoms(indices=list(fixed)))
        return atoms

    def test_a_fixed_atom_is_excluded_from_the_reported_sigma(self):
        """The disagreement lives on the fixed atom, so the free-atom sigma
        must be far smaller than the all-atom one."""
        atoms = self._atoms(fixed=[0])
        with CommitteeCalculator(self._members(0.9, 0.1)) as calc:
            calc.start()
            atoms.calc = calc
            atoms.get_potential_energy()
        assert calc.latest["sigma_max"] == pytest.approx(0.1, abs=1e-9)
        assert calc.latest["sigma_max_all"] == pytest.approx(0.9, abs=1e-9)
        assert calc.latest["n_free_atoms"] == 1

    def test_constraints_survive_ase_copying_the_atoms(self):
        """ASE's Calculator.calculate stores atoms.copy(); the mask is built
        from that copy, so this asserts the copy keeps the constraint."""
        atoms = self._atoms(fixed=[0])
        with CommitteeCalculator(self._members(1.0, 0.0)) as calc:
            calc.start()
            atoms.calc = calc
            atoms.get_potential_energy()
        assert calc.latest["n_free_atoms"] == 1
        assert calc.latest["free_mask"][0].tolist() == [False, False, False]

    def test_an_unconstrained_system_reports_every_atom_free(self):
        atoms = self._atoms()
        with CommitteeCalculator(self._members(1.0, 0.0)) as calc:
            calc.start()
            atoms.calc = calc
            atoms.get_potential_energy()
        assert calc.latest["n_free_atoms"] == 2
        assert calc.latest["unhandled_constraints"] == []

    def test_preflight_masks_the_same_way_calculate_does(self):
        atoms = self._atoms(fixed=[0])
        with CommitteeCalculator(self._members(0.9, 0.1)) as calc:
            calc.start()
            stats = calc.preflight(atoms)
        assert stats["sigma_max"] == pytest.approx(0.1, abs=1e-9)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_calculator.py -v -k Constraint`
Expected: FAIL — `KeyError: 'n_free_atoms'` or a sigma of 0.9 where 0.1 is asserted.

- [ ] **Step 3: Write the implementation**

In `CommitteeCalculator._evaluate`, replace the `stats = committee_statistics(ordered, stacked)` line with:

```python
        free_mask, unhandled = free_component_mask(atoms)
        ordered = [energies[m.name] for m in self.members]
        stats = committee_statistics(ordered, stacked, free_mask=free_mask)
        # Stored, not recomputed downstream: the per-atom CSV must describe
        # the same mask the statistic used, and `atoms` has moved on by the
        # time the run writes it.
        stats["free_mask"] = free_mask
        stats["unhandled_constraints"] = unhandled
        if unhandled:
            logger.warning(
                "committee sigma: constraint type(s) %s are not masked, so "
                "their atoms are counted as free and sigma is over-reported",
                ", ".join(unhandled))
        stats["energies"] = dict(energies)
```

(The existing `ordered = ...` line above the old call is replaced by this block, not duplicated.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_calculator.py tests/test_committee_stats.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/committee/calculator.py tests/test_committee_calculator.py
git commit -m "feat(committee): build the free-component mask from atoms.constraints"
```

---

### Task 3: CSV outputs carry the masked and unmasked numbers

**Files:**
- Modify: `src/mliprun/core/committee/calculator.py` (`CommitteeTraceWriter`, `write_peratom_sigma`)
- Modify: `src/mliprun/core/optimize.py` (the `write_peratom_sigma(...)` call site, currently around line 428)
- Test: `tests/test_committee_outputs.py`

**Interfaces:**
- Consumes: `latest` keys `sigma_max_all`, `n_free_atoms`, `sigma_per_atom_free` from Tasks 1–2.
- Produces:
  - `<stem>_committee.csv` gains columns `sigma_max_all_eV_per_A` and `n_free_atoms`, inserted after `worst_atom` and before `mixed_theory`.
  - `write_peratom_sigma(path, symbols, sigma_per_atom, sigma_free=None, free_mask=None)`; the file gains columns `sigma_free_eV_per_A` and `free_components`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_committee_outputs.py`:

```python
class TestTraceCarriesBothSigmas:
    def test_the_row_carries_the_masked_and_unmasked_maxima(self, tmp_path):
        writer = CommitteeTraceWriter(tmp_path / "t.csv", ["a", "b"], False)
        latest = {
            "energies": {"a": -1.0, "b": -3.0},
            "energy_mean": -2.0, "sigma_max": 0.1, "sigma_mean": 0.05,
            "worst_atom": 1, "sigma_max_all": 0.9, "n_free_atoms": 1,
        }
        row = writer.write_step(0, latest, fmax_value=0.04)
        writer.close()
        assert row["sigma_max_eV_per_A"] == pytest.approx(0.1)
        assert row["sigma_max_all_eV_per_A"] == pytest.approx(0.9)
        assert row["n_free_atoms"] == 1

    def test_the_header_names_the_new_columns(self, tmp_path):
        writer = CommitteeTraceWriter(tmp_path / "t.csv", ["a", "b"], False)
        writer.close()
        header = (tmp_path / "t.csv").read_text().splitlines()[0]
        assert "sigma_max_all_eV_per_A" in header
        assert "n_free_atoms" in header


class TestPerAtomFileMarksConstrainedAtoms:
    def test_every_atom_still_gets_a_row(self, tmp_path):
        """Seeing the frozen atoms is how a reader checks the masking on
        their own run, so constrained atoms are listed, not dropped."""
        path = tmp_path / "p.csv"
        write_peratom_sigma(path, ["Cu", "C"], [0.9, 0.1],
                            sigma_free=[0.0, 0.1],
                            free_mask=[[False, False, False],
                                       [True, True, True]])
        lines = path.read_text().splitlines()
        assert len(lines) == 3
        assert lines[1].split(",")[:2] == ["0", "Cu"]

    def test_the_free_component_count_is_recorded(self, tmp_path):
        path = tmp_path / "p.csv"
        write_peratom_sigma(path, ["Cu", "C"], [0.9, 0.1],
                            sigma_free=[0.0, 0.1],
                            free_mask=[[False, False, False],
                                       [True, True, True]])
        rows = _read(path)
        assert rows[0]["free_components"] == "0"
        assert rows[1]["free_components"] == "3"
        assert float(rows[0]["sigma_free_eV_per_A"]) == pytest.approx(0.0)

    def test_a_partly_fixed_atom_counts_its_free_directions(self, tmp_path):
        path = tmp_path / "p.csv"
        write_peratom_sigma(path, ["Cu"], [0.5], sigma_free=[0.3],
                            free_mask=[[True, True, False]])
        rows = _read(path)
        assert rows[0]["free_components"] == "2"

    def test_omitting_the_mask_treats_every_atom_as_free(self, tmp_path):
        """Keeps the Python API callable with three arguments."""
        path = tmp_path / "p.csv"
        write_peratom_sigma(path, ["Cu", "C"], [0.9, 0.1])
        rows = _read(path)
        assert rows[0]["free_components"] == "3"
        assert float(rows[0]["sigma_free_eV_per_A"]) == pytest.approx(0.9)
```

These use the file's existing `_read(path)` helper (line 29); no new import is needed.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_outputs.py -v -k "Trace or PerAtom"`
Expected: FAIL — `KeyError: 'sigma_max_all_eV_per_A'` and `TypeError: write_peratom_sigma() got an unexpected keyword argument 'sigma_free'`.

- [ ] **Step 3: Write the implementation**

In `CommitteeTraceWriter.__init__`, change the tail of `self._fieldnames` to:

```python
            + ["fmax_eV_per_A", "sigma_max_eV_per_A", "sigma_mean_eV_per_A",
               "worst_atom", "sigma_max_all_eV_per_A", "n_free_atoms",
               "mixed_theory"]
```

In `write_step`, add to the `row` dict after `"worst_atom"`:

```python
            "sigma_max_all_eV_per_A": float(latest["sigma_max_all"]),
            "n_free_atoms": int(latest["n_free_atoms"]),
```

Replace `write_peratom_sigma` with:

```python
def write_peratom_sigma(path, symbols, sigma_per_atom, sigma_free=None,
                        free_mask=None) -> None:
    """Write the final geometry's per-atom force disagreement.

    Final geometry only: a per-atom field at every step would be a large file
    for little gain, and ``worst_atom`` already traces where the disagreement
    lives during the run. This file is the most diagnostically useful output
    -- it says *which* atoms the models disagree about, which is usually the
    adsorbate or the reacting bond.

    Constrained atoms keep their row. ``sigma_eV_per_A`` is the unmasked
    value and ``sigma_free_eV_per_A`` drops the held components, so a reader
    can see for themselves how much of the disagreement sits in a region that
    cannot move. ``free_components`` is 0 for a fully fixed atom.
    """
    symbols = list(symbols)
    sigma_per_atom = np.asarray(sigma_per_atom, dtype=float).reshape(-1)
    if len(symbols) != sigma_per_atom.size:
        raise ValueError(
            f"{len(symbols)} symbols but {sigma_per_atom.size} sigma values")
    if sigma_free is None:
        sigma_free = sigma_per_atom
    sigma_free = np.asarray(sigma_free, dtype=float).reshape(-1)
    if sigma_free.size != sigma_per_atom.size:
        raise ValueError(
            f"{sigma_per_atom.size} sigma values but {sigma_free.size} "
            f"free-component sigma values")
    if free_mask is None:
        free_mask = np.ones((sigma_per_atom.size, 3), dtype=bool)
    free_mask = np.asarray(free_mask, dtype=bool)
    if free_mask.shape != (sigma_per_atom.size, 3):
        raise ValueError(
            f"free_mask must have shape {(sigma_per_atom.size, 3)}, got "
            f"{free_mask.shape}")
    n_free = free_mask.sum(axis=1)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["atom_index", "symbol", "sigma_eV_per_A",
                         "sigma_free_eV_per_A", "free_components"])
        for index, symbol in enumerate(symbols):
            writer.writerow([index, symbol, float(sigma_per_atom[index]),
                             float(sigma_free[index]), int(n_free[index])])
```

In `src/mliprun/core/optimize.py`, update the call site to pass the arrays the calculator already recorded — never recompute the mask here, or the file could describe a different mask than the statistic used:

```python
        write_peratom_sigma(committee_peratom_csv,
                            atoms.get_chemical_symbols(),
                            committee.latest["sigma_per_atom"],
                            sigma_free=committee.latest["sigma_per_atom_free"],
                            free_mask=committee.latest["free_mask"])
```

No new import is needed in `optimize.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_outputs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/committee/calculator.py src/mliprun/core/optimize.py tests/test_committee_outputs.py
git commit -m "feat(committee): CSVs carry the masked and unmasked sigma"
```

---

### Task 4: `uncertainty_summary` stops asserting a verdict it was not given

**Files:**
- Modify: `src/mliprun/core/committee/calculator.py` (`uncertainty_summary`, currently around line 421)
- Test: `tests/test_committee_outputs.py`

**Interfaces:**
- Consumes: `latest` keys from Tasks 1–2.
- Produces: `uncertainty_summary(rows, latest, *, threshold=None, threshold_source="none", symbols=None) -> dict`. `threshold_eV_per_A` may be `None`; `threshold_source` is `"explicit"` or `"none"`; `flagged` is `True`, `False`, or `None`. New keys: `sigma_max_all_final_eV_per_A`, `n_free_atoms`, `unhandled_constraints`, `sigma_max_over_fmax_final`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_committee_outputs.py`, and extend the existing `TestFlaggingRule._rows` helper to include `"fmax_eV_per_A": 0.04` in each row:

```python
class TestNoThresholdAssertsNothing:
    def _rows(self, sigmas):
        return [{"step": i, "sigma_max_eV_per_A": s,
                 "energy_spread_aligned_eV": 0.0, "fmax_eV_per_A": 0.04}
                for i, s in enumerate(sigmas)]

    def test_without_a_threshold_flagged_is_none_not_false(self):
        """`false` must keep meaning "checked and passed". Reusing it for
        "not checked" is the failure PR #46 fixed for device_resolved."""
        summary = uncertainty_summary(
            self._rows([0.30, 0.12]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.02, 0.08]))
        assert summary["flagged"] is None
        assert summary["threshold_eV_per_A"] is None
        assert summary["threshold_source"] == "none"

    def test_the_numbers_are_reported_with_no_threshold(self):
        summary = uncertainty_summary(
            self._rows([0.30, 0.12]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.02, 0.08]))
        assert summary["sigma_max_final_eV_per_A"] == pytest.approx(0.08)
        assert summary["sigma_mean_final_eV_per_A"] is not None

    def test_the_ratio_to_fmax_is_reported(self):
        """The dimensionless number that replaces the pass/fail verdict."""
        summary = uncertainty_summary(
            self._rows([0.30, 0.12]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.02, 0.08]))
        assert summary["sigma_max_over_fmax_final"] == pytest.approx(
            0.08 / 0.04, abs=1e-9)

    def test_an_explicit_threshold_still_flags(self):
        summary = uncertainty_summary(
            self._rows([0.30, 0.12]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.02, 0.08]),
            threshold=0.05, threshold_source="explicit")
        assert summary["flagged"] is True
        assert summary["threshold_eV_per_A"] == pytest.approx(0.05)

    def test_an_empty_trace_reports_nothing_rather_than_crashing(self):
        summary = uncertainty_summary([], None)
        assert summary["flagged"] is None
        assert summary["sigma_max_final_eV_per_A"] is None
        assert summary["sigma_max_over_fmax_final"] is None

    def test_the_unmasked_maximum_and_free_count_reach_the_record(self):
        latest = _latest({"member_a": -1.0, "member_b": -3.0}, [0.02, 0.08])
        latest["sigma_max_all"] = 0.9
        latest["n_free_atoms"] = 1
        latest["unhandled_constraints"] = ["FixBondLength"]
        summary = uncertainty_summary(self._rows([0.30]), latest)
        assert summary["sigma_max_all_final_eV_per_A"] == pytest.approx(0.9)
        assert summary["n_free_atoms"] == 1
        assert summary["unhandled_constraints"] == ["FixBondLength"]
```

Extend the file's existing `_latest` helper (line 14) so the new keys are present by default and the pre-existing tests keep working — add these entries to its returned dict:

```python
        "sigma_per_atom_free": sigma,
        "sigma_max_all": float(sigma[worst]),
        "sigma_mean_all": float(sigma.mean()),
        "worst_atom_all": worst,
        "n_free_atoms": int(sigma.size),
        "unhandled_constraints": [],
```

Also add `"fmax_eV_per_A": 0.04` to each row built by `TestFlaggingRule._rows` (line 136), so the ratio is computable there too.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_outputs.py -v -k NoThreshold`
Expected: FAIL — `TypeError: uncertainty_summary() missing 2 required keyword-only arguments`.

- [ ] **Step 3: Write the implementation**

Replace the signature and body of `uncertainty_summary`:

```python
def uncertainty_summary(rows, latest, *, threshold=None,
                        threshold_source="none", symbols=None) -> dict:
```

Rewrite the docstring's "flagging rule" paragraph to:

```
    With no threshold -- the default -- nothing is asserted: the numbers are
    reported and ``flagged`` is ``None``. ``False`` keeps its meaning of
    "checked against a threshold and passed"; using it for "not checked"
    would put a claim in the record that nobody made.

    When ``--uncertainty-threshold`` is given, a configuration is flagged
    when ``sigma_max`` at the **final** geometry exceeds it. A path that
    passed through a strained geometry but converged to a well-constrained
    minimum is not flagged, which is why the peak is reported separately.

    There is no default threshold. Same-level committees measured on
    cos-cluster disagree by 0.11-0.15 eV/A against convergence targets of
    0.02-0.05, so the previous default (``fmax``) fired on ordinary healthy
    relaxations; and those numbers were measured before constrained atoms
    were excluded from sigma, so they are not a calibration either. See the
    2026-09-08 design note.
```

Replace the summary dict and the tail:

```python
    summary = {
        "n_steps": len(rows),
        "threshold_eV_per_A": None if threshold is None else float(threshold),
        "threshold_source": threshold_source,
        "sigma_max_final_eV_per_A": None,
        "sigma_mean_final_eV_per_A": None,
        "sigma_max_all_final_eV_per_A": None,
        "sigma_max_peak_eV_per_A": None,
        "sigma_max_over_fmax_final": None,
        "peak_step": None,
        "worst_atom": None,
        "worst_atom_symbol": None,
        "n_free_atoms": None,
        "unhandled_constraints": [],
        "energy_spread_aligned_final_eV": None,
        "flagged": None,
    }
    if rows:
        peak = max(rows, key=lambda r: r["sigma_max_eV_per_A"])
        summary["sigma_max_peak_eV_per_A"] = float(peak["sigma_max_eV_per_A"])
        summary["peak_step"] = int(peak["step"])
        summary["energy_spread_aligned_final_eV"] = float(
            rows[-1]["energy_spread_aligned_eV"])
    if latest is not None:
        sigma_max = float(latest["sigma_max"])
        worst = int(latest["worst_atom"])
        summary["sigma_max_final_eV_per_A"] = sigma_max
        summary["sigma_mean_final_eV_per_A"] = float(latest["sigma_mean"])
        summary["worst_atom"] = worst
        if "sigma_max_all" in latest:
            summary["sigma_max_all_final_eV_per_A"] = float(
                latest["sigma_max_all"])
        if "n_free_atoms" in latest:
            summary["n_free_atoms"] = int(latest["n_free_atoms"])
        summary["unhandled_constraints"] = list(
            latest.get("unhandled_constraints", []))
        if symbols is not None and worst < len(symbols):
            summary["worst_atom_symbol"] = symbols[worst]
        final_fmax = rows[-1].get("fmax_eV_per_A") if rows else None
        if final_fmax:
            summary["sigma_max_over_fmax_final"] = sigma_max / float(final_fmax)
        if threshold is not None:
            summary["flagged"] = bool(sigma_max > float(threshold))
    return summary
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_outputs.py -v`
Expected: PASS. The pre-existing `TestFlaggingRule` tests pass an explicit `threshold=` and `threshold_source="fmax"`; change their `threshold_source` to `"explicit"` in the same commit, since `"fmax"` is no longer a source this code can produce.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/committee/calculator.py tests/test_committee_outputs.py
git commit -m "feat(committee)!: no default uncertainty threshold; flagged is tri-state"
```

---

### Task 5: `run_optimization` resolves no threshold, and the schema bumps

**Files:**
- Modify: `src/mliprun/core/optimize.py:292-294` (threshold resolution), `:45-53` (`_committee_parameters`), `:437-455` (the logging branch)
- Modify: `src/mliprun/core/run_record.py:33` (`SCHEMA_VERSION`)
- Test: `tests/test_committee_optimize.py`, plus the five sites that hardcode schema 4

**Interfaces:**
- Consumes: `uncertainty_summary(..., threshold=None, threshold_source="none")` from Task 4.
- Produces: `results.committee_uncertainty` as specified in Task 4; `parameters.uncertainty_threshold` is `None` unless the caller passed one.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_committee_optimize.py`, using the file's existing `_committee(tmp_path)`, `_rattled()` and `_record(tmp_path)` helpers (lines 26–72) and the same `run_optimization(...)` call shape as `TestRunRecord`:

```python
class TestThresholdIsOptIn:
    def _run(self, tmp_path, **kwargs):
        atoms = _rattled()
        committee = _committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=10,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee, **kwargs)
        finally:
            committee.close()
        return _record(tmp_path)

    def test_no_threshold_records_none_and_does_not_flag(self, tmp_path):
        uncertainty = self._run(tmp_path)["stages"][0]["results"][
            "committee_uncertainty"]
        assert uncertainty["threshold_source"] == "none"
        assert uncertainty["threshold_eV_per_A"] is None
        assert uncertainty["flagged"] is None

    def test_the_threshold_does_not_silently_become_fmax(self, tmp_path):
        """The old default. Removing it is the point of this change."""
        uncertainty = self._run(tmp_path)["stages"][0]["results"][
            "committee_uncertainty"]
        assert uncertainty["threshold_eV_per_A"] != pytest.approx(0.05)

    def test_an_explicit_threshold_is_recorded_and_applied(self, tmp_path):
        """Two identical EMT members give sigma exactly 0, so a threshold of
        -1 is the only way to make the flag fire from this harness."""
        uncertainty = self._run(tmp_path, uncertainty_threshold=-1.0)[
            "stages"][0]["results"]["committee_uncertainty"]
        assert uncertainty["threshold_source"] == "explicit"
        assert uncertainty["flagged"] is True

    def test_the_parameter_block_records_no_threshold(self, tmp_path):
        params = self._run(tmp_path)["stages"][0]["parameters"]
        assert params["uncertainty_threshold"] is None

    def test_the_free_atom_count_reaches_the_record(self, tmp_path):
        """_rattled() has no constraints, so every atom is free."""
        uncertainty = self._run(tmp_path)["stages"][0]["results"][
            "committee_uncertainty"]
        assert uncertainty["n_free_atoms"] == len(_rattled())
```

Then update the two pre-existing assertions this change invalidates, in `TestRunRecord.test_the_record_carries_members_and_uncertainty` (line 165): `threshold_source` becomes `"none"`, `threshold_eV_per_A` becomes `None` (drop the `pytest.approx(0.05)` assertion), and `flagged` becomes `None`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_optimize.py -v -k ThresholdIsOptIn`
Expected: FAIL — `assert 'fmax' == 'none'`.

- [ ] **Step 3: Write the implementation**

In `src/mliprun/core/optimize.py`, replace the threshold block:

```python
    # No default. A same-level committee disagrees by 0.11-0.15 eV/A against
    # convergence targets of 0.02-0.05, so defaulting to fmax flagged
    # ordinary healthy relaxations; and those numbers predate the exclusion
    # of constrained atoms from sigma, so they are not a calibration either.
    # The run reports sigma and its ratio to fmax; a verdict is asserted only
    # when a caller chooses a threshold. See the 2026-09-08 design note.
    threshold = (float(uncertainty_threshold)
                 if uncertainty_threshold is not None else None)
    threshold_source = ("explicit" if uncertainty_threshold is not None
                        else "none")
```

Both `uncertainty_summary(...)` call sites already pass `threshold=threshold, threshold_source=threshold_source`, so they need no change.

Replace the `if summary["flagged"]:` logging branch with:

```python
        logger.info(
            "Committee disagreement at the final geometry: sigma_max = "
            "%.4f eV/Ang over %s free atoms (worst: %s #%s), sigma_mean = "
            "%.4f eV/Ang.",
            summary["sigma_max_final_eV_per_A"], summary["n_free_atoms"],
            summary["worst_atom_symbol"], summary["worst_atom"],
            summary["sigma_mean_final_eV_per_A"])
        if summary["flagged"]:
            logger.info(
                "sigma_max exceeds the chosen threshold %.4f eV/Ang; this "
                "configuration deserves a DFT check.", threshold)
```

Update `run_optimization`'s `uncertainty_threshold` docstring paragraph (`optimize.py:254-256`) to say there is no default.

Set `SCHEMA_VERSION = 5` in `src/mliprun/core/run_record.py:33`, and update the five sites that assert 4: `tests/test_committee_optimize.py:182`, `tests/test_committee_cli.py:117`, `tests/test_run_record.py:391` (rename `test_schema_version_is_four` to `test_schema_version_is_five`), `tests/test_run_record.py:399`, `tests/test_core_md.py:151`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/ -q`
Expected: PASS, whole suite, no MLIP installed. Confirm `git status` shows no change under `tests/goldens/`.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/optimize.py src/mliprun/core/run_record.py tests/
git commit -m "feat!: uncertainty threshold is opt-in; run record schema 5"
```

---

### Task 6: The CLI always reports, and warns only when asked to

**Files:**
- Modify: `src/mliprun/cli/commands/optimize.py:202-229` (`_report_flagged_uncertainty`), `:254-261` (the option's help)
- Test: `tests/test_committee_cli.py`

**Interfaces:**
- Consumes: `committee_calc.latest_uncertainty_summary` as produced by Task 5.
- Produces: `_report_committee_uncertainty(committee_calc)` replaces `_report_flagged_uncertainty`; the call site around `optimize.py:340` is renamed with it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_committee_cli.py`, using the file's existing `structure` and `fake_committee_file` fixtures and its module-level `runner` (the same shape as `TestEndToEnd`, line 98):

```python
class TestCommitteeUncertaintyEcho:
    def _invoke(self, structure, path, *extra):
        return runner.invoke(app, ["run", "--structure", str(structure),
                                   "--committee", str(path),
                                   "--max-steps", "3", "--no-verbose",
                                   *extra])

    def test_the_numbers_are_printed_without_a_threshold(
            self, structure, fake_committee_file):
        path, _ = fake_committee_file
        result = self._invoke(structure, path)
        assert result.exit_code == 0, result.output
        assert "Committee disagreement at the final geometry" in result.output
        assert "free atoms" in result.output

    def test_no_warning_is_printed_without_a_threshold(
            self, structure, fake_committee_file):
        """A verdict nobody asked for is what this change removes."""
        path, _ = fake_committee_file
        result = self._invoke(structure, path)
        assert "deserves a DFT check" not in result.output

    def test_a_tripped_explicit_threshold_warns(
            self, structure, fake_committee_file):
        """Two identical EMT members give sigma exactly 0, so -1 is the only
        threshold this harness can exceed."""
        path, _ = fake_committee_file
        result = self._invoke(structure, path,
                              "--uncertainty-threshold", "-1")
        assert result.exit_code == 0, result.output
        assert "deserves a DFT check" in result.output

    def test_an_untripped_explicit_threshold_does_not_warn(
            self, structure, fake_committee_file):
        path, _ = fake_committee_file
        result = self._invoke(structure, path,
                              "--uncertainty-threshold", "1e9")
        assert "deserves a DFT check" not in result.output
        assert "Committee disagreement at the final geometry" in result.output
```

Then update the three pre-existing assertions this change invalidates:
- `TestEndToEnd.test_a_two_member_run_writes_every_output` (line 123): `flagged is False` becomes `flagged is None`.
- `TestFlaggedPath.test_a_genuinely_disagreeing_committee_trips_the_threshold` (line 199): the string counted changes from `"High committee disagreement"` to `"deserves a DFT check"`; keep the `== 1` count and the comment explaining why exactly one channel prints it.
- The `--uncertainty-threshold 0.01` in that same test stays as it is: it is already explicit, which is now the only way to flag.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_cli.py -v -k UncertaintyEcho`
Expected: FAIL — the summary line is absent from stdout.

- [ ] **Step 3: Write the implementation**

Replace `_report_flagged_uncertainty` with:

```python
def _report_committee_uncertainty(committee_calc) -> None:
    """Echo the committee's disagreement at the final geometry.

    Always printed: with no threshold there is no verdict to give, and the
    numbers are the deliverable. The warning below it appears only when a
    caller chose a threshold and the run exceeded it.

    This is the single terminal report at default settings:
    ``run_optimization``'s own log calls are INFO-level (silent unless a
    caller configures logging below WARNING), precisely so this echo is not
    a duplicate of them.

    Reads ``committee_calc.latest_uncertainty_summary`` -- the exact
    ``uncertainty_summary(...)`` dict ``run_optimization`` already computed
    and stored in the run record -- rather than recomputing it, so the
    printed number can never diverge from the recorded one.
    """
    if committee_calc is None:
        return
    summary = committee_calc.latest_uncertainty_summary
    if summary is None or summary["sigma_max_final_eV_per_A"] is None:
        return
    ratio = summary["sigma_max_over_fmax_final"]
    ratio_text = "" if ratio is None else f", {ratio:.1f}x the final fmax"
    typer.echo(
        f"\n📊 Committee disagreement at the final geometry: "
        f"sigma_max = {summary['sigma_max_final_eV_per_A']:.4f} eV/Å"
        f"{ratio_text}, sigma_mean = "
        f"{summary['sigma_mean_final_eV_per_A']:.4f} eV/Å over "
        f"{summary['n_free_atoms']} free atoms. Worst atom: "
        f"{summary['worst_atom_symbol']} (#{summary['worst_atom']}).")
    if summary["unhandled_constraints"]:
        typer.echo(
            f"   Note: constraint type(s) "
            f"{', '.join(summary['unhandled_constraints'])} are not masked, "
            f"so sigma is over-reported for their atoms.")
    if summary["flagged"]:
        typer.echo(
            f"\n⚠️  sigma_max exceeds the threshold you set "
            f"({summary['threshold_eV_per_A']:.4f} eV/Å). The located "
            f"minimum sits inside the committee's own noise; this "
            f"configuration deserves a DFT check.")
```

Rename the call site (currently `_report_flagged_uncertainty(committee_calc)`) to match.

Replace the option's help text:

```python
    uncertainty_threshold: float = typer.Option(
        None, "--uncertainty-threshold",
        help="Flag the final configuration when the committee's per-atom "
             "force disagreement exceeds this (eV/Å). NO DEFAULT: without "
             "it the run reports sigma and its ratio to fmax but asserts no "
             "verdict. There is no calibrated value yet -- same-level "
             "committees disagree by ~0.1 eV/Å. See docs/OUTPUTS.md."),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_committee_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/cli/commands/optimize.py tests/test_committee_cli.py
git commit -m "feat(cli): always report committee disagreement, warn only on an explicit threshold"
```

---

### Task 7: Documentation and changelog

**Files:**
- Modify: `docs/OUTPUTS.md:54-130`, `:159-200`
- Modify: `docs/PYTHON_API.md` (the committee section)
- Modify: `CHANGELOG.md` (`[Unreleased]`)

**Interfaces:**
- Consumes: the final column names and record fields from Tasks 3–5.
- Produces: nothing code depends on.

- [ ] **Step 1: Update `docs/OUTPUTS.md`**

In the trace-CSV column table, change the two sigma rows and add the new ones:

```markdown
| `sigma_max_eV_per_A` | Largest per-atom force disagreement across members at this step, over **free force components only** — the same atoms ASE's `fmax` uses |
| `sigma_mean_eV_per_A` | Mean per-atom force disagreement over the atoms with at least one free component |
| `sigma_max_all_eV_per_A` | The same maximum with no constraint masking, kept so runs from before 2026-09-08 stay comparable |
| `n_free_atoms` | How many atoms retain at least one free force component |
```

In the per-atom-file section, replace the column list with `atom_index`, `symbol`, `sigma_eV_per_A` (unmasked), `sigma_free_eV_per_A` (held components dropped), `free_components` (0–3; 0 means fully fixed), and state that constrained atoms keep their row on purpose.

Replace the "flagged" paragraph and the "default threshold is uncalibrated" paragraph with a single section saying: there is no default threshold; the run always reports `sigma_max`, `sigma_mean`, the worst atom and `sigma_max_over_fmax_final`; `--uncertainty-threshold` is opt-in; `flagged` is `true`, `false` or `null`, and `null` means no threshold was applied — it is not the same as `false`.

In the run-record field list (`:197-199`), replace the `committee_uncertainty` field list with the Task 4 key set, and note `threshold_source` is now `"explicit"` or `"none"` (`"fmax"` no longer occurs).

Add a short note that only `FixAtoms` and `FixCartesian` are masked, and that any other constraint type leaves its atoms counted as free, with the type names recorded in `unhandled_constraints`.

- [ ] **Step 2: Update `docs/PYTHON_API.md`**

Update the `committee_statistics`, `write_peratom_sigma` and `uncertainty_summary` signatures, and document `free_component_mask`. State that `run_optimization(uncertainty_threshold=None)` now means "no threshold" rather than "use fmax".

- [ ] **Step 3: Write the CHANGELOG entry**

Under `[Unreleased]`, with the three breaking changes called out:

```markdown
### Changed
- **Breaking:** the committee's `sigma_max` and `sigma_mean` now exclude
  constrained force components, so they are taken over the same atoms as
  ASE's `fmax`. Previously a frozen slab atom could carry the reported
  maximum. The unmasked values are kept as `sigma_max_all` / `sigma_mean_all`.
  Only `FixAtoms` and `FixCartesian` are masked; other constraint types leave
  their atoms counted as free and are named in `unhandled_constraints`.
- **Breaking:** `--uncertainty-threshold` no longer defaults to `--fmax`.
  With no threshold the run reports the disagreement and its ratio to fmax
  but asserts no verdict, and `flagged` is `null` rather than `false`. The
  old default fired on ordinary healthy relaxations.
- **Breaking:** run record schema 4 → 5. `results.committee_uncertainty`
  gains `sigma_max_all_final_eV_per_A`, `n_free_atoms`,
  `unhandled_constraints` and `sigma_max_over_fmax_final`;
  `threshold_source` is now `"explicit"` or `"none"`.
- `<stem>_committee.csv` gains `sigma_max_all_eV_per_A` and `n_free_atoms`;
  `<stem>_committee_peratom.csv` gains `sigma_free_eV_per_A` and
  `free_components`.
```

- [ ] **Step 4: Verify the whole suite and the coverage gate**

```bash
pytest tests/ -q
COVERAGE_PROCESS_START=$PWD/pyproject.toml pytest --cov=mliprun --cov-report=xml
diff-cover coverage.xml --compare-branch=origin/main --fail-under=90 --exclude setup.py --exclude "scripts/*"
git status --short tests/goldens/
```
Expected: all tests pass with no MLIP installed; diff coverage ≥ 90 %; `tests/goldens/` clean.

- [ ] **Step 5: Commit and open the draft PR**

```bash
git add docs/ CHANGELOG.md
git commit -m "docs(committee): sigma masking and the opt-in threshold"
gh pr create --draft --title "feat(committee)!: sigma excludes constrained atoms; threshold is opt-in" --body-file <path>
```

If `gh pr create`/`gh pr edit` fails with a "Projects (classic) deprecated" GraphQL error, patch the body through `gh api repos/:owner/:repo/pulls/<n> -X PATCH -F body=@<path>` and grep the live body to confirm it applied.

---

### Task 8: MPtrj models share one level of theory

**Files:**
- Modify: `src/mliprun/core/committee/config.py:59` (`LEVEL_TABLE_VERSION`), `:85-89` (`_FIXED_LEVELS`)
- Test: `tests/test_committee_theory.py`, `tests/test_committee_config.py`

**Interfaces:**
- Consumes: nothing from earlier tasks — this is independent of Tasks 1–7 and can be reviewed or dropped on its own.
- Produces: `resolve_level_of_theory("mace") == resolve_level_of_theory("chgnet") == "PBE(+U)/MPtrj"`; `LEVEL_TABLE_VERSION == 2`.

Juan's ruling, 2026-09-08. Rationale in the design note, Decision 3. Only `mace` and `chgnet` change — the three `7net` MPtrj tags stay unlisted until their training data is confirmed from the checkpoints with `sevenn cp <tag>`, per the repo's read-the-package rule.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_committee_theory.py`:

```python
class TestMPtrjModelsShareOneLevel:
    def test_mace_mp_0_and_chgnet_are_the_same_level(self):
        """Both are trained on MPtrj. Their disagreement is architectural at
        a fixed level of theory, which is exactly what a committee measures.
        Juan's ruling, 2026-09-08."""
        assert (resolve_level_of_theory("mace")
                == resolve_level_of_theory("chgnet"))

    def test_the_label_names_the_mixing_not_one_functional(self):
        """MPtrj applies +U to transition-metal oxides and fluorides only, so
        the effective functional depends on the system. A label asserting
        plain PBE or plain PBE+U is wrong for half the compositions."""
        assert resolve_level_of_theory("mace") == "PBE(+U)/MPtrj"

    def test_an_mptrj_committee_is_not_mixed_theory(self):
        assert is_mixed_theory([resolve_level_of_theory("mace"),
                                resolve_level_of_theory("chgnet")]) is False

    def test_omat24_is_still_a_different_level(self):
        """The ruling merges MPtrj labels only; it must not collapse
        genuinely different datasets."""
        assert is_mixed_theory([resolve_level_of_theory("7net-omat"),
                                resolve_level_of_theory("mace")]) is True

    def test_the_unverified_mptrj_sevennet_tags_stay_unknown(self):
        """7net-0 and 7net-l3i5 are believed to be MPtrj too, but that is not
        confirmed from the checkpoints, so they must not be quietly merged."""
        assert resolve_level_of_theory("7net-0") == UNKNOWN_LEVEL
        assert resolve_level_of_theory("7net-l3i5") == UNKNOWN_LEVEL
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_committee_theory.py -v -k MPtrj`
Expected: FAIL — `assert 'PBE/MPtrj' == 'PBE+U/MPtrj'`.

- [ ] **Step 3: Write the implementation**

In `src/mliprun/core/committee/config.py`, set `LEVEL_TABLE_VERSION = 2` and change `_FIXED_LEVELS`:

```python
#: Single-task tags whose level is fixed by the tag alone.
#:
#: ``mace`` (MACE-MP-0 medium) and ``chgnet`` share one label because they
#: share one training set: MPtrj, the Materials Project relaxation
#: trajectories. The label names the dataset and signals the mixing rather
#: than asserting a functional, because MPtrj applies Hubbard U to specific
#: transition metals in oxides and fluorides only -- so an MPtrj-trained
#: model is effectively plain PBE for a metallic slab and PBE+U for an oxide,
#: and no static label can settle that. Juan's ruling, 2026-09-08.
_FIXED_LEVELS = {
    "mace": "PBE(+U)/MPtrj",
    "chgnet": "PBE(+U)/MPtrj",
    "7net-omat": "PBE/OMat24",
}
```

- [ ] **Step 4: Update the five pre-existing assertions this inverts, then run**

- `tests/test_committee_theory.py:30-31`: both rows now expect `"PBE(+U)/MPtrj"`.
- `tests/test_committee_theory.py:82` (`test_two_levels_are_mixed`): `"PBE/MPtrj"` becomes `"PBE(+U)/MPtrj"`.
- `tests/test_committee_config.py:136-138` (`test_provenance_block_carries_every_member`): a chgnet + mace committee is now **same-level**. `level_of_theory` becomes `"PBE(+U)/MPtrj"`, `mixed_theory` becomes `False`, `levels` becomes `["PBE(+U)/MPtrj"]`, and the trailing comment must change from `# PBE+U/MPtrj vs PBE/MPtrj` to say they share one dataset. Do not delete this test to make it pass.
- `tests/test_committee_config.py:152` (`test_two_levels_set_the_flag`, uma + mace — genuinely mixed, keep it that way): `["PBE/MPtrj", "RPBE/OC20"]` becomes `["PBE(+U)/MPtrj", "RPBE/OC20"]`.
- The `LEVEL_TABLE_VERSION` assertions at `tests/test_committee_config.py:531-533` are version-agnostic (`>= 1`) and need no change.

Run: `pytest tests/test_committee_theory.py tests/test_committee_config.py -v`
Expected: PASS. Confirm `test_two_levels_set_the_flag` still asserts `mixed_theory is True` — this ruling must not make every committee look same-level.

- [ ] **Step 5: Commit**

Add to the `[Unreleased]` CHANGELOG section written in Task 7:

```markdown
- **Breaking:** `mace` (MACE-MP-0) and `chgnet` now resolve to one level of
  theory, `PBE(+U)/MPtrj`, because they share one training set. A committee
  of the two is no longer reported as mixed theory, and its spread is an
  error bar rather than a functional comparison. `LEVEL_TABLE_VERSION` 1 → 2.
  The `7net` MPtrj tags are unchanged pending checkpoint confirmation.
```

```bash
git add src/mliprun/core/committee/config.py tests/test_committee_theory.py tests/test_committee_config.py CHANGELOG.md
git commit -m "feat(committee)!: MACE-MP-0 and CHGNet share the PBE(+U)/MPtrj level"
```

---

## Verification not covered by unit tests

These need real MLIPs and belong in a cluster session after the PR is green, not in this plan's tasks:

1. Re-measure `sigma_max` on `run1_samelevel` (O/Pt(111), 3 members, 8 of 17 atoms fixed) and `run7_6member` (CH/FeNi, 6 members, 32 of 82 fixed) and compare against the pre-change 0.1502 and 0.1121. The gap between `sigma_max` and `sigma_max_all` is the answer to whether the old numbers were dominated by frozen substrate.
2. Confirm the terminal prints the summary line and no warning on a default run.
3. Confirm from the checkpoints whether `7net-0`, `7net-0_22may2024` and `7net-l3i5` are MPtrj (`sevenn cp <tag>` in the pysevenn env). If they are, add their rows as `PBE(+U)/MPtrj` and bump `LEVEL_TABLE_VERSION` to 3. Until then they stay `unknown` and a `mace` + `7net-0` committee reports as mixed.

**Blocked until the cluster clone is clean.** `/scratchb/juar/committee-test/mliprun` is on `321013b` with two files hand-copied over it; `git fetch && git checkout main` first, or nothing run there is on a known commit. See `.claude/handoffs/committee-followups.md` item 4.
