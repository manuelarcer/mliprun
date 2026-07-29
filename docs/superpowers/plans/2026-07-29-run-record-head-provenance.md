# Run Record Head/Task Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `mliprun_run.json` name the MLIP head/task that actually ran, so the record can identify the level of theory on its own.

**Architecture:** Two nullable fields (`uma_task`, `mace_head`) added to the `provenance` block of the run record, populated through `collect_provenance` and gated there by model-tag family so a caller passing an inapplicable default cannot mislabel a run. `SCHEMA_VERSION` goes 1 → 2 so readers can distinguish an old record from a genuinely-null field. Both fields join `_PROVENANCE_DIFF_FIELDS` so a resume that switches head is recorded.

**Tech Stack:** Python 3.9+, ASE, typer CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-07-29-run-record-head-provenance-design.md`

## Global Constraints

- **Branch:** `feat/run-record-head-provenance` (already created, spec already committed on it). Draft PRs only; never push to `main`. One concern per PR.
- **Never regenerate golden reference files** (`tests/goldens/*.json`). If a golden test fails, report the numerical delta; updating baselines is a human decision. This change touches no numerics, so no golden should move.
- **CI gates:** full unit suite plus diff coverage ≥ 90% on changed lines. Every new line needs a test that reaches it.
- **Every test must assert a numerical value or invariant.** No smoke-only tests.
- Test command (no MLIP required): `pytest -m "not uma and not mace and not sevenn"`
- Code style: typer CLI, lazy imports for heavy packages, NumPy-style docstrings.
- `uma_task` is non-null only for models whose tag starts with `uma-`; `mace_head` only for tags starting with `mace-mh-`. Enforced inside `collect_provenance`, not at call sites.

---

### Task 1: Head fields, schema bump, diff fields

**Files:**
- Modify: `src/mliprun/core/run_record.py:32` (`SCHEMA_VERSION`), `:52` (`_PROVENANCE_DIFF_FIELDS`), `:159-204` (`collect_provenance`)
- Test: `tests/test_run_record.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `collect_provenance(*, mlip_model, device_requested, device_resolved, uma_task=None, mace_head=None) -> dict` — returned dict gains keys `"uma_task"` and `"mace_head"`, both always present, `None` when the model family does not match. `SCHEMA_VERSION == 2`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_run_record.py`. The existing module-level helpers `_read(tmp_path)` and `_begin(tmp_path, **kwargs)` are already defined at the top of that file — reuse them.

```python
class TestHeadProvenance:
    """CANON C1/C3: the head/task is its own decision and must never be
    inferred. A record naming only the model tag cannot identify the level
    of theory."""

    def test_head_fields_default_to_none(self):
        prov = collect_provenance(mlip_model="chgnet", device_requested="cpu",
                                  device_resolved="cpu")
        assert prov["uma_task"] is None
        assert prov["mace_head"] is None

    def test_records_uma_task_for_uma_model(self):
        prov = collect_provenance(mlip_model="uma-s-1p2", device_requested="cuda",
                                  device_resolved="cuda", uma_task="oc25")
        assert prov["uma_task"] == "oc25"
        assert prov["mace_head"] is None

    def test_records_mace_head_for_mace_model(self):
        prov = collect_provenance(mlip_model="mace-mh-1", device_requested="cpu",
                                  device_resolved="cpu", mace_head="omat_pbe")
        assert prov["mace_head"] == "omat_pbe"
        assert prov["uma_task"] is None

    def test_drops_head_that_does_not_match_the_model_family(self):
        """A CLI passes its parsed --uma-task unconditionally, defaults
        included. A MACE run must not inherit a UMA task it never used."""
        prov = collect_provenance(mlip_model="mace-mh-1", device_requested="cpu",
                                  device_resolved="cpu", uma_task="omat",
                                  mace_head="omat_pbe")
        assert prov["uma_task"] is None
        assert prov["mace_head"] == "omat_pbe"

    def test_tolerates_non_string_model(self):
        prov = collect_provenance(mlip_model=None, device_requested="cpu",
                                  device_resolved="cpu", uma_task="omat")
        assert prov["uma_task"] is None
        assert prov["mace_head"] is None


class TestSchemaVersion:
    def test_schema_version_is_two(self):
        """Bumped so a reader can tell 'no uma_task key because the record
        predates the field' from 'uma_task is null because it was MACE'."""
        assert SCHEMA_VERSION == 2

    def test_written_record_carries_the_new_version(self, tmp_path):
        _begin(tmp_path)
        assert _read(tmp_path)["schema_version"] == 2


class TestHeadInStageProvenance:
    def test_append_records_a_switched_head(self, tmp_path):
        """C3 forbids mixing heads within an energy formula, so a resume
        that switches head is exactly what must not go unrecorded."""
        base = {"mliprun_version": "0.4.0", "hostname": "node-a",
                "device_resolved": "cuda", "mlip_model": "uma-s-1p2",
                "uma_task": "omat", "mace_head": None}
        rec = _begin(tmp_path, provenance=dict(base))
        rec.complete(status="converged")

        rec2 = _begin(tmp_path, provenance=dict(base, uma_task="oc25"),
                      append=True)
        rec2.complete(status="converged")

        data = _read(tmp_path)
        assert data["stages"][1]["stage_provenance"] == {"uma_task": "oc25"}
        assert data["provenance"]["uma_task"] == "omat"

    def test_append_with_same_head_writes_no_stage_provenance(self, tmp_path):
        base = {"mliprun_version": "0.4.0", "hostname": "node-a",
                "device_resolved": "cuda", "mlip_model": "uma-s-1p2",
                "uma_task": "omat", "mace_head": None}
        rec = _begin(tmp_path, provenance=dict(base))
        rec.complete(status="converged")
        rec2 = _begin(tmp_path, provenance=dict(base), append=True)
        rec2.complete(status="converged")
        assert "stage_provenance" not in _read(tmp_path)["stages"][1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_run_record.py -k "HeadProvenance or SchemaVersion or HeadInStage" -v`
Expected: FAIL — `TypeError: collect_provenance() got an unexpected keyword argument 'uma_task'`, and `assert 1 == 2` for the schema version.

- [ ] **Step 3: Bump the schema version**

In `src/mliprun/core/run_record.py`, change line 32:

```python
SCHEMA_VERSION = 2
```

- [ ] **Step 4: Extend the provenance diff fields**

Replace the `_PROVENANCE_DIFF_FIELDS` definition (currently line 52), keeping the existing comment above it:

```python
_PROVENANCE_DIFF_FIELDS = ("mliprun_version", "hostname", "device_resolved",
                           "mlip_model", "uma_task", "mace_head")
```

- [ ] **Step 5: Add the head parameters to `collect_provenance`**

Change the signature (currently lines 159-160) to:

```python
def collect_provenance(*, mlip_model: Any, device_requested: str,
                        device_resolved: str, uma_task: Optional[str] = None,
                        mace_head: Optional[str] = None) -> dict:
```

Append to the existing docstring, before the closing quotes:

```
    ``uma_task`` and ``mace_head`` are gated on the model tag rather than
    trusted from the caller: CLIs pass whatever their ``--uma-task`` /
    ``--mace-head`` options resolved to, defaults included, so a MACE run
    would otherwise be recorded as carrying a UMA task it never used.
    CANON C1 makes the head its own explicit decision and C3 forbids mixing
    heads within an energy formula, so a wrong head is worse than none.
```

Immediately before the `return {` statement (currently line 192), add:

```python
    tag = mlip_model if isinstance(mlip_model, str) else ""
```

Inside the returned dict, directly after the `"mlip_model": mlip_model,` entry, add:

```python
        "uma_task": uma_task if tag.startswith("uma-") else None,
        "mace_head": mace_head if tag.startswith("mace-mh-") else None,
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_run_record.py -v`
Expected: PASS, including every pre-existing test in the file.

- [ ] **Step 7: Run the full unit suite**

Run: `pytest -m "not uma and not mace and not sevenn"`
Expected: PASS. If any golden test fails, STOP and report the numerical delta — do not regenerate.

- [ ] **Step 8: Commit**

```bash
git add src/mliprun/core/run_record.py tests/test_run_record.py
git commit -m "feat(run-record): carry uma_task/mace_head, schema v2

collect_provenance took only the model tag, so a record named uma-s-1p2
without the oc25 task that actually ran. CANON C1 makes the head its own
explicit decision and C3 forbids mixing heads within an energy formula,
so the record could not serve as provenance for any energy entering a
formula.

Both fields are gated on the model tag inside collect_provenance rather
than trusted from callers, which pass their parsed option defaults
unconditionally. SCHEMA_VERSION 1 -> 2 so a reader can distinguish an
old record from a genuinely-null field, and both fields join
_PROVENANCE_DIFF_FIELDS so a resume that switches head is recorded.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Thread the head through the core engines

**Files:**
- Modify: `src/mliprun/core/optimize.py:44-56` (signature), `:152-156` (`collect_provenance` call)
- Modify: `src/mliprun/core/md.py:213-239` (signature), `:323-327` (`collect_provenance` call)
- Modify: `src/mliprun/core/neb.py:662-666` and `:815-819` (both `collect_provenance` calls)
- Test: `tests/test_core_md.py`, `tests/test_core_optimize.py`

**Interfaces:**
- Consumes: `collect_provenance(..., uma_task=None, mace_head=None)` from Task 1.
- Produces: `run_optimization(...)` and `run_md(...)` both accept keyword-only-by-convention `uma_task: Optional[str] = None, mace_head: Optional[str] = None` appended after `device_resolved`. `CustomNEB` needs no new attributes — it already stores `self.uma_task` and `self.mlip`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_core_md.py`:

```python
class TestMDRecordsHead:
    def test_run_md_writes_the_uma_task_into_the_record(self, tmp_path):
        import json
        from ase.build import bulk
        from ase.calculators.emt import EMT
        from mliprun.core.md import run_md

        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms.calc = EMT()
        run_md(atoms, ensemble="nve", steps=2, log_interval=1,
               traj_interval=1, output_dir=tmp_path, model_name="uma-s-1p2",
               uma_task="oc25")

        data = json.loads((tmp_path / "mliprun_run.json").read_text())
        assert data["provenance"]["uma_task"] == "oc25"
        assert data["provenance"]["mace_head"] is None
        assert data["schema_version"] == 2
```

Add to `tests/test_core_optimize.py`:

```python
class TestOptimizeRecordsHead:
    def test_run_optimization_writes_the_mace_head_into_the_record(self, tmp_path):
        import json
        from ase.build import bulk
        from ase.calculators.emt import EMT
        from mliprun.core.optimize import run_optimization

        atoms = bulk("Cu", "fcc", a=3.7) * (2, 2, 2)
        atoms.rattle(stdev=0.05, seed=42)
        atoms.calc = EMT()
        run_optimization(atoms, optimizer="bfgs", fmax=0.05, max_steps=20,
                         output_dir=tmp_path, verbose=False,
                         model_name="mace-mh-1", mace_head="omat_pbe")

        data = json.loads((tmp_path / "mliprun_run.json").read_text())
        assert data["provenance"]["mace_head"] == "omat_pbe"
        assert data["provenance"]["uma_task"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_core_md.py::TestMDRecordsHead tests/test_core_optimize.py::TestOptimizeRecordsHead -v`
Expected: FAIL — `TypeError: run_md() got an unexpected keyword argument 'uma_task'`.

- [ ] **Step 3: Extend `run_optimization`**

In `src/mliprun/core/optimize.py`, add two parameters immediately after `device_resolved: str = "auto",` in the signature:

```python
    uma_task: Optional[str] = None,
    mace_head: Optional[str] = None,
```

Add to the docstring's parameter list, next to the existing `device_resolved` entry:

```
    uma_task : str, optional
        UMA task actually used, recorded for provenance. Ignored for
        non-UMA models.
    mace_head : str, optional
        MACE head actually used, recorded for provenance. Ignored for
        non-MACE models.
```

Change the `collect_provenance` call to:

```python
        provenance=collect_provenance(
            mlip_model=model_name,
            device_requested=device_requested,
            device_resolved=device_resolved,
            uma_task=uma_task,
            mace_head=mace_head,
        ),
```

- [ ] **Step 4: Extend `run_md`**

In `src/mliprun/core/md.py`, add the same two parameters immediately after `device_resolved: str = "auto",` in the `run_md` signature, the same two docstring entries next to the existing `device_resolved` entry, and change the `collect_provenance` call to:

```python
        provenance=collect_provenance(
            mlip_model=model_name,
            device_requested=device_requested,
            device_resolved=device_resolved,
            uma_task=uma_task,
            mace_head=mace_head,
        ),
```

- [ ] **Step 5: Wire both NEB call sites**

In `src/mliprun/core/neb.py`, both `collect_provenance` calls (in the NEB stage at ~line 662 and the AutoNEB stage at ~line 815) become:

```python
            provenance=collect_provenance(
                mlip_model=self.mlip,
                device_requested=self.device,
                device_resolved=resolve_device(self.device),
                uma_task=self.uma_task,
                mace_head=getattr(self, "mace_head", None),
            ),
```

`getattr` rather than `self.mace_head`: `from_restart` sets the attribute on an instance built without it, and line 347 already uses the same defensive read.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_core_md.py tests/test_core_optimize.py tests/test_core_neb.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full unit suite**

Run: `pytest -m "not uma and not mace and not sevenn"`
Expected: PASS, goldens unmoved.

- [ ] **Step 8: Commit**

```bash
git add src/mliprun/core/optimize.py src/mliprun/core/md.py src/mliprun/core/neb.py \
        tests/test_core_md.py tests/test_core_optimize.py
git commit -m "feat(core): pass the head/task through to the run record

run_optimization and run_md take uma_task/mace_head and forward them to
collect_provenance; both NEB stages read them off the CustomNEB instance,
which already stores them.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Thread the head through the CLI

**Files:**
- Modify: `src/mliprun/cli/commands/optimize.py:124-140` (`optimize run`), `:286-302` (`optimize batch`)
- Modify: `src/mliprun/cli/commands/md.py:215-238` (`md run`)
- Test: `tests/test_cli_commands.py`

**Interfaces:**
- Consumes: `run_optimization(..., uma_task=..., mace_head=...)` and `run_md(..., uma_task=..., mace_head=...)` from Task 2.
- Produces: no new API. The CLI commands already parse `uma_task` and `mace_head` into local variables of those names; this task only forwards them.

Note: `neb` and `autoneb` CLI commands need no change — they already pass `uma_task` and `mace_head` into the `CustomNEB` constructor, and Task 2 reads them off the instance. Verify this in Step 3 before concluding the task.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli_commands.py`. This uses typer's in-process runner (the file's existing `_run_cli` shells out, which cannot observe call arguments) and stubs both the calculator setup and the engine, so no MLIP is required:

```python
from typer.testing import CliRunner

from mliprun.cli.commands.md import app as md_app


class TestCLIForwardsHead:
    """The head is the one level-of-theory fact the record cannot infer,
    so the CLI must hand it to the engine (CANON C1)."""

    def test_md_run_forwards_the_uma_task(self, tmp_path, monkeypatch):
        from ase.build import bulk
        from ase.io import write

        structure = tmp_path / "POSCAR"
        write(str(structure), bulk("Cu", "fcc", a=3.6))

        captured = {}
        monkeypatch.setattr("mliprun.cli.commands.md.setup_calculator",
                            lambda atoms, *a, **k: atoms)
        monkeypatch.setattr("mliprun.cli.commands.md.validate_mlip",
                            lambda *a, **k: None)
        monkeypatch.setattr("mliprun.cli.commands.md.run_md",
                            lambda **kwargs: captured.update(kwargs))

        result = CliRunner().invoke(md_app, [
            "run", "--structure", str(structure), "--mlip", "uma-s-1p2",
            "--uma-task", "oc25", "--steps", "1",
            "--output-dir", str(tmp_path),
        ])

        assert result.exit_code == 0, result.output
        assert captured["uma_task"] == "oc25"
        assert captured["mace_head"] is not None or "mace_head" in captured
```

Before running, confirm the exact option names with `md run --help` — if `--output-dir` is not an option on this command, drop it (the command defaults to writing next to the input structure, which is already `tmp_path`).

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cli_commands.py::TestCLIForwardsHead -v`
Expected: FAIL with `KeyError: 'uma_task'` — the CLI does not currently pass it.

- [ ] **Step 3: Forward from `md run`**

In `src/mliprun/cli/commands/md.py`, add to the `run_md(...)` call, next to the existing `device_resolved=_resolve_device(device),`:

```python
        uma_task=uma_task,
        mace_head=mace_head,
```

- [ ] **Step 4: Forward from both optimize entry points**

In `src/mliprun/cli/commands/optimize.py`, add the same two lines to the `run_optimization(...)` call in `run` (next to its `device_resolved=` argument) and to the one in `batch`.

- [ ] **Step 5: Verify neb/autoneb need no change**

Run: `grep -n "CustomNEB(" -A 15 src/mliprun/cli/commands/neb.py src/mliprun/cli/commands/autoneb.py | grep -n "uma_task\|mace_head"`
Expected: both constructors already receive `uma_task=` and `mace_head=`. If either does not, add it — the constructor accepts both.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_cli_commands.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full unit suite**

Run: `pytest -m "not uma and not mace and not sevenn"`
Expected: PASS.

- [ ] **Step 8: Update the run-record docstring pointer**

In `src/mliprun/core/run_record.py`, change the design pointer line in the module docstring to name both documents:

```
Design: docs/superpowers/specs/2026-07-21-unified-run-record-design.md
Amended: docs/superpowers/specs/2026-07-29-run-record-head-provenance-design.md
```

- [ ] **Step 9: Add a CHANGELOG entry**

Under the unreleased/top section of `CHANGELOG.md`, matching the file's existing format:

```markdown
- Run record (`mliprun_run.json`) now records `uma_task` / `mace_head` in its
  `provenance` block, so the record identifies the level of theory on its own.
  Schema version 2.
```

- [ ] **Step 10: Commit**

```bash
git add src/mliprun/cli/commands/md.py src/mliprun/cli/commands/optimize.py \
        src/mliprun/core/run_record.py tests/test_cli_commands.py CHANGELOG.md
git commit -m "feat(cli): forward the head/task to the engines

optimize run, optimize batch, and md run now pass their parsed
--uma-task / --mace-head through to the core, so the run record names the
head that actually ran. neb/autoneb already pass both into CustomNEB.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 11: Open the draft PR**

```bash
git push -u origin feat/run-record-head-provenance
gh pr create --draft --title "feat: carry the MLIP head/task in the run record (schema v2)" \
  --body "$(cat <<'EOF'
`collect_provenance` took only the model tag, so `mliprun_run.json` named
`uma-s-1p2` without the `oc25` task that actually ran. Under CANON C1 the head
is its own explicit decision and under C3 heads must never be mixed within an
energy formula, so the record could not serve as provenance for any energy
entering a formula.

- `uma_task` / `mace_head` as always-present nullable provenance fields, gated
  on the model tag inside `collect_provenance` so a caller passing its parsed
  option default cannot mislabel a run
- `SCHEMA_VERSION` 1 -> 2, so a reader can tell an old record from a
  genuinely-null field
- both fields added to `_PROVENANCE_DIFF_FIELDS`, so a resume that switches
  head is recorded in `stage_provenance`
- threaded through `run_optimization`, `run_md`, both NEB stages, and the
  optimize/md CLI commands

No numerics touched; goldens unmoved.

Design: `docs/superpowers/specs/2026-07-29-run-record-head-provenance-design.md`
Driven by prov's MD capture work, which needs the head to record MD runs.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `collect_provenance` gains `uma_task` / `mace_head` | 1 |
| Always present, null when not applicable | 1 |
| Gated on model family | 1 |
| `SCHEMA_VERSION` 1 → 2 | 1 |
| `_PROVENANCE_DIFF_FIELDS` extended | 1 |
| `run_optimization` threading | 2 |
| `run_md` threading | 2 |
| NEB / AutoNEB call sites | 2 |
| CLI call sites | 3 |
| Golden safety (no regeneration) | Global Constraints + Steps 1.7, 2.7, 3.7 |
| Tests written first | every task, Steps 1-2 |
| Docstring pointer updated | 3.8 |
| Out of scope: no backfill of existing records | not implemented anywhere — correct |

**Placeholder scan:** none. Two steps ask the implementer to verify something before proceeding (3.1's option-name check, 3.5's neb/autoneb check) — both carry the exact command to run and the expected result, and both have a stated action if the expectation fails.

**Type consistency:** `uma_task` and `mace_head` are `Optional[str]` defaulting to `None` at every layer — `collect_provenance`, `run_optimization`, `run_md`, and the CLI locals of the same names. `SCHEMA_VERSION` is `int` 2 in the constant, the assertion, and the written record.
