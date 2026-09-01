# SevenNet Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn mliprun's never-executed SevenNet stub into working support for the SevenNet 0.13.0 model family, with the inference task chosen explicitly by the user and recorded in the run record.

**Architecture:** A module-level tag → task table in `cli/utils.py` becomes the single source of truth for which SevenNet tags exist and which tasks each accepts. `validate_mlip` enforces the table before any model loads. A new `sevennet_task` value threads through the same call path that `uma_task` and `mace_head` already follow, ending in the run record as a schema-3 provenance field.

**Tech Stack:** Python 3.11, typer CLI, ASE, `sevenn` 0.13.0 (`SevenNetCalculator`), pytest. Heavy MLIP imports stay lazy.

**Spec:** `docs/superpowers/specs/2026-09-01-sevennet-refresh-design.md`

## Global Constraints

- **Branch:** `feat/sevennet-refresh`, off `main`. Draft PR only, never push to `main`.
- **One MLIP per environment.** Never add `sevenn` to an env holding `mace-torch`, `fairchem-core`, or `chgnet`.
- **Never regenerate goldens.** `tests/goldens/*.json` are frozen. If a golden test fails, report the numerical delta and stop; do not update the baseline.
- **Every test asserts a number or an invariant.** Loosening a tolerance is acceptable only with the observed delta recorded in the commit message.
- **Unit tests must pass with no MLIP installed:** `pytest -m "not uma and not mace and not sevenn"`.
- **Never name a pytest parametrize id `uma`, `mace`, or `sevenn`.** `tests/conftest.py:65` matches on `item.keywords`, which includes parametrize ids, so such an id silently skips the whole test.
- **Diff coverage ≥ 90%** on changed lines (`.github/workflows/tests.yml`).
- **Lazy imports:** `from sevenn.calculator import SevenNetCalculator` only inside a function, never at module top level.
- **No default for `--sevennet-task`.** CANON C1. The typer default is `None`, and `None` on a multi-task tag is an error, never a fallback.
- **Exit style:** `validate_mlip` raises `typer.Exit(<message string>)`, matching the existing (unusual but established) pattern in that function. Do not change it to `typer.Exit(code=1)` plus an echo; that is a separate concern.
- **Cluster env:** `/scratchb/juar/EWaste2GreenCat/explicit_solvation/.venv/sevenn`, conda prefix env, python 3.11, torch 2.8.0+cu128. Never install or run from `/home`.
- **Never check out this branch in `/app1-cos/mlip-platform/mlip-platform`.** That clone is the shared editable install behind the `uma`, `mace`, and `chgnet` environments; switching its branch changes the code under other people's running jobs. Use `/scratchb/juar/sevennet-dev/mliprun`.

---

### Task 1: Verify the SevenNet model/task table against the installed package  ✅ DONE 2026-09-01

The task strings in the spec were transcribed from the SevenNet documentation. A wrong string in a head table silently changes the level of theory, so the table is verified against the installed package before any of it is committed. This task produces the authoritative table that Task 2 encodes.

**Files:**
- Create: `/scratchb/juar/sevennet-dev/probe_tasks.py` (cluster-side throwaway, not committed)
- Modify: `docs/superpowers/specs/2026-09-01-sevennet-refresh-design.md` (only if a string differs)

**Interfaces:**
- Consumes: nothing.
- Produces: the verified tag → task tuple mapping used verbatim in Task 2.

- [x] **Step 1: Confirm the env is built**

```bash
ssh cos-cluster 'tail -6 /scratchb/juar/sevennet-dev/install_sevenn.log'
```

Expected: the `=== versions ===` block showing `torch 2.8.0 cuda 12.8 avail True` and a `sevenn` version, followed by `DONE`.

- [x] **Step 2: Read the checkpoint metadata for each multi-task model**

`sevenn cp <tag>` prints a checkpoint overview including the modal (task) list, without running the model.

```bash
ssh cos-cluster 'V=/scratchb/juar/EWaste2GreenCat/explicit_solvation/.venv/sevenn
for t in 7net-omni 7net-mf-ompa; do echo "##### $t"; $V/bin/sevenn cp $t 2>&1 | head -40; done'
```

Expected: a modal/task list for each. Record the exact strings.

- [x] **Step 3: Cross-check by instantiating with a task**

If `sevenn cp` does not print the modal list, get it from the checkpoint directly:

```python
# /scratchb/juar/sevennet-dev/probe_tasks.py
import torch
from sevenn.util import pretrained_name_to_path

for tag in ("7net-omni", "7net-mf-ompa"):
    path = pretrained_name_to_path(tag)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    print(tag, "->", cfg.get("_modal_map") or cfg.get("modal_map") or cfg.keys())
```

Run: `ssh cos-cluster '<venv>/bin/python /scratchb/juar/sevennet-dev/probe_tasks.py'`
Expected: a dict or list of task names per tag.

- [x] **Step 4: Locate the checkpoint cache**

```bash
ssh cos-cluster 'du -sh ~/.cache/* 2>/dev/null | sort -h | tail -5; ls -la ~/.cache/sevennet 2>/dev/null | head'
```

Expected: the directory SevenNet downloads weights into, and its size. `/home` is quota'd and non-executable on this cluster; if the cache lands there and is large, note the environment variable or symlink needed to move it to `/scratchb`, and record it in the install recipe in Task 9.

- [x] **Step 5: Reconcile the table**

Compare the observed strings against the spec's table. If any differ, edit the spec's table to the observed values and commit that correction on its own:

```bash
git add docs/superpowers/specs/2026-09-01-sevennet-refresh-design.md
git commit -m "docs(spec): correct SevenNet task strings against sevenn 0.13.0"
```

If they all match, record that in the next commit message instead. **Do not proceed to Task 2 with an unverified table.**

---

### Task 2: Model/task table and `validate_mlip` enforcement

**Files:**
- Modify: `src/mliprun/cli/utils.py` (add table near `_TAG_TO_RECIPE` around line 108; extend `validate_mlip` at lines 209-243)
- Test: `tests/test_cli_utils.py`

**Interfaces:**
- Consumes: the verified table from Task 1.
- Produces:
  - `_SEVENNET_MODELS: dict[str, tuple[str, ...]]` — tag → task tuple; empty tuple means single-task.
  - `validate_mlip(mlip: str, sevennet_task: Optional[str] = None) -> None`
  - `SEVENNET_TASK_HELP: str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_utils.py — add to the imports
from mliprun.cli.utils import _SEVENNET_MODELS

class TestSevenNetTable:
    def test_every_documented_tag_is_present(self):
        # Verified against sevenn 0.13.0:
        # sevenn.util.get_available_pretrained_models(). There are no
        # 7net-nano-* tags in this release despite the documentation.
        expected = {
            "7net-omni", "7net-omni-i8", "7net-omni-i12",
            "7net-mf-ompa", "7net-mf-0",
            "7net-omat", "7net-l3i5", "7net-0", "7net-0_22may2024",
        }
        assert set(_SEVENNET_MODELS) == expected

    def test_multi_task_models_carry_tasks(self):
        assert len(_SEVENNET_MODELS["7net-omni"]) == 13
        assert _SEVENNET_MODELS["7net-mf-ompa"] == ("omat24", "mpa")
        assert "oc20" in _SEVENNET_MODELS["7net-omni"]

    def test_mf_0_tasks_are_uppercase(self):
        # 7net-mf-0 is the one tag whose checkpoint names its modalities in
        # uppercase. Storing them lowercased would send SevenNet a task its
        # checkpoint does not have.
        assert _SEVENNET_MODELS["7net-mf-0"] == ("PBE", "R2SCAN")

    def test_single_task_models_carry_none(self):
        for tag in ("7net-omat", "7net-l3i5", "7net-0", "7net-0_22may2024"):
            assert _SEVENNET_MODELS[tag] == ()

    def test_omni_variants_share_one_task_list(self):
        assert _SEVENNET_MODELS["7net-omni-i8"] == _SEVENNET_MODELS["7net-omni"]
        assert _SEVENNET_MODELS["7net-omni-i12"] == _SEVENNET_MODELS["7net-omni"]


@patch("mliprun.cli.utils.SEVENN_AVAILABLE", True)
class TestValidateSevenNetTask:
    def test_multi_task_without_task_raises(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("7net-omni")
        assert "sevennet-task" in str(exc.value.exit_code)

    def test_multi_task_error_lists_valid_tasks(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("7net-mf-ompa")
        message = str(exc.value.exit_code)
        assert "mpa" in message and "omat24" in message

    def test_invalid_task_raises(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("7net-omni", sevennet_task="not_a_task")
        assert "not_a_task" in str(exc.value.exit_code)

    def test_task_on_single_task_model_raises(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("7net-0", sevennet_task="mpa")
        assert "no selectable task" in str(exc.value.exit_code)

    def test_valid_pair_passes(self):
        validate_mlip("7net-omni", sevennet_task="oc20")
        validate_mlip("7net-mf-ompa", sevennet_task="mpa")
        validate_mlip("7net-mf-0", sevennet_task="PBE")

    def test_task_matching_is_case_sensitive(self):
        # 'pbe' is not a task 7net-mf-0 has; accepting it would invent a name.
        with pytest.raises(typer.Exit):
            validate_mlip("7net-mf-0", sevennet_task="pbe")

    def test_single_task_model_without_task_passes(self):
        validate_mlip("7net-0")

    def test_unknown_7net_tag_passes_with_warning(self, capsys):
        validate_mlip("7net-future-model", sevennet_task="whatever")
        assert "can be validated" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_cli_utils.py -k "SevenNet" -v`
Expected: FAIL — `ImportError: cannot import name '_SEVENNET_MODELS'`.

- [ ] **Step 3: Add the table and the help string**

Insert after `_TAG_TO_RECIPE` in `src/mliprun/cli/utils.py`. Replace the task tuples with the values verified in Task 1 if they differ.

```python
#: SevenNet tag -> selectable inference tasks ("modals" in the SevenNet API).
#: An empty tuple marks a single-task model, which rejects --sevennet-task.
#: Tasks are independent fine-tunes with independent energy zeros (CANON C3),
#: so the task is never defaulted -- see `validate_mlip`.
# Task names are stored exactly as the checkpoints report them (the
# `Modality` line of `sevenn cp <tag>`). 7net-mf-0 uses uppercase where every
# other model uses lowercase, so comparison is exact -- never case-folded.
_SEVENNET_OMNI_TASKS = (
    "omat24", "mpa", "omol25_low", "omol25_high", "matpes_pbe",
    "matpes_r2scan", "mp_r2scan", "oc20", "oc22", "spice", "qcml",
    "odac23", "pet_mad",
)

_SEVENNET_MODELS: dict[str, tuple[str, ...]] = {
    "7net-omni": _SEVENNET_OMNI_TASKS,
    "7net-omni-i8": _SEVENNET_OMNI_TASKS,
    "7net-omni-i12": _SEVENNET_OMNI_TASKS,
    "7net-mf-ompa": ("omat24", "mpa"),
    "7net-mf-0": ("PBE", "R2SCAN"),
    "7net-omat": (),
    "7net-l3i5": (),
    "7net-0": (),
    "7net-0_22may2024": (),
}
```

Add the help constant next to `UMA_TASK_HELP`:

```python
SEVENNET_TASK_HELP = (
    "SevenNet inference task (the 'modal' in SevenNet's API). Required for "
    "multi-task models and rejected for single-task ones -- there is no "
    "default, because tasks are independent fine-tunes with independent "
    "energy zeros. 7net-omni/-i8/-i12: 'mpa' (PBE+U, the SevenNet-recommended "
    "general default), 'oc20' (RPBE catalysis on surfaces), 'oc22', "
    "'matpes_pbe', 'odac23', 'omol25_low', 'omol25_high', 'spice', 'qcml', "
    "'pet_mad', 'mp_r2scan', 'matpes_r2scan'. 7net-mf-ompa: 'omat24' or 'mpa'. "
    "7net-mf-0: 'PBE' or 'R2SCAN' (uppercase). Task names are matched exactly."
)
```

- [ ] **Step 4: Extend `validate_mlip`**

Change the signature and add SevenNet handling. The existing `7net-mf-ompa` special case at line 214 is replaced by the table lookup.

```python
def validate_mlip(mlip: str, sevennet_task: Optional[str] = None) -> None:
    """Validate that the specified MLIP -- and its task, if any -- is usable.

    Parameters
    ----------
    mlip : str
        MLIP model name to validate.
    sevennet_task : str, optional
        SevenNet inference task. Required for multi-task SevenNet tags and
        rejected for single-task ones. Ignored for non-SevenNet models.

    Raises
    ------
    typer.Exit
        If the MLIP is not installed, the tag is unknown, or the task is
        missing, invalid, or given to a model that has none.
    """
    if mlip == "auto":
        return

    if mlip.startswith("7net"):
        if not SEVENN_AVAILABLE:
            raise typer.Exit(_install_message("SevenNet", mlip))
        _validate_sevennet_task(mlip, sevennet_task)
        return

    if mlip == "mace" and not MACE_AVAILABLE:
        ...  # unchanged from here down, minus the old 7net-mf-ompa branch
```

Remove `"7net-mf-ompa"` from the final unknown-tag tuple so the check reads
`if not (mlip in ["mace", "chgnet"] or mlip.startswith("uma-") or mlip.startswith("mace-mh-")):`,
and drop `'7net-mf-ompa'` from that error message's suggestion list in favour of `'7net-omni'`.

Add the helper directly above `validate_mlip`:

```python
def _validate_sevennet_task(mlip: str, sevennet_task: Optional[str]) -> None:
    """Check a SevenNet tag against its task list.

    Unknown ``7net-*`` tags are forwarded to SevenNet unchanged so that new
    checkpoints work without a code change; the trade-off is that neither the
    tag nor the task can be checked, which is stated on stdout.
    """
    if mlip not in _SEVENNET_MODELS:
        typer.echo(
            f"Warning: '{mlip}' is not a tag mliprun knows. It will be passed "
            f"to SevenNet unchanged; neither it nor the task "
            f"'{sevennet_task}' can be validated. Known tags: "
            f"{', '.join(sorted(_SEVENNET_MODELS))}."
        )
        return

    tasks = _SEVENNET_MODELS[mlip]
    if not tasks:
        if sevennet_task is not None:
            raise typer.Exit(
                f"{mlip} is a single-task model and has no selectable task, "
                f"but --sevennet-task {sevennet_task} was given. Drop the flag."
            )
        return

    if sevennet_task is None:
        raise typer.Exit(
            f"{mlip} is a multi-task model: --sevennet-task is required and "
            f"has no default, because its tasks are independent fine-tunes "
            f"with independent energy zeros. Valid tasks: {', '.join(tasks)}."
        )

    if sevennet_task not in tasks:
        raise typer.Exit(
            f"Unknown SevenNet task '{sevennet_task}' for {mlip}. "
            f"Valid tasks: {', '.join(tasks)}."
        )
```

The warning message and the Step 1 test assertion must agree on the substring `"can be validated"`.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_cli_utils.py -v`
Expected: the new SevenNet tests PASS. `test_sevenn_unavailable_raises` still passes. `test_falls_back_to_sevenn` still passes at this point (it is updated in Task 3).

- [ ] **Step 6: Commit**

```bash
git add src/mliprun/cli/utils.py tests/test_cli_utils.py
git commit -m "feat(sevennet): tag/task table with no-default task validation"
```

---

### Task 3: `detect_mlip` picks `7net-omni`; recipe map entries

**Files:**
- Modify: `src/mliprun/cli/utils.py` (`detect_mlip` line ~198; `_TAG_TO_RECIPE` line ~108; `MLIP_HELP` line 16)
- Test: `tests/test_cli_utils.py:48`

**Interfaces:**
- Consumes: `_SEVENNET_MODELS` from Task 2.
- Produces: `detect_mlip()` returning `"7net-omni"` in a SevenNet-only env.

- [ ] **Step 1: Update the existing failing-by-design test**

`tests/test_cli_utils.py:48` currently asserts the old tag. Replace it:

```python
    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", False)
    @patch("mliprun.cli.utils.MACE_AVAILABLE", False)
    @patch("mliprun.cli.utils.SEVENN_AVAILABLE", True)
    def test_falls_back_to_sevenn(self):
        # 7net-omni is SevenNet's recommended model. It is multi-task, so
        # `--mlip auto` in a SevenNet-only env resolves here and then stops
        # for a missing --sevennet-task rather than guessing a head (C1).
        assert detect_mlip() == "7net-omni"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_cli_utils.py::TestDetectMlip::test_falls_back_to_sevenn -v`
Expected: FAIL — `assert '7net-mf-ompa' == '7net-omni'`.

- [ ] **Step 3: Change `detect_mlip` and the tag maps**

In `detect_mlip`, change `return "7net-mf-ompa"` to `return "7net-omni"`, and extend its docstring to say the returned tag still requires an explicit task.

In `_TAG_TO_RECIPE`, replace the single `"7net-mf-ompa": "sevenn.md"` entry with one per tag:

```python
    "7net-omni": "sevenn.md#7net-omni",
    "7net-omni-i8": "sevenn.md#7net-omni-i8",
    "7net-omni-i12": "sevenn.md#7net-omni-i12",
    "7net-mf-ompa": "sevenn.md#7net-mf-ompa",
    "7net-mf-0": "sevenn.md#7net-mf-0",
    "7net-omat": "sevenn.md#7net-omat",
    "7net-l3i5": "sevenn.md#7net-l3i5",
    "7net-0": "sevenn.md#7net-0",
    "7net-0_22may2024": "sevenn.md#7net-0",
```

In `_recipe_for_tag`, add a prefix fallback so unknown `7net-*` tags still get the file:

```python
    if mlip.startswith("7net"):
        return "sevenn.md"
```

In `MLIP_HELP`, replace `'7net-mf-ompa'` with `"any '7net-*' tag (e.g. '7net-omni', which needs --sevennet-task)"`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_cli_utils.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/cli/utils.py tests/test_cli_utils.py
git commit -m "feat(sevennet): auto-detect resolves to 7net-omni; per-tag recipe anchors"
```

---

### Task 4: `build_calculator` / `setup_calculator` take the task

**Files:**
- Modify: `src/mliprun/cli/utils.py:365-451`
- Test: `tests/test_cli_utils.py`

**Interfaces:**
- Consumes: `_SEVENNET_MODELS`.
- Produces:
  - `build_calculator(mlip, uma_task="omat", device="auto", mace_head="omat_pbe", sevennet_task=None)`
  - `setup_calculator(atoms, mlip, uma_task="omat", device="auto", mace_head="omat_pbe", sevennet_task=None)`

- [ ] **Step 1: Write the failing test**

The calculator class is patched, so this runs with no `sevenn` installed.

```python
from unittest.mock import MagicMock, patch
from mliprun.cli.utils import build_calculator

class TestBuildSevenNetCalculator:
    def test_passes_tag_and_task_to_calculator(self):
        fake_cls = MagicMock()
        with patch("mliprun.cli.utils._load_sevenn_calculator", return_value=fake_cls):
            build_calculator("7net-omni", device="cpu", sevennet_task="oc20")
        fake_cls.assert_called_once_with("7net-omni", modal="oc20", device="cpu")

    def test_single_task_model_gets_no_modal(self):
        fake_cls = MagicMock()
        with patch("mliprun.cli.utils._load_sevenn_calculator", return_value=fake_cls):
            build_calculator("7net-0", device="cpu")
        fake_cls.assert_called_once_with("7net-0", device="cpu")

    def test_unknown_tag_forwards_task_when_given(self):
        fake_cls = MagicMock()
        with patch("mliprun.cli.utils._load_sevenn_calculator", return_value=fake_cls):
            build_calculator("7net-future", device="cpu", sevennet_task="mpa")
        fake_cls.assert_called_once_with("7net-future", modal="mpa", device="cpu")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_cli_utils.py -k BuildSevenNet -v`
Expected: FAIL — `build_calculator() got an unexpected keyword argument 'sevennet_task'`.

- [ ] **Step 3: Implement**

Change both signatures to add `sevennet_task: Optional[str] = None`, document the parameter in both NumPy-style docstrings, and replace the `7net-mf-ompa` branch in `build_calculator` with:

```python
    elif mlip.startswith("7net"):
        SevenNetCalculator = _load_sevenn_calculator()
        if sevennet_task is None:
            return SevenNetCalculator(mlip, device=device)
        return SevenNetCalculator(mlip, modal=sevennet_task, device=device)
```

`setup_calculator` forwards it:

```python
    atoms.calc = build_calculator(mlip, uma_task, device=device,
                                  mace_head=mace_head,
                                  sevennet_task=sevennet_task)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_cli_utils.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/cli/utils.py tests/test_cli_utils.py
git commit -m "feat(sevennet): thread sevennet_task into calculator construction"
```

---

### Task 5: Run record schema 3

**Files:**
- Modify: `src/mliprun/core/run_record.py:32` (`SCHEMA_VERSION`), `:54` (`_PROVENANCE_DIFF_FIELDS`), `:166-231` (`collect_provenance`)
- Test: `tests/test_run_record.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `collect_provenance(*, mlip_model, device_requested, device_resolved, uma_task=None, mace_head=None, sevennet_task=None) -> dict` with a `"sevennet_task"` key always present.

- [ ] **Step 1: Write the failing tests**

```python
from mliprun.core.run_record import SCHEMA_VERSION, collect_provenance, _split_provenance

class TestSevenNetProvenance:
    def test_schema_version_is_three(self):
        assert SCHEMA_VERSION == 3

    def test_sevennet_run_carries_the_task(self):
        prov = collect_provenance(
            mlip_model="7net-omni", device_requested="auto",
            device_resolved="cuda", sevennet_task="oc20",
        )
        assert prov["sevennet_task"] == "oc20"
        assert prov["uma_task"] is None
        assert prov["mace_head"] is None

    def test_non_sevennet_run_nulls_the_task(self):
        # A caller passing the flag's value regardless of model must not have
        # it recorded against a model that never used it (CANON C1/C3).
        prov = collect_provenance(
            mlip_model="mace", device_requested="cpu",
            device_resolved="cpu", sevennet_task="oc20",
        )
        assert prov["sevennet_task"] is None

    def test_key_always_present(self):
        prov = collect_provenance(
            mlip_model="uma-s-1p2", device_requested="cpu",
            device_resolved="cpu", uma_task="omat",
        )
        assert "sevennet_task" in prov and prov["sevennet_task"] is None

    def test_task_switch_is_reported_as_changed(self):
        top = {"mlip_model": "7net-omni", "sevennet_task": "mpa"}
        incoming = {"mlip_model": "7net-omni", "sevennet_task": "oc20"}
        changed, new = _split_provenance(incoming, top)
        assert changed == {"sevennet_task": "oc20"}
        assert new == {}

    def test_absent_key_in_legacy_record_is_new_not_changed(self):
        # A schema-2 record has no sevennet_task key at all. That is "unknown",
        # not "different" -- reporting it as changed would be a C3 false
        # positive in the artifact that answers C3 questions.
        top = {"mlip_model": "7net-omni"}
        incoming = {"mlip_model": "7net-omni", "sevennet_task": "mpa"}
        changed, new = _split_provenance(incoming, top)
        assert changed == {}
        assert new == {"sevennet_task": "mpa"}
```

Also update the existing schema assertion: `grep -n "schema_version\|SCHEMA_VERSION" tests/test_run_record.py tests/test_run_record_integration.py` and change any `== 2` to `== 3`.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_run_record.py -k SevenNet -v`
Expected: FAIL — `assert 2 == 3`.

- [ ] **Step 3: Implement**

```python
SCHEMA_VERSION = 3
```

```python
_PROVENANCE_DIFF_FIELDS = ("mliprun_version", "hostname", "device_resolved",
                           "mlip_model", "uma_task", "mace_head",
                           "sevennet_task")
```

Add the keyword-only parameter to `collect_provenance` and the gated key next to its siblings:

```python
        "sevennet_task": sevennet_task if tag.startswith("7net") else None,
```

Extend the docstring paragraph that explains the gating so it names all three fields, not two.

- [ ] **Step 4: Run the full record suite**

Run: `pytest tests/test_run_record.py tests/test_run_record_integration.py -v`
Expected: all PASS. If a golden comparison fails, **stop and report the delta** — do not regenerate.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/run_record.py tests/test_run_record.py tests/test_run_record_integration.py
git commit -m "feat(record): carry sevennet_task in provenance (schema v3)"
```

---

### Task 6: `optimize run`, `optimize batch`, and `md run`

**Files:**
- Modify: `src/mliprun/cli/commands/optimize.py` (options at :55-57 and :184-186; echo at :99-103; record call at :139; `_write_params` at :345-355; batch echo at :245-248 and record at :303)
- Modify: `src/mliprun/cli/commands/md.py` (option at :43-45; echo at :124-156; params at :172-174; record at :238)
- Modify: `src/mliprun/core/optimize.py:56-57,164-165` and `src/mliprun/core/md.py:239-240,335-336`
- Test: `tests/test_cli_commands.py`, `tests/test_core_optimize.py`

**Interfaces:**
- Consumes: `setup_calculator(..., sevennet_task=...)`, `collect_provenance(..., sevennet_task=...)`, `SEVENNET_TASK_HELP`.
- Produces: `--sevennet-task` on `optimize run`, `optimize batch`, `md run`; `run_optimization(..., sevennet_task=None)`; `run_md(..., sevennet_task=None)`.

- [ ] **Step 1: Write the failing tests**

```python
from typer.testing import CliRunner
from mliprun.cli.main import app

runner = CliRunner()

class TestSevenNetTaskOption:
    def test_optimize_run_rejects_multi_task_without_task(self, tmp_path):
        structure = _write_test_poscar(tmp_path)   # existing helper in this file
        with patch("mliprun.cli.utils.SEVENN_AVAILABLE", True):
            result = runner.invoke(app, ["optimize", "run", "--structure",
                                         str(structure), "--mlip", "7net-omni"])
        assert result.exit_code != 0
        assert "--sevennet-task is required" in result.output

    def test_optimize_run_help_lists_the_option(self):
        result = runner.invoke(app, ["optimize", "run", "--help"])
        assert "--sevennet-task" in result.output

    def test_md_run_help_lists_the_option(self):
        result = runner.invoke(app, ["md", "run", "--help"])
        assert "--sevennet-task" in result.output

    def test_optimize_batch_help_lists_the_option(self):
        result = runner.invoke(app, ["optimize", "batch", "--help"])
        assert "--sevennet-task" in result.output
```

Add a params-file test asserting the value is written:

```python
def test_opt_params_records_the_task(tmp_path):
    from mliprun.cli.commands.optimize import _write_params
    path = tmp_path / "opt_params.txt"
    _write_params(path, "7net-omni", "omat", "omat_pbe", "cuda", False,
                  "POSCAR", "bfgs", 0.05, 200, True, tmp_path,
                  sevennet_task="oc20")
    text = path.read_text()
    assert "SevenNet task:     oc20" in text
    assert "UMA task" not in text
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_cli_commands.py -k SevenNetTask -v`
Expected: FAIL — the option does not exist, `--help` has no `--sevennet-task`.

- [ ] **Step 3: Implement the CLI layer**

In each of the three commands, add the option next to `mace_head`:

```python
    sevennet_task: str = typer.Option(None, help=SEVENNET_TASK_HELP),
```

Import `SEVENNET_TASK_HELP` from `mliprun.cli.utils` alongside the other help constants. Add the echo line next to the existing head echoes:

```python
    if mlip.startswith("7net"):
        typer.echo(f"   SevenNet task: {sevennet_task}")
```

Forward it into `setup_calculator(...)` / `build_calculator(...)` and into the `run_optimization` / `run_md` call. `resolve_mlip` must now carry the task through so validation sees it — change its signature to `resolve_mlip(mlip: str, sevennet_task: Optional[str] = None) -> str` and have it call `validate_mlip(mlip, sevennet_task)`. Update every call site (`grep -rn "resolve_mlip(" src/`).

Add the parameter to `_write_params` and write it gated on the tag:

```python
        if mlip.startswith("7net"):
            f.write(f"SevenNet task:     {sevennet_task}\n")
```

- [ ] **Step 4: Implement the core layer**

Add `sevennet_task: Optional[str] = None` to `run_optimization` (`core/optimize.py:56`) and `run_md` (`core/md.py:239`), document it, and forward it into `collect_provenance(...)` at `optimize.py:164` and `md.py:335`.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_cli_commands.py tests/test_core_optimize.py tests/test_core_md.py tests/test_optimize_batch.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mliprun/cli/commands/optimize.py src/mliprun/cli/commands/md.py \
        src/mliprun/core/optimize.py src/mliprun/core/md.py \
        src/mliprun/cli/utils.py tests/
git commit -m "feat(sevennet): --sevennet-task on optimize run/batch and md run"
```

---

### Task 7: NEB and AutoNEB, including the restart round-trip

`CustomNEB` wires calculators itself, separately from `cli/utils.py`, and persists its parameters to `neb_params.txt` for `--restart`. Both need the task or a restart silently resumes on a different energy zero.

**Files:**
- Modify: `src/mliprun/core/neb.py` (`__init__` :103-119; params map :161; `from_restart` :226-308; `setup_calculator` :330-370; record calls :671-686 and :840-850)
- Modify: `src/mliprun/cli/commands/neb.py` (:73-108, :162-163, :183-184, :232-283, :316, :347-348, :381-393)
- Modify: `src/mliprun/cli/commands/autoneb.py` (:25-26, :54-56, :89-90, :118)
- Test: `tests/test_neb_restart.py`, `tests/test_core_neb.py`

**Interfaces:**
- Consumes: `SEVENNET_TASK_HELP`, `collect_provenance(..., sevennet_task=...)`.
- Produces: `CustomNEB(..., sevennet_task: Optional[str] = None)`, `self.sevennet_task`, and the `neb_params.txt` key `"SevenNet task"`.

- [ ] **Step 1: Write the failing round-trip test**

```python
class TestSevenNetTaskRestart:
    def test_task_survives_the_params_round_trip(self, tmp_path):
        from mliprun.core.neb import CustomNEB
        params_file = tmp_path / "neb_params.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "==========================\n"
            "MLIP model:            7net-omni\n"
            "SevenNet task:         oc20\n"
            "Device:                cuda\n"
            "Initial:               initial.vasp\n"
            "Final:                 final.vasp\n"
            "Intermediate images:   5\n"
            "Total images:          7\n"
            "IDPP fmax:             0.1\n"
            "IDPP steps:            100\n"
            "Final fmax:            0.05\n"
            "Spring constant (k):   0.1\n"
            "Climb:                 True\n"
            "Dynamic NEB:           False\n"
            "Scale fmax:            0.0\n"
            "NEB optimizer:         None\n"
        )
        params = CustomNEB._parse_params_file(params_file)
        assert params["sevennet_task"] == "oc20"
        assert params["mlip"] == "7net-omni"

    def test_restart_without_an_override_keeps_the_stored_task(self, tmp_path):
        # A restart that silently dropped the task would resume the band on a
        # different energy zero -- the discontinuity CANON C3 forbids.
        ...  # build via the same fixture the DyNEB restart test uses
```

Use the exact fixture pattern already in `tests/test_neb_restart.py`; read that file first and mirror it rather than inventing a new one.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_neb_restart.py -k SevenNetTask -v`
Expected: FAIL — `KeyError: 'sevennet_task'`.

- [ ] **Step 3: Implement in `core/neb.py`**

Add to the parse map at :161, right after the MACE entry:

```python
            "SevenNet task": ("sevennet_task", str),
```

Add `sevennet_task: Optional[str] = None` to `__init__` (:103) and `self.sevennet_task = sevennet_task` (:119). In `from_restart`, mirror the existing head lines at :292-293 and :307-308 — note the existing `or` defaulting is wrong for SevenNet, which has no default:

```python
        sevennet_task = sevennet_task or params.get("sevennet_task")
        ...
        instance.sevennet_task = sevennet_task
```

In `CustomNEB.setup_calculator`, add the parameter and replace the `7net-mf-ompa` branch:

```python
        sevennet_task = sevennet_task or getattr(self, "sevennet_task", None)
        ...
        if model.startswith("7net"):
            from sevenn.calculator import SevenNetCalculator
            if sevennet_task is None:
                return SevenNetCalculator(model, device=device)
            return SevenNetCalculator(model, modal=sevennet_task, device=device)
```

Forward `sevennet_task=self.sevennet_task` into both `collect_provenance` calls (:680 and :847) and add `"sevennet_task": self.sevennet_task` to the params dicts at :671 and :840.

- [ ] **Step 4: Implement in the two CLI commands**

`neb.py`: add the typer option (`typer.Option(None, ...)`, matching the existing `None` defaults there), thread it through `_handle_restart`, the `params` dict at :162, the echo dict at :183 and :282 (`**({f"SevenNet task:": sevennet_task} if mlip.startswith("7net") else {})`), the restart echo at :105, and the `CustomNEB(...)` call at :316. Unlike `uma_task`/`mace_head` at :232 and :245, do **not** add a `sevennet_task = sevennet_task or "<default>"` line.

`autoneb.py`: the same option, echo, params entry, and pass-through at :118.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_neb_restart.py tests/test_core_neb.py tests/test_neb_dyneb.py tests/test_neb_constrained.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mliprun/core/neb.py src/mliprun/cli/commands/neb.py \
        src/mliprun/cli/commands/autoneb.py tests/
git commit -m "feat(sevennet): --sevennet-task on neb/autoneb with restart round-trip"
```

---

### Task 8: `benchmark run`

**Files:**
- Modify: `src/mliprun/cli/commands/benchmark.py:20-42,69`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `_SEVENNET_MODELS`, `setup_calculator(..., sevennet_task=...)`.
- Produces: `_available_models(sevennet_task: Optional[str] = None) -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
class TestBenchmarkSevenNet:
    @patch("mliprun.cli.commands.benchmark.SEVENN_AVAILABLE", True)
    @patch("mliprun.cli.commands.benchmark.FAIRCHEM_AVAILABLE", False)
    @patch("mliprun.cli.commands.benchmark.MACE_AVAILABLE", False)
    @patch("mliprun.cli.commands.benchmark.CHGNET_AVAILABLE", False)
    def test_auto_list_skips_sevennet_without_a_task(self):
        from mliprun.cli.commands.benchmark import _available_models
        assert _available_models() == []

    @patch("mliprun.cli.commands.benchmark.SEVENN_AVAILABLE", True)
    @patch("mliprun.cli.commands.benchmark.FAIRCHEM_AVAILABLE", False)
    @patch("mliprun.cli.commands.benchmark.MACE_AVAILABLE", False)
    @patch("mliprun.cli.commands.benchmark.CHGNET_AVAILABLE", False)
    def test_auto_list_includes_sevennet_with_a_task(self):
        from mliprun.cli.commands.benchmark import _available_models
        assert _available_models(sevennet_task="mpa") == ["7net-omni"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_benchmark.py -k BenchmarkSevenNet -v`
Expected: FAIL — `_available_models() takes 0 positional arguments`.

- [ ] **Step 3: Implement**

```python
def _available_models(sevennet_task: Optional[str] = None) -> list[str]:
    """MLIP tags installed in this env and runnable without further input.

    SevenNet's recommended model is multi-task, and its tasks have independent
    energy zeros, so it is included only when a task was given. Benchmarking it
    under an assumed task would report a number nobody chose.
    """
    models = []
    if FAIRCHEM_AVAILABLE:
        models.append("uma-s-1p2")
    if MACE_AVAILABLE:
        models.append("mace")
    if SEVENN_AVAILABLE and sevennet_task is not None:
        models.append("7net-omni")
    if CHGNET_AVAILABLE:
        models.append("chgnet")
    return models
```

Add the `--sevennet-task` option to the `run` command, pass it to `_available_models(sevennet_task)`, and print a note when it is skipped:

```python
        if SEVENN_AVAILABLE and sevennet_task is None:
            typer.echo("Note: SevenNet is installed but skipped -- its "
                       "recommended model is multi-task. Pass --sevennet-task "
                       "to include it.\n")
```

Forward it into the calculator call, which currently drops `mace_head` too — keep that as-is:

```python
            setup_calculator(bench_atoms, model, uma_task,
                             sevennet_task=sevennet_task)
```

Check the existing import block at the top of `benchmark.py` actually imports all four availability flags; add any that are missing.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_benchmark.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/cli/commands/benchmark.py tests/test_benchmark.py
git commit -m "feat(sevennet): benchmark includes SevenNet only with an explicit task"
```

---

### Task 9: Documentation

**Files:**
- Modify: `docs/install/sevenn.md` (rewrite), `docs/install/README.md`, `docs/PYTHON_API.md`, `README.md`, `AGENTS.md`, `CONTEXT.md`, `CHANGELOG.md`

**Interfaces:**
- Consumes: the verified table from Task 1 and the cluster numbers from Task 10 (write the recipe last, or leave the measured numbers for a follow-up commit inside Task 10).

- [ ] **Step 1: Rewrite `docs/install/sevenn.md`**

Replace the whole file. It must carry: the real `_Last verified: 2026-09-01_` line; the conda-prefix recipe actually used on cos-cluster (python 3.11, `torch==2.8.0` from the cu128 index, `torch_geometric`, `sevenn`); the full model/task table from Task 1 with one anchor per tag matching `_TAG_TO_RECIPE` (`#7net-omni`, `#7net-omni-i8`, `#7net-omni-i12`, `#7net-mf-ompa`, `#7net-mf-0`, `#7net-omat`, `#7net-l3i5`, `#7net-0`); the checkpoint cache location from Task 1 Step 4 and how to redirect it off a quota'd home; and the measured `mpa` vs `oc20` energy difference from Task 10 as the evidence behind a short "tasks have independent energy zeros (CANON C3)" warning.

- [ ] **Step 2: Update the other docs**

- `docs/install/README.md`: the table row currently lists only `7net-mf-ompa`; list the family and mark `7net-omni` as recommended.
- `README.md`: the `--mlip` bullet at :136, the auto-detection sentence at :115, the SevenNet install block at :101, the mermaid line at :47, and a new `--sevennet-task` bullet beside the `--uma-task` / `--mace-head` ones.
- `AGENTS.md`: amend the "`--mlip auto` falls back to it" paragraph at :41 to state that a SevenNet-only env resolves to `7net-omni` and then stops for a missing task — the "fresh env lands on a usable default" promise no longer holds for SevenNet, and that is deliberate.
- `CONTEXT.md`: the MLIP tag entry lists `7net-mf-ompa`; add the family.
- `docs/PYTHON_API.md`: the `mlip=` bullet at :40 and the `CustomNEB` sentence at :213.
- `CHANGELOG.md`: an `Added` entry for the family and the flag, a `Changed` entry for the schema bump and the auto-detect tag, and a `Breaking` note that `7net-mf-ompa` now requires an explicit task.

- [ ] **Step 3: Verify no stale reference survives**

Run: `grep -rn "7net-mf-ompa" --include='*.md' --include='*.py' . | grep -v '^./.git/'`
Expected: every remaining hit is either a task-table row, a deliberate historical CHANGELOG line, or a test asserting the tag — no hit still presenting it as the only SevenNet tag.

- [ ] **Step 4: Commit**

```bash
git add docs/ README.md AGENTS.md CONTEXT.md CHANGELOG.md
git commit -m "docs(sevennet): model family, --sevennet-task, verified install recipe"
```

---

### Task 10: Cluster validation on GPU

Nothing before this point proves SevenNet runs. This task is the only evidence that it does.

**Files:**
- Create: `/scratchb/juar/sevennet-dev/validate/` (cluster-side, not committed)
- Modify: `docs/install/sevenn.md`, `CHANGELOG.md` (record the measured numbers)

**Interfaces:**
- Consumes: everything above.
- Produces: measured energies and timings for the recipe and changelog.

- [ ] **Step 1: Push the branch and check it out on the cluster**

```bash
git push -u origin feat/sevennet-refresh
ssh cos-cluster 'test -d /scratchb/juar/sevennet-dev/mliprun || \
  git clone /app1-cos/mlip-platform/mlip-platform /scratchb/juar/sevennet-dev/mliprun
cd /scratchb/juar/sevennet-dev/mliprun && \
  git remote set-url origin https://github.com/manuelarcer/mliprun.git && \
  git fetch origin feat/sevennet-refresh && git checkout feat/sevennet-refresh && git log --oneline -1'
```

Expected: the branch head matches the local one. **Confirm the path is `/scratchb/juar/sevennet-dev/mliprun`, not `/app1-cos/...`.**

- [ ] **Step 2: Install it editable into the sevenn env only**

```bash
ssh cos-cluster '/scratchb/juar/EWaste2GreenCat/explicit_solvation/.venv/sevenn/bin/pip \
  install -e /scratchb/juar/sevennet-dev/mliprun'
ssh cos-cluster '/scratchb/juar/EWaste2GreenCat/explicit_solvation/.venv/sevenn/bin/mlip doctor'
```

Expected: `mlip doctor` exits 0 and reports `sevenn` installed with its version.

- [ ] **Step 3: Confirm the error path before the success path**

```bash
ssh cos-cluster 'cd /scratchb/juar/sevennet-dev/validate && \
  ../../EWaste2GreenCat/explicit_solvation/.venv/sevenn/bin/mlip optimize run \
  --structure POSCAR --mlip 7net-omni --device cuda; echo "exit=$?"'
```

Expected: non-zero exit, message naming the 13 valid tasks. A silent run here means the C1 guard does not work and the task is not done.

- [ ] **Step 4: Single-point energies, both models, and the two-task comparison**

```bash
ssh cos-cluster 'V=/scratchb/juar/EWaste2GreenCat/explicit_solvation/.venv/sevenn/bin/python
cd /scratchb/juar/sevennet-dev/validate && $V - <<PY
from ase.build import fcc111, add_adsorbate
from mliprun.cli.utils import build_calculator
slab = fcc111("Pt", size=(2,2,4), vacuum=10.0)
add_adsorbate(slab, "O", 1.5, "fcc")
for tag, task in (("7net-omni","mpa"), ("7net-omni","oc20"), ("7net-mf-ompa","mpa")):
    a = slab.copy()
    a.calc = build_calculator(tag, device="cuda", sevennet_task=task)
    print(tag, task, f"{a.get_potential_energy():.6f} eV")
PY'
```

Expected: three finite energies. The `mpa` vs `oc20` difference on the identical structure is the number that goes into the install recipe as the C3 evidence. Record it exactly.

- [ ] **Step 5: Run optimize, md, and neb end to end on GPU**

Launch each under `setsid nohup` and verify separately — a plain backgrounded ssh command hangs until the remote job exits:

```bash
ssh cos-cluster 'cd /scratchb/juar/sevennet-dev/validate && setsid nohup bash run_validate.sh > validate.log 2>&1 < /dev/null & echo launched'
ssh cos-cluster 'pgrep -af run_validate.sh'
```

`run_validate.sh` runs, with the venv's absolute `mlip` binary: `optimize run --mlip 7net-omni --sevennet-task mpa --device cuda --fmax 0.05`; `md run --mlip 7net-omni --sevennet-task mpa --device cuda --ensemble nvt --temperature 300 --timestep 0.5 --steps 50`; and `neb run --mlip 7net-omni --sevennet-task mpa --device cuda` on a short two-endpoint band.

Expected artifacts: `opt_final.vasp`, `md.traj` + `md_energy.csv`, and the NEB outputs — and an `mliprun_run.json` in each directory.

- [ ] **Step 6: Verify the run record actually carries the task**

```bash
ssh cos-cluster 'cd /scratchb/juar/sevennet-dev/validate && \
  python3 -c "import json,glob;
[print(f, json.load(open(f))[\"provenance\"][\"sevennet_task\"], json.load(open(f))[\"schema_version\"]) for f in glob.glob(\"**/mliprun_run.json\", recursive=True)]"'
```

Expected: `mpa` and `3` for every record. A `None` here means the CLI drops the value somewhere between the flag and `collect_provenance`.

- [ ] **Step 7: Sanity-check the relaxed geometry**

MLIPs relax into broken structures without raising. Compare `opt_final.vasp` against the input — bond lengths, no atoms merged, adsorbate still on the surface — before calling the run good.

- [ ] **Step 8: Record the numbers and commit**

Put the measured energies, the `mpa`/`oc20` delta, and the wall times into `docs/install/sevenn.md` and `CHANGELOG.md`.

```bash
git add docs/install/sevenn.md CHANGELOG.md
git commit -m "docs(sevennet): record cos-cluster validation numbers"
git push
```

- [ ] **Step 9: Open the draft PR**

```bash
gh pr create --draft --base main --head feat/sevennet-refresh \
  --title "feat(sevennet): SevenNet 0.13.0 model family with explicit task selection" \
  --body "<summary, the CANON C1/C3 rationale, the breaking change to 7net-mf-ompa, and the cluster validation numbers>"
```

---

## Self-Review

**Spec coverage.** Table → Tasks 1-2. `--sevennet-task` semantics → Tasks 2, 6, 7, 8. No-default policy → Task 2 Step 4, reinforced in Task 7 Step 3 and Task 10 Step 3. Auto-detect consequence → Task 3. `benchmark` handling → Task 8. No compatibility shim → covered by Task 2 removing the special case; the breaking change is documented in Task 9 and the PR body in Task 10. Schema 3 → Task 5. Call-path changes → Tasks 4, 6, 7. Testing section → the test steps of Tasks 2-8, cluster validation → Task 10. Work order → Tasks 1, 2-9, 10. No gaps.

**Placeholder scan.** One deliberate ellipsis remains, in Task 7 Step 1's second test, with an instruction to mirror the existing `tests/test_neb_restart.py` fixture rather than invent one; the fixture is real and in the repo. Task 9's doc edits list exact files and line numbers rather than code blocks, which is right for prose changes. Task 10 Step 5 describes `run_validate.sh` by its three exact commands rather than quoting the wrapper.

**Type consistency.** `sevennet_task: Optional[str]` everywhere; `_SEVENNET_MODELS: dict[str, tuple[str, ...]]`; `_validate_sevennet_task(mlip, sevennet_task) -> None`; `validate_mlip(mlip, sevennet_task=None)`; `resolve_mlip(mlip, sevennet_task=None)`; `build_calculator(..., sevennet_task=None)`; `setup_calculator(atoms, ..., sevennet_task=None)`; `collect_provenance(..., sevennet_task=None)`; `_available_models(sevennet_task=None)`; params-file key `"SevenNet task"` → dict key `"sevennet_task"`. Consistent across tasks.

**One correction folded in:** Task 2 Step 4 originally contained a double-negative in the warning message ("so neither ... cannot be validated"). The step now states the corrected wording and tells the implementer to keep the test assertion and the message in agreement on `"can be validated"`.
