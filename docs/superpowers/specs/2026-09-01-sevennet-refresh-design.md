# SevenNet refresh: Omni model family, explicit task selection, provenance

**Date:** 2026-09-01
**Amends:** `2026-07-29-run-record-head-provenance-design.md` (schema 2 → 3)
**Status:** approved design, not yet implemented

## Problem

SevenNet has been nominally supported since the earliest releases, but the
support is a stub that has never executed.

`build_calculator` (`src/mliprun/cli/utils.py`) carries a single branch:

```python
elif mlip == "7net-mf-ompa":
    SevenNetCalculator = _load_sevenn_calculator()
    return SevenNetCalculator("7net-mf-ompa", modal="mpa", device=device)
```

Four separate defects follow from those three lines.

**1. Never validated.** `docs/install/sevenn.md` reads
`_Last verified: pending first-run validation_`. No `sevenn` environment exists
on cos-cluster (only `.venv/{uma,mace,chgnet}`), so every SevenNet test in the
suite — `test_md_sevenn.py`, `test_neb_sevenn.py`, `test_milp_single_point.py` —
has always auto-skipped. The code path has never run once.

**2. The head is hardcoded.** `modal="mpa"` is baked into the source. SevenNet's
`modal` is the same concept as UMA's task and MACE's head: an independent
fine-tune with its own energy zero. CANON C1 requires the head to be an explicit
decision, never guessed; CANON C3 forbids mixing heads inside one energy
formula. Neither is enforceable for SevenNet today, because the user cannot see
or set the value.

**3. No provenance.** `collect_provenance` records `uma_task` and `mace_head`
only (`run_record.py:220`). A SevenNet run record cannot state which task
produced its numbers, so under C1/C3 it cannot back any energy entering a
formula.

**4. The pinned model generation is superseded.** SevenNet is at 0.13.0. Its
recommended model is now **SevenNet-Omni** (`7net-omni`), a 13-task
multi-fidelity model whose task list includes `oc20` (RPBE) and `oc22` — surface
and catalysis heads. `7net-mf-ompa` offers only `mpa` and `omat24`, neither of
which is a surface head. The one SevenNet model the platform exposes is the one
least suited to this group's work.

## Change

### 1. Model and task table

A module-level table in `cli/utils.py` maps each documented tag to its task
list. Multi-task models carry a non-empty tuple; single-task models carry an
empty one.

| Tag | Type | Tasks |
|---|---|---|
| `7net-omni` | multi-task | `omat24`, `mpa`, `omol25_low`, `omol25_high`, `matpes_pbe`, `matpes_r2scan`, `mp_r2scan`, `oc20`, `oc22`, `spice`, `qcml`, `odac23`, `pet_mad` |
| `7net-omni-i8` | multi-task | same 13 |
| `7net-omni-i12` | multi-task | same 13 |
| `7net-mf-ompa` | multi-task | `omat24`, `mpa` |
| `7net-mf-0` | multi-task | `PBE`, `R2SCAN` (uppercase — the only tag whose task names are not lowercase) |
| `7net-omat` | single-task | — |
| `7net-l3i5` | single-task | — |
| `7net-0` | single-task | — |
| `7net-0_22may2024` | single-task | — |

Verified 2026-09-01 against `sevenn` 0.13.0 on cos-cluster:
`sevenn.util.get_available_pretrained_models()` for the tag list, and
`sevenn cp <tag>` for each checkpoint's `Modality` line. The documentation
disagreed with the package in three ways, all corrected above:

- **`7net-nano-4.5 / -5.0 / -5.5 / -6.0` do not exist in this release.** The
  documentation lists SevenNet-Nano with those four keywords; the installed
  registry has none of them. They are dropped from the table. If a later
  `sevenn` release adds them, the unknown-tag passthrough already carries them.
- **Two tags the documentation's overview table omits are in the registry:**
  `7net-0_22may2024` (an earlier SevenNet-0 snapshot, single-task) and
  `7net-mf-0` (the first multi-fidelity model, multi-task).
- **`7net-mf-0`'s tasks are uppercase**, `PBE` and `R2SCAN`, unlike every other
  model's lowercase names. Task comparison is therefore exact and
  case-sensitive; lowercasing user input would break this tag, and accepting
  `pbe` for it would be inventing a name the checkpoint does not have.

A wrong string in a head table silently changes the level of theory, so no
value entered this table on documentation authority alone.

Any other `7net-*` tag is forwarded to `SevenNetCalculator` unchanged, with a
warning that neither the tag nor its task can be validated. New SevenNet
checkpoints therefore work without a code change, while the tags actually in use
stay checked.

### 2. `--sevennet-task`

A new CLI option on `optimize run`, `optimize batch`, `md run`, `neb run`,
`autoneb run`, and `benchmark run`. It maps to `SevenNetCalculator(modal=...)`.
The CLI calls it *task*, not *modal*, because SevenNet's own documentation calls
it the task and because it matches the existing `--uma-task`.

Validation rules, enforced in `validate_mlip` before any model is loaded:

| Situation | Result |
|---|---|
| Multi-task tag, task unset | **Error**, listing that model's valid tasks |
| Multi-task tag, task not in list | **Error**, listing that model's valid tasks |
| Single-task tag, task passed | **Error**: model has no selectable task |
| Unknown `7net-*` tag | Forwarded, warning printed |

There is deliberately **no default**. This departs from `--uma-task`
(default `omat`) and `--mace-head` (default `omat_pbe`), which CANON C1 names as
the standing hazard: they execute silently when unset. A new flag has no
installed base to protect, so it is written the way C1 asks for. Aligning the
older two flags is a separate concern and out of scope here.

**Consequence, accepted deliberately.** In a SevenNet-only environment,
`--mlip auto` resolves to a multi-task tag and then exits with the missing-task
error, so `optimize run --structure POSCAR` with no other flags stops working
there. This is preferred over silently applying `mpa`: CANON C1 bans `--mlip
auto` in production regardless, and a loud stop is recoverable where a wrong
energy zero is not. `AGENTS.md` is amended to state this rather than continuing
to promise that a fresh environment always lands on a runnable default.

**`benchmark run`.** Its auto-detected model list includes a SevenNet
multi-task tag only when `--sevennet-task` was given; otherwise SevenNet is
skipped with a printed note, so the benchmark still runs across the other
installed MLIPs. A tag named explicitly through `--models` follows the normal
rules and errors without a task. The pre-existing hole where `benchmark run`
never passes `mace_head` at all (`benchmark.py:69`) is left alone — one concern
per PR.

**No compatibility shim for `7net-mf-ompa`.** It currently runs with an implicit
`mpa`; afterwards it requires `--sevennet-task mpa`. The break is free because
the path has never executed: no run record, result, or script depends on it.

### 3. Run record: schema 3

`SCHEMA_VERSION` 2 → 3. One new provenance field, gated on the tag exactly as
its two siblings are:

```python
"sevennet_task": sevennet_task if tag.startswith("7net") else None,
```

`sevennet_task` joins `_PROVENANCE_DIFF_FIELDS`, so a task switch between stages
of one pipeline is reported as a provenance change — the C3 guard. `7net` is
already mapped to the `sevenn` distribution in `_MLIP_PACKAGES`, so package
resolution is unchanged.

Three parallel fields rather than one normalized `mlip_head`, for the reason the
2026-07-29 amendment already gives: the fields preserve the upstream vocabulary,
and normalizing them is a refactor across every command plus the frozen goldens.

### 4. Call-path changes

`sevennet_task` threads through exactly where `uma_task` and `mace_head` already
do:

- `cli/utils.py` — the tag/task table; `validate_mlip` extended; `MLIP_HELP`
  updated and `SEVENNET_TASK_HELP` added; `build_calculator` and
  `setup_calculator` gain the parameter; `_TAG_TO_RECIPE` entries for the new
  tags; `detect_mlip` returns `7net-omni` instead of `7net-mf-ompa`.
- `cli/commands/{optimize,md,neb,autoneb,benchmark}.py` — the typer option, the
  echo line, the params-file line.
- `core/{optimize,md}.py` — the keyword argument down to `collect_provenance`.
- `core/neb.py` — `CustomNEB` gains `self.sevennet_task`, a `SevenNetCalculator`
  branch in its own `setup_calculator` (it wires calculators itself, separately
  from `cli/utils.py`), and an entry in the `neb_params.txt` parse map at
  `neb.py:161` so `neb run --restart` round-trips the task.
- `core/run_record.py` — as in section 3.

Unifying the three head flags into one `--head` is explicitly **not** part of
this work.

## Testing

Unit tests run with no MLIP installed and assert values or invariants, per the
repository rule. No golden file is added or regenerated.

- Table integrity: every tag resolves; every multi-task model has a non-empty
  task tuple; every single-task model has an empty one.
- `validate_mlip`: raises on missing task, on invalid task, and on a task given
  to a single-task model; passes for each valid (tag, task) pair.
- `detect_mlip` with only `SEVENN_AVAILABLE` patched true resolves to
  `7net-omni`.
- Run record: a `7net-omni` record carries `sevennet_task`; a `mace` record
  carries `sevennet_task: None`; `schema_version` is 3; a schema-2 record still
  loads, and the absent-key versus present-and-null distinction at
  `run_record.py:294` still holds with the added field.
- NEB restart round-trip: write `neb_params.txt` carrying a task, reload, assert
  the task survives — the same shape as the DyNEB restart test.
- Unknown `7net-*` passthrough emits the warning and does not raise.

**Cluster validation** on cos-cluster in a real `sevenn` environment, GPU, with
numbers recorded in `CHANGELOG.md` and the install recipe:

- single-point energy for `7net-omni` (task `mpa`) and `7net-mf-ompa`
  (task `mpa`);
- `optimize run`, `md run`, and `neb run` end to end;
- the same structure evaluated under `7net-omni` task `mpa` versus task `oc20`.
  The energy difference is the concrete demonstration that SevenNet tasks carry
  independent zeros, and it is recorded in the install recipe as the evidence
  behind the C3 warning.

## Work order

Phase 0 runs first because its output is an input to the code.

0. **Cluster, before any code.** Create the `sevenn` conda-prefix env matching
   the existing convention (python 3.11, torch 2.8.0+cu128 — the stack already
   proven on this node by the `uma` env). Confirm the `SevenNetCalculator`
   signature, capture the exact task strings, and locate the checkpoint cache
   so it can be redirected off `/home` if needed.
1. **Implement locally**, TDD, on a branch off `main`.
2. **Validate on the cluster**: push the branch, check it out in a private clone
   under `/scratchb/juar/sevennet-dev/` — never in the shared editable install
   at `/app1-cos/mlip-platform/mlip-platform`, which the `uma`, `mace`, and
   `chgnet` environments all import from — then run the GPU validation.
3. **Docs and changelog**, then a draft PR.

## Files touched

`src/mliprun/cli/utils.py`, `src/mliprun/cli/commands/{optimize,md,neb,autoneb,benchmark}.py`,
`src/mliprun/core/{optimize,md,neb,run_record}.py`, `tests/` (new and extended),
`docs/install/sevenn.md`, `docs/install/README.md`, `docs/PYTHON_API.md`,
`README.md`, `AGENTS.md`, `CONTEXT.md`, `CHANGELOG.md`.
