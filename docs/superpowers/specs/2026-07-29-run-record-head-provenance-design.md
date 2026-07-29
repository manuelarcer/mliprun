# Run record: carry the MLIP head/task (schema v2)

**Date:** 2026-07-29
**Amends:** `2026-07-21-unified-run-record-design.md`
**Driven by:** `~/github/work/prov/docs/superpowers/specs/2026-07-29-md-provenance-capture-design.md` (Part A)
**Status:** approved design, not yet implemented

## Problem

`collect_provenance` takes only the model tag. A real MD record on disk reads:

```json
"mlip_model": "uma-s-1p2",
"device_requested": "cuda",
```

with no mention of the UMA task the run actually used (`oc25`, recoverable only
from `md_params.txt`). The same gap applies to `mace_head`, and to optimize and
NEB records as well as MD.

`mliprun_run.json` presents itself as the canonical run record, but under CANON
C1 the head/task is its own explicit decision, and under C3 heads must never be
mixed within a γ, adsorption-energy, or ΔE formula. A record naming only
`uma-s-1p2` cannot answer "which head produced this number", so it cannot serve
as provenance for any energy that enters a formula.

## Change

Two keyword-only parameters on `collect_provenance`, both defaulting to `None`,
emitted as keys that are **always present and null when not applicable**:

```python
def collect_provenance(*, mlip_model, device_requested, device_resolved,
                       uma_task=None, mace_head=None) -> dict:
    ...
    "uma_task": uma_task,     # non-null only for uma-* models
    "mace_head": mace_head,   # non-null only for mace-mh-* models
```

Two fields rather than one normalized `mlip_head`: it preserves the upstream
vocabulary (UMA calls it a task, MACE calls it a head) and matches the keys the
`prov` ledger's `level_of_theory` already uses, so neither side needs a mapping
layer. Always-present-with-null follows the existing house style — `mlip_package`
is always present with null fields rather than omitted.

Both parameters keep `None` defaults so existing library callers and test doubles
continue to work unchanged.

## Schema version

`SCHEMA_VERSION` 1 → 2.

The bump is load-bearing. Without it a reader cannot distinguish "no `uma_task`
key because the record predates this change" from "`uma_task` is null because
this was a MACE run". `prov` uses exactly that distinction to decide whether to
fall back to parsing `md_params.txt` for the head.

## `_PROVENANCE_DIFF_FIELDS`

Add `"uma_task"` and `"mace_head"`.

That tuple governs which provenance changes an appended stage records in
`stage_provenance`. A resume that switched head is precisely what CANON C3
forbids, so it is the last thing that should go unrecorded.

## Call sites to thread

- `core/optimize.py::run_optimization` — add `uma_task=None, mace_head=None`.
- `core/md.py::run_md` — same.
- `core/neb.py` — `CustomNEB` already stores `self.uma_task` / `self.mace_head`;
  pass them at its `collect_provenance` call site.
- Any other core call site of `collect_provenance`, autoneb path included.
- CLI commands `optimize.py`, `md.py`, `neb.py`, `autoneb.py` pass the values
  they already parse.

## Golden safety

The characterization goldens assert numerics read from `opt_convergence.csv`,
`md_energy.csv`, and final structure files. None reads `mliprun_run.json`, so
adding provenance fields cannot move a golden. **Do not regenerate goldens.**

## Tests (written first)

- `collect_provenance` returns both keys, null by default.
- UMA model + task recorded; MACE model + head recorded.
- `SCHEMA_VERSION == 2`, and a written record carries it.
- End-to-end: `run_md(..., uma_task="oc25")` writes
  `provenance.uma_task == "oc25"`.
- A resumed stage that changes head records it in `stage_provenance`; a
  same-head resume omits `stage_provenance` entirely.

## Out of scope

Backfilling the head into records already on disk. The 7 existing MD records and
every optimize record stay at schema 1; `prov` recovers their head from
`md_params.txt` / `opt_params.txt` instead.
