# 0001 — Install guidance via per-MLIP env recipes, not auto-install

**Status**: accepted

Supported MLIP packages (`mace-torch`, `fairchem-core`, `sevenn`, `chgnet`) pin mutually incompatible versions of `torch`, `torch_geometric`, and `e3nn`. Installing a second MLIP into an env that already contains a different one typically downgrades or upgrades `torch` and silently breaks the first MLIP at import or runtime. The platform therefore documents one **MLIP env** per **MLIP tag** via per-file install recipes under `docs/install/`, and the "MLIP not available" error path prints a URL + local path to the relevant recipe rather than running `pip install` for the user.

## Considered options

- **Auto `pip install` in the current env** — rejected. Silently corrupts working envs when users mix MLIPs.
- **Collision-detection + per-MLIP constraints file** — rejected for v1. The collision check is real work whose main benefit is preventing a footgun that documented recipes already prevent; the actual `pip install` line is the same line a recipe prints.
- **Per-MLIP managed envs with a calculator-in-subprocess bridge** — rejected. Architecturally clean but a multi-month rewrite; current `setup_calculator()` does in-process ASE imports. Filed mentally as a separate future project, not part of this feature.

## Consequences

- Adding a new MLIP requires adding one file under `docs/install/<package>.md` (and a `_TAG_TO_RECIPE` entry in `cli/utils.py`).
- Users who want to switch MLIPs switch envs (conda activate / venv source). The platform does not abstract this away.
- The existing `"Install with: pip install <pkg>"` hints in `validate_mlip()` and `detect_mlip()` are misleading under this policy and must be replaced with recipe links.
- Weight fetching is unchanged: `mace_mp` and `_ensure_mace_foundation_checkpoint` handle MACE; `fairchem` handles UMA via HuggingFace (auth steps documented in the UMA recipe).

## Verification policy

Each recipe carries a `_Last verified: YYYY-MM-DD (torch X.Y.Z, Python A.B)_` line. Maintainer re-runs the recipe before tagging a release and bumps the stamp. No automated CI for env builds in v1 — the single-author maintenance cost outweighs the benefit at this scale.

## Deferred

- Add per-MLIP CI smoke (build each recipe's env on PR-touch + weekly, import the calculator, compute one energy on a 2-atom cell). Blocker for v1: UMA's HF gated-license flow needs a project-owned HF account and a `HUGGINGFACE_TOKEN` repo secret. Revisit when the project has more than one active contributor or starts cutting releases on a cadence.
- Per-MLIP managed envs with a subprocess calculator bridge (see "Considered options"). Revisit only if users start asking for "compare MACE and UMA in one run" workflows.

## 2026-09-07: the revisit trigger fired

The last "Deferred" item's trigger fired: users started asking for exactly
the "compare MACE and UMA in one run" workflow this ADR named. The response
was the calculator-in-subprocess bridge described in "Considered options"
above, a committee of MLIPs, each in its own env, driven together through
`optimize run --committee` (design:
[`2026-09-04-committee-uncertainty-design.md`](../superpowers/specs/2026-09-04-committee-uncertainty-design.md),
plan: [`2026-09-04-committee-uncertainty.md`](../superpowers/plans/2026-09-04-committee-uncertainty.md)).
It shipped for `optimize` at a fraction of the "multi-month rewrite" estimate
that got the option rejected here, because managed environments stayed out
of scope: each committee member is still a plain per-MLIP env the user
builds by hand from the existing install recipes, addressed over a stdin/
stdout protocol by a worker subprocess running that env's own interpreter.
`optimize` itself needed no restructuring: everything downstream only ever
touches `atoms.calc`.

**This ADR's decision is unchanged.** The bridge does not escape the
one-MLIP-per-env rule; it relies on it. Every member still needs its own
env built from a single install recipe, and mixing MLIP packages into one
env remains unsupported and still breaks at least one of them. What changed
is that a user who wants several MLIPs' opinions on one structure no longer
has to run them one env at a time by hand and reconcile the outputs
themselves.

Not extended to `md` or `neb`/`autoneb`, and no batch form (`optimize batch
--committee`) exists yet. Out of scope for the first cut, not ruled out.
