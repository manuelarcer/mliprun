# Committee evaluation: consensus forces and per-configuration uncertainty

**Date:** 2026-09-04
**Amends:** `2026-09-01-sevennet-refresh-design.md` (schema 3 → 4)
**Supersedes the phasing in:** `docs/proposals/ensemble-consensus-uncertainty.md`
**Status:** approved design, not yet implemented

## Problem

A single MLIP returns a point estimate with no error bar. For adsorption
energies, transition-state barriers, and picking which of several candidate
configurations is physically meaningful, the question that actually matters is
*how much can I trust this number*. Nothing in `mliprun` answers it today.

The established answer is a **committee** (query-by-committee): run the
configuration through N potentials, drive the trajectory with their mean force,
and read their disagreement as the uncertainty. Where the models agree the
prediction is well constrained; where they disagree the configuration sits in a
region the potentials extrapolate into, and the result deserves a DFT check.

The obstacle recorded in [ADR 0001](../../adr/0001-per-mlip-envs.md) is that
MLIP packages cannot share a Python environment: `mace-torch`, `fairchem-core`,
`sevenn`, and `chgnet` pin mutually incompatible `torch` / `torch_geometric` /
`e3nn` versions. ADR 0001 priced the escape — a calculator-in-subprocess bridge
— as "a multi-month rewrite" and deferred it, naming its own revisit trigger:
*"Revisit only if users start asking for 'compare MACE and UMA in one run'
workflows."*

That ask has now arrived, and the multi-month estimate was for a larger thing
than what is needed here (see "Feasibility, measured").

## Feasibility, measured

A throwaway probe ran on cos-cluster (`cosrnode01g`, 3× L40S) on 2026-09-04,
before any of this design was written. It held **four members across four
packages in one committee**: `uma-s-1p2@oc20`, `7net-omni@oc20`,
`mace-mp-0-medium`, and `chgnet` — spanning three torch builds (2.5.1+cu121,
2.8.0, 2.8.0+cu128) and two ASE versions (3.28.0, 3.29.0). System: CO on
Pt(111), 2×2, 3 layers, 14 atoms.

Findings that this design is built on:

**1. It works, and the driver needs no MLIP.** A stock ASE `BFGS` relaxed the
system through a committee calculator with no change to the optimizer. The
driver process imported no `torch`, `fairchem`, `mace`, `sevenn`, or `chgnet`
(verified against `sys.modules`).

**2. Cost is `max(member)`, not `sum(member)`.** With a concurrent fan-out the
committee step took 211.7 ms against a slowest member of 206.7 ms.
Inter-process overhead was **5.0 ms per step (2.4%)** on a 992-byte payload,
and that overhead is roughly constant in system size.

**3. Members differ enormously in per-call cost, and UMA's cost is fixed
overhead.** `uma-s-1p2` took 206.7 ms per call where `7net-omni`, `mace` and
`chgnet` took 1.12–1.34 ms. A follow-up measurement inside the `uma` env with
no bridge at all shows why:

| system | fresh `Atoms` per call | reused `Atoms`, positions moved |
|---|---|---|
| 14 atoms | 178.2 ms | 179.4 ms |
| 38 atoms | 171.4 ms | 175.0 ms |
| 82 atoms | 171.8 ms | 173.3 ms |

The cost is **flat in system size** — 6× the atoms for no extra time — and
identical whether the `Atoms` object is rebuilt or reused. It is therefore
fixed per-call overhead inside `fairchem`'s calculator, not model compute
(GPU utilization sat at 7%), and it is not caused by the bridge. Two
consequences: an existing single-model `mlip optimize` run with UMA already
pays this per step, so the committee introduces nothing new; and because the
overhead is constant, its relative weight falls as systems grow.

The same call measured 206.7 ms inside the four-member committee and 178 ms
alone, which is the cost of concurrent fan-out on a contended node: four
workers competing for CPU make each one's CPU-side overhead worse. Budget for
the committee number, not the standalone one.

*Caveat:* the node's 192 cores were saturated by another user's job throughout.
Contention plausibly inflates the absolute number, since the overhead is
CPU-side, but it cannot explain the flatness, and the same contention left
SevenNet, MACE and CHGNet at ~1.2 ms — so the overhead is specific to the
fairchem calculator path.

**4. Energy alignment works, and is unnecessary for differences.** At one fixed
geometry the four raw energies were −73.68, −73.44, −80.63, −82.86 eV, a
4.81 eV spread. Removing each model's own constant offset collapsed the spread
to **0.09 eV** across a CO height scan. A quantity like an adsorption energy
needs no alignment at all, because the offset cancels in the difference.

**5. Mixing levels of theory dominates the spread.** Taking
E(h=2.0) − E(h=3.5) per member:

| member | level of theory | ΔE |
|---|---|---|
| `uma-s-1p2@oc20` | RPBE/OC20 | −1.136 eV |
| `7net-omni@oc20` | RPBE/OC20 | −1.137 eV |
| `mace-mp-0-medium` | PBE/MPtrj | −1.508 eV |
| `chgnet` | PBE+U/MPtrj | −1.490 eV |

Within a level: **0.001** and **0.019 eV**. Across levels: **0.363 eV**. A
mixed-level committee therefore reports roughly twenty times more
"uncertainty", essentially all of it the RPBE-vs-PBE difference rather than
model error. Canon C3 is load-bearing here.

Two limits on that result: it is one system and one scan, not a general claim;
and the within-level agreement is measured on configurations close to both
models' training data, which is exactly where committees look over-confident.
**The spread is a lower bound on the true error.**

**6. Force disagreement behaves as an extrapolation detector.** σ_F(max) was
3.476 eV/Å at a strained geometry (C 1.50 Å above the surface) and
0.25–0.62 eV/Å across the physical range.

**7. Members disagree about what input is valid.** `fairchem`'s UMA calculator
raises `MixedPBCError` on a `pbc=(True, True, False)` slab that MACE, SevenNet
and CHGNet all accept.

**8. A killed member aborts loudly.** Killing one worker produced
`committee incomplete: chgnet: [Errno 32] Broken pipe` and stopped the run,
rather than silently continuing with three members.

## Scope decisions

Settled in the 2026-09-04 brainstorming pass. Alternatives are recorded so a
later reader knows they were considered.

1. **Committee-driven run, not post-hoc evaluation.** One trajectory driven by
   committee-mean forces, disagreement recorded at every step.
   *Rejected:* evaluating already-relaxed structures with N models. Cheaper and
   trivially cross-package, but blind to disagreement along the path.

2. **Cross-package and cross-head members are permitted.** The bridge makes
   genuinely independent committees possible, which is the only way the spread
   becomes a defensible error bar.

3. **Mixed levels of theory warn; they do not refuse.** Juan's call, 2026-09-04.
   *Consequence accepted:* the default path can emit a number that looks like an
   error bar and is a functional comparison. Mitigation is mandatory — the
   warning is stamped into the run record **and** carried as a per-row CSV
   column, so downstream analysis can filter on it without anyone having read
   the terminal.
   *Rejected:* hard refusal with an override flag; and two separate commands
   (uncertainty mode vs comparison mode).

4. **Committee declared by a YAML file**, `--committee committee.yaml`.
   Reproducible, version-controllable, and it hashes into the run record.
   *Rejected:* registered envs plus repeated `--member` flags (the committee is
   then not an artifact); a `discover` helper that generates the file (useful,
   but scope for later).

5. **First deliverable is `optimize` only.** MD and NEB reuse the same layer
   afterwards. MD will additionally need a stride option, since evaluating the
   committee every step makes long trajectories impractical.

6. **Mean force drives the trajectory.** *Rejected for v1:* drive with one model
   and observe the rest. Worth a flag later; the trajectory is then that one
   model's, which answers a different question.

## Design

### Architecture

The committee is an **ASE calculator**. Everything downstream — `optimize`
today, MD and NEB later — only ever touches `atoms.calc`, so no engine needs
restructuring. This was validated in the probe: stock `BFGS` ran through it
unchanged.

*Rejected:* a parallel `run_committee_optimization()` engine, which would
duplicate the optimizer plumbing in `core/optimize.py` and drift from it; and
shelling out to N independent `mlip optimize` runs, which cannot produce
committee-driven forces at all.

New package `src/mliprun/core/committee/`, five modules:

| module | responsibility |
|---|---|
| `protocol.py` | request/response encoding, shared verbatim by driver and worker. **Stdlib only** — it is imported inside every MLIP env and must drag nothing in. |
| `worker.py` | subprocess entry point, `python -m mliprun.core.committee.worker`, run by each MLIP env's own interpreter. Builds its calculator through the existing `build_calculator`. |
| `remote.py` | `RemoteMember`: owns one `Popen`, load handshake, request/response, per-request timeout, teardown. |
| `calculator.py` | `CommitteeCalculator(ase.calculators.calculator.Calculator)`: concurrent fan-out, mean energy and forces, disagreement reduction, trace. |
| `config.py` | parse and validate `committee.yaml`, resolve each member's level of theory, raise the mixed-level warning. |

### Wire protocol

Line-delimited JSON, one object per line, in both directions.

```
-> {"cmd": "load", "mlip": "uma-s-1p2", "uma_task": "oc20", "device": "cuda"}
<- {"ok": true, "t_load": 22.8, "versions": {...}}

-> {"cmd": "calc", "numbers": [...], "positions": [[...]],
    "cell": [[...]], "pbc": [true, true, true]}
<- {"ok": true, "energy": -73.679, "forces": [[...]], "t_calc": 0.207}

-> {"cmd": "quit"}
```

Three constraints, each learned from the probe:

- **Plain arrays only; never a pickled `Atoms`.** One committee held ASE 3.28.0
  and 3.29.0 simultaneously, and pickles do not safely cross that.
- **The worker must protect stdout.** MACE, SevenNet and CHGNet all print
  banners to stdout on import and first inference, which would corrupt the
  stream. The worker takes a private `dup` of file descriptor 1 for protocol
  traffic and points fd 1 at stderr, so library output lands in that member's
  worker log.
- **Constraints stay driver-side.** Workers receive raw geometries and return
  raw forces; ASE applies `FixAtoms` in the driver exactly as today. Nothing
  about the frozen-layer scheme (canon S8) changes or becomes invisible.

### Committee file

```yaml
# committee.yaml
members:
  - env:  /scratchb/juar/.../.venv/uma
    mlip: uma-s-1p2
    uma_task: oc20
    gpu: 0

  - env:  /scratchb/juar/.../.venv/sevenn
    mlip: 7net-omni
    sevennet_task: oc20
    gpu: 1

  - env:  /scratchb/juar/.../.venv/mace
    mlip: mace-mh-1
    mace_head: oc20_usemppbe
    gpu: 2
```

```bash
mlip optimize run --structure POSCAR --committee committee.yaml --fmax 0.05
```

`--committee` and `--mlip` are mutually exclusive; passing both is an error
rather than a silent precedence rule.

`gpu` sets `CUDA_VISIBLE_DEVICES` for that member's process. Omitted, the
member inherits the driver's environment.

### Level of theory

A table maps `(tag, task-or-head)` to a level-of-theory label — for example
`uma-*@oc20`, `7net-*@oc20` and `mace-mh-*@oc20_usemppbe` all resolve to
`RPBE/OC20`; `mace` (MP-0) and `chgnet` resolve to MPtrj PBE variants. Members
resolving to more than one label set `mixed_theory`.

An **unrecognised** tag or task resolves to `unknown` and counts as *possibly*
mixed: it warns rather than passing silently. This matters because
`cli/utils.py` deliberately forwards unknown `uma-*` and `7net-*` tags through
to their packages unchanged.

### Uncertainty metric

Per-atom force disagreement:

    σ_i = ‖ std_across_members( F_i ) ‖

the vector norm of the per-component standard deviation across members
(`ddof=1`). Reduced to `sigma_max` (max over atoms), `sigma_mean`, and the
index of the worst atom.

The committee is a **proper potential**: mean force is exactly the negative
gradient of mean energy, so the effective PES is well defined and line-search
optimizers such as `bfgsls` behave correctly through it. Per-model constant
offsets shift the mean energy by a constant without changing its shape.

### Outputs

`opt_committee.csv`, alongside the existing `opt_convergence.csv`, one row per
optimizer step:

| column | meaning |
|---|---|
| `step` | optimizer step |
| `energy_mean_eV` | mean of members' raw energies — the optimizer's objective. Absolute value is meaningless across packages; differences are not. |
| `energy_spread_aligned_eV` | spread after subtracting each member's own step-0 energy. Composition is fixed during a relaxation, so the offset cancels exactly and this is a real energy uncertainty. |
| `E_<member>_eV` | each member's raw energy, one column per member |
| `fmax_eV_per_A` | max force of the mean-force field |
| `sigma_max_eV_per_A` | per-atom force disagreement, max over atoms |
| `sigma_mean_eV_per_A` | the same, averaged over atoms |
| `worst_atom` | index of the atom with the largest σ |
| `mixed_theory` | the warn-don't-refuse flag, per row |

`opt_committee_peratom.csv`, **final geometry only**: per-atom σ with the atom
index and chemical symbol. This says *which* atoms the models disagree about —
usually the adsorbate or the reacting bond — and is the most diagnostically
useful output. Final geometry only, because a per-atom field at every step
would be a large file for little gain; `worst_atom` already traces where
disagreement lives during the run.

Under `--plot` (opt-in, as elsewhere in the package), the σ trace is added to
the convergence figure. Plotting code follows the `plot-style` conventions.

### Flagging rule

> **Superseded 2026-09-08.** The `fmax` default described below was removed;
> `--uncertainty-threshold` is now opt-in with no default, and `sigma_max`
> was renamed and narrowed to the free atoms. The decision recorded here was
> correct when written — the calibration it asked for is what overturned it.
> See `2026-09-08-committee-sigma-masking-design.md`.

Default: flag the configuration when `sigma_max` at the final geometry exceeds
the `fmax` the run converged to. Self-scaling and physically motivated — if the
models disagree about the forces by more than the convergence tolerance, the
located minimum sits inside the committee's own noise and the geometry is not
resolved. Overridable with `--uncertainty-threshold`.

**This default is uncalibrated.** The probe measured σ_F only across four
mixed-level members, so there is no same-level σ_F number yet, and it is not
known whether a genuine same-level committee sits comfortably below 0.05 eV/Å
or routinely above it. Calibrating it is a natural first use of the feature.
The threshold actually applied, and where it came from (`fmax` or explicit), is
recorded with the flag.

### Run record, schema 3 → 4

`collect_provenance` takes a single `mlip_model` today
(`core/run_record.py:166`); a committee needs a list, which forces the bump.

`provenance.committee` records, per member: tag, task or head, env path,
package and version, torch version, resolved level of theory, and GPU. Plus
the set of levels present, the `mixed_theory` flag, and a SHA-256 of the
`committee.yaml`.

`results.committee_uncertainty` records peak and final `sigma_max`, the step at
which the peak occurred, final `sigma_mean`, the worst atom, the final aligned
energy spread, the flag, the threshold, and the threshold's source.

Single-model runs keep writing exactly what they write today. Schema 3 records
must still load.

### Error handling

**A failed member aborts the whole run.** Never continue with fewer members:
the mean and σ would silently change definition mid-trajectory, so every number
after the failure would come from a different committee than the numbers
before it. The error names the member, quotes its traceback, and gives the path
to its worker log.

The same treatment covers:

- **Per-request timeout** (`--member-timeout`), so a hung worker fails rather
  than stalling overnight.
- **Non-finite energy or forces** from any member, caught and named rather than
  propagating a NaN into the mean.
- **Startup failure** — missing env path, no interpreter, `mliprun` not
  installed there, or the MLIP package absent — reported before the optimizer
  starts, in the style `validate_mlip` already uses, pointing at the relevant
  install recipe.
- **Geometry rejected by a member** — the `MixedPBCError` class of problem.
  Every member evaluates the input geometry once, up front, before the
  optimizer starts. This surfaces in the first seconds rather than on step 400
  of an overnight run.

**Teardown kills every remaining worker on every exit path**, including aborts.
On cos-cluster there is no scheduler to reap orphans, and a leaked worker holds
a CUDA context that makes a GPU look busy to everyone else on the node.

**Partial results survive.** The trace is flushed to CSV as it goes and the run
record completes with `status="failed"`, matching `run_optimization` today. A
committee run that dies at step 300 keeps its first 300 steps of disagreement
data.

### Testing

The whole bridge is testable with **no MLIP installed**, because a worker is
just a subprocess that builds a calculator — and ASE ships EMT.

- **Committee arithmetic, no subprocess.** Stub members returning known force
  arrays; assert mean and σ against hand-computed values. Three members at
  1.0, 3.0, 2.0 eV/Å on one component give mean 2.0 and σ 1.0 exactly.
- **Real subprocess round-trip.** A worker launched under the same interpreter
  building EMT, exercising the actual `Popen`/pipe/JSON path in CI. Assert
  against a directly computed EMT result.
- **Identical-members invariant.** N identical EMT workers must give σ ≡ 0 and
  a mean equal to the single-model result.
- **Non-zero σ with a known answer.** A scaled-EMT stub (forces × 1.1) makes
  the mean and spread analytic.
- **stdout pollution.** A worker that prints garbage to stdout before replying.
- **Dead member, timeout, orphan check.** Assert the loud abort, and that no
  worker process survives.
- **Config validation.** `--committee` with `--mlip` rejected; missing env path;
  mixed-level YAML emits the warning and sets the flag; unknown tag resolves to
  `unknown` and warns.
- **Schema 4 round-trip**, with schema 3 records still loading.

Integration against real MLIPs stays behind the existing `uma` / `mace` /
`sevenn` markers, plus manual cos-cluster verification of the shape the
SevenNet work used.

**Trap specific to this feature: test ids must not contain `uma`, `mace`, or
`sevenn`.** The repo's `conftest` matches those as keywords, not markers, so a
test parametrized over realistic member names such as `uma-s-1p2` would be
silently skipped and still look green. Committee tests want exactly those
names, so they need neutral parametrize ids (`member_a`, `task_a`) with the
real tags in the test body.

## What "done" looks like

`mlip optimize run --structure POSCAR --committee committee.yaml` relaxes the
structure under committee-mean forces across MLIP envs, and writes: the relaxed
structure, `opt_committee.csv`, `opt_committee_peratom.csv`, and a schema-4 run
record carrying the member list, the levels of theory, the `mixed_theory` flag,
and the per-configuration uncertainty summary with its flag. The unit suite
passes with no MLIP installed.

## Open questions

- **Weighting.** Plain mean for v1. A per-member reliability weight is a hook
  left open, not built.
- **UMA's fixed per-call overhead** (finding 3) is ~170 ms regardless of system
  size, measured on a CPU-contended node. Whether it is smaller on an idle node,
  and whether it can be reduced at all, is a `fairchem` question outside this
  design. It bounds how cheap any UMA-containing committee can be.
- **Same-level σ_F calibration.** Needed to justify the default flagging
  threshold; see "Flagging rule".
- **MD stride and NEB cost.** Deferred with those commands.

## Not in scope

- MD and NEB (decision 5).
- A `discover` helper that generates `committee.yaml` (decision 4).
- Managed environments. The user supplies env paths; `mliprun` never creates,
  installs, or owns an MLIP env. This is the part of ADR 0001's deferred item
  that stays deferred, and it is why the bridge is weeks rather than the
  multi-month rewrite that ADR priced.
- A project `CANON.md` for this repo. Raised 2026-09-04; Juan's answer was
  "OK for now". Decision 3 therefore lives in this spec alone.
