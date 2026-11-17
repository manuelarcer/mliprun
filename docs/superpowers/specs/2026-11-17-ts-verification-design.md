# Transition-state verification — design

Date: 2026-11-17
Status: draft, waiting for Juan's approval

## Intent

A climbing-image NEB returns a highest image, not a proven transition state
(TS). The standard proof has two parts: the structure has exactly one
imaginary vibrational mode, and that mode moves the system from the initial
state (IS) towards the final state (FS). mliprun computes frequencies
already (`freq run`) but nothing connects that to a NEB result, so the check
is manual today.

This work adds a TS verification step. It is **not NEB-specific**: Juan
expects other TS search methods later (dimer, improved dimer, others). The
verification therefore takes a TS structure, an IS and an FS, and an
optional reference direction. NEB is the first caller, not the owner.

## Decisions

| # | Decision | By |
|---|---|---|
| D1 | Verify by a vibrational analysis of the TS structure: exactly one imaginary mode. | Juan |
| D2 | Two ways to test that the mode connects IS and FS, **overlap** and **displace-and-relax**. The user chooses one or both. | Juan |
| D3 | The differences, advantages and disadvantages of the two tests are documented clearly for users. | Juan |
| D4 | Available both as an option on `neb run` and as a separate command that works on finished calculations. | Juan |
| D5 | The verification is independent of the TS search method, so later methods can reuse it. | Juan |
| D6 | The separate command is `mlip ts verify`. The `ts` group is where later TS search commands can live. No standalone `ts` console script: that name is taken by a common Unix tool (moreutils). | Claude's call |
| D7 | Defaults: overlap threshold 0.7, displacement 0.1 Å, match tolerance 0.3 Å. All adjustable. Shown to Juan on 2026-11-17, not objected to; these are scientific defaults and remain his to change. | Claude's proposal |
| D8 | A failed verification is a result, not an error: exit code 0, verdict printed and recorded. Only a run that could not be carried out exits non-zero. | Claude's call |
| D9 | A wrong mode count on a TS whose fmax is above a limit (default 0.03 eV/Å) is `inconclusive`, not `not_verified`. | Juan, 2026-11-17 |

## Goals

- `mlip ts verify` verifies a TS from a finished NEB folder, or from three
  structure files (TS, IS, FS) produced by any method.
- `neb run --verify-ts overlap|relax|both` runs the same verification
  automatically after a converged climbing-image NEB.
- One core function does the work for both entry points.
- The verdict and every number behind it are written to a text file and to
  the run record.
- User documentation compares the two tests.

## Non-goals

- Dimer or any other TS search method. Only the verification is built now.
- Committee support (`--committee`). `freq run` supports it; this does not
  yet.
- Web app support.
- AutoNEB integration through a flag. A finished AutoNEB can still be
  checked with the three-file form of `ts verify`.
- Thermochemistry or rate constants from the TS frequencies.
- An intrinsic reaction coordinate (IRC) calculation. Displace-and-relax is
  a plain minimisation from two displaced structures, not a path follower.

## Anti-goals

- Reporting "verified" when a requested check did not actually run or did
  not converge. An incomplete check is `inconclusive`, never a pass.
- Changing any number an existing command produces. `freq run` and `neb run`
  without the new option behave exactly as today; golden files stay frozen.
- A second implementation of the frequency calculation. The existing
  `run_frequencies` is reused.

## Constraints

- AQCat25-EV2 tags are refused, like in `freq` (their forces are not the
  energy gradient). Use `reject_aqcat_unsupported`.
- Unit tests must pass with no MLIP installed, with numerical assertions.
  Use an analytic or EMT calculator on a small system with a known saddle.
- Repo rules: tests first, diff coverage at least 90 %, draft PR, no golden
  regeneration, no test id named `uma`, `mace` or `sevenn`.
- A new stage kind in the run record is additive and does not bump the
  schema (docs/OUTPUTS.md states this rule for `singlepoint` and `freq`).

## Approaches considered

1. **Options on `neb run` only.** Simplest to use. A finished NEB cannot be
   checked without rerunning, and the check is tied to NEB.
2. **A separate command only.** Works on old runs and on any method, but is
   always a second manual step.
3. **Both, over one core function (chosen by Juan).** The NEB option and
   the command are thin callers of the same function. After Juan's remark
   that other TS methods will come, the function takes generic inputs and
   the command sits in a new `ts` group, not under `neb`.

## Design

### Core function

New module `src/mliprun/core/ts_verify.py` with one public function, in
outline:

```python
verify_ts(ts_atoms, initial, final, *, checks, calculator_factory,
          reference_direction=None, reference_label=None,
          output_dir, model_name, ...thresholds..., run_context=None) -> dict
```

- `ts_atoms`, `initial`, `final`: ASE Atoms with the same atoms in the same
  order and the same cell. A mismatch is an error before any force call.
- `checks`: a set drawn from `overlap` and `relax`.
- `calculator_factory`: a callable returning a fresh calculator, because the
  frequency run and the two relaxations each need one.
- `reference_direction`: an (N, 3) array, or None. See "Reference
  direction".

Steps:

1. **Frequencies.** Call the existing `run_frequencies` on `ts_atoms` with
   `write_modes="imaginary"`, writing into `<output_dir>/` with prefix
   `ts_freq`. The displaced atoms are the free atoms of `ts_atoms` (the
   existing `select_indices` rule), or an explicit list.
2. **Mode count.** `n_imaginary` from that result (existing 10 cm⁻¹ floor,
   adjustable). Exactly 1 passes. With 0 or with 2 or more, the mode count
   fails and the connection checks are **not run**; they are reported as
   `not_run`. With 2 or more the frequencies are all listed.
3. **Overlap check**, when requested.
4. **Displace-and-relax check**, when requested.
5. **Verdict**, text report and run-record stage.

The imaginary mode vector is read back from `ts_freq_vibrations.json`
(ASE `VibrationsData`), so `run_frequencies` needs no change in behaviour.

### Reference direction

The overlap check compares the imaginary mode with a direction that
represents the reaction path at the TS.

- **From a NEB band** (`reference_label = "neb_tangent"`): the tangent at
  the highest image, computed from its two neighbour images with the
  improved-tangent definition of Henkelman and Jónsson (the definition the
  NEB itself uses, `method="improvedtangent"`), with the minimum-image
  convention.
- **From three files, no band** (`reference_label = "is_to_fs"`): the
  minimum-image displacement from IS to FS. This is a weaker reference when
  the path is curved, and the report says which reference was used.
- A later TS method passes its own direction (for a dimer, its final mode)
  with its own label.

### Overlap check

- Both vectors are restricted to the displaced atoms, taken as Cartesian
  displacements (not mass-weighted), and normalised.
- `overlap = |mode · reference|`, between 0 and 1. The sign is dropped: a
  mode has no direction.
- Pass when `overlap >= threshold` (default 0.7).
- Cost: no force calls beyond the frequency run.

### Displace-and-relax check

- Scale the imaginary mode so that the largest single-atom displacement is
  `displacement` (default 0.1 Å). Build two structures: TS plus and TS minus
  that vector. Constraints are carried over from `ts_atoms`.
- Relax each with a fresh calculator. Optimizer, fmax and maximum steps
  default to the endpoint settings of `neb run` (bfgs, 0.01 eV/Å, 200) and
  are adjustable.
- Compare each relaxed structure with IS and with FS: the largest
  single-atom minimum-image displacement. A structure **matches** an
  endpoint when that value is at most `match_tolerance` (default 0.3 Å).
  The energy difference to the matched endpoint is reported and does not
  decide the match.
- Result, one of:
  - `connects`: one side matches IS and the other matches FS.
  - `one_side`: one side matches an endpoint, the other matches neither.
  - `same_side`: both sides match the same endpoint.
  - `neither`: no side matches an endpoint.
  - `inconclusive`: a relaxation did not converge within its steps.
- Only `connects` passes.
- Cost: two geometry optimizations.

### A loosely converged TS (D9)

A climbing-image NEB usually stops at fmax 0.05 eV/Å. That can be too loose
for a clean Hessian: a real TS can then show zero, or two or more, imaginary
modes. A plain "not verified" would read as "this is not a TS" when the
honest statement is "this TS is not converged enough to test".

- The fmax of the TS structure over its free force components is always
  reported (`run_frequencies` returns it as `fmax_at_input_free_eV_per_A`).
- `--ts-fmax-limit`, default 0.03 eV/Å.
- Mode count not 1 **and** TS fmax above the limit: verdict `inconclusive`,
  reason "TS not converged enough", with the advice to restart the NEB at a
  tighter fmax.
- Mode count not 1 and TS fmax at or below the limit: `not_verified`.
- Mode count 1: the connection checks run, whatever the fmax. The fmax is
  still printed, with a note when it is above the limit.

### Verdict

- `verified`: one imaginary mode, and every requested check passed.
- `not_verified`: the mode count failed on a TS converged within the fmax
  limit, or a requested check failed.
- `inconclusive`: the mode count failed on a TS above the fmax limit; or one
  imaginary mode, no requested check failed, and at least one check is
  `inconclusive`.

The report always carries the individual numbers, so a user who disagrees
with a threshold can judge directly.

### Command `mlip ts verify`

New file `src/mliprun/cli/commands/ts.py`, a Typer group `ts` with the
command `verify`, registered in `cli/main.py`. Two input forms, mutually
exclusive:

- `--neb-dir DIR`: reads the final band (`A2B.traj`), `neb_parameters.txt`
  (model, task/head options, relaxed atoms) and the NEB run record. TS is
  the highest image; IS and FS are the first and last images; the reference
  is the NEB tangent. Model options given on the command line override the
  ones read. If the record says the last NEB stage was not converged or was
  not a climbing-image run, print a warning and continue: the user asked
  explicitly.
- `--ts FILE --initial FILE --final FILE`: any method. Reference is IS to
  FS. Model options as in `freq run` (`--mlip`, `--uma-task`, `--mace-head`,
  `--sevennet-task`, `--device`).

Options for both forms: `--check overlap|relax|both` (default `overlap`),
`--indices`, `--delta`, `--nfree`, `--imaginary-floor`,
`--ts-fmax-limit`, `--overlap-threshold`, `--displacement`, `--match-tolerance`,
`--relax-fmax`, `--relax-optimizer`, `--relax-max-steps`, `--output-dir`.

Default output directory: `<neb-dir>/ts_verify/`, or `ts_verify/` beside the
TS file.

### Option on `neb run`

- `--verify-ts overlap|relax|both`, off by default. Stored in
  `neb_parameters.txt`, so `--restart` reads it back like the other options.
- Requires climbing image. Given without `--climb`, the command stops at
  start, before any model loads.
- Runs only when the NEB stage ends converged. Otherwise it prints a warning
  that verification was skipped and why. The NEB result is unaffected either
  way, and an exception inside the verification must not turn a finished
  NEB into a failed one: it is reported, and the NEB record stays as it was.
- Uses the default thresholds. A user who wants other settings runs
  `mlip ts verify --neb-dir` afterwards.
- Output goes to `<neb-dir>/ts_verify/`.

### Outputs

In the output directory:

- `ts_freq_frequencies.csv`, `ts_freq_summary.txt`, `ts_freq_vibrations.json`,
  the imaginary-mode trajectory, and the displacement cache: the existing
  `freq` outputs, under prefix `ts_freq`.
- `ts_structure.vasp`: the structure that was analysed.
- `ts_relax_plus.traj`, `ts_relax_minus.traj` and their final structures as
  `.vasp`, when the relax check ran.
- `ts_verify.txt`: the plain-text report: verdict, number of imaginary modes
  and their frequencies, reference used, overlap and threshold, per-side
  match table (distance to IS, distance to FS, energy difference, converged
  or not), and the settings.
- Run record: a stage of kind `ts-verify` whose `results` hold the same
  content as the text report. With `--neb-dir` and with `neb run`, the
  verdict must also be findable from the NEB directory's own record; the
  mechanism (an appended stage there, or a pointer to `ts_verify/`) is left
  to the builder, within the rule that existing fields do not change.

### Documentation

- New `docs/TS_VERIFICATION.md`: what is checked, how to run both entry
  points, and the comparison table below, expanded with one worked example.
- README, AGENTS.md, CONTEXT.md (the terms "TS verification", "overlap
  check", "displace-and-relax check"), docs/OUTPUTS.md, docs/PYTHON_API.md,
  CHANGELOG.

| | Overlap check | Displace-and-relax check |
|---|---|---|
| Question answered | Does the imaginary mode point along the reaction path at the TS? | Does the TS fall to the IS on one side and to the FS on the other? |
| Extra cost | None beyond the frequencies | Two geometry optimizations |
| Strength | Fast, deterministic, a number from 0 to 1 | Tests the actual connection to the endpoints |
| Weakness | Local test only. Weak when the path is curved or the reference is the plain IS-to-FS line. Does not show where the system ends | Can end in a neighbouring or symmetry-equivalent minimum and fail although the TS is right. Depends on displacement size, optimizer and match tolerance |
| Use when | Routine screening of many NEBs | A barrier that will be reported, or an overlap that is borderline |

## Facts confirmed in the repository

- The images in `A2B.traj` keep their `FixAtoms` constraints (`setup_neb`
  restores them after the IDPP interpolation), so the free-atom rule works
  on the highest image.
- `A2B.traj` is written after an energy call on every image, so the
  neighbour energies needed by the improved tangent are in the file.
- The first and last images are the optimized endpoints when
  `--optimize-endpoints` is on, which is the default.
- `neb_parameters.txt` stores the model, task/head options, device, fmax and
  climb, and `CustomNEB._parse_parameters_file` parses them. The last
  stage's `status` and `stage_parameters["climb"]` are in the NEB record.
- `RunRecord.begin(..., stage_kind=...)` appends stages already.

## Smaller rules (Claude's calls)

- **`--indices` and the relax check.** `--indices` limits only the atoms
  displaced for the frequencies. The two relaxations always follow the
  structure's own constraints. The report states both atom sets.
- **Rerun into the same directory.** The frequency displacement cache of a
  previous run with other settings is refused by `run_frequencies` today.
  `ts verify` keeps that behaviour and its message; it does not delete a
  cache silently.
- **Highest image is an endpoint.** The band has no interior maximum, so
  there is no TS to verify: stop with a clear message and a non-zero exit.
- **Unrelaxed IS or FS in the three-file form.** The match can then fail for
  reasons unrelated to the TS. The report prints the fmax of IS and FS (one
  force call each) with a note when it exceeds the relax fmax.

- **The NEB directory's record is not modified** (settled while planning).
  Completing an appended stage rewrites the record's top-level status, so it
  would replace the NEB's `converged`. The verification record is always at
  `<neb-dir>/ts_verify/mliprun_run.json`, and docs/OUTPUTS.md states that
  location.

## Left to the builder

- Internal structure of `ts_verify.py` and its helper names.
- The exact ASE calls for reading the mode and its normalisation, after
  confirming ASE's convention for `VibrationsData.get_modes()`.
- How the NEB directory's record points to the verification result.
- The test systems, provided the assertions are numerical and need no MLIP.
- Wording of console messages, following the style of the existing commands.
