# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Committee evaluation with per-configuration uncertainty**
  (`optimize run --committee committee.yaml`). Several MLIPs, each in its own
  Python environment, evaluate the same structure as subprocess workers; the
  optimization is driven by the consensus energy and mean forces, and every
  step records how far the members disagree. Disagreement is reported as
  `sigma`, the per-atom spread of the force vectors in eV/Å (`ddof=1`), with
  `sigma_max` over atoms as the headline number and the worst atom named.
  Energies are compared as an *aligned* spread, each member offset by its own
  first-step energy, because absolute energies from different training sets
  are not on a common zero and their raw spread is meaningless.
  New outputs beside the existing convergence CSV: `<stem>_committee.csv`
  (per-step trace, one energy column per member) and
  `<stem>_committee_peratom.csv` (final per-atom sigma). The convergence
  figure gains a third panel; its y-scale adapts so that exact zeros are
  never silently dropped by a log axis (linear with an annotation when every
  sigma is zero, `symlog` when only some are, log otherwise).
  `--uncertainty-threshold` flags a configuration whose final `sigma_max`
  exceeds it. **No default**: same-level committees disagree by
  0.11-0.15 eV/Å against typical fmax targets of 0.02-0.05, so a default of
  `fmax` fired on ordinary healthy relaxations and was removed before this
  reached a release — see "Changed" below. Without a threshold the run
  still reports `sigma_max`, `sigma_mean` and their ratio to the final
  `fmax`, but asserts no verdict.
  Committee members are shut down on every exit path, including a failed
  startup and `KeyboardInterrupt`, so a worker never survives holding a GPU
  context on a shared node. Mixed levels of theory warn and still run rather
  than refusing, and the level-of-theory table is deliberately incomplete:
  any combination it does not list resolves to `unknown`, which marks the
  committee mixed. Adding a row changes whether a committee reads as
  same-level, so it is a deliberate act, not a way to silence a warning.
  Available on `optimize run` only; `optimize batch`, `md` and `neb` are out
  of scope for now. Verified against a known answer: two members differing by
  a fixed 2.0 eV/Å force offset give `sigma_max` of exactly 2.0/sqrt(2) =
  1.4142135624 eV/Å, and every column of a 21-step trace was independently
  recomputed from the trajectory with zero mismatches at 1e-8.
  See `examples/committee.yaml` and `docs/OUTPUTS.md`.

- **`optimize run --uncertainty-plot`, an opt-in committee uncertainty
  figure.** Writes `<stem>_uncertainty.png`: one panel, two y-axes against the
  optimizer step. Left, the committee mean energy relative to its own step-0
  value with a ±`energy_spread_aligned_eV` band; right, `fmax` with a
  ±`sigma_max_free` band whose lower edge is clipped at zero, plus the fmax
  target line. Together they answer whether the force is converging into the
  committee's own disagreement. Requires `--committee` (rejected outright
  without it, rather than silently writing nothing) and is independent of
  `--plot`. The energy is plotted relative to step 0 because the band is
  measured that way, which makes the band exactly zero-width at step 0 by
  construction — annotated on the figure rather than hidden. The force axis is
  logarithmic (a relaxation spans orders of magnitude in `fmax`) and switches
  to `symlog`, floored at zero, wherever a band edge is clipped: a log axis
  drops non-positive vertices with no warning, which deforms the band polygon
  rather than merely hiding its edge. The energy axis is linear, since
  energy relative to step 0 is negative. No figure is written for a trace
  shorter than two steps.

- **SevenNet model family and `--sevennet-task`.** The SevenNet backend was a
  stub that had never executed: one hardcoded tag (`7net-mf-ompa`) with
  `modal="mpa"` baked into `build_calculator`, no way to select the task, no
  provenance for it, and an install recipe still marked "pending first-run
  validation". All nine tags in the `sevenn` 0.13.0 registry are now
  supported — `7net-omni`, `7net-omni-i8`, `7net-omni-i12`, `7net-mf-ompa`,
  `7net-mf-0`, `7net-omat`, `7net-l3i5`, `7net-0`, `7net-0_22may2024` — with
  any unrecognised `7net-*` tag forwarded to SevenNet unchanged behind a
  warning, so new checkpoints work without a code change.
  `--sevennet-task` (SevenNet's `modal`) is available on `optimize run`,
  `optimize batch`, `md run`, `neb run`, `autoneb run` and `benchmark run`,
  and is written to the params file and the run record. The tag/task table was
  read from the checkpoints themselves, not from SevenNet's documentation,
  which describes a newer release: the documented `7net-nano-*` tags do not
  exist in 0.13.0, `7net-0_22may2024` and `7net-mf-0` are in the registry but
  absent from the documentation's table, and `7net-mf-0` names its tasks
  `PBE`/`R2SCAN` in uppercase, so task names are matched exactly.
  Validated end to end on cos-cluster (NVIDIA L40S): optimize, MD and NEB all
  run, both error paths exit non-zero, and every run record carries the task.
  On one O/Pt(111) structure `7net-omni` gives −97.583282 eV under `mpa` and
  −88.172501 eV under `oc20` — a 9.410782 eV gap, which is the measured reason
  the task is never defaulted.

- `neb run --dyneb` (with `--scale-fmax`): opt-in DyNEB (ASE
  `ase.mep.dyneb.DyNEB`) — dynamic relaxation freezes images already
  converged below fmax so the serial image loop skips their force calls
  (68 → 40 evaluations on the EMT test path at an identical barrier).
  Off by default; plain-NEB numerics are unchanged. Both settings are
  written to `neb_parameters.txt` and the run record, and `--restart`
  reproduces or overrides them like `--climb`/`--k`.

### Breaking

- **`committee.yaml` now rejects head and task combinations it previously
  accepted and ignored.** Committee members build their calculator inside
  their own environment, which skipped the head/task validation the CLI
  applies, so `mlip: mace` with `mace_head: oc20_usemppbe` ran MACE-MP-0
  while naming the member `mace@oc20_usemppbe`, writing a CSV column
  `E_mace@oc20_usemppbe_eV`, and recording that head in the run record — the
  whole record describing a head that was never used. `7net-omat` with
  `sevennet_task: oc20` was worse: the level of theory resolved from the tag
  to `PBE/OMat24` while the calculator was handed `modal="oc20"`, the one
  path where a well-formed file produced a wrong level label. These files now
  fail at parse time with an error naming the offending member. **This breaks
  any existing `committee.yaml` carrying such a combination**, which is the
  point: a provenance record that names the wrong head is not recoverable
  after the fact, a rejected config is. The validation tables are imported
  from the CLI rather than copied, so the two paths cannot drift apart.

- **Run record schema 3 → 4.** Adds a `committee` block (per member: tag,
  task or head, env, resolved level of theory, GPU, and the versions
  *measured* inside that member's environment rather than declared in the
  YAML) plus a flat `committee_config_sha256`. Single-model runs gain no
  committee keys and are otherwise unchanged. On a committee run,
  `device_requested` and `device_resolved` now both read `"committee"`
  instead of the driver environment's resolved device: the driver has no
  torch by design, so those fields previously asserted `"cpu"` for a run
  whose members were on GPUs. The authoritative per-member `device` and `gpu`
  live in the committee block. A `level_table_version` is stamped alongside,
  so a record written under a later-corrected level-of-theory table can be
  re-judged rather than silently trusted.

- **`--device` is now rejected together with `--committee`.** Each member's
  device comes from `committee.yaml`, so the flag was accepted and silently
  did nothing.

- **`--uma-task` and `--mace-head` are now required, with no defaults.** They
  previously defaulted to `omat` and `omat_pbe` and ran silently when unset —
  the exact hazard CANON C1 names. A `uma-*` model with no `--uma-task`, or a
  `mace-mh-*` model with no `--mace-head`, now exits with an error listing the
  valid values. Plain `mace` (MACE-MP-0) is single-head and now *rejects*
  `--mace-head`, the way a single-task SevenNet tag rejects `--sevennet-task`.
  **This breaks existing command lines and orchestrator scripts** that relied
  on the defaults: they will fail loudly on the next run and need the head
  added. That is the point — a wrong energy zero is not recoverable after the
  fact, a failed launch is. The same rule now applies to `--mlip auto`, to
  library callers (`CustomNEB` raises `ValueError`), and to `benchmark run`,
  where UMA joins the auto-detected list only when a task is given.
  Verified on cos-cluster (NVIDIA L40S) against the real `fairchem-core`
  2.19.0 and `mace-torch` 0.3.15 environments: all three error paths exit
  non-zero (UMA with no task, `mace-mh-1` with no head, plain `mace` given a
  head), and `optimize run` completes with `--uma-task oc20` and with
  `--mace-head oc20_usemppbe`, each run record carrying its own head and
  nulling the other two.
- **The `--uma-task` list was wrong.** It advertised `omat`, `oc20`, `omol`,
  `odac`; `uma-s-1p2`'s own task registry (fairchem-core 2.19.0) also carries
  `omc`, `oc22` and `oc25`. All seven are accepted and documented. The MACE
  head list was verified against the `mace-mh-1` checkpoint and was correct.
- **`7net-mf-ompa` now requires `--sevennet-task mpa`** (or `omat24`) where it
  previously ran on a hardcoded `mpa`. Treated as a free break: the SevenNet
  path had never executed, so no run record, result, or script depends on it.

### Changed

- **Run record schema `2` → `3`.** `provenance.sevennet_task` joins
  `uma_task` and `mace_head`, gated on the model tag the same way, so a
  SevenNet record identifies its own level of theory. The value is recorded
  verbatim (`7net-mf-0`'s tasks are uppercase, and normalising would write a
  task name no checkpoint has). The field also joins the stage-provenance diff,
  so a task switch mid-pipeline is reported as a change — the CANON C3 guard.
- **`--sevennet-task` has no default, unlike `--uma-task` and `--mace-head`.**
  A multi-task SevenNet tag with no task stops the run with an error listing
  that model's valid tasks. SevenNet tasks are independent fine-tunes with
  independent energy zeros, so a guessed task silently changes the level of
  theory (CANON C1). One consequence is deliberate and worth knowing: in a
  SevenNet-only environment `--mlip auto` resolves to `7net-omni` and then
  stops, so the "a fresh env lands on a runnable default" promise does not
  hold for SevenNet.
- **`--mlip auto` picks `7net-omni` instead of `7net-mf-ompa`** when SevenNet
  is the only MLIP installed. `7net-omni` is SevenNet's recommended model and
  the only one of the family with surface heads (`oc20` RPBE, `oc22`). The
  detection *order* (UMA → MACE → SevenNet → CHGNet) is unchanged.
- **Auto-detected model tags are now validated.** `optimize`, `md`, `neb` and
  `autoneb` previously validated only the explicit `--mlip` branch, so a tag
  chosen by auto-detection was never checked.
- **Run record schema `1` → `2`.** `mliprun_run.json` now records the head/task
  that actually ran (`provenance.uma_task` / `provenance.mace_head`), so the
  record identifies the level of theory on its own — a model tag alone does
  not, because the heads are independent fine-tunes with independent energy
  zeros. Each field is recorded only for its own model family, so a MACE run
  is never labelled with the `--uma-task` its command line happened to carry.
  `run_optimization`, `run_md` and `CustomNEB` accept the head/task as
  keywords; `optimize`, `md`, `neb` and `autoneb` forward what they parsed.
  Consumers that pin `schema_version == 1` must be updated.
- An appended stage whose provenance carries a field the stored record has no
  key for (resuming a run that started under schema 1) reports it under
  `stage_provenance_new_fields`, meaning "this field is new; stage 0's value
  was not recorded". Previously such a field landed in `stage_provenance`,
  which claims the value *changed* — so a same-head resume of a legacy run
  wrongly asserted the head had switched. `schema_version` is left as stage 0
  wrote it; see [docs/OUTPUTS.md](docs/OUTPUTS.md#stages).
- **Breaking:** the committee's headline force disagreement now excludes
  constrained force components, so it is taken over the same atoms as
  ASE's `fmax`. Previously a frozen slab atom could carry the reported
  maximum: 8 of 17 atoms were fixed in one measured O/Pt(111) relaxation, 32
  of 82 in a CH/FeNi one. The unmasked values are kept alongside it. Only
  `FixAtoms` and `FixCartesian` are
  masked — the only stock ASE constraints whose `adjust_forces` is a pure
  component mask; every other constraint type leaves its atoms counted as
  free (sigma over-reported, never under-reported) and its type name is
  recorded in `unhandled_constraints`.
- **Breaking: every sigma name now states which atoms it covers.** A name
  carries `free` (the atoms free to move) or `all` (every atom in the
  cell); there is no bare form. A bare `sigma_max` silently meant "free
  atoms only", which is a convention a reader had to already know — and the
  two CSVs had picked *opposite* conventions for the same word. The rename
  is deliberate and total, with no aliases and no duplicate compatibility
  columns; it covers the trace CSV, the per-atom CSV, the run record, and
  the `committee_statistics` dict a Python caller sees:

  | file / dict | before | after |
  |---|---|---|
  | trace CSV | `sigma_max_eV_per_A` | `sigma_max_free_eV_per_A` |
  | trace CSV | `sigma_mean_eV_per_A` | `sigma_mean_free_eV_per_A` |
  | trace CSV | `worst_atom` | `worst_atom_free` |
  | per-atom CSV | `sigma_eV_per_A` | `sigma_all_eV_per_A` |
  | run record | `sigma_max_final_eV_per_A` | `sigma_max_free_final_eV_per_A` |
  | run record | `sigma_mean_final_eV_per_A` | `sigma_mean_free_final_eV_per_A` |
  | run record | `sigma_max_peak_eV_per_A` | `sigma_max_free_peak_eV_per_A` |
  | run record | `sigma_max_over_fmax_final` | `sigma_max_free_over_fmax_final` |
  | run record | `worst_atom` / `worst_atom_symbol` | `worst_atom_free` / `worst_atom_free_symbol` |
  | stats dict | `sigma_per_atom` | `sigma_per_atom_all` |
  | stats dict | `sigma_max` / `sigma_mean` / `worst_atom` | `sigma_max_free` / `sigma_mean_free` / `worst_atom_free` |

  The `_all` names (`sigma_max_all_eV_per_A`, `sigma_max_all_final_eV_per_A`,
  `sigma_per_atom_free`, `sigma_free_eV_per_A`, `n_free_atoms`,
  `free_components`) already named their population and are unchanged.
- **Breaking:** `--uncertainty-threshold` no longer defaults to `--fmax`.
  With no threshold the run still reports `sigma_max_free`,
  `sigma_mean_free`, the worst free atom, and their ratio to fmax
  (`sigma_max_free_over_fmax_final`), but
  asserts no verdict, and `flagged` is `null` rather than `false`. The old
  default fired on ordinary healthy relaxations (same-level committees
  measured 0.11-0.15 eV/Å against convergence targets of 0.02-0.05).
  `flagged` is now tri-state: `true`/`false` when a threshold was checked,
  `null` when none was applied *or* when a threshold was set but the run
  died before its first evaluation — `null` is not the same as `false`.
- **Breaking: run record schema 4 → 5.** Three changes, not one. (1) The
  *meaning* of the reported disagreement changed: it now excludes
  constrained force components, so a schema-5 `sigma` is not comparable to a
  schema-4 one at the same key. That is why this bump matters more than the
  previous two, which only added keys. (2) Every sigma key was renamed to
  carry its population (table above), so the meaning change is visible from
  the key rather than only from the version — a consumer reading a schema-4
  record for `sigma_max_free_final_eV_per_A` gets a `KeyError`, not the
  wrong number. (3) `results.committee_uncertainty` gains
  `sigma_max_all_final_eV_per_A`, `sigma_mean_all_final_eV_per_A`,
  `n_free_atoms`, `unhandled_constraints` and
  `sigma_max_free_over_fmax_final`; `threshold_source`
  is now `"explicit"` or `"none"` (`"fmax"` can no longer be produced).
- **Breaking:** both committee CSVs changed their headers. `<stem>_committee.csv`
  gains `sigma_max_all_eV_per_A`, `sigma_mean_all_eV_per_A` and
  `n_free_atoms` and renames three columns (table above);
  `<stem>_committee_peratom.csv` gains `sigma_free_eV_per_A` and
  `free_components`, renames `sigma_eV_per_A` to `sigma_all_eV_per_A`, and
  still lists every atom, constrained ones included.
- **`ase>=3.23` is now a hard floor.** The constraint masking reads
  `constraint.index` and `constraint.mask` directly, and ASE changed both:
  `FixAtoms`/`FixCartesian` moved onto `IndexedConstraint` at 3.23, and
  before that `FixCartesian` stored the inverted mask. An older ASE would
  either raise or silently flip which components count toward sigma.
- **Breaking:** `mace` (MACE-MP-0) and `chgnet` now resolve to one level of
  theory, `PBE(+U)/MPtrj`, because they share one training set. A committee
  of the two is no longer reported as mixed theory, and its spread is an
  error bar rather than a functional comparison. `LEVEL_TABLE_VERSION` 1 → 2.
  The `7net` MPtrj tags are unchanged pending checkpoint confirmation.

### Fixed

- **A reused committee no longer crashes on a structure that needs no force
  evaluation.** Handing `run_optimization` the same `Atoms` object twice in
  one process, already at `fmax`, makes ASE's calculator cache answer every
  call, so `calculate()` never runs and the committee's `latest` statistics
  stay empty. The final summary block subscripted them anyway: `TypeError`,
  and because it fired before `record.complete()`, the run record was left
  saying `"running"` forever. The block is now guarded; the record gets an
  honest all-null `committee_uncertainty` and no per-atom CSV.
- **A plain `kill` (SIGTERM) on a committee run no longer orphans its
  workers.** SIGTERM's default disposition terminates the interpreter without
  unwinding the stack, so neither the CLI's teardown nor the `atexit` backstop
  ran, and every worker was left running and holding its CUDA context — which
  makes the GPU look busy to everyone else on a shared node with no scheduler
  to reap orphans. Verified by pid on cos-cluster (2026-09-07): driver dead,
  two workers still holding 2948 MiB minutes later. `optimize run --committee`
  now routes SIGTERM into the same unwinding path `Ctrl+C` already took, for
  the committee's whole lifetime including the model loads, and exits **143**
  (128 + SIGTERM) after shutting the members down. The handler is installed by
  the CLI, not by the library, and is removed on the way out: `run_optimization`
  is also a Python API entry point and must not change signal behaviour for a
  caller who installs their own. Runs without `--committee` are untouched.
- **A committee worker now exits when its driver dies without cleaning up**
  (SIGKILL, an OOM kill, a failed node). The documented backstop — the worker
  seeing EOF once the driver's pipe closes — only ever covered an *idle*
  worker: inside a calculation, which is where a member spends nearly all of a
  relaxation's wall clock, the process is not reading stdin and cannot observe
  the close at all. That is why the orphaned cluster workers were in state `R`.
  Each worker now polls its parent pid every 2 s and exits `3` once the driver
  is gone, comparing against the pid recorded at startup rather than against 1,
  so a subreaper does not hide the orphaning.

## [0.4.0] - 2026-07-14

### Added

- `mlip doctor`: environment self-check reporting package versions, asetools health (distinguishes the real GitHub package from the unrelated PyPI `asetools`), installed MLIP packages, what `--mlip auto` resolves to, and torch/CUDA status. Exits 0 iff at least one MLIP is installed, so scripts and CI can assert on it. (#31)
- `AGENTS.md` and `CLAUDE.md`: committed orientation for AI coding assistants and new users — install rules, verification, testing commands, and repo conventions. (#31)
- CI: `install-smoke.yml` workflow proves the README Quick Start on a clean runner (base install → `[neb]` extra → CPU MACE → `mlip doctor` gate → small relaxation with a finite-energy assertion), weekly and on packaging-file changes. (#31)
- `optimize batch`: relax a series of structures in one process, loading the MLIP model **only once** and reusing it across every relaxation (avoids the per-run model-load cost). Discovers one input structure per immediate subdirectory of `--parent` (default `--input-name '*.vasp'`; the platform's own `*_final.vasp` outputs are ignored), relaxes each in place, continues past failures, and writes a `batch_summary.csv` into `--parent`. Supports `--skip-existing` to resume a partial batch.
- `build_calculator()` in `mliprun.cli.utils`: builds and returns an ASE calculator without attaching it, so the loaded model can be reused across many structures. `setup_calculator()` is now a thin build-and-attach wrapper around it.
- `optimize run` / `optimize batch` now also write a fixed-name `CONTCAR` (copy of the final relaxed structure) so a follow-up DFT run managed by asetools can restart from the directory.
- `--plot / --no-plot` flag on `optimize run`, `optimize batch`, `md run`, and `neb run`. Plotting is now **opt-in**: with `--plot` the command writes its PNG figures; without it (the default) only the CSVs are written. Threaded through `run_optimization(plot=...)`, `run_md(plot=...)`, and `CustomNEB.run_neb(plot=...)`.
- `neb run --device {cpu|cuda}`: run NEB on GPU. Defaults to `cpu` (unlike `optimize`/`md`, which default to `auto`); pass `--device cuda` for GPU runs. The device is recorded in `neb_parameters.txt` and honored on `--restart`.
- `neb run` support for multi-head MACE foundation models via `--mace-head` (same head values as the other commands); the resolved MACE head is logged in the NEB parameter file.
- CI: `tests.yml` GitHub Actions workflow runs the unit suite with a diff-coverage gate (>= 90% on changed lines) on every push/PR, plus a `weekly.yml` scheduled quality-check run. New baseline characterization tests (golden outputs + physics invariants) guard against regressions.

### Changed

- **Project renamed: `mlip-platform` → `mliprun`.** The GitHub repository is now `manuelarcer/mliprun` (old URLs redirect), the distribution is `mliprun`, and the Python import package is `mliprun` (was `mlip_platform`). CLI commands are unchanged (`mlip`, `optimize`, `md`, `neb`, …). Existing editable installs must be refreshed after pulling: `pip uninstall mlip-platform -y && pip install -e .` — and any personal scripts using `import mlip_platform` need updating to `import mliprun`.
- **Packaging moved from `setup.py` to `pyproject.toml`** (PEP 621); `setup.py` remains as a compatibility shim. (#32)
- **asetools is no longer a base dependency.** The bare PyPI name `asetools` resolves to an unrelated Aseprite tool, and the only in-code use is the guarded NEB interpolation sanity check — so it is now an optional extra installed from GitHub: `pip install -e ".[neb]"`. A plain install runs NEB **without** the interpolation distance check (by design); `mlip doctor` reports the state. (#32, #31)
- CI workflows install `.[dev,neb]` instead of a separate `pip install git+…` asetools step, exercising the same install path users run. (#31)
- Dependency `typer[all]` → `typer` (the `[all]` extra is empty in modern typer and produced a pip warning on every fresh install). (#31)
- **Plotting is now opt-in across `optimize`, `md`, and `neb`.** Previously these commands always wrote PNG figures (`optimize` had a `--no-plot` opt-out; `md`/`neb` had no way to disable it). Now no PNG is written unless `--plot` is passed. The CSV data (`*_convergence.csv`, `md_energy.csv`, `neb_convergence.csv`, `neb_data.csv`) is always written, so nothing is lost — figures can be regenerated from it. Motivation: the figure/save is IO that dominates short runs (measured ~3x speedup on frozen-surface site scans). The old `--no-plot` still works as the off side of `--plot/--no-plot`, so existing scripts are unaffected.
- **Auto-detection order changed to UMA → MACE → SevenNet → CHGNet** (was UMA → SevenNet → MACE → CHGNet). UMA is still preferred when installed, but MACE is now the preferred fallback ahead of SevenNet/CHGNet: UMA is gated on Hugging Face and unusable without an access request, whereas `pip install mace-torch` gives a working model immediately. A fresh environment therefore lands on a usable default.
- NEB now uses ASE's `improvedtangent` tangent method instead of the older default (`aseneb`), which improves convergence on curved reaction paths.
- All parameter files (`opt_params.txt`, `md_params.txt`, `neb_parameters.txt`) now record the resolved MLIP head/task, so a run's provenance is fully captured in its parameter file.
- `md_energy.png` plots kinetic energy on a twin right axis for readability alongside total/potential energy (when `--plot` is used).

## [0.3.0] - 2026-05-08

### Added

- `mlip` parent CLI exposing every subcommand under one namespace (`mlip optimize`, `mlip md`, `mlip neb`, `mlip autoneb`, `mlip autoneb-results`, `mlip benchmark`). The standalone aliases (`optimize`, `md`, …) continue to work.
- `md run --resume`: continue an MD run from the last frame of `md.traj`. Preserves momenta (no Maxwell-Boltzmann re-init), appends to `md.traj` and `md_energy.csv`, treats `--steps` as additional steps. `setup_dynamics` now accepts `set_velocities`; `run_md` accepts `resume`.
- `MLIP_HELP` and `UMA_TASK_HELP` constants in `mlip_platform.cli.utils` — single source of truth for the model / task help wording, imported by all four CLI commands.
- `--models tag1,tag2` flag on `benchmark run` for explicit model selection.
- `--output bench.json` flag on `benchmark run` to persist results to a JSON file.
- New documentation: `docs/PYTHON_API.md` (public Python API reference), `docs/OUTPUTS.md` (canonical output-files reference), `docs/MD_REFERENCE.md` (MD CLI reference, replacing the old design doc), `CHANGELOG.md`, and `CONTRIBUTING.md`.

### Changed

- `mlip_platform.core.optimize.run_optimization`: default `optimizer` flipped from `"fire"` to `"bfgs"`. The CLI `optimize run` already defaulted to `bfgs`; the function default now matches.
- `benchmark run` rewritten as an in-process loop. Replaces the previous subprocess shell-out to `bench_driver.py` (which only worked from the repo root). Now auto-detects every installed MLIP (UMA, SevenNet, MACE, CHGNet) and is independent of the working directory.
- README and Windows setup guide updated to describe the `mlip` namespace, list CHGNet alongside the other MLIPs, and document the auto-detection priority (UMA → SevenNet → MACE → CHGNet) explicitly.
- `docs/UMA_USAGE_GUIDE.md` rewritten end-to-end: all CLI examples now use the `<command> run` subcommand pattern, `uma-s-1p2` is documented as the only current model, and a new "Hugging Face access" section covers gated-repo setup.
- `tests/README.md` replaced — previous content described an unrelated project ("MILP 1") and was wholly inaccurate.
- Validator error message in `validate_mlip` reworded to reflect the wildcard `uma-*` behavior instead of pretending only `uma-s-1p1` and `uma-m-1p1` are accepted.
- Help strings in all four CLI commands now state that any `uma-*` tag is forwarded to FAIRChem unchanged.

### Removed

- `src/mlip_platform/core/mlip_bench.py` — dead module (109 lines), not imported by anything live.
- `docs/MD_ENSEMBLE_DESIGN.md` — replaced by `docs/MD_REFERENCE.md`. The implementation-plan / Phase 1–3 / file-architecture sections were stale; the implementation has been complete for some time.
- `bench_driver.py` (root) — superseded by the rewritten in-process `benchmark run`. The "Benchmarking with bench_driver.py" subsection in `UMA_USAGE_GUIDE.md` now points at `benchmark run` instead.
- `docs/Python_MLIP_UMA_Setup_Guide.docx` and `.pdf` moved to `docs/legacy/` — `windows_setup_guide.md` is the canonical Markdown version.

### Fixed

- README example commands no longer reference `uma-s-1p1` after the project default switched to `uma-s-1p2`.
- README and CLI help no longer claim a `mlip` command exists when it doesn't (now actually wired).
- `md run --resume` step accounting: the previous draft called `dyn.run(prior_steps + steps)` after `dyn.nsteps = prior_steps`, which double-counted because ASE's `dyn.run(N)` runs `N` *additional* steps. A resume of `--steps 50` from step 50 now correctly ends at step 100, not step 150.

## [0.2.0] - 2026-03-13

### Added

- Shared utility modules: `mlip_platform.cli.utils` (MLIP detection / validation, calculator setup, relax-atoms parsing) and `mlip_platform.core.utils` (`calc_fmax`, GPa conversion).
- Parameter-file I/O helpers in `mlip_platform.core.params_io` (`write_parameters_file`, `write_endpoint_results`).
- AutoNEB support: new `autoneb` and `autoneb-results` CLI commands, `CustomNEB.run_autoneb`, MIC-aware path interpolation.
- NEB `--restart` flag with `bkup_<timestamp>/` automatic backup folder, parameter override / lock rules, image-count consistency check.
- Highly-constrained NEB mode: `--relax-atoms 0,1,5` keeps only the listed atoms mobile (others fixed); skips IDPP automatically.
- Automatic endpoint optimization (default-on) for both `neb` and `autoneb`, with similarity check between input and relaxed endpoints.
- BFGS and L-BFGS as NEB optimizers (`--neb-optimizer`).
- MD ensemble support: NVE, NVT (Langevin / Nose-Hoover / Berendsen), NPT (MTK / Berendsen). New `--ensemble`, `--thermostat`, `--barostat`, `--friction`, `--ttime`, `--taut`, `--taup` flags on `md run`.
- `--maximum-steps` limit for NEB optimization.
- Detailed iteration logging with custom log-file option for NEB.
- UMA-1.2 (`uma-s-1p2`) added as the new default UMA model; old `uma-s-1p1` still loads if requested.
- Windows setup guide (`docs/windows_setup_guide.md`).
- Comprehensive pytest suite — 17 test files, EMT-based unit tests plus marker-gated MLIP integration tests (`uma`, `mace`, `sevenn`, `slow`).

### Changed

- Default UMA model changed from `uma-s-1p1` to `uma-s-1p2`.
- Default NEB optimizer changed from MDMin to FIRE.
- Default `optimize` optimizer changed from FIRE to BFGS.
- NEB output now goes to the current working directory (was previously inconsistent across commands).
- NEB convergence plot shows barrier height instead of absolute max energy.
- NEB energy reference is now the initial structure (was previously the band minimum).
- NEB force logging uses actual NEB forces instead of cached image forces.
- Lazy imports throughout the CLI — heavy MLIP packages are loaded on demand only, eliminating PyTorch warnings and slow startup.
- `optimize` convergence filenames use the `--logfile` stem so two runs with different logs don't collide.

### Fixed

- NEB restart: parameter file is no longer overwritten before backup; copied to backup, then a fresh restart-tagged file is written.
- NEB restart: handles `None` overrides correctly when reusing loaded parameters.
- NEB restart: skips re-creation of the NEB instance when loading from restart.
- NEB endpoint optimization: `FixAtoms` constraints now correctly applied when `--relax-atoms` is set.
- NEB output directory respects symlinks.
- Endpoint optimization: convergence check uses the converged criterion, not the optimizer's exit code.
- AutoNEB: initial-image energies now computed before AutoNEB starts (avoids crash).
- AutoNEB: `n_simul` defaults to 1 with a warning when `>1` is requested without MPI.
- AutoNEB: stale `*.traj` and `AutoNEB_iter/` from prior runs are cleaned up before starting a new run.
- F-string syntax error in restart validation.
- Removed `egg-info` from git tracking (was a build artifact).

## [0.1.0]

Initial public iteration.

### Added

- Standalone CLI commands: `optimize`, `md`, `neb`, `benchmark`.
- Single-point energy + timing benchmark via `bench_driver.py` (called as a subprocess from `benchmark run`).
- Initial UMA model support alongside MACE and SevenNet.
- Initial pytest scaffolding for MACE and SevenNet single-point energy validation.
- Project README and `.gitignore`.

The 0.1 series predates this CHANGELOG; entries are reconstructed from the git history. There was no formal 0.1.0 release tag; `setup.py` jumped from project inception to `0.2.0` during the 2026-03-13 refactor.

[Unreleased]: https://github.com/manuelarcer/mliprun/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/manuelarcer/mliprun/releases/tag/v0.4.0
[0.3.0]: https://github.com/manuelarcer/mliprun/releases/tag/v0.3.0
[0.2.0]: https://github.com/manuelarcer/mliprun/releases/tag/v0.2.0
