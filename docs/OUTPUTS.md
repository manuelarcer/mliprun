# Output Files Reference

Canonical list of every file each CLI command writes. Search this page to find what produces a given filename, what format it's in, and which command's output directory it ends up in.

## Where output goes

| Command | Output directory |
|---------|------------------|
| `optimize run` | Directory containing `--structure` (i.e. `Path(--structure).parent`) |
| `optimize batch` | One relaxation per immediate subdirectory of `--parent` (outputs written into each subdir); a `batch_summary.csv` is written into `--parent` |
| `md run` | Directory containing `--structure` |
| `neb run` | Current working directory at invocation |
| `autoneb run` | Current working directory at invocation |
| `autoneb-results results` | The directory passed via `--directory` (default `.`) |
| `benchmark run` | Nothing on disk by default; `--output bench.json` writes a JSON file there |

This is not always the same directory the user is sitting in. `optimize` and `md` write *next to the input structure*; `neb` and `autoneb` write *into the cwd*. Set up the working directory accordingly before running NEB / AutoNEB.

---

## `optimize run`

| File | Format | Contents |
|------|--------|----------|
| `opt.traj` | ASE trajectory (binary) | Every optimizer step |
| `opt.log` | text | ASE optimizer log (step, fmax, energy) |
| `opt_convergence.csv` | CSV | columns: `step`, `energy(eV)`, `fmax(eV/A)` |
| `opt_convergence.png` | PNG | Energy and fmax vs step — **only with `--plot`** (plotting is opt-in; the CSV is always written) |
| `opt_final.vasp` | VASP POSCAR (vasp5, direct) | Final relaxed structure |
| `CONTCAR` | VASP POSCAR (vasp5, direct) | Copy of the final relaxed structure, named so a follow-up DFT run (e.g. managed by asetools) can restart from this directory |
| `opt_params.txt` | plain text, key/value | Echo of run parameters (MLIP, optimizer, fmax, max_steps, etc.) |
| `mliprun_run.json` | JSON | Canonical run record: every resolved parameter with the source of its value, provenance (versions, device, host, timings), and per-stage outcome. See [The run record](#the-run-record). |

If you change `--logfile <name>.log`, the convergence CSV / PNG and final POSCAR are renamed accordingly: `<name>.log`, `<name>_convergence.csv`, `<name>_convergence.png`, `<name>_final.vasp`. The `CONTCAR` filename is fixed (it does not follow `--logfile`). The trajectory filename comes from `--trajectory`. **Two relaxations launched in the same directory will overwrite each other** unless you set `--logfile` and `--trajectory` to different names.

With `--committee committee.yaml`, two more files are always written (see [Committee outputs](#committee-outputs) below):

| File | Format | Contents |
|------|--------|----------|
| `opt_committee.csv` | CSV | Per-step disagreement trace |
| `opt_committee_peratom.csv` | CSV | Per-atom disagreement at the final geometry |
| `committee_<member>.log` | text | One per member: that env's library banners and any remote traceback |

These also follow `--logfile <name>.log`: `<name>_committee.csv`, `<name>_committee_peratom.csv`. With `--plot`, `<name>_convergence.png` gains a third panel (see [Committee outputs](#committee-outputs)).

---

## Committee outputs

A committee is two or more MLIPs, each in its own environment, relaxing the
same structure together (`optimize run --structure POSCAR --committee
committee.yaml`). The relaxation follows their **mean** force; the outputs
below report how much they **disagree**, as a diagnostic. Force disagreement
(every `sigma_*` column) is in eV/Å; energy disagreement
(`energy_spread_aligned_eV`) is in eV. It is
never a substitute for DFT validation. Not supported by `optimize batch`,
`md`, or `neb`/`autoneb`: committees run through `optimize run` only.

### Constraint masking

**Every sigma name says which atoms it covers.** A name carrying `free`
(`sigma_max_free_eV_per_A`, `sigma_free_eV_per_A`, `worst_atom_free`, …) is
taken over **free force components only** — the same population ASE's
`fmax` uses for its convergence test, not every atom in the cell. A name
carrying `all` (`sigma_max_all_eV_per_A`, `sigma_all_eV_per_A`, …) covers
every atom in the cell, constrained ones included. There is no bare
`sigma_max`: which population a number covers is the one thing you should
never have to remember.

A constrained atom cannot move regardless of how much the members disagree
about its force, so including it would compare against the wrong
population; see the 2026-09-08 design note
(`docs/superpowers/specs/2026-09-08-committee-sigma-masking-design.md`).

Only `FixAtoms` and `FixCartesian` are masked: they are the only stock ASE
constraints whose `adjust_forces` is a pure component mask, so the component
they hold is exactly zero and dropping it is unambiguous. Every other
constraint type (`FixScaled`, `FixedPlane`, `FixedLine`, `FixBondLength`, …)
projects rather than masks, so it is left **unhandled**: its atoms stay
counted as free, sigma is over-reported for them rather than silently
under-reported, and the type names are recorded in `unhandled_constraints`
(in `results.committee_uncertainty`, and echoed once by the CLI as a
warning naming the constraint types).

If every atom in the structure is fully constrained, the `free` values fall
back to the `all` ones since there is no free population left to reduce
over.

The pre-masking numbers survive in full — `sigma_max_all_eV_per_A` and
`sigma_mean_all_eV_per_A` (trace CSV), `sigma_max_all_final_eV_per_A` and
`sigma_mean_all_final_eV_per_A` (run record), `sigma_all_eV_per_A`
(per-atom CSV) — so runs from before 2026-09-08 stay comparable.

### `<name>_committee.csv`

One row per optimizer step, flushed as it is written, so a run that dies at
step 300 keeps its first 300 rows.

| Column | Meaning |
|--------|---------|
| `step` | Optimizer step index |
| `energy_mean_eV` | Mean of the members' raw energies (the optimizer's objective) |
| `energy_spread_aligned_eV` | Standard deviation (`ddof=1`, not a range) across members after each member's own step-0 energy is subtracted |
| `E_<member>_eV` | One column per member, its raw energy |
| `fmax_eV_per_A` | Max atomic force at this step (same quantity as `opt_convergence.csv`) |
| `sigma_max_free_eV_per_A` | Largest per-atom force disagreement across members at this step, over **free force components only** — the same atoms ASE's `fmax` uses |
| `sigma_mean_free_eV_per_A` | Mean per-atom force disagreement at this step, over the atoms with at least one free component |
| `worst_atom_free` | Index of the atom with the largest free-component disagreement at this step |
| `sigma_max_all_eV_per_A` | The same maximum with no constraint masking, kept so runs from before 2026-09-08 stay comparable |
| `sigma_mean_all_eV_per_A` | The same mean with no constraint masking, over every atom in the cell |
| `n_free_atoms` | How many atoms retain at least one free force component |
| `mixed_theory` | Whether the committee spans more than one level of theory (see below); repeated on every row so a downstream filter needs no terminal output |

`energy_mean_eV` is not comparable in absolute terms across MLIP packages,
because each package carries its own constant energy offset, but
*differences* along the trajectory are meaningful. `energy_spread_aligned_eV` removes that
per-member offset before taking the spread, so it is a real energy
uncertainty; it is exactly zero on step 0 by construction (that is the row
each member's offset is measured against).

### `<name>_committee_peratom.csv`

One row per atom, at the **final** geometry only (not every step: a
per-atom field at every step would be a large file for little gain, and
`worst_atom_free` above already traces where the disagreement lived during
the run). This is usually the most diagnostically useful committee output. It
names *which* atoms the members disagree about, typically the adsorbate or
the bond being formed or broken, not just that they disagree.

| Column | Meaning |
|--------|---------|
| `atom_index` | Index into the structure |
| `symbol` | Chemical symbol |
| `sigma_all_eV_per_A` | Per-atom force disagreement, unmasked (all three components) |
| `sigma_free_eV_per_A` | The same quantity with constrained components dropped |
| `free_components` | How many of the atom's 3 force components are free (0-3; `0` means the atom is fully fixed) |

**Every atom keeps its row, constrained ones included.** Seeing the frozen
atoms alongside `sigma_all_eV_per_A` and `sigma_free_eV_per_A` side by side
is how a reader checks, on their own run, how much of the disagreement sits
in a region that cannot move.

### `committee_<member>.log`

One file per declared member (named after the member's `name` in
`committee.yaml`), receiving that env's stderr: library import banners and,
if the member fails, its remote traceback. Referenced by path in any
committee error message.

### The flagging rule

**There is no default threshold.** The run always reports
`sigma_max_free`, `sigma_mean_free`, the worst free atom, and
`sigma_max_free_over_fmax_final` (the ratio of `sigma_max_free` to the
final `fmax`) — those numbers are the deliverable whether
or not anyone sets a threshold. `--uncertainty-threshold` (eV/Å) is
**opt-in**: pass it and a configuration is **flagged** when
`sigma_max_free` at the *final* geometry exceeds it. The reasoning is that
if the members disagree about the forces by more than the convergence
tolerance, the located minimum sits inside the committee's own noise. When
flagged, the CLI prints one warning naming the worst atom.

There used to be a default (`--fmax` itself); it is gone. Same-level
committees measured on cos-cluster disagreed by 0.11-0.15 eV/Å against
convergence targets of 0.02-0.05, so the old default fired on ordinary
healthy relaxations. Those numbers also predate excluding constrained atoms
from the reported maximum, so they are not a calibration for a new default
either.
See the 2026-09-08 design note
(`docs/superpowers/specs/2026-09-08-committee-sigma-masking-design.md`).

**`flagged` is tri-state:** `true` (checked against a threshold and it was
exceeded), `false` (checked and it passed), or `null` (no verdict was
reached). `null` happens two ways: no threshold was applied (the default
now), *or* a threshold was applied but nothing was ever evaluated — a run
that died before its first optimizer step. **`null` is not the same as
`false`.** A script that filters `flagged == false` to select healthy runs
would otherwise count a run that crashed before its first force call as
healthy.

With `--relax-cell`, the two sides of that comparison are not quite the same
quantity. `fmax_eV_per_A` then includes the cell virials the cell filter emits
alongside the atomic forces, because that is what the optimizer's convergence
test uses, while `sigma_max_free` is disagreement about **atomic forces
only**: the members are asked for forces, never for a stress. Read a flag on
a cell relaxation as "the members disagree about the atomic forces by more
than the combined force/virial tolerance", not as a like-for-like ratio.

On a run that **failed** (a member rejecting the geometry, a member dying),
`sigma_max_free_final_eV_per_A` and the other `*_final_*` fields describe the last
**successful** evaluation, not the final geometry, because there is no
converged final geometry to describe. When the run failed before any member
completed an evaluation, every one of those fields is `null` and `flagged` is
also `null` — the second `null` case above. `n_steps` says how many steps the
trace actually holds.

### Mixed levels of theory

`committee.yaml` lets members mix levels of theory (e.g. an RPBE/OC20 head
alongside a PBE/OMat24 one). mliprun does not refuse this; it warns once, at
startup, because the level-of-theory table it checks against
(`src/mliprun/core/committee/config.py`) is deliberately incomplete: any
tag/task combination it does not recognise resolves to `unknown`, which also
counts as possibly mixed. Adding a row to that table is a deliberate act
that changes whether a committee is reported as same-level, not a way to
silence the warning. Every edit to the table bumps `LEVEL_TABLE_VERSION` in
that same file, and the value is stamped into the run record as
`provenance.committee.level_table_version`. That exists for the table's one
*silent* failure mode: an incomplete table announces itself through `unknown`,
but a **wrong** row does not, because two entries carrying the same label for
genuinely different datasets simply read as same-level. The stored version
lets a record written under a later-corrected table be re-judged rather than
trusted blindly. A mixed committee's spread is a comparison *between*
levels of theory, not an error bar on one of them: the probe behind this
feature measured `energy_spread_aligned_eV` at 0.363 eV across an RPBE and a
PBE member, against 0.001-0.019 eV between members sharing one level.
`mixed_theory=True` is stamped on every CSV row and into the run record
either way.

### The convergence plot

With `--plot`, a committee run's `<name>_convergence.png` gains a third
panel plotting `sigma_max_free` and `sigma_mean_free` per step, on the same fmax
reference line as the force panel. Its y-axis adapts to what is actually
being plotted rather than defaulting to log, because a plain log scale
silently drops non-positive values. It is linear, with an on-panel note,
when sigma is exactly zero at every step: an exactly-agreeing committee,
which does happen and must not be hidden by the axis choice. It is
`symlog` when some steps agree exactly and others don't, and log when every
value is a real, positive disagreement.

### The run record (committee fields)

`mliprun_run.json` gains the following, present only when a committee
actually ran (a single-model record is unchanged apart from the schema
number: see [The run record](#the-run-record) below). `provenance.committee`
and `provenance.committee_config_sha256` were added in schema 4;
`results.committee_uncertainty`'s content below is schema 5, amended by the
2026-09-08 sigma-masking change — a schema-4 record instead has a bare
`sigma_max_final_eV_per_A` (unmasked), a bare `worst_atom`, and a
`threshold_source` of `"fmax"` or `"explicit"`. The key names changed with
the meaning on purpose: a consumer that reads a schema-4 record for
`sigma_max_free_final_eV_per_A` gets a `KeyError`, not the wrong number.

- `provenance.committee`: one entry per member (`name`, `mlip`, `uma_task`,
  `mace_head`, `sevennet_task`, `env`, `python`, `device`, `gpu`,
  `level_of_theory`, `measured`), plus `levels` (the distinct level-of-theory
  labels present), `mixed_theory`, `level_table_version`, `config_sha256`, and
  `config_path`.
- **Declared vs measured.** Every member field except `measured` is what
  `committee.yaml` *declared*. `measured` is what that member's own
  environment *reported back* once its model had loaded: `python` (a version
  string, where the declared `python` is an interpreter path), `executable`,
  `mliprun`, `ase`, `torch`, `package` and `package_version`. Any individual
  entry can be `null` when the lookup failed there (a CHGNet-only env has no
  `torch`); `measured` itself is `null` when that member never started. The
  declared value is the intent and the measured value is the fact, and only
  the second one answers "which MACE version produced this number".
- `provenance.committee_config_sha256`: the same SHA-256, promoted to a flat
  field, so a later stage that ran a *different* committee shows up as a
  one-string diff without comparing the full member list.
- `provenance.device_requested` and `provenance.device_resolved` are both the
  literal string `"committee"`, not a torch device. The driver process
  resolves no device at all: by design it imports no torch (ADR 0001), so
  asking it would answer `"cpu"` even when every member is on its own GPU. The
  authoritative value is per member, in `provenance.committee.members[i]`
  (`device` and `gpu`).
- `results.committee_uncertainty`: `n_steps`, `threshold_eV_per_A`,
  `threshold_source` (`"explicit"` or `"none"` — `"fmax"` can no longer be
  produced, see [The flagging rule](#the-flagging-rule) above),
  `sigma_max_free_final_eV_per_A`, `sigma_mean_free_final_eV_per_A`,
  `sigma_max_all_final_eV_per_A` and `sigma_mean_all_final_eV_per_A` (the
  pre-masking pair, for comparison against runs from before 2026-09-08),
  `sigma_max_free_peak_eV_per_A`,
  `sigma_max_free_over_fmax_final` (`sigma_max_free_final_eV_per_A` divided
  by the final `fmax`, or `null` when there are no trace rows or the final
  `fmax` is exactly zero), `peak_step`,
  `worst_atom_free`, `worst_atom_free_symbol`, `n_free_atoms`,
  `unhandled_constraints`
  (list of constraint type names, empty when none are present),
  `energy_spread_aligned_final_eV`, `flagged` (tri-state: `true`, `false`, or
  `null` — see [The flagging rule](#the-flagging-rule) above).

---

## `optimize batch`

Relaxes a series of structures in one process, **loading the MLIP model only once** and reusing it across every relaxation (avoids the per-run model-load cost). Discovers one input structure per immediate subdirectory of `--parent` (default `--input-name '*.vasp'`, which expects exactly one `.vasp` file per subdir; the platform's own `*_final.vasp` outputs are ignored). Each structure is optimized in place, producing the same per-directory files as `optimize run` (`opt_final.vasp`, `CONTCAR`, etc.).

A structure that errors or fails to converge is logged and the batch continues. Pass `--skip-existing` to skip subdirectories that already contain a `CONTCAR` (resume a partial batch). `optimize batch` has no `--committee` option: committees are supported by `optimize run` only.

| File | Format | Contents |
|------|--------|----------|
| `<subdir>/...` | — | Same files as `optimize run`, one set per subdirectory |
| `<subdir>/mliprun_run.json` | JSON | Canonical run record for that subdirectory's relaxation. Every record from one `batch` invocation shares `run.batch.batch_id`. **Not** written into `--parent` — see [The run record](#the-run-record). |
| `batch_summary.csv` | CSV (in `--parent`) | columns: `subdir`, `status` (`converged` / `not_converged` / `error` / `no_input` / `skipped`), `converged`, `steps`, `energy_eV`, `walltime_s`, `detail` |

---

## `md run`

Always written:

| File | Format | Contents |
|------|--------|----------|
| `md.traj` | ASE trajectory (binary) | Every `interval` steps |
| `md_energy.csv` | CSV | columns: `step`, `time(fs)`, `temperature(K)`, `total_energy(eV)`, `potential_energy(eV)`, `kinetic_energy(eV)` |
| `md_params.txt` | plain text | Echo of run parameters |
| `mliprun_run.json` | JSON | Canonical run record: every resolved parameter with the source of its value, provenance (versions, device, host, timings), and per-stage outcome. A `--resume` appends a new `md-resume` stage rather than overwriting. See [The run record](#the-run-record). |

Only with `--plot` (plotting is opt-in; the CSV above is always written):

| File | Format | Contents |
|------|--------|----------|
| `md_energy.png` | PNG | Total / potential / kinetic energy vs time |
| `md_temperature.png` | PNG | Temperature vs time, with target line for NVT/NPT |

NPT-only extras (also require `--plot`):

| File | Format | Contents |
|------|--------|----------|
| `md_pressure.png` | PNG | Pressure vs time, target line shown |
| `md_volume.png` | PNG | Volume vs time |

NPT also adds `pressure(GPa)` and `volume(A^3)` columns to `md_energy.csv`.

---

## `neb run`

Endpoint optimization (only when `--optimize-endpoints` is enabled, the default):

| File | Format | Contents |
|------|--------|----------|
| `initial_opt.traj` / `final_opt.traj` | ASE trajectory | Endpoint relaxations |
| `initial_opt.log` / `final_opt.log` | text | ASE optimizer logs |
| `endpoint_optimization.txt` | plain text | Energies before/after, displacement summary |

IDPP interpolation (skipped in highly-constrained mode):

| File | Format | Contents |
|------|--------|----------|
| `idpp.traj` | ASE trajectory | IDPP iterations |
| `idpp.log` | text | IDPP log |

NEB run:

| File | Format | Contents |
|------|--------|----------|
| `A2B.traj` | ASE trajectory | Final NEB band (one frame per image) |
| `A2B_full.traj` | ASE trajectory | All NEB iteration steps; required for `--restart` |
| `neb.log` (or `--log` value) | text | Per-iteration log: step, fmax, current barrier |
| `neb_convergence.csv` | CSV | columns: `step`, `fmax(eV/A)`, `barrier(eV)` |
| `neb_convergence.png` | PNG | Two-panel: fmax vs step, barrier vs step — **only with `--plot`** |
| `neb_data.csv` | CSV | One row per image: index, energy, relative energy, force info |
| `neb_energy.png` | PNG | Smoothed energy profile across the band, barrier annotated — **only with `--plot`** |
| `neb_parameters.txt` | plain text | Echo of run parameters; required for `--restart` |
| `mliprun_run.json` | JSON | Canonical run record: every resolved parameter with the source of its value, provenance (versions, device, host, timings), and per-stage outcome. See [The run record](#the-run-record). |
| `00/POSCAR`, `01/POSCAR`, ... | VASP POSCAR | One directory per image, including endpoints |

Restart side-effect:

| File / folder | Notes |
|---------------|-------|
| `bkup_YYYY.MM.DD_HH.MM.SS/` | Created when `--restart` is used. The previous run's outputs (everything above plus `0N/` POSCAR folders) are *moved* into this folder; `neb_parameters.txt` is *copied* so the original record is preserved. `mliprun_run.json` is neither moved nor copied — it stays in place and gains a `neb-restart` stage, so one record spans the whole plain-NEB-then-CI-NEB history. |

---

## `autoneb run`

Endpoint optimization (only when `--optimize-endpoints` is enabled, the default): same files as for `neb run` (`initial_opt.*`, `final_opt.*`, `endpoint_optimization.txt`).

AutoNEB run:

| File / folder | Format | Contents |
|---------------|--------|----------|
| `autoneb000.traj`, `autoneb001.traj`, ... | ASE trajectory | One file per image; suffix is the image index (3-digit padded). The `autoneb` prefix can be changed with `--prefix`. |
| `AutoNEB_iter/` | folder of trajectories | Per-iteration history written by ASE's AutoNEB |
| `autoneb_parameters.txt` | plain text | Echo of run parameters |
| `mliprun_run.json` | JSON | Canonical run record: every resolved parameter with the source of its value, provenance (versions, device, host, timings), and per-stage outcome. See [The run record](#the-run-record). |

**Note:** `autoneb run` does *not* produce `neb_convergence.csv` / `neb_convergence.png` / `neb_data.csv` / `neb_energy.png`. To get an energy profile after AutoNEB finishes, run `autoneb-results results` (next section).

---

## `autoneb-results results`

| File | Format | Contents |
|------|--------|----------|
| `<prefix>_energy_profile.csv` | CSV | columns: `image`, `energy`, `rel_energy` |
| `<prefix>_energy_profile.png` | PNG | Spline-smoothed energy profile with TS annotated |
| `image_00/POSCAR`, `image_01/POSCAR`, ... | VASP POSCAR | Only when `--export-poscars` is set |

`<prefix>` defaults to `autoneb` and matches the prefix of the trajectory files this command reads.

---

## `benchmark run`

Stdout:

- A summary block per model (energy in eV, time in seconds), plus a final JSON summary block. Failed models are recorded as a string starting with `failed:` and the loop continues.

Optional file (only when `--output` is set):

| File | Format | Contents |
|------|--------|----------|
| `<--output path>` | JSON | `{model_tag: {"energy_eV": ..., "time_s": ...} or "failed: ..."}` |

---

## Parameter file conventions

`*_params.txt` and `*_parameters.txt` are written by `mliprun.core.params_io.write_parameters_file`. Same two-column layout (`{key:<23}{value}`) for every command. The keys include their trailing colon. These files are plain text and intended to be diffed across runs.

`endpoint_optimization.txt` (NEB / AutoNEB) is written by `write_endpoint_results` from the same module and has its own structured layout — see [PYTHON_API.md](PYTHON_API.md#parameter-file-io) if you need to consume it programmatically.

---

## The run record

Every command writes `mliprun_run.json` into its output directory. Unlike the
`*_params.txt` files (which are kept, and which NEB restart still parses), this
one file has the same schema for every command and is written by the core
layer — so a script that calls `run_optimization` directly gets one too.

### Top-level keys

| Key | Meaning |
|-----|---------|
| `schema_version` | Currently `5`. Check it before parsing. Version 2 added `provenance.uma_task` and `provenance.mace_head` (a version-1 record simply lacks those keys, which is not the same as null); version 3 added `provenance.sevennet_task`; version 4 added `provenance.committee` and `provenance.committee_config_sha256`, present only on a committee run (see [Committee outputs](#committee-outputs)); version 5 changed the *meaning* of `results.committee_uncertainty`'s reported disagreement (constrained force components excluded, see [Constraint masking](#constraint-masking)), renamed every sigma key so that meaning is on the key itself (`sigma_max_final_eV_per_A` → `sigma_max_free_final_eV_per_A`, and so on), and made `--uncertainty-threshold` opt-in (`threshold_source` is now `"explicit"` or `"none"`; `"fmax"` can no longer be produced). A schema-4 record predates all three changes. |
| `command` | `optimize`, `md`, `neb` or `autoneb`. |
| `status` | Status of the **latest** stage: `running`, `converged`, `not_converged` or `failed`. A record left saying `running` means the job died without reporting back. |
| `run.mode` | `one-off` or `batch`. |
| `run.batch` | `null` for one-off runs; otherwise `batch_id`, `driver`, `argv`, `root`, `config_file`. Every run of one batch shares a `batch_id`. |
| `inputs` | For `optimize` and `md`: structure filename and absolute path, atom count, formula. For `neb` and `autoneb`: `n_images` and `n_atoms` (there is no single input structure). |
| `parameters` | Every resolved parameter as `{"value": ..., "source": ...}`. |
| `provenance` | Versions (mliprun, ASE, the MLIP package), model **and the head/task it ran** (`uma_task` / `mace_head`), requested vs resolved device, Python, hostname, timestamps, wall time. Always describes stage 0's environment — see below. |
| `stages` | One entry per invocation in this directory. A NEB restart or MD resume **appends**. |

`provenance` is fixed at stage 0 and is never rewritten by a later stage.
`provenance.started_at` is stage 0's start time; `provenance.finished_at` is
the *latest* stage's completion. `provenance.walltime_s` is the **sum of every
stage's** `walltime_s` — compute actually spent, not the wall-clock span
between `started_at` and `finished_at`. A NEB restarted a week after stage 0
reports the two hours of compute the two stages took, not the week of
wall-clock gap in between.

#### The head/task fields

`provenance.uma_task` and `provenance.mace_head` name the head that actually
ran, so the record identifies the level of theory on its own — a model tag
alone does not, because the heads are independent fine-tunes with independent
energy zeros. At most one is ever non-null: each is recorded only when the
model tag belongs to its family (`uma-*` and `mace-mh-*` respectively), so a
MACE run never inherits the `--uma-task` its command line happened to carry.
Both are `null` for models with no head at all (CHGNet, SevenNet, plain
`mace`), and `null` also means "not determined" for a library caller that
passed neither.

Use these two fields before combining energies from different directories: an
energy produced under one head must not enter a formula with an energy
produced under another.

### Parameter sources

`source` is one of:

| Value | Meaning |
|-------|---------|
| `user` | Given on the command line (or in a config file). |
| `default` | Not given; the command's default applied. |
| `env` | Taken from an environment variable. |
| `prompt` | Typed at an interactive prompt. |
| `unspecified` | A library caller supplied no context. mliprun does not guess: a caller passing `fmax=0.05` explicitly is indistinguishable from one that omitted it. |

### Stages

`stages` is an array because a workflow can be several invocations in one
directory — most commonly a plain NEB followed by a CI-NEB restart, or an MD
run extended with `--resume`. Each stage records its own `kind`
(`optimize`, `md`, `md-resume`, `neb`, `neb-restart`, `autoneb`), `status`,
`steps`, `walltime_s`, any `parameters` that stage changed, and its `results`.
A stage's terminal status is never rewritten, so a converged stage 0 followed
by a failed stage 1 keeps both facts.

A stage carrying `"prior_history_unknown": true` was appended to a directory
with no readable prior record — an older run, or one whose record was damaged.

An appended stage that ran in a different environment than stage 0 — a
restart moved to another cluster, run after a version bump, or **switched to
a different head** — carries a `stage_provenance` object holding **only** the
fields that differ from the top-level `provenance`, drawn from
`mliprun_version`, `hostname`, `device_resolved`, `mlip_model`, `uma_task`
and `mace_head`. The key is omitted entirely when nothing differs, and stage 0
never carries it (there is nothing yet to compare it against).

A stage may also carry `stage_provenance_new_fields`. It means something
different, and the two never overlap:

| Key | Claim |
|-----|-------|
| `stage_provenance` | This field **changed**. Stage 0's value is in the top-level `provenance`; this stage's value is here. |
| `stage_provenance_new_fields` | This field is **new**. The stored record has no key for it at all, so stage 0's value was **not recorded** and is not knowable from this file. This stage's value is here. |

In practice `stage_provenance_new_fields` appears when a run started under
schema 1 (before `uma_task`/`mace_head` existed) is resumed by a current
mliprun. Read it as "the head is now *X*; what stage 0 ran is unrecorded" —
**not** as "the head changed to *X*". The `*_params.txt` file written
alongside the record by stage 0 is the remaining evidence of what stage 0
actually used; check it before combining the two stages' energies.

Such a record is a hybrid: `schema_version` stays at the value stage 0 wrote
(`1`), while the appended stage carries a schema-2 key. The version is
deliberately not rewritten — bumping it would assert that stage 0's
provenance was collected under schema 2, which it was not. Parse defensively:
treat `schema_version` as describing the top-level `provenance` block, and
read each stage's own keys for what that stage recorded.

### Results by command

**optimize** — `converged`, `final_energy_eV`, `final_fmax_eV_per_A`.

**md** — `n_samples`, `mean_temperature_K`, `std_temperature_K`,
`mean_total_energy_eV`, `std_total_energy_eV`, `mean_potential_energy_eV`,
`std_potential_energy_eV`, `total_energy_drift_eV_per_atom_per_ps`, and
`decile_mean_total_energy_eV` (ten block means over the segment, for judging
equilibration by eye). NPT adds `mean_pressure_GPa` and `mean_volume_A3`.
Statistics describe **that stage's segment only**, not the whole trajectory.

There is deliberately no equilibration-window detection and no production
average: choosing a production region is a methodological decision, and MD
samples are autocorrelated, so a naive standard-error criterion truncates too
early and understates uncertainty.

**neb / autoneb** — `forward_barrier_eV`, `reverse_barrier_eV`,
`reaction_energy_eV`, `ts_image_index`, `n_images`, and `ts_at_endpoint`. The
last is a sanity flag: `true` means the energy maximum sits on the first or
last image, so no saddle was bracketed and the "barrier" is not a transition
state. `neb` records forward **and** reverse barriers (the old convergence log
recorded only the forward one).

`final_fmax_eV_per_A` is recorded for `neb` stages only (plain and
`--restart` alike). `autoneb` stages omit it — ASE's `AutoNEB.run()` doesn't
expose a comparable per-iteration fmax through this path — and also omit
`steps`. `autoneb`'s stage status is always `converged` on success, since
`AutoNEB.run()` either completes or raises; there is no `not_converged`
outcome for this command, only `converged` or `failed`.

### Failure behavior

A record write can never abort a run. Writes are atomic (serialize, then
replace), so a crash never leaves truncated JSON. An unparseable record found
on restart is moved to `mliprun_run.json.corrupt-<timestamp>` and a fresh one
started. `NaN` and `Inf` are written as `null`, since neither is valid JSON.
