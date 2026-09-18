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
| `singlepoint run` | Directory containing `--structure`, unless `--output-dir` is given |
| `freq run` | Directory containing `--structure`, unless `--output-dir` is given |

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

These also follow `--logfile <name>.log`: `<name>_committee.csv`, `<name>_committee_peratom.csv`. With `--plot`, `<name>_convergence.png` gains a third panel; with `--uncertainty-plot`, a separate `<name>_uncertainty.png` is written (see [Committee outputs](#committee-outputs)).

---

## Committee outputs

A committee is two or more MLIPs, each in its own environment, relaxing the
same structure together (`optimize run --structure POSCAR --committee
committee.yaml`). The relaxation follows their **mean** force; the outputs
below report how much they **disagree**, as a diagnostic. Force disagreement
(every `sigma_*` column) is in eV/Å; energy disagreement
(`energy_spread_aligned_eV`) is in eV. It is
never a substitute for DFT validation. The per-step trace
(`<name>_committee.csv`) and the convergence/uncertainty plots documented
below exist only for a relaxation, so they are `optimize run`-only; not
supported by `optimize batch`, `md`, or `neb`/`autoneb` at all.
`singlepoint run --committee` (see [`singlepoint run`](#singlepoint-run))
shares the two outputs that do not depend on there being a trajectory —
`<name>_committee_peratom.csv` and `results.committee_uncertainty` —
evaluated at the one structure it was given rather than a relaxation's final
geometry.

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

The same caveat applies to `sigma_max_free_over_fmax_final`, and it applies
on *every* `--relax-cell` run, not only a flagged one: the ratio is printed
and recorded unconditionally, and on a cell relaxation its denominator
carries virials its numerator does not. It is still useful as a trend across
comparable runs; it is not "sigma in units of the force tolerance" there.

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

### `<name>_uncertainty.png`

Written only with `--uncertainty-plot`, which requires `--committee` and is
independent of `--plot` — ask for either, both or neither. One panel, two
y-axes, against the optimizer step:

| Element | Axis | Meaning |
|---------|------|---------|
| Energy trace | left | `energy_mean_eV` minus its own step-0 value |
| Energy band | left | ± `energy_spread_aligned_eV` |
| Force trace | right | `fmax_eV_per_A` |
| Force band | right | `fmax` ± `sigma_max_free_eV_per_A`, lower edge clipped at 0 |
| Dashed red line | right | the `--fmax` convergence target |

The point of the single panel is the comparison the target line makes
possible: whether the force is converging into the committee's own
disagreement. On the measured same-level runs it is, which is what the
uncalibrated threshold discussion is about (see [The flagging
rule](#the-flagging-rule)).

Three things about it are deliberate:

- **The energy is plotted relative to step 0** because the band is measured
  that way — `energy_spread_aligned_eV` removes each member's own offset
  first. An absolute committee energy carries a per-package offset of tens of
  eV while the band is ~0.01 eV wide, so plotting the raw mean would render
  the band as a line. A consequence the figure annotates rather than hides:
  the band has exactly zero width at step 0, by construction and not by
  agreement.
- **The force band is an upper bound, not the error bar on the plotted
  number.** `sigma_max_free` is the largest disagreement anywhere in the free
  region, and the atom carrying it need not be the atom carrying `fmax`.
- **The force axis is logarithmic; the energy axis is linear.** A relaxation
  spans orders of magnitude in `fmax`, so a linear force axis buries every
  step after the first few. Energy relative to step 0 is negative whenever the
  relaxation went downhill, so that axis can only be linear.
- **The force axis becomes `symlog` when a band edge is clipped to zero**, on
  the same reasoning as the sigma panel: matplotlib drops non-positive
  vertices from a log axis with no warning, which does not merely hide the
  clipped edge but deforms the whole band polygon. `symlog` treats
  `|y| <= linthresh` linearly and keeps the edge where it belongs, with the
  axis floored at zero so the scale's symmetric negative half — decades of
  negative force — never appears. A clipped edge means sigma exceeded `fmax`
  at that step, which is exactly the case worth seeing.

No figure is written when the trace has fewer than two steps (a run that
converged at step 0); the run says so on the terminal rather than emitting a
one-point plot.

### Committee frequencies

`freq run --committee committee.yaml` (see [`freq run`](#freq-run) below)
gets one Hessian per member from a **single** displacement sweep:
`CommitteeVibrations` captures each member's own forces at every
displacement alongside the mean forces ASE already sees, so a per-member
Hessian is assembled from data the sweep was already computing. Cost is
therefore **one sweep, not one per member** — members are queried
concurrently, exactly as in a committee relaxation, so wall time is the
slowest member's, not the sum of all of them. This is measured on EMT only
and not yet verified against real MLIP potentials.

Two caveats belong here because without them a spread reads as a clean
number instead of the qualified one it is:

- **Mode pairing is by index.** Each member's Hessian is diagonalized
  independently and its eigenvalues come back sorted ascending, so for
  near-degenerate modes member A's mode 7 and member B's mode 7 need not be
  the same physical mode — comparing them by index is then not a
  like-for-like comparison, and the per-mode standard deviation would
  silently absorb an ordering swap as disagreement. The `<member>_overlap`
  columns in `<prefix>_committee_frequencies.csv` are how that becomes
  visible: for each member and each mode, the absolute overlap
  `|⟨u_member,i | u_committee,i⟩|` between that member's mode vector and the
  committee's, each **normalised to unit Cartesian length first** (ASE's own
  mode vectors are unit-normalised in the *mass-weighted* basis instead, not
  in Cartesian space, so the raw dot product would be mass-dependent rather
  than a clean-match indicator). Close to 1 is a clean match; when any
  overlap falls below 0.9 the run warns, naming the modes, and
  `mode_pairing_suspect` is set `true` in the run record. 0.9 is a
  diagnostic trigger for a warning, not a scientific verdict — the overlaps
  themselves are in the CSV for anyone who disagrees with it.
  **Expect this flag to be noisy, and treat it as untested against real
  potentials.** It takes its minimum over *every* mode, near-zero ones
  included, and the eigenvectors of a near-zero frustrated translation or
  rotation are an arbitrary basis that differs freely between members — so
  the flag may well fire on runs where nothing is wrong. It has been
  exercised against EMT only. Read `worst_mode_overlap` and the per-mode
  `<member>_overlap` columns next to the frequencies before acting on the
  flag: an overlap that is low only for modes at a few cm⁻¹ says nothing
  about the modes you are reporting. Whether to apply a frequency floor
  below which the diagnostic is skipped is an open question for the project
  owner, not something this command decides.
- **At `delta = 0.01 Å` (the default), a merely noisy member contributes to
  the spread alongside genuine model disagreement, and one sweep cannot
  separate the two.** The force differences being divided are small at that
  displacement, and MLIP forces carry their own numerical noise at that
  scale. Running at two values of `--delta` distinguishes noise from
  disagreement. Do not present `frequency_member_std_cm-1` as pure model
  uncertainty without checking that.

`<prefix>_committee_frequencies.csv`:

| Column | Meaning |
|--------|---------|
| `mode_index` | As in `<prefix>_frequencies.csv` |
| `frequency_committee_cm-1` | The headline value, from the **mean** forces — one Hessian, not the mean of the per-member frequencies below (those are different numbers, D8 in the design note) |
| `imaginary` | bool, for the committee (headline) value |
| `<member>_cm-1` | One column per member, magnitude |
| `frequency_member_std_cm-1` | Standard deviation across members, `ddof=1` |
| `<member>_overlap` | One column per member — see above |

Per-member ZPE, plus its mean and standard deviation across members, is in
the run record's `results.committee_frequencies`, not in the CSV:
`zpe_eV_per_member` (one value per member), `zpe_mean_eV`, `zpe_std_eV`,
`frequency_member_std_cm-1` (the same values as the CSV column),
`worst_mode_overlap`, and `mode_pairing_suspect`.

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

A structure that errors or fails to converge is logged and the batch continues. Pass `--skip-existing` to skip subdirectories that already contain a `CONTCAR` (resume a partial batch). `optimize batch` has no `--committee` option.

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

## `singlepoint run`

Evaluates a structure once and stops: energy, per-atom forces, and (when the
cell allows it) stress. No optimizer, no trajectory, no relaxed structure.
Replaces the `optimize run --max-steps 0` workaround, which performed the
same single evaluation but reported it as a failed relaxation — `status:
not_converged`, the "increase max_steps" advice block, a trajectory and a
`CONTCAR` for a geometry that never moved, and no per-atom forces anywhere.

| File | Format | Contents |
|------|--------|----------|
| `<prefix>_forces.csv` | CSV | One row per atom: the raw forces the model predicts, plus the free-component mask |
| `mliprun_run.json` | JSON | Canonical run record; stage kind `singlepoint` (see [The run record](#the-run-record)) |

With `--committee committee.yaml`, one more file is always written:

| File | Format | Contents |
|------|--------|----------|
| `<prefix>_committee_peratom.csv` | CSV | Per-atom committee disagreement at this one configuration — same layout and columns as `optimize run`'s `<name>_committee_peratom.csv` (see [Committee outputs](#committee-outputs)) |

`<prefix>` defaults to `singlepoint` and follows `--prefix`.

| Option | Default | Meaning |
|--------|---------|---------|
| `--output-dir` | `Path(--structure).parent` | Directory all of the above files are written into. Created if missing, including intermediate parents. **Why it exists:** `mliprun_run.json` is replaced wholesale by the next command that writes in the same directory — running `optimize` then `singlepoint` in one directory silently drops the optimize stage from the record, with no error and no warning. Point `--output-dir` at a separate folder (e.g. `sp/`) to keep a single-point evaluation from destroying the record of the optimization that produced its input structure. |

### `<prefix>_forces.csv`

| Column | Meaning |
|--------|---------|
| `index` | Index into the structure |
| `symbol` | Chemical symbol |
| `fx`, `fy`, `fz` | Raw force components (eV/Å) |
| `f_norm` | Magnitude of the raw force vector |
| `free_x`, `free_y`, `free_z` | Whether that component is free (`True`) or held by a masked constraint (`False`) |

**These are the raw forces, not the constrained ones.** `atoms.get_forces()`
zeroes whatever component a constraint holds, and a CSV of zeros on a fixed
layer says nothing about what the model actually predicts there. This file
bypasses the constraint machinery (`atoms.calc.get_forces(atoms)`) and writes
what the calculator returns; `fmax_free_eV_per_A` below is computed from the
constrained forces instead, because that is the number a relaxation would
converge against. The mask columns say which rows `fmax_free` covers.

### `results`

| Key | Meaning |
|-----|---------|
| `energy_eV` | Potential energy |
| `fmax_free_eV_per_A` | Max force magnitude from `atoms.get_forces()` — constraints applied, the population an optimizer would converge against |
| `fmax_all_eV_per_A` | Max force magnitude from `atoms.calc.get_forces(atoms)` — no constraints applied, what the model predicts before anything is held fixed |
| `n_free_atoms` | How many atoms retain at least one free force component |
| `worst_force_atom_all` | Index of the atom with the largest raw (unconstrained) force |
| `worst_force_atom_all_symbol` | Its chemical symbol |
| `worst_force_atom_free` | Index of the atom with the largest constrained force, or `null` when every atom is fixed — there is then no free atom to be the worst one |
| `worst_force_atom_free_symbol` | Its chemical symbol, `null` alongside a `null` index |
| `unhandled_constraints` | Constraint type names left unmasked (see [Constraint masking](#constraint-masking)); their atoms count as free, so `fmax_free` over-reports for them |
| `stress_eV_per_A3`, `stress_GPa` | Voigt-order stress tensor, or both `null` when not attempted or not available |
| `stress_unavailable_reason` | Why stress is `null`: `"not requested"` (`--no-stress`), a pbc message (see [Stress](#stress) below), or the calculator's own exception; `null` when stress was reported |
| `committee_uncertainty` | Present only with `--committee`: the same block `optimize run` writes, evaluated at this one configuration rather than a relaxation's final geometry (see [Committee outputs](#committee-outputs)) |

**Two fmax values, and they answer different questions — never average them
or treat them as redundant.** `fmax_free_eV_per_A` comes from
`atoms.get_forces()`, which applies the structure's constraints; it is the
number an optimizer would actually converge against, and the one comparable
to a relaxation's `final_fmax_eV_per_A`. `fmax_all_eV_per_A` comes from the
calculator directly, bypassing constraints entirely; it is what the model
predicts before anything is held fixed, and is normally the larger of the
two on a constrained structure. They can (and often do) name different
atoms as the worst offender — see the next point.

**`worst_force_atom_free` and `committee_uncertainty.worst_atom_free` are
two different atoms answering two different questions.**
`worst_force_atom_free` is the atom carrying the **greatest force** among
the free ones — a statement about the model's own prediction at this
geometry. `committee_uncertainty.worst_atom_free` is the atom carrying the
**greatest disagreement between committee members** — a statement about how
much the members disagree, not about which force is largest. Nothing ties
these together, and on a real structure they are usually not the same atom.

### Stress

Attempted only when the cell is periodic in all three directions
(`pbc=(True, True, True)`): a slab's stress along the vacuum direction is
not a physical quantity, and reporting a number there invites it to be
used. `--stress` forces the attempt regardless of `pbc`; `--no-stress` skips
it unconditionally. A calculator that does not implement stress at all
(`PropertyNotImplementedError`) is recorded in `stress_unavailable_reason`
and never aborts the run — a missing stress must not cost the energy.

---

## `freq run`

Computes vibrational frequencies by finite differences of forces
(`ase.vibrations.Vibrations`) and reports the frequencies, the imaginary-mode
count, and the zero-point energy (ZPE). With `--committee committee.yaml`,
one displacement sweep additionally yields one Hessian per member: see
[Committee frequencies](#committee-frequencies) above.

| File | Format | Contents |
|------|--------|----------|
| `<prefix>_frequencies.csv` | CSV | One row per mode: magnitude, energy, imaginary flag (see below) |
| `<prefix>_summary.txt` | text | ASE's own `vib.summary()` table — the format users already recognise from other ASE-driven work |
| `<prefix>_vibrations.json` | JSON | `VibrationsData.write()` output: the full Hessian and the atoms. Reloads through `VibrationsData.read` (see [PYTHON_API.md](PYTHON_API.md#vibrational-frequencies)) |
| `<prefix>/` | folder | ASE's per-displacement JSON cache. An interrupted sweep resumes at the displacement it stopped on. The cache is keyed by displacement, not by model, so a reusing run first verifies it belongs to that run's calculator — see [The displacement cache](#the-displacement-cache-and-what-it-is-checked-against) |
| `<prefix>.<n>.traj` | ASE trajectory | One animated trajectory per written mode, `n` its mode index. Which modes get one follows `--write-modes` (`none`, `imaginary` — the default, or `all`); a clean minimum under the default writes nothing |
| `mliprun_run.json` | JSON | Canonical run record; stage kind `freq` (see [The run record](#the-run-record)) |

With `--committee committee.yaml`, one more file is always written:

| File | Format | Contents |
|------|--------|----------|
| `<prefix>_committee_frequencies.csv` | CSV | Per-member frequencies, the per-mode standard deviation across members, and the per-member mode-overlap diagnostic — see [Committee frequencies](#committee-frequencies) above |

`<prefix>` defaults to `freq` and follows `--prefix`.

**Give a frequency run its own output directory.** A run record is
*replaced*, not appended to, by the next command that writes into the same
directory — measured: `optimize run` followed by `singlepoint run` in one
directory leaves the `optimize` stage gone from `mliprun_run.json`. `freq`
is no different, so running it in the directory that produced the structure
risks losing that structure's `optimize` provenance the next time something
writes there. `--output-dir` puts a frequency run's outputs in their own
folder instead (default: next to `--structure`, as `optimize`/`singlepoint`/
`md` do). This only moves where *this run's* files land — the
stationary-point warning below still looks up the fmax expectation in the
*structure's own* directory regardless of `--output-dir`, so the warning
keeps working.

### `<prefix>_frequencies.csv`

| Column | Meaning |
|--------|---------|
| `mode_index` | 0-based, in ASE's ascending-eigenvalue order |
| `frequency_cm-1` | **magnitude**, always positive |
| `energy_meV` | The same mode's energy, in meV — a **magnitude** like `frequency_cm-1`, positive for an imaginary mode too |
| `imaginary` | bool |

**The frequency column is a magnitude plus a boolean, never a signed
number.** Writing an imaginary frequency as a negative one is the widespread
convention elsewhere, and it is a silent trap here: anything that sums or
sorts this column would treat an imaginary mode as an unusually soft real
one rather than flagging it. `energy_meV` follows the same rule: it is the
same mode's energy magnitude, so the two numeric columns on a row always
describe the same mode in two units (`energy_meV = frequency_cm-1 ×
ase.units.invcm × 1000`), imaginary rows included.

A mode counts as imaginary when `abs(energy.imag) > 1e-8` eV — the same
threshold ASE's own `im_tol` uses in
`VibrationsData._tabulate_from_energies`, applied to the same quantity (the
mode **energy**, never the frequency in cm⁻¹). This alignment matters
because `freq` writes both this CSV and ASE's own `<prefix>_summary.txt`
from the same run; a mismatched threshold, or the same threshold applied to
a different quantity, would let the two files disagree about which modes
are imaginary. It does **not** settle whether a *larger* tolerance should
suppress genuine near-zero modes — a frustrated translation or rotation on
a slab can pick up an arbitrary tiny sign from finite differences, so a mode
at a few cm⁻¹ may be numerical noise rather than real negative curvature.
That is an open scientific question (recorded in the design note), and it
bears directly on any "exactly one imaginary mode" transition-state check
built on this output.

**ZPE counts the real modes only.** ASE's zero-point energy sums the real
parts of the mode energies, so an imaginary mode contributes exactly zero to
it. A structure with imaginary modes therefore has **no well-defined
zero-point energy**, and `zpe_eV` in the run record is the ZPE of its real
modes only.

### The stationary-point warning

A frequency analysis assumes the input geometry sits at a stationary point.
`freq` measures fmax at the input geometry — at no extra cost, since
`Vibrations.run()` evaluates the undisplaced geometry first, before any
displacement — and compares it against an expectation, in order:

1. `--expect-fmax X`, when given (`fmax_expectation_source: "explicit"`).
2. Otherwise, the fmax a **converged** `optimize` stage in the *structure's
   own directory* actually met, read from that directory's
   `mliprun_run.json` (`fmax_expectation_source: "run_record"`). This is not
   an invented constant: it is the convergence criterion that was actually
   applied to this structure, with its provenance attached. A
   not-converged `optimize` stage supplies nothing — the fmax it was aiming
   at is not one it met.
3. Otherwise, no comparison (`fmax_expectation_source: "none"`,
   `fmax_warning: null`). The measured fmax is still reported, with a line
   saying no relaxation provenance was found next to the structure.

**Two fmax values are reported, and the comparison uses the free one.**
`fmax_at_input_free_eV_per_A` covers only the force components no constraint
holds, exactly as `singlepoint`'s `fmax_free_eV_per_A` does; it is what the
expectation above is measured against, because that expectation is the
*constrained* criterion an `optimize` stage converged to.
`fmax_at_input_all_eV_per_A` is the same forces with no mask applied — what
the model predicts before anything is held fixed. On a slab with frozen
layers they differ by an order of magnitude (measured on a relaxed Pt(111)
2×2×4 + H slab: 0.0198 eV/Å free against 0.3809 eV/Å over all atoms), so
comparing the all-atom number against an `optimize` record's fmax would
raise the warning on every correctly relaxed slab. The masking follows
[Constraint masking](#constraint-masking): a projecting constraint
(`FixedPlane`, `FixedLine`, …) is left unmasked, so the free value
over-reports for those atoms and their type names appear in
`unhandled_constraints`.

**This warning never refuses.** Exceeding the expectation prints a warning
and sets `fmax_warning: true`; the run completes regardless. A geometry that
is not a stationary point produces spurious imaginary modes that look, by
eye, indistinguishable from a real transition state — the warning is the
only defence against that, and it is only ever advisory.

The fmax-expectation lookup always reads the *structure's own* directory
(`Path(--structure).parent`), never `--output-dir`: keeping a frequency
run's outputs in their own folder (see above) does not disconnect the
warning from the `optimize` stage that produced the structure.

### Which atoms move

Every atom **not** held by a `FixAtoms` constraint — the structure's own
answer, and also ASE's own default for `Vibrations`. `--indices` overrides
it explicitly (accepts `0,1,5`, `12-30`, or a mix, inclusive ranges).

Any other constraint type (`FixCartesian`, `FixedPlane`, `FixedLine`,
`FixBondLength`/`FixBondLengths`, `FixScaled`, …) **warns and is displaced
in full**: ASE's `indices` selects whole atoms, so a partially-held atom has
no partial Hessian to express through it. Those atoms' held components then
enter the Hessian as if free. The warned-about type names are recorded in
`unhandled_constraints`; pass `--indices` to exclude those atoms explicitly
if that is not acceptable.

### Cost

Force calls: `1 + 6 × n_displaced` at `--nfree 2` (the default), `1 + 12 ×
n_displaced` at `--nfree 4` (five-point stencil, doubling the cost). The
leading `1` is the equilibrium evaluation. Restart is nearly free — an
interrupted sweep resumes only the displacements not already in the
`<prefix>/` cache, and pays one extra force evaluation for the cache check
described next.

### The displacement cache, and what it is checked against

The `<prefix>/` folder is ASE's own per-displacement JSON cache. It is keyed
by the displacement, **not** by the model: its name is
`<output_dir>/<prefix>`, and `--prefix` defaults to `freq` whatever
`--mlip` says. A second `freq run` in the same directory with a *different*
potential would therefore reuse the first potential's forces and report them
under its own provenance — a wrong number carrying a false attribution.
Measured before this check existed: EMT then Lennard-Jones on N₂ in one
directory gave run 2 zero force calls, EMT's 928.1448 cm⁻¹ top frequency,
and `provenance.mlip_model: "lj"`.

So whenever a run reuses anything from the cache, it re-evaluates the
undisplaced geometry once with its own calculator and compares that against
the cached equilibrium forces. The comparison is `numpy.allclose(rtol=0,
atol=1e-6)` in eV/Å, not exact equality: a real MLIP on a GPU is not
bit-reproducible between runs, while a different model differs by orders of
magnitude (~1 eV/Å for the EMT/Lennard-Jones pair above), so that tolerance
separates the two cases cleanly. On a mismatch the run **stops** with an
error naming the cache directory, the run record is completed as `failed`,
and the message gives both remedies: delete the `<prefix>/` folder, or pass
a different `--prefix` so this run gets its own cache. The check costs one
force evaluation on a restart, against the `6 × n_displaced` a restart
saves, and it is not counted in `n_force_calls` (which reports the sweep's
own cost).

**A committee run can only restart a committee sweep.** A committee restart
is otherwise as cheap as a single-model one — the per-member forces survive
ASE's JSON cache, so a resumed committee run keeps its spread and still
reports `committee_uncertainty` (that block never comes from the cache; it
comes from one explicit evaluation of the input geometry after the sweep).
But a cache written by a **single-model** run holds the consensus forces
only: there are no per-member forces in it for a committee to build one
Hessian per member from. `freq run` followed by `freq run --committee` in
one directory — both default to the same directory and the same prefix — is
therefore refused by the same check, with the same two remedies, rather
than crashing partway through with the run record left saying `running`.

### `freq run --committee`

A committee run reports **two different things**, not one:

1. **Per-member frequencies** — one Hessian per member from a single
   displacement sweep, in `<prefix>_committee_frequencies.csv` and
   `results.committee_frequencies`. This is `freq`'s own mechanism; see
   [Committee frequencies](#committee-frequencies) above.
2. **The consensus force disagreement** — the same `committee_uncertainty`
   block `optimize run --committee` and `singlepoint run --committee` write,
   with the same keys and the same meaning (see [Committee
   outputs](#committee-outputs)). `freq` measures it **at the input
   geometry of the sweep**, not at a displaced one: `Vibrations.run()`
   restores the positions after every displacement, so the members are
   evaluated once, explicitly, at the geometry the frequencies describe.
   That one evaluation is also why a fully cached restart still reports this
   block, with the same numbers as a fresh run.

`--uncertainty-threshold X` (eV/Å) is opt-in here exactly as it is on the
other two commands: there is **no default**, and without it sigma is
reported and no verdict asserted (`threshold_source: "none"`,
`flagged: null`). With it, `sigma_max_free` above `X` sets `flagged: true`
and prints a warning saying the configuration deserves a DFT check. It
never stops the run, and it is independent of `--expect-fmax`, which
answers a different question (is this geometry a stationary point?) against
a different quantity.

The terminal echo after a committee run names the geometry it describes —
"Committee disagreement at the input geometry" — because this is the same
shared reporter `optimize` uses at a relaxation's final geometry.

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
| `schema_version` | Currently `5`. Check it before parsing. Version 2 added `provenance.uma_task` and `provenance.mace_head` (a version-1 record simply lacks those keys, which is not the same as null); version 3 added `provenance.sevennet_task`; version 4 added `provenance.committee` and `provenance.committee_config_sha256`, present only on a committee run (see [Committee outputs](#committee-outputs)); version 5 changed the *meaning* of `results.committee_uncertainty`'s reported disagreement (constrained force components excluded, see [Constraint masking](#constraint-masking)), renamed every sigma key so that meaning is on the key itself (`sigma_max_final_eV_per_A` → `sigma_max_free_final_eV_per_A`, and so on), and made `--uncertainty-threshold` opt-in (`threshold_source` is now `"explicit"` or `"none"`; `"fmax"` can no longer be produced). A schema-4 record predates all three changes. The `singlepoint` and `freq` stage kinds are additive and do not bump the schema: `provenance` is untouched, and no existing field changes meaning. |
| `command` | `optimize`, `md`, `neb`, `autoneb`, `singlepoint` or `freq`. |
| `status` | Status of the **latest** stage: `running`, `converged`, `not_converged`, `completed` (a `singlepoint` or `freq` stage: there is nothing to converge) or `failed`. A record left saying `running` means the job died without reporting back. |
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
(`optimize`, `md`, `md-resume`, `neb`, `neb-restart`, `autoneb`,
`singlepoint`, `freq`), `status`, `steps`, `walltime_s`, any `parameters`
that stage changed, and its `results`.
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

**singlepoint** — `energy_eV`, `fmax_free_eV_per_A`, `fmax_all_eV_per_A`,
`n_free_atoms`, `worst_force_atom_all`, `worst_force_atom_all_symbol`,
`worst_force_atom_free`, `worst_force_atom_free_symbol`,
`unhandled_constraints`, `stress_eV_per_A3`, `stress_GPa`,
`stress_unavailable_reason`, and `committee_uncertainty` with `--committee`.
Status is always `completed` on success — there is nothing to converge — or
`failed`. See [`singlepoint run`](#singlepoint-run) above for what each key
means.

**freq** — `n_modes`, `n_imaginary`, `frequencies_cm-1` (list, magnitudes),
`imaginary_mask` (list, bool), `zpe_eV` (real modes only, see [`freq
run`](#freq-run) above), `fmax_at_input_free_eV_per_A`,
`fmax_at_input_all_eV_per_A`, `fmax_expectation`,
`fmax_expectation_source`, `fmax_warning`, `n_displaced_atoms`,
`n_force_calls`, `unhandled_constraints`, and — with `--committee` — both
`committee_frequencies` (see [Committee
frequencies](#committee-frequencies) above) and `committee_uncertainty`
(the same consensus-disagreement block `optimize` and `singlepoint` write,
here measured at the **input geometry** of the displacement sweep; see
[`freq run --committee`](#freq-run---committee) below). Status is always
`completed` on success — there is nothing to converge — or `failed`.

### Failure behavior

A record write can never abort a run. Writes are atomic (serialize, then
replace), so a crash never leaves truncated JSON. An unparseable record found
on restart is moved to `mliprun_run.json.corrupt-<timestamp>` and a fresh one
started. `NaN` and `Inf` are written as `null`, since neither is valid JSON.
