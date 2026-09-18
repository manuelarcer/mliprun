# Single-point and frequency analysis — design

Date: 2026-09-17
Status: approved in chat, pending spec review
Scope: two new commands, `mlip singlepoint` and `mlip freq`, both with
committee support.

## Summary

`mliprun` can relax a structure, run MD on it, and find a transition state
between two of them. It cannot evaluate one structure and stop, and it cannot
say anything about curvature. This design adds both.

**`singlepoint`** evaluates one structure once and reports energy, per-atom
forces, fmax and stress. This is reachable today only by abuse:
`optimize run --max-steps 0` performs exactly one force evaluation (verified
against ASE 3.29 — one callback, `nsteps=0`), but it reports `converged=False`,
prints the "increase max_steps" advice block, writes a trajectory and a
`CONTCAR`, records `status: "not_converged"`, and never writes per-atom forces
anywhere. The record therefore misdescribes what happened. This command is
about a correct label and a correct output shape, not new capability.

**`freq`** computes vibrational frequencies by finite differences of forces,
using `ase.vibrations.Vibrations`, and reports frequencies, the imaginary-mode
count, and the zero-point energy. With a committee it additionally reports each
member's own frequencies and the per-mode spread across members.

## Decisions

Recorded here because several were Juan's and are not re-openable by an
implementer.

| # | Decision | Who, when |
|---|---|---|
| D1 | The second method is **committee frequencies** — per-member Hessians from one displacement sweep. VACF power spectra from MD and `ase.phonons` are both rejected for this work. | Juan, 2026-09-17 |
| D2 | Which atoms are displaced comes from the **input structure's constraints**. `--indices` overrides. | Juan, 2026-09-17 |
| D3 | **No thermochemistry in this work.** Frequencies and ZPE only. The design must leave free energies reachable without recomputing forces. | Juan, 2026-09-17 |
| D4 | **Committee support on `singlepoint`**: yes. | Juan, 2026-09-17 |
| D5 | A structure that is not at a stationary point produces a **warning, never a refusal**. | Juan, 2026-09-17 |
| D6 | A constraint type other than `FixAtoms` produces a **warning, never a refusal**. | Juan, 2026-09-17 |
| D7 | Command names `singlepoint` and `freq`, each also a standalone console entry point, matching `optimize`/`md`/`neb`. | Juan, 2026-09-17 |
| D8 | The headline committee frequency is the frequency of the **mean potential** (one Hessian built from the mean forces), with per-member frequencies reported beside it. It is not the mean of the members' frequencies; those are different numbers. | Claude, 2026-09-17, unopposed |

## Command 1 — `singlepoint`

### CLI surface

`mlip singlepoint run`, plus a standalone `singlepoint` console script.

| Option | Default | Meaning |
|---|---|---|
| `--structure` | required, prompted | Input structure, any format `ase.io.read` accepts |
| `--mlip` | `auto` | MLIP tag, as elsewhere |
| `--uma-task` / `--mace-head` / `--sevennet-task` | none | As elsewhere; the same `validate_mlip` rules apply |
| `--device` | `auto` | As elsewhere |
| `--committee` | none | Path to a `committee.yaml`. Mutually exclusive with the model-selection options above, enforced by parameter source exactly as `optimize run` does |
| `--member-timeout` | `DEFAULT_CALC_TIMEOUT_S` | As `optimize run` |
| `--uncertainty-threshold` | none | As `optimize run`: opt-in, no default, `flagged` is null without one |
| `--prefix` | `singlepoint` | Stem for this command's output files |
| `--stress / --no-stress` | attempt when `all(atoms.get_pbc())` | Ask the calculator for the stress tensor |

Outputs are written next to the input structure, as `optimize` and `md` do.

### Behaviour

1. Read the structure, attach the calculator (or start the committee, with the
   same `ExitStack` / SIGTERM teardown guarantees `optimize run` has).
2. One evaluation. Energy from `atoms.get_potential_energy()`.
3. Two force arrays, both reported:
   - **constrained** — `atoms.get_forces()`, which applies the structure's
     constraints, zeroing held components. `fmax_free_eV_per_A` is
     `calc_fmax` of this array. It is what an optimizer converges against.
   - **raw** — `atoms.calc.get_forces(atoms)`, which bypasses the constraint
     machinery. `fmax_all_eV_per_A` is `calc_fmax` of this array, and this is
     what the per-atom CSV carries, because it is what the model actually
     predicts.

   Two numbers, never one. This follows the `_free` / `_all` naming discipline
   settled for committee sigma in the 2026-09-08 design: the population a
   statistic covers belongs on its key, not in the reader's memory.
4. Stress when asked for and available. A calculator that does not implement
   it raises `PropertyNotImplementedError`; that is caught, `stress_*` fields
   are recorded null, and `stress_unavailable_reason` carries the exception
   text. A cell that is not periodic in all three directions skips the attempt
   with that as the recorded reason — a slab at `pbc=(True, True, False)` has
   a stress component along the vacuum that means nothing, and reporting it
   would invite it to be used. `--stress` forces the attempt anyway. A failure
   to produce a stress never fails the run.
5. Nothing is moved, so nothing is written back: no trajectory, no `CONTCAR`,
   no `_final.vasp`.

### Outputs

`<prefix>_forces.csv` — one row per atom:

| column | meaning |
|---|---|
| `index` | atom index in the input structure |
| `symbol` | chemical symbol |
| `fx`, `fy`, `fz` | raw force components, eV/Å |
| `f_norm` | `|f|`, eV/Å |
| `free_x`, `free_y`, `free_z` | whether each component is free, from `free_component_mask` |

`mliprun_run.json` — a stage of kind `singlepoint`, `results`:

```
energy_eV                 float
fmax_free_eV_per_A        float
fmax_all_eV_per_A         float
n_free_atoms              int
worst_force_atom_all      int, with worst_force_atom_all_symbol
worst_force_atom_free     int, with worst_force_atom_free_symbol
stress_eV_per_A3          list[6] (Voigt) or null
stress_GPa                list[6] (Voigt) or null
stress_unavailable_reason str or null
unhandled_constraints     list[str]
committee_uncertainty     the uncertainty_summary block, committee runs only
```

`worst_force_atom_*` is named for the force deliberately. On a committee run
the record also carries `committee_uncertainty.worst_atom_free`, which is the
atom of greatest *disagreement*. Two different atoms, two different questions,
so they must not share a key name.

Committee runs additionally write `<prefix>_committee_peratom.csv` through the
existing `write_peratom_sigma`, unchanged.

Terminal output: energy, both fmax values, the worst atom, and — with a
committee — the existing `_report_committee_uncertainty` echo.

## Command 2 — `freq`

### CLI surface

`mlip freq run`, plus a standalone `freq` console script. The model-selection,
committee and `--uncertainty-threshold` options are identical to `singlepoint`.
Additionally:

| Option | Default | Meaning |
|---|---|---|
| `--indices` | from constraints (see below) | Atoms to displace. Accepts `0,1,5` and `12-30`, mixed |
| `--delta` | `0.01` | Displacement, Å. ASE's own default |
| `--nfree` | `2` | 2 or 4. 4 uses a five-point stencil, doubling the cost |
| `--direction` | `central` | `central`, `forward` or `backward` |
| `--method` | `standard` | `standard` or `frederiksen` (acoustic sum-rule correction, useful on slabs) |
| `--write-modes` | `imaginary` | `none`, `imaginary` or `all` — which modes get a trajectory file |
| `--expect-fmax` | from the run record, else none | Warn when fmax at the input geometry exceeds this. Never stops the run (D5) |
| `--prefix` | `freq` | Stem for this command's output files |

### Which atoms are displaced (D2)

`ase.vibrations.Vibrations` already defaults `indices` to every atom not held
by a `FixAtoms` constraint. That is exactly the required behaviour, so the
structure's constraints drive the selection with no code of ours in the path.
`--indices` overrides it.

**The gap, and what we do about it (D6).** ASE derives that default from
`FixAtoms` alone. An atom under any other constraint — `FixCartesian`,
`FixedPlane`, `FixedLine`, `FixBondLength`, `FixScaled` — is displaced in
full, and its held components enter the Hessian as though they were free.
ASE's `indices` selects whole atoms, so a partial Hessian cannot be expressed
through it.

The command therefore **warns and continues**. `select_indices` returns the
sorted names of the constraint types it could not honour; that list is echoed,
logged, and recorded as `unhandled_constraints`. Wording names the affected
types and states the consequence: those atoms were displaced in full and their
held components are in the Hessian as if free, so `--indices` is the way to
exclude them.

**Correction, 2026-09-18 (Task 8).** This document originally said detection
could reuse `free_component_mask`'s second return value, and that "one rule
covers both" this and the committee sigma path. **That is wrong**, and the
error is worth recording because the two look interchangeable and are not.

`free_component_mask` answers *which force components are free*, for a
statistic over components. It **handles** `FixCartesian` — it masks the held
components and reports nothing unhandled, which is correct there: a held
component simply does not enter the sum.

`select_indices` answers *which whole atoms to displace*. ASE's `indices`
selects whole atoms, so a `FixCartesian` atom is displaced in all three
directions no matter what, and its held components land in the Hessian as
though free. For this question `FixCartesian` is precisely **not** handled.

Verified on ASE 3.29 against a `FixCartesian(0, mask=(True, True, False))`
slab: `free_component_mask` returns `unhandled == []` with mask row
`[False, False, True]`, while `select_indices` returns
`unhandled == ["FixCartesian"]` and still displaces atom 0. Both are right for
their own question. The functions must therefore scan constraints separately,
and `free_component_mask`'s own frozen test pins the behaviour that makes
reuse impossible.

**Also note the type name.** ASE 3.29's `FixBondLength(a, b)` is a deprecated
factory that constructs a `FixBondLengths` instance, so the name that reaches
`unhandled_constraints` is the plural. The same trap is already documented in
this repo's committee tests.

### The stationary-point warning (D5)

A frequency analysis assumes the geometry sits at a stationary point. When it
does not, the measured curvature is not the curvature of a minimum, and the
failure shows up as a handful of spurious imaginary modes that are
indistinguishable by eye from a real transition state.

`Vibrations.run()` evaluates the undisplaced geometry first — `iterdisplace`
yields the `eq` displacement before any other — so fmax at the input geometry
costs nothing extra and is already in the displacement cache. It is always
measured, always echoed, and always recorded.

What it is compared against, in order:

1. `--expect-fmax X` when given. `fmax_expectation_source: "explicit"`.
2. Otherwise, the converged fmax found in an `mliprun_run.json` sitting in the
   input structure's directory, when one is there and carries a converged
   `optimize` stage. `fmax_expectation_source: "run_record"`. This is not an
   invented constant: it is the convergence criterion that was actually
   applied to this structure, with its provenance attached.
3. Otherwise no comparison. `fmax_expectation_source: "none"`,
   `fmax_warning: null`. The measured number is still reported, with a line
   saying no relaxation provenance was found next to the structure.

Exceeding the expectation prints a warning and sets `fmax_warning: true`. It
never stops the run.

### Outputs

`<prefix>_frequencies.csv`:

| column | meaning |
|---|---|
| `mode_index` | 0-based, in ASE's ascending-eigenvalue order |
| `frequency_cm-1` | **magnitude**, always positive |
| `energy_meV` | the same mode in meV |
| `imaginary` | bool |

The frequency column is a magnitude with a separate boolean, not a signed
number. The widespread convention of writing an imaginary frequency as a
negative one is a silent trap for any consumer that sums or sorts the column.

`<prefix>_summary.txt` — ASE's own `vib.summary()` table, which is the format
users already recognise from other ASE-driven work.

`<prefix>_vibrations.json` — `VibrationsData.write()`. This carries the full
Hessian and the atoms, and it is how D3 is honoured: a later `thermo` command
reads this file and runs `HarmonicThermo` or `IdealGasThermo` without
recomputing a single force.

`<prefix>/` — ASE's per-displacement JSON cache directory. Restart is free: an
interrupted sweep resumes at the displacement it stopped on, because every
completed displacement is already on disk. The name is not ours to choose
freely: `Vibrations` takes one `name` and derives both this directory and the
mode filenames from it, so the command passes `name = <output_dir>/<prefix>`.

`<prefix>.<n>.traj` — one animated trajectory per written mode, `n` being the
mode index. ASE's `write_mode` composes this path as `f"{vib.name}.{n}.traj"`,
which is why the cache directory above carries the bare prefix. Which modes get
written follows `--write-modes`; the default writes only the imaginary ones,
which is what a transition-state check needs to look at.

**A trap in `vib.summary()`**: its `log` argument opens a path in *append*
mode, so a restarted run would write a second table into the same file and the
result would read as twice as many modes. The command passes an open handle in
write mode instead.

`mliprun_run.json` — a stage of kind `freq`, `results`:

```
n_modes                      int
n_imaginary                  int
frequencies_cm-1             list[float], magnitudes
imaginary_mask               list[bool]
zpe_eV                       float
fmax_at_input_free_eV_per_A  float
fmax_expectation             float or null
fmax_expectation_source      "explicit" | "run_record" | "none"
fmax_warning                 bool or null
n_displaced_atoms            int
n_force_calls                int
unhandled_constraints        list[str]
committee_frequencies        committee runs only, see below
committee_uncertainty        committee runs only, at the input geometry
```

`parameters` carries `delta`, `nfree`, `direction`, `method`, `indices`,
`write_modes` and `prefix`, each tagged with its source by the existing
`param_sources_from_ctx` machinery.

**On ZPE and imaginary modes.** ASE's zero-point energy sums the real parts of
the mode energies, so an imaginary mode contributes exactly zero. That is a
defensible convention but it is not a well-defined zero-point energy, and the
documentation must say so wherever `zpe_eV` appears: a structure with imaginary
modes has no ZPE, and the number reported is the ZPE of its real modes only.

**What counts as an imaginary mode, added 2026-09-18 (Task 10 review).** A mode
is classified imaginary when `abs(energy.imag) > IMAGINARY_ENERGY_TOL_EV`, with
that constant set to `1e-8` — matching ASE's own `im_tol` in
`VibrationsData._tabulate_from_energies`, and applied to the mode **energy in
eV** rather than the frequency in cm⁻¹.

That alignment is a consistency requirement, not a preference. `freq` writes
both its own `<prefix>_frequencies.csv` and ASE's `<prefix>_summary.txt` from
the same run; if the two use different thresholds, or the same threshold on
different quantities, one file can call a mode imaginary while the other calls
it real. An earlier draft classified with `> 0` on the frequency and had
exactly that defect.

**[OPEN — Juan's decision, not settled here.]** Aligning with ASE removes the
disagreement between our two files. It does not answer the separate scientific
question: whether a *larger* tolerance should suppress near-zero modes
altogether. An adsorbate on a slab carries frustrated translations and
rotations at low frequency, and finite differences give those eigenvalues
arbitrary tiny signs — so a mode at a few cm⁻¹ with a negative eigenvalue may
be numerical noise rather than a real negative curvature.

This matters because it changes the transition-state test. Confirming a saddle
means finding **exactly one** imaginary mode, and that count is taken against
whatever threshold this spec sets. Until the question is settled, the reported
count is "imaginary by ASE's own definition", which is the most defensible
position available without a ruling, and the docs say so. Task 15's
transition-state check inherits this and should not be read as a verdict on
the threshold.

### Cost

Force calls are `1 + 6 × n_displaced` at `nfree=2`, and `1 + 12 × n_displaced`
at `nfree=4`. The leading 1 is the equilibrium evaluation. `--direction
forward` or `backward` does not reduce the number of displacements ASE
computes; it changes only which of them the Hessian assembly reads.

## Committee frequencies (D1)

### The obstacle

With a `CommitteeCalculator` attached, ASE sees only the mean force. One
Hessian comes out, and there is no spread. Per-member frequencies need each
member's own force array at every displacement, and `CommitteeCalculator`
currently discards them: `_evaluate` builds `stats` from the per-member forces
and keeps only `forces_mean` and the sigma derivatives.

### The mechanism

Verified end to end against ASE 3.29 before this document was written, with a
synthetic two-member committee on EMT:

1. **`CommitteeCalculator._evaluate` gains one key**, `forces_per_member`, an
   `(M, N, 3)` array in member order. This is the only change to existing
   committee code. Memory is negligible — 72 KB for six members on 500 atoms —
   and `latest` is overwritten each evaluation.

2. **`CommitteeVibrations(Vibrations)`** overrides `calculate()` to copy that
   array into the returned results dict. ASE stores the dict in its JSON cache,
   and the array survives the round trip as an `(M, N, 3)` list. Restart
   therefore works for committee runs exactly as it does for single-model runs.

3. **`assemble_hessian(forces_by_displacement, indices, delta, nfree,
   direction, method)`** — a pure function mirroring ASE's own assembly in
   `Vibrations.read`, which is **public** API in ASE 3.29 (this document
   originally called it `_read`; there is no such private method, corrected
   2026-09-18 during Task 7). Applied to the mean forces it reproduces ASE's
   `vib.H` with a maximum absolute difference of **exactly 0.0** in the probe,
   and the test suite asserts that equality rather than a tolerance.

   This narrows the fragility this design carries. The arithmetic being
   mirrored comes from public API; the only ASE-private names used anywhere
   in this work are `Vibrations._disp` and `_eq_disp`, which read the
   displacement cache, and those are confined to reading cached forces back
   out — see the note under the restart discussion.

4. **Per member**: `VibrationsData.from_2d(atoms, H_member, indices)` yields
   that member's frequencies and ZPE. Mass weighting and diagonalization stay
   ASE's; none of that physics is reimplemented here.

The headline frequencies remain ASE's own, from the mean forces (D8).

### Cost

One sweep, not `M` sweeps. Members are queried concurrently through the
committee's existing thread pool, so wall time is `max(member)`, not
`sum(member)` — the same economics the committee relaxation already has. A slab
with 50 free atoms is 301 committee evaluations at `nfree=2`.

### Output

`<prefix>_committee_frequencies.csv`:

| column | meaning |
|---|---|
| `mode_index` | as in the main CSV |
| `frequency_committee_cm-1` | the headline value, from the mean forces |
| `imaginary` | bool, for the committee value |
| `<member>_cm-1` | one column per member, magnitudes |
| `frequency_member_std_cm-1` | standard deviation across members, `ddof=1` |
| `<member>_overlap` | one column per member, see below |

Per-member ZPE, together with its mean and standard deviation across members,
goes in the run record under `committee_frequencies`, not in the CSV.

Every column name says which population it describes, in keeping with the
convention the 2026-09-08 sigma work established.

### Mode ordering across members — a correctness caveat

Each member's Hessian is diagonalized independently, and its eigenvalues come
back sorted ascending. For near-degenerate modes, member A's mode 7 and member
B's mode 7 need not be the same physical mode, and comparing them by index is
then not a like-for-like comparison. The per-mode standard deviation would
silently absorb an ordering swap as disagreement.

Pairing is by index, and the design makes the risk visible rather than
correcting it: for each member and each mode, the column `<member>_overlap`
carries `|⟨u_member,i | u_committee,i⟩|`, the absolute overlap between that
member's mode vector and the committee's. A clean match is close to 1. When any
overlap falls below 0.9 the command warns, naming the modes, and states that
their spread is not a like-for-like comparison. The 0.9 is a diagnostic trigger
for a warning, not a scientific verdict, and the overlaps themselves are in the
CSV for anyone who disagrees with it.

### A caveat the documentation must carry

At `delta = 0.01 Å` the force differences being divided are small, and MLIP
forces carry their own numerical noise at that scale. A member that is merely
noisy contributes to the per-mode spread alongside genuine model disagreement,
and one sweep cannot separate the two. Running at two values of `--delta`
distinguishes them. The docs say this rather than presenting the spread as pure
model uncertainty.

## Thermochemistry — out of scope, kept reachable (D3)

Not built here. What this design owes it is that building it later costs no
forces: `<prefix>_vibrations.json` carries the Hessian and the atoms, which is
everything `HarmonicThermo` needs and everything `IdealGasThermo` needs beyond
its own inputs (symmetry number, spin, geometry).

The decision that work will force, and which is not settled here: what happens
to imaginary modes in a free energy — discarded, taken as their absolute value,
or the free energy refused outright. Silently discarding them is the common
practice and it is a silent error. ASE 3.29 also ships `QuasiHarmonicThermo`
and `MSRRHOThermo`, which handle low-frequency modes differently again; which
of them is right is a scientific choice, not an implementation detail.

## Files

New:

- `src/mliprun/core/singlepoint.py` — `run_singlepoint(...)`
- `src/mliprun/core/vibrations.py` — `run_frequencies(...)`, `CommitteeVibrations`,
  `assemble_hessian(...)`, `select_indices(...)`
- `src/mliprun/cli/commands/singlepoint.py`
- `src/mliprun/cli/commands/freq.py`
- `src/mliprun/cli/committee_session.py` — see below

Changed:

- `src/mliprun/core/committee/calculator.py` — one new key in `_evaluate`
- `src/mliprun/cli/commands/optimize.py` — imports the extracted committee
  session helpers instead of defining them
- `src/mliprun/cli/main.py` — two new sub-apps
- `pyproject.toml` — two new console scripts
- `docs/OUTPUTS.md`, `docs/PYTHON_API.md`, `README.md`, `AGENTS.md`,
  `CHANGELOG.md`

### The committee-session extraction

`optimize.py` currently holds `_reject_conflicting_options`, `_build_committee`,
`_Terminated`, `_sigterm_as_interrupt`, `_started_committee` and
`_report_committee_uncertainty` as private functions — roughly 150 lines. Both
new commands need all of them.

Copying that into three files would put the SIGTERM teardown fix from PR #47 in
three places that can drift, on the exact code path whose failure left GPU
workers orphaned on a shared node. They move to
`src/mliprun/cli/committee_session.py` and `optimize.py` imports them.

This is a behaviour-preserving move, not a rewrite. It lands as its own commit,
with the existing committee suite green before and after, before any new
command is written. Nothing in `optimize run`'s behaviour changes.

### Run record schema

No schema bump, and the reasoning should be checked rather than taken on
trust. Versions 2, 3 and 4 each bumped because a new key appeared in the
top-level `provenance` block, which every reader parses. Version 5 bumped
because the *meaning* of an existing `results` field changed. Neither applies
here: `stages[].kind` is already a free string, `provenance` is untouched, and
the two new kinds add `results` shapes alongside the existing ones without
changing or reinterpreting any field a current reader already knows.

`docs/OUTPUTS.md` gains `singlepoint` and `freq` in its list of stage kinds,
and the two `results` blocks above.

## Testing

Everything below runs with **no MLIP installed**, on ASE's EMT, per the rule in
`AGENTS.md`. Every test asserts a numerical value or an invariant.

Single-point:

1. Energy and both fmax values for a fixed EMT structure, against hard-coded
   numbers.
2. `fmax_free` differs from `fmax_all` on a structure carrying `FixAtoms`, and
   `fmax_free` equals the `calc_fmax` of `atoms.get_forces()`.
3. `<prefix>_forces.csv` row count, column set, and the free-component columns
   against the constraint.
4. Stress recorded on a periodic EMT structure; `stress_unavailable_reason`
   populated and the run still successful on a calculator without stress.
5. Run record stage kind, status and results keys.

Frequencies:

6. `assemble_hessian` against ASE's `vib.H`, **exact equality**, for
   `nfree=2` central. Separate cases for `nfree=4`, forward, backward, and
   Frederiksen.
7. Frequencies of a fixed EMT system against hard-coded cm⁻¹ values.
8. Indices derived from `FixAtoms`: only the free atoms are displaced, and
   `n_force_calls` equals `1 + 6 × n_free`.
9. `--indices` overrides the constraint-derived default, including range
   syntax.
10. A non-`FixAtoms` constraint warns, names the type, records it in
    `unhandled_constraints`, and the run still completes.
11. The stationary-point warning in all three source states: `explicit`,
    `run_record` (a fixture directory carrying an `mliprun_run.json` from a
    converged optimize stage), and `none`.
12. Restart: interrupt after `k` displacements, re-run, and assert the second
    run makes only the remaining calls.
13. An imaginary mode is produced by a deliberately non-stationary structure,
    counted, flagged in the CSV as a positive magnitude plus `imaginary: true`,
    and excluded from `zpe_eV`.
14. `VibrationsData` round-trip: the written JSON reloads and reproduces the
    same frequencies.

Committee:

15. `forces_per_member` present on `latest`, correct shape and member order,
    and unchanged mean forces — the existing optimize committee tests are the
    regression guard for the rest.
16. Per-member frequencies differ and the per-mode standard deviation is
    non-zero, driven through the existing `emt` worker path with
    `tests/committee_stubs/biased_worker.py`.
17. A committee of identical members gives a per-mode standard deviation of
    exactly 0.0 and overlaps of exactly 1.0.
18. The overlap warning fires on a constructed pair of near-degenerate modes.
19. Committee restart: cached `forces_per_member` reloads and reproduces the
    same per-member frequencies.

CLI, through typer's `CliRunner`, matching `tests/test_cli_commands.py`:

20. Both commands' `--help`.
21. `--committee` with `--mlip`, `--uma-task`, `--mace-head`,
    `--sevennet-task` or `--device` is rejected on both commands, by parameter
    source rather than by value, so a typed `--mlip auto` is still caught.
22. `--nfree` outside `{2, 4}`, and an unknown `--direction`, `--method` or
    `--write-modes`, each fail with a message naming the allowed values rather
    than raising from inside ASE.
23. The output file listing each command prints matches the files actually on
    disk afterwards, including the case where `--write-modes imaginary` finds
    no imaginary modes and writes none.

Parametrize ids stay neutral (`member_a`, `case_a`), never the bare strings
`uma`, `mace` or `sevenn`, per the auto-skip trap in `tests/conftest.py`. After
adding parametrized tests, confirm `0 skipped` among the new cases.

## Out of scope

- **Thermochemistry** (D3), kept reachable as described above.
- **VACF power spectra** from MD trajectories, and **`ase.phonons`** supercell
  phonons (D1). Both were offered and declined.
- **IR intensities.** ASE's `Infrared` needs dipole moments, which none of the
  supported MLIP calculators provide. Not a scheduling decision — it is not
  possible with these models.
- **Batch mode.** Neither command gets an `optimize batch` equivalent here.
- **`md` and `neb` committee support.** Unchanged, still out of scope.
