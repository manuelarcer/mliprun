# Committee sigma: constrained-atom masking, and an opt-in threshold

Design note, 2026-09-08. Amends
`2026-09-04-committee-uncertainty-design.md`; everything that document says
still holds except where this one contradicts it.

Both changes were ruled on by Juan on 2026-09-08 after the cos-cluster
verification of PR #46. They are recorded here because they are decisions,
not derivations -- a later reader must be able to see what was chosen and
why, not just what the code does.

## Context

PR #46 shipped `sigma`, the per-atom force disagreement across committee
members, and a flagging rule comparing `sigma_max` to a threshold that
defaults to the run's `fmax`. Two same-level cluster relaxations then
measured:

| system | members | fmax reached | sigma_max | sigma_mean | aligned E spread |
|---|---|---|---|---|---|
| O/Pt(111), 17 atoms, 27 free DOF | 3 | 0.05 | 0.1502 | 0.0643 | 0.0313 eV |
| CH/FeNi, 82 atoms, 150 free DOF | 6 | 0.02 | 0.1121 | 0.0343 | 0.0507 eV |

Both were flagged. A verdict that fires on healthy work is not a verdict.

## Decision 1: sigma excludes constrained force components

**What was wrong.** `committee_statistics` reduced sigma over *every* atom
and every component. `fmax` -- the quantity the default threshold compared it
against -- is ASE's max over *free* atoms only, after constraints are
applied. The default comparison was therefore between two maxima taken over
different sets of atoms.

This is not a small difference in the runs above: 8 of 17 atoms in the O/Pt
statistic are fixed, and 32 of 82 in the CH/FeNi one. A `sigma_max` sitting
on a frozen bottom-layer atom reports disagreement in a region that cannot
move, cannot change the geometry, and largely cancels out of any adsorption
energy or barrier computed as a difference against a common slab.

**The rule.** Mask componentwise, not per atom. A force component that a
constraint holds fixed contributes nothing to sigma; an atom free in x and y
but fixed in z still contributes its x and y disagreement. `sigma_max_free`
and `sigma_mean_free` are then taken over the atoms that retain at least one
free component.

**Which constraints are handled, and why only those.** `FixAtoms` and
`FixCartesian` are the only stock ASE constraints whose `adjust_forces` is a
pure component mask (`forces[index] = 0.0` and
`forces[index] *= ~mask[None, :]` respectively, verified byte-identical against
both ase 3.26.0 and ase 3.29.0 -- 3.29.0 is what the repo's `.venv` runs, so it
is the version that matters).
For those the masked component is exactly zero and excluding it is
unambiguous. Every other constraint type -- `FixScaled`, `FixedPlane`,
`FixedLine`, `FixBondLength` -- projects rather than masks, and a projection
does not carry over to a standard deviation, which is not a vector.

Those are treated as **unhandled**: their atoms stay free, sigma is
over-reported rather than under-reported, and the type names are recorded in
the run record so the fallback is visible.

The rejected alternative was a generic probe -- push an array of ones through
each constraint's `adjust_forces` and treat a zeroed component as fixed. It
has a silent and dangerous failure mode: a `FixedPlane` whose normal is
parallel to (1,1,1) zeroes the probe exactly, masking components that a real
force would keep. Over-reporting is the safe direction; under-reporting is
not.

**Energy is unaffected.** The energy is one number for the whole cell. There
is no per-atom decomposition to mask, so the aligned energy spread is
untouched by this change.

**The old numbers are kept.** `sigma_max_all` / `sigma_mean_all` /
`worst_atom_all` carry the pre-change definition alongside the new
free-atom values. Cost is a few floats; the benefit is that the two measured
datapoints above stay comparable to every later run, and the gap between the
two numbers says directly how much of the old figure was frozen substrate --
which is the evidence any future threshold calibration needs.

**Naming: every sigma name states which atoms it covers.** Juan's ruling,
2026-09-09. A sigma name carries `free` (atoms free to move) or `all` (every
atom in the cell); there is no bare form, in the CSV columns, in the run
record, or in the internal statistics dict.

The first attempt at this made the exception explicit and left the common case
bare -- `sigma_max_all_eV_per_A` announced itself while `sigma_max_eV_per_A`
silently meant free-atoms-only. That is the wrong way round: the bare name is
the one people actually read, so it is the one that gets misread, and a reader
had to already know the convention to know what they were looking at. The
concrete failure it produced was two sibling output files using opposite
polarity for the same distinction, so `df_peratom.sigma_eV_per_A.max()` stopped
equalling the trace's final `sigma_max_eV_per_A` -- silently, on exactly the
constrained runs this feature exists for.

`free` rather than `relaxed`, for consistency with `n_free_atoms` and
`free_components`, and with ASE's degrees-of-freedom vocabulary.

The internal dict keys follow the same rule, not only the user-facing columns.
A `stats["sigma_max"]` meaning "free" behind a column named
`sigma_max_free_eV_per_A` would rebuild the same implicit convention one layer
down, where the next maintainer meets it.

## Decision 2: the threshold is opt-in, and `flagged` is tri-state

**What changes.** `--uncertainty-threshold` no longer defaults to `fmax`. Left
unset, no threshold is applied and no verdict is asserted. The committee
always reports `sigma_max_free`, `sigma_mean_free`, the worst free atom and
the ratio `sigma_max_free / fmax`; a warning appears only when a threshold was
chosen and exceeded.

**Why not simply loosen the default to the measured spread.** Two datapoints
are not a calibration, and they were measured on the pre-masking statistic --
Decision 1 changes both numbers. Picking 0.15 today means picking it from
data that is about to move.

**Why this is not cosmetic.** `flagged: true` is a machine-readable claim,
written permanently into the run record, that a downstream script can filter
on. A threshold turns a measurement into an assertion; asserting from an
arbitrary number puts an arbitrary claim into the provenance. Adding a
calibrated default later is easy. Un-asserting `flagged` from records already
written is not.

**`flagged` becomes tri-state**: `true`, `false`, or `null`. `false` must keep
meaning "checked against a threshold and passed" -- reusing it for "not
checked" is exactly the failure PR #46 fixed when `device_resolved` reported
`"cpu"` for a GPU run it had not resolved.

`null` therefore means **no verdict was reached**, which happens two ways: no
threshold was applied, *or* a threshold was applied but nothing was ever
evaluated (a run that died before its first optimizer step, where `latest` is
`None`). The second case was underspecified in the first draft of this note
and is recorded here because it is the one a consumer gets wrong: a script
filtering `flagged == false` to select healthy runs would otherwise count a
run that crashed before its first force call as healthy.

**What a threshold is for, when one is wanted.** Three uses want different
things, and only the third needs an absolute number: a single careful study
(read the numbers, a verdict adds nothing); batch screening (rank within the
batch -- a relative criterion self-calibrates where an absolute one does
not); an automated trigger such as active learning or a DFT hand-off. The
third is not yet in use here.

## Decision 3: MPtrj models share one level, labelled `PBE(+U)/MPtrj`

Juan's ruling, 2026-09-08. `mace` (MACE-MP-0 medium) and `chgnet` are both
trained on MPtrj -- the Materials Project relaxation trajectories -- and were
carrying different labels (`PBE/MPtrj` and `PBE+U/MPtrj`), so a committee of
the two reported as mixed theory and its spread was presented as a functional
comparison rather than an error bar. They now share one label.

The label names the dataset and signals the mixing rather than asserting a
functional, because **the +U question is composition-dependent**: MPtrj
applies Hubbard U to specific transition metals in oxides and fluorides under
MP's GGA/GGA+U mixing scheme, so an MPtrj-trained model is effectively plain
PBE for a metallic slab and PBE+U for an oxide. No static label can settle
that. `PBE/MPtrj` is wrong for the oxides and `PBE+U/MPtrj` is wrong for the
metals; `PBE(+U)/MPtrj` is accurate for both, and since the code only ever
compares these strings for equality, nothing is lost by being accurate in the
display.

**What this unlocks:** a MACE-MP-0 + CHGNet committee is now read as an error
bar. Those two share no architecture, only training data, which makes them a
less correlated committee than several heads off one backbone -- the
correlated-members weakness `.claude/ideas.md` flagged for this feature.

**What is deliberately not added.** `7net-0`, `7net-0_22may2024` and
`7net-l3i5` are believed to be MPtrj as well, but that has not been confirmed
from the checkpoints. The repo's rule is to read the installed package, never
the documentation -- it is what caught three documentation errors on
2026-09-01 -- so those rows wait for `sevenn cp <tag>` on cos-cluster. Until
then a `mace` + `7net-0` committee still reports as mixed. The remaining
missing rows (molecular and molecular-crystal heads: UMA `omc`/`omol`,
MACE-MH `omol`/`spice_wB97M`, SevenNet `omol25_*`/`spice`/`qcml`) stay
unlisted on purpose: they cannot appear in a slab committee, so filling them
would add liability with no benefit.

`LEVEL_TABLE_VERSION` goes 1 -> 2.

## Not changing

- The committee remains a proper potential: mean force is the negative
  gradient of the mean energy. Nothing here touches the forces the optimizer
  sees -- only the statistic reported alongside them.
- Mixed levels of theory still warn and never refuse.
- The per-atom CSV keeps a row for every atom, constrained ones included.
  Seeing the frozen atoms is how a reader checks Decision 1 on their own run.
- The remaining unlisted level-of-theory rows stay unlisted; see
  Decision 3 for which and why.

## Consequences

- Run record schema 4 -> 5. Schema 4's `sigma_max_final_eV_per_A` counted every
  atom; schema 5 renames it `sigma_max_free_final_eV_per_A` and reduces over
  free components only. A changed *meaning* is stronger than the two previous
  bumps, which only added fields — a reader must be able to tell which
  definition a record was written under, and here the name itself says so.
- The two cluster datapoints above are superseded as calibration inputs. Any
  future threshold must be measured after this change.
- No frozen golden contains sigma (the four in `tests/goldens/` predate the
  committee feature), so none may change.
