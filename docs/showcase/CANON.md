# CANON decisions — mliprun front-page showcase runs

Master: `~/.claude/CANON.md` — read it first, together with its VASP-parameter
sub-file `~/.claude/CANON-VASP-PARAMS.md` (CV items). Entries below are this
project's decisions; they override the master's defaults. Any canon item not recorded here
and not settled in the master is a blocking question (see the master's law).

Scope: the real calculations behind the README and landing-site animations (O/Pt(111)
relaxation, committee relaxation, O fcc→hcp NEB with TS verification, H2O bilayer MD).
They are demonstrations of the tool, not publication results.

Format (master law 9):
- The **Contents** list has one line per decision: the decision in plain words, the ID in
  brackets at the end. Add the line when you add the entry.
- Each entry: the heading is the decision in plain words, then the ID in brackets. Under
  it, bullets with fixed fields in this order, empty ones left out: **Decision**, **Why**,
  **Never**, **Open**, **Note**, **Evidence**, **Decided** (date, and "Juan" when he
  decided it in person). One idea per bullet, no paragraphs.
- A revision replaces the entry text; it never appends to it. Evidence goes to
  `CANON-EVIDENCE.md` beside this file, under the same ID.
- In chat, logs and reports, state a rule in words first and give the ID in brackets
  after it (master law 11). Never cite a bare ID.

## Contents

Bulk
- The Pt bulk is built from the fcc prototype [B1]
- The bulk relaxes cell and atoms, with fcc symmetry checked afterwards [B2]

Surface
- The slab is cut with ase.build.fcc111 from the relaxed lattice constant [S1]
- The slab has 4 layers; one layer is one close-packed Pt plane [S2]
- Vacuum is 15 Å [S5]
- The bottom 2 layers are frozen [S8]

Adsorption
- O sits at the fcc and hcp hollows [A1]
- O binds monodentate through itself; water lies as a √3-type bilayer [A2]

Calculation parameters
- Every run uses uma-s-1p2 with the oc20 task; the committee is the OC20-level trio [C1]
- fmax is 0.02 eV/Å for every relaxation and NEB endpoint [C2]
- MD is Langevin NVT at 300 K, 0.5 fs, 5 ps on a 6-H2O bilayer [C5]
- NEB uses 7 images, k = 0.1, climbing image on [C6]
- Band explosion allowance is 1.0 Å + 1.0 × each atom's IS→FS travel [C7]
- TS check uses ts verify mode both, 10 cm⁻¹ floor, overlap ≥ 0.7 [C8]

Bookkeeping
- Every run executes on cos-cluster on one GPU card [K-p1]

## Bulk (B)

### The Pt bulk is built from the fcc prototype [B1]
- **Decision:** Pt bulk is built with `ase.build.bulk("Pt", "fcc")`, not downloaded.
- **Why:** single-element fcc metal; the prototype is unambiguous.
- **Decided:** 2026-11-17, Juan.

### The bulk relaxes cell and atoms, with fcc symmetry checked afterwards [B2]
- **Decision:** primitive fcc cell relaxed with `mlip optimize run --relax-cell`; symmetry is not enforced.
- **Decision:** after the run, the cell must still be fcc (equal lengths and 60° angles, within 1e-4 Å and 0.01°); otherwise stop.
- **Why:** mliprun has no FixSymmetry; a primitive fcc cell has no symmetry-breaking freedom, so a check suffices and the step stays one mliprun command.
- **Decided:** 2026-11-17, Juan.

## Surface (S)

### The slab is cut with ase.build.fcc111 from the relaxed lattice constant [S1]
- **Decision:** `ase.build.fcc111("Pt", size=(3, 3, 4), a=<relaxed a>, vacuum=7.5)`.
- **Never:** pass the experimental or default lattice constant (slabs come only from the optimized bulk, S7).
- **Note:** the (3×3) cell is about 8.4 Å in plane, above the 7.5 Å minimum (S6).
- **Decided:** 2026-11-17, Juan.

### The slab has 4 layers; one layer is one close-packed Pt plane [S2]
- **Decision:** 4 Pt(111) planes.
- **Note:** thin for publication energetics; adequate for a demonstration.
- **Decided:** 2026-11-17, Juan.

### Vacuum is 15 Å [S5]
- **Decision:** 15 Å total vacuum (7.5 Å each side in the `fcc111` call).
- **Decided:** 2026-11-17, Juan.

### The bottom 2 layers are frozen [S8]
- **Decision:** the two bottom Pt planes are fixed through a `FixAtoms` constraint written into the input structure.
- **Decided:** 2026-11-17, Juan.

## Adsorption (A)

### O sits at the fcc and hcp hollows [A1]
- **Decision:** one O per (3×3) cell, at the fcc hollow (NEB initial state) and at the hcp hollow (NEB final state).
- **Decided:** 2026-11-17, Juan.

### O binds monodentate through itself; water lies as a √3-type bilayer [A2]
- **Decision:** atomic O, monodentate at the hollow.
- **Decision:** the MD system is 6 H2O in a √3-type bilayer on the (3×3) cell (2 H2O per √3 cell), with no O adatom.
- **Decision:** H-down bilayer: in each √3 cell one water lies flat and one points an H toward the surface.
- **Decided:** 2026-11-17, Juan.

## Calculation parameters (C)

### Every run uses uma-s-1p2 with the oc20 task; the committee is the OC20-level trio [C1]
- **Decision:** single-model runs use `--mlip uma-s-1p2 --uma-task oc20`, including the bulk relaxation.
- **Decision:** the committee is `uma-s-1p2`/oc20, `7net-omni`/oc20 and `mace-mh-1`/`oc20_usemppbe`, as in `examples/committee.yaml`.
- **Why:** one head per energy formula (C3); all committee members sit at the same OC20 (RPBE) level, so their spread is an error bar.
- **Never:** add CHGNet or any non-OC20 member to the committee.
- **Decided:** 2026-11-17, Juan.

### fmax is 0.02 eV/Å for every relaxation and NEB endpoint [C2]
- **Decision:** `--fmax 0.02` for bulk, slab, adsorbate, committee and NEB endpoint relaxations, and for the NEB band.
- **Note:** the existing `docs/images/committee_uncertainty.png` was made at 0.05 and is replaced by a 0.02 rerun.
- **Decided:** 2026-11-17, Juan.

### MD is Langevin NVT at 300 K, 0.5 fs, 5 ps on a 6-H2O bilayer [C5]
- **Decision:** NVT, 300 K, 0.5 fs timestep, 10 000 steps (5 ps).
- **Decision:** Langevin thermostat, friction 0.01 fs⁻¹, passed explicitly.
- **Why:** 0.5 fs because the system contains H.
- **Decided:** 2026-11-17, Juan.

### NEB uses 7 images, k = 0.1, climbing image on [C6]
- **Decision:** 7 intermediate images, spring constant 0.1 eV/Å², climbing image on.
- **Decided:** 2026-11-17, Juan.

### Band explosion allowance is 1.0 Å + 1.0 × each atom's IS→FS travel [C7]
- **Decision:** floor 1.0 Å, factor 1.0, checked on every interior image of the final band.
- **Note:** mliprun has no built-in explosion check; a short script performs it.
- **Decided:** 2026-11-17, Juan.

### TS check uses ts verify mode both, 10 cm⁻¹ floor, overlap ≥ 0.7 [C8]
- **Decision:** `neb run --verify-ts both`: overlap with the NEB tangent plus displace-and-relax to both endpoints.
- **Decision:** ts verify defaults: imaginary floor 10 cm⁻¹, overlap threshold 0.7.
- **Decided:** 2026-11-17, Juan.

## Bookkeeping (K)

### Every run executes on cos-cluster on one GPU card [K-p1]
- **Decision:** all showcase calculations run on cos-cluster using a single GPU card; committee members share it.
- **Decision:** at most two jobs (two `mlip` commands) run on that card at the same time; a committee run counts as one job.
- **Never:** run showcase calculations on the local machine.
- **Decided:** 2026-11-17, Juan.
