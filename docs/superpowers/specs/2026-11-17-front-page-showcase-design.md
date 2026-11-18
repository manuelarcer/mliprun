# Front-page showcase: five alternatives for the README and a landing site

Date: 2026-11-17. Owner: Juan (science, final pick). Status: draft for approval.

## Intent

Replace the mostly-text GitHub README front page with something that shows, in seconds,
what mliprun does. The primary audience is Juan's own group and collaborators: people who
will run it on the group's clusters. They should leave knowing (1) what the five core
workflows look like, (2) that each is one command, and (3) how to run one on cos-cluster,
from the CLI or the web app.

Deliver **five distinct design alternatives**. Juan compares them and picks one, or a
combination, for the final front page.

## Goals

- Each alternative = a landing page (GitHub Pages, JavaScript allowed, live 3D) **plus**
  a README version of the same concept (no JavaScript: GIF / animated SVG / MP4) that
  links to the landing page.
- Every animation and figure comes from a **real mliprun run** with recorded provenance.
- Five use cases, shared by all alternatives (below).
- Quality bar: "not easy to please". A page reaches Juan only after an independent
  design review and a manager review.

## Non-goals

- No changes to mliprun's Python code, CLI, or web app.
- No new documentation content beyond what the front page needs.
- Not a publication: the runs demonstrate the tool and are labeled as such.

## Anti-goals (failure even if it "works")

- Illustrative or faked trajectories, or numbers that do not trace to a run.
- Stock-template look (generic hero + three feature cards + gradient blobs).
- Animation that hides the science: no axis-free curves, no unlabeled units.
- A README that breaks on GitHub (broken media, JS that is stripped, > 10 MB assets).
- Five alternatives that differ only in colour.

## The five use cases (shared data)

All on Pt(111), one system story. Settings are fixed in `docs/showcase/CANON.md`
(uma-s-1p2/oc20, fmax 0.02 eV/Å, (3×3)×4 slab cut from the UMA-relaxed bulk, 15 Å
vacuum, bottom 2 layers frozen).

| # | Use case | Command | Run | Media produced |
|---|---|---|---|---|
| 1 | Adsorbate relaxation | `mlip optimize run` | O at fcc hollow on Pt(111) | relaxation trajectory; E and fmax vs step |
| 2 | Committee uncertainty | `mlip optimize run --committee` | same start, OC20-level trio | committee CSVs; regenerated `committee_uncertainty.png` at 0.02 |
| 3 | Reaction path + TS check | `mlip neb run --verify-ts` | O fcc→hcp, 7 images, CI on | band trajectory; energy profile; imaginary-mode trajectory; verify report |
| 4 | Finite-temperature MD | `mlip md run` | 6-H2O √3 bilayer, NVT 300 K, 0.5 fs, 5 ps | MD trajectory; T(t), E(t) |
| 5 | Remote run from web app | `mlip webapp` + Machines | one optimize job sent to cos-cluster | screen recording: submit → queue → result |

Features without a strong animation (frequencies/ZPE, AutoNEB, benchmark, SevenNet and
CHGNet support, `mlip doctor`) appear as a compact capability list, not as hero content.

## The five design concepts

1. **Terminal hero.** An animated terminal types a real command (`mlip neb run …`); its
   real output streams and the energy profile draws itself. Use cases follow as
   command → result pairs.
2. **One system, full story.** Scroll-driven: the same O/Pt(111) goes relax → committee →
   NEB → TS → MD (water). Each chapter replaces the 3D scene.
3. **Live gallery.** Five looping trajectory tiles, each with its one-line command and a
   key number. Dense and scannable.
4. **Group handbook.** Practical first: which machine, which env, copy-paste recipe for
   each use case; animations support the recipes.
5. **Cinematic 3D.** Full-screen rotatable MD (water on Pt) behind the title; use cases
   as overlays/cards over the scene.

Each concept ships both forms (landing page + README variant).

## Approaches considered (how it gets built)

- **A. Data first, then design in parallel (chosen).** One data agent produces all runs
  and a shared, validated data pack. Five design agents then build from the same pack.
  Trade-off: designs wait for the runs (hours), but every alternative shows identical,
  real numbers, so comparisons are fair.
- B. Each design agent runs its own calculations. Rejected: five times the cluster time,
  inconsistent numbers between alternatives, provenance scattered.
- C. Design on mock data, real runs only for the winner. Rejected by Juan: drafts would
  mislead about how the real data looks.

## How it is built

### Phase 1 — Runs (cos-cluster, one GPU card)

- Run directory on cos-cluster under the shared scratch (`/scratchb/juar/mliprun-showcase/`),
  one subfolder per use case, using the existing per-MLIP venvs (`/scratchb/juar/envs/*`).
- Order: bulk relax → slab → O fcc and O hcp relaxations → committee relax → NEB with
  `--verify-ts` → H2O bilayer build + relax → MD.
- All committee members get `gpu: 0` (one card).
- Band explosion check (1.0 Å + 1.0 × travel) by a small script on the final band.
- Each run labeled converged / not-converged; each recorded in the prov ledger
  (project `mliprun-showcase`).
- Results pulled back to `docs/showcase/data/` (raw outputs gitignored except the small
  files the pages need).

### Phase 2 — Data pack

`docs/showcase/data/pack/` holds web-ready files every design uses:
- trajectories as multi-frame XYZ (subsampled, ≤ ~300 frames each);
- per-use-case JSON: series (step, energy, fmax, σ bands, T, image energies), key numbers
  with units and convergence labels, the exact command line run;
- `PROVENANCE.md`: which run produced which file (prov ledger ids).

### Phase 3 — Five alternatives (parallel)

- `site/alt-<n>-<slug>/index.html` (+ assets): the landing page. Allowed libraries:
  3Dmol.js (already used by the web app) for 3D, plain JS/CSS or a small, pinned CDN lib
  for charts. Static only; must load from GitHub Pages without a build step.
- `site/alt-<n>-<slug>/README.md`: the README variant, using media rendered from the same
  pack (GIF/MP4/animated SVG under `site/alt-<n>-<slug>/media/`), each ≤ ~5 MB.
- Each landing page is also published as a private claude.ai Artifact for Juan to preview.

### Phase 4 — Review

- An independent reviewer agent (not the builder) critiques each alternative against
  this document: real-data fidelity, design quality, GitHub rendering of the README,
  mobile layout, load weight, accessibility basics.
- Builders fix; manager (main session) reviews last. Only passing alternatives reach Juan.

### Phase 5 — Selection and integration (after Juan picks)

- The chosen design (or combination) replaces `README.md`; the landing page moves to the
  GitHub Pages location with a Pages workflow; draft PR only (repo rule).

## Constraints

- No calculations on the local machine (`docs/showcase/CANON.md`, K-p1).
- cos-cluster has no scheduler and is shared: one card, check `nvidia-smi` for a free one.
- README on GitHub: no JS, no iframes; videos via GitHub-hosted MP4 or GIF.
- Draft PRs only; frozen goldens untouched; no source changes in `src/`.

## Decided by Juan

README + landing site; real runs only; audience = own group/collaborators; systems O hop +
H2O/Pt(111); all five concepts; uma-s-1p2/oc20; fmax 0.02; slab recipe, NEB, MD, TS and
explosion settings as in `docs/showcase/CANON.md`; one GPU card on cos-cluster.

## Left to the builders (manager's calls)

- Visual identity per concept (type, palette, motion), within the anti-goals above.
- Rendering pipeline for README media (e.g. headless browser capture of the 3Dmol scene,
  so README and site look alike).
- Subsampling rates and frame counts.

## Settled after the builder check

- Bulk: plain `--relax-cell` on the primitive cell, then an fcc check (stop if broken).
- NEB: `--verify-ts both`. MD: Langevin, friction 0.01 fs⁻¹, explicit.
- MD system: 6 H2O H-down √3 bilayer, no O adatom; concept 2's story moves from the O
  adatom (relax → committee → NEB/TS) to a new water scene for MD.
- Committee: new `docs/showcase/committee.yaml`, all members `gpu: 0`, fmax 0.02.
- The 0.02 committee figure goes to the data pack only; `docs/images/committee_uncertainty.png`
  is replaced only in Phase 5, after Juan picks a design.
- Prov ledger: project `mliprun-showcase`, root = repo root, fixed once.
- Use case 5: an agent drives a headless browser through the web app, using the saved
  `cos-cluster` machine entry and Juan's open SSH connection (Juan is asked to open it if
  it is down). The job is an O fcc relaxation rerun, ledger-recorded.
