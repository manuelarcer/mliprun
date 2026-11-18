# Front-Page Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce real mliprun runs on cos-cluster, a shared web-ready data pack, and five
reviewed front-page alternatives (landing page + README variant each) for Juan to choose from.

**Architecture:** Phase 1 runs all calculations on cos-cluster (one GPU card) from small
local scripts copied over with rsync. Phase 2 turns the outputs into one data pack
(`docs/showcase/data/pack/`) that every design reads. Phase 3 builds five static
alternatives in `site/alt-<n>-<slug>/` in parallel; Phase 4 reviews them independently.

**Tech Stack:** mliprun CLI (`optimize run`, `neb run --verify-ts`, `md run`, `webapp`),
ASE, Python 3 + pytest for the helper scripts, static HTML/CSS/JS with 3Dmol.js 2.4.2
(jsDelivr, same as the web app), Playwright (headless Chromium) for media capture,
ffmpeg for GIF/MP4.

**Spec:** `docs/superpowers/specs/2026-11-17-front-page-showcase-design.md`; calculation
settings: `docs/showcase/CANON.md` (both must be read by every executor).

## Global Constraints

- Every calculation runs on cos-cluster, one GPU card; never on the local machine (K-p1).
- Model: `--mlip uma-s-1p2 --uma-task oc20` on every single-model run; never `--mlip auto` (C1).
- Committee: `uma-s-1p2`/oc20, `7net-omni`/oc20, `mace-mh-1`/`oc20_usemppbe`, all `gpu: 0`; no other member (C1).
- `--fmax 0.02` on every relaxation, NEB endpoint (`--endpoint-fmax 0.02`) and NEB band (C2, C4).
- Optimizer: `bfgs` for standalone and endpoint relaxations alike (C4, manager's call).
- Slab: `fcc111("Pt", size=(3,3,4), a=<relaxed a>, vacuum=7.5)`, bottom 2 layers `FixAtoms` (S1, S2, S5, S8).
- NEB: `--num-images 7 --k 0.1 --climb --verify-ts both` (C6, C8).
- Band explosion allowance per atom: 1.0 Å + 1.0 × that atom's IS→FS travel (C7).
- MD: `--ensemble nvt --thermostat langevin --friction 0.01 --temperature 300 --timestep 0.5 --steps 10000` (C5).
- Every run labeled converged / not-converged in the pack (K1) and recorded with `prov update <repo root> mliprun-showcase` (K2).
- cos-cluster pre-flight (`/app1-cos/checkNodeUserStatus.sh`, `nvidia-smi`) before each launch; launch with `setsid nohup … & echo launched`, verify with `pgrep -af`.
- Never pull raw `md.traj` to the laptop; subsample on the cluster.
- No changes under `src/` or `tests/`; `docs/images/committee_uncertainty.png` and `README.md` untouched (Phase 5 only).
- Draft PR only; branch `showcase/front-page`.
- README media ≤ 5 MB each; landing pages static, no build step, script URLs with exact versions.

## Review Focus

1. A run that "finishes" but is not converged (fmax never reached, NEB step cap hit): the pack must carry `converged: false` and the pages must not present it as a result — test in Task 6.
2. MLIP-broken geometry (O sinks into the slab, water desorbs or dissociates during MD): flagged by a geometry check, not silently animated — test in Task 6.
3. Bulk cell drifts off fcc: Task 2 stops — test in Task 2.
4. README variant viewed on GitHub (JS stripped, relative media paths, dark mode): every media link resolves and nothing depends on script — check in Task 9.
5. Landing page at phone width (375 px) and with `prefers-reduced-motion`: no horizontal scroll; animations pause — check in Task 9.

---

### Task 1: Branch, scaffolding, cluster readiness

**Files:**
- Create: `docs/showcase/committee.yaml`, `docs/showcase/.gitignore`, `docs/showcase/README.md` (one paragraph: what this folder is, pointer to CANON.md and the spec)

**Interfaces:**
- Produces: remote root `REMOTE=/scratchb/juar/mliprun-showcase`; env paths `UMA_BIN`, `SEVENN_BIN`, `MACE_BIN` written into `docs/showcase/cluster.env` (shell `KEY=value` lines) for all later tasks.

- [ ] **Step 1:** `git checkout -b showcase/front-page` from `main`; confirm `git branch --show-current` prints `showcase/front-page`.
- [ ] **Step 2:** Check the SSH master: `mkdir -p /tmp/ssh_mux && ssh -O check cos-cluster`. If no master runs, STOP and ask Juan to run `! ssh -fN cos-cluster`.
- [ ] **Step 3:** The shared install `/app1-cos/mlip-platform/mlip-platform` is at `7868eb8`, behind `d7ecf27` (TS verification). Never touch it. Instead place `main` at `2b3044b` (contains d7ecf27 TS verification and #66 `--relax-cell all|volume|shape`) in `/scratchb/juar/mliprun-showcase/mliprun` (`git clone https://github.com/manuelarcer/mliprun.git` on the cluster; if that fails, `git archive 2b3044b | ssh cos-cluster 'mkdir -p … && tar -x -C …'` from the local repo) and export `PYTHONPATH=/scratchb/juar/mliprun-showcase/mliprun/src` in `cluster.env`. Verify from every env: `PYTHONPATH=… <env>/bin/python -c "import mliprun; print(mliprun.__file__)"` prints the `/scratchb/juar/mliprun-showcase/` path, and `PYTHONPATH=… <env>/bin/mlip neb run --help` lists `--verify-ts`. Any env that resolves to `/app1-cos` → STOP.
- [ ] **Step 4:** Resolve the three envs: test `/scratchb/juar/envs/{uma,sevenn,mace}/bin/mlip` first, then `/scratchb/juar/EWaste2GreenCat/explicit_solvation/.venv/{uma,mace}/bin/mlip`. For each found env run `<bin>/mlip doctor` and record which MLIP it reports. No sevenn env → STOP and report (committee cannot be same-level without it).
- [ ] **Step 5:** Write `committee.yaml` from `examples/committee.yaml` with the resolved env paths and `gpu: 0` for all three members; write `cluster.env`.
- [ ] **Step 6:** `.gitignore` in `docs/showcase/`: ignore `data/runs/**` except `*.csv`, `*.json`, `*_final.vasp`, `verify_ts*`, `*.txt` logs ≤ 1 MB; never `*.traj`.
- [ ] **Step 7:** Commit: `chore(showcase): scaffolding, committee file and cluster env`.

### Task 2: Bulk, slab and O fcc/hcp relaxations

**Files:**
- Create: `docs/showcase/scripts/build_structures.py`, `docs/showcase/scripts/run_01_relax.sh`
- Test: `docs/showcase/tests/test_build_structures.py`

**Interfaces:**
- Produces: `build_structures.py` CLI with subcommands `bulk OUT`, `check-fcc RELAXED_BULK` (exit 1 when not fcc), `slab RELAXED_BULK OUT`, `adsorb SLAB {fcc,hcp} OUT`, `bilayer SLAB OUT`; functions `is_fcc(atoms, tol_len=1e-4, tol_ang=0.01) -> bool`, `make_slab(a: float) -> Atoms`, `add_o(slab, site: str) -> Atoms`, `make_bilayer(slab) -> Atoms`.
- Remote outputs: `$REMOTE/01_relax/{bulk,slab,o_fcc,o_hcp}/opt_final.vasp`.

- [ ] **Step 1: Write failing tests** (run locally with any ASE env — structure building only, no MLIP):
  - `test_is_fcc_true_for_prototype`: `is_fcc(bulk("Pt","fcc",a=3.97))` is True.
  - `test_is_fcc_false_when_strained`: cell scaled by `[1, 1, 1.001]` → False.
  - `test_slab_geometry`: `make_slab(3.97)` has 36 atoms, in-plane vector lengths ≥ 7.5 Å, vacuum gap 15.0 ± 0.01 Å, exactly 18 atoms fixed and they are the lowest two layers.
  - `test_add_o_sites`: O at fcc sits above no second-layer atom; O at hcp sits above a second-layer atom (lateral distance < 0.1 Å); O height above top layer 1.0–1.4 Å initial.
  - `test_bilayer`: 6 O + 12 H added; per √3 cell one water roughly flat (H–O–H plane within 15° of surface) and one with an H pointing down (H below its O); no H–H closer than 1.4 Å between waters; constraint preserved.
- [ ] **Step 2:** `pytest docs/showcase/tests/test_build_structures.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement `build_structures.py`. Bilayer: place water O on top sites of a √3 sub-lattice at ~2.3–2.7 Å height (flat at lower, H-down at higher), H–O 0.97 Å, H–O–H 104.5°.
- [ ] **Step 4:** Tests PASS.
- [ ] **Step 5:** `run_01_relax.sh` (sources `cluster.env`, `set -euo pipefail`): bulk `optimize run --relax-cell all` → `check-fcc` (stop on fail) → `slab` → `optimize run` slab → `adsorb` fcc and hcp on the relaxed slab → two `optimize run`. Every run: `--mlip uma-s-1p2 --uma-task oc20 --device cuda --fmax 0.02 --optimizer bfgs`, `CUDA_VISIBLE_DEVICES=<free card>`.
- [ ] **Step 6:** Pre-flight, rsync scripts to `$REMOTE`, launch with setsid, verify with `pgrep`; wait (background, no polling subagent); confirm four `opt_final.vasp` files exist and each log ends converged. Record relaxed `a` and the fcc check output in the task report.
- [ ] **Step 7:** Commit scripts + tests: `feat(showcase): structure builders and relaxation run`.

### Task 3: Committee relaxation

**Files:**
- Create: `docs/showcase/scripts/run_02_committee.sh`

- [ ] **Step 1:** Script: `mlip optimize run --structure $REMOTE/01_relax/o_fcc/<unrelaxed input> --committee committee.yaml --fmax 0.02 --uncertainty-plot` in `$REMOTE/02_committee/`. Start from the same unrelaxed O-fcc structure as Task 2 (so the animation shows the full relaxation).
- [ ] **Step 2:** Pre-flight, launch, verify; on completion confirm `*_committee.csv` and the uncertainty PNG exist and all three members appear as columns.
- [ ] **Step 3:** Commit: `feat(showcase): committee relaxation run`.

### Task 4: NEB with TS verification and band explosion check

**Files:**
- Create: `docs/showcase/scripts/band_check.py`, `docs/showcase/scripts/run_03_neb.sh`
- Test: `docs/showcase/tests/test_band_check.py`

**Interfaces:**
- Produces: `band_check(initial, final, seeded: list[Atoms], final_band: list[Atoms], floor=1.0, factor=1.0) -> dict` with keys `exploded: bool`, `worst: {image, atom, drift, allowance}`, `max_interimage_jump: float`, `jump_warning: bool` (jump > 1.0 Å). CLI: `band_check.py NEB_DIR` prints JSON and writes `band_check.json`.

- [ ] **Step 1: Failing tests:**
  - `test_healthy_band`: straight-line band, final = seeded + 0.3 Å noise on the mover → `exploded is False`.
  - `test_exploded_band`: one image's fixed-in-travel atom (travel 0) displaced 1.2 Å → `exploded is True`, `worst.atom` is that index, `worst.allowance == 1.0`.
  - `test_allowance_scales_with_travel`: mover with travel 1.6 Å displaced 2.5 Å from seed → not exploded (allowance 2.6); displaced 2.7 Å → exploded.
  - `test_jump_warning`: consecutive images 1.1 Å apart → `jump_warning is True`, `exploded is False`.
  - Drift uses minimum-image displacement under PBC.
- [ ] **Step 2:** Tests FAIL. **Step 3:** implement. **Step 4:** tests PASS.
- [ ] **Step 5:** `run_03_neb.sh`: `mlip neb run --initial o_fcc/opt_final.vasp --final o_hcp/opt_final.vasp --mlip uma-s-1p2 --uma-task oc20 --device cuda --num-images 7 --k 0.1 --climb --fmax 0.02 --optimize-endpoints --endpoint-fmax 0.02 --endpoint-optimizer bfgs --verify-ts both --plot`, then `band_check.py`. Find the seeded-path file name the NEB writes (read `docs/OUTPUTS.md` NEB section); if no seeded band is saved, STOP and report.
- [ ] **Step 6:** Launch, verify, wait. Report: barrier (eV), imaginary frequency (cm⁻¹), number of imaginary modes, overlap, relax-check result, band_check verdict. Any failure → label not-converged and report; do not rerun with changed settings.
- [ ] **Step 7:** Commit: `feat(showcase): NEB run and band explosion check`.

### Task 5: Water bilayer relaxation and MD

**Files:**
- Create: `docs/showcase/scripts/run_04_md.sh`, `docs/showcase/scripts/subsample.py`

**Interfaces:**
- Produces: `subsample.py TRAJ OUT.xyz --frames N` (evenly spaced, always keeps first and last); remote `$REMOTE/04_md/md_300frames.xyz`.

- [ ] **Step 1:** Script: `bilayer` on relaxed slab → `optimize run` (fmax 0.02, bfgs) → `md run` with the Global MD flags, `--traj-interval 10`, `--resume`-safe.
- [ ] **Step 2:** Launch, verify, wait (~hours; background notification, no polling agent). Completion: last step in `md_energy.csv` ≥ 10000.
- [ ] **Step 3:** On the cluster: `subsample.py md.traj md_300frames.xyz --frames 300`. Report mean T over the last 4 ps and whether any water left the surface (O height > 5 Å) or dissociated (O–H > 1.3 Å).
- [ ] **Step 4:** Commit: `feat(showcase): water bilayer MD run`.

### Task 6: Pull back, data pack, provenance

**Files:**
- Create: `docs/showcase/scripts/build_pack.py`, `docs/showcase/data/pack/PROVENANCE.md`
- Test: `docs/showcase/tests/test_build_pack.py`

**Interfaces:**
- Produces in `docs/showcase/data/pack/`: `relax.xyz`, `committee.json`, `neb_band.xyz`, `ts_mode.xyz`, `md.xyz`, and one JSON per use case `{relax,committee,neb,md,webapp}.json` with schema:
  `{"use_case": str, "command": str, "converged": bool, "flags": [str], "key_numbers": [{"label": str, "value": float, "unit": str}], "series": {name: [float]}, "frames": str|null}`.
- Frame files ≤ 300 frames, multi-frame XYZ with lattice in extended-XYZ header.

- [ ] **Step 1:** rsync (`--update`, no `*.traj` except subsampled) `$REMOTE` → `docs/showcase/data/runs/`.
- [ ] **Step 2: Failing tests** on small synthetic fixtures:
  - `test_not_converged_is_labeled`: a fake optimize log whose last fmax 0.03 > 0.02 → `converged is False`.
  - `test_frames_capped`: 1000-frame input → output has 300 frames, first and last kept.
  - `test_schema`: every JSON has all schema keys and every key number has a non-empty unit.
  - `test_geometry_flag`: O below top-layer height → `flags` contains `"o_subsurface"`; water O–H > 1.3 Å → `"water_dissociated"`.
  - `test_command_is_verbatim`: `command` equals the command line recorded in the run's log/run record.
- [ ] **Step 3:** FAIL → implement → PASS.
- [ ] **Step 4:** Build the pack from the real runs. `ts_mode.xyz`: 30 frames of the TS ± 0.3 Å along the imaginary mode (sinusoid), from the verify output.
- [ ] **Step 5:** `prov update /Users/juar/github/work/mlip_platform mliprun-showcase --subtree docs/showcase/data/runs` then `prov sync`; `prov show mliprun-showcase` lists every run. Write `PROVENANCE.md` mapping each pack file to its run directory and prov id.
- [ ] **Step 6:** Commit pack + scripts + tests: `feat(showcase): data pack with provenance`.

### Task 7: Web app remote-run recording (use case 5)

**Files:**
- Create: `docs/showcase/scripts/record_webapp.py`, `docs/showcase/data/pack/webapp.mp4`, `webapp.json`

- [ ] **Step 1:** Confirm SSH master to cos-cluster alive (else ask Juan). Start `mlip webapp` locally (it only submits; the calculation runs on cos-cluster).
- [ ] **Step 2:** `record_webapp.py`: Playwright, 1440×900, records video while it uploads `01_relax/o_fcc` input, picks machine `cos-cluster` and the UMA env, sets uma-s-1p2/oc20/fmax 0.02/bfgs, submits, waits for "done", opens the result view. Use the app's `data-testid` selectors.
- [ ] **Step 3:** Trim to ≤ 30 s with ffmpeg (speed up the waiting part, label it "×N"); write `webapp.json` (command = equivalent CLI line, converged from the job's result).
- [ ] **Step 4:** Add the job dir to `prov` (Task 6 Step 5 command). Commit: `feat(showcase): web app remote-run recording`.

### Task 8: Shared media renderer

**Files:**
- Create: `site/shared/viewer.js` (3Dmol scene loader: `loadScene(el, xyzUrl, opts) -> {play(), pause(), setFrame(i)}`; consistent element colours, Pt as metallic grey, O red, H white; respects `prefers-reduced-motion`), `site/shared/capture.py` (Playwright: renders a scene page and writes PNG frames → ffmpeg GIF and MP4, ≤ 5 MB each)
- Test: `site/shared/test_capture.py` (one GIF from `relax.xyz` exists, ≤ 5 MB, frame count > 10)

- [ ] **Step 1–4:** failing test → implement → pass.
- [ ] **Step 5:** Commit: `feat(showcase): shared 3D viewer and media capture`.

### Task 9 (×5, parallel): Alternative n

One task per concept, same structure, run in parallel by five builders after Tasks 6–8.

| n | slug | concept (spec §"five design concepts") |
|---|---|---|
| 1 | terminal | Terminal hero |
| 2 | story | One system, full story (O chapters, then water scene) |
| 3 | gallery | Live gallery |
| 4 | handbook | Group handbook |
| 5 | cinematic | Cinematic 3D |

**Files:** `site/alt-<n>-<slug>/index.html`, `style.css`, `main.js`, `README.md`, `media/*`

**Interfaces:** Consumes the pack (Task 6 schema), `site/shared/viewer.js`, `site/shared/capture.py`. Reads data only from `../../docs/showcase/data/pack/` paths copied into `site/alt-<n>-<slug>/data/` at build (static hosting).

- [ ] **Step 1:** Builder loads the `frontend-design:frontend-design` and `artifact-design` skills; reads the spec's anti-goals.
- [ ] **Step 2:** Landing page shows all five use cases with real numbers from the pack (no hard-coded numbers; each number has a unit; `converged: false` items shown as such).
- [ ] **Step 3:** README variant: same concept, GitHub-safe (no script, relative media links, works in GitHub light and dark), links to the landing page, keeps install pointer and the badges.
- [ ] **Step 4: Checks** (Playwright script `site/alt-<n>-<slug>/check.py`): no console errors; no horizontal scroll at 375 px; with reduced motion no animation runs; every `<img>/<video>` in README.md resolves to an existing file; total landing page weight ≤ 15 MB.
- [ ] **Step 5:** Publish `index.html` as a private Artifact (preview); record the URL in the task report.
- [ ] **Step 6:** Commit: `feat(showcase): alternative <n> — <slug>`.

### Task 10: Independent review and fixes

- [ ] **Step 1:** For each alternative, a fresh reviewer (not its builder) scores against: spec anti-goals, data fidelity (every number traces to the pack), visual quality (hierarchy, typography, motion purpose), GitHub rendering, mobile, accessibility (contrast AA, alt text). Output: ranked must-fix / should-fix list with screenshots.
- [ ] **Step 2:** Builder fixes must-fix items; reviewer re-checks once.
- [ ] **Step 3:** Manager reviews all five (screenshots at 1440 and 375 px, plays every animation), rejects anything below bar back to its builder.
- [ ] **Step 4:** Open a draft PR from `showcase/front-page` with the five preview links, a one-line pitch per alternative, and the run summary (barrier, imaginary frequency, committee σ, MD T) with convergence labels.

Phase 5 (integrating the chosen design into `README.md` and GitHub Pages) is a separate plan after Juan picks.
