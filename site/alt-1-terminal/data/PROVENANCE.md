# Provenance of the showcase data pack

Every number and frame in this folder comes from a run in `../runs/`, pulled back from
cos-cluster (`/scratchb/juar/mliprun-showcase`, node `cosrnode01g`, one GPU card) on
2026-11-17. `../../scripts/build_pack.py` builds the pack from those files; it runs no
calculation.

## Code and environment

- **mliprun code that ran:** commit `2b3044b` (`2b3044b69ebd135cf3dd2a78365be88389d8d33a`,
  "feat(optimize): --relax-cell all|volume|shape (#66)"). It was a clone at
  `/scratchb/juar/mliprun-showcase/mliprun`, put ahead of the shared install through
  `PYTHONPATH` (see `../../cluster.env`).
- The run records say `mliprun_version` 0.4.0. That string is the package metadata of the
  environment, not the code that ran; the code is commit `2b3044b`.
- ASE 3.28.0, Python 3.11.15, fairchem-core 2.19.0 (uma-s-1p2, task oc20) for every
  single-model run. The committee members ran in their own environments (uma, sevenn, mace)
  listed in `../runs/02_committee/opt_params.txt`.
- Settings: `../../CANON.md`. Run scripts: `../../scripts/run_0{1,2,3,4}_*.sh`.

## Pack file → run directory → prov id

The prov ledger is `calc-provenance/mliprun-showcase/` (ledger commit `2d0b254`). A prov id
is the run directory relative to the repository root.

| Pack file | Source run directory (under `../runs/`) | Source file(s) | prov id |
|---|---|---|---|
| `relax.json` | `01_relax/o_fcc` (+ `bulk`, `slab`, `o_hcp`) | `mliprun_run.json`, `opt.log`, `opt_convergence.csv`, `opt_final.vasp`, `bulk/fcc_check.txt` | `docs/showcase/data/runs/01_relax/o_fcc`; also `.../01_relax/bulk`, `.../01_relax/slab`, `.../01_relax/o_hcp` |
| `relax.xyz` | `01_relax/o_fcc` | `opt_traj.xyz` (all 13 steps of `opt.traj`, converted on the cluster with `subsample.py`) | `docs/showcase/data/runs/01_relax/o_fcc` |
| `committee.json` | `02_committee` | `mliprun_run.json`, `opt.log`, `opt_committee.csv`, `opt_committee_peratom.csv`, `opt_final.vasp` | `docs/showcase/data/runs/02_committee` |
| `neb.json` | `03_neb` (+ `03_neb/ts_verify`) | `mliprun_run.json`, `neb_data.csv`, `neb_convergence.csv`, `band_check.json`, `ts_verify/mliprun_run.json` | `docs/showcase/data/runs/03_neb` |
| `neb_band.xyz` | `03_neb` | `A2B_band.xyz` (the 9-image final band, `A2B.traj`) | `docs/showcase/data/runs/03_neb` |
| `neb_seed.xyz` | `03_neb` | `idpp_seed.xyz` (the IDPP seed band, `idpp.traj`) | `docs/showcase/data/runs/03_neb` |
| `ts_mode.xyz` | `03_neb/ts_verify` | `ts_freq_vibrations.json` (Hessian, imaginary mode), `ts_structure.vasp` | `docs/showcase/data/runs/03_neb` (ts_verify is a stage of the NEB run, not a separate prov node) |
| `md.json` | `04_md/bilayer_relax` | `mliprun_run.json`, `opt.log`, `opt_final.vasp` | `docs/showcase/data/runs/04_md/bilayer_relax` |
| `md_bilayer_built.xyz` | `04_md/bilayer_relax` | `POSCAR` (as built by `build_structures.py bilayer`) | `docs/showcase/data/runs/04_md/bilayer_relax` |
| `md_bilayer_relaxed.xyz` | `04_md/bilayer_relax` | `opt_final.vasp` | `docs/showcase/data/runs/04_md/bilayer_relax` |

## Status of each use case

- **relax:** converged (last fmax 0.0134 eV/Å, target 0.02 eV/Å). bulk, slab and o_hcp
  converged as well.
- **committee:** converged (committee-mean force 0.0173 eV/Å, target 0.02 eV/Å). The
  final `sigma_max_free` (0.0637 eV/Å) is 3.7 times the final fmax.
- **neb:** converged (band fmax 0.0177 eV/Å, target 0.02 eV/Å). TS verified: one imaginary
  mode of 128.3i cm⁻¹, overlap with the NEB tangent 0.962, and the displace-and-relax check
  connects both ways. Flag: ts_verify measured endpoint fmax 0.0134 (IS) and 0.0174 (FS)
  eV/Å, above its internal 0.01 eV/Å relax fmax; the verdict is unaffected. The band did
  not explode.
- **md:** status `pending`; the MD did **not** run. The 6-H2O bilayer relaxation with
  uma-s-1p2/oc20 converged, and the intact water layer ended at O heights 4.7–5.2 Å above
  the top Pt layer (water check limit 4.0 Å), so the run script stopped before MD. How to
  proceed is pending the scientist's decision (flag `pending_scientist_decision`).
  `md.json` carries only the bilayer-relaxation facts, labelled as relaxation, with
  `converged: false` and no frames. There is no `md.xyz`. Two one-frame structures are in
  the pack: `md_bilayer_built.xyz` (as built, the input to the bilayer relaxation) and
  `md_bilayer_relaxed.xyz` (after the uma-s-1p2/oc20 relaxation; the planned MD start in
  `run_04_md.sh`).
- **webapp:** converged (last fmax 0.0134 eV/Å, target 0.02 eV/Å, 12 BFGS steps). The
  local web app (`mlip webapp`) submitted the unrelaxed `01_relax/o_fcc/POSCAR` to
  cos-cluster env `cos-cluster-uma` (shared install at commit `2b3044b`), GPU card 0, on
  2026-11-18. Job directory `../runs/05_webapp/showcase-o-fcc-webapp-2` (remote
  `/scratchb/juar/mlip-webapp/showcase-o-fcc-webapp-2`); final energy differs from
  `relax.json` by +2.2e-6 eV. `webapp.json` and `webapp.mp4` are built by
  `../../scripts/record_webapp.py`, not by `build_pack.py`. Not in the prov ledger: prov has
  no adapter for a web app optimize job (no `opt_params.txt`).

## What was not pulled

Raw `*.traj` files, `*.png` plots, the ts_verify finite-difference cache
(`ts_verify/ts_freq/`, its Hessian is in `ts_freq_vibrations.json`), the mliprun clone,
`cluster.env`, `committee.yaml`, `scripts/` and `diagnostics/` stay on the cluster.
Inputs without an extension (`POSCAR`, `CONTCAR`) were pulled but are not committed
(`../../.gitignore`), except `04_md/bilayer_relax/POSCAR`, which the pack build reads; prov
recorded them all.
