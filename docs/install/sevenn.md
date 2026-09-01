# SevenNet install recipe

_Last verified: 2026-09-01 — `sevenn` 0.13.0, torch 2.8.0+cu128, on cos-cluster (NVIDIA L40S, CUDA driver 595.71.05)._

Package: [`sevenn`](https://github.com/MDIL-SNU/SevenNet) — supplies the `SevenNetCalculator` used for every `7net-*` tag.

> **Don't install `sevenn` into an env that already has `mace-torch`, `fairchem-core`, or `chgnet`.** The torch / `torch_geometric` pins collide. Create a fresh env per [ADR 0001](../adr/0001-per-mlip-envs.md).

## Conda recipe (preferred for HPC)

This is the exact sequence used to build the verified env.

```bash
conda create -n mlip-sevenn python=3.11 -y
conda activate mlip-sevenn

# Pick ONE torch line depending on your hardware.
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
# pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu   # CPU

# torch_geometric is a hard requirement of sevenn; install it after torch.
pip install torch_geometric
pip install sevenn ase
pip install -e /path/to/mliprun   # or: pip install mliprun
```

`sevenn` does not pin `torch` itself, so the version is yours to choose. 2.8.0+cu128 is the one verified here; it is also the stack already proven on cos-cluster by the UMA env.

## Venv recipe (preferred for local dev)

```bash
python3.11 -m venv .venv-sevenn
source .venv-sevenn/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install sevenn ase
pip install -e /path/to/mliprun
```

## Sanity check

```bash
python -c "from sevenn.calculator import SevenNetCalculator; SevenNetCalculator('7net-omni', modal='mpa', device='cpu'); print('OK')"
```

No Hugging Face auth and no access request: the checkpoints are public.

### Where the weights go

Contrary to what this file said before it was verified, the weights are **not** bundled in the wheel. They are downloaded on first use into the installed package itself:

```
<env>/lib/python3.11/site-packages/sevenn/pretrained_potentials/<Model>/checkpoint_*.pth
```

Roughly 103 MB per checkpoint. Two consequences:

- The download lands inside the environment, not in `~/.cache`, so it follows the env's filesystem. On a cluster with a quota'd home, put the env on scratch and nothing else needs redirecting.
- An air-gapped compute node needs the checkpoint fetched beforehand. `sevenn cp <tag>` downloads it and prints the checkpoint summary without running the model.

## Models and tasks

SevenNet's multi-fidelity models expose several **inference tasks** (SevenNet's API calls them "modals"). A task is an independent fine-tune with its **own energy zero**, exactly like a UMA task or a MACE head.

> **CANON C3 applies.** Never mix tasks inside one energy formula (adsorption energy, reaction energy, γ). Choosing a task for one leg of a pipeline forces it on every leg.

Because of that, `--sevennet-task` has **no default**. A multi-task model with no task stops the run rather than picking one.

| Tag | Type | Tasks |
| --- | --- | --- |
| `7net-omni` | multi-task | `omat24`, `mpa`, `omol25_low`, `omol25_high`, `matpes_pbe`, `matpes_r2scan`, `mp_r2scan`, `oc20`, `oc22`, `spice`, `qcml`, `odac23`, `pet_mad` |
| `7net-omni-i8` | multi-task | same 13 |
| `7net-omni-i12` | multi-task | same 13 |
| `7net-mf-ompa` | multi-task | `omat24`, `mpa` |
| `7net-mf-0` | multi-task | `PBE`, `R2SCAN` |
| `7net-omat` | single-task | — |
| `7net-l3i5` | single-task | — |
| `7net-0` | single-task | — |
| `7net-0_22may2024` | single-task | — |

The list comes from `sevenn.util.get_available_pretrained_models()` and each checkpoint's `Modality` line, read on 2026-09-01. Three things differ from SevenNet's own documentation, which describes a release newer than 0.13.0:

- **`7net-nano-4.5` / `-5.0` / `-5.5` / `-6.0` are documented but absent** from the 0.13.0 registry. mliprun does not list them. If a later `sevenn` adds them they still work — any unrecognised `7net-*` tag is forwarded to SevenNet unchanged, with a warning that the tag and task cannot be checked.
- **`7net-0_22may2024` and `7net-mf-0` are in the registry** but missing from the documentation's overview table.
- **`7net-mf-0` names its tasks in uppercase** (`PBE`, `R2SCAN`) where every other model uses lowercase. Task names are matched **exactly**: `--sevennet-task pbe` is rejected, because `pbe` is not a task that checkpoint has.

An unknown task prints the valid list for that model, so `mlip optimize run --mlip 7net-omni` with no task tells you what to choose.

### Measured: the tasks really do sit on different energy zeros

Verified on cos-cluster (NVIDIA L40S, `--device cuda`) on 2026-09-01. System: O adsorbed on a Pt(111) 2x2x4 slab, 17 atoms, 10 A vacuum, bottom two layers fixed. These are software smoke-test numbers, not converged science.

| Model | Task | Single-point energy | Model load | Single point |
| --- | --- | --- | --- | --- |
| `7net-omni` | `mpa` | −97.583282 eV | 15.4 s | 0.98 s |
| `7net-omni` | `oc20` | −88.172501 eV | 8.1 s | 1.05 s |
| `7net-mf-ompa` | `mpa` | −97.404488 eV | 7.4 s | 1.07 s |
| `7net-0` | — (single-task) | −97.958290 eV | 5.6 s | 0.19 s |

**The same model on the same structure differs by 9.410782 eV between `mpa` and `oc20`.** That is the concrete reason `--sevennet-task` has no default and why CANON C3 forbids mixing tasks inside one formula: an adsorption energy built from one leg at `mpa` and another at `oc20` would be wrong by roughly that amount, and nothing in the output would say so.

The `7net-0` row is the single-task path: no `modal` argument is passed at all, which is what that checkpoint requires.

### End-to-end validation

Same machine and date, `7net-omni` with `--sevennet-task mpa` on `--device cuda`:

- `optimize run` (fmax 0.05 eV/A, BFGS): converged. Relaxed geometry checked rather than assumed — maximum atomic displacement 0.163 A, O in a threefold hollow at 2.044 A from three Pt, no merged atoms, formula unchanged.
- `md run` (NVT, Langevin, 300 K, 0.5 fs, 50 steps): completed.
- `neb run` (5 intermediate images, k = 0.1, climbing image, fmax 0.05 eV/A): completed, maximum 0.377 eV above the initial image on a deliberately coarse smoke-test band.
- Both error paths confirmed to exit non-zero: a multi-task tag with no task, and a task passed to a single-task tag.
- All three `mliprun_run.json` records carry `schema_version: 3`, `provenance.sevennet_task: "mpa"`, and `device_resolved: "cuda"`; `uma_task` and `mace_head` are null.


---

## Tag: `7net-omni`

SevenNet's recommended model and the tag `--mlip auto` resolves to when `sevenn` is the only MLIP installed. Trained on 15 open *ab initio* datasets, l_max = 3, 5 interaction layers.

It is the only SevenNet model carrying **surface heads** — `oc20` (RPBE) and `oc22` — which is what makes it useful for catalysis work; `7net-mf-ompa` has neither.

Use `mpa` (PBE+U) as the general-purpose PBE-level task, and `oc20` for adsorption on surfaces. Being multi-task, it requires `--sevennet-task`:

```bash
optimize run --structure POSCAR --mlip 7net-omni --sevennet-task oc20 --device cuda
```

## Tag: `7net-omni-i8`

Same training strategy and task list as `7net-omni`, with 8 interaction layers instead of 5. More accurate, slower.

## Tag: `7net-omni-i12`

Same again with 12 interaction layers and partial parity. The most accurate SevenNet checkpoint in this release, and the slowest.

## Tag: `7net-mf-ompa`

The earlier multi-fidelity model (MPtrj, sAlex, OMat24). Two tasks, `omat24` and `mpa`, neither of them a surface head. Superseded by `7net-omni` for most purposes; kept because existing work may reference it.

## Tag: `7net-mf-0`

The first multi-fidelity SevenNet model. Tasks `PBE` and `R2SCAN`, uppercase.

## Tag: `7net-omat`

Single-task, trained on OMat24. No `--sevennet-task`; passing one is an error.

## Tag: `7net-l3i5`

Single-task, trained on MPtrj, l_max = 3.

## Tag: `7net-0`

The original SevenNet model, single-task, trained on MPtrj. `7net-0_22may2024` is an earlier snapshot of it.
