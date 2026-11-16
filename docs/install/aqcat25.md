# AQCat25-EV2 install recipe

_Status: the CPU-only variant was built and run end to end on macOS arm64 (Python 3.11.10) on 2026-11-16: `mlip doctor`, `singlepoint run`, `optimize run` and `neb run` against the real `aqcat25-ev2-midfilm` checkpoint. The GPU (CUDA 12.1) steps follow the upstream model card and have not been run by us yet._

Models: SandboxAQ's AQCat25-EV2 checkpoints, EquiformerV2 with FiLM conditioning on spin and DFT fidelity (Allam et al., *npj Comput. Mater.* **12**, 226 (2026)). They run on **fairchem-core 1.10.0** patched with two SandboxAQ files. That distribution has the same name, `fairchem-core`, as the 2.x line UMA uses, so mliprun tells them apart by major version: 2.x is UMA, 1.x is AQCat25-EV2.

> **Give this its own env.** Never install it into an env that holds UMA (`fairchem-core` 2.x), `mace-torch`, `sevenn` or `chgnet`. The torch and `torch_geometric` pins collide, and two `fairchem-core` lines cannot coexist at all. See [ADR 0001](../adr/0001-per-mlip-envs.md).

## Access and licence

- The checkpoints are on the **gated** Hugging Face repo [`SandboxAQ/aqcat25-ev2`](https://huggingface.co/SandboxAQ/aqcat25-ev2). Fill in the access form on the repo page; approval is automatic.
- Licence: **CC BY-NC-SA 4.0**, non-commercial use only.
- mliprun ships **none** of the SandboxAQ files (checkpoints or patched code). You download them yourself, under that licence.
- mliprun pins the repo revision `2a7d5574aa7f0addc0678080eb7621a47a892abd`, so a tag always names the same checkpoint file.

## Venv recipe (GPU, CUDA 12.1)

Python 3.10 is what the model card specifies.

```bash
python3.10 -m venv .venv-aqcat
source .venv-aqcat/bin/activate

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install e3nn submitit torchtnt hydra-core pymatgen ase orjson wandb tensorboard \
    lmdb huggingface-hub numba datasets pandas tqdm requests "numpy<2" "scipy<1.15"

# Log in to Hugging Face (after your access request was approved).
hf auth login
```

Download the patched code (and the FiLM model definitions) from the gated repo:

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="SandboxAQ/aqcat25-ev2",
    allow_patterns=["ev2_film/*", "patched_code/*"],
    local_dir="./aqcat25-ev2",
)
EOF
```

Build fairchem-core 1.10.0 with the two SandboxAQ files on top:

```bash
git clone https://github.com/facebookresearch/fairchem.git
cd fairchem
git checkout -b aqcat-ev2 tags/fairchem_core-1.10.0

cp ../aqcat25-ev2/ev2_film/equiformer_v2_film.py \
   packages/fairchem-core/src/fairchem/core/models/equiformer_v2/
cp ../aqcat25-ev2/patched_code/ase_utils.py \
   packages/fairchem-core/src/fairchem/core/common/relaxation/ase_utils.py

pip install -e packages/fairchem-core --no-deps
cd ..
```

Then mliprun itself, and the check:

```bash
pip install -e /path/to/mliprun
mlip doctor
```

`mlip doctor` should list `AQCat25-EV2  fairchem-core  1.10.…` as installed, say that `--mlip auto` does not resolve in this env, and exit 0. If it lists the version under UMA instead, the env holds `fairchem-core` 2.x, not the patched 1.10.0.

If the patch is missing, the first run stops with an error saying the env has fairchem-core 1.x but not the SandboxAQ patch (`patched_calc` is absent from `fairchem.core.common.relaxation.ase_utils`).

## CPU-only variant

Verified on macOS arm64 with Python 3.11.10 (the model card asks for 3.10; 3.11 works). Only the torch and PyG wheel lines differ from the GPU recipe; every other step is the same.

```bash
python3.11 -m venv .venv-aqcat
source .venv-aqcat/bin/activate

pip install torch==2.4.0
pip install "numpy<2" torch_geometric
pip install --only-binary=:all: torch_sparse torch_scatter \
    -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install e3nn submitit torchtnt hydra-core pymatgen ase orjson wandb tensorboard \
    lmdb huggingface-hub numba datasets pandas tqdm requests "numpy<2" "scipy<1.15"
```

Versions this resolved to: torch 2.4.0, torch_geometric 2.8.1, torch_scatter 2.1.2, torch_sparse 0.6.18, e3nn 0.6.0, numpy 1.26.4, scipy 1.14.1, ase 3.29.0, fairchem-core 1.10.0.

- **`scipy<1.15` is required.** The model card has no scipy pin, and with a newer scipy fairchem 1.10.0 fails at import (`cannot import name 'sph_harm' from 'scipy.special'`).
- Speed on a laptop CPU: about 1.5 s to load the model and about 1.3 s per evaluation for a 50-atom slab.
- On load, fairchem prints its whole model configuration and two deprecation warnings about the head class names. Both are harmless.

## First run

- **Checkpoint download.** The checkpoint (about 126 MB) downloads into the Hugging Face cache on first use. That needs the login above; without it, mliprun stops with an error that points back to this recipe.
- **Air-gapped compute nodes.** Pre-fetch the checkpoint on a node with network access, into the same Hugging Face cache the compute node will read (`HF_HOME`):

  ```bash
  python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('SandboxAQ/aqcat25-ev2', \
  'checkpoints_aqcat_ev2/ev2-in+midFiLM-AQCat25+OC20-20M_20251008_223220.pt', \
  revision='2a7d5574aa7f0addc0678080eb7621a47a892abd')"
  ```

  For `aqcat25-ev2-infilm`, use the file name in its section below.
- **Device.** `cpu` or `cuda` only. Any other resolved device (for example `mps`) is an error.

## Using the models in mliprun

```bash
optimize run --structure POSCAR --mlip aqcat25-ev2-midfilm --aqcat-spin off --device cuda
singlepoint run --structure POSCAR --mlip aqcat25-ev2-midfilm --aqcat-spin on --aqcat-fidelity low
neb run --initial IS.vasp --final FS.vasp --mlip aqcat25-ev2-infilm --aqcat-spin off --device cuda
```

- **`--aqcat-spin on|off` is required**, with no default. Spin on versus off changes the predicted energy by up to several eV, so mliprun never lets upstream guess it from the elements.
- **`--aqcat-fidelity high|low`** is optional and defaults to `high`. `high` is the AQCat25 DFT settings (RPBE, 500 eV cutoff); `low` is the OC20 settings (RPBE, 350 eV).
- Both values go into `opt_params.txt` / `neb_parameters.txt` and into the run record (`provenance.aqcat_spin`, `provenance.aqcat_fidelity`; schema 6). A NEB restart reads them back.
- `--mlip auto` never picks these models. Pass the tag explicitly.

### Supported commands

`singlepoint run`, `optimize run`, `optimize batch` and `neb run` (new run and `--restart`). Everything else refuses the tags: `md`, `freq`, `autoneb`, `benchmark`, committee members, and the web app.

Inside `optimize`, two more refusals, both raised before any model loads:

- `--relax-cell`: the models predict no stress.
- Optimizers `bfgsls` and `gpmin`: they assume the forces are the gradient of the energy, and these are not (see below). `fire`, `bfgs`, `lbfgs` and `mdmin` work.

### One flag set per process

Upstream's `patched_calc` applies the spin and fidelity by patching a class attribute on every build, so within one Python process the most recent flags apply to every AQCat calculator. mliprun therefore allows one (spin, fidelity) pair per process: building a second calculator with a different pair is an error. The same pair again is fine (NEB builds one calculator per image). To compare spin on and spin off, run two separate commands.

### What the numbers mean

- **Energy:** an OC20-style **referenced adsorption energy**, not a DFT total energy. Do not combine it with total energies from VASP or from another MLIP.
- **Forces:** a direct output head, so they are **not the gradient** of the energy. An optimizer that converges on these forces is converging on the force head, not on the energy surface. One measurement of the size of the mismatch (CO on a Co(0001) slab at an unrelaxed geometry, maximum force 3.2 eV/Å, `aqcat25-ev2-midfilm`, spin on): the model's force on the C atom along z was −3.19 eV/Å, while the finite difference of the model's own energy gave −4.44 eV/Å. The mismatch near a minimum or a saddle point has not been measured. Treat NEB barriers with care: the band follows the forces, while the barrier is read from the energies.
- **No stress.** `singlepoint` reports the stress as unavailable.

Each supported command prints this caveat next to the spin and fidelity lines.

### Limits stated by the authors

- Weak on purely organic materials.
- Antiferromagnetic orderings are not distinguished.

---

## Tag: `aqcat25-ev2-midfilm`

The authors' recommended model. Trained from scratch on AQCat25 plus 20M OC20 structures.

Checkpoint: `checkpoints_aqcat_ev2/ev2-in+midFiLM-AQCat25+OC20-20M_20251008_223220.pt`

## Tag: `aqcat25-ev2-infilm`

Cotuned from the OC20 EquiformerV2-31M model on AQCat25 plus 20M OC20 structures.

Checkpoint: `checkpoints_aqcat_ev2/ev2-inFiLM-OC20-ft-AQCat25+OC20-20M_20251008_223143.pt`
