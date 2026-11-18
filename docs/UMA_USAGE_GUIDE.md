# UMA Models Usage Guide

This guide explains how to use UMA (Universal Models for Atoms, FAIRChem) models in the mliprun.

## Supported Model

| Model | Description |
|-------|-------------|
| **uma-s-1p2** | Current default. Used when `--mlip auto` resolves to UMA, or when `--mlip uma-s-1p2` is set explicitly. |

The platform's MLIP validator accepts any tag that starts with `uma-` and forwards it to FAIRChem (see `src/mliprun/cli/utils.py`), so older releases such as `uma-s-1p1` or `uma-m-1p1` still load if you specify them and have the corresponding weights cached. They are no longer documented here; new work should use `uma-s-1p2`.

UMA supports four task heads:

| Task | Use case |
|------|----------|
| **omat** | General inorganic materials, crystals, surfaces (default) |
| **oc20** | Catalysis, adsorbates on metal surfaces |
| **omol** | Molecular systems, organic molecules |
| **odac**  | ODAC dataset (direct air capture) |

Pass the task with `--uma-task` (CLI) or `uma_task=` (Python API).

---

## Installation

### 1. Install fairchem-core

```bash
pip install fairchem-core
```

Then re-install mliprun so the entry points are registered against the right environment:

```bash
pip install -e .
```

### 2. Hugging Face access

UMA weights live in a gated repo on Hugging Face. Without access the first model load fails with a 401. Steps:

1. Create or sign in to a Hugging Face account at <https://huggingface.co/join>. Use a work or institutional address — personal accounts are sometimes rejected.
2. Open the UMA model page (<https://huggingface.co/facebook/UMA>) and click the access-request button.
3. Wait for the request to switch from `PENDING` to `ACCEPTED` under **Settings → Gated Repositories**.
4. Generate a **Read** token under **Settings → Access Tokens**. Copy it; HF will not show it again.
5. Log in from a terminal:

   ```bash
   pip install huggingface_hub  # if not already installed
   hf auth login
   ```

   Paste the token when prompted. The credential is cached to `~/.cache/huggingface/token` (or `~/.huggingface/token` on older versions) and persists across sessions.

For a Windows / PowerShell walkthrough of the same steps, see [windows_setup_guide.md](windows_setup_guide.md).

---

## CLI Usage

All CLI examples below assume the package is installed (`pip install -e .`) so `optimize`, `md`, `neb`, and `autoneb` are on `PATH`. Every command takes a `run` subcommand. Use `<command> run --help` for the full option list.

### Geometry optimization

```bash
# Auto-detect MLIP (resolves to uma-s-1p2 if fairchem-core is installed)
optimize run --structure structure.vasp

# Explicit model + task
optimize run --structure structure.vasp --mlip uma-s-1p2 --uma-task omat

# Tighter convergence
optimize run --structure structure.vasp --mlip uma-s-1p2 --uma-task omat --fmax 0.01

# Different optimizer (default is bfgs; fire and lbfgs are common alternatives)
optimize run --structure structure.vasp --mlip uma-s-1p2 --optimizer fire
optimize run --structure structure.vasp --mlip uma-s-1p2 --optimizer lbfgs
```

Task variations:

```bash
# Catalysis / adsorbates on surfaces
optimize run --structure slab.vasp --mlip uma-s-1p2 --uma-task oc20

# Molecules
optimize run --structure mol.vasp  --mlip uma-s-1p2 --uma-task omol

# ODAC
optimize run --structure cell.vasp --mlip uma-s-1p2 --uma-task odac
```

### Nudged Elastic Band (NEB)

```bash
# Auto-detect MLIP
neb run --initial initial.vasp --final final.vasp

# Explicit UMA + task
neb run --initial initial.vasp --final final.vasp \
    --mlip uma-s-1p2 --uma-task omat

# More images, tighter fmax
neb run --initial initial.vasp --final final.vasp \
    --mlip uma-s-1p2 --uma-task omat \
    --num-images 9 --fmax 0.01
```

### Molecular dynamics

```bash
# Auto-detect MLIP, NVT (Langevin) defaults
md run --structure structure.vasp --steps 1000 --temperature 300

# Explicit UMA + task
md run --structure structure.vasp \
   --mlip uma-s-1p2 --uma-task omat \
   --steps 1000 --temperature 300 --timestep 1.0

# Higher temperature with shorter timestep
md run --structure structure.vasp \
   --mlip uma-s-1p2 --uma-task omat \
   --steps 5000 --temperature 1000 --timestep 0.5
```

### AutoNEB

```bash
autoneb run --initial initial.vasp --final final.vasp \
    --mlip uma-s-1p2 --uma-task omat \
    --n-max 11 --fmax 0.03
```

### Benchmarking

```bash
benchmark run --structure structure.vasp --models uma-s-1p2 --uma-task omat
benchmark run --structure structure.vasp --models uma-s-1p2,mace --output bench.json
```

`benchmark run` times one `get_potential_energy()` call per model in-process. With no `--models`, it benchmarks every MLIP installed in the current environment. See the [command reference](CLI.md#benchmark) for the full option list.

---

## Model selection guidelines

`uma-s-1p2` covers the majority of solid-state and surface-chemistry workflows on this platform. Practical guidance:

- **Pick the right task head, not the model size.** Switching from `omat` to `oc20` for a catalysis problem typically matters more than swapping model size.
- **Tighten `--fmax` for production runs.** The model is fast enough that 0.01 eV/Å is realistic for relaxation and NEB endpoints.
- **For very large systems (>500 atoms),** memory becomes the limit before runtime. Drop intermediate trajectories you don't need (`--trajectory /dev/null` is not supported; instead, run optimization and delete the file afterwards).

### Performance characteristics (qualitative)

- `uma-s-1p2` is fast and CPU-friendly; energies + forces evaluate in fractions of a second per call on modest hardware for ~100-atom cells.
- Accuracy is sufficient for screening and most NEB barrier work; for publication-quality energetics, validate against DFT on a small representative set.
- Memory scales roughly linearly with the number of atoms; expect to need GPU or a high-memory CPU node well before 1000 atoms.

(No fixed wall-time numbers are quoted here — they vary too much across hardware to be useful as documentation. Use `benchmark run --models uma-s-1p2 --output bench.json` to measure on your own machine.)

---

## Programmatic usage (Python API)

### Geometry optimization

```python
from ase.io import read
from ase.optimize import FIRE
from fairchem.core import pretrained_mlip, FAIRChemCalculator

atoms = read("structure.vasp")

predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cpu")
atoms.calc = FAIRChemCalculator(predictor, task_name="omat")

opt = FIRE(atoms, trajectory="opt.traj")
opt.run(fmax=0.05, steps=200)

print(f"Final energy: {atoms.get_potential_energy():.6f} eV")
```

The platform also exposes a thin helper that does the same calculator wiring:

```python
from ase.io import read
from mliprun.cli.utils import setup_calculator

atoms = read("structure.vasp")
atoms = setup_calculator(atoms, mlip="uma-s-1p2", uma_task="omat")
```

### NEB with the platform's wrapper

```python
from ase.io import read
from ase.optimize import FIRE
from mliprun.core.neb import CustomNEB

initial = read("initial.vasp", format="vasp")
final   = read("final.vasp",   format="vasp")

neb = CustomNEB(
    initial=initial, final=final,
    num_images=7,
    fmax=0.05,
    mlip="uma-s-1p2", uma_task="omat",
    output_dir=".",
)
neb.interpolate_idpp()
neb.run_neb(optimizer=FIRE, climb=True, max_steps=600)
df = neb.process_results()
neb.plot_results(df)
neb.export_poscars()
```

`CustomNEB` itself wires the FAIRChem calculator onto each image; you do not need to call `setup_calculator` separately when using it.

---

## Troubleshooting

### `Could not find model 'uma-s-1p2'`

`fairchem-core` is not installed in the active environment.

```bash
pip install fairchem-core
```

### 401 / "You don't have access to this gated repo"

You haven't been granted access to `facebook/UMA` on Hugging Face, or you haven't logged in on this machine. Re-run the steps in [Installation → Hugging Face access](#2-hugging-face-access).

### Memory errors on large systems

The platform configures UMA for CPU only. For systems beyond ~500 atoms you will likely need to:

- Reduce the cell (run on a subsystem first),
- Or move the calculation to a node with substantially more RAM,
- Or switch to GPU by editing `cli/utils.py` (`setup_calculator` hard-codes `device="cpu"`).

### Wrong task head silently selected

`--uma-task` defaults to `omat` everywhere. If you forget to set it for an OC20-style problem you will get plausible but wrong energies. Always set `--uma-task` explicitly when the system is not bulk inorganic.

---

## Output files

The output files produced by each command are listed in [OUTPUTS.md](OUTPUTS.md). UMA-specific runs add the model name and (for UMA) the task name to the parameter file written by every command (`opt_params.txt`, `md_params.txt`, `neb_parameters.txt`, `autoneb_parameters.txt`).

---

## Complete example workflow

A typical workflow runs NEB (which relaxes both endpoints by default) and then probes the saddle point with MD:

```bash
# 1. NEB between two endpoint guesses. With --optimize-endpoints (the default),
#    this relaxes initial.vasp and final.vasp before interpolating.
neb run --initial initial.vasp --final final.vasp \
    --mlip uma-s-1p2 --uma-task omat \
    --num-images 7 --fmax 0.05

# 2. Probe the saddle with finite-temperature MD.
#    Pick the highest-energy image from the NEB output (check neb_data.csv);
#    each image's structure is written to NN/POSCAR.
md run --structure 04/POSCAR \
   --mlip uma-s-1p2 --uma-task omat \
   --ensemble nvt --temperature 500 --steps 5000 --timestep 1.0
```

If you want separate `optimize run` passes on each endpoint, run them in different directories. The optimizer writes to the directory containing its input, and the final structure filename (`opt_final.vasp`) is fixed, so two relaxations in the same directory will overwrite each other unless you also pass `--logfile <name>.log` to change the output prefix.

---

## Auto-detection priority

When `--mlip auto` is used (the default for every command), `cli/utils.py:detect_mlip()` resolves in this order:

1. **UMA** (`uma-s-1p2`) if `fairchem-core` is importable.
2. **SevenNet** (`7net-mf-ompa`) if `sevenn` is importable.
3. **MACE** (`mace`) if `mace-torch` is importable.
4. Otherwise, the command exits with an error.

To bypass auto-detection, pass `--mlip <name>` explicitly. Any tag starting with `uma-` is forwarded to FAIRChem unchanged.
