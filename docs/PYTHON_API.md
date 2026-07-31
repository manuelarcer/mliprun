# Python API Reference

The CLI commands wrap a small set of pure-Python functions and one class. This page documents the public surface that you can call directly from a script or notebook. For CLI usage see the [main README](../README.md); for UMA-specific examples see [UMA_USAGE_GUIDE.md](UMA_USAGE_GUIDE.md).

## Module map

| Module | Public symbol | Purpose |
|--------|---------------|---------|
| `mliprun.cli.utils` | `build_calculator(mlip, uma_task="omat", device="auto", mace_head="omat_pbe")` | Build and **return** an MLIP calculator (loads model weights once) without attaching it |
| `mliprun.cli.utils` | `setup_calculator(atoms, mlip, uma_task="omat", device="auto", mace_head="omat_pbe")` | Attach the right MLIP calculator to an `Atoms` object (thin wrapper over `build_calculator`) |
| `mliprun.cli.utils` | `detect_mlip()` / `validate_mlip(name)` / `resolve_mlip(name)` | Auto-detect / validate MLIP availability |
| `mliprun.core.optimize` | `run_optimization(atoms, ...)` | Geometry optimization with progress logging and plots |
| `mliprun.core.optimize` | `OPTIMIZER_MAP` | dict of supported ASE optimizers |
| `mliprun.core.md` | `setup_dynamics(atoms, ...)` | Build a configured ASE dynamics object |
| `mliprun.core.md` | `run_md(atoms, ...)` | Full MD run with logging, CSV, and plots |
| `mliprun.core.neb` | `CustomNEB(initial, final, ...)` | NEB/AutoNEB orchestration class |
| `mliprun.core.params_io` | `write_parameters_file(path, title, params)` | Standard parameter file writer used by every command |
| `mliprun.core.params_io` | `write_endpoint_results(path, results)` | Endpoint-optimization summary writer |
| `mliprun.core.utils` | `calc_fmax(forces)` | Maximum atomic force magnitude (eV/Å) |
| `mliprun.core.utils` | `GPA_TO_EV_PER_ANG3` | Pressure unit conversion constant |

Lazy imports keep startup fast: heavy MLIP packages (`fairchem`, `mace`, `sevenn`, `chgnet`) are only imported when their calculator is actually needed.

---

## Calculator setup

```python
from ase.io import read
from mliprun.cli.utils import setup_calculator

atoms = read("structure.vasp")
setup_calculator(atoms, mlip="uma-s-1p2", uma_task="omat")  # mutates atoms.calc
```

`setup_calculator` accepts:

- `mlip="mace"` → MACE-MP-0 medium
- `mlip="mace-mh-1"` (or any `mace-mh-*`) → multi-head MACE foundation model; `mace_head` selects the head (`omat_pbe` default, `oc20_usemppbe`, `matpes_r2scan`, `mp_pbe_refit_add`, `omol`, `spice_wB97M`)
- `mlip="7net-mf-ompa"` → SevenNet 7net-mf-ompa (mpa modal)
- `mlip="uma-..."` → any FAIRChem UMA tag, with `uma_task` selecting the head (`omat`, `oc20`, `omol`, `odac`)
- `mlip="chgnet"` → CHGNet default model

`device` selects the compute device (`"auto"` → cuda if available else cpu, or force `"cuda"` / `"cpu"`). `mace_head` is ignored for non-MH models; `uma_task` is ignored for non-UMA models.

To relax or evaluate **many** structures, load the model once with `build_calculator` and reuse the returned calculator, instead of calling `setup_calculator` (which rebuilds it) per structure. This is what `optimize batch` does internally:

```python
from mliprun.cli.utils import build_calculator

calc = build_calculator(mlip="uma-s-1p2", uma_task="omat")  # expensive: loads weights once
for atoms in many_structures:
    atoms.calc = calc                                        # cheap: reuse across all
    ...
```

Auto-detection is also exposed:

```python
from mliprun.cli.utils import detect_mlip, resolve_mlip

detect_mlip()              # returns first installed, in order: "uma-s-1p2" / "mace" / "7net-mf-ompa" / "chgnet"
resolve_mlip("auto")       # detect + echo to stdout — convenience wrapper
resolve_mlip("uma-s-1p2")  # validate availability + echo
```

---

## Geometry optimization

```python
from ase.io import read
from mliprun.cli.utils import setup_calculator
from mliprun.core.optimize import run_optimization

atoms = read("structure.vasp")
setup_calculator(atoms, "uma-s-1p2", "omat")

converged = run_optimization(
    atoms,
    optimizer="bfgs",   # or "fire", "lbfgs", "bfgsls", "gpmin", "mdmin"
    fmax=0.05,
    max_steps=200,
    relax_cell=False,   # True → also relax the cell (VASP ISIF=3-equivalent)
    output_dir="./relaxed",
    model_name="uma-s-1p2",
    uma_task="omat",    # or mace_head="omat_pbe" — see "Run-record keywords"
    plot=False,         # default: no PNG. Set True to write *_convergence.png (CSV always written)
)
```

Side effects (written to `output_dir`):

- `opt.traj` — full ASE trajectory
- `opt.log` — per-step force/energy log
- `opt_convergence.csv` / `opt_convergence.png` — step vs energy & fmax
- `opt_final.vasp` — final relaxed structure

`OPTIMIZER_MAP` is the canonical dict of supported optimizer names; access it if you need to validate or list options programmatically.

---

## Molecular dynamics

For an end-to-end run with logging, CSV, and plots:

```python
from ase.io import read
from mliprun.cli.utils import setup_calculator
from mliprun.core.md import run_md

atoms = read("structure.vasp")
setup_calculator(atoms, "uma-s-1p2", "omat")

run_md(
    atoms,
    ensemble="nvt",
    thermostat="langevin",
    temperature=300,
    timestep=1.0,
    friction=0.01,
    steps=10000,
    log_interval=10,    # rows appended to md_energy.csv every N steps
    traj_interval=100,  # frames written to md.traj every N steps
    output_dir="./md",
    model_name="uma-s-1p2",
    uma_task="omat",    # or mace_head="omat_pbe" — see "Run-record keywords"
)
```

For full control of the dynamics object (e.g. to attach extra observers, drive the integrator step-by-step, or stop early), use the lower-level builder:

```python
from mliprun.core.md import setup_dynamics

dyn = setup_dynamics(
    atoms,
    ensemble="npt",
    barostat="npt",     # MTK
    temperature=300,
    pressure=0.0,
    timestep=1.0,
    ttime=25.0,
)

dyn.attach(my_callback, interval=100)
dyn.run(50000)
```

`setup_dynamics` returns the corresponding ASE dynamics object (`Langevin`, `NoseHoover`, `NVTBerendsen`, `NPT`, `NPTBerendsen`, or `VelocityVerlet`). It is also where Maxwell-Boltzmann velocity initialization happens for NVT/NPT.

For the full parameter list and unit conventions, see [MD_REFERENCE.md](MD_REFERENCE.md).

---

## Run-record keywords

`run_optimization` and `run_md` both write a `mliprun_run.json` run record
([OUTPUTS.md](OUTPUTS.md#the-run-record)). These keywords exist only to fill
it in — none of them changes the physics — and all are optional:

| Keyword | Default | Notes |
|---------|---------|-------|
| `uma_task` | `None` | The UMA task head this run used. Recorded only when `model_name` starts with `uma-`. |
| `mace_head` | `None` | The MACE head this run used. Recorded only when `model_name` starts with `mace-mh-`. |
| `device_requested` | `"auto"` | The device as asked for. |
| `device_resolved` | `"auto"` | The device actually used (e.g. `"cuda"`). |
| `run_context` | `None` | A `RunContext` declaring the command, batch identity, and where each parameter value came from. Without it every parameter is tagged `unspecified` — mliprun never guesses. |

Pass the same head/task you gave `setup_calculator` / `build_calculator`.
These functions receive an `Atoms` object with a calculator already attached
and cannot interrogate it for the head, so an omitted `uma_task` is recorded
as "not determined" rather than guessed — CANON C1: the head is an explicit
decision, never inferred. The mismatched one is dropped rather than trusted,
so passing both is harmless.

The record is what lets you check, later, that two energies you are about to
subtract came from the same head. Filling these in is the difference between
a record that identifies the level of theory and one that does not.

---

## NEB / AutoNEB

`CustomNEB` is the orchestration class used by both the `neb` and `autoneb` CLI commands. It owns the MLIP setup, IDPP interpolation, endpoint optimization, NEB/AutoNEB runs, and result post-processing.

### Minimal NEB

```python
from ase.io import read
from ase.optimize import FIRE
from mliprun.core.neb import CustomNEB

initial = read("initial.vasp", format="vasp")
final   = read("final.vasp",   format="vasp")

neb = CustomNEB(
    initial=initial,
    final=final,
    num_images=7,        # intermediate images; total = num_images + 2
    fmax=0.05,
    mlip="uma-s-1p2",
    uma_task="omat",
    output_dir="./neb",
)
neb.interpolate_idpp()
neb.run_neb(optimizer=FIRE, climb=True, max_steps=600)
df = neb.process_results()
neb.plot_results(df)
neb.export_poscars()
```

`CustomNEB` itself wires the FAIRChem / MACE / SevenNet / CHGNet calculator onto each image, so you do not need to call `setup_calculator` separately.

### Constructor

| Parameter | Default | Notes |
|-----------|---------|-------|
| `initial`, `final` | required | ASE `Atoms`. `final` is re-celled to match `initial`. |
| `num_images` | `9` | Intermediate images only |
| `interp_fmax` | `0.1` | IDPP interpolation force threshold |
| `interp_steps` | `1000` | IDPP iteration limit |
| `fmax` | `0.05` | NEB convergence threshold |
| `mlip` | `"7net-mf-ompa"` | Pass `"uma-s-1p2"` etc. as needed |
| `uma_task` | `"omat"` | Used only when `mlip` starts with `"uma-"` |
| `output_dir` | `"."` | Created if missing |
| `relax_atoms` | `None` | List of indices to keep mobile (highly-constrained mode). All others are constrained with `FixAtoms`. IDPP is skipped in this mode. |
| `logfile` | `"neb.log"` | NEB iteration log filename |

### Methods

| Method | What it does |
|--------|--------------|
| `interpolate_idpp()` | Run IDPP interpolation between initial and final. Skipped automatically in highly-constrained mode. |
| `optimize_endpoints(endpoint_fmax=0.01, optimizer="BFGS", max_steps=200)` | Pre-relax both endpoints, return a results dict and write `initial_opt.*` / `final_opt.*` |
| `run_neb(optimizer=FIRE, trajectory="A2B.traj", full_traj="A2B_full.traj", climb=False, max_steps=600)` | Run NEB optimization, return the final image list |
| `run_autoneb(n_simul, n_max, k, climb, optimizer, space_energy_ratio, interpolate_method, maxsteps, prefix)` | Run AutoNEB with dynamic image insertion |
| `process_results()` | Returns a pandas `DataFrame` with one row per image and columns `image_index`, `energy`, `relative_energy` |
| `plot_results(df)` | Saves `neb_energy.png` (smoothed energy profile, barrier annotation) |
| `export_poscars()` | Writes `00/POSCAR`, `01/POSCAR`, ... for every image |
| `CustomNEB.load_from_restart(output_dir, mlip=None, uma_task=None, fmax=None, logfile=None, k=None, climb=None, neb_optimizer=None, neb_max_steps=None)` | Class method; reload a previous run from `A2B_full.traj` + `neb_parameters.txt`. Any non-`None` keyword overrides the corresponding saved parameter |

`load_from_restart` is the entry point used by `neb run --restart`. It returns `(neb_instance, loaded_params)`.

---

## Parameter file I/O

Every CLI command writes a `*_params.txt` echo of its arguments using a shared helper. The same helper is available for downstream tooling:

```python
from pathlib import Path
from mliprun.core.params_io import write_parameters_file

write_parameters_file(
    Path("./run/params.txt"),
    title="My Custom Run",
    params={
        "MLIP model:": "uma-s-1p2",
        "Temperature (K):": 300,
        "Steps:": 10000,
    },
)
```

Two-column layout (`{key:<23}{value}`); keys should already include their trailing colon if you want one. `write_endpoint_results(path, results)` is the matching writer for the dict returned by `CustomNEB.optimize_endpoints()`.

---

## Small helpers

```python
from mliprun.core.utils import calc_fmax, GPA_TO_EV_PER_ANG3

fmax = calc_fmax(atoms.get_forces())          # max-magnitude force, eV/Å
pressure_eV_per_A3 = 1.0 * GPA_TO_EV_PER_ANG3 # 1 GPa in ASE internal units
```

`calc_fmax` is the same function used internally by `run_optimization` and `CustomNEB`, so any custom convergence check stays consistent with the CLI's reporting.
