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
| `mliprun.core.committee.config` | `load_committee(path)` | Parse and validate a `committee.yaml` into a `CommitteeConfig` |
| `mliprun.core.committee.remote` | `RemoteMember(name, python_exe, ...)` | One committee member's worker subprocess, addressed as a calculator |
| `mliprun.core.committee.calculator` | `CommitteeCalculator(members, ...)` | ASE calculator over N members: mean force drives the relaxation, their spread is reported |
| `mliprun.core.committee.calculator` | `free_component_mask(atoms)` | The `(N, 3)` free/fixed mask `committee_statistics` needs, from `atoms.constraints` |
| `mliprun.core.committee.calculator` | `committee_statistics(energies, forces, free_mask=None)` | Reduce one evaluation's per-member energies/forces to a consensus and a masked + unmasked spread |
| `mliprun.core.committee.calculator` | `write_peratom_sigma(path, symbols, sigma_all, ...)` | Write `<stem>_committee_peratom.csv` for the final geometry |
| `mliprun.core.committee.calculator` | `uncertainty_summary(rows, latest, *, threshold=None, ...)` | Reduce a run's trace into the `results.committee_uncertainty` block |
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
- `mlip="mace-mh-1"` (or any `mace-mh-*`) → multi-head MACE foundation model; `mace_head` selects the head and is **required, with no default** (`omat_pbe`, `oc20_usemppbe`, `matpes_r2scan`, `mp_pbe_refit_add`, `omol`, `spice_wB97M`). Rejected for plain `mace`.
- `mlip="7net-omni"` (or any `7net-*`) → SevenNet; `sevennet_task` selects the task, which SevenNet's API calls the `modal`. Required for multi-task checkpoints (`7net-omni`, `7net-omni-i8`, `7net-omni-i12`, `7net-mf-ompa`, `7net-mf-0`) and rejected for single-task ones (`7net-omat`, `7net-l3i5`, `7net-0`). No default: the tasks have independent energy zeros.
- `mlip="uma-..."` → any FAIRChem UMA tag, with `uma_task` selecting the head and **required, with no default** (`omat`, `omc`, `omol`, `oc20`, `oc22`, `oc25`, `odac`)
- `mlip="chgnet"` → CHGNet default model

`device` selects the compute device (`"auto"` → cuda if available else cpu, or force `"cuda"` / `"cpu"`). Each head argument is ignored for models outside its own family. None of the three has a default: a multi-head model with no head raises rather than picking one.

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

detect_mlip()              # returns first installed, in order: "uma-s-1p2" / "mace" / "7net-omni" / "chgnet"
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
    uncertainty_plot=False,  # committee runs only: *_uncertainty.png, energy + fmax with bands
)
```

Side effects (written to `output_dir`):

- `opt.traj` — full ASE trajectory
- `opt.log` — per-step force/energy log
- `opt_convergence.csv` / `opt_convergence.png` — step vs energy & fmax
- `opt_final.vasp` — final relaxed structure

`OPTIMIZER_MAP` is the canonical dict of supported optimizer names; access it if you need to validate or list options programmatically.

---

## Committee evaluation

A committee runs several MLIPs, each in its own environment, against the
same structure: the relaxation follows their mean force, and their
disagreement is reported as a per-configuration uncertainty (see
[OUTPUTS.md](OUTPUTS.md#committee-outputs) for what the numbers mean and the
CLI's `optimize run --committee` for the equivalent one-liner). Supported by
`run_optimization` only, not `run_md` or `CustomNEB`.

```python
from ase.io import read

from mliprun.core.committee.calculator import CommitteeCalculator
from mliprun.core.committee.config import load_committee
from mliprun.core.committee.remote import RemoteMember
from mliprun.core.optimize import run_optimization

config = load_committee("committee.yaml")
members = [
    RemoteMember(spec.name, spec.python_exe, mlip=spec.mlip,
                 uma_task=spec.uma_task, mace_head=spec.mace_head,
                 sevennet_task=spec.sevennet_task,
                 device=spec.device, gpu=spec.gpu,
                 log_path=f"committee_{spec.name}.log")
    for spec in config.members
]
atoms = read("POSCAR")
with CommitteeCalculator(members, mixed_theory=config.mixed_theory,
                          levels=config.levels) as committee:
    committee.start()
    committee.preflight(atoms)   # evaluate once, up front: members disagree
                                  # about what input is valid, and that must
                                  # surface now, not on step 400 tonight
    atoms.calc = committee
    run_optimization(atoms, fmax=0.05, output_dir=".",
                      model_name="committee", committee=committee,
                      committee_config=config,
                      uncertainty_plot=True)   # optional: *_uncertainty.png
```

Pass every field of the `spec` through, `device=spec.device` included.
`RemoteMember` defaults `device` to `"auto"`, so dropping it silently discards
a per-member `device:` from `committee.yaml` while the run record still reports
the declared value: a record that says `cpu` for a member that ran on the
default device.

The `with` block is the teardown contract, not a convenience: each member is
a subprocess holding a loaded model (and, on GPU, a CUDA context). A
committee that is never closed leaks every one of them. `close()` is
idempotent and safe to call again in a `finally`, which is what the CLI
does; call it yourself if you build a `CommitteeCalculator` without the
context manager.

**Signals are yours to handle.** The library installs no signal handlers, so
that a caller who installs their own keeps them. That matters for `SIGTERM`
in particular: its default disposition terminates the interpreter *without*
unwinding the stack, so a plain `kill` on your driver runs neither the `with`
block above nor the `atexit` backstop, and leaves every member running and
holding its CUDA context. `mlip optimize run --committee` handles this itself
by routing `SIGTERM` into a `KeyboardInterrupt` for the committee's whole
lifetime; a Python API caller who wants the same protection must do the
equivalent. `SIGKILL` cannot be handled at all, so the worker covers that end
itself: it polls its parent pid every 2 s and exits `3` once the driver is
gone (`mliprun.core.committee.worker.ORPHAN_POLL_S`). That poll exists
because the worker's other defence — seeing EOF when the driver's pipe closes
— is only reachable *between* calculations, and a member spends nearly all of
a relaxation inside one.

`committee.start()` loads every member's model, sequentially, and raises
`mliprun.core.committee.remote.MemberError` naming the failing member if any
one of them cannot load. It closes every member first, so a failed startup
never leaves an orphaned worker. `committee.preflight(atoms)` runs one
evaluation before the optimizer starts, because a geometry one member
rejects (fairchem's UMA calculator raises `MixedPBCError` on a
`pbc=(True, True, False)` slab that MACE, SevenNet and CHGNet all accept)
should fail in the first seconds, not deep into an overnight run.

`load_committee` raises `mliprun.core.committee.config.CommitteeConfigError`
for anything wrong with the file itself (missing env, unknown key, fewer than
two members) **and** for any head/task combination the CLI's `validate_mlip`
rejects: a head on single-head `mace`, a task on a single-task `7net-*` tag, a
missing or unknown head on `mace-mh-*`, a missing or unknown task on a
verified `uma-*` tag. Those combinations used to be accepted and then ignored
by the calculator, which named the member, its CSV column and its provenance
block after a head or task that never ran. Unregistered `uma-*` and `7net-*`
tags still forward their task unchecked, exactly as the CLI does, so a newer
checkpoint works without a code change; the level then resolves to `unknown`,
which flags the committee as mixed. What `load_committee` does *not* check is
whether the member's MLIP package is installed. Only that member's own env can
answer that, and it does, at member start.

`committee.start()` returns `{member name: versions}` and keeps the same dict
on `committee.member_versions`: the interpreter, ASE, torch and MLIP package
each member *measured* inside its own env. `run_optimization` merges it into
`provenance.committee.members[i].measured`, alongside the declared values from
the YAML. If you write your own record, read it from there rather than
assuming the declared version is the one that loaded.

`config.mixed_theory` is `True` when the members span more than one level of
theory, or when any member's tag/task is not in the level-of-theory table at
all (`unknown` counts as possibly mixed). It does not stop the run: mixing
levels is allowed, and the flag rides into the run record and every CSV row
regardless of whether anyone printed it. The CLI prints
`config.mixed_theory_warning()` once at startup when it is set; a script
calling `run_optimization` directly should check and print it too, or the
warning is silent. See [OUTPUTS.md](OUTPUTS.md#mixed-levels-of-theory).

### Uncertainty statistics

`run_optimization` calls these itself when a `committee=` is passed; they
are documented here for a caller writing its own driver instead of going
through `run_optimization`, or post-processing a committee's raw per-member
forces some other way. All four live in `mliprun.core.committee.calculator`.
See [OUTPUTS.md#constraint-masking](OUTPUTS.md#constraint-masking) and
[OUTPUTS.md#the-flagging-rule](OUTPUTS.md#the-flagging-rule) for the
reasoning; this section is signatures and return values only.

```python
free_component_mask(atoms) -> (mask, unhandled_constraints)
```

Builds the `(N, 3)` boolean mask (`True` where a force component is free)
that `committee_statistics` needs for its `free_mask` argument. Only
`ase.constraints.FixAtoms` and `FixCartesian` are recognised — the only
stock ASE constraints whose `adjust_forces` is a pure component mask, so
excluding the component they hold is unambiguous. Every other constraint
type on `atoms` (`FixScaled`, `FixedPlane`, `FixedLine`, `FixBondLength`,
…) is left free rather than masked (over-reporting sigma, never
under-reporting it) and its type name is added to `unhandled_constraints`,
returned as a sorted `list[str]` (empty when nothing is unhandled).

```python
committee_statistics(energies, forces, free_mask=None) -> dict
```

Reduces one evaluation's per-member `energies` (shape `(M,)`) and `forces`
(shape `(M, N, 3)`) to a consensus and a spread. `free_mask` (shape
`(N, 3)`, from `free_component_mask`) is optional; omitting it reproduces
the pre-2026-09-08 unmasked behaviour exactly. Raises `ValueError` for
fewer than two members, mismatched shapes, or a wrongly-shaped `free_mask`.

Returns a dict with `energy_mean`, `forces_mean` (`(N, 3)`),
`sigma_per_atom_all` (`(N,)`, unmasked), `sigma_per_atom_free` (`(N,)`,
masked), `sigma_max_free` / `sigma_mean_free` / `worst_atom_free` (free
components only — what a threshold should compare against),
`sigma_max_all` / `sigma_mean_all` / `worst_atom_all` (the pre-masking
values), `n_free_atoms` (atoms with at least one free component), and
`all_constrained` (`True` when every atom is fully fixed, in which case
`sigma_max_free`/`sigma_mean_free` fall back to the unmasked values because
there is no free population to reduce over).

Every key naming a sigma carries a `_free` or `_all` suffix. There is no
bare `sigma_max`: which population a number covers is the one thing a
reader must not have to remember.

```python
write_peratom_sigma(path, symbols, sigma_all, sigma_free=None,
                     free_mask=None) -> None
```

Writes `<stem>_committee_peratom.csv` (columns: `atom_index`, `symbol`,
`sigma_all_eV_per_A`, `sigma_free_eV_per_A`, `free_components`) for the final
geometry. `sigma_all` is `committee_statistics()`'s `sigma_per_atom_all`;
pass its `sigma_per_atom_free` and the same `free_mask` as `sigma_free` /
`free_mask` to get the masked column and the free-component count.
Omitting `sigma_free` or `free_mask` writes the pre-masking file (every
atom reported fully free). Every atom gets a row, constrained ones
included — see the `<name>_committee_peratom.csv` section of
[OUTPUTS.md](OUTPUTS.md#committee-outputs).

```python
uncertainty_summary(rows, latest, *, threshold=None,
                     threshold_source="none", symbols=None) -> dict
```

Reduces a whole run's trace `rows` (as written by `CommitteeTraceWriter`)
plus the final evaluation's `latest` stats (`None` when the run died before
evaluating anything) into the `results.committee_uncertainty` block. With
no `threshold` — the default — nothing is asserted: the numbers are
reported and `flagged` is `None`. `threshold_source` is `"none"` or
`"explicit"`; there is no `"fmax"` any more. `run_optimization`'s own
`uncertainty_threshold=None` means exactly this: no default, not "use
`fmax`". See [OUTPUTS.md#the-flagging-rule](OUTPUTS.md#the-flagging-rule)
for what each returned key means and the `flagged` tri-state.

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
| `device_requested` | `"auto"` | The device as asked for. Ignored on a committee run: see below. |
| `device_resolved` | `"auto"` | The device actually used (e.g. `"cuda"`). Ignored on a committee run: see below. |
| `run_context` | `None` | A `RunContext` declaring the command, batch identity, and where each parameter value came from. Without it every parameter is tagged `unspecified` — mliprun never guesses. |

Pass the same head/task you gave `setup_calculator` / `build_calculator`.
These functions receive an `Atoms` object with a calculator already attached
and cannot interrogate it for the head, so an omitted `uma_task` is recorded
as "not determined" rather than guessed — CANON C1: the head is an explicit
decision, never inferred. The mismatched one is dropped rather than trusted,
so passing both is harmless.

On a committee run (`committee=` passed to `run_optimization`), both device
keywords are overridden and recorded as the literal string `"committee"`,
whatever you pass. The driver process resolves no device at all: it imports no
torch by design (ADR 0001), so any value it computed would read `"cpu"` even
with every member on its own GPU. The per-member `device` and `gpu` in
`provenance.committee.members[i]` are the authoritative record.

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

`CustomNEB` itself wires the FAIRChem / MACE / SevenNet / CHGNet calculator onto each image, so you do not need to call `setup_calculator` separately. For a multi-task SevenNet checkpoint it raises `ValueError` when `sevennet_task` is unset, rather than letting the failure surface from inside SevenNet.

### Constructor

| Parameter | Default | Notes |
|-----------|---------|-------|
| `initial`, `final` | required | ASE `Atoms`. `final` is re-celled to match `initial`. |
| `num_images` | `9` | Intermediate images only |
| `interp_fmax` | `0.1` | IDPP interpolation force threshold |
| `interp_steps` | `1000` | IDPP iteration limit |
| `fmax` | `0.05` | NEB convergence threshold |
| `mlip` | `"7net-mf-ompa"` | A historical default, and a multi-task SevenNet tag: leaving it means also passing `sevennet_task`, or the constructor's calculator setup raises. Pass `"uma-s-1p2"` etc. as needed. |
| `uma_task` | `None` | Used only when `mlip` starts with `"uma-"`. Required for those models — no default, because the heads have independent energy zeros. |
| `sevennet_task` | `None` | Used only when `mlip` starts with `"7net"`. Required for multi-task checkpoints — no default, because the tasks have independent energy zeros. |
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
