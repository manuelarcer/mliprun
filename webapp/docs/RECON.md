# mliprun reconnaissance for the web app

Branch `feature/webapp` at `c0c8250`, read 2026-10-15. Paths are relative to the repo root.
`src/` is abbreviated to `mliprun/` (for example `mliprun/core/md.py:366` means
`src/mliprun/core/md.py`, line 366). Nothing in the package was modified.

---

## 0. Summary of findings

| Topic | Finding |
|---|---|
| Architecture | Two layers. A **core** layer of pure-Python functions takes an `ase.Atoms` object that already has a calculator attached (`core/optimize.py`, `core/singlepoint.py`, `core/vibrations.py`, `core/md.py`, `core/neb.py`). A **CLI** layer (typer) reads files, validates the MLIP choice, attaches the calculator, writes `*_params.txt`, and prints to the terminal. The web app should call the core layer. |
| Calculator injection | The core functions accept **any** ASE calculator, so `atoms.calc = EMT()` works for optimize, singlepoint, freq and md. The goldens are produced exactly this way (`tests/golden_inputs.py`). NEB and AutoNEB are the exception: `CustomNEB` builds its own calculators through the method `CustomNEB.setup_calculator` (`core/neb.py:344`), which you override on the instance or in a subclass. |
| Live per-step hooks | **None are exposed.** Every engine attaches a private closure to its ASE optimizer or dynamics object, and that closure is not reachable from outside. A backward-compatible `observer=None` keyword is needed in 4 places (see §5). For NEB and AutoNEB there is already a zero-change route: pass an optimizer *subclass* as the `optimizer=` argument. |
| Results | `run_optimization` returns only a `bool`, `run_md` returns `None` and `run_autoneb` returns `None`. The authoritative numbers are in `mliprun_run.json`, which is written with `status: running` when the run begins and completed at the end. `run_singlepoint` and `run_frequencies` return the results dict directly. |
| Config files | Calculations are configured through CLI flags or core keyword arguments only. The single config file is `committee.yaml`, for committees. There are no dataclass configs for single-model runs. |
| Concurrency hazards | `run_autoneb` calls `os.chdir` (process-global). The plots use the global state of matplotlib `pyplot`. `MDLogger` writes to `sys.stdout`. The library code raises `typer.Exit`. The committee SIGTERM handler is installed only on the main thread. **Run every job in its own subprocess, with its own output directory.** |
| Python envs | **No interpreter on this machine has `ase` or `mliprun`**, and none has `mace-torch`, `torch` or `fastapi`. The system Python is 3.14.7 only (`/usr/bin/python3.14`, with no conda, uv or pyenv), so the test suite was **not run**. A new venv is needed (§8). |

---

## 1. Entry points and configuration

### 1.1 Console scripts (`pyproject.toml` `[project.scripts]`)

| Script | Target | Subcommands |
|---|---|---|
| `mlip` | `mliprun.cli.main:app` (`cli/main.py:15-27`) | `optimize {run,batch}`, `singlepoint run`, `freq run`, `md run`, `neb run`, `autoneb run`, `autoneb-results results`, `benchmark run`, `doctor` |
| `optimize` | `cli/commands/optimize.py:33` | `run` (:59), `batch` (:273) |
| `singlepoint` | `cli/commands/singlepoint.py:30` | `run` (:47). The callback at :33 keeps `run` addressable. |
| `freq` | `cli/commands/freq.py` | `run` (:55). Callback at :42. |
| `md` | `cli/commands/md.py:27` | `run` (:64) |
| `neb` | `cli/commands/neb.py:18` | `run` (:350) |
| `autoneb` | `cli/commands/autoneb.py:13` | `run` (:16) |
| `autoneb-results` | `cli/commands/autoneb_results.py` | `results` (:15) |
| `benchmark` | `cli/commands/benchmark.py` | `run` (:44) |
| (none) | `cli/commands/doctor.py:77` | `mlip doctor`. Exits 1 if no MLIP is installed. |

`--structure`, `--initial`, `--final` and `--parent` carry `prompt=True`. If the web app omits one, typer blocks waiting on stdin.

### 1.2 Python API (the layer the web app should call)

| Function or class | Location | Returns |
|---|---|---|
| `build_calculator(mlip, uma_task=None, device="auto", mace_head=None, sevennet_task=None)` | `cli/utils.py:607` | ASE calculator, or `None` for an unknown tag (`:703`) |
| `setup_calculator(atoms, ...)` | `cli/utils.py:706` | the `atoms` object, with `.calc` set |
| `detect_mlip()`, `validate_mlip(...)`, `resolve_mlip(...)` | `cli/utils.py:258`, `:420`, `:478` | tag string. These raise `typer.Exit` and print with `typer.echo`. |
| `run_optimization(atoms, ...)` | `core/optimize.py:320` | `bool` (converged) |
| `OPTIMIZER_MAP` | `core/optimize.py:22` | module-level dict; it can be monkeypatched |
| `run_singlepoint(atoms, ...)` | `core/singlepoint.py:78` | results dict |
| `run_frequencies(atoms, ...)` | `core/vibrations.py:902` | results dict |
| `setup_dynamics(atoms, ...)` / `run_md(atoms, ...)` | `core/md.py:139` / `:366` | ASE dynamics object / `None` |
| `CustomNEB(initial, final, ...)` | `core/neb.py:59` (`__init__` :98) | object. Methods: `load_from_restart` :229 (classmethod), `setup_calculator` :344, `optimize_endpoints` :437, `setup_neb` :576, `interpolate_idpp` :614, `run_neb` :665 (returns a list of images), `run_autoneb` :848 (returns `None`), `process_results` :1010, `plot_results` :1028, `export_poscars` :1066 |
| `summarize_neb_path(energies)` | `core/neb.py:30` | barrier dict |
| `load_committee(path)` | `core/committee/config.py:440` | `CommitteeConfig` |
| `RemoteMember(...)`, `CommitteeCalculator(members, ...)` | `core/committee/remote.py:78`, `core/committee/calculator.py:202` | |
| `build_committee`, `started_committee` (context manager) | `cli/committee_session.py:46`, `:128` | started `CommitteeCalculator` |
| `RunContext`, `BatchInfo`, `RunRecord` | `core/run_record.py:284`, `:264`, `:349` | |
| `write_parameters_file`, `write_endpoint_results` | `core/params_io.py:5`, `:26` | |

**Documentation drift.** `docs/PYTHON_API.md:9-10` documents the defaults as `uma_task="omat"` and `mace_head="omat_pbe"`. The code defaults to `None`, and a multi-head model with no head **raises** `ValueError` (`cli/utils.py:658-697`). The docstrings at `cli/utils.py:622-631` are stale in the same way.

### 1.3 Configuration channels

- **CLI flags**: typer options (tables in §2). Where each value came from is captured by `param_sources_from_ctx` (`cli/utils.py:767`) and passed to the core layer through `RunContext.param_sources`. Without that context, every parameter is recorded as `"unspecified"` (`core/run_record.py:303-313`). The web app can pass its own `RunContext(command=..., param_sources={name: "user"|"default"})`.
- **`committee.yaml`**: parsed by `load_committee`, `core/committee/config.py:440`. Example at `examples/committee.yaml`. It is the only config file.
- **No other YAML or dataclass config.** The `neb_parameters.txt` file written by `neb run` is read back on `--restart` (`core/neb.py:145-222`).

---

## 2. Calculation types and their parameters (source for the web form)

### 2.0 Shared model-selection parameters (all single-model commands)

| Name | Type | Default | Valid values | Notes |
|---|---|---|---|---|
| `mlip` | str | `auto` (the `neb` CLI uses `None`, then `auto`, `neb.py:242`) | `auto`; any `uma-*` (verified: `uma-s-1p2`, `uma-s-1p1`, `uma-m-1p1`, `utils.py:193`); `mace`; `mace-mh-0`, `mace-mh-1`; `7net-omni`, `7net-omni-i8`, `7net-omni-i12`, `7net-mf-ompa`, `7net-mf-0`, `7net-omat`, `7net-l3i5`, `7net-0`, `7net-0_22may2024` (`utils.py:170-180`); `chgnet` | `auto` resolves in the order UMA → MACE → SevenNet → CHGNet (`utils.py:284-293`) |
| `uma_task` | str | None (**required** for `uma-*`) | `omat`, `omc`, `omol`, `oc20`, `oc22`, `oc25`, `odac` (`utils.py:188`) | The value is unchecked for unverified `uma-*` tags (`:375`) |
| `mace_head` | str | None (**required** for `mace-mh-*`, **rejected** for `mace`) | `omat_pbe`, `mp_pbe_refit_add`, `matpes_r2scan`, `oc20_usemppbe`, `omol`, `spice_wB97M` (`utils.py:197`) | |
| `sevennet_task` | str | None (**required** for multi-task, **rejected** for single-task) | omni / -i8 / -i12: `omat24`, `mpa`, `omol25_low`, `omol25_high`, `matpes_pbe`, `matpes_r2scan`, `mp_r2scan`, `oc20`, `oc22`, `spice`, `qcml`, `odac23`, `pet_mad`; `7net-mf-ompa`: `omat24`, `mpa`; `7net-mf-0`: `PBE`, `R2SCAN` (uppercase, case-sensitive) | `utils.py:164-180` |
| `device` | str | `auto` (`neb`: `cpu`; `autoneb`: none, so `CustomNEB` uses `cpu`; `benchmark`: none, so `auto`) | `auto`, `cuda`, `cpu` | **Not validated.** Any other string is passed unchanged to torch (`core/utils.py:8-23`) |

The dependent-field rule for the form: show the task or head field according to the tag prefix. Each one is required, has no default, and is rejected where it does not apply.

### 2.1 `optimize run` (`cli/commands/optimize.py:59-106`) → `run_optimization` (`core/optimize.py:320`)

| Name | Type | Default | Range / choices | Description |
|---|---|---|---|---|
| structure | path | required | any ASE-readable file | input structure. The output directory is its parent. |
| (model block) | | | §2.0 | |
| committee | path | None | committee.yaml | mutually exclusive with `--mlip`, `--uma-task`, `--mace-head`, `--sevennet-task`, `--device` (`committee_session.py:23`) |
| member_timeout | float s | 300.0 (`remote.py:27`) | >0 | per-call limit for each member (committee only). The load limit is 1800 s (`remote.py:34`). |
| uncertainty_threshold | float eV/Å | None | >0 | flags σ_max_free. With None, no verdict is given. |
| optimizer | str | `bfgs` | `fire`, `bfgs`, `lbfgs`, `bfgsls`, `gpmin`, `mdmin` (`core/optimize.py:22-29`) | |
| fmax | float eV/Å | 0.05 | >0 | convergence criterion (includes the cell virials when relax_cell is set) |
| max_steps | int | 200 | ≥0 | |
| relax_cell | bool | False | | `FrechetCellFilter` (falls back to `ExpCellFilter`), `core/optimize.py:32` |
| trajectory | str | `opt.traj` | filename | |
| logfile | str | `opt.log` | filename | its stem names `*_convergence.csv`, `*_final.vasp` and the committee CSVs |
| verbose | bool | True (batch: False) | | selects path or handle for the ASE log file; both write `opt.log` |
| plot | bool | False | | `<stem>_convergence.png` |
| uncertainty_plot | bool | False | committee only (the CLI rejects it otherwise, :150) | `<stem>_uncertainty.png` |
| core-only: `output_dir`, `model_name` (label), `run_context`, `device_requested`, `device_resolved`, `committee`, `committee_config` | | `"."`, `"mlip"` | | |

### 2.2 `optimize batch` (`optimize.py:273-300`, a CLI-only loop over `run_optimization`)

`parent` (dir, required), `input_name` (glob, `*.vasp`; files ending `_final.vasp` are excluded, :36-56), the model block, `optimizer`, `fmax`, `max_steps`, `relax_cell`, `trajectory`, `logfile`, `skip_existing` (bool False; skips a subdir that has a `CONTCAR`), `verbose` (False), `plot`. No committee. The calculator is built once (:353). Writes `<parent>/batch_summary.csv` (:439). The `steps` column is always blank (:426).

### 2.3 `singlepoint run` (`singlepoint.py:47-87`) → `run_singlepoint` (`core/singlepoint.py:78`)

| Name | Type | Default | Choices | Description |
|---|---|---|---|---|
| structure | path | required | | |
| (model block, committee, member_timeout, uncertainty_threshold) | | as in §2.1 | | |
| stress | bool or None | None | None means "only if pbc is all True" (`core/singlepoint.py:40`); True forces; False skips | flag form `--stress/--no-stress` |
| prefix | str | `singlepoint` | | file stem |
| output_dir | path | structure's parent | | |

### 2.4 `freq run` (`freq.py:55-119`, validation :128-151) → `run_frequencies` (`core/vibrations.py:902`)

| Name | Type | Default | Choices / range | Description |
|---|---|---|---|---|
| structure | path | required | | |
| (model block, committee, member_timeout, uncertainty_threshold) | | as in §2.1 | | |
| indices | str | None (= every atom not under `FixAtoms`) | `"0,1,5"`, `"12-30"` (inclusive), or a mix; 0..N-1 (`vibrations.py:180`) | atoms to displace |
| delta | float Å | 0.01 | >0 | displacement |
| nfree | int | 2 | 2, 4 (`vibrations.py:29`) | 4 costs twice as much |
| direction | str | `central` | `central`, `forward`, `backward` | |
| method | str | `standard` | `standard`, `frederiksen` | |
| write_modes | str | `imaginary` | `none`, `imaginary`, `all` | writes `<prefix>.<n>.traj` per mode |
| imaginary_floor | float cm⁻¹ | 10.0 (`vibrations.py:49`) | ≥0 | |
| expect_fmax | float eV/Å | None, then the fmax of a converged optimize stage in `structure_dir/mliprun_run.json` | >0 | gives a warning only |
| output_dir | path | structure's parent | | |
| prefix | str | `freq` | | also names the displacement cache dir `<out>/<prefix>/` |
| core-only: `structure_dir` (defaults to output_dir) | | | | |

Cost is `1 + 6·n` force calls with nfree=2, or `1 + 12·n` with nfree=4 (`vibrations.py:604-613`). A reused cache with a different delta, nfree or model raises `FrequencyCacheError` (`vibrations.py:82`, `:487`).

### 2.5 `md run` (`md.py:64-154`) → `run_md` (`core/md.py:366`) / `setup_dynamics` (`:139`)

| Name | Type | Default | Choices / range | Description |
|---|---|---|---|---|
| structure | path | required | | the output directory is its parent |
| ensemble | str | `nvt` | `nve`, `nvt`, `npt` | |
| steps | int | 1000 | ≥0 | with resume, these are additional steps |
| temperature | float K | 300 | >0 for nvt/npt (CLI :177) | |
| pressure | float GPa | 0.0 | | NPT |
| timestep | float fs | 1.0 | >0 | |
| thermostat | str | `langevin` | `langevin`, `nose-hoover` (needs `ase.md.nose_hoover`, `core/md.py:26-30`), `berendsen` | NVT only |
| barostat | str | `npt` | `npt` (ASE NPT, Melchionna), `berendsen` | NPT only |
| barostat_mask | str `"x,y,z"` | `1,1,1` | each 0/1; a non-default value only with npt (`core/md.py:89`, CLI :186) | the berendsen barostat with a mask switches to `Inhomogeneous_NPTBerendsen` |
| friction | float 1/fs | 0.01 | | Langevin |
| ttime | float fs | 25.0 | | Nosé-Hoover / NPT |
| taut | float fs | 100.0 | | Berendsen T |
| taup | float fs | 1000.0 | | Berendsen P |
| compressibility | float 1/GPa | 0.457 (`core/md.py:63`) | >0; **the CLI rejects an explicit value unless the barostat is berendsen** (:200) | |
| pfactor | float GPa·fs² (CLI); ASE units (core) | None, then 5.625e5 GPa·fs² (`core/md.py:77`) | >0; npt barostat only (:205) | the CLI multiplies by `PFACTOR_GPA_FS2` (:221) |
| (model block) | | | §2.0 | no committee |
| resume | bool | False | needs `md.traj` and `md_energy.csv` in the output dir | starts from the last frame of `md.traj` (CLI :162-167) |
| csv_flush_every | int | 100 | ≥0 (0 = only at the end) | **set this to 1 for live CSV** |
| log_interval | int | 10 | ≥1 | one CSV row and one stdout `MDLogger` line every N steps |
| traj_interval | int | 100 | ≥1 | a frame every N steps |
| plot | bool | False | | `md_energy.png`, `md_temperature.png`, and for NPT `md_pressure.png`, `md_volume.png` |
| core-only: `log_path` (None; a second `MDLogger` file), `output_dir`, `model_name` | | | | The CLI never passes `log_path`. |

### 2.6 `neb run` (`neb.py:350-377`; defaults applied at :237-254 with `or`, so a value of **0 cannot be entered** for k, fmax, interp_fmax and similar)

| Name | Type | Default | Choices | Description |
|---|---|---|---|---|
| restart | bool | False | | forbids initial, final, num_images, relax_atoms and optimize_endpoints (:384-393). Needs `A2B_full.traj` and `neb_parameters.txt` in the **cwd**. Moves the old outputs to `bkup_<ts>/`. On a frame-count mismatch it **sleeps 15 s** (:146-155). |
| initial / final | path | required unless restart | read with `format="vasp"` (forced, :256-257) | the same atom count is required |
| num_images | int | 5 | ≥1 | intermediate images. Total = n + 2. (The class default is 9.) |
| interp_fmax | float | 0.1 | | IDPP |
| interp_steps | int | 100 | | IDPP (the class default is 1000) |
| fmax | float eV/Å | 0.05 | | NEB convergence |
| (model block) | | `auto` | §2.0 | `--device` default `cpu`; help lists `cpu`, `cuda` |
| relax_atoms | str `"0,1,5"` | None | indices 0..N-1 (`cli/utils.py:515`) | fixes all other atoms and **skips IDPP** |
| log | str | `neb.log` | | |
| k | float eV/Å² | 0.1 | | spring constant |
| climb | bool | True | `--climb/--no-climb` | CI-NEB (the core `run_neb` default is False) |
| dyneb | bool | False | | `ase.mep.dyneb.DyNEB` |
| scale_fmax | float | 0.0 | ≥0 | DyNEB only |
| neb_optimizer | str | `fire` | `fire`, `mdmin`, `bfgs`, `lbfgs` (:20; an unknown name silently falls back to FIRE, :420) | |
| neb_max_steps | int | 600 | | |
| optimize_endpoints | bool | True | | |
| endpoint_fmax | float | 0.01 | | |
| endpoint_optimizer | str | `bfgs` | `bfgs`, `lbfgs`, `fire` (unknown → BFGS, `core/neb.py:463`) | |
| endpoint_max_steps | int | 200 | | |
| plot | bool | False | | |

Output directory: **`Path.cwd()`** (:380). The core class takes an `output_dir` argument.

### 2.7 `autoneb run` (`autoneb.py:16-38`) → `CustomNEB.run_autoneb` (`core/neb.py:848`)

| Name | Type | Default | Notes |
|---|---|---|---|
| initial / final | path | required (prompt) | `format="vasp"` forced (:46-47) |
| n_max | int | 9 | total images, endpoints included |
| n_simul | int | 1 | >1 needs MPI |
| fmax | float | 0.05 | |
| (model block, no device) | | `auto` | device is fixed at the `CustomNEB` default `cpu` |
| climb | bool | True | |
| k | float | 0.1 | |
| space_energy_ratio | float | 0.5 | 0 = energy gaps … 1 = geometric gaps |
| interpolate_method | str | `idpp` | `idpp`, `linear` |
| maxsteps | int | 10000 | per relaxation |
| prefix | str | `autoneb` | `autoneb000.traj`, … |
| relax_atoms | str | None | a warning says this may not work |
| optimize_endpoints | bool | True | |
| endpoint_fmax / endpoint_optimizer / endpoint_max_steps | float / str / int | 0.01 / `bfgs` / 200 | |

The optimizer is hard-coded to `FIRE` in the CLI (:143). The core accepts any optimizer class. Output: cwd. The core `chdir`s into `output_dir` (`core/neb.py:917-972`) and **deletes** `{prefix}*.traj` and `AutoNEB_iter/` there before it starts (:920-924).

### 2.8 `autoneb-results results` (`autoneb_results.py:15-19`), post-processing only

`directory` (path, `.`), `prefix` (str, `autoneb`), `export_poscars` (bool, False). Reads the last frame of each `{prefix}*.traj`. Writes `{prefix}_energy_profile.csv` and `.png` (always), and `image_XX/POSCAR` optionally. No core function exists: the logic is inline in the CLI.

### 2.9 `benchmark run` (`benchmark.py:44-53`), CLI only

`structure` (path), `models` (comma-separated tags; default is every installed model, with UMA and SevenNet included only if a task is given, `:23-41`), `uma_task`, `sevennet_task`, `output` (JSON path, optional). It has **no `mace_head` and no `device`**, so `mace-mh-*` always fails here. It times one `get_potential_energy()` per model, and a failed model is recorded as `"failed: ..."` with the loop continuing.

### 2.10 Committee variants (only `optimize run`, `singlepoint run`, `freq run`)

`committee.yaml` schema (`core/committee/config.py:163-166`, `:411-437`): a top-level `members:` list with ≥2 entries. Per member: `env` (required; the env root, which must contain `bin/python`), `mlip` (required; a tag, or the reserved `emt`, `worker.py:27`), `uma_task`, `mace_head`, `sevennet_task` (the same rules as §2.0), `name` (matching `^[A-Za-z0-9][A-Za-z0-9._@+-]*$`; default `<mlip>@<task>`), `device` (default `auto`), and `gpu` (int; sets `CUDA_VISIBLE_DEVICES`). Unknown keys are rejected. Every member env must have mliprun installed. The bridge is POSIX-only. Each member runs as a subprocess: `python -m mliprun.core.committee.worker` (`remote.py:78`).

### 2.11 Other

`mlip doctor` (`doctor.py:77`) takes no parameters and gives a useful environment summary for an "environment" page.

---

## 3. MLIP and calculator selection

| Step | Code |
|---|---|
| Availability is detected **once, when the module is imported**, from `importlib.metadata.distribution()` | `cli/utils.py:75-119`. A package installed later is not seen until restart. |
| `auto` resolution: `uma-s-1p2` → `mace` → `7net-omni` → `chgnet`, else `typer.Exit` | `cli/utils.py:258-293` |
| Validation (availability, then task/head rules) | `cli/utils.py:420-475`; raises `typer.Exit(<message>)` |
| Build: `mace` → `mace_mp(model="medium")`; `mace-mh-*` → `MACECalculator(model_paths=<downloaded to ~/.cache/mace>, head=...)`; `7net*` → `SevenNetCalculator(tag, modal=task)`; `uma-*` → `pretrained_mlip.get_predict_unit` + `FAIRChemCalculator(task_name=)`; `chgnet` → `CHGNetCalculator(use_device=)` | `cli/utils.py:607-703`. The heavy imports are lazy, through `lru_cache` loaders at `:553-574`. |
| UMA gating | a Hugging Face access request is needed (see `docs/UMA_USAGE_GUIDE.md`); the code does not check for it, so a failure surfaces as a fairchem error at load time |
| SevenNet with `auto` | resolves to `7net-omni`, which is multi-task, and then **stops** for a missing task (`utils.py:504`) |
| Device | `resolve_device("auto")` returns `cuda` if `torch.cuda.is_available()`, else `cpu` (`core/utils.py:8-23`). The resolved and requested values are both recorded. This machine has no GPU (`nvidia-smi` is absent) and 8 CPUs. |
| NEB duplicate | `CustomNEB.setup_calculator` (`core/neb.py:344-431`) re-implements the build. It is called once per image or endpoint at `:474`, `:774`, `:929` and `:943`. |

### Injecting a plain ASE calculator (EMT)

| Calc type | Injection point | Evidence |
|---|---|---|
| optimize, singlepoint, freq, md (core) | Set `atoms.calc = EMT()` yourself; `model_name` is only a label. The provenance then shows `mlip_package: {name: None, version: None}`. | `tests/golden_inputs.py:18-37`; `tests/test_characterization_golden.py` |
| neb, autoneb (core) | `neb.setup_calculator = lambda *a, **k: EMT()` on the instance, after construction (`__init__` attaches no calculators), or a subclass that overrides `setup_calculator` | `tests/test_core_neb.py:128`, `tests/test_run_record_integration.py:393` |
| any CLI command | monkeypatch the name bound inside the module: `mliprun.cli.commands.<cmd>.setup_calculator` or `.build_calculator` (batch). Note that `validate_mlip` still runs first. | `tests/test_cli_run_plot.py:34`, `tests/test_optimize_batch.py:157` |
| committee | the reserved member tag `mlip: emt`, with `env` pointing at the env of the current interpreter | `tests/conftest.py:134-150` (`fake_committee_file`) |

The narrowest backward-compatible change, if one is wanted later, is an optional `calculator_factory=None` keyword on `CustomNEB.__init__` that `setup_calculator` uses when it is set. Nothing is needed for the other calc types.

---

## 4. What each run writes, and where

### 4.1 Output directory rules

| Path | Rule |
|---|---|
| CLI optimize, md | `Path(structure).parent` (`optimize.py:164`, `md.py:161`) |
| CLI singlepoint, freq | `--output-dir`, else the structure's parent |
| CLI neb, autoneb | `Path.cwd()` |
| CLI autoneb-results | `--directory` |
| CLI benchmark | nothing, unless `--output` is given |
| Core API | every function takes `output_dir` (`CustomNEB(output_dir=...)`) and creates it |

**One `mliprun_run.json` per directory.** A new command in the same directory *replaces* the record. Only `md --resume` and `neb --restart` append a stage (`core/run_record.py:358-451`). The web app should give each job its own directory.

### 4.2 The run record (`core/run_record.py`)

- `RunRecord.begin()` (:359) atomically writes `{schema_version: 5, command, status: "running", run, inputs, parameters: {k: {value, source}}, provenance, stages: [{kind, status: "running", started_at, ...}]}` **before the calculation starts**.
- `complete()` (:453) sets the stage and top-level `status` (`converged`, `not_converged`, `completed` or `failed`), plus `walltime_s`, `steps` and `results`.
- On an exception, every engine writes `status: failed, results: {error: str}` and re-raises. A record still showing `running` therefore means the process died.
- A record write never raises, NaN is stored as null, and an unreadable record is moved aside as `*.corrupt-<ts>`.
- Per-command `results` keys are listed in `docs/OUTPUTS.md` ("Results by command", line 1027).

### 4.3 Files, and when they are written

| Calc | Written **during** the run (live) | Written **at the end** |
|---|---|---|
| optimize | `opt.traj` (ASE `trajectory=`, every step) and `opt.log` (ASE optimizer log, one line per step), both set up at `core/optimize.py:562/569`; with a committee, `<stem>_committee.csv` (flushed every step, `calculator.py:498-522`) and `committee_<member>.log` | `<stem>_convergence.csv` (`step, energy(eV), fmax(eV/A)`, :654, **only at the end**), `<stem>_final.vasp` and `CONTCAR` (:651-652), `mliprun_run.json`, the optional PNGs, `<stem>_committee_peratom.csv`; CLI: `opt_params.txt` (`optimize.py:453`) |
| singlepoint | `mliprun_run.json` (running) | `<prefix>_forces.csv` (`index, symbol, fx, fy, fz, f_norm, free_x, free_y, free_z`), `<prefix>_committee_peratom.csv` |
| freq | `<prefix>_cache.json` (an identity sidecar written before the sweep, `vibrations.py:513`); `<out>/<prefix>/cache.*` (ASE writes one JSON per displacement) | `<prefix>_vibrations.json`, `<prefix>_summary.txt`, `<prefix>_frequencies.csv` (`mode_index, frequency_cm-1, energy_meV, imaginary`; signed), `<prefix>_committee_frequencies.csv`, `<prefix>.<n>.traj` mode animations (`vibrations.py:1122-1172`) |
| md | `md.traj` (every `traj_interval`, :539); `md_energy.csv` (header first, then appended every `csv_flush_every` log calls, :550-593; columns `step, time(fs), temperature(K), total_energy(eV), potential_energy(eV), kinetic_energy(eV)` plus, for NPT, `pressure(GPa), pressure_xx/yy/zz(GPa), volume(A^3)`); `MDLogger` on **stdout** (:542); CLI: `md_params.txt` before the run (`md.py:283`) | final CSV flush, `mliprun_run.json` (`_md_statistics`, :314), PNGs |
| neb | `initial_opt.traj/.log` and `final_opt.traj/.log` (endpoints, `core/neb.py:477-481`); `idpp.traj/.log` (:624-629); `neb.log` (one line per iteration, flushed: `Step N Fmax: x Barrier: y`, :792-795); `A2B_full.traj` (**all images, every iteration**, :801-802); CLI: `neb_parameters.txt` (before the run) and `endpoint_optimization.txt` | `neb_convergence.csv` (`step, fmax(eV/A), barrier(eV)`, :816-818), `A2B.traj` (final band, :826-829), `mliprun_run.json` (`summarize_neb_path`); CLI after `run_neb`: `neb_data.csv` (`image_index, energy, relative_energy`), `00/POSCAR` …, PNGs |
| autoneb | ASE-internal: `autoneb000.traj` …, `AutoNEB_iter/` (per-iteration history); `autoneb_parameters.txt` | `mliprun_run.json`; no CSV |
| autoneb-results | | `{prefix}_energy_profile.csv/.png`, `image_XX/POSCAR` |
| benchmark | stdout | optional JSON |

---

## 5. Step-level data access and live hooks

**No calc type exposes a per-step callback today.** Each engine uses a private closure, shown in the table. Two routes work with no code change: an exception raised inside an attached callback propagates out of `opt.run` / `dyn.run`, the record is completed as `failed`, and the exception is re-raised. That gives a clean **cancellation** mechanism once an observer exists.

| Calc | Existing internal hook | Zero-change workaround | Minimal backward-compatible hook to add |
|---|---|---|---|
| optimize | `log_convergence()` closure, `opt.attach(..., interval=1)` (`core/optimize.py:545-563`) | (a) tail `opt.log` and `opt.traj`; (b) register a subclass in the module-level `OPTIMIZER_MAP` (`:22`), for example `OPTIMIZER_MAP["bfgs_obs"] = cls`, whose `__init__` calls `self.attach(cb)`. This is hacky because it mutates global state. | `observer: Optional[Callable[[dict], None]] = None` on `run_optimization` (:320), called at the end of `log_convergence` with `{"step", "energy", "fmax", "committee_row"?}`. Optionally also `"positions"`/`"cell"` from `atoms`. |
| singlepoint | none (one evaluation) | the record status | none needed; report started and finished |
| freq | `CountingVibrations.calculate()` (`core/vibrations.py:399-402`), called once per displacement actually computed, with the counter `n_force_calls` | poll the count of `<out>/<prefix>/cache.*` against `1 + 6n` or `1 + 12n` (`_expected_force_calls`, :604) | `observer=None` on `run_frequencies` (:902), stored on the Vibrations instance created at :1042 and called in `CountingVibrations.calculate` with `{"done": n_force_calls, "total": _expected_force_calls(len(indices), nfree)}` |
| md | `log_properties()` closure, `dyn.attach(..., interval=log_interval)` (`core/md.py:565-595`) | `csv_flush_every=1` makes `md_energy.csv` grow by one row every `log_interval` steps; `md.traj` grows every `traj_interval`. Alternatively use `setup_dynamics` (:139) directly and attach your own callbacks, which loses the run record, CSV and stats. | `observer=None` on `run_md` (:366), called at the end of `log_properties` with `row` (step, time, T, Etot, Epot, Ekin, plus P and V for NPT) |
| neb | `log_iteration()` closure (`core/neb.py:784-802`) | **pass an optimizer subclass** as `run_neb(optimizer=...)`: `class ObsFIRE(FIRE): def __init__(s, *a, **k): super().__init__(*a, **k); s.attach(lambda: cb(s))`. `s.atoms` is the NEB object, so `.images` gives the per-image energies. Or tail `neb.log` and `A2B_full.traj`. | `observer=None` on `CustomNEB.run_neb` (:665), called in `log_iteration` with `{"step", "fmax", "barrier", "energies": [...]}` |
| neb endpoints | none (`opt.run` at `core/neb.py:482`) | tail `initial_opt.log` / `final_opt.log` | `observer=None` on `optimize_endpoints` (:437), attached to `opt` with a `label` |
| neb IDPP | none (`idpp_interpolate`, :626) | tail `idpp.log` | (optional) phase events only |
| autoneb | none (ASE `AutoNEB.run()`, :959) | the same optimizer-subclass trick: `run_autoneb(optimizer=ObsFIRE)`, because ASE builds a fresh optimizer for each sub-NEB; or watch `AutoNEB_iter/` and `{prefix}NNN.traj` appear | a thin wrapper over the `optimizer` argument inside `run_autoneb` (:848), same signature |
| optimize batch, benchmark | CLI loops (`optimize.py:360`, `benchmark.py:87`) | re-implement the loop in the web app (it is trivial) | n/a |
| committee | `CommitteeCalculator.latest` (`calculator.py:404`) holds the stats of the last evaluation | read `committee.latest` from inside an optimizer observer | covered by the optimize observer |

Recommended observer contract (one keyword per engine, default `None`, never changing numerics): `observer(event: dict)`, called synchronously in the engine thread. The observer must be cheap; it should push to a queue or pipe. It must not touch `atoms` beyond read-only getters, because those hit the calculator's cache. The golden tests guarantee that a `None` observer changes nothing.

---

## 6. Structure input

| Path | Reader |
|---|---|
| optimize, singlepoint, freq, md, benchmark (CLI) | `ase.io.read(path)` with **automatic format detection** from the file name or extension (`optimize.py:114`, `singlepoint.py:89`, `freq.py:121`, `md.py:169`, `benchmark.py:61`). That covers `POSCAR`, `CONTCAR`, `*.vasp`, `OUTCAR`, `*.cif`, `*.xyz` (extxyz), `*.traj`, `*.json`/`*.db` and everything else ASE supports. **A multi-frame file yields its last frame** (ASE's default `index=-1`). |
| md --resume | `read(<outdir>/md.traj, index=-1)` (`md.py:166`) |
| neb, autoneb (CLI) | `read(path, format="vasp")`, **forced** (`neb.py:256-257`, `autoneb.py:46-47`) |
| neb --restart | `read(A2B_full.traj, index=f"-{n+2}:")` (`core/neb.py:279`) |
| autoneb-results | `read(traj, index=-1)` for each `{prefix}*.traj` |
| Core API | takes `ase.Atoms`; the web app controls the reading |

For uploads, keep the original file name, because format detection depends on it (`POSCAR` has no extension), or pass `format=` explicitly. Constraints matter: VASP selective dynamics becomes `FixAtoms`/`FixCartesian`. These drive the free mask in singlepoint and the committee σ (`calculator.py:19`), and the default displaced atoms in freq (`vibrations.py:236`). Structures are written as VASP 5, direct coordinates.

---

## 7. Test suite

| Item | Value |
|---|---|
| Runner | pytest. `pytest.ini`: `pythonpath = src`, `testpaths = tests`, `addopts = --maxfail=0 --continue-on-collection-errors` |
| Markers | `uma`, `mace`, `sevenn`, `slow` (registered, but no test uses `slow`). `conftest.py:59-67` skips a test automatically when its package is missing. **Trap:** a parametrize id of exactly `uma`, `mace` or `sevenn` silently skips the case (AGENTS.md "Testing"). |
| Layout | 49 `tests/test_*.py` files, about 770 `def test_` functions before parametrization. Committee tests use subprocess workers with the `emt` tag (`tests/committee_stubs/`). The MLIP smoke tests are `test_md_mace.py`, `test_neb_mace.py`, `test_integration_uma.py` and others. |
| Fixtures | `simple_atoms` (Cu with EMT), `initial_final_pair`, `fake_committee_file` and others (`tests/conftest.py`) |
| Frozen goldens | `tests/goldens/{optimize_cu_rattled,md_nve_cu,neb_idpp_profile,utils_calc_fmax}.json`, checked by `tests/test_characterization_golden.py` with EMT inputs from `tests/golden_inputs.py`. **Never regenerate them**; `scripts/generate_goldens.py` refuses to overwrite. If one fails, report the numerical delta. Every test must assert a number or an invariant. |
| CI | `.github/workflows/tests.yml`: Ubuntu, Python 3.11, `pip install -e ".[dev,neb]"`, with a coverage `.pth` armed for subprocess workers; runs `pytest --cov`, then `diff-cover --fail-under=90` against `origin/main`. `install-smoke.yml`: a fresh venv, MACE on CPU, and a 3-step `optimize run`. `weekly.yml`: pip-audit and scoped mutmut. |
| Commands | `pytest -m "not uma and not mace and not sevenn"` (no MLIP needed; needs only `ase`, `typer`, `pandas`, `matplotlib`, `scipy`, `pyyaml`); `pytest` (all) |
| Duration | Unknown. Nothing in the repo records it, and the suite could not be run here. |
| **Run result** | **Not run.** No interpreter on this machine has `ase` or `mliprun` installed, and the instructions were not to install anything. |

Any web-app code that touches `src/` must keep CI's 90% diff-coverage gate green. Observer hooks therefore need tests (EMT is enough).

---

## 8. Python environment situation (2026-10-15)

| Interpreter | Python | Relevant packages present |
|---|---|---|
| `/usr/bin/python3` (= `/usr/bin/python`, 3.14) | 3.14.7 | `pyyaml` only. No `ase`, `mliprun`, `mace-torch`, `torch`, `fastapi` or `pytest`. PEP 668 managed, so a bare `pip install` fails. |
| `~/github/ibkr-agent/.venv` | 3.14.7 | `uvicorn 0.52.4`, `pytest 9.1.1`, `pandas`, `pyyaml`, `websockets`, `python-multipart`. No `ase` or `fastapi`. (Another project; do not reuse it.) |
| `~/tools/Basecamp-MCP-Server/venv` | 3.14.7 | `uvicorn`, `pytest 7.4`, `typer 0.27.2`, `httpx`, `pydantic 2.13`, `sse-starlette`. No `ase` or `fastapi`. |
| `~/github/board-games-lab/.venv` | 3.14.7 | `pytest 9.1.1` |
| conda, mamba, uv, pyenv, pipx | not installed | |
| `mliprun/.venv` | **does not exist** | |
| Playwright | Node-based only: `~/.local/bin/playwright` is a mise shim (`npm:playwright`); node 26.8.1 comes through mise; `/usr/bin/chromium` is present. No Python `playwright`, and no `~/.cache/ms-playwright`. | |
| MACE weights cache `~/.cache/mace` | absent | |

Recommendation (needs the user's approval, because it installs things): create `/home/juan/github/mliprun/.venv` with `python3.14 -m venv`, then `pip install -e ".[dev,neb]"`, then `torch` (CPU) and `mace-torch`, then `fastapi uvicorn` (plus `pytest-playwright` if browser tests go through Python). It must be a separate env from any other MLIP (ADR 0001). For a test of the web layer alone, `pip install -e ".[dev]"` without MACE is enough, because every engine runs on EMT.

---

## 9. Risks and open issues for the web-app design

1. **Process-global side effects.** `run_autoneb` calls `os.chdir` (`core/neb.py:918`). matplotlib `pyplot` is not thread-safe (every `_plot_*`). `MDLogger` writes to `sys.stdout` (`core/md.py:542`). `typer.echo` prints from `resolve_mlip` and `validate_mlip`. The committee SIGTERM handler is installed on the main thread only (`committee_session.py:106`). Use **one subprocess per job**, which also gives a hard cancel (kill), and set `MPLBACKEND=Agg`.
2. **`typer.Exit` raised from library code** (`detect_mlip`, `validate_mlip`, `_validate_*`, `_ensure_mace_foundation_checkpoint`; all in `cli/utils.py`). Catch `click.exceptions.Exit` (or `typer.Exit`) and read the message from `exc.args`. Some of these calls print first, then raise with exit code 1 (`parse_relax_atoms`).
3. **Availability flags are frozen at import** (`cli/utils.py:116-119`), so the server must restart after a package install.
4. **The NEB restart blocks for 15 s** on a frame mismatch (`neb.py:146-155`). The NEB CLI `or`-defaults make 0 impossible to enter (`neb.py:238-254`). The class and CLI defaults differ (num_images 9 vs 5, interp_steps 1000 vs 100, climb False vs True).
5. **`CustomNEB.__init__` mutates `final`**: `set_cell(initial.cell, scale_atoms=True)` (`core/neb.py:116`).
6. **AutoNEB deletes earlier `{prefix}*.traj` files and `AutoNEB_iter/` in output_dir** (`core/neb.py:920-924`). It has no device option and runs on CPU.
7. **Run records are replaced per directory.** The web app needs one directory per job.
8. **`run_optimization` opens a second, unused `Trajectory(traj_file, "w")`** (`core/optimize.py:535`) on the same file the optimizer writes, and never closes it. This is harmless in practice, but beware when tailing `opt.traj`. `<stem>_convergence.csv` exists only after the run finishes.
9. **Python 3.14 is the only interpreter.** `pyproject` classifiers list 3.9-3.12 (CI uses 3.11), and whether torch, mace-torch and e3nn work on 3.14 here has not been checked. CI's ASE is 3.29; the code has an explicit guard for ASE ≤3.28 (`vibrations.py:848`).
10. **Benchmark** has no `mace_head` and no `device`, so `mace-mh-*` always fails there.
11. **No cancellation API.** Use a subprocess kill. An exception raised from an observer is the cooperative alternative, and the record then reads `failed` with the message.
12. **Docs drift** in `docs/PYTHON_API.md` about the head/task defaults (see §1.2).
