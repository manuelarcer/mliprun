# mliprun web app: implementation plan and contract

Status: authoritative contract for Phase 2 (implementation), Phase 3 (tests) and
Phase 4 (docs). Branch `feature/webapp`. Written 2026-10-15 from `RECON.md` and
the source at `c0c8250`; updated 2026-10-16 after integration to describe what
was actually built (see section 14, "Changes during implementation").

Three implementers work in parallel and do not talk to each other:
**Backend** (server, routes, jobs/queue, uploads, launch command, packaging),
**Bridge** (worker, per-calc runners, event emission, parameter schema,
calculator handling, core hooks) and **Frontend** (`static/`). A later
**Tester** sees only this file. Where this file says MUST, it is a contract
another party relies on. Where it is silent, the owner decides.

Paths are repository-relative. `src/mliprun/` is the real package path
(verified: `src/mliprun/{cli,core}`; `package-dir = {"" = "src"}`).

---

## 0. Fixed decisions (not open for change)

- Stack: FastAPI + uvicorn, WebSocket for live events. Vanilla HTML/CSS/JS
  frontend, ES modules, **no Node build step**. 3Dmol.js and Plotly from
  pinned CDN URLs. New runtime deps only `fastapi`, `uvicorn[standard]`,
  `python-multipart`, in optional extra `webapp`. Tests also use `httpx`,
  `pytest`, optional `playwright`.
- Bind `127.0.0.1` by default.
- Code in `src/mliprun/webapp/`, static assets in `src/mliprun/webapp/static/`,
  tests in `tests/webapp/`. Launch: `mlip webapp [--host --port --runs-dir
  --no-browser]` and `python -m mliprun.webapp`.
- In-memory + on-disk job registry; one queue runner; jobs run **strictly one
  at a time**; each job is its own OS subprocess
  `python -m mliprun.webapp.worker <run_dir>`. The worker appends JSON lines to
  `run_dir/events.jsonl` (flushed per line); the server tails the file and
  broadcasts over WebSocket. Cancel = SIGTERM, then SIGKILL after a grace
  period. A worker crash or nonzero exit marks the job failed; the server is
  unaffected. On server restart, jobs left `running` are marked failed.
- Core hooks: optional `observer=None` keyword on exactly `run_optimization`,
  `run_md`, `run_frequencies`, `CustomNEB.run_neb`. Default `None` gives
  identical behaviour. AutoNEB uses the optimizer-subclass trick.
- Calculators: installed MLIPs (via the package's own detection) plus `emt`.
  Device selector `auto`/`cpu`/`cuda`.
- `run_dir = <runs-dir>/<job_id>/` containing inputs, `job.json`,
  `events.jsonl`, the package's normal outputs, `summary.json`, `worker.log`,
  `reproduce.sh`. Default runs-dir `~/.mliprun/webapp-runs`.
- Live structure updates throttled by `structure_every` (default 5).

## 0.1 Decisions made in this plan (beyond the fixed ones)

| # | Decision | Why |
|---|---|---|
| D1 | Calc types offered: `singlepoint`, `optimize`, `freq`, `md`, `neb`, `autoneb`. **Not offered**: committees, `optimize batch` (replaced by multi-structure submit), `benchmark`, `md --resume`, `neb --restart`, `autoneb-results` (AutoNEB post-processing is done by the worker). | Committees need per-member env paths; restart modes depend on cwd state and a 15 s sleep (RECON §9.4). |
| D2 | `events.jsonl` has at most one writer at any moment: the server writes `job_status` lines only while no worker process exists for that job; the worker writes everything else. `seq` = 0-based line index. | Avoids interleaved writes and seq races. |
| D3 | Server-side validation is done by a Bridge function (`schema.validate_submission`), driven by the same data the form is generated from. The Backend never hard-codes calc parameters. | One source of truth. |
| D4 | Fields hidden by `visible_if` / `input_modes` MUST be omitted by the client; the server rejects them if present. | Mirrors the CLI, which refuses knobs the chosen ensemble/barostat never reads (e.g. `--compressibility`, `--pfactor`). |
| D5 | Mutating HTTP requests require header `X-Mliprun: 1`; the server sends no CORS headers; WebSocket rejects foreign `Origin`; a Host-header allow-list blocks DNS rebinding. | A web page in the user's browser could otherwise POST to `127.0.0.1` (multipart POSTs need no CORS preflight) and run jobs or read files. |
| D6 | On server restart, `pending` jobs are re-queued in creation order (they never started). `running` ones become `failed`, and a still-alive orphan worker for that run_dir is killed. | Fixed decision covers `running`; pending loses nothing by re-queueing. |
| D7 | Non-NEB calcs use the **last frame** of a multi-frame upload (ASE default, same as CLI). NEB `band` mode uses all frames. | CLI parity. |
| D8 | NEB/AutoNEB accept any ASE-readable format (CLI forces VASP). The worker also writes `inputs/initial.vasp`/`inputs/final.vasp` so `reproduce.sh` works. | Usability; CLI reproducibility preserved. |
| D9 | The bridge resolves `device="auto"` with `mliprun.core.utils.resolve_device` before passing it anywhere, because `CustomNEB.setup_calculator` forwards the device string unresolved. | `"auto"` would reach SevenNet/MACE raw in NEB. |
| D10 | `httpx` is added to the existing `dev` extra (test-only). CI installs `.[dev,neb,webapp]`. | CI's 90% diff-coverage gate would otherwise fail on untested webapp lines. |
| D11 | Tier-2 smoke test uses calculator `mace`, which the package builds as **MACE-MP-0 medium** (`build_calculator`, `cli/utils.py:654`). "small" is not reachable without a core change. | Only the package's own build path is used. |
| D12 | NEB endpoint optimization and IDPP interpolation emit `phase` events only (no per-step data): they are not among the four hooked functions. | Fixed list of hooks. |

---

## 1. Architecture

### 1.1 Diagram

```
 Browser (static/, ES modules; 3Dmol.js + Plotly from CDN)
   |  HTTP JSON (REST)                         ^  WebSocket /ws
   v                                           |  (event envelopes)
 +-------------------------------------------------------------------+
 | uvicorn  ->  FastAPI app  (server.py)          [Backend]          |
 |   routes: /api/*  /ws  /  /static/*                               |
 |   uploads.py : store + parse uploads (ase.io.read, in threadpool) |
 |   jobs.py    : JobRegistry (memory + state.json per run_dir)      |
 |                QueueRunner (one asyncio task, FIFO, 1 at a time)  |
 |                EventTailer (polls events.jsonl, 0.2 s)            |
 |                Broadcaster (WS fan-out)                           |
 |   imports (Bridge): schema.get_calc_types / validate_submission,  |
 |                     calculators.list_calculators / detect_devices,|
 |                     events.EventWriter / read_events              |
 +-------------------------------|-----------------------------------+
                                 | spawn: [python, -m, mliprun.webapp.worker, run_dir]
                                 | cwd=run_dir, stdout+stderr -> worker.log
                                 | SIGTERM (cancel) ... SIGKILL after grace
                                 v
 +-------------------------------------------------------------------+
 | worker.py  (separate OS process)                [Bridge]          |
 |   read job.json -> write reproduce.sh/.json -> read inputs        |
 |   -> build calculator (calculators.py) -> runners.py:<calc_type>  |
 |   -> core function with observer=ObserverAdapter                  |
 |   -> events.jsonl lines (events.py) -> summary.json -> exit code  |
 +-------------------------------|-----------------------------------+
                                 v
 mliprun.core.{singlepoint,optimize,vibrations,md,neb}  (+ observer hooks)
 writes the package's normal outputs + mliprun_run.json into run_dir
```

### 1.2 Files and owners (disjoint)

| File | Owner | Purpose |
|---|---|---|
| `src/mliprun/webapp/__init__.py` | Backend | Empty except docstring. MUST NOT import fastapi/uvicorn/ase-heavy modules. |
| `src/mliprun/webapp/__main__.py` | Backend | `python -m mliprun.webapp` -> same as `mlip webapp` (`typer.run(webapp_command)`). |
| `src/mliprun/webapp/cli.py` | Backend | `webapp_command(host, port, runs_dir, no_browser)` typer function; lazy-imports fastapi/uvicorn, prints install hint and exits 1 if missing. |
| `src/mliprun/webapp/server.py` | Backend | `create_app(...)`, all routes, WebSocket, middleware, error handlers. |
| `src/mliprun/webapp/jobs.py` | Backend | JobRegistry, state machine, QueueRunner, EventTailer, Broadcaster, restart recovery. |
| `src/mliprun/webapp/uploads.py` | Backend | Upload store, ASE parsing, structure payloads (xyz strings). |
| `src/mliprun/webapp/errors.py` | Backend | `ApiError` and the error JSON format (section 5). |
| `src/mliprun/cli/main.py` (edit) | Backend | Register `webapp` command (lazy). |
| `pyproject.toml` (edit) | Backend | `webapp` extra; `httpx` into `dev`; package data for `static/**`. |
| `.github/workflows/tests.yml` (edit) | Backend | Install `.[dev,neb,webapp]`. |
| `MANIFEST.in` (create/edit if needed) | Backend | Ship `static/` in sdists. |
| `src/mliprun/webapp/schema.py` | Bridge | `CALC_TYPES` data, `get_calc_types()`, `validate_submission()`, `ValidationFailed`. |
| `src/mliprun/webapp/calculators.py` | Bridge | `list_calculators()`, `detect_devices()`, `make_calculator()`, EMT element set. |
| `src/mliprun/webapp/events.py` | Bridge | `EventWriter`, `read_events`, `encode_frame`, `to_jsonable`. Used by Backend too. |
| `src/mliprun/webapp/worker.py` | Bridge | Worker entry point (`python -m mliprun.webapp.worker <run_dir>`). |
| `src/mliprun/webapp/runners.py` | Bridge | One runner per calc type; result payloads; ObserverAdapter; AutoNEB optimizer subclass. |
| `src/mliprun/webapp/reproduce.py` | Bridge | Builds `reproduce.sh` + `reproduce.json`. |
| `src/mliprun/core/optimize.py` (edit) | Bridge | Hook (section 9). |
| `src/mliprun/core/md.py` (edit) | Bridge | Hook. |
| `src/mliprun/core/vibrations.py` (edit) | Bridge | Hook. |
| `src/mliprun/core/neb.py` (edit) | Bridge | Hook. |
| `src/mliprun/webapp/static/index.html` | Frontend | Single page. |
| `src/mliprun/webapp/static/css/app.css` | Frontend | Styles. |
| `src/mliprun/webapp/static/js/main.js` | Frontend | Bootstrapping, routing between panels (hash-based `#/jobs/<id>`). |
| `src/mliprun/webapp/static/js/api.js` | Frontend | REST wrapper (adds `X-Mliprun: 1`). |
| `src/mliprun/webapp/static/js/ws.js` | Frontend | WebSocket client with reconnect + replay. |
| `src/mliprun/webapp/static/js/form.js` | Frontend | Form generated from `/api/calc-types` and `/api/calculators`. |
| `src/mliprun/webapp/static/js/structures.js` | Frontend | Upload panel, structure list, previews. |
| `src/mliprun/webapp/static/js/viewer.js` | Frontend | 3Dmol wrapper (load xyz, update frame, draw cell, play band/modes). |
| `src/mliprun/webapp/static/js/plots.js` | Frontend | Plotly charts per calc type. |
| `src/mliprun/webapp/static/js/jobs.js` | Frontend | Job list and job detail views. |
| `tests/webapp/conftest.py` | Backend | Shared fixtures (section 10.1). MUST NOT importorskip at module level. |
| `tests/webapp/fake_worker.py` | Backend | Scripted stand-in worker for backend tests. |
| `tests/webapp/test_api_*.py`, `test_ws.py`, `test_launch.py` | Backend | Backend unit tests. |
| `tests/webapp/test_schema.py`, `test_calculators.py`, `test_events.py`, `test_worker.py`, `test_core_hooks.py`, `test_reproduce.py` | Bridge | Bridge unit tests. |
| `tests/webapp/test_tier1_emt.py`, `test_tier2_mace.py`, `test_robustness.py`, `test_browser.py` | Tester (Phase 3) | Integration tests. |
| `docs/WEBAPP.md`, README section, `AGENTS.md` line | Phase 4 | Docs. |

No other file may be edited. In particular `tests/goldens/*` MUST NOT change.

### 1.3 Python constraints for all webapp Python code

- `from __future__ import annotations`; Python >= 3.9 syntax (no `match`,
  no `X | Y` at runtime). Lazy-import heavy packages (torch, MLIPs) as the
  rest of the package does.
- `mliprun.webapp.schema`, `calculators`, `events`, `reproduce`, `runners`,
  `worker` MUST import without fastapi installed.
- No new runtime dependency beyond the fixed three.

---

## 2. On-disk layout

```
<runs-dir>/                         default ~/.mliprun/webapp-runs (expanduser, created)
  _uploads/<upload_id>/
      <sanitized filename>          the uploaded bytes, unchanged
      meta.json                     UploadInfo (section 4.3) minus xyz/frames_xyz, plus "format"
  <job_id>/                         run_dir
      job.json                      written by Backend at submission, never modified
      state.json                    written by Backend (atomic tmp + os.replace) on every transition
      events.jsonl                  section 6
      inputs/<role>/<filename>      copy of the upload; role in structure|initial|final|band
      inputs/initial.vasp, inputs/final.vasp   neb/autoneb only, written by worker (D8)
      reproduce.sh, reproduce.json  written by worker before the calculation
      worker.log                    worker stdout+stderr (append)
      summary.json                  written by worker at the end; by Backend if the worker died
      ... the package's normal outputs (opt.traj, mliprun_run.json, md_energy.csv, 00/POSCAR ...)
```

- `job_id` format: `YYYYMMDD-HHMMSS-<calc_type>-<6 lowercase hex>` (UTC), regex
  `^[0-9]{8}-[0-9]{6}-[a-z]+-[0-9a-f]{6}$`. Sorts by creation time.
- `upload_id`: 12 lowercase hex chars, regex `^[0-9a-f]{12}$`.
- Filename sanitizing: take the basename; replace every char outside
  `[A-Za-z0-9._+-]` with `_`; strip leading dots; empty -> `structure`.
  `POSCAR`/`CONTCAR` names are preserved.
- Package outputs go directly into `run_dir` (the core `output_dir`). None of
  the package's output names collide with the webapp's own files.
- REST timestamps are ISO-8601 UTC with milliseconds and `Z`
  (`2026-10-15T14:23:01.123Z`). Event `t` is Unix epoch seconds (float).

### 2.1 `job.json` (schema_version 1)

```json
{
  "schema_version": 1,
  "job_id": "20261015-142301-optimize-a1b2c3",
  "calc_type": "optimize",
  "label": "POSCAR",
  "created_at": "2026-10-15T14:23:01.123Z",
  "calculator": {"name": "emt", "device": "auto",
                  "uma_task": null, "mace_head": null, "sevennet_task": null},
  "params": {"optimizer": "bfgs", "fmax": 0.05, "max_steps": 200,
             "relax_cell": false, "plot": false},
  "param_sources": {"optimizer": "default", "fmax": "user", "max_steps": "default",
                    "relax_cell": "default", "plot": "default"},
  "structure_every": 5,
  "inputs": {
    "structure": {"upload_id": "0123456789ab", "filename": "POSCAR",
                  "path": "inputs/structure/POSCAR", "format": "vasp"}
  }
}
```

- `params` = output of `validate_submission` (all applicable fields, defaults
  filled, hidden fields absent).
- `inputs` keys: `structure` (single), or `initial`+`final` (pair), or `band`.
- `format` = the ASE format string detected or forced at upload; the worker
  MUST pass it to `ase.io.read(format=...)`.

### 2.2 `state.json` (Backend-owned)

```json
{"job_id": "...", "status": "running", "created_at": "...", "started_at": "...",
 "finished_at": null, "pid": 12345, "returncode": null,
 "cancel_requested": false, "reason": null, "result_brief": null}
```

---

## 3. Parameter schema (`schema.py`, Bridge) served at `GET /api/calc-types`

### 3.1 Response format

```json
{
  "schema_version": 1,
  "calc_types": [
    {"name": "optimize", "label": "Geometry optimization",
     "description": "Relax atomic positions (optionally the cell) to a force threshold.",
     "inputs": "single",            // "single" | "pair" | "pair_or_band"
     "cli_command": "optimize run",
     "fields": [FieldSpec, ...]}
  ]
}
```

`FieldSpec` (every key always present; use `null`/`false` when not applicable):

| Key | Type | Meaning |
|---|---|---|
| `name` | str | param key used in `params` |
| `label` | str | short UI label |
| `type` | str | `float`, `int`, `bool`, `choice`, `indices`, `mask3` |
| `default` | any | value used when the field is applicable and omitted |
| `unit` | str/null | display unit, e.g. `eV/Å` |
| `help` | str | one or two sentences |
| `min`, `max` | number/null | bounds |
| `min_exclusive`, `max_exclusive` | bool | strict bounds |
| `choices` | list/null | allowed values for `choice` (strings, or ints for `nfree`) |
| `nullable` | bool | `null` is a valid value |
| `advanced` | bool | UI shows it in a collapsed "Advanced" section |
| `visible_if` | list/null | null = always. Else list of clauses; field applies if **any** clause holds; a clause is `{field: [allowed values]}` and holds if **every** listed field's current (effective) value is in its list |
| `input_modes` | list/null | NEB only: `["pair"]` means the field applies only with initial+final input. null = all modes |
| `cli_flag` | str/null | the `mlip` CLI flag (for help/reproduce), null if no CLI equivalent |

Type semantics: `float` accepts JSON numbers (not bools); `int` accepts JSON
integers or floats with zero fractional part (not bools); `bool` only
`true`/`false`; `choice` exact match (case-sensitive); `indices` is a string
`"0,1,5"`, `"12-30"` or a mix (inclusive ranges, whitespace ignored), every
index in `0..natoms-1`, non-empty, parsed with
`mliprun.core.vibrations.parse_indices`; `mask3` is a string `"x,y,z"` of
0/1, validated with `mliprun.core.md.normalize_barostat_mask`.

Effective value for `visible_if`: submitted value if present, else default.
Visibility is evaluated after defaults are applied to visible fields, in field
order (all controlling fields precede the fields they control).

### 3.2 Fields per calc type

Units: energy eV, length Å, force eV/Å, time fs, temperature K, pressure GPa.

**singlepoint** (`inputs: single`, CLI `singlepoint run`)

| name | type | default | bounds/choices | visible_if | help |
|---|---|---|---|---|---|
| stress | choice | `auto` | auto, on, off | - | Stress tensor. auto = only when the cell is periodic in all three directions. Maps to core `stress=None/True/False`. |

**optimize** (`single`, `optimize run`)

| name | type | default | bounds/choices | adv | help |
|---|---|---|---|---|---|
| optimizer | choice | bfgs | bfgs, lbfgs, fire, bfgsls, gpmin, mdmin | | ASE optimizer |
| fmax | float | 0.05 | (0, 10] eV/Å | | Convergence: max force (includes cell virials with relax_cell) |
| max_steps | int | 200 | [0, 100000] | | Step limit |
| relax_cell | bool | false | | | Also relax the cell (FrechetCellFilter) |
| plot | bool | false | | yes | Write convergence PNG |

**freq** (`single`, `freq run`)

| name | type | default | bounds/choices | adv | help |
|---|---|---|---|---|---|
| indices | indices | null (nullable) | 0..N-1 | | Atoms to displace; empty = all atoms not under FixAtoms |
| delta | float | 0.01 | (0, 0.1] Å | | Displacement |
| nfree | choice | 2 | 2, 4 | | Points per coordinate (4 = twice the cost) |
| direction | choice | central | central, forward, backward | yes | Finite-difference direction |
| method | choice | standard | standard, frederiksen | yes | Hessian method |
| write_modes | choice | imaginary | none, imaginary, all | | Write `freq.<n>.traj` mode animations |
| imaginary_floor | float | 10.0 | [0, 1000] cm⁻¹ | yes | Imaginary modes below this magnitude are not counted |
| expect_fmax | float | null (nullable) | (0, 10] eV/Å | yes | Warn if the input geometry's fmax exceeds this |

**md** (`single`, `md run`)

| name | type | default | bounds/choices | visible_if | adv |
|---|---|---|---|---|---|
| ensemble | choice | nvt | nve, nvt, npt | - | |
| steps | int | 1000 | [1, 10000000] | - | |
| timestep | float | 1.0 | (0, 10] fs | - | |
| temperature | float | 300 | (0, 100000] K | `[{"ensemble":["nvt","npt"]}]` | |
| thermostat | choice | langevin | langevin, nose-hoover, berendsen | `[{"ensemble":["nvt"]}]` | |
| barostat | choice | npt | npt, berendsen | `[{"ensemble":["npt"]}]` | |
| friction | float | 0.01 | (0, 10] 1/fs | `[{"ensemble":["nvt"],"thermostat":["langevin"]}]` | yes |
| ttime | float | 25.0 | (0, 100000] fs | `[{"ensemble":["nvt"],"thermostat":["nose-hoover"]},{"ensemble":["npt"],"barostat":["npt"]}]` | yes |
| taut | float | 100.0 | (0, 100000] fs | `[{"ensemble":["nvt"],"thermostat":["berendsen"]},{"ensemble":["npt"],"barostat":["berendsen"]}]` | yes |
| pressure | float | 0.0 | [-1000, 1000] GPa | `[{"ensemble":["npt"]}]` | |
| taup | float | 1000.0 | (0, 1e6] fs | `[{"ensemble":["npt"],"barostat":["berendsen"]}]` | yes |
| compressibility | float | 0.457 | (0, 100] 1/GPa | `[{"ensemble":["npt"],"barostat":["berendsen"]}]` | yes |
| pfactor | float | null (nullable) | (0, 1e12] GPa·fs² | `[{"ensemble":["npt"],"barostat":["npt"]}]` | yes |
| barostat_mask | mask3 | `"1,1,1"` | each 0/1 | `[{"ensemble":["npt"]}]` | yes |
| log_interval | int | 10 | [1, 100000] | - | |
| traj_interval | int | 100 | [1, 100000] | - | |
| plot | bool | false | | - | yes |

Field order is the table order: `barostat` directly follows `thermostat`
because `ttime` and `taut` are controlled by it (rule of 3.1: controlling
fields precede the fields they control).

Help notes the Bridge MUST include: NVE starts from rest (no velocities are
drawn, `core/md.py` only draws Maxwell-Boltzmann for nvt/npt); `pfactor` null
= automatic (75 fs)² × 100 GPa; one `md_step` event and one CSV row every
`log_interval` steps. Bridge converts `pfactor` with
`core.md.PFACTOR_GPA_FS2` before calling `run_md` (CLI parity, `md.py:221`).
Not exposed: `resume`, `log_path`, `csv_flush_every` (core default 100).

**neb** (`pair_or_band`, `neb run`)

| name | type | default | bounds/choices | visible_if | input_modes | adv |
|---|---|---|---|---|---|---|
| n_images | int | 5 | [1, 30] | - | pair | |
| interpolation | choice | idpp | idpp, linear | - | pair | |
| interp_fmax | float | 0.1 | (0, 10] | `[{"interpolation":["idpp"]}]` | pair | yes |
| interp_steps | int | 100 | [1, 100000] | `[{"interpolation":["idpp"]}]` | pair | yes |
| fmax | float | 0.05 | (0, 10] eV/Å | - | - | |
| k | float | 0.1 | (0, 100] eV/Å² | - | - | |
| climb | bool | true | | - | - | |
| neb_optimizer | choice | fire | fire, mdmin, bfgs, lbfgs | - | - | |
| neb_max_steps | int | 600 | [1, 100000] | - | - | |
| dyneb | bool | false | | - | - | yes |
| scale_fmax | float | 0.0 | [0, 100] | `[{"dyneb":[true]}]` | - | yes |
| optimize_endpoints | bool | true | | - | pair | |
| endpoint_fmax | float | 0.01 | (0, 10] eV/Å | `[{"optimize_endpoints":[true]}]` | pair | |
| endpoint_optimizer | choice | bfgs | bfgs, lbfgs, fire | `[{"optimize_endpoints":[true]}]` | pair | |
| endpoint_max_steps | int | 200 | [1, 100000] | `[{"optimize_endpoints":[true]}]` | pair | |
| relax_atoms | indices | null (nullable) | 0..N-1 | - | pair | yes |
| plot | bool | false | | - | - | yes |

Help MUST state: `n_images` = intermediate images (total = n + 2);
`relax_atoms` fixes all other atoms and skips IDPP; in band mode the uploaded
frames are the starting band (no interpolation, no endpoint optimization).

**autoneb** (`pair`, `autoneb run`)

| name | type | default | bounds/choices | visible_if | adv |
|---|---|---|---|---|---|
| n_max | int | 9 | [3, 50] | - | |
| fmax | float | 0.05 | (0, 10] | - | |
| k | float | 0.1 | (0, 100] | - | |
| climb | bool | true | | - | |
| interpolate_method | choice | idpp | idpp, linear | - | |
| space_energy_ratio | float | 0.5 | [0, 1] | - | yes |
| maxsteps | int | 10000 | [1, 1000000] | - | yes |
| optimize_endpoints | bool | true | | - | |
| endpoint_fmax | float | 0.01 | (0, 10] | `[{"optimize_endpoints":[true]}]` | |
| endpoint_optimizer | choice | bfgs | bfgs, lbfgs, fire | `[{"optimize_endpoints":[true]}]` | |
| endpoint_max_steps | int | 200 | [1, 100000] | `[{"optimize_endpoints":[true]}]` | |

Fixed internally: `n_simul=1` (MPI not supported), `prefix="autoneb"`,
optimizer FIRE (CLI parity). `relax_atoms` not offered (core warns it may not
work).

### 3.3 `validate_submission` (Bridge API used by Backend)

```python
class ValidationFailed(Exception):
    def __init__(self, errors: list[dict]): ...   # each {"field": str, "message": str}
    errors: list[dict]

def get_calc_types() -> dict: ...                 # the JSON of 3.1, deep-copied

def validate_submission(calc_type: str, calculator: dict, params: dict,
                        inputs: dict[str, dict]) -> dict:
    """Return {"calculator": <normalized>, "params": <normalized>,
               "param_sources": {name: "user"|"default"}}.
    Raise ValidationFailed with ALL errors found (not just the first)."""
```

`inputs` maps role -> StructureMeta built by Backend from the upload:
`{"natoms": int, "symbols": [str], "n_frames": int, "pbc": [bool,bool,bool]}`
(symbols of the frame that will be used; for `band`, of frame 0, and Backend
adds `"frames_same_symbols": bool`). Roles: `structure`, or `initial`+`final`,
or `band`.

Rules (field paths in errors use dotted form: `calc_type`,
`calculator.name`, `calculator.uma_task`, `params.fmax`, `inputs.final`):

1. `calc_type` must be one of section 3.2.
2. Input roles must match `inputs` kind (`single`: `structure`; `pair`:
   `initial`+`final`; `pair_or_band`: either pair or `band`).
3. Unknown param key -> error "unknown parameter". Param not applicable
   (hidden by `visible_if` or `input_modes`) but present -> error
   "does not apply when <controlling field>=<value>".
4. Type/bounds/choices/nullable per FieldSpec.
5. `indices` fields: parsed and range-checked against `natoms`; normalized
   value stays the string as submitted (stripped); runners convert to
   `list[int]`.
6. NEB/AutoNEB pair: `initial.natoms == final.natoms` and symbol lists equal
   (same order). Band: `n_frames >= 3`, `n_frames <= 52`,
   `frames_same_symbols` true.
7. Calculator (section 7.2 rules).
8. EMT: all symbols in `{H, C, N, O, Al, Ni, Cu, Pd, Ag, Pt, Au}` else
   error on `calculator.name`, listing the unsupported elements.

Normalized calculator dict always has the five keys `name, device, uma_task,
mace_head, sevennet_task` (unused task keys `null`).

---

## 4. REST API (Backend)

All JSON responses `Content-Type: application/json`. All POST/DELETE requests
MUST carry header `X-Mliprun: 1`, else 403 `forbidden` (D5). Errors use
section 5.

### 4.1 Summary table

| Method | Path | Success | Errors |
|---|---|---|---|
| GET | `/` | 200 `index.html` | |
| GET | `/static/{path}` | 200 file | 404 |
| GET | `/api/health` | 200 `{"status":"ok","mliprun_version":str,"runs_dir":str,"queue_length":int,"running_job":str|null}` | |
| GET | `/api/calc-types` | 200 section 3.1 | |
| GET | `/api/calculators` | 200 section 7.1 | |
| POST | `/api/uploads` | 200 `{"uploads":[UploadInfo...]}` | 400 no files; 403; 413 >50 files |
| GET | `/api/uploads` | 200 `{"uploads":[UploadInfo without xyz, frames_xyz]}` newest first | |
| GET | `/api/uploads/{upload_id}` | 200 UploadInfo (full) | 404 |
| POST | `/api/jobs` | 201 `{"jobs":[JobSummary...]}` | 400, 403, 404 unknown upload id, 422 validation |
| GET | `/api/jobs?status=a,b&limit=n` | 200 `{"jobs":[JobSummary...]}` newest first; `limit` default 500, max 5000 | 422 bad status |
| GET | `/api/jobs/{job_id}` | 200 JobDetail | 404 |
| POST | `/api/jobs/{job_id}/cancel` | 200 `{"job": JobSummary}` | 403, 404, 409 already terminal |
| GET | `/api/jobs/{job_id}/events?since=0&limit=1000` | 200 `{"events":[...],"next_seq":int,"terminal":bool}` | 404, 422 |
| GET | `/api/jobs/{job_id}/files` | 200 `{"files":[{"path":str,"size":int,"mtime":ISO}]}` sorted by path | 404 |
| GET | `/api/jobs/{job_id}/files/{path:path}` | 200 file (`?inline=1` -> `text/plain; charset=utf-8` for text, max 5 MB else 413) | 400 traversal, 404 |
| GET | `/api/jobs/{job_id}/zip` | 200 `application/zip`, `Content-Disposition: attachment; filename="<job_id>.zip"` | 404 |
| WS | `/ws` | section 6 | close 1008 bad origin |

### 4.2 POST /api/uploads

`multipart/form-data`: one or more parts named `files`; optional text part
`format` (an ASE format name applied to every file in this request; empty =
auto-detect). Per file limit 50 MB (oversize -> per-file error, not 413).
Max 50 files per request (more -> 413).

Processing per file (in a threadpool): save to a temp dir; detect format with
`ase.io.formats.filetype(path)` unless `format` given; `frames =
ase.io.read(path, index=":", format=fmt)`; on any exception -> per-file error,
file not stored. On success move into `_uploads/<id>/` and write `meta.json`.

HTTP status is 200 whenever the request was well-formed, even if every file
failed (each entry has `ok`).

### 4.3 UploadInfo

```json
{"id": "0123456789ab", "filename": "POSCAR", "ok": true, "format": "vasp",
 "created_at": "...", "formula": "Cu32", "natoms": 32, "n_frames": 1,
 "symbols": ["Cu", "..."], "cell": [[..],[..],[..]], "pbc": [true,true,true],
 "has_constraints": false,
 "xyz": "<extended XYZ of the LAST frame>",
 "frames_xyz": null}
```
Failure entry: `{"id": null, "filename": "x.foo", "ok": false, "error": "message"}`.

- `formula`, `natoms`, `symbols`, `cell`, `pbc`, `has_constraints` describe the
  **last** frame (the one non-NEB calcs use).
- `xyz` format (defined once, Backend generates, Frontend parses with 3Dmol
  `"xyz"`): line 1 natoms; line 2
  `Lattice="a1x a1y a1z a2x ... a3z" Properties=species:S:1:pos:R:3 pbc="T T F"`;
  then `"%s %.6f %.6f %.6f"` per atom. Valid extended XYZ.
- `frames_xyz`: concatenation of the same per-frame blocks for all frames,
  when `2 <= n_frames <= 500`; else `null`.

### 4.4 POST /api/jobs

```json
{
  "calc_type": "optimize",
  "calculator": {"name": "emt", "device": "auto"},
  "params": {"fmax": 0.02},
  "structure_every": 5,
  "label": null,
  "structure_ids": ["0123456789ab", "ba9876543210"]
}
```
- single-input types: `structure_ids` (1..200 upload ids) -> one job per id,
  in the given order.
- `neb`: exactly one of (`initial_id` + `final_id`) or `band_id` -> one job.
- `autoneb`: `initial_id` + `final_id` -> one job.
- `params`: the browser client sends only visible fields whose value differs
  from the default, so `param_sources` records exactly the changed values as
  `user` (an API client may send defaults explicitly; they are then `user`).
- `structure_every`: int 0..100000, default 5 (0 = never send positions).
- `label`: optional string <= 80 chars; default = filename (pair:
  `"<initial> -> <final>"`).
- Unknown top-level keys -> 422. Missing/extra id keys for the calc type -> 422.
- Unknown upload id or failed upload -> 404 `not_found` with field path.
- Backend calls `validate_submission` once per job. For multi-structure
  submits, errors from structure k get field prefix `structure_ids[k].`
  (e.g. `structure_ids[1].params.indices`). **All-or-nothing**: if any job
  fails validation, no job is created.
- On success, for each job: create run_dir, copy inputs, write `job.json`,
  `state.json` (pending), append `job_status pending` event (seq 0), enqueue.
  Response 201 with JobSummary list in submission order.

### 4.5 JobSummary and JobDetail

```json
JobSummary = {
  "job_id": "...", "calc_type": "md", "label": "POSCAR", "status": "running",
  "created_at": "...", "started_at": "..."|null, "finished_at": "..."|null,
  "calculator": {"name": "emt", "device": "auto"},
  "formula": "Cu32", "natoms": 32,
  "queue_position": null,              // 1-based among pending, else null
  "cancel_requested": false,
  "progress": {"step": 120, "total": 1000, "energy": -3.21, "fmax": null,
               "text": "step 120/1000"} | null,
  "result_brief": {...} | null,
  "error": "message" | null
}
```
`progress` is derived by Backend from the latest progress event:
`opt_step` -> step, total=`params.max_steps`, energy, fmax;
`md_step` -> step, total=`params.steps`, energy=`etot`;
`neb_iter` -> step=`iteration`, total=`neb_max_steps` (neb) or null (autoneb),
energy=`barrier`, fmax; `freq_progress` -> step=`done`, total=`total`.
`text` is a short human string of the owner's choice.

`result_brief` = these keys copied from the `result` event when present:
singlepoint `energy_eV, fmax_free_eV_per_A`; optimize `converged,
final_energy_eV, final_fmax_eV_per_A, steps`; freq `n_imaginary, zpe_eV`; md
`mean_temperature_K, mean_total_energy_eV`; neb/autoneb `forward_barrier_eV,
reverse_barrier_eV, reaction_energy_eV`.

`error` = `reason` of the final failed `job_status`, else the last `error`
event's `message`, else null.

```json
JobDetail = JobSummary + {
  "params": {...}, "param_sources": {...}, "structure_every": 5,
  "calculator": {all five keys},
  "inputs": {"<role>": {"upload_id","filename","natoms","symbols","cell","pbc",
                         "xyz","frames_xyz"}},
  "run_dir": "/abs/path",
  "returncode": int|null,
  "result": <result event payload, section 6.3> | null,
  "reproduce": {"status": "exact"|"approximate"|"no", "notes": [str]} | null,
  "last_seq": int
}
```
`result` comes from the in-memory last `result` event, or after a restart
from re-reading `events.jsonl` (last line of type `result`).

### 4.6 Events, files, zip

- `GET /events`: returns events with `seq >= since`, at most `limit`
  (1..5000), in seq order; `next_seq` = last returned seq + 1 (or `since` if
  none); `terminal` = job status is terminal **and** `next_seq` equals the
  number of complete lines in the file.
- Files: recursive listing of run_dir (max 10000 entries); paths relative,
  `/`-separated. Download resolves `(run_dir / path).resolve()` and MUST be
  inside `run_dir.resolve()` (else 400 `bad_request`); directories -> 404.
- Zip: all files under run_dir, archive paths prefixed `<job_id>/`. Allowed
  at any status (snapshot). Built into a `tempfile.SpooledTemporaryFile` and
  streamed.

### 4.7 Server construction (for tests)

```python
def create_app(runs_dir: Path, *, start_runner: bool = True,
               allowed_hosts: Optional[list[str]] = None,   # None = no Host check
               worker_cmd: Optional[list[str]] = None,      # default [sys.executable, "-m", "mliprun.webapp.worker"]
               cancel_grace_s: float = 10.0,
               poll_interval_s: float = 0.2) -> FastAPI
```
The run_dir path is appended as the last argv element of `worker_cmd`. The
queue runner starts/stops with the FastAPI lifespan (so `with
TestClient(app):` runs it). On shutdown: cancel the running job (SIGTERM,
wait grace, SIGKILL), write its final status, then exit.

### 4.8 `mlip webapp`

`mlip webapp [--host 127.0.0.1] [--port 8765] [--runs-dir ~/.mliprun/webapp-runs] [--no-browser]`

- If fastapi/uvicorn missing: print `The web app needs extra packages: pip
  install "mliprun[webapp]"` (and the editable form `pip install -e
  ".[webapp]"`), exit 1.
- Host loopback (`127.0.0.1`, `localhost`, `::1`): `allowed_hosts =
  ["127.0.0.1:<port>", "localhost:<port>", "[::1]:<port>"]`. Otherwise print a
  warning that there is no authentication and anyone who can reach the port
  can run jobs and read files, and disable the Host check.
- Prints the URL; unless `--no-browser`, opens it with `webbrowser.open`
  about 1 s after startup.
- `cli/main.py` registration MUST NOT import fastapi at import time (the
  command module imports it inside the function). `mlip --help` must work
  without the extra.

---

## 5. Error format and validation

Every non-2xx JSON response (including FastAPI's own request-validation
errors, which Backend MUST convert):

```json
{"error": {"code": "validation_error",
           "message": "2 problems with the submission",
           "details": [{"field": "params.fmax", "message": "must be > 0"},
                       {"field": "calculator.uma_task", "message": "required for uma-s-1p2; one of omat, omc, ..."}]}}
```

| HTTP | code | When |
|---|---|---|
| 400 | `bad_request` | malformed body, no files, path traversal |
| 403 | `forbidden` | missing `X-Mliprun` header, Host not allowed |
| 404 | `not_found` | unknown job/upload/file/route |
| 409 | `conflict` | cancel of a terminal job |
| 413 | `too_large` | too many files, inline file too big |
| 422 | `validation_error` | schema/param/calculator/structure validation; `details` non-empty |
| 500 | `internal_error` | unexpected; message is the exception text, no traceback |

`details` is `[]` for non-validation errors.

---

## 6. Events and WebSocket

### 6.1 Envelope

Each line of `events.jsonl` and each WebSocket event message:

```json
{"job_id": "20261015-...", "seq": 17, "t": 1760537000.123, "type": "opt_step", ...payload}
```
Payload fields are top-level siblings of the envelope. `seq` = 0-based line
index in that job's `events.jsonl` (D2). JSON encoding: numpy -> Python,
NaN/inf -> `null` (`events.to_jsonable`), `allow_nan=False`.

### 6.2 Frame (positions payload, defined once)

```json
Frame = {"positions": [[x, y, z], ...], "cell": [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]}
```
Å, rounded to 4 decimals, atom order = input order. `cell` always present
(zeros when undefined). Symbols are not repeated: the client takes them from
JobDetail `inputs` (constant for all calc types; AutoNEB images share them).

Throttling (`structure_every = S`): the ObserverAdapter counts calls per
event type from 0; it attaches `frame` (or `band`) on call k iff `S > 0` and
`k % S == 0`. The first event therefore carries a frame. `result` always
carries final frames regardless of S.

### 6.3 Event types

| type | Writer | Payload | Units / notes |
|---|---|---|---|
| `job_status` | server | `status`, `reason` (str/null), `returncode` (int/null) | seq 0 is always `pending` |
| `phase` | worker | `phase`, `message` | phases: `setup`, `load_model`, `endpoints`, `interpolation`, `run`, `postprocess` |
| `log` | worker | `level` (`info`/`warning`/`error`), `message`, `logger` | Python logging from `mliprun.*` at INFO+, plus `warnings`. Rate limit 50/s; excess dropped and one warning `"N log messages suppressed"` emitted |
| `opt_step` | worker | `step` int, `energy` eV, `fmax` eV/Å, `frame`? | every optimizer step (incl. step 0) |
| `md_step` | worker | `step` int, `time_fs`, `epot`, `ekin`, `etot` eV, `temperature` K, `pressure_GPa`? `volume_A3`? (NPT only), `frame`? | every `log_interval` steps |
| `neb_iter` | worker | `iteration` int, `energies` [eV abs, per image, endpoints included], `rel_energies` [E-E0], `fmax` eV/Å, `barrier` eV (= max(E) - E0), `highest_image` int, `n_images` int (incl. endpoints), `band`? [Frame per image], `autoneb`? `{"sub_step": int}` | NEB: every iteration. AutoNEB: per sub-relaxation step, energies/band of the **sub-band** being relaxed; an energy that cannot be read is `null` |
| `freq_progress` | worker | `done` int, `total` int | per computed displacement; `total = 1 + 6n` (nfree 2) or `1 + 12n` |
| `result` | worker | section 6.4 | exactly one per successful run, just before exit 0 |
| `error` | worker | `message`, `exception_type`, `traceback` | before exit 1/2 |

### 6.4 `result` payloads

Common keys: `calc_type`, `record_status` (top-level `status` of
`mliprun_run.json`: `converged`/`not_converged`/`completed`),
`record_results` (its `results` block verbatim), `walltime_s` (float/null).

| calc_type | Extra keys |
|---|---|
| singlepoint | `energy_eV`, `fmax_free_eV_per_A`, `fmax_all_eV_per_A`, `stress_GPa` (6-list Voigt or null), `stress_unavailable_reason`, `forces`: list of `{index, symbol, fx, fy, fz, f_norm, free}` in eV/Å (from `singlepoint_forces.csv`; `free` = any of free_x/y/z), `final_frame` |
| optimize | `converged`, `final_energy_eV`, `final_fmax_eV_per_A`, `steps`, `final_frame` |
| freq | `modes`: list of `{index, frequency_cm1, energy_meV, imaginary}` (signed: imaginary modes have negative frequency, as in `freq_frequencies.csv`), `n_modes`, `n_imaginary`, `n_imaginary_raw`, `imaginary_floor_cm1`, `zpe_eV`, `fmax_at_input_free_eV_per_A`, `fmax_warning`, `mode_files` (list of relative paths `freq.<n>.traj` that exist), `final_frame` (input geometry) |
| md | `steps`, `ensemble`, all keys of record results (`mean_temperature_K`, `mean_total_energy_eV`, ...), `final_frame` |
| neb | `converged`, `steps`, `energies_eV`, `relative_energies_eV`, `forward_barrier_eV`, `reverse_barrier_eV`, `reaction_energy_eV`, `ts_image_index`, `ts_at_endpoint`, `final_fmax_eV_per_A`, `band` (final Frames) |
| autoneb | same as neb, `converged` = record_status == "converged", `steps` null, `final_fmax_eV_per_A` null; energies/band from the last frame of each `autoneb???.traj` in index order |

### 6.5 WebSocket `/ws`

- Origin check: if an `Origin` header is present and its host:port is not the
  request Host, close with code 1008.
- Client -> server messages (JSON text):
  - `{"action": "subscribe", "job_id": "<id>", "since": <int>|null}`:
    per-job subscription. With `since` = N, the server first replays all
    events with `seq >= N` from the file, then streams live ones, with no gap
    or duplicate (the replay and the live stream are joined under the job's
    lock). `since: null` = live only.
  - `{"action": "subscribe", "job_id": "*"}`: all jobs, live only, and only
    types `job_status`, `result`, `error` (to keep the job list light).
  - `{"action": "unsubscribe", "job_id": "<id>"|"*"}`.
- Server -> client control messages (no `seq`):
  `{"type": "ws_ack", "action": "subscribe", "job_id": ..., "replayed": int}`
  (sent **after** the replayed events, so `replayed` is the final count),
  `{"type": "ws_error", "code": "not_found"|"bad_request", "message": ...}`,
  `{"type": "ws_ping", "t": float}` every 20 s.
- An event is delivered at most once per connection even if both `*` and the
  per-job subscription match.
- Recommended client pattern (Frontend MUST do this): subscribe `*`, then GET
  `/api/jobs`; for the open job, GET JobDetail then subscribe with
  `since = last_seq + 1`; on reconnect, resubscribe with `since = last seen
  seq + 1`.

---

## 7. Calculators (`calculators.py`, Bridge) served at `GET /api/calculators`

### 7.1 Response

```json
{
  "calculators": [
    {"name": "emt", "label": "EMT (ASE built-in; demo/testing only)", "family": "emt",
     "task_field": null, "task_choices": [], "task_required": false,
     "elements": ["Ag","Al","Au","C","Cu","H","N","Ni","O","Pd","Pt"],
     "cli_reproducible": false, "note": "Not an mlip CLI model."},
    {"name": "uma-s-1p2", "label": "UMA small 1.2 (requires Hugging Face access)", "family": "uma",
     "task_field": "uma_task", "task_choices": ["omat","omc","omol","oc20","oc22","oc25","odac"],
     "task_required": true, "elements": null, "cli_reproducible": true, "note": "..."}
  ],
  "default": "mace",
  "devices": {"choices": ["auto","cpu","cuda"], "default": "auto",
              "torch_installed": true, "cuda_available": false},
  "note": "Restart the server after installing an MLIP package."
}
```

- Built from `mliprun.cli.utils` availability flags (`FAIRCHEM_AVAILABLE`,
  `MACE_AVAILABLE`, `SEVENN_AVAILABLE`, `CHGNET_AVAILABLE`) and its tables
  (`_KNOWN_UMA_MODELS`, `_UMA_TASKS`, `_MACE_MH_HEADS`, `_SEVENNET_MODELS`).
  Only installed families are listed, plus `emt` (always, listed **last**).
  - fairchem: `uma-s-1p2`, `uma-s-1p1`, `uma-m-1p1`; task `uma_task`, required.
  - mace: `mace` (no head), `mace-mh-0`, `mace-mh-1` (`mace_head`, required).
  - sevenn: every key of `_SEVENNET_MODELS`; `sevennet_task` with that tuple;
    required iff non-empty.
  - chgnet: `chgnet`.
- `default` = `detect_mlip()` result if any MLIP installed, else `"emt"`
  (catch its `typer.Exit`).
- `devices.cuda_available`: from `detect_devices()`, which runs
  `[sys.executable, "-c", "import torch, json; print(json.dumps(torch.cuda.is_available()))"]`
  with a 60 s timeout, cached for the process lifetime; `null` until known or
  on failure; `false` with `torch_installed: false` when torch is absent.
  Backend calls it once in a background thread at startup; it MUST NOT import
  torch in the server process.

### 7.2 Calculator validation rules (inside `validate_submission`)

- `name` in the listed calculators (else error on `calculator.name`,
  mentioning the install recipe message from `cli.utils._install_message` when
  it is a known but uninstalled tag).
- The calculator's `task_field`: required and in `task_choices` when
  `task_required`; must be null/absent when the calculator has no task field
  or a single-task SevenNet tag. Other two task keys must be null/absent.
- `device` in `auto`/`cpu`/`cuda` (default `auto`); `cuda` with
  `cuda_available == false` -> error. For `emt` device is recorded only.

### 7.3 `make_calculator(spec: dict)` (worker side)

`emt` -> `ase.calculators.emt.EMT()`. Else `validate_mlip(...)` then
`build_calculator(name, uma_task=..., device=resolved_device, mace_head=...,
sevennet_task=...)` from `mliprun.cli.utils`. Catch `click.exceptions.Exit`
(typer.Exit): message = `exc.exit_code` if it is a `str`, else `"exited with
code N"`; re-raise as `RuntimeError(message)`. In current typer `typer.Exit`
is itself a `RuntimeError` subclass, so the `Exit` handler MUST come before
any generic `RuntimeError` handling.

---

## 8. Worker (`worker.py`, `runners.py`, `events.py`, `reproduce.py`; Bridge)

### 8.1 Process contract (relied on by Backend)

- Invocation: `<python> -m mliprun.webapp.worker <run_dir>`; cwd = run_dir;
  env = `os.environ` copy + `MPLBACKEND=Agg`, `PYTHONUNBUFFERED=1` (Backend
  sets these; the worker also calls `matplotlib.use("Agg")` defensively before
  any core import). stdout/stderr -> `run_dir/worker.log` (Backend opens it in
  append mode). Backend starts it with `start_new_session=True`.
- Exit codes: `0` result written; `1` calculation raised; `2` unreadable
  `job.json` or setup failure before the calculation; `3` cancelled
  (SIGTERM handled). Killed by signal -> negative returncode (Popen).
- SIGTERM handler raises `JobCancelled(Exception)` in the main thread. Being
  an `Exception`, the core marks `mliprun_run.json` as `failed` and
  re-raises; the worker emits `log` (`"cancelled by user"`), writes
  `summary.json` with status `cancelled`, exits 3.
- The worker never writes `job_status` events and never touches
  `state.json`.
- Test hook: if env `MLIPRUN_WEBAPP_TEST_CRASH=after_first_event`, the worker
  calls `os._exit(7)` immediately after writing its first event (no summary).
  Used by robustness tests to simulate a crash.

### 8.2 Steps

1. Read `job.json` (exit 2 with an `error` event on failure if the events file
   is writable).
2. Open `EventWriter(run_dir/"events.jsonl", job_id)`.
3. Install SIGTERM handler; attach a logging handler to logger `mliprun`
   (INFO) that emits `log` events; `logging.captureWarnings(True)` routed the
   same way.
4. `phase setup`; read inputs: `ase.io.read(path, index=-1, format=fmt)`
   (single, initial, final) or `index=":"` (band). For neb/autoneb pair,
   write `inputs/initial.vasp` and `inputs/final.vasp` (`format="vasp"`,
   direct, VASP 5).
5. Write `reproduce.sh` and `reproduce.json` (8.4).
6. `phase load_model`; `make_calculator`.
7. `phase run`; call the runner (8.3) with an ObserverAdapter.
8. `phase postprocess`; build result payload (6.4); emit `result`.
9. Write `summary.json`; exit 0.

`summary.json` (worker): `{"job_id","calc_type","status":"done"|"failed"|"cancelled",
"started_at","finished_at","walltime_s","result":<result payload without
frame/final_frame/band>|null,"error":{"message","exception_type"}|null,
"reproduce":{"status","notes"}}`.

### 8.3 Runners (all write into `output_dir=run_dir`)

Common: `model_name = calculator name`; `device_requested = spec.device`;
`device_resolved = resolve_device(spec.device)` (`"cpu"` for emt); task keys
passed for provenance; `run_context = RunContext(command=<cli command name:
"singlepoint"|"optimize"|"freq"|"md"|"neb"|"autoneb">, mode="one-off",
param_sources=job.param_sources, extra_inputs={"source": "webapp", "job_id":
..., "structure_filename": ...})`.

- singlepoint: `run_singlepoint(atoms, output_dir, prefix="singlepoint",
  stress={"auto": None, "on": True, "off": False}[...], ...)`.
- optimize: `run_optimization(atoms, optimizer, fmax, max_steps,
  trajectory="opt.traj", logfile="opt.log", output_dir, verbose=True,
  relax_cell, plot, ..., observer=adapter)`. Result numbers read from
  `mliprun_run.json` (`results`, `steps`).
- freq: `run_frequencies(atoms, output_dir, prefix="freq", indices=<list or
  None>, delta, nfree, direction, method, write_modes, imaginary_floor,
  expect_fmax, ..., observer=adapter)`. `energy_meV` from
  `freq_frequencies.csv`.
- md: `run_md(atoms, ensemble, thermostat, barostat, barostat_mask,
  temperature, pressure, timestep, friction, ttime, pfactor=<None or
  value*PFACTOR_GPA_FS2>, taut, taup, compressibility, steps, log_interval,
  traj_interval, output_dir, resume=False, plot, ..., observer=adapter)`.
  Hidden params use core defaults.
- neb, pair mode (mirrors `cli/commands/neb.py::_handle_new_neb` + `run`):
  `CustomNEB(initial, final, num_images=n_images, interp_fmax, interp_steps,
  fmax, mlip=name (emt: "emt"), uma_task, mace_head, sevennet_task,
  output_dir, relax_atoms=<list or None>, logfile="neb.log",
  device=resolved)`; for emt set `neb.setup_calculator = lambda *a, **k:
  EMT()` right after construction. If `optimize_endpoints`: `phase
  endpoints`, `optimize_endpoints(endpoint_fmax, endpoint_optimizer,
  endpoint_max_steps)`, `neb.images = neb.setup_neb()`,
  `write_endpoint_results(run_dir/"endpoint_optimization.txt", ...)`. If
  `interpolation == "idpp"`: `phase interpolation`, `interpolate_idpp()`
  (core skips it when relax_atoms is set; the worker logs a warning). Then
  `run_neb(optimizer=NEB_OPTIMIZER_MAP[neb_optimizer], climb,
  max_steps=neb_max_steps, k, dyneb, scale_fmax, plot, run_context,
  observer=adapter)`; then `process_results()`, `plot_results(df)` if plot,
  `export_poscars()`.
- neb/autoneb `phase load_model`: only `validate_mlip` runs here (no
  calculator is built in the worker), because `CustomNEB.setup_calculator`
  builds one per image itself; emt skips validation.
- neb pair with `interpolation == "linear"`: the band is the linear one from
  `setup_neb()`, which drops `FixAtoms` from intermediate images (normally
  restored by `interpolate_idpp`); the worker re-applies the endpoint
  `FixAtoms` to every intermediate image so fixed atoms stay fixed.
- neb, band mode: `CustomNEB(frames[0], frames[-1],
  num_images=len(frames)-2, fmax, ..., relax_atoms=None)`, then
  `neb.images = [neb.initial, *frames[1:-1], neb.final]`; no endpoint
  optimization, no interpolation; then as above from `run_neb`.
- autoneb (mirrors `cli/commands/autoneb.py`): `CustomNEB(..., num_images=5,
  device=resolved)`, emt override as above, optional `optimize_endpoints` +
  `write_endpoint_results`, then `run_autoneb(n_simul=1, n_max, k, climb,
  optimizer=ObservedFIRE, space_energy_ratio, interpolate_method, maxsteps,
  prefix="autoneb", run_context)` where `ObservedFIRE` is a FIRE subclass
  made by a factory in runners.py: its `__init__(self, atoms, *a, **kw)`
  stores `atoms` (the sub-NEB object), calls `super().__init__`, then
  `self.attach(callback, interval=1)`; the callback reads
  `target.images` energies (try/except -> null), `calc_fmax(target.get_forces())`,
  positions, and emits `neb_iter` with a global iteration counter and
  `autoneb: {"sub_step": self.nsteps}`. `run_autoneb` `chdir`s inside the
  worker process only (harmless there). Result energies are read from the
  last frame of each `autoneb???.traj` with
  `get_potential_energy(allow_calculation=False)` (stored results, never a
  recomputation); the worker also writes `autoneb_energy_profile.csv` into
  run_dir.

ObserverAdapter maps core observer payloads (section 9) to events: optimize
-> `opt_step`; md -> `md_step` (keys from CSV column names: `time(fs)` ->
`time_fs`, `potential_energy(eV)` -> `epot`, `kinetic_energy(eV)` -> `ekin`,
`total_energy(eV)` -> `etot`, `temperature(K)` -> `temperature`,
`pressure(GPa)` -> `pressure_GPa`, `volume(A^3)` -> `volume_A3`); freq ->
`freq_progress`; neb -> `neb_iter` (`rel_energies`, `highest_image` =
argmax, `n_images` computed). It also checks a cancellation flag (no-op;
SIGTERM is the mechanism).

### 8.4 `reproduce.sh` / `reproduce.json`

`reproduce.sh` (mode 0755):
```bash
#!/usr/bin/env bash
# mliprun webapp job <job_id> (<calc_type>), created <created_at>
# Reproducible: exact | approximate | no
# Note: <one line per note>
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p reproduce
cp inputs/<role>/<file> reproduce/          # neb/autoneb: inputs/initial.vasp inputs/final.vasp
cd reproduce
mlip <cmd> run --structure <file> --mlip <tag> [--uma-task ...] --device <d> --<flag> <value> ...
```
- Every applicable param is passed explicitly (defaults too), with CLI flag
  spellings taken from the typer options of `cli/commands/<cmd>.py`
  (`cli_flag` in the schema). Bools use the CLI's `--x/--no-x` or flag form
  as declared there. `relax_atoms` is expanded to an explicit comma list.
  md: `--pfactor` only when not null; `--compressibility` only with npt +
  berendsen; `--barostat-mask` only with npt. Outputs go into `reproduce/` so
  the webapp results are not overwritten.
- Status/notes rules:
  - `no`: calculator `emt` (no CLI equivalent). The `mlip` line is written
    commented out (`# mlip ...`) with `--mlip REPLACE_WITH_AN_MLIP`; the
    script then `echo`es the note and exits 1.
  - `approximate` (notes accumulate): neb band input (CLI cannot start from a
    band); neb `interpolation=linear` without relax_atoms (CLI always runs
    IDPP); autoneb with a device other than cpu (CLI autoneb has no
    `--device`); md NVT/NPT (random initial velocities, no seed); input
    format not VASP for neb/autoneb (converted to VASP).
  - `exact`: otherwise.
- `reproduce.json`: `{"status": ..., "notes": [...], "argv": [<the mlip argv list>]}`.

### 8.5 `events.py` API (used by Backend and worker)

```python
class EventWriter:
    def __init__(self, path: Path, job_id: str):
        """Open for append. If the file ends with a partial line (no trailing
        newline), truncate it. Next seq = number of complete lines."""
    def emit(self, type: str, **payload) -> dict:
        """Write one JSON line {"job_id","seq","t","type",**payload} with
        to_jsonable, '\n', flush. Return the dict. seq increments by 1."""
    def close(self) -> None: ...

def read_events(path: Path, since: int = 0, limit: Optional[int] = None) -> list[dict]:
    """Complete lines only; skip lines that fail to parse (they still occupy a seq)."""
def count_complete_lines(path: Path) -> int: ...
def encode_frame(positions, cell) -> dict:   # section 6.2, 4-decimal rounding
def to_jsonable(obj): ...                    # numpy -> python, NaN/inf -> None, recursive
```

---

## 9. Core hooks (Bridge; backward compatible, default None = identical)

Common contract: new last keyword `observer: Optional[Callable[[dict], None]]
= None`. Called synchronously in the engine thread only when not None. Must
not change numerics: it reads values the existing callback already computed,
plus `get_positions()` / `get_cell()` copies. Exceptions raised by the
observer propagate (the engine's existing `except` marks the record `failed`
and re-raises). Docstrings updated (NumPy style).

| File | Function | Where called | Payload |
|---|---|---|---|
| `src/mliprun/core/optimize.py` | `run_optimization(..., observer=None)` | end of `log_convergence()` | `{"step": int, "energy": float, "fmax": float, "positions": ndarray (N,3), "cell": ndarray (3,3)}` (positions/cell of `atoms`, not the filter) |
| `src/mliprun/core/md.py` | `run_md(..., observer=None)` | end of `log_properties()` | `{"row": dict(row), "positions": ndarray, "cell": ndarray}`; `row` keys are exactly the CSV column names |
| `src/mliprun/core/vibrations.py` | `run_frequencies(..., observer=None)` | `CountingVibrations.calculate` after `n_force_calls += 1` | `{"done": int, "total": int}`. Implementation: class attributes `observer = None`, `expected_force_calls = None` on `CountingVibrations`; after constructing `vib` (line ~1042) set `vib.observer = observer` and `vib.expected_force_calls = _expected_force_calls(len(chosen), nfree)`. `CommitteeVibrations` inherits it. |
| `src/mliprun/core/neb.py` | `CustomNEB.run_neb(..., observer=None)` | end of `log_iteration()` | `{"step": int, "fmax": float, "barrier": float, "energies": [float], "positions": [ndarray per image], "cell": ndarray}` |

AutoNEB: no core change (optimizer subclass, section 8.3). NEB endpoints and
IDPP: no hook (D12).

Hook tests (Bridge, `tests/webapp/test_core_hooks.py`, EMT, no fastapi
needed): (a) observer call count equals the CSV row count
(`opt_convergence.csv`, `md_energy.csv`, `neb_convergence.csv`) and values
match the CSV to 1e-10; freq last `done` == `results["n_force_calls"]` ==
`total` on an empty cache; (b) an observer that raises at call 3 makes the
function raise that exception and leaves `mliprun_run.json` with status
`failed`; (c) existing `tests/test_characterization_golden.py` passes
unchanged.

---

## 10. Job lifecycle (Backend)

### 10.1 State machine

```
            submit                 runner picks (no job running)
  (none) ----------> pending ---------------------------------> running
                        |                                         |
                        | cancel (user)                           | worker exits
                        v                                         v
                    cancelled          cancel_requested ? -> cancelled
                                       rc == 0 and summary.status == "done" -> done
                                       otherwise -> failed
```

| From | To | Trigger | Who |
|---|---|---|---|
| (none) | pending | POST /api/jobs | API handler |
| pending | running | head of FIFO and nothing running; Backend writes `job_status running` (with no worker yet), then spawns | QueueRunner |
| pending | cancelled | POST cancel | API handler (immediate) |
| running | cancelled | POST cancel sets `cancel_requested`, sends SIGTERM; SIGKILL after `cancel_grace_s`; transition when the process has exited (any rc) | API handler + QueueRunner |
| running | done | process exit, rc 0, `summary.json` status `done` | QueueRunner |
| running | failed | rc != 0 (not cancelled), or rc 0 without done summary, or spawn error (`reason` = exception text) | QueueRunner |
| running | failed | server start finds `running` in state.json; `reason = "server restarted while the job was running"` | startup recovery |
| pending | pending (re-queued) | server start (D6) | startup recovery |

Terminal states: `done`, `failed`, `cancelled`; no transitions out. `done`
means the calculation finished, including a non-converged optimization or
NEB (see `result.converged`).

Rules:
- Every transition: update memory, write `state.json` atomically, append the
  `job_status` event (only when no worker process is alive for the job),
  broadcast. Final status line is appended **after** the worker process is
  reaped and the tailer has drained the file.
- Failed `reason` text: `"worker exited with code <rc>"` / `"worker killed by
  signal <n>"` / the `error` event message when present.
- If the worker died without `summary.json`, Backend writes one with status
  `failed` (or `cancelled`) and `error.message = reason`. A job cancelled while
  `pending` never had a worker and gets **no** `summary.json` (its state is in
  `state.json` and `events.jsonl`).
- Startup recovery: for each `<runs-dir>/<job_id>/state.json`; if `running`
  and `pid` alive and `/proc/<pid>/cmdline` contains `mliprun.webapp.worker`
  and the run_dir: SIGTERM, wait grace, SIGKILL. Truncate a partial final
  line of `events.jsonl` (EventWriter does this) before appending. Dirs
  without a readable `state.json` are ignored (logged).
- Tailer: for the running job only; offset in bytes; parse complete lines;
  update `progress`, `result`, last error; broadcast each event.

---

## 11. Frontend (static/)

Single page served at `/`. CDN scripts (exact URLs, `<script>` tags in
`index.html`):
- `https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js`
- `https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js`

If either global (`$3Dmol`, `Plotly`) is missing after load, show a banner
(`data-testid="lib-error-banner"`) saying viewers/plots need internet access;
the rest of the UI MUST still work.

### 11.1 Behaviour

- **Structures panel**: drop zone + file input (multiple); optional format
  select (auto + common ASE formats: vasp, extxyz, cif, traj, vasp-out);
  results list with formula, natoms, n_frames, pbc, error text; click =
  preview in a 3Dmol viewer with the unit cell drawn as 12 lines from `cell`.
  Multi-frame uploads show a frame slider (from `frames_xyz`).
- **New calculation form**: calc type select; calculator select (from
  `/api/calculators`, preselect `default`); task select shown only when the
  calculator has `task_field` (choices from `task_choices`); device select;
  fields generated from FieldSpec (bool -> checkbox, choice -> select,
  float/int -> number input with min/max/step, indices/mask3 -> text);
  `advanced` fields in a collapsed section; `visible_if` / `input_modes`
  re-evaluated on every change; hidden fields are **not sent**. Unit shown
  after inputs; `help` as tooltip/caption. `structure_every` numeric input.
  Inputs: single -> checkbox per OK upload (submit one job per checked
  structure); pair/band -> radio for mode and selects for initial/final or
  band (band select lists uploads with n_frames >= 3). Client-side checks are
  convenience only; 422 `details` are shown next to the matching field
  (`params.<name>` -> that field; others in the submit error box).
- **Job list**: live (WS `*`), newest first, status badge, calc type, label,
  formula, progress text, cancel button for pending/running.
- **Job detail** (`#/jobs/<id>`): header (status, calculator, params);
  live 3D viewer updated from `frame`/`band`; plots:
  - optimize: energy vs step and fmax vs step (log y), fmax target line.
  - md: etot/epot vs time_fs with ekin in a stacked subplot below (shared x;
    two scales are never put on twin y-axes of one plot); temperature vs
    time_fs with the target line; NPT adds a third plot (`plot-extra`) of
    pressure and volume, also stacked.
  - neb/autoneb: current band rel_energies vs image index (updated per
    iteration); fmax vs iteration (log y) with the barrier in a stacked
    subplot. Viewer can play the band.
  - freq: progress bar (done/total); on result, stick plot of
    `frequency_cm1` per mode with imaginary modes in a distinct colour, and a
    table (index, cm⁻¹, meV, imaginary).
  - singlepoint: energy, fmax, stress; forces table; atoms coloured by
    `f_norm` in the viewer.
  - result panel with all result scalars; log panel (log/phase/error events);
    files list with download links; zip link; reproduce status + notes + a
    link to `reproduce.sh` inline view.
- Reconnect WS with backoff (1, 2, 5, 10 s) and resubscribe with `since`.
- The job list also re-fetches `GET /api/jobs` every 2.5 s while any listed
  job is `pending` or `running` (the `*` subscription carries no progress
  events, so this keeps progress text and queue positions current).
- Uploading files ticks them in the structure list (selected for the next
  single-structure submission).
- Routing to `#/jobs/<id>` scrolls the detail panel into view; panels carry
  `scroll-margin-top` so the sticky top bar does not cover the job title.
- No mock/offline-demo mode ships; the page always talks to the real server.

### 11.2 Stable selectors (Tester relies on these)

`data-testid` values: `upload-input`, `upload-dropzone`, `upload-format`,
`structure-list`, `structure-item` (attr `data-upload-id`, `data-ok`),
`structure-select` (checkbox inside a structure-item), `structure-error`,
`calc-type-select`, `calculator-select`, `task-select`, `device-select`,
`structure-every`, `field-<name>` (wrapper; contains exactly one
input/select), `advanced-toggle`, `neb-mode-pair`, `neb-mode-band`,
`neb-initial-select`, `neb-final-select`, `neb-band-select`, `submit-button`,
`submit-error`, `field-error` (inside a `field-<name>` wrapper),
`job-list`, `job-row` (attrs `data-job-id`, `data-status`),
`job-status`, `job-detail` (attr `data-job-id`, `data-status`),
`cancel-button`, `viewer`, `plot-main`, `plot-secondary`, `result-panel`,
`forces-table`, `freq-table`, `log-panel`, `files-list`, `zip-link`,
`reproduce-panel`, `lib-error-banner`, plus `plot-extra` (NPT pressure and
volume) and `structure-preview` (the upload preview viewer).

`cancel-button` and `job-status` appear both in job rows and in the job
detail header: tests MUST scope them, e.g.
`[data-testid=job-detail] [data-testid=cancel-button]`, or inside one
`job-row`.

`data-status` always equals the job's current status string.

---

## 12. Phases and acceptance criteria

### Phase 2A, Backend
- [ ] Files of section 1.2 (Backend rows); `pip install -e ".[webapp]"`
      installs fastapi, uvicorn[standard], python-multipart; `static/**`
      ships as package data (`pip install .` into a fresh venv serves `/`).
- [ ] `mlip --help` works without the extra; `mlip webapp --help` lists the
      four options; without fastapi, `mlip webapp` exits 1 with the hint.
- [ ] All endpoints of section 4 with the exact JSON shapes; error format of
      section 5 for every error including FastAPI's 422.
- [ ] D5: POST without `X-Mliprun` -> 403; WS with foreign Origin closed
      1008; Host check when `allowed_hosts` given.
- [ ] Queue: never two workers alive; FIFO; state machine of 10.1; restart
      recovery; cancel pending and running; SIGKILL after grace.
- [ ] Tests (with `fake_worker.py`, no MLIP): uploads (POSCAR without
      extension, extxyz multi-frame, garbage file), submit/validation 422
      passthrough, all-or-nothing multi-submit, events replay `since`, WS
      replay-then-live with no gap/duplicate, files traversal 400, zip
      contents, serial execution, cancel, restart recovery. Existing suite
      still green.
- `conftest.py` fixtures: `runs_dir` (tmp), `make_app(**kw)`, `client`
  (TestClient as context manager, `allowed_hosts=None`), `upload_atoms(client,
  atoms, filename, fmt=None) -> UploadInfo` (writes the file with ase.io.write
  and POSTs it), `wait_for_status(client, job_id, statuses, timeout=60) ->
  JobDetail`, header constant `H = {"X-Mliprun": "1"}`. fastapi imported
  inside fixtures.

### Phase 2B, Bridge
- [ ] Core hooks of section 9 + their tests; `pytest tests/test_characterization_golden.py`
      unchanged and green; full non-MLIP suite green.
- [ ] `schema.py` data exactly as section 3; `test_schema.py`: every
      calc type has fields; every `visible_if` references an earlier field;
      defaults pass their own validation; each rule of 3.3 has a failing case
      (hidden field present, out of range, wrong type, bad indices, NEB atom
      mismatch, EMT unsupported element, missing uma_task, task on
      single-task model).
- [ ] `calculators.py` with emt always present; `detect_devices` never
      imports torch in-process.
- [ ] `events.py` with partial-line truncation test.
- [ ] Worker: for each calc type, a test writes a `job.json` (EMT, Cu) and
      runs `python -m mliprun.webapp.worker <dir>` as a subprocess; asserts
      exit 0, one `result` event, seq contiguous from the starting count,
      `summary.json` status done, `reproduce.sh` present; NEB in both pair
      and band mode; SIGTERM during md -> exit 3, record `failed`, summary
      `cancelled`.
- [ ] `test_reproduce.py`: for every calc type, the generated argv parses
      with `typer.main.get_command(mliprun.cli.main.app).make_context("mlip",
      argv)` without error (MLIP tag e.g. `mace`, not run); emt -> status `no`.

### Phase 2C, Frontend
- [ ] No build step; only the two CDN scripts; ES modules from
      `/static/js/`.
- [ ] Form generated purely from `/api/calc-types` + `/api/calculators` (no
      field names hard-coded except `structure_every` and the calculator
      block); hidden fields not sent.
- [ ] All `data-testid` of 11.2 present at the documented places.
- [ ] Manual run with EMT: upload Cu POSCAR, run each calc type, see live
      plots and structure updates, cancel an md job, download zip.
- [ ] Works at 1280x800 and is usable at 390 px width (stacked panels).

### Phase 2D, Integration (orchestrator)
- [x] With real Backend + Bridge + Frontend: `mlip webapp --no-browser`,
      run all six calc types with EMT through the browser; fix contract
      mismatches by amending this file first.

### Phase 3, Tests (Tester; tests/webapp/)
- Tier 1, `test_tier1_emt.py` (no MLIP; real worker via `create_app` default
  `worker_cmd`), structures built with `ase.build`, elements Cu, Pt, Au, Ni
  (parametrize ids `cu`, `pt`, `au`, `ni`):
  - singlepoint on rattled fcc bulk 2x2x2 (`rattle(0.05, seed=1)`):
    `result.energy_eV` equals a direct `EMT()` energy to 1e-8; forces table
    has natoms rows and matches direct forces to 1e-8.
  - optimize same structure, fmax 0.05: `done`, `converged` true,
    `final_fmax_eV_per_A <= 0.05`, energy equal to a direct
    `run_optimization` with the same params to 1e-6; number of `opt_step`
    events = rows in `opt_convergence.csv`; with `structure_every=5`, frames
    present exactly on events 0, 5, 10, ...
  - freq on a relaxed small cell (e.g. 4-atom fcc conventional): `n_modes =
    3N`, `freq_progress` last `done == total == 1 + 6N`, frequencies equal a
    direct `run_frequencies` to 1e-6 cm⁻¹.
  - md NVE 50 steps, timestep 1 fs, log_interval 5 on rattled bulk:
    `md_step` count = rows in `md_energy.csv`; |etot drift| < 1e-2 eV.
    NVT Langevin 50 steps: completes, mean temperature finite.
  - neb pair: fcc(100) 3x3x3 slab of the metal with an adatom moved between
    two neighbouring hollow sites (bottom layer fixed), n_images 3, idpp,
    fmax 0.1: `done`; `forward_barrier_eV > 0`; `ts_at_endpoint` false;
    barrier equal to a direct CustomNEB run with the same settings to 1e-6;
    band mode with the resulting `A2B.traj` uploaded: `done`.
  - autoneb (Cu only, n_max 5, fmax 0.1): `done`, barrier > 0.
  - reproduce: `reproduce.json` status `no` for emt everywhere.
- `test_robustness.py`: 3 jobs submitted at once run strictly in series
  (`started_at[i+1] >= finished_at[i]`); cancel pending -> `cancelled`
  without a worker; cancel running md (10^6 steps) -> `cancelled` within
  `cancel_grace_s + 5` s and record `failed`; `MLIPRUN_WEBAPP_TEST_CRASH`
  -> `failed`, server still serves `/api/health`, next job runs; restart:
  new app on a runs_dir with a `running` state.json (dead pid) -> `failed`
  with the restart reason, `pending` -> runs; validation 422 shapes; path
  traversal; `X-Mliprun` enforcement; WS replay since=0 equals GET events.
- Tier 2, `test_tier2_mace.py`, marked `@pytest.mark.mace` (auto-skipped
  without mace-torch; parametrize ids must not be `mace`): calculator `mace`,
  device `cpu`; singlepoint on Cu bulk equals direct
  `build_calculator("mace", device="cpu")` energy to 1e-5; optimize
  `max_steps=3` reaches `done`; reproduce status `exact` and argv parses.
- Browser, `test_browser.py`: skip unless `playwright.sync_api` importable and
  a Chromium launches (try bundled, then `executable_path="/usr/bin/chromium"`)
  and the CDN is reachable; uvicorn in a thread on a free port; upload a Cu
  POSCAR via `upload-input`, choose optimize + emt, submit, wait for
  `job-detail[data-status=done]`, assert `plot-main` contains a Plotly graph
  and `result-panel` shows the converged energy; md job cancel via
  `cancel-button` -> `data-status=cancelled`.
- Acceptance: `pytest tests/webapp -m "not mace"` green with zero
  unexpected skips except browser/tier2; full repo suite green; webapp
  Python diff coverage >= 90 %.

### Phase 4, Docs
- [ ] `docs/WEBAPP.md`: install (`pip install -e ".[webapp]"`), launch,
      runs-dir layout, calc types and what is not offered (D1), EMT caveat,
      reproduce.sh meaning, security note (localhost, no auth), offline
      limitation (CDN), restart-after-install note.
- [ ] README: short "Web app" section linking it. AGENTS.md: one line on
      `tests/webapp` and the `webapp` extra.

---

## 13. Risks (from RECON) and handling

| # | Risk | Handling |
|---|---|---|
| 1 | Process-global side effects: `os.chdir` in `run_autoneb`, pyplot global state, `MDLogger` on stdout, `typer.echo` | One subprocess per job, cwd = run_dir, `MPLBACKEND=Agg`, stdout -> worker.log; strictly one job at a time. |
| 2 | `typer.Exit` raised from library code with the message as `exit_code` | `make_calculator` and runners catch `click.exceptions.Exit`/`click.UsageError` and convert to `RuntimeError(message)`; server-side validation catches most cases before a job starts. |
| 3 | MLIP availability frozen at import | `/api/calculators` `note`; docs; server restart needed. |
| 4 | NEB restart 15 s sleep; `or`-defaults make 0 unenterable; class vs CLI defaults differ | Restart not offered; webapp calls core directly with explicit values (0 allowed where the schema allows it, e.g. `scale_fmax`); schema defaults = CLI defaults. |
| 5 | `CustomNEB.__init__` mutates `final` (cell rescale) | Worker reads fresh Atoms from files; nothing shared. |
| 6 | AutoNEB deletes `{prefix}*.traj` and `AutoNEB_iter/` in output_dir | Own run_dir per job; prefix `autoneb` never matches webapp files. |
| 7 | One `mliprun_run.json` per directory | One run_dir per job. |
| 8 | `opt.traj` opened twice; `*_convergence.csv` only at end | Live data comes from the observer, not file tailing. |
| 9 | Python 3.14 only locally; ASE version guards | Webapp code is 3.9+-compatible; CI (3.11) runs webapp tests via D10. |
| 10 | Benchmark lacks mace_head/device | Benchmark not offered. |
| 11 | No cancellation API | SIGTERM -> `JobCancelled` (record marked failed), SIGKILL fallback. |
| 12 | Docs drift on head/task defaults | Schema requires tasks/heads explicitly; no defaults. |
| - | Device "auto" passed raw by `CustomNEB.setup_calculator` | D9: resolve before passing. |
| - | NEB CLI reads only VASP | D8: convert for reproduce.sh. |
| - | Random MD velocities (no seed) | reproduce `approximate`; tests use invariants for NVT. |
| - | Local web server reachable by any web page in the browser | D5 (custom header, Origin and Host checks), 127.0.0.1 bind. |
| - | Large events (many atoms x many steps) | `structure_every`, 4-decimal frames, `*` subscription carries no frames, events endpoint paging. |
| - | CDN unavailable offline | Banner; rest of the UI works; documented. |
| - | Orphan worker after server crash | `start_new_session=True` + startup recovery kills a matching live pid. |
| - | CI diff-coverage gate (90 %) on core hook lines and webapp code | Hook tests with EMT; webapp tests run in CI (D10); worker subprocess coverage already handled by the repo's `COVERAGE_PROCESS_START` setup because the worker inherits the environment. |

---

## 14. Changes during implementation

Recorded by the implementers (PROGRESS.md) and at integration; the sections
above are already edited to match.

- Frontend: stacked subplots instead of twin y-axes; NPT plot `plot-extra`;
  preview viewer `structure-preview` (11.1, 11.2); only non-default visible
  params are sent (4.4); duplicate `cancel-button`/`job-status` testids must
  be scoped (11.2); job list polls every 2.5 s while jobs are active (11.1).
- Frontend: the development mock (`static/js/mock.js`, `?mock=1`) was
  removed at integration.
- Backend: `ws_ack` is sent after the replay (6.5); a job cancelled while
  pending gets no `summary.json` (10.1); rc 0 without a `done` summary is
  `failed` (10.1).
- Bridge: `typer.Exit` is a `RuntimeError` subclass and is caught first
  (7.3); md field order puts `barostat` right after `thermostat` (3.2);
  neb/autoneb `load_model` only validates (8.3); linear interpolation
  re-applies `FixAtoms` (8.3); AutoNEB energies via
  `allow_calculation=False` plus `autoneb_energy_profile.csv` (8.3).
- Integration fixes: the job title was hidden under the sticky top bar after
  routing (added `scroll-margin-top`; the "header mid-page" in full-page
  screenshots is only a screenshot artifact of `position: sticky`); the
  singlepoint result table no longer shows an empty "Stress unavailable
  reason" row; `tests/webapp/conftest.py::cu_upload` writes into `tmp_path`
  instead of a leaked `mkdtemp` directory.
- Outside the webapp: `tests/test_neb_mace.py` wrote its NEB outputs
  (`mliprun_run.json`, `neb.log`, `A2B*.traj`, `idpp.*`,
  `neb_convergence.csv`) into the current directory, i.e. the repo root; it
  now uses `output_dir=tmp_path` and absolute fixture paths.
