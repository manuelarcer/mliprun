# TS Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify a transition state (TS) by vibrational analysis: exactly one imaginary mode, plus an overlap check and/or a displace-and-relax check that the mode connects the initial state (IS) and final state (FS).

**Architecture:** One method-independent core function, `verify_ts`, in a new module `src/mliprun/core/ts_verify.py`. It reuses the existing `run_frequencies`. Two thin callers: a new command `mlip ts verify` and a new option `neb run --verify-ts`.

**Tech Stack:** Python, ASE (`ase.vibrations`, `ase.optimize`, `ase.geometry.find_mic`), Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-ts-verification-design.md`. Read it before any task.

## Global Constraints

- Branch `feat/ts-verify`. Confirm with `git branch --show-current` before every commit. Never commit to `main`. Draft PR only.
- Tests first: each task writes the failing test before the code.
- Unit tests run with no MLIP installed: `.venv/bin/pytest -m "not uma and not mace and not sevenn" --ignore=tests/webapp`. Baseline before this work: 936 passed, 6 skipped.
- Diff coverage against `origin/main` at least 90 % (command in `AGENTS.md`).
- Never regenerate or edit anything under `tests/goldens/`.
- No pytest parametrize id or test name containing `uma`, `mace` or `sevenn` (such a test is silently skipped).
- `freq run` and `neb run` without the new option must produce exactly what they produce today.
- AQCat25-EV2 tags are refused (`reject_aqcat_unsupported(mlip, "ts verify")`).
- Defaults, verbatim from the spec: imaginary floor 10 cm⁻¹ (`IMAGINARY_FLOOR_CM1`), TS fmax limit 0.03 eV/Å, overlap threshold 0.7, displacement 0.1 Å, match tolerance 0.3 Å, relax optimizer `bfgs`, relax fmax 0.01 eV/Å, relax max steps 200, default check `overlap`.
- Verdict strings: `verified`, `not_verified`, `inconclusive`. Relax results: `connects`, `one_side`, `same_side`, `neither`, `inconclusive`. Check statuses: `pass`, `fail`, `inconclusive`, `not_run`. Reference labels: `neb_tangent`, `is_to_fs`.
- No file is written by shell heredoc or `sed -i`. No stray report files in the tree.
- Commit messages end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## Decisions made while planning (not in the spec)

- **The NEB directory's own run record is not modified.** `RunRecord.complete` rewrites the top-level `status`, so an appended stage would replace the NEB's `converged`. The verification record lives at the fixed location `<neb-dir>/ts_verify/mliprun_run.json`; `docs/OUTPUTS.md` documents that location as the pointer.
- **The record in `ts_verify/`** has top-level `command: "ts-verify"` and two stages: `freq` (written by `run_frequencies`) then `ts-verify` (appended). This needs one new keyword on `run_frequencies`: `record_command: str = "freq"`.
- **`CustomNEB.run_neb` stores `self.converged`** (bool) so the CLI can decide whether to verify.

## Review Focus

1. IS, TS and FS that differ in atom count, atom order or cell: an error before any force call, naming the mismatch.
2. A zero reference direction (IS identical to FS, or a mode with no component on the displaced atoms): a clear error, never a NaN overlap.
3. Highest image of the band is image 0 or the last image: non-zero exit with a clear message.
4. Rerun into a `ts_verify/` that holds a displacement cache from other settings: the existing cache refusal message is printed, exit code 1, no traceback.
5. An exception inside the verification after `neb run`: the NEB outputs and NEB record are unchanged and the command still exits 0, with the failure printed.

---

## File structure

| File | Responsibility |
|---|---|
| `src/mliprun/core/ts_verify.py` (new) | Pure helpers, `verify_ts`, `load_neb_for_verification`, report writer |
| `src/mliprun/core/vibrations.py` (modify) | `record_command` keyword on `run_frequencies` |
| `src/mliprun/core/neb.py` (modify) | `self.converged`; parse `Verify TS` in `_parse_parameters_file` |
| `src/mliprun/cli/commands/ts.py` (new) | Typer group `ts`, command `verify` |
| `src/mliprun/cli/main.py` (modify) | register `ts` |
| `src/mliprun/cli/commands/neb.py` (modify) | `--verify-ts` |
| `tests/ts_helpers.py` (new) | `DoubleWell` analytic calculator |
| `tests/test_ts_verify_helpers.py`, `tests/test_core_ts_verify.py`, `tests/test_cli_ts.py`, `tests/test_neb_verify_ts.py` (new) | tests |
| `docs/TS_VERIFICATION.md` (new) and README, AGENTS, CONTEXT, OUTPUTS, PYTHON_API, CHANGELOG | documentation |

---

### Task 1: Pure helpers

**Files:**
- Create: `src/mliprun/core/ts_verify.py`
- Test: `tests/test_ts_verify_helpers.py`

**Interfaces — Produces:**

```python
def neb_tangent(images: list, index: int) -> np.ndarray            # (N, 3), not normalised
def is_to_fs_direction(initial, final) -> np.ndarray               # (N, 3), minimum image
def mode_overlap(mode, reference, indices) -> float                # |cos| in [0, 1]
def scale_mode(mode, displacement: float) -> np.ndarray            # largest atom norm == displacement
def endpoint_distance(atoms, endpoint) -> float                    # largest single-atom MIC distance, Å
def classify_connection(plus: dict, minus: dict) -> str            # relax result string
def decide_verdict(n_imaginary, ts_fmax, ts_fmax_limit, check_statuses: dict) -> tuple[str, str]
def check_same_system(ts_atoms, initial, final) -> None            # raises ValueError
```

- `neb_tangent`: the improved tangent of Henkelman and Jónsson (J. Chem. Phys. 113, 9978 (2000)) at image `index`, from the minimum-image displacements to the two neighbours and the three energies (`get_potential_energy()` of the stored images). At an interior maximum it is the energy-weighted sum of the two segment vectors. Raises `ValueError` when `index` is 0 or the last image.
- `mode_overlap`: restrict both (N, 3) arrays to `indices`, normalise each, return the absolute dot product. Raises `ValueError` when either restricted vector has norm below 1e-12.
- `classify_connection`: `plus` and `minus` each carry `converged: bool` and `matched: "initial" | "final" | None`. Any `converged` False gives `inconclusive`; one `initial` and one `final` gives `connects`; both the same endpoint gives `same_side`; exactly one matched gives `one_side`; none gives `neither`.
- `decide_verdict` returns `(verdict, reason)`, following the spec sections "A loosely converged TS (D9)" and "Verdict". `check_statuses` maps `"overlap"` / `"relax"` to a status string, only for requested checks.
- `check_same_system`: same length, same chemical symbols in order, cells equal within 1e-6 Å; the message names which of the three differs and how.

- [ ] **Step 1: Write the failing tests**

```python
def test_overlap_of_parallel_and_antiparallel_vectors_is_one():
    v = np.array([[1.0, 0, 0], [0, 2.0, 0]])
    assert mode_overlap(v, -3 * v, [0, 1]) == pytest.approx(1.0)

def test_overlap_of_orthogonal_vectors_is_zero():
    assert mode_overlap(np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]), [0]) == pytest.approx(0.0)

def test_overlap_ignores_atoms_outside_the_indices():
    mode = np.array([[1.0, 0, 0], [0, 5.0, 0]]); ref = np.array([[1.0, 0, 0], [0, 0, 9.0]])
    assert mode_overlap(mode, ref, [0]) == pytest.approx(1.0)

def test_a_zero_reference_is_an_error_not_nan():
    with pytest.raises(ValueError, match="zero"):
        mode_overlap(np.ones((1, 3)), np.zeros((1, 3)), [0])

def test_scale_mode_sets_the_largest_atom_displacement():
    scaled = scale_mode(np.array([[3.0, 4.0, 0], [1.0, 0, 0]]), 0.1)
    assert np.linalg.norm(scaled, axis=1).max() == pytest.approx(0.1)
    assert np.linalg.norm(scaled[1]) == pytest.approx(0.02)

def test_endpoint_distance_uses_the_minimum_image():
    # one atom at x=0.1 and one at x=9.9 in a 10 A periodic cell are 0.2 A apart
    ...assert endpoint_distance(a, b) == pytest.approx(0.2)

def test_tangent_at_a_symmetric_maximum_is_the_sum_of_both_segments():
    # three single-atom images at x = 0, 1, 2 with energies 0, 1, 0
    ...assert tangent / np.linalg.norm(tangent) == pytest.approx([[1.0, 0, 0]])

def test_tangent_at_an_endpoint_is_an_error(): ...

@pytest.mark.parametrize("plus, minus, expected", [
    (("initial", True), ("final", True), "connects"),
    (("final", True), ("initial", True), "connects"),
    (("initial", True), ("initial", True), "same_side"),
    (("initial", True), (None, True), "one_side"),
    ((None, True), (None, True), "neither"),
    (("initial", True), ("final", False), "inconclusive"),
])
def test_classify_connection(plus, minus, expected): ...

@pytest.mark.parametrize("n_imag, fmax, statuses, expected", [
    (1, 0.01, {"overlap": "pass"}, "verified"),
    (1, 0.10, {"overlap": "pass"}, "verified"),            # one mode: fmax does not block
    (1, 0.01, {"overlap": "pass", "relax": "fail"}, "not_verified"),
    (1, 0.01, {"overlap": "pass", "relax": "inconclusive"}, "inconclusive"),
    (0, 0.01, {"overlap": "not_run"}, "not_verified"),
    (2, 0.01, {"overlap": "not_run"}, "not_verified"),
    (2, 0.031, {"overlap": "not_run"}, "inconclusive"),    # D9, limit 0.03
    (0, 0.031, {"overlap": "not_run"}, "inconclusive"),
])
def test_decide_verdict(n_imag, fmax, statuses, expected): ...

def test_the_d9_reason_names_convergence():
    assert "not converged enough" in decide_verdict(2, 0.05, 0.03, {"overlap": "not_run"})[1]

def test_check_same_system_names_the_mismatch():   # Review Focus 1
    # different atom count, different symbol order, different cell: three cases
    ...
```

- [ ] **Step 2: Run and confirm failure** — `.venv/bin/pytest tests/test_ts_verify_helpers.py -q`, expected: import error.
- [ ] **Step 3: Implement the eight functions** in `src/mliprun/core/ts_verify.py`.
- [ ] **Step 4: Run** the same command, expected: all pass.
- [ ] **Step 5: Commit** — `feat(ts): geometry and verdict helpers for TS verification`.

---

### Task 2: `verify_ts` with frequencies, overlap, report and record

**Files:**
- Modify: `src/mliprun/core/ts_verify.py`, `src/mliprun/core/vibrations.py` (`run_frequencies` signature and its `RunRecord.begin` call)
- Create: `tests/ts_helpers.py`, `tests/test_core_ts_verify.py`

**Interfaces — Consumes:** Task 1 helpers; `run_frequencies` (returns `n_imaginary`, `frequencies_cm-1`, `imaginary_mask`, `fmax_at_input_free_eV_per_A`); `select_indices`; `RunRecord.begin(..., append=True)`; `collect_provenance`.

**Interfaces — Produces:**

```python
def verify_ts(ts_atoms, initial, final, *, calculator_factory,
              checks=("overlap",), reference_direction=None, reference_label=None,
              output_dir="ts_verify", model_name="mlip", indices=None,
              delta=0.01, nfree=2, imaginary_floor=IMAGINARY_FLOOR_CM1,
              ts_fmax_limit=0.03, overlap_threshold=0.7,
              displacement=0.1, match_tolerance=0.3,
              relax_fmax=0.01, relax_optimizer="bfgs", relax_max_steps=200,
              provenance=None, extra_inputs=None, run_context=None) -> dict
```

- `calculator_factory()` returns a fresh ASE calculator. `provenance` is a dict of keyword arguments for `collect_provenance` (model, device, task and head), passed through to `run_frequencies` as well.
- `reference_direction=None` means `is_to_fs_direction(initial, final)` with label `is_to_fs`.
- Returned dict (also the `results` of the `ts-verify` stage):

```python
{"verdict", "reason", "checks_requested": [...],
 "n_imaginary", "imaginary_frequencies_cm-1": [...],      # magnitudes of the counted modes
 "ts_fmax_free_eV_per_A", "ts_fmax_limit_eV_per_A", "ts_fmax_above_limit": bool,
 "displaced_indices": [...],
 "overlap": {"status", "value", "threshold", "reference"},  # only when requested
 "relax": {...},                                            # Task 3
 "n_force_calls_frequencies"}
```

- Files written in `output_dir`: `ts_structure.vasp`, the `ts_freq_*` files (prefix `ts_freq`, `write_modes="imaginary"`), `ts_verify.txt`, `mliprun_run.json`.
- `run_frequencies` gains `record_command: str = "freq"`, used as `command=` in its `RunRecord.begin`. `verify_ts` passes `record_command="ts-verify"`, then appends its own stage with `RunRecord.begin(output_dir, command="ts-verify", stage_kind="ts-verify", append=True, ...)`.
- The imaginary mode is read from `ts_freq_vibrations.json` with `ase.vibrations.VibrationsData`, as Cartesian displacements for all atoms (zeros on atoms not displaced). Confirm in the installed ASE that `get_modes(all_atoms=True)` is not mass-weighted; if it is, convert. The mode index is the single True entry of `imaginary_mask`.
- Order: `check_same_system` first (no force call), then frequencies, then mode count, then checks, then verdict. When the mode count is not 1, every requested check has status `not_run`.
- Any exception after the stage began completes the stage as `failed` with the error text, then re-raises.
- `ts_verify.txt`: verdict and reason on the first two lines, then mode count and frequencies, TS fmax and limit, reference label, overlap and threshold, the relax table (Task 3), and the settings.

**Test calculator** (`tests/ts_helpers.py`): `DoubleWell(Calculator)`, properties energy and forces, acting on atom 0 relative to a `centre`:
`E = a (x² − 1)² + by · y² + bz · z²`, defaults `a=1.0`, `by=2.0`, `bz=2.0` (eV, Å). Saddle at `centre`, minima at `centre ± (1, 0, 0)`, barrier 1.0 eV. `by=-2.0` gives a second imaginary mode. Helper `single_atom(position)` returns one H atom in a 10 Å cubic cell, `pbc=False`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_true_saddle_passes_the_overlap_check(tmp_path):
    r = verify_ts(ts, initial, final, calculator_factory=DoubleWell, output_dir=tmp_path)
    assert r["verdict"] == "verified"
    assert r["n_imaginary"] == 1
    assert r["overlap"]["value"] == pytest.approx(1.0, abs=1e-4)
    assert r["overlap"]["reference"] == "is_to_fs"
    assert r["overlap"]["status"] == "pass"

def test_a_mode_orthogonal_to_the_reference_fails(tmp_path):
    ref = np.array([[0.0, 1.0, 0.0]])
    r = verify_ts(..., reference_direction=ref, reference_label="neb_tangent", output_dir=tmp_path)
    assert r["overlap"]["value"] == pytest.approx(0.0, abs=1e-4)
    assert r["verdict"] == "not_verified"
    assert r["overlap"]["reference"] == "neb_tangent"

def test_the_overlap_threshold_is_adjustable(tmp_path):
    ref = np.array([[1.0, 1.0, 0.0]])                      # overlap 0.7071
    assert verify_ts(..., reference_direction=ref, reference_label="x")["verdict"] == "verified"
    assert verify_ts(..., reference_direction=ref, reference_label="x",
                     overlap_threshold=0.8, output_dir=tmp_path / "b")["verdict"] == "not_verified"

def test_two_imaginary_modes_at_a_converged_geometry_is_not_verified(tmp_path):
    r = verify_ts(..., calculator_factory=lambda: DoubleWell(by=-2.0), output_dir=tmp_path)
    assert r["n_imaginary"] == 2 and len(r["imaginary_frequencies_cm-1"]) == 2
    assert r["verdict"] == "not_verified"
    assert r["overlap"]["status"] == "not_run"

def test_two_imaginary_modes_at_a_loose_geometry_is_inconclusive(tmp_path):
    # TS displaced 0.05 A along x: |F| = 4*0.05*(1-0.0025) = 0.1995 eV/A > 0.03
    r = verify_ts(ts_off, ..., calculator_factory=lambda: DoubleWell(by=-2.0), output_dir=tmp_path)
    assert r["ts_fmax_free_eV_per_A"] == pytest.approx(0.1995, abs=1e-3)
    assert r["verdict"] == "inconclusive" and "not converged enough" in r["reason"]

def test_a_minimum_has_no_imaginary_mode(tmp_path):
    # "TS" placed at the minimum centre + (1, 0, 0)
    assert r["n_imaginary"] == 0 and r["verdict"] == "not_verified"

def test_outputs_and_record(tmp_path):
    for name in ("ts_structure.vasp", "ts_freq_frequencies.csv", "ts_freq_vibrations.json",
                 "ts_verify.txt", "mliprun_run.json"):
        assert (tmp_path / name).exists()
    record = json.loads((tmp_path / "mliprun_run.json").read_text())
    assert record["command"] == "ts-verify"
    assert [s["kind"] for s in record["stages"]] == ["freq", "ts-verify"]
    assert record["stages"][1]["results"]["verdict"] == "verified"
    assert (tmp_path / "ts_verify.txt").read_text().splitlines()[0].endswith("verified")

def test_mismatched_structures_stop_before_any_force_call(tmp_path):   # Review Focus 1
    # a counting calculator factory; assert it was never called and ValueError raised

def test_identical_endpoints_are_an_error_not_nan(tmp_path):           # Review Focus 2
    with pytest.raises(ValueError, match="zero"): verify_ts(ts, initial, initial, ...)

def test_freq_run_record_command_is_unchanged_by_default(tmp_path):
    # run_frequencies without record_command still writes command == "freq"
```

- [ ] **Step 2: Run and confirm failure** — `.venv/bin/pytest tests/test_core_ts_verify.py -q`.
- [ ] **Step 3: Implement** `record_command` in `run_frequencies`, then `verify_ts` (frequencies, mode count, overlap, verdict, report, record). The relax check raises `NotImplementedError` if requested, until Task 3.
- [ ] **Step 4: Run** `tests/test_core_ts_verify.py tests/test_core_vibrations.py tests/test_cli_freq.py`, expected: all pass.
- [ ] **Step 5: Commit** — `feat(ts): verify_ts with mode count and overlap check`.

---

### Task 3: Displace-and-relax check

**Files:**
- Modify: `src/mliprun/core/ts_verify.py`
- Test: `tests/test_core_ts_verify.py`

**Interfaces — Produces** the `relax` block of the result:

```python
{"status", "result", "displacement_A", "match_tolerance_A",
 "fmax_eV_per_A", "optimizer", "max_steps",
 "initial_fmax_eV_per_A", "final_fmax_eV_per_A", "endpoints_relaxed": bool,
 "plus":  {"converged", "steps", "distance_to_initial_A", "distance_to_final_A",
           "matched", "energy_difference_eV"},
 "minus": {...same keys...}}
```

- Two structures: `ts_atoms` positions plus and minus `scale_mode(mode, displacement)`, constraints copied from `ts_atoms`, each with `calculator_factory()`.
- Optimizer from `{"bfgs", "lbfgs", "fire"}` (same map as `CustomNEB.optimize_endpoints`); an unknown name is a `ValueError` before any force call. Trajectories `ts_relax_plus.traj`, `ts_relax_minus.traj`, logs `ts_relax_plus.log`, `ts_relax_minus.log`; final structures `ts_relax_plus.vasp`, `ts_relax_minus.vasp`.
- `matched` is the endpoint with the smaller `endpoint_distance`, when that distance is at most `match_tolerance`; otherwise `None`. `energy_difference_eV` is relaxed energy minus the matched endpoint's energy (one force call per endpoint with a fresh calculator), `None` when unmatched.
- `initial_fmax_eV_per_A` and `final_fmax_eV_per_A` come from those same endpoint calls (free components); `endpoints_relaxed` is False when either exceeds `relax_fmax`, and the report then prints the note from the spec ("Unrelaxed IS or FS").
- `status`: `pass` for `connects`, `inconclusive` for `inconclusive`, otherwise `fail`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_true_saddle_connects_both_minima(tmp_path):
    r = verify_ts(..., checks=("overlap", "relax"), output_dir=tmp_path)
    assert r["relax"]["result"] == "connects" and r["verdict"] == "verified"
    sides = {r["relax"]["plus"]["matched"], r["relax"]["minus"]["matched"]}
    assert sides == {"initial", "final"}
    assert r["relax"]["plus"]["energy_difference_eV"] == pytest.approx(0.0, abs=1e-4)
    for name in ("ts_relax_plus.traj", "ts_relax_minus.vasp"): assert (tmp_path / name).exists()

def test_relax_only_does_not_report_an_overlap(tmp_path):
    r = verify_ts(..., checks=("relax",)); assert "overlap" not in r and r["verdict"] == "verified"

def test_a_wrong_final_state_is_one_side(tmp_path):
    # final given at centre + (0, 3, 0), which is not a minimum
    assert r["relax"]["result"] == "one_side" and r["verdict"] == "not_verified"
    assert r["relax"]["endpoints_relaxed"] is False

def test_an_unconverged_relaxation_is_inconclusive(tmp_path):
    r = verify_ts(..., checks=("relax",), relax_max_steps=1)
    assert r["relax"]["result"] == "inconclusive" and r["verdict"] == "inconclusive"

def test_the_match_tolerance_decides_the_match(tmp_path):
    # final given at centre + (1.2, 0, 0): distance 0.2 A from the true minimum
    assert verify_ts(..., checks=("relax",))["relax"]["result"] == "connects"                 # 0.2 <= 0.3
    assert verify_ts(..., checks=("relax",), match_tolerance=0.1)["relax"]["result"] == "one_side"

def test_the_displacement_size_is_applied(tmp_path):
    # first frame of ts_relax_plus.traj is 0.1 A from the TS; with displacement=0.25 it is 0.25 A

def test_an_unknown_optimizer_stops_before_any_force_call(tmp_path): ...

def test_constraints_are_carried_into_the_relaxations(tmp_path):
    # two atoms, atom 1 fixed with FixAtoms and placed far away; after relax its position is unchanged
```

- [ ] **Step 2: Run and confirm failure.**
- [ ] **Step 3: Implement** the relax check and its section of `ts_verify.txt`.
- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_core_ts_verify.py tests/test_ts_verify_helpers.py -q`, expected: all pass.
- [ ] **Step 5: Commit** — `feat(ts): displace-and-relax check`.

---

### Task 4: Command `mlip ts verify`

**Files:**
- Create: `src/mliprun/cli/commands/ts.py`, `tests/test_cli_ts.py`
- Modify: `src/mliprun/cli/main.py`, `src/mliprun/core/ts_verify.py`

**Interfaces — Produces:**

```python
# core/ts_verify.py
def load_neb_for_verification(neb_dir) -> dict
# {"images": [...], "ts_index": int, "params": dict,          # params from CustomNEB._parse_parameters_file
#  "last_stage_status": str | None, "last_stage_climb": bool | None}
```

- Reads `A2B.traj` (all images), `neb_parameters.txt`, and the last stage of `mliprun_run.json` whose kind is `neb` or `neb-restart` (missing record gives `None` values). `ts_index` is the highest-energy image. Raises `ValueError` when `A2B.traj` or `neb_parameters.txt` is missing, and when `ts_index` is 0 or the last image ("no interior maximum").
- `ts.py`: `app = typer.Typer(help="Transition-state tools.")` with a no-op `@app.callback()` (same reason as in `freq.py`) and `@app.command() def verify(ctx, ...)`. Registered in `main.py` as `app.add_typer(ts.app, name="ts", help="Transition-state verification")`. No entry in `[project.scripts]`.
- Options: `--neb-dir`, `--ts`, `--initial`, `--final`, `--check` (default `overlap`; `overlap|relax|both`), `--mlip`, `--uma-task`, `--mace-head`, `--sevennet-task`, `--device`, `--indices`, `--delta`, `--nfree`, `--imaginary-floor`, `--ts-fmax-limit`, `--overlap-threshold`, `--displacement`, `--match-tolerance`, `--relax-fmax`, `--relax-optimizer`, `--relax-max-steps`, `--output-dir`.
- `--neb-dir` and the three files are mutually exclusive; the three files are all-or-none. Wrong combination, invalid `--check`, invalid `--nfree`, `--indices` parse error: message and exit 1 before any model loads.
- Model: with `--neb-dir`, the values from `neb_parameters.txt`, each overridden by its command-line option when given. With three files, as in `freq run` (`auto` detection, `validate_mlip`). `reject_aqcat_unsupported(mlip, "ts verify")` in both forms.
- Calculator factory: `lambda: build_calculator(mlip, uma_task, device=device, mace_head=mace_head, sevennet_task=sevennet_task)`.
- With `--neb-dir`: reference is `neb_tangent(images, ts_index)`, label `neb_tangent`; a last NEB stage that is not `converged`, or not climbing, prints a warning line and continues.
- Default output directory: `<neb-dir>/ts_verify`, or `ts_verify` beside the TS file.
- `FrequencyCacheError` and `ValueError` from the core: print the message, exit 1. A completed verification exits 0 for every verdict. The console prints the verdict, the reason, the key numbers and the path of `ts_verify.txt`.

- [ ] **Step 1: Write the failing tests.** Patch `mliprun.cli.commands.ts.build_calculator` to return `DoubleWell()` and `validate_mlip` / `detect_mlip` as `tests/test_cli_freq.py` does. Build a NEB folder fixture by writing a three-image `A2B.traj` (single atom at x = −1, 0, +1 around the centre, energies from `DoubleWell` via `SinglePointCalculator`), a `neb_parameters.txt` through `write_parameters_file` with `MLIP model`, `Intermediate images`, `Final fmax`, `Climb`, and a minimal `mliprun_run.json`.

```python
def test_three_files_verify_and_exit_zero(...):
    assert result.exit_code == 0 and "verified" in result.stdout
    assert (ts_file.parent / "ts_verify" / "ts_verify.txt").exists()

def test_a_neb_folder_uses_the_tangent_reference(...):
    record = json.loads((neb_dir / "ts_verify" / "mliprun_run.json").read_text())
    assert record["stages"][-1]["results"]["overlap"]["reference"] == "neb_tangent"
    assert record["stages"][-1]["results"]["verdict"] == "verified"

def test_the_neb_record_is_not_modified(...):       # bytes of neb_dir/mliprun_run.json unchanged
def test_check_both_runs_the_relax_check(...):      # results["relax"]["result"] == "connects"
def test_a_failed_verification_still_exits_zero(...):   # --overlap-threshold 1.1 -> not_verified, exit 0
def test_neb_dir_and_ts_together_are_refused(...):  # exit 1
def test_a_partial_three_file_set_is_refused(...):  # exit 1
def test_an_invalid_check_value_is_refused(...):    # exit 1
def test_an_aqcat_tag_is_refused(...):              # exit 1
def test_a_band_whose_maximum_is_an_endpoint_exits_nonzero(...):      # Review Focus 3, "no interior maximum"
def test_an_unconverged_neb_warns_and_continues(...):                 # exit 0, "not converged" in stdout
def test_a_foreign_cache_is_reported_without_a_traceback(...):        # Review Focus 4: run twice, second with --delta 0.02; exit 1, "Traceback" not in output
def test_mlip_ts_is_registered():                   # runner.invoke(main.app, ["ts", "--help"]) lists verify
```

For message assertions that name an option, strip ANSI codes first (Typer colours errors on CI).

- [ ] **Step 2: Run and confirm failure** — `.venv/bin/pytest tests/test_cli_ts.py -q`.
- [ ] **Step 3: Implement** `load_neb_for_verification`, `ts.py`, and the registration.
- [ ] **Step 4: Run** `tests/test_cli_ts.py tests/test_cli.py tests/test_cli_commands.py`, expected: all pass.
- [ ] **Step 5: Commit** — `feat(ts): mlip ts verify command`.

---

### Task 5: `neb run --verify-ts`

**Files:**
- Modify: `src/mliprun/cli/commands/neb.py`, `src/mliprun/core/neb.py`
- Test: `tests/test_neb_verify_ts.py`

**Interfaces — Consumes:** `verify_ts`, `neb_tangent`, `reject_aqcat_unsupported`.

- `run_neb` sets `self.converged = bool(converged)` (and `False` before the run starts). Nothing else in `run_neb` changes.
- New option `verify_ts: str = typer.Option(None, "--verify-ts", help=...)`, values `overlap|relax|both`.
- Stored as the line `Verify TS:` in `neb_parameters.txt` (only when set), parsed by `_parse_parameters_file` into `params["verify_ts"]`, and read back on `--restart` exactly as `climb` is; a value given on the restart command line overrides it.
- At start, before any model loads: an invalid value exits 1; `--verify-ts` with climb resolved to False exits 1 with a message that verification needs a climbing image; an AQCat tag with `--verify-ts` exits 1.
- After the NEB and the existing post-processing, when `verify_ts` is set:
  - `neb_obj.converged` False: print that TS verification was skipped because the NEB did not converge. Nothing else.
  - Highest image is an endpoint: print that verification was skipped (no interior maximum).
  - Otherwise call `verify_ts` on a copy of the highest image, with the first and last images, `reference_direction=neb_tangent(...)`, label `neb_tangent`, `calculator_factory=neb_obj.setup_calculator`, `output_dir=output_dir / "ts_verify"`, default thresholds, and the same provenance values the NEB record uses. Print the verdict and the report path.
  - The whole block is inside `try / except Exception`: on failure print `TS verification failed: <error>` and continue. The exit code stays 0.

- [ ] **Step 1: Write the failing tests.** Drive `neb run` in a temporary working directory the way `tests/test_neb_restart.py` and `tests/test_neb_dyneb.py` do (read them first and reuse their calculator patching). Use an EMT system with a real saddle: an Al adatom on a 2x2 Al(100) slab of 2 layers, bottom layer fixed, hopping between neighbouring hollow sites, 3 intermediate images.

```python
def test_a_converged_neb_is_verified(...):
    assert result.exit_code == 0
    results = json.loads(Path("ts_verify/mliprun_run.json").read_text())["stages"][-1]["results"]
    assert results["n_imaginary"] == 1 and results["overlap"]["reference"] == "neb_tangent"
    assert results["overlap"]["value"] > 0.7 and results["verdict"] == "verified"

def test_the_option_is_written_and_read_back_on_restart(...):
    assert "Verify TS:" in Path("neb_parameters.txt").read_text()
    # then --restart without the option: ts_verify/ts_verify.txt exists again

def test_without_the_option_nothing_is_verified(...):      # no ts_verify directory
def test_an_unconverged_neb_skips_verification(...):       # --neb-max-steps 1; "skipped" in stdout; no ts_verify.txt
def test_verify_ts_without_climb_is_refused_at_start(...): # --no-climb; exit 1; no neb.log written
def test_an_invalid_value_is_refused(...):
def test_a_failure_inside_verification_does_not_fail_the_neb(...):   # Review Focus 5
    # monkeypatch mliprun.cli.commands.neb.verify_ts to raise RuntimeError("boom")
    assert result.exit_code == 0 and "TS verification failed: boom" in result.stdout
    assert json.loads(Path("mliprun_run.json").read_text())["status"] == "converged"
```

If the EMT adatom band does not converge or does not give one imaginary mode with the default NEB settings, stop and report the numbers; do not loosen the assertions.

- [ ] **Step 2: Run and confirm failure** — `.venv/bin/pytest tests/test_neb_verify_ts.py -q`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_neb_verify_ts.py tests/test_neb_restart.py tests/test_neb_dyneb.py tests/test_core_neb.py tests/test_characterization_golden.py`, expected: all pass, no golden file changed (`git status --short tests/goldens` empty).
- [ ] **Step 5: Commit** — `feat(neb): --verify-ts runs TS verification after a converged CI-NEB`.

---

### Task 6: Documentation, full suite, coverage

**Files:**
- Create: `docs/TS_VERIFICATION.md`
- Modify: `README.md`, `AGENTS.md`, `CONTEXT.md`, `docs/OUTPUTS.md`, `docs/PYTHON_API.md`, `CHANGELOG.md`

- `docs/TS_VERIFICATION.md`: what is verified and why; both entry points with one example command each; the verdict rules including the loosely converged TS rule (limit 0.03 eV/Å, advice to restart the NEB at a tighter fmax); every option with its default; the comparison table from the spec section "Documentation", copied in full; a short "which one should I use" paragraph (overlap for routine screening; add displace-and-relax for a barrier that will be reported or a borderline overlap); the limits (the three-file reference is the straight IS-to-FS line; symmetry-equivalent minima can fail the match; AQCat tags, committees and the web app are not supported).
- `docs/OUTPUTS.md`: the files in `ts_verify/`; the `ts-verify` stage and its `results` keys; the statement that the NEB directory's record is not modified and the verification record is at `<neb-dir>/ts_verify/mliprun_run.json`; the note that the `ts-verify` stage kind is additive and does not bump the schema (extend the existing sentence about `singlepoint` and `freq`).
- `CONTEXT.md`: the terms "TS verification", "overlap check", "displace-and-relax check", "reference direction".
- `docs/PYTHON_API.md`: `verify_ts` signature and a short example. `README.md` and `AGENTS.md`: one entry each in the command lists. `CHANGELOG.md`: an "Added" entry under the unreleased section.
- Plain international English, no emojis in the docs, no em-dashes.

- [ ] **Step 1: Write the documentation.**
- [ ] **Step 2: Run the full unit suite** — `.venv/bin/pytest -m "not uma and not mace and not sevenn" --ignore=tests/webapp -q`. Expected: 936 plus the new tests passed, 6 skipped, 0 failed.
- [ ] **Step 3: Run diff coverage** with the command in `AGENTS.md`. Expected: at least 90 % on changed lines. Add tests for any uncovered branch; do not add `# pragma: no cover`.
- [ ] **Step 4: Check** `git status --short` shows no untracked artefacts and nothing under `tests/goldens/`.
- [ ] **Step 5: Commit** — `docs: TS verification`.
