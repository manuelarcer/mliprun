# Transition-State Verification

A climbing-image NEB returns the highest image of a band. That image is a candidate for the transition state (TS), not a proof. The usual proof has two parts: the structure has exactly one imaginary vibrational mode, and that mode moves the system between the initial state (IS) and the final state (FS).

`mlip ts verify` tests both parts on a TS structure. It does not depend on how the TS was found, so later search methods can use it too. `neb run --verify-ts` runs the same test automatically after a converged climbing-image NEB.

For the files the verification writes, see [OUTPUTS.md](OUTPUTS.md#ts-verify). For the Python function, see [PYTHON_API.md](PYTHON_API.md#transition-state-verification).

---

## What is verified

1. **Frequencies.** The existing frequency calculation (`freq run`) runs on the TS structure. The atoms displaced are the free atoms, that is, all atoms not held by a `FixAtoms` constraint, unless `--indices` says otherwise.
2. **Mode count.** The number of imaginary modes must be exactly 1. A mode counts as imaginary only above the imaginary floor (default 10 cm^-1).
3. **Connection checks.** Only when the mode count is 1, the one or two checks you requested run (see below). With 0 or 2 or more imaginary modes, the checks are not run and are reported as `not_run`.
4. **Verdict.** One of `verified`, `not_verified` or `inconclusive`.

A failed verification is a result, not an error. The command exits with code 0, prints the verdict and records it. It exits with a non-zero code only when the verification could not be carried out, for example when an input file is missing.

---

## Two ways to run it

### From a finished NEB folder

```bash
mlip ts verify --neb-dir path/to/neb_run --check both
```

The command reads the final band (`A2B.traj`) and `neb_parameters.txt`. The TS is the highest-energy image. IS and FS are the first and last images. The model is the one the NEB used, unless you give model options on the command line. The reference direction for the overlap check is the NEB tangent at the TS image.

If the last NEB stage did not converge, or was not a climbing-image run, the command prints a warning and continues, because you asked for it explicitly. If the highest image is an endpoint, the band has no interior maximum and there is no TS to verify. The command then stops with an error.

### From three structure files

```bash
mlip ts verify --ts ts.vasp --initial is.vasp --final fs.vasp --mlip uma-s-1p2 --uma-task oc20
```

This form works for a TS from any method. The three structures must have the same atoms in the same order and the same cell. The reference direction is the straight IS-to-FS displacement, with the minimum-image convention for periodic cells.

The two forms cannot be mixed. `--neb-dir` cannot be combined with `--ts`, `--initial` or `--final`.

### Automatically after a NEB

```bash
neb run --initial POSCAR_A --final POSCAR_B --mlip uma-s-1p2 --verify-ts both
```

`--verify-ts` takes `overlap`, `relax`, `both` or `none` (off). It is off by default and is stored in `neb_parameters.txt`, so `neb run --restart` uses it again. It needs a climbing image: with `--no-climb` the command stops at the start, before any model loads. The verification runs only when the NEB stage ends converged. Otherwise a message says it was skipped. A problem inside the verification never turns a finished NEB into a failed one: the error is printed and the NEB result stays as it was. This path uses the default settings listed below. For other settings, run `mlip ts verify --neb-dir` afterwards.

---

## The two connection checks

| | Overlap check | Displace-and-relax check |
|---|---|---|
| Question answered | Does the imaginary mode point along the reaction path at the TS? | Does the TS fall to the IS on one side and to the FS on the other? |
| Extra cost | None beyond the frequencies | Two geometry optimizations |
| Strength | Fast, deterministic, a number from 0 to 1 | Tests the actual connection to the endpoints |
| Weakness | Local test only. Weak when the path is curved or the reference is the plain IS-to-FS line. Does not show where the system ends | Can end in a neighbouring or symmetry-equivalent minimum and fail although the TS is right. Depends on displacement size, optimizer and match tolerance |
| Use when | Routine screening of many NEBs | A barrier that will be reported, or an overlap that is borderline |

### Overlap check

The imaginary mode and the reference direction are both restricted to the displaced atoms, taken as Cartesian displacements (not mass-weighted), and normalised. The overlap is the absolute value of their dot product, a number between 0 and 1. The sign is dropped, because a mode has no direction. The check passes when the overlap is at least the threshold (default 0.7).

### Displace-and-relax check

The imaginary mode is scaled so that the largest single-atom displacement equals `--displacement` (default 0.1 A). Two structures are built, the TS plus this vector and the TS minus this vector. Each is relaxed with a fresh calculator. The relaxed structures keep the constraints of the TS structure, whatever `--indices` says.

Each relaxed structure is compared with IS and with FS. The distance is the largest single-atom minimum-image displacement. A structure matches an endpoint when this distance is at most `--match-tolerance` (default 0.3 A). The energy difference to the matched endpoint is reported but does not decide the match.

The result is one of:

| Result | Meaning |
|---|---|
| `connects` | One side matches IS and the other side matches FS. This passes. |
| `one_side` | One side matches an endpoint, the other matches neither. |
| `same_side` | Both sides match the same endpoint. |
| `neither` | No side matches an endpoint. |
| `inconclusive` | A relaxation did not converge within its steps, or a relaxation returned to the TS (see below). |

Only `connects` passes. The other results, except `inconclusive`, fail the check.

**A relaxation that returns to the TS.** If a relaxed structure matches neither endpoint and lies within the match tolerance of the TS itself, the relaxation did not test the connection. The check is then `inconclusive`, with the note "a relaxation returned to the TS; use a larger displacement or another optimizer". Each side reports `distance_to_ts_A` and `returned_to_ts`. This was seen with the `lbfgs` optimizer on a test potential. The default `bfgs` and `fire` reached the minima in the same test.

**Unrelaxed endpoints.** The report prints the fmax of IS and FS. When it exceeds the relax fmax, a note says that a failed match can reflect the endpoints and not the TS.

---

## Verdict rules

| Verdict | When |
|---|---|
| `verified` | One imaginary mode, and every requested check passed. |
| `not_verified` | The mode count is not 1 on a TS whose fmax is at or below the limit, or a requested check failed. |
| `inconclusive` | The mode count is not 1 on a TS whose fmax is above the limit; or one imaginary mode, no check failed, and at least one check is `inconclusive`. |

### A loosely converged TS

A NEB usually stops at fmax 0.05 eV/A. That can be too loose for a clean Hessian, so a real TS can show zero, or two or more, imaginary modes. Calling this "not verified" would read as "this is not a TS". The honest statement is "this TS is not converged enough to test". Therefore:

- The fmax of the TS structure over its free force components is always reported.
- If the mode count is not 1 and the TS fmax is above `--ts-fmax-limit` (default 0.03 eV/A), the verdict is `inconclusive`. The printed reason ends with the advice "restart the NEB with a tighter fmax, or relax the TS further, and verify again", for example `neb run --restart --fmax 0.02`.
- When the TS fmax is above the limit, `ts_verify.txt` and the console add the note "the TS fmax is above the limit; the geometry is loosely converged", whatever the verdict.
- If the mode count is not 1 and the TS fmax is at or below the limit, the verdict is `not_verified`.
- If the mode count is 1, the verdict does not depend on the TS fmax. The fmax and whether it is above the limit are still reported.

A NEB converged at the default fmax of 0.05 eV/A will usually show a TS fmax above 0.03 eV/A. This does not matter while the mode count is 1. It matters only when the mode count is not 1.

---

## Options

Options of `mlip ts verify`, with defaults:

| Option | Default | Meaning |
|---|---|---|
| `--neb-dir` | none | A finished NEB folder. Alternative to the three files. |
| `--ts`, `--initial`, `--final` | none | The three structure files. All three are needed together. |
| `--check` | `overlap` | `overlap`, `relax` or `both`. |
| `--mlip` | NEB's model with `--neb-dir`, otherwise `auto` | Model tag, as in `freq run`. |
| `--uma-task`, `--mace-head`, `--sevennet-task`, `--device` | from the NEB with `--neb-dir`, otherwise as in `freq run` | Model options. Values given on the command line override the NEB's. |
| `--indices` | all atoms not held by `FixAtoms` | Atoms displaced for the frequencies, e.g. `0,1,5` or `12-30`. It does not change which atoms the two relaxations move. |
| `--delta` | 0.01 | Displacement in A for the finite-difference Hessian. |
| `--nfree` | 2 | 2 (three-point) or 4 (five-point). |
| `--imaginary-floor` | 10 | cm^-1. A mode counts as imaginary only above this magnitude. |
| `--ts-fmax-limit` | 0.03 | eV/A. Above this fmax a wrong mode count gives `inconclusive`. |
| `--overlap-threshold` | 0.7 | Minimum overlap for the overlap check to pass. |
| `--displacement` | 0.1 | A. Largest single-atom displacement along the mode (relax check). |
| `--match-tolerance` | 0.3 | A. A relaxed structure matches an endpoint within this distance (relax check). |
| `--relax-fmax` | 0.01 | eV/A. Convergence of the two relaxations (relax check). |
| `--relax-optimizer` | `bfgs` | `bfgs`, `lbfgs` or `fire` (relax check). |
| `--relax-max-steps` | 200 | Step limit per relaxation (relax check). |
| `--output-dir` | `<neb-dir>/ts_verify`, or `ts_verify` beside the TS file | Where the outputs go. |

`neb run --verify-ts` takes the same three choices as `--check` and uses all other settings at the defaults above.

---

## Running the verification again

Running into a directory that already holds a verification (for example after `neb run --restart --fmax 0.02` moved the TS) follows three rules. Nothing is deleted silently.

- **The TS geometry changed** (different atoms, cell, or any position by more than 1e-6 A). Every entry of the output directory is moved into a new subdirectory `previous_<n>/` (`n` = 1, 2, ... the first unused number), and the verification starts on an empty directory. `results["archived_previous"]` holds the name, and `ts_verify.txt` and the console say: "A verification of a different TS geometry was found here and moved to previous_<n>/." Earlier `previous_*` directories stay where they are.
- **The geometry is the same.** The directory stays and the displacement cache is reused, so the frequencies cost no new force calls. The old `ts_verify.txt` and any `ts_relax_plus.*` and `ts_relax_minus.*` files are removed first, so that a run that fails, or that asks for fewer checks, never leaves an older verdict or older relaxation files beside the new record.
- **The geometry is the same but the settings are not** (for example another `--delta`). The command refuses the cache with the same message as `freq run`. Use `--output-dir` to give the new run its own directory.

To switch off a stored `--verify-ts` on a restart, give `neb run --restart --verify-ts none`. This also removes the reason to refuse `--no-climb`. On a new run `none` means off, and no `Verify TS:` line is written.

---

## Worked example

The imaginary frequency, the overlap and the barrier below are pinned by a test in the test suite. The system is an Al adatom hopping between neighbouring hollow sites on Al(100): a 2x2 cell, 2 layers, bottom layer fixed. The potential is EMT, a simple test potential built into ASE. It is used here only to illustrate the output, and the numbers have no physical meaning for a real MLIP calculation.

The NEB used 3 intermediate images and default settings. It converged in about 25 FIRE steps. The barrier is 0.238 eV. The verification then reported:

- Imaginary modes: 1, at 64.5i cm^-1.
- Overlap of the imaginary mode with the NEB tangent: 0.998, above the threshold 0.7.
- Verdict: `verified`.

The same information is in `ts_verify/ts_verify.txt` and in the run record, see [OUTPUTS.md](OUTPUTS.md#ts-verify).

---

## Which check should I use

Use `overlap` for routine screening of many NEBs. It costs nothing beyond the frequencies. Add the displace-and-relax check, with `--check both`, for a barrier that you will report, or when the overlap is close to the threshold. The relax check tests the connection to the endpoints directly, but it can fail on a correct TS, so read its distances and not only its result.

---

## Limits

- The three-file form uses the straight IS-to-FS line as reference. When the real path is curved, the overlap with this line can be low for a correct TS. The report states which reference was used (`neb_tangent` or `is_to_fs`).
- Symmetry-equivalent minima can fail the match. A relaxed structure can be the same minimum as an endpoint, but with equivalent atoms exchanged. The largest single-atom distance is then large, and the check reports `neither` or `one_side`.
- The overlap check is a local test. It does not show where the system ends.
- AQCat25-EV2 models are refused, because their forces are a direct output head and not the gradient of the energy (see the item "Forces" in [install/aqcat25.md](install/aqcat25.md)). Committees (`--committee`) and the web app are not supported. A finished AutoNEB can still be checked with the three-file form.
- This is not an intrinsic reaction coordinate calculation. The relax check is a plain minimisation from two displaced structures.
