# mliprun

A CLI + Python API that drives ASE-based optimisation, MD, and NEB workflows using one of several supported machine-learning interatomic potentials (MLIPs).

## Language

**MLIP tag**:
The user-facing identifier passed to `--mlip` (e.g. `mace`, `mace-mh-1`, `uma-s-1p2`, `7net-omni`, `chgnet`). Maps one-to-many to package names.
_Avoid_: model name, calculator name.

**Package name**:
The PyPI distribution that provides an MLIP (`mace-torch`, `fairchem-core`, `sevenn`, `chgnet`). One package can serve several **MLIP tags** (e.g. `mace-torch` serves both `mace` and every `mace-mh-*`; `sevenn` serves every `7net-*`).
_Avoid_: distribution.

**MLIP env**:
A Python environment (venv or conda) that contains exactly one MLIP package plus its pinned dependency stack. The platform assumes a separate **MLIP env** per MLIP because MLIP packages have mutually incompatible torch / torch-geometric / e3nn version ranges. See [ADR 0001](./docs/adr/0001-per-mlip-envs.md).
_Avoid_: virtualenv (too generic), conda env (only one tool).

**Install recipe**:
A per-MLIP markdown file under `docs/install/<tag-or-package>.md` containing tested conda *and* venv instructions for creating an **MLIP env**, plus any first-run notes (HF auth for UMA, checkpoint pre-download for MACE-MH). The platform never executes installs; it points users at the recipe.

**Calculator**:
The ASE `Calculator` subclass instantiated from an MLIP package (e.g. `MACECalculator`, `FAIRChemCalculator`, `SevenNetCalculator`, `CHGNetCalculator`). Lives in the **MLIP env**, attached to an `ase.Atoms` object by `setup_calculator()` in `src/mliprun/cli/utils.py`.

**Committee**:
Two or more **Members** evaluated together against the same structure, driving one relaxation trajectory with their mean force and reporting the members' disagreement as a per-configuration uncertainty (sigma, eV/Å). Declared in a `committee.yaml` and run via `optimize run --committee`; not supported by `optimize batch`, `md`, or `neb`/`autoneb`.

**Member**:
One **MLIP tag** in one **MLIP env**, addressed by the driver process through a worker subprocess. A **Committee** needs at least two.

**Level of theory**:
The label a member's tag, together with its task or head, resolves to (a table in `src/mliprun/core/committee/config.py`). Members sharing one label are same-level; a tag/task combination the table does not recognise resolves to `unknown`, which counts as *possibly* mixed rather than assumed same-level.

## Relationships

- A run command (`optimize`, `md`, `neb`) requires exactly one **MLIP tag**, which resolves to one **Package name**, which lives in one **MLIP env**.
- An **MLIP env** contains one **Package name** and is documented by one **Install recipe**.
- An **MLIP tag** that fails `validate_mlip()` produces an error pointing at the relevant **Install recipe**.
- A **Committee** contains two or more **Members**, each of which is one **MLIP tag** in one **MLIP env**.

## Example dialogue

> **User:** "I want to compare MACE and UMA on the same structure — can I install both?"
> **Maintainer:** "Not in one env. `mace-torch` and `fairchem-core` pin incompatible torch versions. Create a separate **MLIP env** per **MLIP tag** — see `docs/install/mace.md` and `docs/install/uma.md`. Switch envs between runs."

## Flagged ambiguities

- "install MACE" was used ambiguously between "install `mace-torch`" (package) and "fetch the MACE-MH checkpoint" (weights). Resolved: package install is an **Install recipe** concern; weight fetching is handled lazily by `setup_calculator()` (`_ensure_mace_foundation_checkpoint` and `mace_mp`'s built-in downloader).
