# Contributing to mliprun

Bug reports, feature requests, and pull requests are welcome. This guide covers the minimum you need to know to land a change.

## Issues

- **Bug reports**: please include the failing command, the MLIP and version (`pip show fairchem-core` / `mace-torch` / `sevenn` / `chgnet`), the structure file (or a minimal reproducer), and the full error.
- **Feature requests**: describe the use case and the user-visible behavior you want. If a similar flag exists, mention it.

Issue templates live under `.github/ISSUE_TEMPLATE/`.

## Setting up a development environment

```bash
git clone https://github.com/manuelarcer/mliprun.git
cd mliprun
python3 -m venv .venv && source .venv/bin/activate   # or a conda env — never system Python
pip install -e ".[dev,neb]"   # dev = pytest + coverage tools; neb = optional asetools

# Optionally install ONE MLIP for integration tests. One MLIP per environment —
# the packages pin incompatible torch/e3nn versions (see docs/adr/0001-per-mlip-envs.md
# and the recipes in docs/install/README.md):
pip install mace-torch      # simplest — works immediately, no access request
mlip doctor                 # verify what the environment can run
```

The `mlip` (including `mlip doctor`), `optimize`, `md`, `neb`, `autoneb`, `autoneb-results`, and `benchmark` commands all become available after `pip install -e .`. See the [command reference](docs/CLI.md) for usage and [docs/PYTHON_API.md](docs/PYTHON_API.md) for the public Python API.

## Running tests

```bash
# Unit tests only — no MLIP required, fastest
pytest -m "not uma and not mace and not sevenn"

# Everything that can run in the current environment (auto-skips missing MLIPs)
pytest

# Specific MLIP suite (requires that MLIP installed)
pytest -m uma
pytest -m mace
pytest -m sevenn

# Skip slow tests
pytest -m "not slow"
```

`pytest.ini` registers four markers: `uma`, `mace`, `sevenn`, `slow`. The fixtures and auto-skip logic are described in [tests/README.md](tests/README.md).

When adding a new test that needs a specific MLIP, mark the file or function so it auto-skips elsewhere:

```python
import pytest
pytestmark = pytest.mark.uma   # entire module gated on fairchem-core
```

## Code style

- **NumPy-style docstrings** for any new public function or class. The existing modules under `src/mliprun/core/` are the reference.
- **No new emojis** in source files unless they were already present in the surrounding context.
- **Lazy imports** for heavy MLIP packages — load them inside the function that needs the calculator, not at module top. See `src/mliprun/cli/utils.py` for the established pattern (`@functools.lru_cache(maxsize=1)` around the importer).
- **Type hints** on new public functions where they don't add noise. Internal helpers do not require them.

## CLI conventions

- Every subcommand uses Typer and exposes a single `run` (or `results`) entry. Match the existing files under `src/mliprun/cli/commands/`.
- The `--mlip` and `--uma-task` help strings are centralized in `src/mliprun/cli/utils.py` (`MLIP_HELP`, `UMA_TASK_HELP`). Reuse them.
- Output files belong in the directory documented in [docs/OUTPUTS.md](docs/OUTPUTS.md). New files added by an existing command should be added to that table in the same PR.
- Echo the resolved MLIP name once per run (`Auto-detected MLIP: …` / `Using MLIP: …`) — `resolve_mlip` already does this.

## Pull requests

1. Branch from `main`.
2. Keep the PR focused. One feature or one fix per PR.
3. Update documentation in the same PR:
   - User-visible flag changes → `docs/CLI.md` and the relevant `docs/*.md` page. The README is generated (`site/alt-5-cinematic/build_readme.py --root`); do not edit it by hand.
   - New output files → `docs/OUTPUTS.md`.
   - Public Python API changes → `docs/PYTHON_API.md`.
4. Add or update tests. Unit tests should run without any MLIP installed; mark MLIP-dependent tests as above.
5. Add an entry to `CHANGELOG.md` under `[Unreleased]` describing the user-visible change. Follow the existing structure (Added / Changed / Fixed / Removed / Deprecated).
6. Run the unit suite locally before pushing: `pytest -m "not uma and not mace and not sevenn"`.

## Continuous integration

Every push and pull request runs `.github/workflows/tests.yml`:

- The unit suite runs under coverage (`pytest --cov=mliprun --cov-report=xml`).
- A **diff-coverage gate** requires **>= 90% coverage on the lines your PR changes** (`diff-cover coverage.xml --compare-branch=origin/main --fail-under=90`, excluding `setup.py` and `scripts/*`). A PR that adds untested code will fail here — add tests for new lines before pushing.

A separate scheduled workflow (`.github/workflows/weekly.yml`) runs broader quality checks weekly; you do not need to trigger it for a PR.

## Versioning and releases

Versioning follows [Semantic Versioning](https://semver.org/). The version lives in `pyproject.toml`. To cut a release:

1. Move entries from `[Unreleased]` to a new `[X.Y.Z] - YYYY-MM-DD` section in `CHANGELOG.md`.
2. Bump the `version` string in `pyproject.toml`.
3. Tag the commit `vX.Y.Z`.

There is no automated PyPI release at present.
