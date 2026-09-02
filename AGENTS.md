# Agent guide — mliprun

Orientation for AI coding assistants (Claude Code, Cursor, Codex, …) and new
users driving this repo. Everything here is also human-readable; the README
covers usage in more depth.

## What this is

A CLI + Python API that drives ASE-based **geometry optimization, molecular
dynamics, and NEB/AutoNEB transition-state searches** using machine-learning
interatomic potentials (MLIPs): UMA (fairchem-core), MACE (mace-torch),
SevenNet (sevenn), and CHGNet (chgnet). `CONTEXT.md` defines the project
vocabulary (MLIP tag vs package vs env) — read it before discussing installs.

## Install (fresh machine)

```bash
git clone https://github.com/manuelarcer/mliprun.git
cd mliprun
python3 -m venv .venv && source .venv/bin/activate   # REQUIRED: bare pip fails
                          # on PEP 668 (externally-managed) system Pythons
pip install -e .          # base install: CLI + ASE, no MLIP yet
pip install mace-torch    # simplest working MLIP (no access request needed)
mlip doctor               # verify: exits 0 when the env is usable
```

Rules that are easy to get wrong:

1. **One MLIP package per Python environment.** mace-torch, fairchem-core,
   sevenn, and chgnet pin mutually incompatible torch/e3nn versions.
   Installing two into one env can silently break at least one. Tested
   per-MLIP recipes (conda and venv, CPU and CUDA): `docs/install/README.md`;
   rationale: `docs/adr/0001-per-mlip-envs.md`.
2. **Never `pip install asetools`.** PyPI's `asetools` is an unrelated
   Aseprite tool that shadows the real dependency. asetools is optional here
   (it enables the NEB interpolation sanity check); the correct way to get it
   is the extra `pip install -e ".[neb]"`, which pulls it from GitHub. If the
   impostor is installed: `pip uninstall asetools`, then reinstall the extra.
3. **UMA is gated.** fairchem-core needs a Hugging Face access request before
   any UMA model runs — see `docs/UMA_USAGE_GUIDE.md`. MACE works immediately,
   which is why `--mlip auto` falls back to it (order: UMA → MACE → SevenNet
   → CHGNet). **The "a fresh env lands on a runnable default" promise does not
   hold for SevenNet**, deliberately: a SevenNet-only env resolves to
   `7net-omni`, which is multi-task, and the run then stops for a missing
   `--sevennet-task`. SevenNet tasks are independent fine-tunes with
   independent energy zeros, so guessing one would silently change the level
   of theory; a loud stop is recoverable where a wrong energy zero is not.
4. **Model weights download on first use** into `~/.cache/mace/` (the Hugging
   Face cache for UMA; for SevenNet, into the installed package itself under
   `site-packages/sevenn/pretrained_potentials/`, ~103 MB per checkpoint). Air-gapped compute nodes need the checkpoint
   pre-fetched — commands are in the install recipes.

`mlip doctor` reports all of the above states (versions, asetools health,
installed MLIPs, `--mlip auto` resolution, torch/CUDA) and exits 1 if no
MLIP is installed — script against its exit code.

## Running

- Entry points: `mlip <subcmd>` or standalone `optimize`, `md`, `neb`,
  `autoneb`, `autoneb-results`, `benchmark`. All support `--help`.
- Typical: `optimize run --structure POSCAR --fmax 0.05`. Model selection via
  `--mlip` (default `auto`), device via `--device` (note: `neb` defaults to
  CPU, the others to auto).
- Plots are **opt-in** (`--plot`); CSV outputs are always written.
- Output locations differ: `optimize`/`md` write next to the input structure;
  `neb`/`autoneb` write into the current working directory. Full file
  reference: `docs/OUTPUTS.md`.
- Python API (for scripts/notebooks): `docs/PYTHON_API.md`.

## Testing

```bash
pip install -e ".[dev,neb]"
pytest -m "not uma and not mace and not sevenn"   # unit tests, no MLIP needed
pytest                                             # + integration (needs MLIPs)
```

- CI gates: full unit suite + diff coverage ≥ 90% on changed lines
  (`.github/workflows/tests.yml`).
- **Never regenerate golden reference files** (`tests/goldens/*.json`). If a
  golden test fails, report the numerical delta; updating baselines is a
  human decision. `scripts/generate_goldens.py` refuses to overwrite.
- Every test must assert a numerical value or invariant. Loosening a
  tolerance (recording the observed delta) is acceptable; silently changed
  numerics are not — numerical correctness is this package's top priority.

## Contribution conventions

- Draft PRs only; never push to main. One concern per PR.
- Match the existing code style: typer CLI, lazy imports for heavy packages,
  NumPy-style docstrings. See `CONTRIBUTING.md`.
