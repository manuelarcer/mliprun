# Installing mliprun

> Setting up with an AI coding assistant (Claude Code, Cursor, Codex, …)? Point it at [AGENTS.md](../AGENTS.md) — it covers the install rules, verification, and repo conventions in one place.

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/manuelarcer/mliprun.git
cd mliprun
```

2. **Create and activate a virtual environment** (don't skip this)
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\activate
```

Installing into the system Python fails outright on modern Linux/macOS (`error: externally-managed-environment`) and, where it doesn't, pollutes system packages. A dedicated environment is also how this project isolates MLIPs from each other (one MLIP per environment — [ADR 0001](adr/0001-per-mlip-envs.md)). Prefer conda, or setting up on an HPC/GPU machine? Use the per-MLIP recipes in [docs/install/](install/README.md) instead — they include tested conda and venv blocks.

3. **Install the package** (into the activated environment)
```bash
pip install -e .
```

Optional: the NEB interpolation sanity check uses [asetools](https://github.com/manuelarcer/asetools). Enable it with:
```bash
pip install -e ".[neb]"
```

> **Warning**: Never `pip install asetools` directly — the package named `asetools` on PyPI is an unrelated project (Aseprite tooling). The `[neb]` extra pulls the correct one from GitHub. If the wrong one is already installed: `pip uninstall asetools`, then `pip install -e ".[neb]"`.

4. **Install an MLIP model** (each in its own environment — see [install recipes](install/README.md))

```bash
# MACE - readily usable, no access request (good default for a fresh setup)
pip install mace-torch

# UMA models (FAIRChem) - most accurate, but gated on Hugging Face
pip install fairchem-core

# SevenNet (7net-omni recommended; needs --sevennet-task)
pip install torch_geometric sevenn

# CHGNet
pip install chgnet

# AQCat25-EV2 (SandboxAQ): patched fairchem-core 1.10.0, gated, non-commercial
# licence. No one-line install: follow docs/install/aqcat25.md
```

5. **Verify the install**
```bash
mlip doctor
```

Reports Python/package versions, asetools health, installed MLIP packages, what `--mlip auto` resolves to, and torch/CUDA status. Exits non-zero if no MLIP is installed, so it can be scripted.

> **Note**: With `--mlip auto` (the default), the CLI picks the first available in the order **UMA → MACE → SevenNet → CHGNet**, or you can pass `--mlip <name>` to force a specific one. Note that a SevenNet-only environment resolves to `7net-omni`, which is multi-task and then stops for a missing `--sevennet-task` — deliberately, since guessing a task would silently change the level of theory. Prefer one MLIP per environment: the packages pin mutually incompatible torch/e3nn versions, so installing several into one env can silently break — see [ADR 0001](adr/0001-per-mlip-envs.md). UMA is preferred when installed (it is the most accurate), but it is gated on Hugging Face and unusable without an access request — so MACE is placed ahead of SevenNet/CHGNet as the readily-usable fallback. A fresh environment with only `pip install mace-torch` lands on a working model with no access request. UMA's Hugging Face setup is covered in [UMA_USAGE_GUIDE.md](UMA_USAGE_GUIDE.md#2-hugging-face-access). The AQCat25-EV2 models are never picked by `auto`: in an env where they are the only MLIP, `auto` stops and names the tags and `--aqcat-spin` to pass.

## Windows Setup

For detailed Windows installation instructions, see: [Windows Setup Guide](windows_setup_guide.md)
