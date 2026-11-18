# Setup Python, mliprun, and UMA Model (FAIRChem) on Windows

This guide walks you through setting up Python, mliprun, and the UMA Model from FAIRChem on Windows using PowerShell terminal in VSCode.

## Table of Contents

- [Install Python](#install-python)
- [Setup a Virtual Environment](#setup-a-virtual-environment)
- [Setup mliprun](#setup-mliprun)
- [Setup UMA Model (FAIRChem)](#setup-uma-model-fairchem)

---

## Install Python

You can install Python on Windows by following this tutorial:
[https://www.digitalocean.com/community/tutorials/install-python-windows-10](https://www.digitalocean.com/community/tutorials/install-python-windows-10)

> **Important:** Python 3.11 is recommended. FAIRChem currently supports Python >=3.10 and <3.13. See [fairchem-core on PyPI](https://pypi.org/project/fairchem-core/) for the latest compatibility information.

---

## Setup a Virtual Environment

A virtual environment isolates your Python packages for this project from other projects, preventing conflicts between dependencies.

Reference: [Python Packaging Guide - Virtual Environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments)

### 1. Create a virtual environment called 'uma'

In PowerShell within VSCode, navigate to a folder where you want your project. For example:

```powershell
cd C:\Users\YourUsername\Projects\MLIP
```

> **Note:** Replace `C:\Users\YourUsername\Projects\MLIP` with your preferred project directory path.

Then create the virtual environment:

```powershell
python -m venv uma
```

**Expected result:** You should see a new folder named `uma` created in your current directory. This folder contains the isolated Python environment.

### 2. Activate the virtual environment

```powershell
.\uma\Scripts\activate
```

**Expected result:** After running this command, your PowerShell prompt should change to show `(uma)` at the beginning. This indicates the virtual environment is active.

For example, your prompt should change from:
```
PS C:\Users\YourUsername\Projects\MLIP>
```
to:
```
(uma) PS C:\Users\YourUsername\Projects\MLIP>
```

### 3. Install pip

pip is the Python package manager used to install and update packages.

First, ensure pip is up to date:

```powershell
python -m pip install --upgrade pip
```

**Expected output:** You should see output showing the download and installation progress, ending with something like:
```
Successfully installed pip-25.3
```

Then verify pip is available:

```powershell
python -m pip --version
```

**Expected output:** You should see the pip version and its location, for example:
```
pip 25.3 from C:\Users\YourUsername\Projects\MLIP\uma\lib\site-packages\pip (python 3.11)
```

Now your isolated environment is ready to install the required packages.

---

## Setup mliprun

### 1. Install Git

Git is required to clone the mliprun repository.

**Installation steps:**

1. Go to the official Git website: [https://git-scm.com/downloads](https://git-scm.com/downloads)
2. Download the latest x64 version of Git for Windows (e.g., version 2.51.2)
3. Double-click the downloaded `.exe` file to start installation
4. During installation, use the default settings, but pay attention to these key options:

| Setup Screen | Recommended Choice | Why |
|--------------|-------------------|-----|
| **Adjusting your PATH environment** | ✓ "Git from the command line and also from 3rd-party software" | Allows you to use `git` in PowerShell or VSCode terminal |
| **Choosing the default editor** | "Vim" or "VS Code" (your preference) | VS Code is easier for most users |
| **Configuring line endings** | "Checkout Windows-style, commit Unix-style line endings" | Recommended default for Windows |
| **Other options** | Leave as default | Safe for most users |

**Verify the installation:**

Open the PowerShell terminal in VSCode and run:

```powershell
git --version
```

**Expected output:**
```
(uma) PS C:\Users\YourUsername\Projects\MLIP> git --version
git version 2.51.0.windows.1
```

### 2. Clone the mliprun repository

Before cloning, make sure you're in your project directory and have activated the virtual environment:

```powershell
cd C:\Users\YourUsername\Projects\MLIP
.\uma\Scripts\activate
```

Clone the repository:

```powershell
git clone https://github.com/manuelarcer/mliprun.git
cd mliprun
```

**Expected output:** You should see progress messages showing the cloning process:
```
Cloning into 'mliprun'...
remote: Enumerating objects: 836, done.
remote: Counting objects: 100% (157/157), done.
remote: Compressing objects: 100% (90/90), done.
remote: Total 836 (delta 64), reused 132 (delta 56), pack-reused 679 (from 1)
Receiving objects: 100% (836/836), 21.25 MiB | 9.95 MiB/s, done.
Resolving deltas: 100% (331/331), done.
```

After changing directory, your prompt should show:
```
(uma) PS C:\Users\YourUsername\Projects\MLIP\mliprun>
```

### 3. Install mliprun package

While in the `mliprun` directory, install the package in editable mode:

```powershell
pip install -e .
```

**What this does:** The `-e .` option installs the package in editable (development) mode, which means:
- Changes you make to the source code are immediately reflected without reinstalling
- A set of CLI commands becomes available in your environment

**Expected output (last few lines):**
```
Successfully built mliprun
Installing collected packages: typing-extensions, click, olefile, fonttools, cycler, colorama, scipy, ...
Successfully installed ase-3.26.0 click-8.1.8 ... mliprun-0.2.0
```

**Verify installation:** Each of the following commands should work whenever the 'uma' virtual environment is active:

| Command | What it runs |
|---------|--------------|
| `mlip --help` | Top-level namespace; lists every subcommand below |
| `optimize --help` | Geometry optimization |
| `md --help` | Molecular dynamics (NVE / NVT / NPT) |
| `neb --help` | Nudged Elastic Band |
| `autoneb --help` | AutoNEB with dynamic image insertion |
| `autoneb-results --help` | Extract / plot results from a completed AutoNEB run |
| `benchmark --help` | MACE / SevenNet timing benchmarks |
| `mlip doctor` | Environment self-check: installed MLIPs, torch/CUDA, asetools health. Exits with an error until an MLIP is installed (next section) — that is expected at this stage. |

`mlip <name>` is equivalent to running `<name>` directly — for example, `mlip md run --structure POSCAR` and `md run --structure POSCAR` do the same thing. Use whichever you prefer.

---

## Setup UMA Model (FAIRChem)

The FAIRChem-Core library from the FAIR Chemistry team provides access to the UMA model and ASE integration.

Reference: [FAIRChem GitHub Repository](https://github.com/facebookresearch/fairchem?tab=readme-ov-file#fairchem-by-the-fair-chemistry-team)

### 1. Install FAIRChem (fairchem-core)

From PowerShell (with your 'uma' venv activated), navigate back to your projects directory:

```powershell
cd ..
```

Your prompt should now show:
```
(uma) PS C:\Users\YourUsername\Projects\MLIP>
```

Install fairchem-core:

```powershell
pip install fairchem-core
```

Verify the installation:

```powershell
pip show fairchem-core
```

**Expected output:**
```
Name: fairchem-core
Version: 2.10.0
Summary: Machine learning models for chemistry and materials science by the FAIR Chemistry team
Home-page:
Author-email:
License: MIT License
Location: C:\Users\YourUsername\Projects\MLIP\uma\lib\site-packages
Requires: ase, ase-db-backend, Clustercope, e3nn, huggingface-hub, hydra-core, lmdb, monty, numba, numpy, orjson, pymatgen, requests, submitit, torch, torchviz, tqdm, wandb, websockets
```

### 2. Login to Hugging Face for UMA Model Access

The UMA model is gated on Hugging Face, requiring authentication to access.

#### Step 1: Create a Hugging Face account

If you don't already have one, sign up at: [https://huggingface.co/join](https://huggingface.co/join)

> **Important:** Use your work email address when signing up. Requests from personal email accounts may be rejected.

#### Step 2: Request access to the UMA model

1. Log in to your Hugging Face account
2. Navigate to the UMA model page: [https://huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA)
3. Click the button to request access or agree to the terms for the UMA model
4. Submit your access request

#### Step 3: Check your access status

1. Go to your **profile icon** (top right)
2. Select **Settings** → **Gated Repositories**
3. Look for the UMA model in the list

**What to look for:**
- **Status: ACCEPTED** means you have been granted access
- **Status: PENDING** means your request is under review
- The table shows columns: Repo Name, Type, Date, Request Status

#### Step 4: Generate an access token

1. Go to **profile icon** → **Settings** → **Access Tokens**
2. Click **"New Token"**
3. **Token name:** Give it a descriptive name (e.g., "uma-read-2025-11-03")
4. **Token type:** Select **"Read"** (this provides read-only access to gated models)
5. Click **"Create token"**
6. **Important:** Copy and save the generated token immediately - it will not be shown again!

#### Step 5: Login to Hugging Face from your terminal

Install the Hugging Face CLI tool (if not already installed):

```powershell
pip install huggingface_hub
```

Run the login command:

```powershell
hf auth login
```

**What happens:**
- You'll be prompted: `Token:` or `User is already logged in.`
- Paste your access token when prompted (it won't be visible as you paste)
- Press Enter

**Expected output on successful login:**
```
(uma) PS C:\Users\YourUsername\Projects\MLIP> hf auth login
User is already logged in.
```

Or if logging in for the first time, you'll see confirmation that the token has been saved.

**Note:** The token is saved in your user profile (`~/.huggingface/token`), so you typically only need to do this once per machine.

For more details, see: [Hugging Face Hub Quick Start](https://huggingface.co/docs/huggingface_hub/quick-start)

---

## You're All Set!

Your environment is now fully configured with:
- ✓ Python virtual environment ('uma')
- ✓ mliprun installed and `mlip` CLI available
- ✓ FAIRChem-Core installed with UMA model access

**Next steps:**
- Keep the 'uma' virtual environment activated when working with mliprun
- Use `mlip --help` to explore the full command namespace, or call any subcommand directly (e.g. `optimize --help`, `md --help`)
- Refer to the [command reference](CLI.md) for usage examples and documentation
- For UMA-specific examples, see [UMA Models Usage Guide](UMA_USAGE_GUIDE.md)

**Troubleshooting tips:**
- If commands are not found, ensure the 'uma' virtual environment is activated (look for `(uma)` in your prompt)
- If UMA model downloads fail, verify your Hugging Face access status and that you've logged in with `hf auth login`
- On Windows, if you encounter execution policy errors, you may need to run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
