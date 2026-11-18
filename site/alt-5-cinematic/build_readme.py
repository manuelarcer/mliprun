"""Write README.md for alternative 5 from the data pack (no typed numbers).

    python3 site/alt-5-cinematic/build_readme.py

Media come from build_media.py. GitHub-safe: no script, no iframes, relative
media links; the hero GIF has light and dark variants through <picture>.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPO = "https://github.com/manuelarcer/mliprun"


def load(name):
    return json.loads((DATA / name).read_text())


def kn(j, sub):
    for k in j["key_numbers"]:
        if sub in k["label"]:
            return k
    raise KeyError(f"{j['use_case']}.json: no key number matching {sub!r}")


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def num(v, unit=""):
    a = abs(v)
    if unit in ("steps", "models", "atoms"):
        return str(round(v))
    if unit == "cm⁻¹":
        return f"{v:.1f}"
    if a and a < 1e-3:
        e = math.floor(math.log10(a))
        return f"{v / 10 ** e:.1f}×10{str(e).translate(SUP)}"
    if a >= 100:
        return f"{v:.3f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{float(f'{v:.3g}'):g}"


def q(k):
    u = "" if k["unit"] == "dimensionless" else f" {k['unit']}"
    return f"{num(k['value'], k['unit'])}{u}"


def short(cmd):
    cmd = re.sub(r"^\S*/mlip\s", "mlip ", cmd)
    return re.sub(r"(?:/[^\s/]+)+/([^\s/]+)", r"\1", cmd)


def label(j, detail):
    if j.get("status") == "pending":
        return f"**Pending:** {detail}"
    return f"**{'Converged' if j['converged'] else 'Not converged'}:** {detail}"


def h2o(s):
    return s.replace("H2O", "H₂O")


def main():
    relax, com, neb, md, web = (load(f"{n}.json") for n in ("relax", "committee", "neb", "md", "webapp"))
    model = re.search(r"--mlip\s+(\S+)", relax["command"]).group(1)
    task = re.search(r"--uma-task\s+(\S+)", relax["command"]).group(1)
    members = [k["label"].replace("final energy, member ", "").split(" (")[0]
               for k in com["key_numbers"] if k["label"].startswith("final energy, member ")]
    ts = neb["ts_image_index"]
    built = md["structures"]["built"]
    n_images = len(neb["series"]["image_index"])
    slab = next((n for n in relax["notes"] if n.startswith("Slab:")), "")

    rows = []
    rows.append((
        '<img src="media/relax.gif" width="300" alt="O relaxes in the fcc hollow of Pt(111)">',
        "**Relax an adsorbate.**",
        f"`{short(relax['command'])}`",
        f"O height {q(kn(relax, 'O height above top Pt layer, fcc'))}, "
        f"{num(kn(relax, 'BFGS steps, O at fcc hollow')['value'], 'steps')} BFGS steps.",
        label(relax, f"fmax {q(kn(relax, 'final fmax, O at fcc hollow'))}, target {q(kn(relax, 'fmax target'))}."),
    ))
    rows.append((
        '<img src="media/committee.svg" width="300" alt="Committee-mean fmax and force sigma against BFGS step, log scale">',
        f"**Relax with a committee** of {len(members)} models.",
        f"`{short(com['command'])}`",
        f"Force σ at the end {q(kn(com, 'sigma_max_free, final step'))} "
        f"({num(kn(com, 'sigma_max_free / fmax, final step')['value'])} × fmax).",
        label(com, f"mean-force fmax {q(kn(com, 'final fmax of the committee-mean force'))}, target {q(kn(com, 'fmax target'))}."),
    ))
    rows.append((
        '<img src="media/ts_mode.gif" width="300" alt="Imaginary mode at the TS: mode, not dynamics"><br>'
        '<img src="media/neb_profile.svg" width="300" alt="NEB energy profile against reaction coordinate">',
        "**Find the path, verify the TS.** Imaginary mode shown (mode, not dynamics).",
        f"`{short(neb['command'])}`",
        f"Barrier {q(kn(neb, 'forward barrier'))}, reaction energy {q(kn(neb, 'reaction energy'))}, "
        f"{num(kn(neb, 'TS imaginary frequency')['value'], 'cm⁻¹')}i cm⁻¹.",
        label(neb, f"band fmax {q(kn(neb, 'final band fmax'))}, target {q(kn(neb, 'fmax target (band and endpoints)'))}; TS verified."),
    ))
    rows.append((
        f'<img src="media/md_bilayer_built.gif" width="300" alt="{h2o(built["label"])}. Static structure, not a simulation.">',
        "**Finite-temperature MD.**",
        "",
        f"{h2o(built['label'])}. Static structure, not a simulation.",
        label(md, "MD not run; next step pending a decision."),
    ))
    rows.append((
        '<a href="data/webapp.mp4"><img src="media/webapp-result.jpg" width="300" alt="The web app after a job finished: status done, energy and max-force plots"></a>',
        "**Run it on your own GPU machine from the web app.** No preconfigured machines: add your own (Machines, Add machine).",
        "",
        f"Showcase job on `{web['target']['machine']}`: {q(kn(web, 'job wall time on cos-cluster'))} wall time. Click for the recording.",
        label(web, f"fmax {q(kn(web, 'final fmax'))}, target {q(kn(web, 'fmax target'))}."),
    ))

    table = ["<table>"]
    for img, head, cmd, nums, lab in rows:
        cell = "<br>".join(x for x in (head, cmd, nums, lab) if x)
        table.append(f"<tr>\n<td width=\"320\">{img}</td>\n<td>\n\n{cell}\n\n</td>\n</tr>")
    table.append("</table>")

    out = f"""<!-- Generated by site/alt-5-cinematic/build_readme.py from data/*.json. Do not edit by hand. -->
<!-- TODO (promotion, Phase 5): relative links (index.html, data/, media/, ../../) become the GitHub Pages URL and repository-root paths. -->

# mliprun

[![Tests]({REPO}/actions/workflows/tests.yml/badge.svg)]({REPO}/actions/workflows/tests.yml)
[![Fresh-install smoke test]({REPO}/actions/workflows/install-smoke.yml/badge.svg)]({REPO}/actions/workflows/install-smoke.yml)

<a href="index.html"><img src="media/hero-dark.gif" width="100%" alt="O hopping from the fcc to the hcp hollow of Pt(111): the {n_images} converged NEB images as gold ghosts joined through the O centres, with fcc and hcp tags and the energy of each image"></a>

<sub>O hops fcc → hcp on Pt(111); NEB, barrier {num(kn(neb, 'forward barrier')['value'])} eV. Images, not dynamics.</sub>

**One CLI and one web app for machine-learning interatomic potentials** (UMA, MACE, SevenNet, CHGNet). **[Open the interactive page](index.html).**

{chr(10).join(table)}

<sub>Demonstration runs, {model}/{task}. [Provenance](../../docs/showcase/data/pack/PROVENANCE.md).</sub>

**Also:** `freq run`, `autoneb run`, `benchmark`, `singlepoint run`, `mlip doctor`.

## Installation

One MLIP per environment: see [`docs/INSTALL.md`](../../docs/INSTALL.md), then run `mlip doctor`.
"""
    (HERE / "README.md").write_text(out)
    print(f"wrote {HERE / 'README.md'}")


def essential(cmd, keep):
    """The run command cut to its subcommand plus the flags in `keep` (values kept)."""
    toks = short(cmd).split()
    out, i = [], 0
    while i < len(toks) and not toks[i].startswith("--"):
        out.append(toks[i])
        i += 1
    while i < len(toks):
        flag = toks[i]
        vals = []
        i += 1
        while i < len(toks) and not toks[i].startswith("--"):
            vals.append(toks[i])
            i += 1
        if flag in keep:
            out += [flag, *vals]
    return " ".join(out)


def details(cmd):
    return f"<details><summary>command as run</summary>\n\n`{short(cmd)}`\n\n</details>"


def root(pages_url=None):
    """Write the repository-root README.md: visuals first, essential commands, links to docs/."""
    relax, com, neb, web = (load(f"{n}.json") for n in ("relax", "committee", "neb", "webapp"))
    model = re.search(r"--mlip\s+(\S+)", relax["command"]).group(1)
    task = re.search(r"--uma-task\s+(\S+)", relax["command"]).group(1)
    members = [k for k in com["key_numbers"] if k["label"].startswith("final energy, member ")]
    n_images = len(neb["series"]["image_index"])
    m = "site/alt-5-cinematic/media"
    pack = "docs/showcase/data/pack"
    rows = [
        (f'<img src="{m}/relax.gif" width="360" alt="O relaxes in the fcc hollow of Pt(111)">',
         "**Relax an adsorbate.**",
         f"{num(kn(relax, 'BFGS steps, O at fcc hollow')['value'], 'steps')} BFGS steps. "
         + label(relax, f"fmax {q(kn(relax, 'final fmax, O at fcc hollow'))}."),
         essential(relax["command"], {"--structure", "--mlip", "--uma-task", "--fmax"}),
         relax["command"]),
        (f'<img src="{m}/committee.svg" width="360" alt="Committee-mean fmax and force sigma against BFGS step, log scale">',
         f"**Know when your models disagree.** A committee of {len(members)} MLIPs relaxes together.",
         f"Force σ at the end {q(kn(com, 'sigma_max_free, final step'))}, "
         f"{num(kn(com, 'sigma_max_free / fmax, final step')['value'])} × fmax. "
         + label(com, f"mean-force fmax {q(kn(com, 'final fmax of the committee-mean force'))}."),
         essential(com["command"], {"--structure", "--committee", "--uncertainty-plot"}),
         com["command"]),
        (f'<img src="{m}/ts_mode.gif" width="360" alt="Imaginary mode at the TS: mode, not dynamics">',
         "**Find the path, verify the TS.**",
         f"Barrier {q(kn(neb, 'forward barrier'))}; one imaginary mode, "
         f"{num(kn(neb, 'TS imaginary frequency')['value'], 'cm⁻¹')}i cm⁻¹ (mode, not dynamics). "
         + label(neb, f"band fmax {q(kn(neb, 'final band fmax'))}; TS verified."),
         essential(neb["command"], {"--initial", "--final", "--mlip", "--uma-task", "--climb", "--verify-ts"}),
         neb["command"]),
        (f'<img src="{m}/webapp.gif" width="360" alt="Screen recording: an optimize job is filled in the mliprun web app, sent to a remote GPU machine, runs, and returns its result">',
         "**Send jobs to your own GPU machine from the web app.**",
         "No preconfigured machines: add yours under Machines → Add machine. "
         f"This job: {q(kn(web, 'job wall time on cos-cluster'))} wall time.",
         "mlip webapp",
         None),
    ]
    table = ["<table>"]
    for img, head, nums, cmd, full in rows:
        cell = f"{head}<br>{nums}\n\n`{cmd}`" + (f"\n\n{details(full)}" if full else "")
        table.append(f"<tr>\n<td width=\"380\">{img}</td>\n<td>\n\n{cell}\n\n</td>\n</tr>")
    table.append("</table>")

    hero = (f'<img src="{m}/hero-dark.gif" width="100%" alt="O hopping from the fcc to the hcp hollow of Pt(111): '
            f'the {n_images} converged NEB images as gold ghosts, with the energy of each image">')
    if pages_url:
        hero = f'<a href="{pages_url}">{hero}</a>'
    live = f" **[Open the interactive page]({pages_url}).**" if pages_url else ""

    out = f"""<!-- Generated by site/alt-5-cinematic/build_readme.py --root from site/alt-5-cinematic/data/*.json. Do not edit by hand. -->

# mliprun

[![Tests]({REPO}/actions/workflows/tests.yml/badge.svg)]({REPO}/actions/workflows/tests.yml)
[![Fresh-install smoke test]({REPO}/actions/workflows/install-smoke.yml/badge.svg)]({REPO}/actions/workflows/install-smoke.yml)

{hero}

<sub>O hops fcc → hcp on Pt(111); NEB, barrier {num(kn(neb, 'forward barrier')['value'])} eV. Images, not dynamics.</sub>

**Machine-learning interatomic potentials from one command:** UMA, MACE, SevenNet and CHGNet, from a CLI or a web app.{live}

{chr(10).join(table)}

<sub>Showcase runs, {model}/{task}, on {web['target']['machine']} (one GPU card). [Provenance]({pack}/PROVENANCE.md).</sub>

**Also:** `md run` (NVE, NVT, NPT), `freq run`, `autoneb run`, `singlepoint run`, `benchmark`, `mlip doctor`.

## Install

```bash
git clone {REPO}.git && cd mliprun
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[neb]"
pip install mace-torch   # or UMA, SevenNet, CHGNet: one MLIP per environment
mlip doctor
```

Full guide, per-MLIP recipes and UMA access: [docs/INSTALL.md](docs/INSTALL.md).

## Docs

- [Command reference](docs/CLI.md): every command and option
- [Output files](docs/OUTPUTS.md) · [TS verification](docs/TS_VERIFICATION.md) · [MD reference](docs/MD_REFERENCE.md) · [Python API](docs/PYTHON_API.md)
- [Web app](webapp/README.md), including running on your own remote machine
- [Developing](docs/DEVELOPMENT.md) · [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)
- Using an AI coding assistant? Point it at [AGENTS.md](AGENTS.md).

## Acknowledgments

We gratefully acknowledge the contributions of:

- **Yifan Niu** (National University of Singapore) - Windows setup documentation and testing
- **Lee Yuan Zhang** (National University of Singapore) - Development and testing support

Special thanks to all contributors who have helped improve this platform.
"""
    path = HERE.parent.parent / "README.md"
    path.write_text(out)
    print(f"wrote {path}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args[:1] == ["--root"]:
        root(args[1] if len(args) > 1 else None)
    else:
        main()
