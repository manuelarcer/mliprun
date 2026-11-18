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


if __name__ == "__main__":
    main()
