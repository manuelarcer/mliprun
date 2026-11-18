"""Build README.md (and its media) for alternative 4, the handbook.

    site/.venv/bin/python site/alt-4-handbook/build_readme.py            # README + media
    site/.venv/bin/python site/alt-4-handbook/build_readme.py --no-media # README text only

Every number and every command in README.md comes from data/*.json (copied from the
data pack by build_data.py). Recipes follow the same rule as the landing page: the
binary path becomes `mlip`, and only path values become placeholders; flags and
values stay exactly as run. GIFs are rendered with site/shared/capture.py (outside the
sandbox: Chromium), in a light and a dark variant for GitHub's two themes. Charts are
plain SVG written here, also in two variants.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
DATA = HERE / "data"
MEDIA = HERE / "media"
PY = sys.executable

LIGHT = {"bg": "#ffffff", "ink": "#1f2328", "muted": "#59636e", "grid": "#e4e8ec",
         "c1": "#1c5d86", "c2": "#b2405f", "c3": "#5f7f22", "ok": "#2b7647"}
DARK = {"bg": "#0d1117", "ink": "#e6edf3", "muted": "#9198a1", "grid": "#262c33",
        "c1": "#7cb5dc", "c2": "#e47f9a", "c3": "#a8c86a", "ok": "#74c493"}
GH_DARK_BG = "#0d1117"

LANDING = "index.html"  # TODO(Phase 5): replace with the GitHub Pages URL once Pages is set up
ROOT = False  # --root: paths relative to the repository root (README promoted in Phase 5)


def link(path):
    """Repo path (e.g. 'docs/INSTALL.md') as a link from this README's location."""
    return path if ROOT else "../../" + path


def media(name):
    return ("site/alt-4-handbook/media/" if ROOT else "media/") + name


def landing():
    return "site/alt-4-handbook/index.html" if ROOT else LANDING


def load(name):
    return json.loads((DATA / name).read_text())


P = {k: load(f"{k}.json") for k in ("relax", "committee", "neb", "md", "webapp")}
P["md_planned"] = load("md_planned.json")
META = load("meta.json")
YAML = (DATA / "committee.yaml").read_text()


# ---------------------------------------------------------------- numbers
def kn(uc, prefix):
    for k in P[uc]["key_numbers"]:
        if k["label"].startswith(prefix):
            return k
    raise SystemExit(f"key number not found: {uc} | {prefix}")


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt_value(v, unit):
    a = abs(v)
    if unit in ("steps", "models", "atoms"):
        return str(round(v))
    if a != 0 and a < 1e-3:
        m, e = f"{v:.1e}".split("e")
        return f"{m} × 10{str(int(e)).translate(SUP)}"
    if unit == "eV" and a >= 10:
        return f"{v:.4f}"
    if unit in ("Å",):
        return f"{v:.3f}"
    if unit in ("cm⁻¹", "s"):
        return f"{v:.1f}"
    s = f"{float(f'{v:.3g}'):g}" if a >= 1e-4 else f"{v:.3g}"
    return s


def num(k, imaginary=False):
    u = "" if k["unit"] == "dimensionless" else " " + k["unit"]
    return f"{fmt_value(k['value'], k['unit'])}{'i' if imaginary else ''}{u}"


def cap(s):
    return s[:1].upper() + s[1:]


def status(d):
    if d.get("status") == "pending":
        return "pending"
    s = "converged" if d["converged"] else "**not converged**"
    if "ts_verified" in d.get("flags", []):
        s += ", TS verified"
    return s


# ---------------------------------------------------------------- recipes
def recipe(cmd, subs=None, pre=()):
    subs = subs or {}
    t = cmd.split()
    if re.search(r"(^|/)mlip$", t[0]):
        t[0] = "mlip"
    head, i = [], 0
    while i < len(t) and not t[i].startswith("--"):
        head.append(t[i]); i += 1
    groups = []
    while i < len(t):
        flag = t[i]; g = [flag]; i += 1
        while i < len(t) and not t[i].startswith("--"):
            g.append(subs.get(flag, t[i])); i += 1
        groups.append(g)
    lines, cur, cur_len = [" ".join(head)], None, 0
    for g in groups:
        ln = sum(len(p) + 1 for p in g)
        if cur is None or cur_len + ln > 52:
            cur = []; lines.append(cur); cur_len = 0
        cur.extend(g); cur_len += ln
    out = list(pre)
    rows = [lines[0]] + ["  " + " ".join(c) for c in lines[1:]]
    out += [r + (" \\" if j < len(rows) - 1 else "") for j, r in enumerate(rows)]
    return "\n".join(out)


def flag_value(cmd, flag):
    t = cmd.split()
    return t[t.index(flag) + 1] if flag in t else None


# ---------------------------------------------------------------- SVG charts
def nice_ticks(lo, hi, n=4):
    span = hi - lo or abs(hi) or 1
    mag = 10 ** math.floor(math.log10(span / n))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if span / (m * mag) <= n), 10 * mag)
    v = math.ceil(lo / step - 1e-9) * step
    out = []
    while v <= hi + step * 1e-9:
        out.append(float(f"{v:.12g}")); v += step
    return out


def svg_chart(path, theme, x, series, xlabel, ylabel, log=False, thr=None, mark=None,
              y_zero=False, w=560, h=230):
    T = theme
    ml, mr, mt, mb = 54, 16, 28, 44
    ys = [v for s in series for v in s["y"]] + ([thr[0]] if thr else [])
    xmin, xmax = min(x), max(x)
    if log:
        lo = math.floor(math.log10(min(v for v in ys if v > 0)))
        hi = math.ceil(math.log10(max(ys)))
        yt = [10.0 ** e for e in range(lo, hi + 1)]
        ty = lambda v: mt + (h - mt - mb) * (1 - (math.log10(v) - lo) / (hi - lo))
    else:
        lo, hi = min(ys), max(ys)
        if y_zero:
            lo = min(lo, 0)
        pad = (hi - lo) * 0.08
        lo = lo if (y_zero and lo == 0) else lo - pad
        hi += pad
        yt = nice_ticks(lo, hi)
        st = yt[1] - yt[0] if len(yt) > 1 else 1
        while yt[-1] < max(ys):
            yt.append(float(f"{yt[-1] + st:.12g}"))
        y0, y1 = min(lo, yt[0]), max(hi, yt[-1])
        ty = lambda v: mt + (h - mt - mb) * (1 - (v - y0) / (y1 - y0))
    xt = nice_ticks(xmin, xmax, 6)
    tx = lambda v: ml + (w - ml - mr) * (v - xmin) / (xmax - xmin)
    lab = lambda v: f"{v:g}"
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
         f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif" font-size="11">',
         f'<rect width="{w}" height="{h}" fill="{T["bg"]}"/>']
    for v in yt:
        y = ty(v)
        o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="{T["grid"]}"/>')
        o.append(f'<text x="{ml-6}" y="{y+3.5:.1f}" text-anchor="end" fill="{T["muted"]}">{lab(v)}</text>')
    for v in xt:
        xx = tx(v)
        o.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{h-mb}" y2="{h-mb+4}" stroke="{T["muted"]}"/>')
        o.append(f'<text x="{xx:.1f}" y="{h-mb+16}" text-anchor="middle" fill="{T["muted"]}">{lab(v)}</text>')
    o.append(f'<path d="M{ml},{mt-4}V{h-mb}H{w-mr}" fill="none" stroke="{T["muted"]}"/>')
    o.append(f'<text x="{ml}" y="13" fill="{T["ink"]}" font-weight="600">{ylabel}</text>')
    o.append(f'<text x="{(ml+w-mr)/2}" y="{h-8}" text-anchor="middle" fill="{T["muted"]}">{xlabel}</text>')
    if thr:
        y = ty(thr[0])
        o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="{T["muted"]}" stroke-width="1.2" stroke-dasharray="1.5 3.5" stroke-linecap="round"/>')
        o.append(f'<text x="{ml+8}" y="{y-5:.1f}" fill="{T["muted"]}">{thr[1]}</text>')
    for j, s in enumerate(series):
        c = T[f"c{j+1}"]
        d = "".join(f'{"L" if i else "M"}{tx(x[i]):.1f},{ty(v):.1f}' for i, v in enumerate(s["y"]))
        dash = ' stroke-dasharray="5 3"' if j == 2 else ""
        o.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2"{dash}/>')
        if s.get("dots"):
            o += [f'<circle cx="{tx(x[i]):.1f}" cy="{ty(v):.1f}" r="3.2" fill="{c}"/>' for i, v in enumerate(s["y"])]
    if len(series) > 1:  # legend, top right
        lx = w - mr
        for j, s in reversed(list(enumerate(series))):
            tw = 6.2 * len(s["label"]) + 26
            lx -= tw
            c = T[f"c{j+1}"]
            dash = ' stroke-dasharray="5 3"' if j == 2 else ""
            o.append(f'<line x1="{lx:.0f}" x2="{lx+16:.0f}" y1="24" y2="24" stroke="{c}" stroke-width="2"{dash}/>')
            o.append(f'<text x="{lx+20:.0f}" y="27.5" fill="{T["muted"]}">{s["label"]}</text>')
    if mark:
        i, label = mark
        cx, cy = tx(x[i]), ty(series[0]["y"][i])
        o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{T["c2"]}"/>')
        o.append(f'<text x="{cx:.1f}" y="{cy-10:.1f}" text-anchor="middle" fill="{T["ink"]}" font-weight="600">{label}</text>')
    o.append("</svg>")
    path.write_text("\n".join(o) + "\n")


def charts():
    rs, cs, ns = P["relax"]["series"], P["committee"]["series"], P["neb"]["series"]
    fT, cT = kn("relax", "fmax target"), kn("committee", "fmax target")
    ts = P["neb"]["ts_image_index"]
    for name, T in (("light", LIGHT), ("dark", DARK)):
        svg_chart(MEDIA / f"relax-fmax-{name}.svg", T, rs["step"],
                  [{"y": rs["fmax_eV_per_A"], "dots": True, "label": "fmax"}],
                  "BFGS step", "fmax (eV/Å, log scale)", log=True, thr=(fT["value"], f"target {num(fT)}"))
        svg_chart(MEDIA / f"committee-force-{name}.svg", T, cs["step"],
                  [{"y": cs["fmax_eV_per_A"], "label": "fmax, committee-mean force"},
                   {"y": cs["sigma_max_free_eV_per_A"], "label": "max force σ, free"},
                   {"y": cs["sigma_mean_free_eV_per_A"], "label": "mean force σ, free"}],
                  "BFGS step", "force (eV/Å, log scale)", log=True, thr=(cT["value"], f"fmax target {num(cT)}"))
        svg_chart(MEDIA / f"neb-profile-{name}.svg", T, ns["reaction_coordinate_A"],
                  [{"y": ns["relative_energy_eV"], "dots": True, "label": "image"}],
                  "reaction coordinate (Å)", "E − E(IS) (eV)", y_zero=True,
                  mark=(ts, f"TS, image {ts} (0–{len(ns['image_index']) - 1}): {num(kn('neb', 'forward barrier'))}"))


# ---------------------------------------------------------------- GIFs
def gifs():
    cap = SITE / "shared/capture.py"
    jobs = {
        "relax": (DATA / "relax.xyz", ["--elevation", "14", "--azimuth", "-30", "--fps", "4", "--hold", "1.2"]),
        "neb-band": (DATA / "neb_band.xyz", ["--elevation", "62", "--azimuth", "-30", "--fps", "3",
                                             "--hold", "0.8", "--bounce", "--image-tint", "0.55"]),
        "ts-mode": (DATA / "ts_mode.xyz", ["--view", "tilted", "--fps", "20", "--hold", "0", "--image-tint", "0.55"]),
    }
    for stem, (xyz, extra) in jobs.items():
        for name, bg in (("light", "#ffffff"), ("dark", GH_DARK_BG)):
            out = MEDIA / f"{stem}-{name}"
            args = [a if not (name == "dark" and a == "0.55") else "0.3" for a in extra]
            subprocess.run([PY, str(cap), str(xyz), str(out), "--width", "640", "--height", "400",
                            "--background", bg, *args], check=True)
            mp4 = out.with_suffix(".mp4")
            if mp4.exists():
                mp4.unlink()
    # web app recording: one GIF (the recording itself is light UI), 480 px, 8 fps
    vid = DATA / "webapp.mp4"
    out = MEDIA / "webapp.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(vid), "-vf",
                    "fps=6,scale=560:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=sierra2_4a",
                    "-loop", "0", str(out)], check=True)


# ---------------------------------------------------------------- README
def picture(stem, ext, alt, width=480):
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{media(stem + "-dark." + ext)}">\n'
            f'  <img src="{media(stem + "-light." + ext)}" alt="{alt}" width="{width}">\n</picture>')


def asrun_block(d, extra=()):
    rows = [d["command"]]
    if d.get("cwd"):
        rows.append(f"# folder: {d['cwd']}")
    rows += [f"# {e}" for e in extra]
    return "Exactly as run:\n\n```bash\n" + "\n".join(rows) + "\n```"


def asrun(d, extra=()):
    rows = [d["command"]]
    if d.get("cwd"):
        rows.append(f"# folder: {d['cwd']}")
    rows += [f"# {e}" for e in extra]
    return ("<details><summary>Exactly as run</summary>\n\n```bash\n" + "\n".join(rows) + "\n```\n\n</details>")


READABLE = [("sigma_max_free / fmax", "max force σ on free atoms / fmax"),
            ("sigma_max_free", "max force σ on free atoms"), ("sigma_mean_free", "mean force σ on free atoms")]


def readable(label):
    for a, b in READABLE:
        label = label.replace(a, b)
    return label


def numbers_table(uc, picks, imaginary_prefix=None):
    out = ["| Quantity | Value |", "|---|---|"]
    for p in picks:
        k = kn(uc, p)
        label = readable(re.sub(r"\s+\([^()]*\)$", "", k["label"]))
        out.append(f"| {label} | {num(k, imaginary=bool(imaginary_prefix and p.startswith(imaginary_prefix)))} |")
    return "\n".join(out)


def env_rows():
    body = YAML[YAML.index("members:"):]
    rows = []
    for block in re.split(r"\n\s*-\s+", body)[1:]:
        get = lambda k: (re.search(rf"{k}:\s*(\S+)", block) or [None, None])[1]
        env, mlip = get("env"), get("mlip")
        head = get("uma_task") or get("sevennet_task") or get("mace_head")
        pkg = "fairchem-core" if mlip.startswith("uma") else "sevenn" if mlip.startswith("7net") else "mace-torch"
        users = [n for k, n in (("relax", "relax"), ("neb", "NEB"), ("md", "bilayer relax"))
                 if P[k]["command"].startswith(env + "/bin/")]
        if P["webapp"]["target"]["python"].startswith(env + "/bin/"):
            users.append("web app job")
        users.append("committee member")
        rows.append(f"| `{mlip}` / `{head}` | {pkg} | {', '.join(users)} | `{env}` |")
    return "\n".join(rows)


def readme():
    R, C, N, M, W, MP = (P[k] for k in ("relax", "committee", "neb", "md", "webapp", "md_planned"))
    act = "source <your uma env>/bin/activate"
    yaml_members = re.sub(r"(env:\s+)(\S+)", lambda m: f"{m.group(1)}<your {m.group(2).rsplit('/', 1)[-1]} env>",
                          YAML[YAML.index("members:"):].rstrip())
    fT = kn("relax", "fmax target")
    slab = next(n for n in R["notes"] if n.startswith("Slab:"))
    energies = next(n for n in R["notes"] if n.startswith("Energies are"))
    md_note = next(n for n in M["notes"] if n.startswith("command is"))
    fI = kn("neb", "TS imaginary frequency")
    speed = kn("webapp", "video speed-up")
    s = f"""<!-- Generated by site/alt-4-handbook/build_readme.py from data/*.json. Do not edit by hand. -->
# mliprun

[![Tests](https://github.com/manuelarcer/mliprun/actions/workflows/tests.yml/badge.svg)](https://github.com/manuelarcer/mliprun/actions/workflows/tests.yml)
[![Fresh-install smoke test](https://github.com/manuelarcer/mliprun/actions/workflows/install-smoke.yml/badge.svg)](https://github.com/manuelarcer/mliprun/actions/workflows/install-smoke.yml)

**Handbook: set up mliprun on your own machines.** No preconfigured machines: you add your own.
Every result below is a real run (mliprun `{META['code_commit']}`, uma-s-1p2/oc20; showcase runs on
cos-cluster, one GPU card). Live 3D version: [landing page]({landing()}).

Install: see [Installation]({link('docs/INSTALL.md')}) and [AGENTS.md]({link('AGENTS.md')}).

## Five results at a glance

| Job | Command | Result | Status |
|---|---|---|---|
| [Relax an adsorbate](#relax-an-adsorbate) | `mlip optimize run` | final fmax {num(kn('relax', 'final fmax, O at fcc'))} (target {num(fT)}) | {status(R)} |
| [Relax with a committee](#relax-with-a-committee) | `mlip optimize run --committee` | max force σ on free atoms {num(kn('committee', 'sigma_max_free, final step'))} at the final step | {status(C)} |
| [Find and check a TS](#find-and-check-a-transition-state) | `mlip neb run --verify-ts` | barrier {num(kn('neb', 'forward barrier'))}, one imaginary mode {num(fI, True)} | {status(N)} |
| [Run MD](#run-md-pending) | `mlip md run` | MD not run; next step pending a decision | {status(M)} |
| [Send a job from the web app](#send-a-job-from-the-web-app) | `mlip webapp` | {num(kn('webapp', 'job wall time'))} wall time of the showcase job | {status(W)} |

## Setup

Do this once on each machine you use. The recipes assume it is done.

### Which machine

Your laptop, or a remote GPU machine you add yourself (password-free SSH, Linux, an environment
with mliprun). No scheduler? Check the GPU table and use one free card.

```bash
# on your GPU machine: which cards are busy?
nvidia-smi
export CUDA_VISIBLE_DEVICES=<free card>
```

### Which environment

One MLIP per Python environment, with mliprun installed in each.

```bash
git clone https://github.com/manuelarcer/mliprun.git
cd mliprun
python3 -m venv <your uma env>     # a folder you choose
source <your uma env>/bin/activate
pip install -e .
pip install fairchem-core      # UMA; one MLIP package in this env only
```

<details><summary>Environments the showcase runs used (provenance only, not for you to use)</summary>

| MLIP / head | Package | Used by | Where these runs happened |
|---|---|---|---|
{env_rows()}

</details>

UMA weights are gated on Hugging Face ([UMA guide]({link('docs/UMA_USAGE_GUIDE.md')})). Never
`pip install asetools` (the PyPI package is unrelated); use `pip install -e ".[neb]"`.
Per-MLIP recipes: [docs/install]({link('docs/install/README.md')}).

### Check the environment

```bash
source <your uma env>/bin/activate
mlip doctor        # versions, installed MLIPs, CUDA; non-zero exit if no MLIP
```

Always name the model and its head (`--mlip uma-s-1p2 --uma-task oc20`); each head has its own
energy zero.

### Add your machine in the web app

```bash
pip install -e ".[webapp]"     # on the laptop only
mlip webapp                    # opens http://127.0.0.1:8765/
```

1. **Machines → Add machine.** Name: any. Host: `<your ssh alias>` (from `~/.ssh/config`).
   Default remote location: `<your remote folder>`. Press **Add machine**.
2. **Add environment**, on the saved machine's card. Python path: `<your uma env>/bin/python`, plus a Name.
3. **Test connection**, on the saved machine (host and GPU table; it does not check the load).
4. **Run on.** In the job form, choose your environment; submit.

## Recipes

The command that ran; fill in every `<…>`.

### Relax an adsorbate

{picture('relax', 'gif', 'Relaxation of O on Pt(111): one frame per BFGS step of the real run', 640)}

O relaxes in the fcc hollow of Pt(111). {cap(status(R))}.

```bash
{recipe(R['command'], {'--structure': '<your POSCAR>'}, [act])}
```

{numbers_table('relax', ['final fmax, O at fcc', 'BFGS steps, O at fcc', 'total energy, O at fcc', 'O height above top Pt layer, fcc', 'Pt lattice constant'])}

<details><summary>Convergence chart, run details, exactly as run</summary>

{picture('relax-fmax', 'svg', 'fmax against BFGS step, log scale, with the fmax target', 560)}

{slab}

{asrun_block(R)}

</details>

### Relax with a committee

{picture('committee-force', 'svg', 'Committee force disagreement and committee-mean fmax against BFGS step, log scale', 640)}

Three MLIPs relax together; their disagreement is the uncertainty. {cap(status(C))}.

```bash
# members run in the envs named in committee.yaml
{recipe(C['command'], {'--structure': '<your POSCAR>'}, [act])}
```

```yaml
{yaml_members}
```

{numbers_table('committee', ['sigma_max_free, final step', 'sigma_max_free / fmax', 'final fmax of the committee-mean', 'energy spread across members, final step', 'BFGS steps'])}

<details><summary>Run details, exactly as run</summary>

Same start structure; all members at the OC20 (RPBE) level, each in its own environment.
`gpu: N` is the physical card for that member.

{asrun_block(C)}

</details>

### Find and check a transition state

{picture('neb-band', 'gif', 'Converged NEB band, image by image, O moving from the fcc to the hcp hollow', 640)}

O hops fcc → hcp on Pt(111); barrier {num(kn('neb', 'forward barrier'))}. {cap(status(N))}.

```bash
{act}
# NEB writes into the current folder
cd <neb run folder>
{recipe(N['command'], {'--initial': '<IS POSCAR>', '--final': '<FS POSCAR>'})}
```

{numbers_table('neb', ['forward barrier', 'reverse barrier', 'reaction energy', 'TS imaginary frequency', 'overlap of imaginary mode', 'final band fmax'], imaginary_prefix='TS imaginary')}

<details><summary>Energy profile, TS mode (mode, not dynamics), 1 flag, exactly as run</summary>

{picture('neb-profile', 'svg', 'Energy relative to the initial state against reaction coordinate, TS marked', 560)}

{picture('ts-mode', 'gif', 'TS imaginary mode; mode, not dynamics', 480)}

Climbing-image NEB with {flag_value(N['command'], '--num-images')} intermediate + 2 endpoint images ({len(N['series']['image_index'])} in all), then the
TS frequency check and a displace-and-relax check both ways. NEB defaults to CPU: pass `--device cuda`.

**Flag.** {N['notes'][0]}

{asrun_block(N)}

</details>

### Run MD (pending)

**MD not run; next step pending a decision.** Water bilayer on Pt(111), NVT. Planned command,
from `{MP['source']}`:

```bash
# planned: this command has NOT run yet
{recipe(MP['command'], {'--structure': '<relaxed bilayer POSCAR>'}, [act])}
```

<details><summary>Start structure, and what did run (the bilayer relaxation, not MD)</summary>

Start structure, from `md.json`: {M['structures']['relaxed_oc20']['label']}.

```bash
{M['command']}
# {md_note}
```

</details>

### Send a job from the web app

<img src="{media('webapp.gif')}" alt="Screen recording: the web app submits an optimize job to a remote GPU machine and shows the result" width="640">

Laptop submits, remote GPU machine runs; recording {num(kn('webapp', 'video length'))}, waiting part ×{fmt_value(speed['value'], 'dimensionless')}. {cap(status(W))}.

1. `mlip webapp` → **Optimize**, upload `<your POSCAR>`.
2. **Run on**: your environment ([add it first](#add-your-machine-in-the-web-app)).
3. Set model, head, optimizer and fmax as below; **Submit**.

```bash
# equivalent command, from reproduce.sh
{recipe(W['command'])}
```

{numbers_table('webapp', ['job wall time', 'relaxation wall time', 'BFGS steps', 'energy difference, web app'])}

<details><summary>Exactly as run</summary>

{asrun_block(W, [f"machine: {W['target']['machine']}, env: {W['target']['env']}"])}

</details>

## Other commands

| What | Command |
|---|---|
| Vibrational frequencies and ZPE | `mlip freq run --structure POSCAR --mlip uma-s-1p2 --uma-task oc20` |
| AutoNEB | `mlip autoneb run --initial IS.vasp --final FS.vasp --mlip uma-s-1p2 --uma-task oc20` |
| Benchmark installed MLIPs | `mlip benchmark run --structure POSCAR --uma-task oc20` |
| SevenNet (task required) | `--mlip 7net-omni --sevennet-task oc20` |
| CHGNet | `--mlip chgnet` |
| Check an environment | `mlip doctor` |

## Where the numbers come from

Every number is generated from the showcase data pack
([PROVENANCE.md]({link('docs/showcase/data/pack/PROVENANCE.md')})). Tool demonstrations, not
publication results. {energies}
"""
    (HERE / ("README.root.md" if ROOT else "README.md")).write_text(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-media", action="store_true")
    ap.add_argument("--root", action="store_true",
                    help="write README.root.md with paths relative to the repository root (Phase 5)")
    a = ap.parse_args()
    global ROOT
    ROOT = a.root
    MEDIA.mkdir(exist_ok=True)
    charts()
    if not a.no_media:
        gifs()
    readme()
    for f in sorted(MEDIA.iterdir()):
        print(f"{f.name:28s} {f.stat().st_size/1e6:6.2f} MB")


if __name__ == "__main__":
    main()
