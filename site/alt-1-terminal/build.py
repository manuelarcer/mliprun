"""Build the data folder, the README media (SVG) and README.md for alternative 1 (terminal).

Every number and every log line comes from the showcase data pack or the run logs:

  docs/showcase/data/pack/*.json, *.xyz, webapp.mp4   -> data/
  docs/showcase/data/runs/... (real logs, verbatim)    -> data/logs/
  docs/showcase/scripts/run_04_md.sh                   -> data/md_planned.json
                                                          (the MD command the run script
                                                          would issue; it has NOT run)

Then it writes, from those copies only:
  media/terminal-light.svg, media/terminal-dark.svg    animated SVG terminal (CSS only)
  media/committee-light.svg, media/committee-dark.svg  committee chart
  media/relax-light.svg, media/relax-dark.svg          relaxation chart
  README.md                                            the README variant

GIFs come from site/shared/capture.py (run with --capture; needs Chromium, so outside
the Claude Code sandbox) and from ffmpeg (webapp.gif).

Usage (from the repo root):
  site/.venv/bin/python site/alt-1-terminal/build.py            # data + SVG + README
  site/.venv/bin/python site/alt-1-terminal/build.py --capture  # also the GIFs
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PACK = REPO / "docs/showcase/data/pack"
RUNS = REPO / "docs/showcase/data/runs"
SCRIPTS = REPO / "docs/showcase/scripts"
DATA = HERE / "data"
MEDIA = HERE / "media"
PAGES_URL = "https://manuelarcer.github.io/mliprun/"  # landing page (Phase 5 location)
# Absolute links survive promotion of README.md to the repo root (ruling AB).
REPO_URL = "https://github.com/manuelarcer/mliprun/blob/main"
RELAX_VIEW = {"elevation": 38, "azimuth": -30, "zoom": 2.1}  # same camera as main.js RELAX_VIEW
MODE_ZOOM = 3.2  # TS-mode GIF zoom; main.js uses 4.5 for its smaller square inset (same framing idea)

PACK_FILES = ["relax.json", "committee.json", "neb.json", "md.json", "webapp.json",
              "relax.xyz", "ts_mode.xyz", "webapp.mp4",
              "PROVENANCE.md"]
LOGS = {  # data/logs/<name>  <-  runs/<path>   (copied verbatim, never edited)
    "neb_stdout.txt": "03_neb/neb_stdout.txt",
    "neb.log": "03_neb/neb.log",
    "neb_run.log": "03_neb/run.log",
    "ts_verify.txt": "03_neb/ts_verify/ts_verify.txt",
    "relax_opt.log": "01_relax/o_fcc/opt.log",
    "committee_stdout.txt": "02_committee/mlip_stdout.txt",
}


# ------------------------------------------------------------------ helpers
def elide(tok: str) -> str:
    """Shorten an absolute path to '…/' + its last three parts (same rule as main.js)."""
    if tok.startswith("/"):
        parts = [p for p in tok.split("/") if p]
        if len(parts) > 3:
            return "…/" + "/".join(parts[-3:])
    return tok


def command_lines(cmd: str, width: int = 58) -> list[str]:
    """Split a command into shell continuation lines: head, then flags packed to `width`."""
    toks = [elide(t) for t in cmd.split()]
    head, i = [], 0
    while i < len(toks) and not toks[i].startswith("--"):
        head.append(toks[i])
        i += 1
    groups = []
    while i < len(toks):
        g = [toks[i]]
        i += 1
        while i < len(toks) and not toks[i].startswith("--"):
            g.append(toks[i])
            i += 1
        groups.append(" ".join(g))
    lines, cur = [" ".join(head)], ""
    for g in groups:
        if cur and len(cur) + 1 + len(g) > width:
            lines.append(cur)
            cur = g
        else:
            cur = f"{cur} {g}".strip()
    if cur:
        lines.append(cur)
    out = []
    for k, ln in enumerate(lines):
        pre = "$ " if k == 0 else "    "
        out.append(pre + ln + (" \\" if k < len(lines) - 1 else ""))
    return out


def kn(d: dict, label: str) -> dict:
    """Key number by exact label prefix; fail loudly if the pack changed."""
    hits = [k for k in d["key_numbers"] if k["label"] == label] or \
        [k for k in d["key_numbers"] if k["label"].startswith(label)]
    if len(hits) != 1:
        raise KeyError(f"{d['use_case']}: {len(hits)} key numbers start with {label!r}")
    return hits[0]


def fmt(k: dict, nd: int) -> str:
    return f"{k['value']:.{nd}f} {k['unit']}"


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (m * step) <= n:
            step *= m
            break
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


def hms(t: str) -> int:
    h, m, x = (int(v) for v in t.split(":"))
    return h * 3600 + m * 60 + x


def dur(sec: int) -> str:
    return f"{sec // 60} min {sec % 60} s" if sec >= 60 else f"{sec} s"


# ------------------------------------------------------------------ data
def copy_data() -> None:
    if DATA.exists():
        shutil.rmtree(DATA)  # only what the page reads; stale copies never linger
    (DATA / "logs").mkdir(parents=True, exist_ok=True)
    for f in PACK_FILES:
        shutil.copy2(PACK / f, DATA / f)
    for name, src in LOGS.items():
        shutil.copy2(RUNS / src, DATA / "logs" / name)
    DATA.joinpath("md_planned.json").write_text(json.dumps(md_planned(), indent=1) + "\n")
    centre_on_adsorbate(DATA / "ts_mode.xyz", DATA / "ts_mode_centred.xyz", "O")
    centre_on_adsorbate(DATA / "relax.xyz", DATA / "relax_centred.xyz", "O")


def centre_on_adsorbate(src: Path, dst: Path, elem: str) -> None:
    """Rigid in-plane translation of a periodic slab so `elem` (frame 0) sits at the cell centre.

    A periodic system looks the same from any origin; this only moves the origin so that the
    zoomed TS-mode views (page and README) show the O in the middle of the middle periodic
    image. Atoms are wrapped back into the cell with the wrap decided on frame 0 and applied
    unchanged to every frame, so no atom jumps between frames. z is untouched.
    """
    lines = src.read_text().splitlines()
    frames, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        frames.append((lines[i + 1], [ln.split() for ln in lines[i + 2:i + 2 + n]]))
        i += 2 + n
    lat = [float(v) for v in re.search(r'Lattice="([^"]+)"', frames[0][0]).group(1).split()]
    a, b = (lat[0], lat[1]), (lat[3], lat[4])
    det = a[0] * b[1] - a[1] * b[0]

    def frac(x, y):
        return ((x * b[1] - y * b[0]) / det, (a[0] * y - a[1] * x) / det)

    ref = next(r for r in frames[0][1] if r[0] == elem)
    fo = frac(float(ref[1]), float(ref[2]))
    shift = (0.5 - fo[0], 0.5 - fo[1])
    wraps = []
    for r in frames[0][1]:
        f = frac(float(r[1]), float(r[2]))
        wraps.append((math.floor(f[0] + shift[0]), math.floor(f[1] + shift[1])))
    out = []
    for comment, rows in frames:
        # only species and positions are written, so the header must say so
        out += [str(len(rows)), re.sub(r"Properties=\S+", "Properties=species:S:1:pos:R:3", comment)]
        for r, (wa, wb) in zip(rows, wraps):
            f = frac(float(r[1]), float(r[2]))
            u, v = f[0] + shift[0] - wa, f[1] + shift[1] - wb
            x, y = u * a[0] + v * b[0], u * a[1] + v * b[1]
            out.append(f"{r[0]:<2} {x:16.8f} {y:16.8f} {float(r[3]):16.8f}")
    dst.write_text("\n".join(out) + "\n")


def md_planned() -> dict:
    """Read the MD command the stage-04 run script would issue. It has not run."""
    src = SCRIPTS / "run_04_md.sh"
    text = src.read_text()

    def arr(name: str) -> list[str]:
        m = re.search(rf"^{name}=\((.*?)\)", text, re.S | re.M)
        return m.group(1).split()

    steps = re.search(r"^TARGET_STEPS=(\d+)", text, re.M).group(1)
    md = json.loads((PACK / "md.json").read_text())
    mlip_bin = md["command"].split()[0]                      # same env as the relaxation
    remote = md["command"].split("--structure ")[1].split("/04_md/")[0]
    structure = f"{remote}/04_md/md/POSCAR"
    cmd = " ".join([mlip_bin, "md", "run", "--structure", structure,
                    *arr("MODEL"), *arr("MD_FLAGS"), "--steps", steps])
    return {
        "status": "planned, not run",
        "command": cmd,
        "source": "docs/showcase/scripts/run_04_md.sh (MODEL, MD_FLAGS, TARGET_STEPS)",
        "note": "Built from the stage-04 run script. The MD has not run; md.json says pending.",
    }


# ------------------------------------------------------------------ SVG media
THEMES = {
    "light": dict(bg="#ffffff", panel="#f6f8fa", ink="#1b2430", muted="#5b6775",
                  rule="#d0d7de", accent="#b3302a", ok="#1f7a4a", prompt="#3b5bdb",
                  grid="#e6eaee", band="#b3302a22"),
    "dark": dict(bg="#0d1117", panel="#11161d", ink="#e2e8ee", muted="#94a0ad",
                 rule="#30363d", accent="#f2705f", ok="#5cc28a", prompt="#8aa2ff",
                 grid="#1f262e", band="#f2705f2e"),
}
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def neb_transcript() -> tuple[list[tuple[str, str]], int]:
    """(kind, text) lines for the README terminal; all text is from the real logs."""
    neb = json.loads((DATA / "neb.json").read_text())
    stdout = (DATA / "logs/neb_stdout.txt").read_text().splitlines()
    steps = [ln for ln in (DATA / "logs/neb.log").read_text().splitlines()
             if ln.startswith("Step")]
    every = 9
    pick = [s for i, s in enumerate(steps) if i % every == 0 or i == len(steps) - 1]
    out = [("cmd", ln) for ln in command_lines(neb["command"], 60)]
    out.append(("out", next(ln for ln in stdout if ln.startswith("Using MLIP")).strip()))
    runlog = (DATA / "logs/neb_run.log").read_text().splitlines()
    fire = [ln for ln in (DATA / "logs/neb.log").read_text().splitlines() if ln.startswith("FIRE:")]
    t0 = hms(re.search(r"T(\d\d:\d\d:\d\d)", next(l for l in runlog if "neb run (" in l)).group(1))
    gap = hms(fire[0].split()[2]) - t0
    out.append(("note", f"# not shown: {dur(gap)} of model load, endpoint relaxation and IDPP"))
    out.append(("out", next(ln for ln in stdout if "Running NEB" in ln).strip()))
    out.append(("note", f"# neb.log, every {every}th of {len(steps)} Step lines"))
    out += [("step", s) for s in pick]
    t_exit = hms(re.search(r"T(\d\d:\d\d:\d\d)", next(l for l in runlog if "neb run exited" in l)).group(1))
    gap2 = t_exit - hms(fire[-1].split()[2])
    out.append(("note", f"# not shown: {dur(gap2)} (results, TS verification)"))
    i = next(k for k, ln in enumerate(stdout) if ln.startswith("TS verification:"))
    for ln in stdout[i:]:
        if not ln.strip() or ln.lstrip().startswith("📄"):
            continue
        if ln.strip().startswith(("Reason",)):
            continue
        out.append(("ts", ln.strip()))
    return out, len(steps)


def profile_svg(t: dict, x0: float, y0: float, w: float, h: float, anim: bool,
                t_start: float) -> str:
    """NEB image-energy profile with axes and the barrier annotated."""
    neb = json.loads((DATA / "neb.json").read_text())
    xs = neb["series"]["reaction_coordinate_A"]
    ys = neb["series"]["relative_energy_eV"]
    ts = neb["ts_image_index"]
    fwd = kn(neb, "forward barrier")
    dE = kn(neb, "reaction energy")
    ml, mb, mt, mr = 44, 40, 14, 12
    pw, ph = w - ml - mr, h - mb - mt
    xmax = math.ceil(max(xs) * 2) / 2
    ymin, ymax = -0.1, math.ceil(max(ys) * 10) / 10 + 0.1
    X = lambda v: x0 + ml + v / xmax * pw
    Y = lambda v: y0 + mt + (ymax - v) / (ymax - ymin) * ph
    s = [f'<g font-family="{SANS}" font-size="11" fill="{t["muted"]}">']
    for v in nice_ticks(ymin, ymax, 5):
        s.append(f'<line x1="{X(0):.1f}" x2="{X(xmax):.1f}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')
        s.append(f'<text x="{X(0) - 6:.1f}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
    for v in nice_ticks(0, xmax, 4):
        s.append(f'<text x="{X(v):.1f}" y="{Y(ymin) + 16:.1f}" text-anchor="middle">{v:g}</text>')
    s.append(f'<line x1="{X(0):.1f}" x2="{X(xmax):.1f}" y1="{Y(ymin):.1f}" y2="{Y(ymin):.1f}" '
             f'stroke="{t["rule"]}"/>')
    s.append(f'<text x="{X(xmax / 2):.1f}" y="{y0 + h - 4:.1f}" text-anchor="middle">'
             f'reaction coordinate (Å)</text>')
    s.append(f'<text transform="translate({x0 + 11:.1f} {Y((ymin + ymax) / 2):.1f}) rotate(-90)" '
             f'text-anchor="middle">E − E<tspan font-size="8" dy="2">IS</tspan>'
             f'<tspan dy="-2"> (eV)</tspan></text>')
    s.append("</g>")
    pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs, ys))
    plen = sum(math.hypot(X(xs[i + 1]) - X(xs[i]), Y(ys[i + 1]) - Y(ys[i]))
               for i in range(len(xs) - 1))
    cls = ' class="draw"' if anim else ""
    sty = (f' style="stroke-dasharray:{plen + 4:.0f};animation-delay:{t_start:.2f}s"'
           if anim else "")
    s.append(f'<polyline{cls}{sty} points="{pts}" fill="none" stroke="{t["accent"]}" '
             f'stroke-width="2" stroke-linejoin="round"/>')
    g = f' class="pop" style="animation-delay:{t_start + 1.2:.2f}s"' if anim else ""
    s.append(f"<g{g}>")
    for i, (a, b) in enumerate(zip(xs, ys)):
        r = 4.5 if i == ts else 3
        s.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="{r}" fill="'
                 f'{t["accent"] if i == ts else t["bg"]}" stroke="{t["accent"]}" stroke-width="1.6"/>')
    # barrier bracket at the TS
    xb = X(xs[ts])
    s.append(f'<line x1="{X(0):.1f}" x2="{xb:.1f}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" '
             f'stroke="{t["muted"]}" stroke-dasharray="3 3"/>')
    s.append(f'<line x1="{xb:.1f}" x2="{xb:.1f}" y1="{Y(0):.1f}" y2="{Y(ys[ts]) + 6:.1f}" '
             f'stroke="{t["ink"]}" stroke-width="1"/>')
    s.append(f'<text x="{xb - 6:.1f}" y="{Y(ys[ts] / 2):.1f}" text-anchor="end" '
             f'font-family="{SANS}" font-size="12" fill="{t["ink"]}">E<tspan font-size="9" dy="3">a</tspan>'
             f'<tspan dy="-3"> = {fwd["value"]:.3f} {fwd["unit"]}</tspan></text>')
    s.append(f'<text x="{X(xs[-1]):.1f}" y="{Y(ys[-1]) + 18:.1f}" text-anchor="end" '
             f'font-family="{SANS}" font-size="12" fill="{t["ink"]}">ΔE = +{dE["value"]:.3f} {dE["unit"]}</text>')
    s.append(f'<text x="{X(0) + 2:.1f}" y="{Y(0) + 16:.1f}" font-family="{SANS}" font-size="11" '
             f'fill="{t["muted"]}">O fcc</text>')
    s.append(f'<text x="{X(xs[-1]):.1f}" y="{Y(ys[-1]) + 34:.1f}" text-anchor="end" '
             f'font-family="{SANS}" font-size="11" fill="{t["muted"]}">O hcp</text>')
    s.append("</g>")
    return "\n".join(s)


def terminal_svg(theme: str) -> str:
    t = THEMES[theme]
    lines, nsteps = neb_transcript()
    W, LH, top = 960, 17, 46
    TW = 590
    H = top + LH * len(lines) + 24
    body, tcur = [], 0.4
    for k, (kind, text) in enumerate(lines):
        y = top + LH * (k + 1)
        color = {"cmd": t["ink"], "out": t["muted"], "note": t["muted"], "step": t["ink"],
                 "ts": t["ok"] if "verified" in text or "connects" in text or "pass" in text
                 else t["ink"]}[kind]
        style = ' font-style="italic"' if kind == "note" else ""
        if kind == "cmd":
            n = len(text)
            dur = n * 0.011
            body.append(
                f'<g><text x="18" y="{y}" fill="{color}" xml:space="preserve">{escape(text)}</text>'
                f'<rect class="type" x="18" y="{y - 13}" width="{TW - 30}" height="{LH}" '
                f'fill="{t["panel"]}" style="animation-duration:{dur:.2f}s;'
                f'animation-delay:{tcur:.2f}s;animation-timing-function:steps({n})"/></g>')
            tcur += dur + 0.05
        else:
            step = 0.08 if kind == "step" else 0.06
            if kind == "out" and lines[k - 1][0] == "cmd":
                tcur += 0.25
            if kind == "ts" and lines[k - 1][0] != "ts":
                tcur += 0.25
            body.append(f'<text class="ln" x="18" y="{y}" fill="{color}"{style} '
                        f'xml:space="preserve" style="animation-delay:{tcur:.2f}s">'
                        f'{escape(text)}</text>')
            tcur += step
    prof = profile_svg(t, TW + 20, 40, W - TW - 30, min(H - 60, 330), True, tcur - 1.0)
    css = f"""
  .ln, .pop {{ animation: show .01s step-end backwards; }}
  .type {{ transform-box: fill-box; transform-origin: right center;
           animation-name: type; animation-fill-mode: backwards; transform: scaleX(0); }}
  .draw {{ animation: draw 1.6s ease-out backwards; stroke-dashoffset: 0; }}
  @keyframes show {{ from {{ opacity: 0 }} to {{ opacity: 1 }} }}
  @keyframes type {{ from {{ transform: scaleX(1) }} to {{ transform: scaleX(0) }} }}
  @keyframes draw {{ from {{ stroke-dashoffset: var(--len, 1200) }} to {{ stroke-dashoffset: 0 }} }}
  @media (prefers-reduced-motion: reduce) {{ .ln, .pop, .type, .draw {{ animation: none !important; }} }}
"""
    # stroke-dashoffset start must equal the path length; set via inline var on the polyline
    prof = prof.replace('class="draw" style="stroke-dasharray:',
                        'class="draw" style="--len:9999;stroke-dasharray:', 1)
    m = re.search(r"stroke-dasharray:(\d+)", prof)
    prof = prof.replace("--len:9999", f"--len:{m.group(1)}", 1)
    neb = json.loads((DATA / "neb.json").read_text())
    head = elide(neb["cwd"])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Animated terminal: mlip neb run with --verify-ts both on O fcc to hcp on Pt(111); real log lines and the resulting image-energy profile">
<style>{css}</style>
<rect width="{W}" height="{H}" rx="10" fill="{t['bg']}" stroke="{t['rule']}"/>
<rect x="8" y="8" width="{TW - 8}" height="{H - 16}" rx="7" fill="{t['panel']}" stroke="{t['rule']}"/>
<text x="18" y="28" font-family="{SANS}" font-size="11.5" fill="{t['muted']}">ran on cos-cluster  {escape(head)}   ·   real output, excerpted</text>
<line x1="8" x2="{TW}" y1="38" y2="38" stroke="{t['rule']}"/>
<g font-family="{MONO}" font-size="11.6">
{chr(10).join(body)}
</g>
<text x="{TW + 20}" y="28" font-family="{SANS}" font-size="12" font-weight="600" fill="{t['ink']}">Image energies, final band ({len(neb['series']['image_index'])} images)</text>
{prof}
</svg>
"""


def line_chart_svg(theme: str, title: str, series: list[dict], xlabel: str, ylabel: str,
                   logy: bool, hline: dict | None, w: int = 520, h: int = 260) -> str:
    """Small static chart: series = [{x, y, color_key, label, dash}]."""
    t = THEMES[theme]
    ml, mr, mt, mb = 58, 16, 30, 40
    pw, ph = w - ml - mr, h - mt - mb
    xs = [v for s in series for v in s["x"]]
    ys = [v for s in series for v in s["y"]] + ([hline["y"]] if hline else [])
    xmin, xmax = min(xs), max(xs)
    if logy:
        lo, hi = math.floor(math.log10(min(ys))), math.log10(max(ys) * 1.4)
        T = lambda v: math.log10(v)
        ticks = [10 ** k for k in range(lo, math.floor(hi) + 1)]
    else:
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        T = lambda v: v
        ticks = nice_ticks(lo, hi, 4)
    X = lambda v: ml + (v - xmin) / (xmax - xmin) * pw
    Y = lambda v: mt + (hi - T(v)) / (hi - lo) * ph
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}">',
         f'<rect width="{w}" height="{h}" rx="8" fill="{t["bg"]}" stroke="{t["rule"]}"/>',
         f'<g font-family="{SANS}" font-size="11" fill="{t["muted"]}">']
    for v in ticks:
        lab = f"{v:g}"
        o.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{t["grid"]}"/>')
        o.append(f'<text x="{ml - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{lab}</text>')
    for v in nice_ticks(xmin, xmax, 6):
        o.append(f'<text x="{X(v):.1f}" y="{mt + ph + 16}" text-anchor="middle">{v:g}</text>')
    o.append(f'<text x="{ml + pw / 2}" y="{h - 6}" text-anchor="middle">{escape(xlabel)}</text>')
    o.append(f'<text transform="translate(13 {mt + ph / 2}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>')
    o.append(f'<text x="{ml}" y="18" fill="{t["ink"]}" font-size="12" font-weight="600">{escape(title)}</text>')
    o.append("</g>")
    if hline:
        o.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(hline["y"]):.1f}" y2="{Y(hline["y"]):.1f}" '
                 f'stroke="{t["muted"]}" stroke-dasharray="4 3"/>')
        o.append(f'<text x="{ml + 6}" y="{Y(hline["y"]) + 14:.1f}" '
                 f'font-family="{SANS}" font-size="11" fill="{t["muted"]}">{escape(hline["label"])}</text>')
    lx = ml + 8
    for s in series:
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(s["x"], s["y"]))
        col = t[s["color"]]
        dash = ' stroke-dasharray="5 3"' if s.get("dash") else ""
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"{dash}/>')
        for a, b in zip(s["x"], s["y"]):
            o.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="2.4" fill="{col}"/>')
        o.append(f'<line x1="{lx}" x2="{lx + 18}" y1="{mt + 8}" y2="{mt + 8}" stroke="{col}" stroke-width="2"{dash}/>')
        o.append(f'<text x="{lx + 22}" y="{mt + 12}" font-family="{SANS}" font-size="11" fill="{t["ink"]}">{escape(s["label"])}</text>')
        lx += 30 + 6.3 * len(s["label"])
    o.append("</svg>\n")
    return "\n".join(o)


def write_media() -> None:
    MEDIA.mkdir(exist_ok=True)
    rel = json.loads((DATA / "relax.json").read_text())
    com = json.loads((DATA / "committee.json").read_text())
    for th in THEMES:
        (MEDIA / f"terminal-{th}.svg").write_text(terminal_svg(th))
        c = com["series"]
        (MEDIA / f"committee-{th}.svg").write_text(line_chart_svg(
            th, "Committee: force being minimized vs. member disagreement",
            [dict(x=c["step"], y=c["fmax_eV_per_A"], color="ink", label="fmax of mean force"),
             dict(x=c["step"], y=c["sigma_max_free_eV_per_A"], color="accent",
                  label="σ_max (free atoms)", dash=True)],
            "BFGS step", "force (eV/Å)", True,
            dict(y=kn(com, "fmax target")["value"],
                 label=f"target {kn(com, 'fmax target')['value']:g} eV/Å")))


# ------------------------------------------------------------------ README
def md_status_text(md: dict) -> str:
    """Ruling AC wording, bound to md.json flags."""
    parts = []
    if "md_not_run" in md.get("flags", []):
        parts.append("MD not run")
    if "pending_scientist_decision" in md.get("flags", []):
        parts.append("next step pending a decision")
    return "; ".join(parts) or md.get("status", "")



def picture(stem: str, alt: str, width: int | None = None) -> str:
    w = f' width="{width}"' if width else ""
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="media/{stem}-dark.{{ext}}">\n'
            f'  <img src="media/{stem}-light.{{ext}}" alt="{escape(alt)}"{w}>\n</picture>')


def readme() -> str:
    rel = json.loads((DATA / "relax.json").read_text())
    com = json.loads((DATA / "committee.json").read_text())
    neb = json.loads((DATA / "neb.json").read_text())
    md = json.loads((DATA / "md.json").read_text())
    web = json.loads((DATA / "webapp.json").read_text())
    plan = json.loads((DATA / "md_planned.json").read_text())

    def cmd(c: str) -> str:
        return "```bash\n" + "\n".join(command_lines(c, 64)).replace("$ ", "", 1) + "\n```"

    def conv(d: dict) -> str:
        return "converged" if d["converged"] else "not converged"

    n_step_lines = sum(1 for ln in (DATA / "logs/neb.log").read_text().splitlines()
                       if ln.startswith("Step"))
    md_state = md_status_text(md)
    sig_ratio = kn(com, "sigma_max_free / fmax")["value"]
    members = [n.split("member ")[1].split(" (")[0] for n in
               (k["label"] for k in com["key_numbers"]) if n.startswith("final energy, member")]
    vid_speed = kn(web, "video speed-up")["value"]
    ts_mode = kn(neb, "TS mode animation peak displacement")
    term = picture("terminal", "Terminal replay of mlip neb run --verify-ts both: the command, "
                   "excerpted neb.log steps, the TS verification verdict, and the final image-energy "
                   "profile", None).replace("{ext}", "svg")
    com_pic = picture("committee", "Committee relaxation: fmax of the mean force and the largest "
                      "per-atom force disagreement per step, log scale", 640).replace("{ext}", "svg")
    mode_pic = picture("ts-mode", "The imaginary mode at the transition state, drawn as a "
                       "displacement of 0.3 Å peak (an illustration of the mode, not dynamics)",
                       520).replace("{ext}", "gif")
    relax_gif = picture("relax", "The 13 BFGS steps of O relaxing in the fcc hollow on "
                        "Pt(111)", 520).replace("{ext}", "gif").replace("relax-", "relax3d-")
    return f"""<!-- Generated by site/alt-1-terminal/build.py from docs/showcase/data/pack. Do not edit by hand. -->
# mliprun

[![Tests](https://github.com/manuelarcer/mliprun/actions/workflows/tests.yml/badge.svg)](https://github.com/manuelarcer/mliprun/actions/workflows/tests.yml)
[![Fresh-install smoke test](https://github.com/manuelarcer/mliprun/actions/workflows/install-smoke.yml/badge.svg)](https://github.com/manuelarcer/mliprun/actions/workflows/install-smoke.yml)

Machine-learning interatomic potentials (UMA, MACE, SevenNet, CHGNet), one command at a time.
Real runs of O on Pt(111), uma-s-1p2/oc20. **[Live page]({PAGES_URL})**

{term}

<sub>Real log, excerpted and sped up. Runs ran on cos-cluster; elided paths are where they ran.</sub>

---

### `mlip optimize run` · {conv(rel)}

{relax_gif}

E {fmt(kn(rel, 'total energy, O at fcc'), 3).replace('-', '−', 1)} · fmax {fmt(kn(rel, 'final fmax'), 4)} · {int(kn(rel, 'BFGS steps')['value'])} BFGS steps · 3×3 periodic images

{cmd(rel['command'])}

### `mlip optimize run --committee` · {conv(com)}

{com_pic}

σ max {fmt(kn(com, 'sigma_max_free, final step'), 4)}, {sig_ratio:.1f}× the final fmax. Members: {', '.join('`' + m + '`' for m in members)}.

{cmd(com['command'])}

### `mlip neb run --verify-ts both` · {conv(neb)}, TS verified

{mode_pic}

Barrier {fmt(kn(neb, 'forward barrier'), 3)} · imaginary mode {kn(neb, 'TS imaginary frequency')['value']:.1f}i {kn(neb, 'TS imaginary frequency')['unit']} (mode, not dynamics)

<details><summary>Flag</summary>

{neb['notes'][0]}

</details>

### `mlip md run` · {md_state}

{cmd(plan['command'])}

Planned command from the run script; no output yet.

### `mlip webapp` · {conv(web)}

<a href="media/webapp.mp4"><img src="media/webapp.gif" alt="Screen recording: a relaxation submitted from the mliprun web app to a remote GPU machine" width="760"></a>

The same relaxation from a browser (here sent to cos-cluster; waiting part ×{vid_speed:g}). No machines are preconfigured; you add your own.

<details><summary>Add your machine</summary>

1. Run `mlip webapp` on your laptop.
2. Machines → Add machine: Name, Host `<your ssh alias>` (an alias from `~/.ssh/config`),
   Default remote location `<your remote folder>`. Press Add machine.
3. On the saved machine: Add environment, Python path `<your uma env>/bin/python`, and a name.
4. Test connection on the saved machine, then submit a job to it.

</details>

---

**Also:** `mlip freq run` · `mlip autoneb run` · `mlip benchmark run` · SevenNet, CHGNet · `mlip doctor`

### Installation

One MLIP per environment; full recipe in [docs/INSTALL.md]({REPO_URL}/docs/INSTALL.md) and [AGENTS.md]({REPO_URL}/AGENTS.md).

```bash
git clone https://github.com/manuelarcer/mliprun.git && cd mliprun
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[neb]"
pip install mace-torch        # or fairchem-core (UMA, gated), sevenn, chgnet
mlip doctor
```

<sub>[Provenance]({REPO_URL}/docs/showcase/data/pack/PROVENANCE.md) · generated by `site/alt-1-terminal/build.py`.</sub>
"""


# ------------------------------------------------------------------ capture
def capture() -> None:
    py = REPO / "site/.venv/bin/python"
    cap = REPO / "site/shared/capture.py"
    bgs = {"light": "#ffffff", "dark": "#0d1117"}
    for th, bg in bgs.items():
        subprocess.run([py, cap, DATA / "ts_mode_centred.xyz", MEDIA / f"ts-mode-{th}",
                        "--width", "520", "--height", "360", "--replicate", "3", "3",
                        "--view", "top", "--elevation", "90", "--azimuth", "-30",
                        "--zoom", str(MODE_ZOOM), "--fps", "20", "--hold", "0",
                        "--background", bg], check=True)
        subprocess.run([py, cap, DATA / "relax_centred.xyz", MEDIA / f"relax3d-{th}",
                        "--width", "520", "--height", "340", "--replicate", "3", "3",
                        "--view", "tilted", "--elevation", str(RELAX_VIEW["elevation"]),
                        "--azimuth", str(RELAX_VIEW["azimuth"]), "--zoom", str(RELAX_VIEW["zoom"]),
                        "--fps", "6", "--hold", "1.5",
                        "--background", bg], check=True)
    for f in MEDIA.glob("*.mp4"):
        f.unlink()  # README uses the GIFs; the page uses the live viewer
    ff = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    shutil.copy2(DATA / "webapp.mp4", MEDIA / "webapp.mp4")
    subprocess.run([ff, "-v", "error", "-y", "-i", DATA / "webapp.mp4", "-vf",
                    "fps=6,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=4", "-loop", "0",
                    MEDIA / "webapp.gif"], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    a = ap.parse_args()
    copy_data()
    write_media()
    if a.capture:
        capture()
    (HERE / "README.md").write_text(readme())
    for f in sorted(MEDIA.iterdir()):
        print(f"{f.stat().st_size / 1e6:6.2f} MB  {f.relative_to(HERE)}")


if __name__ == "__main__":
    sys.exit(main())
