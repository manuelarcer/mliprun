"""Build README.md and its SVG charts for alternative 2 ("story") from the data pack.

    python3 site/alt-2-story/build_readme.py

Every number in the README is formatted here from data/*.json (copied from
docs/showcase/data/pack/); none is typed by hand. Charts are plain SVG, written once
for GitHub light and once for GitHub dark, and used through <picture>. 3D media come
from render_media.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MEDIA = HERE / "media"
LANDING = "https://manuelarcer.github.io/mliprun/"  # GitHub Pages target (Phase 5)
REPO = "https://github.com/manuelarcer/mliprun"

THEMES = {
    "light": dict(ink="#1f2328", muted="#59636e", grid="#d1d9e0", oxy="#b8352a",
                  cool="#2b6a91", band="rgba(43,106,145,0.20)", bar="#d1d9e0"),
    "dark": dict(ink="#e6edf3", muted="#9198a1", grid="#30363d", oxy="#ff7466",
                 cool="#79b8de", band="rgba(121,184,222,0.25)", bar="#3d444d"),
}
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"

P = {n: json.loads((DATA / f"{n}.json").read_text()) for n in
     ("relax", "committee", "neb", "md", "webapp")}


# ------------------------------------------------------------------ numbers
def kn(name, label):
    for k in P[name]["key_numbers"]:
        if k["label"] == label or k["label"].startswith(label):
            return k
    raise KeyError(f"{name}: {label}")


def fmt(v, dp):
    s = f"{abs(v):.{dp}f}"
    return ("−" if v < 0 and float(s) != 0 else "") + s


def sci(v, dp=1):
    e = math.floor(math.log10(abs(v)))
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{fmt(v / 10 ** e, dp)} × 10{str(e).translate(sup)}"


def q(name, label, dp, unit=None):
    k = kn(name, label)
    u = k["unit"] if unit is None else unit
    v = k["value"]
    s = sci(v) if 0 < abs(v) < 10 ** -max(dp, 3) else fmt(v, dp)
    return s if u in ("", "dimensionless") else f"{s} {u}"


def short_cmd(cmd):
    toks = cmd.split()
    toks = [t.split("/")[-1] if (i == 0 or t.startswith("/")) else t for i, t in enumerate(toks)]
    lines, cur = [], []
    for t in toks:
        if t.startswith("--") and cur:
            lines.append(" ".join(cur))
            cur = []
        cur.append(t)
    lines.append(" ".join(cur))
    return " \\\n".join(("  " if i else "") + l for i, l in enumerate(lines))


def state(d):
    if d.get("status") == "pending":
        return "pending"
    return "converged" if d.get("converged") else "not converged"


# ------------------------------------------------------------------ SVG helpers
class Svg:
    def __init__(self, w, h, t):
        self.w, self.h, self.t, self.parts = w, h, t, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, fill=None, anchor="start", size=12):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill or self.t["muted"]}" '
                 f'font-size="{size}" text-anchor="{anchor}">{s}</text>')

    def line(self, x1, y1, x2, y2, stroke, w=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke}" stroke-width="{w}"{d}/>')

    def path(self, pts, stroke, w=1.8, fill="none", close=False):
        d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + ("Z" if close else "")
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" '
                 f'stroke-linejoin="round"/>')

    def dot(self, x, y, fill, r=3):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>')

    def save(self, path, title):
        body = "\n".join(self.parts)
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}" font-family="{FONT}" role="img">'
            f"<title>{title}</title>\n{body}\n</svg>\n")


def panel(s, x0, y0, w, h, xd, yd, xticks, yticks, xlabel, ylabel, ylog=False, xdp=0, ydp=1):
    t = s.t
    X = lambda v: x0 + (v - xd[0]) / (xd[1] - xd[0]) * w  # noqa: E731
    if ylog:
        a, b = math.log10(yd[0]), math.log10(yd[1])
        Y = lambda v: y0 + h - (math.log10(max(v, yd[0])) - a) / (b - a) * h  # noqa: E731
    else:
        Y = lambda v: y0 + h - (v - yd[0]) / (yd[1] - yd[0]) * h  # noqa: E731
    for v in yticks:
        s.line(x0, Y(v), x0 + w, Y(v), t["grid"])
        s.text(x0 - 6, Y(v) + 4, (f"{v:g}" if ylog else fmt(v, ydp)), anchor="end", size=11)
    for v in xticks:
        s.line(X(v), y0 + h, X(v), y0 + h + 4, t["muted"])
        s.text(X(v), y0 + h + 16, fmt(v, xdp), anchor="middle", size=11)
    s.add(f'<path d="M{x0},{y0}V{y0 + h}H{x0 + w}" fill="none" stroke="{t["muted"]}"/>')
    s.text(x0 + w, y0 + h + 31, xlabel, fill=t["ink"], anchor="end")
    s.text(x0 - 44, y0 - 8, ylabel, fill=t["ink"])
    return X, Y


def chart_relax(theme):
    t = THEMES[theme]
    s = Svg(720, 250, t)
    r = P["relax"]["series"]
    e0 = r["energy_eV"][0]
    dE = [e - e0 for e in r["energy_eV"]]
    n = len(r["step"]) - 1
    X, Y = panel(s, 56, 26, 270, 180, (0, n), (min(dE) * 1.1, 0.02), [0, 4, 8, 12],
                 [0, -0.1, -0.2], "BFGS step", "E − E₀ (eV)")
    s.path([(X(a), Y(b)) for a, b in zip(r["step"], dE)], t["ink"])
    for a, b in zip(r["step"], dE):
        s.dot(X(a), Y(b), t["ink"], 2.6)
    X, Y = panel(s, 420, 26, 280, 180, (0, n), (0.005, 3), [0, 4, 8, 12], [0.01, 0.1, 1],
                 "BFGS step", "fmax (eV/Å)", ylog=True)
    tgt = kn("relax", "fmax target")["value"]
    s.line(420, Y(tgt), 700, Y(tgt), t["oxy"], dash="5 4")
    s.text(424, Y(tgt) + 15, f"target {fmt(tgt, 2)}", fill=t["oxy"], size=11)
    s.path([(X(a), Y(b)) for a, b in zip(r["step"], r["fmax_eV_per_A"])], t["oxy"])
    for a, b in zip(r["step"], r["fmax_eV_per_A"]):
        s.dot(X(a), Y(b), t["oxy"], 2.6)
    s.save(MEDIA / f"chart-relax-{theme}.svg", "Relaxation: energy and fmax per BFGS step")


def chart_committee(theme):
    t = THEMES[theme]
    s = Svg(720, 250, t)
    c = P["committee"]["series"]
    n = len(c["step"]) - 1
    floor = 0.005
    fm, sg = c["fmax_eV_per_A"], c["sigma_max_free_eV_per_A"]
    X, Y = panel(s, 56, 26, 290, 180, (0, n), (floor, 3), [0, 5, 10], [0.01, 0.1, 1],
                 "BFGS step", "force (eV/Å)", ylog=True)
    top = [(X(a), Y(g)) for a, g in zip(c["step"], sg)]
    s.path(top + [(X(c["step"][-1]), Y(floor)), (X(0), Y(floor))], "none", w=0, fill=t["band"], close=True)
    s.path(top, t["cool"])
    s.path([(X(a), Y(f)) for a, f in zip(c["step"], fm)], t["oxy"])
    tgt = kn("committee", "fmax target")["value"]
    s.line(56, Y(tgt), 346, Y(tgt), t["oxy"], dash="5 4")
    s.text(60, Y(tgt) + 15, f"target {fmt(tgt, 2)}", fill=t["oxy"], size=11)
    s.text(62, Y(3) + 14, "fmax", fill=t["oxy"], size=12)
    s.text(100, Y(3) + 14, "σmax", fill=t["cool"], size=12)
    cross = next((i for i in range(len(fm)) if fm[i] < sg[i]), None)
    if cross is not None:
        xc = X(c["step"][cross])
        s.line(xc, Y(3), xc, Y(floor), t["muted"], dash="2 3")
        s.text(xc - 5, Y(0.4), "fmax &lt; σmax", fill=t["ink"], anchor="end", size=12)
        s.text(xc - 5, Y(0.4) + 15, f"from step {c['step'][cross]:g}", anchor="end", size=11)
    sp = c["energy_spread_aligned_eV"]
    X, Y = panel(s, 440, 26, 260, 180, (0, n), (0, max(sp) * 1.25), [0, 5, 10],
                 [0, 0.004, 0.008, 0.012], "BFGS step", "energy spread, std. dev. (eV)", ydp=3)
    s.path([(X(a), Y(b)) for a, b in zip(c["step"], sp)], t["cool"])
    s.save(MEDIA / f"chart-committee-{theme}.svg", "Committee: fmax and the largest force disagreement per step, and the energy spread")


def chart_neb(theme):
    t = THEMES[theme]
    s = Svg(720, 250, t)
    nb = P["neb"]["series"]
    rc, e = nb["reaction_coordinate_A"], nb["relative_energy_eV"]
    fwd = kn("neb", "forward barrier")["value"]
    rev = kn("neb", "reverse barrier")["value"]
    rxn = kn("neb", "reaction energy")["value"]
    X, Y = panel(s, 56, 26, 500, 180, (0, rc[-1]), (-0.05, max(e) * 1.22),
                 [0, 0.5, 1.0, 1.5], [0, 0.2, 0.4, 0.6], "reaction coordinate (Å)",
                 "E − E(fcc) (eV)", xdp=1)
    s.path([(X(a), Y(b)) for a, b in zip(rc, e)], t["ink"])
    ts = P["neb"]["ts_image_index"]
    for i, (a, b) in enumerate(zip(rc, e)):
        s.dot(X(a), Y(b), t["oxy"] if i == ts else t["ink"], 4 if i == ts else 3)
    s.text(X(rc[ts]), Y(e[ts]) - 11, f"TS, image {ts}", fill=t["ink"], anchor="middle")
    xr = 556

    def br(x, ya, yb):
        s.add(f'<path d="M{x - 3},{ya:.1f}H{x + 3}M{x},{ya:.1f}V{yb:.1f}M{x - 3},{yb:.1f}H{x + 3}" '
              f'fill="none" stroke="{t["ink"]}"/>')
    s.line(56, Y(fwd), xr + 16, Y(fwd), t["grid"], dash="3 3")
    s.line(X(rc[-1]), Y(rxn), xr + 16, Y(rxn), t["grid"], dash="3 3")
    br(70, Y(0), Y(fwd))
    s.text(78, Y(fwd * 0.62), f"Ea {fmt(fwd, 3)} eV")
    br(xr + 16, Y(rxn), Y(fwd))
    s.text(xr + 24, Y((fwd + rxn) / 2) + 4, f"reverse {fmt(rev, 3)} eV")
    br(xr + 16, Y(0), Y(rxn))
    s.text(xr + 24, Y(rxn / 2) + 4, f"ΔE {fmt(rxn, 3)} eV")
    s.save(MEDIA / f"chart-neb-{theme}.svg", "NEB energy profile, fcc to hcp")


def pic(stem, ext, alt, width=None):
    w = f' width="{width}"' if width else ""
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="media/{stem}-dark.{ext}">\n'
            f'  <img src="media/{stem}-light.{ext}" alt="{alt}"{w}>\n</picture>')


# ------------------------------------------------------------------ README
def readme():
    r, c, nb, md, wa = (P[k] for k in ("relax", "committee", "neb", "md", "webapp"))
    rs = r["series"]
    e = rs["energy_eV"]
    drop = e[0] - e[-1]
    k90 = next(i for i, x in enumerate(e) if e[0] - x >= 0.9 * drop)
    h = []  # O height above the top Pt layer per frame, from relax.xyz
    lines = (DATA / r["frames"]).read_text().splitlines()
    i = 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        atoms = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        zpt = [float(a[3]) for a in atoms if a[0] == "Pt"]
        top = [z for z in zpt if z > max(zpt) - 0.6]
        h.append(max(float(a[3]) for a in atoms if a[0] == "O") - sum(top) / len(top))
        i += n + 2
    cs = c["series"]
    cross = next((i for i in range(len(cs["step"])) if cs["fmax_eV_per_A"][i] < cs["sigma_max_free_eV_per_A"][i]), None)
    members = [k["label"].split("member ")[1].split(" (")[0] for k in c["key_numbers"] if k["label"].startswith("final energy, member ")]
    model = r["command"].split("--mlip ")[1].split()[0]
    head = r["command"].split("--uma-task ")[1].split()[0]
    nimg = len(nb["series"]["image_index"]) - 2
    wall = kn("webapp", "job wall time")["value"]
    opt = kn("webapp", "relaxation wall time")["value"]
    nwater = sum(1 for line in (DATA / md["structures"]["built"]["file"]).read_text().splitlines()[2:] if line.split()[:1] == ["O"])
    ts_ok = "ts_verified" in nb["flags"]

    L = []
    a = L.append
    a("# mliprun\n")
    a(f"[![Tests]({REPO}/actions/workflows/tests.yml/badge.svg)]({REPO}/actions/workflows/tests.yml)")
    a(f"[![Fresh-install smoke test]({REPO}/actions/workflows/install-smoke.yml/badge.svg)]({REPO}/actions/workflows/install-smoke.yml)\n")
    a("Machine-learning interatomic potentials from one command line: relaxations, model committees, "
      "NEB with transition-state checks, MD, and remote runs on GPU machines you add yourself. "
      "UMA, MACE, SevenNet and CHGNet, each in its own environment.\n")
    a(f"**One oxygen atom, followed from start to finish.** O on Pt(111), {model}/{head}, force target "
      f"{q('relax', 'fmax target', 2)}. Each chapter below is one real run from the showcase data pack. "
      f"The [interactive version]({LANDING}) replays the same runs in 3D as you scroll.\n")
    a("To run any of this yourself, start with the [installation steps](docs/INSTALL.md) "
      "(one MLIP per environment).\n")

    a(f"## 1. Relax the adsorbate ({state(r)})\n")
    a("Start every surface study here: `mlip optimize run` moves the free atoms until the largest "
      "force on any of them drops below `--fmax`.\n")
    a("```bash\n" + short_cmd(r["command"]) + "\n```\n")
    a(pic("chart-relax", "svg", "Energy relative to step 0 and fmax per BFGS step, with the fmax target"))
    a("")
    a(f"Forces fall from {fmt(rs['fmax_eV_per_A'][0], 2)} eV/Å to {q('relax', 'final fmax, O at fcc hollow', 4)} "
      f"in {q('relax', 'BFGS steps, O at fcc hollow', 0)}. The energy drops by {fmt(drop, 3)} eV, "
      f"90% of it in the first {k90} steps. The geometry barely moves: the O atom sinks from "
      f"{h[0]:.2f} Å to {h[-1]:.2f} Å above the top Pt layer, too small to see in a picture of the slab, so this chapter shows the chart only. "
      f"The same command at the hcp hollow gives E(hcp) − E(fcc) = {q('relax', 'E(hcp) − E(fcc)', 3)}.\n")

    a(f"## 2. Ask a committee how sure it is ({state(c)})\n")
    a(f"Add `--committee` and the relaxation runs with {fmt(kn('committee', 'committee members')['value'], 0)} "
      f"models at once ({', '.join(members)}), follows their mean force, and records how far they disagree.\n")
    a("```bash\n" + short_cmd(c["command"]) + "\n```\n")
    a(pic("committee", "png", "Pt(111) slab with each atom coloured by the committee's per-atom force disagreement at the final step", 560))
    a("")
    a(pic("chart-committee", "svg", f"Committee relaxation: fmax and σmax per BFGS step, shaded zone below σmax, fmax drops below σmax at step {cross}; right panel: energy spread per step"))
    a("")
    a("**The blue line, σmax, is the largest force disagreement between the models over the free atoms. "
      "Once the force being minimised (fmax, red) drops below it, into the shaded zone, the remaining force "
      "is smaller than what these models can tell apart.** "
      f"Here fmax drops below σmax at step {cross}. At the last step the disagreement is "
      f"{q('committee', 'sigma_max_free, final step', 3)}, {q('committee', 'sigma_max_free / fmax', 1)} times the final force "
      f"of {q('committee', 'final fmax of the committee-mean force', 4)}. The energy spread ends at "
      f"{q('committee', 'energy spread across members, final step', 4)} (standard deviation after each member's step-0 energy is subtracted).\n")
    a("The atom colours are the per-atom disagreement at the committee's final step, drawn on the chapter 1 geometry "
      "(the committee run saves numbers, not frames).\n")

    a(f"## 3. Find the path between the hollows ({state(nb)})\n")
    a("`mlip neb run` builds a band from the fcc to the hcp minimum, relaxes it, and lets the highest "
      "image climb to the saddle point.\n")
    a("```bash\n" + short_cmd(nb["command"]) + "\n```\n")
    a(pic("neb", "gif", "Top view of the final NEB band, O moving from the fcc to the hcp hollow and back", 560))
    a("")
    a(pic("chart-neb", "svg", "NEB energy profile with forward and reverse barriers and the reaction energy"))
    a("")
    a(f"{nimg} images between the relaxed endpoints, converged to {q('neb', 'final band fmax', 4)} in "
      f"{q('neb', 'NEB optimizer steps', 0)}. Forward barrier {q('neb', 'forward barrier', 3)}, reverse "
      f"{q('neb', 'reverse barrier', 3)}, reaction energy {q('neb', 'reaction energy', 3)}. No atom drifted "
      f"more than {q('neb', 'band check: largest drift', 3)} from the seeded band (allowance "
      f"{q('neb', 'band check: allowance', 2)}).\n")

    a(f"## 4. Check that the top is a real saddle ({'TS verified' if ts_ok else 'TS not verified'})\n")
    a("`--verify-ts both` asks three questions of the climbing image: is there exactly one imaginary mode, "
      "does it point along the path, and does relaxing off it land in the two minima.\n")
    a(pic("ts", "gif", "The transition state displaced along its imaginary mode, an illustration of the mode and not dynamics", 560))
    a("")
    nu = kn("neb", "TS imaginary frequency")
    a(f"One imaginary mode at {fmt(nu['value'], 1)}i {nu['unit']}. Overlap with the NEB tangent "
      f"{q('neb', 'overlap of imaginary mode', 3)} (threshold {q('neb', 'overlap threshold', 1)}); displaced "
      f"along the mode and relaxed, the structure connects to both endpoints. The animation is the mode scaled "
      f"so the most-displaced atom moves {q('neb', 'TS mode animation peak displacement', 1)}: an illustration, not dynamics.\n")
    if "ts_verify_endpoint_fmax_above_relax_fmax" in nb["flags"]:
        a(f"> Flag raised by the check: {nb['notes'][0]}\n")

    a(f"## 5. Water at finite temperature ({state(md)})\n")
    a("**MD not run; next step pending a decision.**\n")
    a(f"The story moves to a new system, a bilayer of {nwater} water molecules on Pt(111). "
      "The picture shows the "
      f"{md['structures']['built']['label']}: a static structure, not a simulation.\n")
    a(pic("water", "png", "Static structure, not a simulation: water bilayer on Pt(111) as built", 560))
    a("")

    a(f"## 6. Run it on your own GPU machine from the browser ({state(wa)})\n")
    a("`mlip webapp` sends a job to a remote machine that you add yourself and brings the result back: "
      "the GPU does the work, your laptop only submits. mliprun ships with no preconfigured machines.\n")
    a("1. In the web app, open **Machines** and choose **Add machine**: your SSH alias (`<your ssh alias>`) "
      "and your remote folder (`<your remote folder>`).")
    a("2. Point it at your own MLIP environment (`<your uma env>`).")
    a("3. Press **Test connection**, then submit. The web app runs the same command as the CLI:\n")
    a("```bash\n" + short_cmd(wa["command"]) + "\n```\n")
    a(f'<img src="media/webapp.gif" alt="Screen recording: a relaxation submitted from the mliprun web app to {wa["target"]["machine"]}, where the showcase runs ran, and its result">\n')
    a(f"In the recording, the chapter 1 relaxation runs on {wa['target']['machine']}, the machine where all "
      "showcase runs in this README ran (one GPU card); your machine will be the one you add. "
      "Timing: "
      f"{fmt(wall, 1)} s in total, {fmt(opt, 1)} s of it relaxation and the rest loading the model. "
      f"Same {q('webapp', 'BFGS steps', 0)} as the command line, energy difference "
      f"{q('webapp', 'energy difference, web app − CLI', 1)} (GPU floating-point noise). "
      f"The waiting part of the recording is sped up {fmt(kn('webapp', 'video speed-up')['value'], 0)}× and labelled so. "
      f"[Full-quality video](data/{wa['video']}).\n")

    a("## Also in mliprun\n")
    a("- `mlip freq run`: finite-difference frequencies, imaginary-mode count and zero-point energy, with per-member spread from a committee.")
    a("- `mlip autoneb run`: NEB with images inserted where the path needs them.")
    a("- `mlip benchmark run`: times one energy call for each installed MLIP on the same structure.")
    a("- `--mlip`: UMA, MACE, SevenNet and CHGNet, each in its own environment.")
    a("- `mlip doctor`: checks that an environment can import and run its model.\n")
    a(f"Where each number comes from: [data pack provenance]({REPO}/blob/main/docs/showcase/data/pack/PROVENANCE.md). "
      "Installation: [docs/INSTALL.md](docs/INSTALL.md).\n")
    (HERE / "README.md").write_text("\n".join(L))


def main():
    MEDIA.mkdir(exist_ok=True)
    for th in THEMES:
        chart_relax(th)
        chart_committee(th)
        chart_neb(th)
    readme()
    print("wrote README.md and", len(list(MEDIA.glob("chart-*.svg"))), "charts")


if __name__ == "__main__":
    main()
