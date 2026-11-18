"""Build the README variant of the gallery (alternative 3) from the data pack.

    python site/alt-3-gallery/build_readme.py [--prefix site/alt-3-gallery/]

Writes media/strip-*-{light,dark}.svg (the labelled data strips) and README.md.
Every number in the README is formatted here from data/*.json; none is typed.
--prefix is prepended to media paths, for when this README replaces the repo
root README (Phase 5); the default keeps paths relative to this folder.
Run render_media.py first for the GIF/PNG media.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MEDIA = HERE / "media"
# Landing-page address after Phase 5 (GitHub Pages). Not live yet: see report.
LANDING_URL = "https://manuelarcer.github.io/mliprun/"
REPO = "https://github.com/manuelarcer/mliprun"

THEME = {
    "light": dict(fg="#1f2328", muted="#59636e", grid="#d1d9e0", accent="#2443c4", accent2="#8a96b8",
                  band="#2443c4", band_op=0.16, sig="#7a3fb0", sig_op=0.22, ts="#c0392b"),
    "dark": dict(fg="#e6edf3", muted="#9198a1", grid="#3d444d", accent="#86a0ff", accent2="#6e7a99",
                 band="#86a0ff", band_op=0.22, sig="#c69bf0", sig_op=0.24, ts="#ff7a6b"),
}


def load(name):
    return json.loads((DATA / name).read_text())


def K(j, prefix):
    for k in j["key_numbers"]:
        if k["label"].startswith(prefix):
            return k
    raise KeyError(f"{j['use_case']}: {prefix}")


def sci(v, d=1):
    e = math.floor(math.log10(abs(v)))
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{v / 10 ** e:.{d}f} × 10{sup}"


def cmd_tokens(cmd):
    t = cmd.split()
    for i, x in enumerate(t):
        if x == "mlip" or x.endswith("/mlip"):
            return t[i + 1:]
    return t


def cmd_short(cmd, keep):
    t = cmd_tokens(cmd)
    out = t[:2]
    for i in range(2, len(t)):
        if t[i] in keep:
            out.append(t[i])
            if i + 1 < len(t) and not t[i + 1].startswith("--"):
                out.append(t[i + 1])
    return "mlip " + " ".join(out)


# ---------------------------------------------------------------- SVG strips
def nice_ticks(lo, hi, n=4):
    span = (hi - lo) or abs(hi) or 1
    step = 10 ** math.floor(math.log10(span / n))
    err = span / n / step
    step *= 10 if err >= 7.5 else 5 if err >= 3.5 else 2 if err >= 1.5 else 1
    v = math.ceil(lo / step - 1e-9) * step
    out = []
    while v <= hi + step * 1e-9:
        out.append(round(v, 12))
        v += step
    return out, step


def tick_label(v, step):
    d = max(0, -math.floor(math.log10(step) + 1e-9))
    s = f"{v:.{d}f}"
    return "0" if float(s) == 0 else s


def log_ticks(lo, hi, max_n=6):
    allt = [m * 10.0 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
            for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
    dec = [v for v in allt if abs(math.log10(v) - round(math.log10(v))) < 1e-9]
    return dec if len(allt) > max_n and len(dec) >= 2 else allt


def log_label(v):
    return f"{v:.3g}" if v >= 1 else f"{v:.2g}"


def svg_chart(theme, *, x, series, x_label, y_label, y_log=False, refs=(), band=None, marks=(),
              ptexts=(), texts=(), bracket=None, title="", W=420, H=140, gutter=10):
    """Same conventions as main.js chart(): log axes clipped to the data,
    end labels and gutter refs in a right gutter, pushed apart, never on the lines."""
    c = THEME[theme]
    M = dict(l=44, r=gutter, t=32, b=28)
    pw, ph = W - M["l"] - M["r"], H - M["t"] - M["b"]
    ys = [v for s in series for v in s["y"]] + [r["y"] for r in refs]
    if band:
        ys += list(band["lo"]) + list(band["hi"])
    if y_log:
        lo, hi = min(v for v in ys if v > 0) / 1.25, max(ys) * 1.25
        yt = log_ticks(lo, hi, max(3, int(ph // 16)))
        ymap = lambda v: M["t"] + ph - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * ph  # noqa: E731
        ylab = log_label
    else:
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.08 or 0.01
        lo, hi = lo - pad, hi + pad
        yt, ystep = nice_ticks(lo, hi, 3)
        lo, hi = min(lo, yt[0]), max(hi, yt[-1])
        ymap = lambda v: M["t"] + ph - (v - lo) / (hi - lo) * ph  # noqa: E731
        ylab = lambda v: tick_label(v, ystep)  # noqa: E731
    x0, x1 = min(x), max(x)
    xmap = lambda v: M["l"] + (v - x0) / (x1 - x0 or 1) * pw  # noqa: E731
    xt, xstep = nice_ticks(x0, x1, 5)
    font = "font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif\""
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" {font} font-size="11">']
    o.append(f'<text x="0" y="12" fill="{c["fg"]}" font-size="12" font-weight="600">{title}</text>')
    for v in yt:
        y = ymap(v)
        o.append(f'<line x1="{M["l"]}" x2="{M["l"] + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c["grid"]}"/>')
        o.append(f'<text x="{M["l"] - 5}" y="{y + 3.5:.1f}" text-anchor="end" fill="{c["muted"]}">{ylab(v)}</text>')
    for v in xt:
        xx = xmap(v)
        o.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{M["t"] + ph}" y2="{M["t"] + ph + 3}" stroke="{c["muted"]}"/>')
        o.append(f'<text x="{xx:.1f}" y="{M["t"] + ph + 14}" text-anchor="middle" fill="{c["muted"]}">{tick_label(v, xstep)}</text>')
    o.append(f'<path d="M{M["l"]},{M["t"]} V{M["t"] + ph} H{M["l"] + pw}" fill="none" stroke="{c["muted"]}"/>')
    o.append(f'<text x="{M["l"] - 5}" y="{M["t"] - 8}" fill="{c["fg"]}">{y_label}</text>')
    o.append(f'<text x="{M["l"] + pw}" y="{H - 2}" text-anchor="end" fill="{c["fg"]}">{x_label}</text>')
    if band:
        pts = [f"{xmap(a):.1f},{ymap(b):.1f}" for a, b in zip(x, band["hi"])]
        pts += [f"{xmap(a):.1f},{ymap(b):.1f}" for a, b in reversed(list(zip(x, band["lo"])))]
        o.append(f'<polygon points="{" ".join(pts)}" fill="{c[band.get("color", "sig")]}" fill-opacity="{c["sig_op"]}"/>')
    gut = []
    for r in refs:
        y = ymap(r["y"])
        o.append(f'<line x1="{M["l"]}" x2="{M["l"] + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c["fg"]}" stroke-dasharray="3 3"/>')
        if r.get("at") == "gutter":
            gut.append([y, r["label"], c["fg"], "400"])
        else:
            o.append(f'<text x="{M["l"] + 4}" y="{y - 4:.1f}" fill="{c["fg"]}">{r["label"]}</text>')
    for s in series:
        col = c[s.get("color", "accent")]
        d = "".join(f'{"L" if i else "M"}{xmap(a):.1f},{ymap(b):.1f}' for i, (a, b) in enumerate(zip(x, s["y"])))
        if s.get("line", True):
            dash = ' stroke-dasharray="4 3"' if s.get("dash") else ""
            o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.6"{dash}/>')
        if s.get("points"):
            for a, b in zip(x, s["y"]):
                o.append(f'<circle cx="{xmap(a):.1f}" cy="{ymap(b):.1f}" r="2.4" fill="{col}"/>')
        if s.get("end"):
            gut.append([ymap(s["y"][-1]), s["end"], col, "600"])
    gut.sort(key=lambda q: q[0])
    for k in range(1, len(gut)):
        gut[k][0] = max(gut[k][0], gut[k - 1][0] + 11)
    over = gut[-1][0] - (M["t"] + ph + 4) if gut else 0
    for q in gut:
        q[0] -= max(0, over)
        o.append(f'<text x="{M["l"] + pw + 6}" y="{q[0] + 3.5:.1f}" fill="{q[2]}" font-weight="{q[3]}">{q[1]}</text>')
    if bracket:
        xx = xmap(x[bracket["i"]]) - 10
        ya, yb = ymap(bracket["a"]), ymap(bracket["b"])
        o.append(f'<path d="M{xx + 4:.1f},{ya:.1f} H{xx:.1f} V{yb:.1f} H{xx + 4:.1f}" fill="none" stroke="{c["fg"]}" stroke-width="1.2"/>')
        o.append(f'<text x="{xx - 4:.1f}" y="{(ya + yb) / 2 + 4:.1f}" text-anchor="end" fill="{c["fg"]}" font-weight="600">{bracket["label"]}</text>')
    for i, lab in marks:
        xx, yy = xmap(x[i]), ymap(series[0]["y"][i])
        o.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="3.8" fill="{c["ts"]}"/>')
        o.append(f'<text x="{xx:.1f}" y="{yy - 9:.1f}" text-anchor="middle" fill="{c["ts"]}" font-weight="600">{lab}</text>')
    for p in ptexts:
        s = series[p.get("s", 0)]
        o.append(f'<text x="{xmap(x[p["i"]]) + p.get("dx", 0):.1f}" y="{ymap(s["y"][p["i"]]) + p.get("dy", 0):.1f}" '
                 f'text-anchor="{p.get("anchor", "middle")}" fill="{c["fg"]}">{p["text"]}</text>')
    for fx, fy, txt, anchor in texts:
        o.append(f'<text x="{M["l"] + fx * pw:.1f}" y="{M["t"] + fy * ph:.1f}" text-anchor="{anchor}" fill="{c["fg"]}">{txt}</text>')
    o.append("</svg>")
    return "\n".join(o)


def write_strips(relax, committee, neb, webapp):
    tgt = K(relax, "fmax target")["value"]
    for th in THEME:
        s = relax["series"]
        (MEDIA / f"strip-relax-{th}.svg").write_text(svg_chart(
            th, x=s["step"], series=[{"y": s["fmax_eV_per_A"], "points": True}], y_log=True,
            refs=[{"y": tgt, "label": f"target {tgt}"}], x_label="optimizer step", y_label="fmax / eV Å⁻¹",
            title="Max force against step"))
        s = committee["series"]
        ct = K(committee, "fmax target")["value"]
        last = len(s["step"]) - 1
        ratio = K(committee, "sigma_max_free / fmax, final")["value"]
        (MEDIA / f"strip-committee-{th}.svg").write_text(svg_chart(
            th, x=s["step"], y_log=True, W=420, H=170, gutter=96,
            band={"lo": s["sigma_mean_free_eV_per_A"], "hi": s["sigma_max_free_eV_per_A"]},
            series=[{"y": s["fmax_eV_per_A"], "points": True, "end": f"fmax {s['fmax_eV_per_A'][-1]:.3f}"},
                    {"y": s["sigma_max_free_eV_per_A"], "color": "sig", "end": f"σ max {s['sigma_max_free_eV_per_A'][-1]:.3f}"},
                    {"y": s["sigma_mean_free_eV_per_A"], "color": "sig", "dash": True, "end": f"σ mean {s['sigma_mean_free_eV_per_A'][-1]:.3f}"}],
            refs=[{"y": ct, "label": f"target {ct}", "at": "gutter"}],
            bracket={"i": last, "a": s["fmax_eV_per_A"][last], "b": s["sigma_max_free_eV_per_A"][last], "label": f"σ max = {ratio:.1f}× fmax"},
            x_label="optimizer step", y_label="force / eV Å⁻¹",
            title="Remaining force against the committee's force σ"))
        s = neb["series"]
        n = len(s["image_index"])
        fwd, dE = K(neb, "forward barrier")["value"], K(neb, "reaction energy")["value"]
        (MEDIA / f"strip-neb-{th}.svg").write_text(svg_chart(
            th, x=s["reaction_coordinate_A"], series=[{"y": s["relative_energy_eV"], "points": True}],
            marks=[(neb["ts_image_index"], f"TS {fwd:.3f} eV")],
            ptexts=[{"i": 0, "text": "fcc", "dx": 4, "dy": -9, "anchor": "start"},
                    {"i": n - 1, "text": f"hcp, ΔE {'+' if dE > 0 else ''}{dE:.3f} eV", "dy": 18, "anchor": "end"}],
            x_label="reaction coordinate / Å", y_label="E − E(fcc) / eV",
            title=f"Energy along the final band ({n - 2} intermediate + 2 endpoint images)", W=720, H=170))
        s = webapp["series"]
        eref = relax["series"]["energy_eV"][-1]
        d = K(webapp, "energy difference, web app")["value"]
        (MEDIA / f"strip-webapp-{th}.svg").write_text(svg_chart(
            th, x=s["step"],
            series=[{"y": [e - eref for e in relax["series"]["energy_eV"]], "color": "accent2"},
                    {"y": [e - eref for e in s["energy_eV"]], "line": False, "points": True}],
            texts=[(1, 0.3, f"final difference {'+' if d > 0 else ''}{sci(d)} eV", "end")],
            x_label="optimizer step", y_label="E − E(final, CLI) / eV",
            title="Same input: CLI run (line), web app job (dots)", W=720, H=150))


# ---------------------------------------------------------------- README
def pic(prefix, name, ext, width, alt):
    return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{prefix}media/{name}-dark.{ext}">'
            f'<img src="{prefix}media/{name}-light.{ext}" width="{width}" alt="{alt}"></picture>')


def build(prefix=""):
    relax, committee, neb, md, webapp = (load(f"{n}.json") for n in ("relax", "committee", "neb", "md", "webapp"))
    caps = load("capabilities.json")["items"]
    MEDIA.mkdir(exist_ok=True)
    write_strips(relax, committee, neb, webapp)
    toks = cmd_tokens(relax["command"])
    level = f"{toks[toks.index('--mlip') + 1]}/{toks[toks.index('--uma-task') + 1]}"
    tgt = K(relax, "fmax target")
    conv = lambda j: "converged" if j["converged"] else "not converged"  # noqa: E731
    # docs link: relative to the repo root once this README is the root README
    install = "docs/INSTALL.md" if prefix else "../../docs/INSTALL.md"

    rs, rf = K(relax, "BFGS steps, O at fcc"), K(relax, "final fmax, O at fcc")
    members = [k[2:-3] for k in committee["series"] if k.startswith("E_")]
    ratio, smax = K(committee, "sigma_max_free / fmax, final"), K(committee, "sigma_max_free, final")
    sig = committee["series"]["peratom_final_sigma_free_eV_per_A"]
    fwd, rev = K(neb, "forward barrier"), K(neb, "reverse barrier")
    nu, ov, ovt = K(neb, "TS imaginary frequency"), K(neb, "overlap of imaginary mode"), K(neb, "overlap threshold")
    bf, ns = K(neb, "final band fmax"), K(neb, "NEB optimizer steps")
    amp = K(neb, "TS mode animation peak displacement")
    nimg = len(neb["series"]["image_index"])
    ts_ok = "ts_verified" in neb["flags"]
    wall, dE = K(webapp, "job wall time"), K(webapp, "energy difference, web app")
    speed, vlen = K(webapp, "video speed-up"), K(webapp, "video length")
    machine, env = webapp["target"]["machine"], webapp["target"]["env"]
    slab = next((n for n in relax["notes"] if n.startswith("Slab:")), "")
    rt = cmd_tokens(relax["command"])
    rt = ["POSCAR" if i and rt[i - 1] == "--structure" else t for i, t in enumerate(rt)]
    run_cmd = "mlip " + " ".join(rt)  # the run's own structure path is provenance, not a path to use

    L = []
    L.append("# mliprun\n")
    L.append(f"[![Tests]({REPO}/actions/workflows/tests.yml/badge.svg)]({REPO}/actions/workflows/tests.yml)")
    L.append(f"[![Fresh-install smoke test]({REPO}/actions/workflows/install-smoke.yml/badge.svg)]({REPO}/actions/workflows/install-smoke.yml)\n")
    L.append("Machine-learning interatomic potentials from one command: relaxations, committees, NEB with a "
             "transition-state check, MD, and jobs sent to a remote GPU machine you add yourself. UMA, MACE, SevenNet and CHGNet.\n")
    L.append(f"**[Open the live gallery]({LANDING_URL})** for rotatable 3D panels, "
             f"**[install]({install})**, or **[run it on your own GPU machine](#run-it-on-your-own-gpu-machine)**.\n")
    L.append(f"Plate 1. Five real runs on O/Pt(111), {level}, all on {machine} (one GPU card). Numbers are generated from the run records.\n")
    L.append("<table>")
    # row 1: a, NEB (wide)
    L.append("<tr><td colspan=\"2\">")
    L.append(f"<b>a</b>&nbsp; <b>O hops fcc → hcp, barrier {fwd['value']:.3f} {fwd['unit']}</b> ({conv(neb)}{', TS verified' if ts_ok else ''})<br>")
    L.append(pic(prefix, "neb", "gif", 460, "NEB band: O hops from the fcc to the hcp hollow on Pt(111)") +
             pic(prefix, "tsmode", "gif", 300, f"Imaginary mode at the TS, {nu['value']:.1f}i cm⁻¹; mode, not dynamics"))
    L.append(f"<br><sub>TS mode: {nu['value']:.1f}i {nu['unit']}, overlap {ov['value']:.3f} (≥ {ovt['value']}); mode, not dynamics.</sub>")
    L.append("<br>" + pic(prefix, "strip-neb", "svg", 760, "Energy along the final band with the TS, fcc and hcp marked"))
    L.append(f"<br><code>{cmd_short(neb['command'], ['--num-images', '--climb', '--verify-ts'])}</code>")
    L.append("</td></tr>")
    # row 2: b relax | c committee
    L.append("<tr><td width=\"50%\" valign=\"top\">")
    L.append(f"<b>b</b>&nbsp; <b>Relaxation, {int(rs['value'])} BFGS steps</b> ({conv(relax)})<br>")
    L.append(pic(prefix, "relax", "gif", 400, "Relaxation of O at the fcc hollow of Pt(111), every optimizer step"))
    L.append("<br>" + pic(prefix, "strip-relax", "svg", 400, "Max force against optimizer step, with the target line"))
    L.append(f"<br><code>{cmd_short(relax['command'], ['--mlip', '--uma-task', '--fmax'])}</code>")
    L.append("</td><td width=\"50%\" valign=\"top\">")
    L.append(f"<b>c</b>&nbsp; <b>Committee: σ max {ratio['value']:.1f}× fmax</b> ({conv(committee)})<br>")
    L.append(pic(prefix, "committee", "png", 400, f"Force sigma per atom at the last step, 0 to {max(sig):.3f} eV/Å; O red"))
    L.append(f"<br><sub>Force σ per atom, last step: pale 0 → violet {max(sig):.3f} eV/Å; O red.</sub>")
    L.append("<br>" + pic(prefix, "strip-committee", "svg", 400, "Committee-mean fmax against step with the largest and mean force sigma"))
    L.append(f"<br><code>{cmd_short(committee['command'], ['--committee', '--fmax'])}</code>")
    L.append("</td></tr>")
    # row 3: d MD | capability list
    L.append("<tr><td valign=\"top\">")
    if md.get("status") == "pending" or not md.get("frames"):
        L.append("<b>d</b>&nbsp; <b>MD: not run; next step pending a decision</b><br>")
        L.append(pic(prefix, "md", "png", 400, f"{md['structures']['built']['label']}; static, not a simulation"))
        L.append(f"<br><sub>{md['structures']['built']['label']}. Static, not a simulation.</sub>")
        L.append("<br><code>mlip md run</code>")
    else:
        L.append(f"<b>d</b>&nbsp; <b>Finite-temperature MD</b> ({conv(md)})<br>MD media: rerun render_media.py once md.json has frames.")
    L.append("</td><td valign=\"top\">")
    L.append("<b>Also in mliprun</b>")
    L.append("<ul>" + "".join(f"<li>{'<code>' + c['cmd'] + '</code> ' if c['cmd'] else ''}{c['text']}</li>" for c in caps) + "</ul>")
    L.append("</td></tr>")
    # row 4: e web app (wide)
    L.append("<tr><td colspan=\"2\">")
    L.append(f"<b>e</b>&nbsp; <b>Web app → {machine}: {wall['value']:.1f} {wall['unit']} job</b> ({conv(webapp)})<br>")
    L.append(pic(prefix, "webapp-stills", "png", 780, "Web app: submit, running, done"))
    L.append(f"<br><sub>Submit, running, done. <a href=\"{prefix}data/webapp.mp4\">Recording</a> ({vlen['value']:.1f} s).</sub>")
    L.append("<br>" + pic(prefix, "strip-webapp", "svg", 760, "Energy against step for the web app job and the CLI run; they coincide"))
    L.append("<br><code>mlip webapp</code>")
    L.append("</td></tr>")
    L.append("</table>\n")
    L.append(f"<sub>[Data and provenance]({REPO}/blob/main/docs/showcase/data/pack/PROVENANCE.md).</sub>\n")
    L.append("## Run it on your own GPU machine\n")
    L.append("No machines are preconfigured: add your own. CLI, on your GPU machine:\n")
    L.append("```bash\n" + run_cmd + "\n```\n")
    L.append("Web app (`mlip webapp`): *Machines* → *Add machine* (*Host* `<your ssh alias>`, *Default remote location* "
             "`<your remote folder>`) → *Add environment* (*Python path* `<your uma env>/bin/python`) → *Test connection*. "
             f"[Guide]({REPO}/blob/main/webapp/README.md#running-on-another-machine-ssh).\n")
    L.append("## Installation\n")
    L.append(f"See **[docs/INSTALL.md]({install})**.")
    (HERE / "README.md").write_text("\n".join(L) + "\n")
    print("wrote", HERE / "README.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="")
    build(ap.parse_args().prefix)
