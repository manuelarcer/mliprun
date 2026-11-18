"""Render the README media for alternative 5 from the data pack.

    site/.venv/bin/python site/alt-5-cinematic/build_media.py   (outside the sandbox)

3D clips go through site/shared/capture.py (the same viewer.js as the page),
so README and landing page match. Charts are static SVG files drawn here from
the JSONs, with labelled axes and units, on their own dark panel so they read
in GitHub light and dark themes. The web-app clip is cut from the pack MP4.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MEDIA = HERE / "media"
PY = sys.executable
CAPTURE = HERE.parent / "shared" / "capture.py"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
DARK = "#0e1724"
LIGHT = "#f6f8fa"

# Panel palette (matches style.css tokens)
P = {"bg": DARK, "rule": "#2c3e56", "muted": "#9fb0c3", "text": "#e8edf3",
     "key": "#f1c98a", "cool": "#86c7dc"}


def cap(xyz: str, stem: str, *args: str) -> None:
    cmd = [PY, str(CAPTURE), str(DATA / xyz), str(MEDIA / stem), *args]
    print("+", " ".join(cmd[1:]), flush=True)
    for attempt in range(3):  # capture.py's page load times out on a busy machine
        if subprocess.run(cmd).returncode == 0:
            return
        print(f"capture failed (attempt {attempt + 1}); retrying", flush=True)
        time.sleep(30)
    raise RuntimeError(f"capture failed three times: {stem}")


def keep_gif_only(stem: str) -> None:
    (MEDIA / f"{stem}.mp4").unlink(missing_ok=True)


# ---------------------------------------------------------------- SVG charts
def ticks_lin(a: float, b: float, n: int = 4) -> list[float]:
    raw = (b - a) / n
    p = 10 ** math.floor(math.log10(raw))
    step = next(s * p for s in (1, 2, 2.5, 5, 10) if s * p >= raw)
    v = math.ceil(a / step) * step
    out = []
    while v <= b + step * 1e-6:
        out.append(round(v, 10))
        v += step
    return out


def fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) < 0.01:
        return f"{v:.0e}"
    return f"{v:.3g}"


def svg_chart(path: Path, *, x, y, series, hlines=(), marks=(), legend=(), title="",
              w=640, h=300) -> None:
    m = dict(l=64, r=18, t=36 if legend else 16, b=46)
    W, H = w - m["l"] - m["r"], h - m["t"] - m["b"]
    ylog = y.get("log", False)

    def ty(v):
        if ylog:
            a, b = math.log10(y["min"]), math.log10(y["max"])
            return m["t"] + H - (math.log10(v) - a) / (b - a) * H
        return m["t"] + H - (v - y["min"]) / (y["max"] - y["min"]) * H

    def tx(v):
        return m["l"] + (v - x["min"]) / (x["max"] - x["min"]) * W

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
         f'role="img" aria-label="{title}" font-family="Helvetica, Arial, sans-serif">',
         f'<rect width="{w}" height="{h}" rx="6" fill="{P["bg"]}"/>']
    yt = ([10 ** e for e in range(math.ceil(math.log10(y["min"])), math.floor(math.log10(y["max"])) + 1)]
          if ylog else ticks_lin(y["min"], y["max"]))
    for v in yt:
        o.append(f'<line x1="{m["l"]}" x2="{m["l"] + W}" y1="{ty(v):.1f}" y2="{ty(v):.1f}" stroke="{P["rule"]}" stroke-opacity="0.5"/>')
        o.append(f'<text x="{m["l"] - 8}" y="{ty(v) + 4:.1f}" text-anchor="end" font-size="12" fill="{P["muted"]}">{fmt_tick(v)}</text>')
    for v in x.get("ticks") or ticks_lin(x["min"], x["max"], 5):
        o.append(f'<line x1="{tx(v):.1f}" x2="{tx(v):.1f}" y1="{m["t"] + H}" y2="{m["t"] + H + 5}" stroke="{P["rule"]}"/>')
        o.append(f'<text x="{tx(v):.1f}" y="{m["t"] + H + 19}" text-anchor="middle" font-size="12" fill="{P["muted"]}">{fmt_tick(v)}</text>')
    o.append(f'<path d="M{m["l"]},{m["t"]}V{m["t"] + H}H{m["l"] + W}" stroke="{P["rule"]}" fill="none"/>')
    o.append(f'<text x="{m["l"] + W / 2}" y="{h - 8}" text-anchor="middle" font-size="13" fill="{P["muted"]}">{x["label"]}</text>')
    cy = m["t"] + H / 2
    o.append(f'<text x="16" y="{cy}" text-anchor="middle" font-size="13" fill="{P["muted"]}" transform="rotate(-90 16 {cy})">{y["label"]}</text>')
    for hl in hlines:
        o.append(f'<line x1="{m["l"]}" x2="{m["l"] + W}" y1="{ty(hl["y"]):.1f}" y2="{ty(hl["y"]):.1f}" stroke="{P["muted"]}" stroke-dasharray="5 4"/>')
        o.append(f'<text x="{m["l"] + 6}" y="{ty(hl["y"]) + 15:.1f}" text-anchor="start" font-size="12" fill="{P["muted"]}">{hl["label"]}</text>')
    for s in series:
        d = "".join(f'{"L" if i else "M"}{tx(a):.1f},{ty(b):.1f}' for i, (a, b) in enumerate(zip(s["xs"], s["ys"])))
        dash = ' stroke-dasharray="2 4"' if s.get("dash") else ""
        o.append(f'<path d="{d}" stroke="{s["color"]}" stroke-width="2" fill="none"{dash}/>')
        if s.get("dots"):
            for a, b in zip(s["xs"], s["ys"]):
                o.append(f'<circle cx="{tx(a):.1f}" cy="{ty(b):.1f}" r="3.2" fill="{s["color"]}"/>')
    for mk in marks:
        o.append(f'<circle cx="{tx(mk["x"]):.1f}" cy="{ty(mk["y"]):.1f}" r="6" fill="{P["bg"]}" stroke="{P["key"]}" stroke-width="2"/>')
        o.append(f'<text x="{tx(mk["x"]):.1f}" y="{ty(mk["y"]) - 12:.1f}" text-anchor="middle" font-size="12" fill="{P["text"]}">{mk["label"]}</text>')
    lx = m["l"]
    for color, label, dash in legend:
        da = ' stroke-dasharray="2 4"' if dash == "dot" else (' stroke-dasharray="5 4"' if dash == "dash" else "")
        o.append(f'<line x1="{lx}" x2="{lx + 18}" y1="18" y2="18" stroke="{color}" stroke-width="2"{da}/>')
        o.append(f'<text x="{lx + 24}" y="22" font-size="12" fill="{P["muted"]}">{label}</text>')
        lx += 30 + 6.6 * len(label)
    o.append("</svg>")
    path.write_text("\n".join(o) + "\n")
    print(f"svg: {path.relative_to(HERE)}")


def kn(j, sub):
    return next(k for k in j["key_numbers"] if sub in k["label"])


def charts() -> None:
    c = json.loads((DATA / "committee.json").read_text())
    s = c["series"]
    tgt = kn(c, "fmax target")["value"]
    allv = s["fmax_eV_per_A"] + s["sigma_max_free_eV_per_A"] + s["sigma_mean_free_eV_per_A"] + [tgt]
    svg_chart(MEDIA / "committee.svg", title="Committee-mean fmax and force sigma against BFGS step",
              x={"min": 0, "max": s["step"][-1], "label": "BFGS step",
                 "ticks": list(range(0, int(s["step"][-1]) + 1, 2))},
              y={"min": 10 ** math.floor(math.log10(min(allv))), "max": max(allv) * 1.6,
                 "label": "force (eV/Å), log scale", "log": True},
              series=[{"xs": s["step"], "ys": s["fmax_eV_per_A"], "color": P["key"], "dots": True},
                      {"xs": s["step"], "ys": s["sigma_max_free_eV_per_A"], "color": P["cool"]},
                      {"xs": s["step"], "ys": s["sigma_mean_free_eV_per_A"], "color": P["cool"], "dash": True}],
              hlines=[{"y": tgt, "label": f"fmax target {tgt:g} eV/Å"}],
              legend=[(P["key"], "fmax of the mean force", ""), (P["cool"], "σ max, free atoms", ""),
                      (P["cool"], "σ mean, free atoms", "dot")])
    n = json.loads((DATA / "neb.json").read_text())
    s = n["series"]
    ts = n["ts_image_index"]
    xs, ys = s["reaction_coordinate_A"], s["relative_energy_eV"]
    svg_chart(MEDIA / "neb_profile.svg", title="NEB energy profile, O fcc to hcp on Pt(111)",
              x={"min": 0, "max": max(xs), "label": "reaction coordinate (Å)"},
              y={"min": min(0, min(ys)) - 0.05, "max": max(ys) * 1.25, "label": "E − E(fcc) (eV)"},
              series=[{"xs": xs, "ys": ys, "color": P["key"], "dots": True}],
              marks=[{"x": xs[ts], "y": ys[ts], "label": f"TS, image {ts}"}])


# ---------------------------------------------------------------- page clips
# The README hero and TS-mode clips are cut from the LIVE PAGE (viewer
# snapshot() includes the page's shapes: NEB ghosts, path line, mode arrows),
# cropped around the central-cell O so the motion reads at README width.
FRAME_JS = r"""async ([k, crop, out, labels]) => {
  const st = window.__alt5, c = st.current.ctrl;
  c.setFrame(k);
  const url = c.snapshot();
  const img = new Image();
  await new Promise((r) => { img.onload = r; img.src = url; });
  const layer = st.current.layer.getBoundingClientRect();
  const sx = img.width / layer.width;
  const toCanvas = (p) => ({ x: (p.x - scrollX - layer.left) * sx, y: (p.y - scrollY - layer.top) * sx });
  const centre = toCanvas(c.viewer.modelToScreen(crop.at));
  const w = crop.w * sx, h = crop.h * sx;
  const x0 = centre.x - w / 2, y0 = centre.y - h * crop.fy;
  const cv = document.createElement("canvas");
  cv.width = out.w; cv.height = out.h;
  const g = cv.getContext("2d");
  g.fillStyle = "#0e1724"; g.fillRect(0, 0, out.w, out.h);
  g.drawImage(img, x0, y0, w, h, 0, 0, out.w, out.h);
  const f = out.w / w;
  g.font = `600 ${Math.round(out.h / 16)}px Archivo, Helvetica, Arial, sans-serif`;
  g.textBaseline = "bottom";
  const tp = (labels.tags || []).map((t) => toCanvas(c.viewer.modelToScreen(t.at)));
  const mx = tp.reduce((a, p) => a + p.x, 0) / (tp.length || 1), my = tp.reduce((a, p) => a + p.y, 0) / (tp.length || 1);
  for (const [i, t] of (labels.tags || []).entries()) {
    const p = tp[i];
    let dx = p.x - mx, dy = p.y - my;
    const nn = Math.hypot(dx, dy) || 1;
    const X = (p.x - x0) * f + (dx / nn) * out.h / 7, Y = (p.y - y0) * f + (dy / nn) * out.h / 7 - out.h / 30;
    const tw = g.measureText(t.text).width;
    g.fillStyle = "rgba(14,23,36,0.8)"; g.fillRect(X - tw / 2 - 6, Y - out.h / 16 - 6, tw + 12, out.h / 16 + 10);
    g.fillStyle = "#f1c98a"; g.textAlign = "center"; g.fillText(t.text, X, Y);
  }
  const line = labels.lines ? labels.lines[k] : labels.text;
  if (line) {
    g.font = `500 ${Math.round(out.h / 20)}px Archivo, Helvetica, Arial, sans-serif`;
    g.textAlign = "left";
    const tw = g.measureText(line).width;
    g.fillStyle = "rgba(14,23,36,0.85)"; g.fillRect(8, out.h - out.h / 20 - 22, tw + 20, out.h / 20 + 16);
    g.fillStyle = "#e8edf3"; g.fillText(line, 18, out.h - 12);
  }
  return cv.toDataURL("image/png");
}"""


def page_clips() -> None:
    import base64
    sys.path.insert(0, str(HERE))
    import check as C  # route handler + Chromium launcher of the page checks
    from playwright.sync_api import sync_playwright
    neb = json.loads((DATA / "neb.json").read_text())
    rel = neb["series"]["relative_energy_eV"]
    n = len(rel)
    with sync_playwright() as pw:
        C._launch(pw).close()
        b = pw.chromium.launch(executable_path=C.LAST_CHROMIUM["executable"],
                               args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
        try:
            C.DPR = 2
            ctx, page, errors = C.new_page(b, 1440, 900, reduced=True)
            page.goto(C.PAGE, wait_until="domcontentloaded")
            C.wait_scene(page, "hero")
            page.wait_for_timeout(500)
            mid = page.evaluate("() => { const t = window.__alt5.tags; const a = t[0].at, b = t[1].at;"
                                " return {x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2}; }")
            tags = page.evaluate("() => window.__alt5.tags.map((t) => ({text: t.text, at: t.at}))")
            lines = [f"NEB image {k} of {n}: E − E(fcc) = {rel[k]:.3f} eV" + (" (TS)" if k == neb["ts_image_index"] else "")
                     for k in range(n)]
            seq = list(range(n)) + [n - 1] * 3 + list(range(n - 2, 0, -1)) + [0] * 3
            write_clip(page, "hero-dark", seq, 4, {"at": mid, "w": 640, "h": 253, "fy": 0.5},
                       {"w": 960, "h": 380}, {"tags": tags, "lines": lines})
            C.go(page, "neb")
            C.wait_scene(page, "neb")
            page.click("#btn-mode")
            C.wait_scene(page, "ts")
            page.wait_for_timeout(500)
            o = page.evaluate("() => { const c = window.__alt5.current.ctrl; c.setFrame(0);"
                              " const a = c.viewer.getModel().selectedAtoms({elem: 'O'}).find((x) => !x.properties.image);"
                              " return {x: a.x, y: a.y, z: a.z}; }")
            nts = page.evaluate("() => window.__alt5.current.ctrl.nFrames")
            freq = next(k["value"] for k in neb["key_numbers"] if k["label"].startswith("TS imaginary frequency"))
            write_clip(page, "ts_mode", list(range(nts)), 20, {"at": o, "w": 260, "h": 173, "fy": 0.5},
                       {"w": 480, "h": 320}, {"text": f"imaginary mode {freq:.1f}i cm⁻¹; arrows: mode direction"})
            if errors:
                raise RuntimeError(f"page errors: {errors}")
        finally:
            b.close()


def write_clip(page, stem, seq, fps, crop, out, labels) -> None:
    import base64
    tmp = Path(tempfile.mkdtemp(prefix=f"alt5-{stem}-"))
    try:
        cache = {}
        for i, k in enumerate(seq):
            if k not in cache:
                uri = page.evaluate(FRAME_JS, [k, crop, out, labels])
                cache[k] = base64.b64decode(uri.split(",", 1)[1])
            (tmp / f"f{i:05d}.png").write_bytes(cache[k])
        gif = MEDIA / f"{stem}.gif"
        vf = ("split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
              "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle")
        subprocess.run([FFMPEG, "-y", "-v", "error", "-framerate", str(fps), "-i", str(tmp / "f%05d.png"),
                        "-filter_complex", vf, "-loop", "0", str(gif)], check=True)
        shutil.copy(tmp / "f00000.png", MEDIA.parent / "shots" / f"clip-{stem}-first.png") if (MEDIA.parent / "shots").is_dir() else None
        print(f"clip: {gif.relative_to(HERE)} {gif.stat().st_size / 1e6:.2f} MB, {len(seq)} frames")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def webapp_gif() -> None:
    """The whole web app recording as a GIF for the repository README.

    Original speed: the recording already carries its own on-screen speed-up
    label, which a second speed-up here would make wrong.
    """
    vf = ("fps=8,scale=720:-1:flags=lanczos,split[a][b];"
          "[a]palettegen=max_colors=96:stats_mode=diff[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    gif = MEDIA / "webapp.gif"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(DATA / "webapp.mp4"),
                    "-filter_complex", vf, "-loop", "0", str(gif)], check=True)
    print(f"webapp: {gif.relative_to(HERE)} {gif.stat().st_size / 1e6:.2f} MB")


def main() -> int:
    if sys.argv[1:] == ["--clips-only"]:
        page_clips()
        return 0
    if sys.argv[1:] == ["--webapp-gif"]:
        webapp_gif()
        return 0
    MEDIA.mkdir(exist_ok=True)
    charts()
    row = ["--width", "360", "--height", "240", "--background", DARK, "--image-tint", "0.72", "--timeout", "300"]
    cap("relax.xyz", "relax", *row, "--replicate", "3", "3", "--elevation", "22", "--azimuth", "-30",
        "--zoom", "1.5", "--fps", "3", "--hold", "1.4")
    keep_gif_only("relax")
    cap("md_bilayer_built.xyz", "md_bilayer_built", *row, "--replicate", "2", "2", "--style", "ballstick",
        "--elevation", "22", "--azimuth", "-30", "--hold", "0")
    keep_gif_only("md_bilayer_built")
    page_clips()  # hero-dark.gif and ts_mode.gif, cut from the live page

    # Page still for software-WebGL browsers: the TS image (climbing image) of
    # the band, at the live hero's framing, as a JPEG.
    neb = json.loads((DATA / "neb.json").read_text())
    lines = (DATA / "neb_band.xyz").read_text().splitlines()
    n = int(lines[0])
    ts = neb["ts_image_index"]
    tmp = Path(tempfile.mkdtemp(prefix="alt5-poster-"))
    try:
        (tmp / "ts.xyz").write_text("\n".join(lines[ts * (n + 2):(ts + 1) * (n + 2)]) + "\n")
        subprocess.run([PY, str(CAPTURE), str(tmp / "ts.xyz"), str(tmp / "poster"), "--width", "1440",
                        "--height", "900", "--replicate", "3", "3", "--elevation", "44", "--azimuth", "-24",
                        "--zoom", "2.0", "--image-tint", "0.72", "--background", DARK, "--hold", "0",
                        "--scale", "1", "--timeout", "300", "--keep-frames", str(tmp / "frames")], check=True)
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(tmp / "frames" / "f00000.png"), "-q:v", "3",
                        str(MEDIA / "hero-poster.jpg")], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Web app: a legible still of the finished job (result, plots), linked to the MP4.
    web = json.loads((DATA / "webapp.json").read_text())
    length = next(k["value"] for k in web["key_numbers"] if k["label"] == "video length")
    t = length - 6.0  # the result view is on screen during the last seconds of the recording
    subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{t:.1f}", "-i", str(DATA / "webapp.mp4"),
                    "-frames:v", "1", "-vf", "crop=960:760:320:40", "-q:v", "3",
                    str(MEDIA / "webapp-result.jpg")], check=True)
    webapp_gif()
    (MEDIA / "hero-light.gif").unlink(missing_ok=True)
    for f in sorted(MEDIA.iterdir()):
        print(f"{f.name:28s} {f.stat().st_size / 1e6:6.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
