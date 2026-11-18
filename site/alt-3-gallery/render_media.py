"""Render the README media for the gallery (alternative 3) from the data pack.

Run outside the sandbox (Chromium cannot start inside it):
    site/.venv/bin/python site/alt-3-gallery/render_media.py

Writes media/<name>-light.* and media/<name>-dark.* (GitHub light / dark
backgrounds) with site/shared/capture.py, so README and page use one viewer:
  relax      13 optimizer steps, looping            (GIF)
  neb        9-image final band, forward and back   (GIF)
  tsmode     imaginary mode at the TS, 1x1 cell     (GIF)
  committee  start geometry, top view, free atoms coloured by final force sigma, O red (PNG)
  md         water bilayer as built, static         (PNG; MD pending)
  webapp     three cropped stills of the recording (PNG) and a README composite
Every GIF is kept <= 2 MB (re-encoded narrower if not).

Elevations match main.js; since viewer.js dee6bbe, snapshot() and capture.py
return the screen orientation, so no post-processing flip is applied.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
DATA = HERE / "data"
MEDIA = HERE / "media"
sys.path.insert(0, str(SITE / "shared"))
import capture  # noqa: E402

PY = sys.executable
CAP = str(SITE / "shared" / "capture.py")
THEMES = {"light": "#ffffff", "dark": "#0d1117"}
LIMIT = 2 * 1024 * 1024
FFMPEG = capture.FFMPEG


def run(cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def shrink_gif(path: Path, width: int):
    """Re-encode a GIF narrower until it is <= 2 MB."""
    w = width
    while path.stat().st_size > LIMIT and w > 200:
        w = int(w * 0.85) // 2 * 2
        tmp = path.with_suffix(".tmp.gif")
        run([FFMPEG, "-y", "-loglevel", "error", "-i", path, "-filter_complex",
             f"scale={w}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=sierra2_4a",
             "-loop", "0", tmp])
        tmp.replace(path)


def anim(name, xyz, theme, bg, elevation, extra, width=420, height=300):
    stem = MEDIA / f"{name}-{theme}"
    run([PY, CAP, DATA / xyz, stem, "--width", width, "--height", height, "--background", bg,
         "--elevation", elevation, "--azimuth", -30, *extra])
    stem.with_suffix(".mp4").unlink(missing_ok=True)  # README uses GIF only
    shrink_gif(stem.with_suffix(".gif"), width)


def still(name, xyz_text, theme, bg, elevation, js=None, width=420, height=300, **opts):
    out = MEDIA / f"{name}-{theme}.png"
    view = {"elevation": elevation, "azimuth": -30}
    with capture.scene_page(xyz_text, width=width, height=height, scale=2, background=bg, view=view, **opts) as page:
        if js:
            page.evaluate(js)
        url = page.evaluate("() => window.__scene.snapshot()")
    out.write_bytes(base64.b64decode(url.split(",", 1)[1]))
    print("wrote", out, out.stat().st_size)


COMMITTEE_JS = """() => {
  const s = window.__scene, sig = %s, mx = Math.max(...sig);
  const stops = [[0xd8,0xe2,0xff],[0x6b,0x7c,0xf0],[0x45,0x27,0xb8]];
  const ramp = (t) => { const u = Math.max(0, Math.min(1, t)) * 2, k = Math.min(1, Math.floor(u)), f = u - k;
    return '#' + stops[k].map((v, j) => Math.round(v + (stops[k+1][j] - v) * f).toString(16).padStart(2, '0')).join(''); };
  const R = {Pt: 1.39, O: 0.74, H: 0.4};
  for (const e of Object.keys(R)) s.viewer.setStyle({elem: e}, {sphere: {radius: R[e],
    colorfunc: (a) => { const v = sig[a.properties.src]; return e === 'O' ? '#e0352b' : v > 0 ? ramp(v / mx) : '#9da2ab'; }}});
  s.viewer.render();
}"""


# Web app recording: three stills (submit, running, done), cropped to the
# status regions of the 1280x800 screen recording. Times and crop boxes are
# display choices, picked by looking at the frames (shots/vid/).
STILLS = {
    "submit": (3.5, (380, 406, 0, 234)),    # New calculation form: Run on cos-cluster-uma, UMA, oc20, cuda
    "running": (12.0, (886, 236, 380, 44)),  # queue + job header: running, Runs on cos-cluster-uma
    "done": (20.0, (886, 236, 380, 44)),     # queue + job header: done, converged, wall time
}


def webapp_stills(video: Path):
    for name, (t, (w, h, x, y)) in STILLS.items():
        out = MEDIA / f"webapp-{name}.png"
        run([FFMPEG, "-y", "-loglevel", "error", "-ss", t, "-i", video, "-frames:v", 1,
             "-vf", f"crop={w}:{h}:{x}:{y}", "-update", 1, out])
    # README composite (>= 960 px wide): submit on the left, running over done
    # on the right, on a GitHub-theme ground; the dark one is dimmed a little.
    for theme, (ground, dim) in {"light": ("0xf6f8fa", "1"), "dark": ("0x161b22", "0.82")}.items():
        out = MEDIA / f"webapp-stills-{theme}.png"
        f = (f"[0]format=rgb24,scale=-2:484,pad=iw+16:ih:0:0:color={ground}[l];"
             f"[1]format=rgb24,pad=iw:ih+12:0:0:color={ground}[r1];[2]format=rgb24[r2];[r1][r2]vstack=2[r];"
             f"[l][r]hstack=2,pad=iw+24:ih+24:12:12:color={ground},"
             f"colorchannelmixer=rr={dim}:gg={dim}:bb={dim}")
        run([FFMPEG, "-y", "-loglevel", "error",
             "-i", MEDIA / "webapp-submit.png", "-i", MEDIA / "webapp-running.png", "-i", MEDIA / "webapp-done.png",
             "-filter_complex", f, "-frames:v", 1, "-update", 1, out])


def main():
    only = sys.argv[1:] and sys.argv[1] == "--only-webapp"
    MEDIA.mkdir(exist_ok=True)
    committee = json.loads((DATA / "committee.json").read_text())
    md = json.loads((DATA / "md.json").read_text())
    webapp = json.loads((DATA / "webapp.json").read_text())
    relax_txt = (DATA / "relax.xyz").read_text().splitlines()
    nat = int(relax_txt[0])
    frame0 = "\n".join(relax_txt[: nat + 2]) + "\n"
    sig = committee["series"]["peratom_final_sigma_free_eV_per_A"]

    webapp_stills(DATA / webapp["video"])
    if only:
        return
    for theme, bg in THEMES.items():
        # views match main.js
        anim("relax", "relax.xyz", theme, bg, 20, ["--fps", "5", "--hold", "1.4", "--image-tint", "0.55"])
        anim("neb", "neb_band.xyz", theme, bg, 52, ["--fps", "4", "--hold", "1.0", "--bounce", "--image-tint", "0.55"])
        anim("tsmode", "ts_mode.xyz", theme, bg, 30, ["--fps", "20", "--hold", "0", "--replicate", "1", "1"], width=300, height=240)
        still("committee", frame0, theme, bg, 90, js=COMMITTEE_JS % json.dumps(sig))
        built = (DATA / md["structures"]["built"]["file"]).read_text()
        still("md", built, theme, bg, 26, style="ballstick")
    for f in sorted(MEDIA.iterdir()):
        print(f"{f.name:28s} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
