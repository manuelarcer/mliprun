"""Render the README media for alternative 2 ("story") from the data pack.

    site/.venv/bin/python site/alt-2-story/render_media.py      (outside the sandbox)

Uses site/shared/capture.py (scene_page, encode_gif) so the frames come from the same
viewer.js, with the same cameras, as the landing page. It drives scene_page directly
(not the CLI) because the committee still needs per-atom colours set in the page.

Writes, for light (#ffffff) and GitHub-dark (#0d1117) backgrounds:
  (relax_gif() can render a cropped relax GIF with readouts; not used in the README)
  media/committee-{light,dark}.png  atoms coloured by final per-atom force sigma
  media/neb-{light,dark}.gif        final band, fcc -> hcp -> fcc
  media/ts-{light,dark}.gif         imaginary mode at the TS (illustration)
  media/water-{light,dark}.png      bilayer as built (static structure, not MD)
  media/webapp.gif                  the web app recording, from data/webapp.mp4 (960 px wide)
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
import capture  # noqa: E402

DATA = HERE / "data"
MEDIA = HERE / "media"
BGS = {"light": "#ffffff", "dark": "#0d1117"}
RAMPS = {"light": ["#d3d9e0", "#9fb8cc", "#5d8fb3", "#2a6593", "#123f66"],  # as main.js
         "dark": ["#5a6470", "#46708f", "#4f95c4", "#80c3ea", "#d2ecfb"]}
INK = {"light": ("#17202b", "#566272"), "dark": ("#e6edf3", "#9198a1")}


def parse_xyz(text):
    """Minimal extxyz reader: [(symbols, positions, cell)] per frame."""
    lines, i, out = text.splitlines(), 0, []
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        com = lines[i + 1]
        cell = None
        if 'Lattice="' in com:
            v = [float(x) for x in com.split('Lattice="')[1].split('"')[0].split()]
            cell = [v[0:3], v[3:6], v[6:9]]
        sy, pos = [], []
        for ln in lines[i + 2:i + 2 + n]:
            t = ln.split()
            sy.append(t[0])
            pos.append([float(t[1]), float(t[2]), float(t[3])])
        out.append((sy, pos, cell))
        i += n + 2
    return out


SLICE_HALF_W, SLICE_FRONT, SLICE_DEPTH = 6.5, 0.5, 6.0  # Å, as main.js


def crop_relax(frames):
    """Same slice as main.js cropRelax: the O atom plus every Pt within SLICE_HALF_W along
    x and from SLICE_FRONT in front of the O to SLICE_DEPTH behind it along +y (away from
    the camera at azimuth 0). Positions unchanged. Returns xyz text and an empty dict."""
    sy, p0, cell = frames[0]
    o = max(i for i, s in enumerate(sy) if s == "O")
    picks = []
    for i in range(len(sy)):
        for u in (-2, -1, 0, 1, 2):
            for v in (-2, -1, 0, 1, 2):
                sh = [u * cell[0][k] + v * cell[1][k] for k in range(3)]
                if i == o:
                    if u == 0 and v == 0:
                        picks.append((i, sh))
                    continue
                dx = p0[i][0] + sh[0] - p0[o][0]
                dy = p0[i][1] + sh[1] - p0[o][1]
                if abs(dx) <= SLICE_HALF_W and -SLICE_FRONT <= dy <= SLICE_DEPTH:
                    picks.append((i, sh))
    text = ""
    for k, (s2, pk, _) in enumerate(frames):
        text += f"{len(picks)}\nProperties=species:S:1:pos:R:3 step={k}\n"
        text += "\n".join(f"{s2[i]} " + " ".join(f"{pk[i][c] + sh[c]:.6f}" for c in range(3)) for i, sh in picks) + "\n"
    return text, {}


def o_heights(frames):
    out = []
    for sy, pos, _ in frames:
        zpt = [p[2] for s, p in zip(sy, pos) if s == "Pt"]
        top = max(zpt)
        layer = [z for z in zpt if z > top - 0.6]
        zo = max(p[2] for s, p in zip(sy, pos) if s == "O")
        out.append(zo - sum(layer) / len(layer))
    return out


def relax_gif(relax, plan, fps, width, height):
    frames = parse_xyz((DATA / relax["frames"]).read_text())
    text, g = crop_relax(frames)
    h = o_heights(frames)
    s = relax["series"]
    n = len(s["step"]) - 1
    for theme, bg in BGS.items():
        ink, muted = INK[theme]
        tmp = Path(tempfile.mkdtemp(prefix="story-media-"))
        raw = tmp / "raw"
        raw.mkdir()
        try:
            with capture.scene_page(text, width=width, height=height, scale=2, background=bg,
                                    replicate=[1, 1], view={"elevation": 18, "azimuth": 0},
                                    fill=0.8) as page:
                page.evaluate("""([g, col]) => {
                  const d = document.createElement('div'); d.id = 'hud';
                  d.style.cssText = 'position:absolute;left:14px;top:10px;font:13px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:' + col;
                  document.body.appendChild(d);
                }""", [g, muted])
                for k, idx in enumerate(plan):
                    page.evaluate("(i) => window.__scene.setFrame(i)", idx)
                    rows = [("step", f"{idx} of {n}"), ("O height", f"{h[idx]:.3f} Å"),
                            ("fmax", f"{s['fmax_eV_per_A'][idx]:.4f} eV/Å"),
                            ("E − E₀", f"{s['energy_eV'][idx] - s['energy_eV'][0]:+.3f} eV".replace("-", "−"))]
                    html = "".join(f"<div><span style='display:inline-block;width:72px'>{a}</span>"
                                   f"<b style='color:{ink}'>{b}</b></div>" for a, b in rows)
                    html += "<div style='margin-top:4px'>cropped; positions not scaled</div>"
                    page.evaluate("(h) => { document.getElementById('hud').innerHTML = h; }", html)
                    page.screenshot(path=str(raw / f"f{k:05d}.png"))
            info = capture.encode_gif(raw, fps, MEDIA / f"relax-{theme}.gif", width)
            print(f"relax-{theme}.gif  {info['bytes'] / 1e6:.2f} MB  {info}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)



RECOLOR_JS = """([sig, ramp, oColor, radii]) => {
  const v = window.__scene.viewer, smax = Math.max(...sig);
  const hex = (h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));
  const col = (t) => { t = Math.max(0, Math.min(1, t)) * (ramp.length - 1);
    const i = Math.min(ramp.length - 2, Math.floor(t)), f = t - i, a = hex(ramp[i]), b = hex(ramp[i + 1]);
    return '#' + a.map((x, k) => Math.round(x * (1 - f) + b[k] * f).toString(16).padStart(2, '0')).join(''); };
  for (const e of ['Pt', 'O'])
    v.setStyle({elem: e}, {sphere: {radius: radii[e], colorfunc: (at) => e === 'O' ? oColor : col((sig[at.properties.src] || 0) / smax)}});
  v.render();
}"""


def frames_to_dir(xyz: Path, plan, out: Path, *, width, height, bg, view, style="spacefill",
                  recolor=None):
    raw = out / "raw"
    raw.mkdir(parents=True)
    with capture.scene_page(xyz.read_text(), width=width, height=height, scale=2,
                            background=bg, replicate=[2, 2], view=view, style=style,
                            fill=0.88) as page:
        for k, idx in enumerate(plan):
            page.evaluate("(i) => window.__scene.setFrame(i)", idx)
            if recolor:
                page.evaluate(RECOLOR_JS, recolor)
            uri = page.evaluate("() => window.__scene.snapshot()")
            (raw / f"f{k:05d}.png").write_bytes(base64.b64decode(uri.split(",", 1)[1]))
    return raw


def gif(xyz, stem, plan, fps, **kw):
    for theme, bg in BGS.items():
        tmp = Path(tempfile.mkdtemp(prefix="story-media-"))
        try:
            d = frames_to_dir(xyz, plan, tmp, bg=bg, **kw)
            info = capture.encode_gif(d, fps, MEDIA / f"{stem}-{theme}.gif", kw["width"])
            print(f"{stem}-{theme}.gif  {info['bytes'] / 1e6:.2f} MB  {info}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def still(xyz, stem, idx, **kw):
    for theme, bg in BGS.items():
        tmp = Path(tempfile.mkdtemp(prefix="story-media-"))
        try:
            kw2 = dict(kw)
            if kw2.get("recolor"):
                kw2["recolor"] = [kw2["recolor"][0], RAMPS[theme], *kw2["recolor"][2:]]
            d = frames_to_dir(xyz, [idx], tmp, bg=bg, **kw2)
            # downscale the 2x supersampled frame to the display width
            out = MEDIA / f"{stem}-{theme}.png"
            subprocess.run([capture.FFMPEG, "-v", "error", "-y", "-i", str(sorted(d.glob("f*.png"))[0]),
                            "-vf", f"scale={kw['width']}:-2:flags=lanczos", "-frames:v", "1", "-update", "1", str(out)], check=True)
            print(f"{out.name}  {out.stat().st_size / 1e6:.2f} MB")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def hold(plan, n):
    return plan + [plan[-1]] * n


def main():
    MEDIA.mkdir(exist_ok=True)
    committee = json.loads((DATA / "committee.json").read_text())
    neb = json.loads((DATA / "neb.json").read_text())
    md = json.loads((DATA / "md.json").read_text())
    relax = json.loads((DATA / "relax.json").read_text())
    W, H = 560, 320

    n_relax = len(relax["series"]["step"])
    # relax: the README uses the chart alone. The atoms move 0.07 Å, so even the cropped
    # view with readouts reads as a still image (design review, fix round 1).
    # relax_gif(relax, hold(list(range(n_relax)), 6), 3, 720, H)

    sig = committee["series"]["peratom_final_sigma_free_eV_per_A"]
    still(DATA / relax["frames"], "committee", n_relax - 1, width=W, height=H,
          view="tilted",
          recolor=[sig, None, "#e0352b", {"Pt": 1.39, "O": 0.74}])

    n_img = len(neb["series"]["image_index"])
    fwd = list(range(n_img))
    gif(DATA / neb["frames"], "neb", hold(fwd, 4) + hold(fwd[-2::-1], 4), 5,
        width=W, height=H, view={"elevation": 64, "azimuth": -30})

    gif(DATA / neb["frames_extra"]["ts_mode"], "ts", list(range(30)), 24,
        width=W, height=H, view="tilted")

    still(DATA / md["structures"]["built"]["file"], "water", 0, width=W, height=H,
          view="tilted", style="ballstick")

    # web app recording -> GIF (GitHub READMEs do not play a relative-path MP4 inline)
    out = MEDIA / "webapp.gif"
    for width, fps, cols in ((960, 6, 128), (960, 5, 96), (960, 4, 64), (800, 4, 64)):
        vf = (f"fps={fps},scale={width}:-2:flags=lanczos,split[a][b];"
              f"[a]palettegen=max_colors={cols}:stats_mode=diff[p];"
              f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run([capture.FFMPEG, "-v", "error", "-y", "-i", str(DATA / "webapp.mp4"),
                        "-filter_complex", vf, "-loop", "0", str(out)], check=True)
        if out.stat().st_size <= capture.LIMIT:
            break
    print(f"webapp.gif  {out.stat().st_size / 1e6:.2f} MB  width {width} fps {fps} colours {cols}")
    print(f"chromium: {capture.LAST_CHROMIUM['version']}")


if __name__ == "__main__":
    main()
