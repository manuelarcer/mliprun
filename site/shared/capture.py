"""Render a multi-frame extended-XYZ trajectory to GIF + MP4 with the shared viewer.

The same viewer.js that the landing pages use draws every frame in headless
Chromium (Playwright), so README media and the live pages look identical.
Frames are stepped deterministically through window.__scene.setFrame(i) and
read back as PNG from the WebGL canvas, then ffmpeg encodes:

  OUT_STEM.gif  palettegen/paletteuse, loops forever
  OUT_STEM.mp4  H.264, yuv420p, +faststart, no audio track

Each file is kept <= 5 MB. GIF: at full width it tries 256, then 128
colours, then 96 colours at 0.75x fps, then 64 colours at 0.6x fps; if none
fits it repeats that sequence at 0.85, 0.7, 0.55 and 0.45x width. MP4: CRF 18,
22, 26, 30 at full width, then the same at 0.8, 0.6 and 0.45x width.

--keep-frames DIR refuses a directory that already holds f*.png frames.
The Chromium build used is printed at the end of each run.

No local web server is used: every request from the page is answered by a
Playwright route handler (works inside sandboxes that forbid binding a port).
3Dmol.js is served from site/.cache/ when a copy is present there (it is
downloaded once from the pinned CDN URL), otherwise fetched from the CDN.

Usage:
    site/.venv/bin/python site/shared/capture.py XYZ OUT_STEM \
        [--frames N] [--width W --height H] [--replicate NX NY] [--view preset]
        [--style spacefill|ballstick] [--fps F] [--hold S] [--bounce]
        [--spin DEG] [--background CSS] [--elevation D --azimuth D --zoom Z]
        [--image-tint T] [--keep-frames DIR]
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
CACHE = SITE / ".cache"
THREEDMOL_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"
THREEDMOL_CACHE = CACHE / "3Dmol-2.4.2-min.js"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
ORIGIN = "http://mliprun.capture"
LIMIT = 5 * 1024 * 1024


# ---------------------------------------------------------------- planning
def frame_plan(n_src: int, n_frames: int | None, bounce: bool = False,
               hold: int = 0) -> list[int]:
    """Source-frame index for every output frame.

    Picks n_frames evenly spaced source frames (always first and last), then
    `hold` repeats of the last; with bounce, plays back to frame 0 and holds it.
    """
    if n_src < 1:
        raise ValueError("no frames")
    n = n_src if not n_frames or n_frames >= n_src else max(2, n_frames)
    if n == 1:
        fwd = [0]
    else:
        fwd = sorted({round(k * (n_src - 1) / (n - 1)) for k in range(n)})
    plan = fwd + [fwd[-1]] * hold
    if bounce and len(fwd) > 1:
        plan += fwd[-2:0:-1] + [0] * max(1, hold)
    return plan


# ---------------------------------------------------------------- browser
def _chromium_candidates():
    root = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH",
                               Path.home() / "Library/Caches/ms-playwright"))
    pats = ["chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell",
            "chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/"
            "Google Chrome for Testing",
            "chromium-*/chrome-linux*/chrome"]
    found = []
    for p in pats:
        found += sorted(glob.glob(str(root / p)), reverse=True)
    return found


LAST_CHROMIUM = {"executable": None, "version": None}


def _launch(pw):
    browser = _launch_any(pw)
    LAST_CHROMIUM["version"] = browser.version
    return browser


def _launch_any(pw):
    # SwiftShader (software WebGL) renders identically on every machine and
    # needs no GPU, so captured frames are reproducible.
    args = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
            "--ignore-gpu-blocklist"]
    exe = os.environ.get("MLIPRUN_CHROMIUM")
    if exe:
        LAST_CHROMIUM["executable"] = exe
        return pw.chromium.launch(executable_path=exe, args=args)
    try:
        LAST_CHROMIUM["executable"] = pw.chromium.executable_path
        return pw.chromium.launch(args=args)
    except Exception as err:  # bundled revision missing: use a cached one
        for cand in _chromium_candidates():
            try:
                LAST_CHROMIUM["executable"] = cand
                return pw.chromium.launch(executable_path=cand, args=args)
            except Exception:
                continue
        raise RuntimeError(f"no usable Chromium found: {err}") from err


def _threedmol_bytes() -> bytes | None:
    if not THREEDMOL_CACHE.exists():
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(THREEDMOL_URL, timeout=30) as r:
                THREEDMOL_CACHE.write_bytes(r.read())
        except Exception:
            return None
    return THREEDMOL_CACHE.read_bytes()


def _page_html(width: int, height: int, background: str, opts: dict,
               below_px: int = 0) -> str:
    bg = "transparent" if background == "transparent" else background
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>html,body{{margin:0;padding:0;background:{bg};{'overflow:hidden' if not below_px else ''}}}
#scene{{width:{width}px;height:{height}px}}</style></head>
<body><div id="scene"></div><div id="below" style="height:{below_px}px"></div>
<script type="module">
import {{ loadScene }} from "./viewer.js";
const opts = {json.dumps(opts)};
loadScene(document.getElementById("scene"), "./scene.xyz", opts)
  .catch((e) => {{ window.__sceneError = String(e && e.stack || e); }});
</script></body></html>"""


@contextlib.contextmanager
def scene_page(xyz_text: str, width: int = 640, height: int = 480, scale: float = 2,
               reduced_motion: bool = False, background: str = "#ffffff",
               timeout_s: float = 120.0, has_touch: bool = False,
               is_mobile: bool = False, below_px: int = 0, **opts):
    """Yield a Playwright page whose window.__scene is a ready viewer.

    Extra keyword arguments are passed to loadScene as opts (camelCase keys).
    timeout_s bounds page load and scene readiness (auto-fit is slow under
    SwiftShader on a loaded machine). has_touch / is_mobile emulate a phone;
    below_px adds page content under the scene (for scroll tests).
    """
    timeout_ms = int(timeout_s * 1000)
    from playwright.sync_api import sync_playwright

    scene_opts = {"capture": True, "autoplay": False, "background": background,
                  "pauseOffscreen": False, "interactive": False, **opts}
    html = _page_html(width, height, background, scene_opts, below_px)
    viewer_js = (HERE / "viewer.js").read_bytes()
    lib = _threedmol_bytes()

    def handle(route):
        url = route.request.url
        if url.startswith(THREEDMOL_URL.rsplit("/", 1)[0]) and lib is not None:
            return route.fulfill(body=lib, content_type="application/javascript")
        if url.startswith(ORIGIN):
            path = url[len(ORIGIN):].split("?")[0]
            if path in ("/", "/index.html"):
                return route.fulfill(body=html, content_type="text/html")
            if path == "/viewer.js":
                return route.fulfill(body=viewer_js, content_type="application/javascript")
            if path == "/scene.xyz":
                return route.fulfill(body=xyz_text, content_type="text/plain")
            return route.fulfill(status=404, body="not found")
        return route.continue_()

    with sync_playwright() as pw:
        browser = _launch(pw)
        try:
            ctx = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=scale, has_touch=has_touch, is_mobile=is_mobile,
                reduced_motion="reduce" if reduced_motion else "no-preference")
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.route("**/*", handle)
            page.goto(ORIGIN + "/index.html", timeout=timeout_ms)
            page.wait_for_function(
                "() => (window.__scene && window.__scene.ready) || window.__sceneError",
                timeout=timeout_ms)
            err = page.evaluate("() => window.__sceneError || null")
            if err:
                raise RuntimeError(f"viewer failed: {err}")
            if errors:
                raise RuntimeError("page errors: " + "; ".join(errors))
            yield page
        finally:
            browser.close()


# ---------------------------------------------------------------- encoding
def _run(cmd, timeout=600):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed (exit {e.returncode}): {' '.join(cmd)}\n"
                           f"{(e.stderr or '').strip()}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out after {timeout} s: {' '.join(cmd)}") from e


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def encode_gif(frames_dir: Path, fps: float, out: Path, width: int) -> dict:
    """GIF under LIMIT. Returns the settings that were used."""
    ladder = []
    for wf in (1.0, 0.85, 0.7, 0.55, 0.45):
        for colors, fpsf in ((256, 1.0), (128, 1.0), (96, 0.75), (64, 0.6)):
            ladder.append((wf, colors, fpsf))
    for wf, colors, fpsf in ladder:
        w = _even(width * wf)
        out_fps = fps * fpsf
        vf = (f"fps={out_fps:g},scale={w}:-2:flags=lanczos,split[a][b];"
              f"[a]palettegen=max_colors={colors}:stats_mode=full:reserve_transparent=0[p];"
              f"[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle")
        _run([FFMPEG, "-y", "-v", "error", "-framerate", f"{fps:g}",
              "-i", str(frames_dir / "f%05d.png"), "-filter_complex", vf,
              "-loop", "0", str(out)])
        if out.stat().st_size <= LIMIT:
            return {"width": w, "colors": colors, "fps": out_fps,
                    "bytes": out.stat().st_size}
    raise RuntimeError(f"could not get {out} under 5 MB")


def encode_mp4(frames_dir: Path, fps: float, out: Path, width: int) -> dict:
    """H.264 MP4 under LIMIT: yuv420p, faststart, no audio."""
    for wf in (1.0, 0.8, 0.6, 0.45):
        w = _even(width * wf)
        for crf in (18, 22, 26, 30):
            _run([FFMPEG, "-y", "-v", "error", "-framerate", f"{fps:g}",
                  "-i", str(frames_dir / "f%05d.png"),
                  "-vf", f"scale={w}:-2:flags=lanczos,format=yuv420p",
                  "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                  "-profile:v", "high", "-pix_fmt", "yuv420p",
                  "-movflags", "+faststart", "-an", str(out)])
            if out.stat().st_size <= LIMIT:
                return {"width": w, "crf": crf, "bytes": out.stat().st_size}
    raise RuntimeError(f"could not get {out} under 5 MB")


# ---------------------------------------------------------------- pipeline
def render_frames(xyz: Path, frames_dir: Path, *, n_frames=None, width=640, height=480,
                  replicate=(2, 2), view="tilted", style="spacefill", fps=20.0,
                  hold=0.8, bounce=False, spin=0.0, background="#ffffff",
                  camera=None, image_tint=0.0, scale=2,
                  timeout_s=120.0) -> list[int]:
    """Write f00000.png ... into frames_dir; return the frame plan."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(frames_dir.glob("f*.png"))
    if stale:
        raise FileExistsError(f"{frames_dir} already holds {len(stale)} f*.png frames; "
                              "use an empty directory (stale frames would be encoded)")
    text = Path(xyz).read_text()
    opts = {"replicate": list(replicate), "view": view, "style": style,
            "imageTint": image_tint, "fitSpin": bool(spin)}
    with scene_page(text, width=width, height=height, scale=scale,
                    background=background, timeout_s=timeout_s, **opts) as page:
        n_src = page.evaluate("() => window.__scene.nFrames")
        if camera:
            page.evaluate("(v) => window.__scene.setView(v)", camera)
        base_az = page.evaluate("() => window.__scene.getView().azimuth")
        plan = frame_plan(n_src, n_frames, bounce=bounce, hold=round(hold * fps))
        cache = {}
        for k, idx in enumerate(plan):
            if spin:
                page.evaluate("(a) => window.__scene.setView({azimuth: a}, {fit: false})",
                              base_az + spin * k / len(plan))
                key = None
            else:
                key = idx
            if key is None or key not in cache:
                uri = page.evaluate("(i) => { window.__scene.setFrame(i);"
                                    " return window.__scene.snapshot(); }", idx)
                data = base64.b64decode(uri.split(",", 1)[1])
                if key is not None:
                    cache[key] = data
            else:
                data = cache[key]
            (frames_dir / f"f{k:05d}.png").write_bytes(data)
    return plan


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("xyz", type=Path)
    p.add_argument("out_stem", type=Path)
    p.add_argument("--frames", type=int, default=None,
                   help="number of source frames to use (evenly spaced; default all)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--replicate", type=int, nargs=2, default=[2, 2], metavar=("NX", "NY"))
    p.add_argument("--view", default="tilted", help="tilted | top | side | hero")
    p.add_argument("--elevation", type=float, default=None)
    p.add_argument("--azimuth", type=float, default=None)
    p.add_argument("--zoom", type=float, default=None)
    p.add_argument("--style", default="spacefill", choices=["spacefill", "ballstick"])
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--hold", type=float, default=0.8, help="seconds held on the last frame")
    p.add_argument("--bounce", action="store_true", help="play forward then back (NEB, modes)")
    p.add_argument("--spin", type=float, default=0.0,
                   help="total azimuth turn over the clip (deg); 360 loops seamlessly")
    p.add_argument("--background", default="#ffffff",
                   help="CSS colour (GIF/MP4 have no alpha; default white)")
    p.add_argument("--image-tint", type=float, default=0.0,
                   help="0..1 fade of periodic images toward the background")
    p.add_argument("--scale", type=float, default=2.0,
                   help="supersampling factor (render at scale x size, downscale)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="seconds allowed for page load and scene auto-fit (default 120)")
    p.add_argument("--keep-frames", type=Path, default=None,
                   help="write the PNG frames here (must be new or hold no f*.png)")
    a = p.parse_args(argv)

    camera = {k: v for k, v in (("elevation", a.elevation), ("azimuth", a.azimuth),
                                ("zoom", a.zoom)) if v is not None}
    a.out_stem.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="mliprun-capture-"))
    frames_dir = a.keep_frames or tmp / "frames"
    try:
        plan = render_frames(a.xyz, frames_dir, n_frames=a.frames, width=a.width,
                             height=a.height, replicate=a.replicate, view=a.view,
                             style=a.style, fps=a.fps, hold=a.hold, bounce=a.bounce,
                             spin=a.spin, background=a.background, camera=camera or None,
                             image_tint=a.image_tint, scale=a.scale,
                             timeout_s=a.timeout)
        gif = encode_gif(frames_dir, a.fps, a.out_stem.with_suffix(".gif"), a.width)
        mp4 = encode_mp4(frames_dir, a.fps, a.out_stem.with_suffix(".mp4"), a.width)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"chromium: {LAST_CHROMIUM['version']} ({LAST_CHROMIUM['executable']})")
    print(f"frames: {len(plan)} output ({len(set(plan))} distinct source frames)")
    print(f"gif: {a.out_stem.with_suffix('.gif')}  {gif['bytes'] / 1e6:.2f} MB  "
          f"width {gif['width']} px, {gif['colors']} colours, {gif['fps']:g} fps")
    print(f"mp4: {a.out_stem.with_suffix('.mp4')}  {mp4['bytes'] / 1e6:.2f} MB  "
          f"width {mp4['width']} px, crf {mp4['crf']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
