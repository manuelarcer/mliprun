"""Tests for the shared viewer (viewer.js) and media capture (capture.py).

Run: site/.venv/bin/python -m pytest site/shared/test_capture.py -v
Input is the SYNTHETIC fixture fixtures/relax_fixture.xyz (see make_fixture.py).
"""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "relax_fixture.xyz"
PACK = HERE.parents[1] / "docs" / "showcase" / "data" / "pack"
FFPROBE = shutil.which("ffprobe")
if FFPROBE is None:
    pytest.skip("ffprobe not on PATH", allow_module_level=True)
LIMIT = 5 * 1024 * 1024

sys.path.insert(0, str(HERE))
import capture  # noqa: E402


def _probe(path, *fields):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=" + ",".join(fields), "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("cap")
    stem = root / "relax"
    rc = capture.main([str(FIXTURE), str(stem), "--frames", "24",
                       "--width", "480", "--height", "360", "--replicate", "2", "2",
                       "--keep-frames", str(root / "frames")])
    assert rc == 0
    return stem


def test_gif_exists_small_and_animated(media):
    gif = media.with_suffix(".gif")
    assert gif.exists()
    assert 0 < gif.stat().st_size <= LIMIT
    s = _probe(gif, "nb_read_frames", "width", "height")
    assert int(s["nb_read_frames"]) > 10
    assert int(s["width"]) == 480


def test_first_and_last_frames_differ(media):
    """The animation must actually move: first and final captured frames differ."""
    pngs = sorted((media.parent / "frames").glob("f*.png"))
    assert len(pngs) > 10
    assert pngs[0].read_bytes() != pngs[-1].read_bytes()


def test_keep_frames_refuses_non_empty_dir(media):
    with pytest.raises(FileExistsError):
        capture.render_frames(FIXTURE, media.parent / "frames", n_frames=3,
                              width=160, height=120)


def test_mp4_exists_small_h264(media):
    mp4 = media.with_suffix(".mp4")
    assert mp4.exists()
    assert 0 < mp4.stat().st_size <= LIMIT
    s = _probe(mp4, "codec_name", "pix_fmt", "nb_read_frames", "width")
    assert s["codec_name"] == "h264"
    assert s["pix_fmt"] == "yuv420p"
    assert int(s["nb_read_frames"]) > 10
    # yuv420p needs even dimensions
    assert int(s["width"]) % 2 == 0


def test_mp4_has_no_audio(media):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a",
                          "-show_entries", "stream=index", "-of", "csv=p=0",
                          str(media.with_suffix(".mp4"))],
                         check=True, capture_output=True, text=True).stdout
    assert out.strip() == ""


def test_scene_api_and_reduced_motion():
    """loadScene API: nFrames, setFrame clamps, reduced motion rests on last frame."""
    xyz = FIXTURE.read_text()
    with capture.scene_page(xyz, width=320, height=240, reduced_motion=True) as page:
        state = page.evaluate(
            """() => { const s = window.__scene;
                 const out = {n: s.nFrames, frame: s.frame, rm: s.reducedMotion,
                              played: s.play(), playing: s.playing,
                              synthetic: s.frames[0].synthetic,
                              cell: s.cell.length,
                              atoms: s.viewer.getModel().selectedAtoms({}).length};
                 out.clampHi = s.setFrame(999); out.clampLo = s.setFrame(-5);
                 out.forced = s.play({force: true}); s.pause();
                 return out; }""")
    assert state["n"] == 40
    assert state["frame"] == 39          # rests on the final frame
    assert state["rm"] is True
    assert state["played"] is False and state["playing"] is False
    assert state["synthetic"] is True
    assert state["cell"] == 3
    assert state["atoms"] == 4 * 37      # default replicate is [2, 2]
    assert state["clampHi"] == 39 and state["clampLo"] == 0
    assert state["forced"] is True


def test_play_advances_and_pause_stops():
    xyz = FIXTURE.read_text()
    with capture.scene_page(xyz, width=320, height=240, fps=30, bounce=True) as page:
        seen = page.evaluate(
            """async () => { const s = window.__scene; const seen = new Set();
                 s.setFrame(0);
                 s.play();
                 for (let k = 0; k < 40; k++) {
                   await new Promise(r => setTimeout(r, 25)); seen.add(s.frame); }
                 s.pause(); const f = s.frame;
                 await new Promise(r => setTimeout(r, 200));
                 return {n: seen.size, still: s.frame === f, playing: s.playing}; }""")
    assert seen["n"] > 5
    assert seen["still"] is True and seen["playing"] is False


def test_destroy_one_scene_keeps_the_other_alive():
    """3Dmol shares one WebGL context per page: destroy() must not kill it."""
    xyz = FIXTURE.read_text()
    with capture.scene_page(xyz, width=320, height=240) as page:
        r = page.evaluate(
            """async () => { const { loadScene } = await import('./viewer.js');
                 const a = window.__scene;
                 const host = document.createElement('div');
                 host.style.cssText = 'width:200px;height:150px';
                 document.body.append(host);
                 const b = await loadScene(host, './scene.xyz', {pauseOffscreen: false});
                 const gl = b.viewer.renderer._gl; b.destroy();
                 a.setFrame(3);
                 return {lost: gl.isContextLost(), empty: host.children.length === 0,
                         n: a.snapshot().length}; }""")
    assert r["lost"] is False and r["empty"] is True and r["n"] > 1000


def test_snapshot_is_not_blank():
    """The rendered frame must contain atoms, not just background."""
    xyz = FIXTURE.read_text()
    with capture.scene_page(xyz, width=320, height=240) as page:
        frac = page.evaluate(
            """async () => { const img = new Image();
                 img.src = window.__scene.snapshot(); await img.decode();
                 const c = document.createElement('canvas');
                 c.width = img.width; c.height = img.height;
                 const g = c.getContext('2d'); g.drawImage(img, 0, 0);
                 const d = g.getImageData(0, 0, c.width, c.height).data;
                 let fg = 0;
                 for (let i = 0; i < d.length; i += 4)
                   if (Math.abs(d[i]-255) + Math.abs(d[i+1]-255) + Math.abs(d[i+2]-255) > 30) fg++;
                 return fg / (d.length / 4); }""")
    assert frac > 0.15


BBOX_JS = """async () => { const img = new Image();
  img.src = window.__scene.snapshot(); await img.decode();
  const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
  const g = c.getContext('2d'); g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, c.width, c.height).data;
  let x0 = c.width, y0 = c.height, x1 = -1, y1 = -1;
  for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
    const i = 4 * (y * c.width + x);
    if (Math.abs(d[i]-255) + Math.abs(d[i+1]-255) + Math.abs(d[i+2]-255) > 30) {
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; } }
  return [x0 / c.width, y0 / c.height, (x1 + 1) / c.width, (y1 + 1) / c.height]; }"""


@pytest.mark.parametrize("rep", [[1, 1], [2, 2], [3, 3]], ids=["r1", "r2", "r3"])
def test_autofit_frames_slab_without_cropping(rep):
    """Content must not touch the edges, must fill the frame, and be centred."""
    xyz = FIXTURE.read_text()
    with capture.scene_page(xyz, width=400, height=300, replicate=rep) as page:
        x0, y0, x1, y1 = page.evaluate(BBOX_JS)
    assert x0 > 0.01 and y0 > 0.01 and x1 < 0.99 and y1 < 0.99   # not cropped
    assert max(x1 - x0, y1 - y0) > 0.8                            # fills the frame
    assert abs((x0 + x1) / 2 - 0.5) < 0.05 and abs((y0 + y1) / 2 - 0.5) < 0.05


# ------------------------------------------------ orientation (screen truth)
CHIRAL_XYZ = """3
chirality: Pt at origin, O on +x, H on +y
Pt 0.0 0.0 0.0
O 3.0 0.0 0.0
H 0.0 3.0 0.0
"""

# Colour-class centroids (x, y as image fractions; y grows DOWN) of a PNG data URL.
CENTROIDS_JS = """async (src) => {
  const img = new Image(); img.src = src; await img.decode();
  const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
  const g = c.getContext('2d'); g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, c.width, c.height).data;
  const acc = {red: [0, 0, 0], white: [0, 0, 0], grey: [0, 0, 0]};
  for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
    const i = 4 * (y * c.width + x), r = d[i], gg = d[i+1], b = d[i+2];
    let k = null;
    if (r > 150 && gg < 90 && b < 90) k = 'red';
    else if (r > 225 && gg > 225 && b > 225) k = 'white';
    else if (r > 60 && Math.abs(r - gg) < 20 && b >= r && r < 215) k = 'grey';
    if (k) { acc[k][0] += x / c.width; acc[k][1] += y / c.height; acc[k][2]++; } }
  const out = {};
  for (const [k, v] of Object.entries(acc)) out[k] = v[2] ? [v[0] / v[2], v[1] / v[2]] : null;
  return out; }"""


def _screen_and_snapshot(page):
    """Centroids from what the screen shows (element screenshot) and from snapshot()."""
    shot = page.locator("#scene").screenshot()
    screen = "data:image/png;base64," + base64.b64encode(shot).decode()
    return (page.evaluate(CENTROIDS_JS, screen),
            page.evaluate(CENTROIDS_JS, page.evaluate("() => window.__scene.snapshot()")))


def test_top_view_chirality_on_screen_and_snapshot_agree():
    """Top view: +x right, +y up on SCREEN; snapshot() is not mirrored."""
    with capture.scene_page(CHIRAL_XYZ, width=300, height=300, background="#000000",
                            replicate=[1, 1], view="top",
                            ambientOcclusion=False) as page:
        page.evaluate("() => window.__scene.setView({elevation: 90, azimuth: 0})")
        screen, snap = _screen_and_snapshot(page)
    pt, o, h = screen["grey"], screen["red"], screen["white"]
    assert o[0] > pt[0] + 0.2 and abs(o[1] - pt[1]) < 0.05     # O to the right
    assert h[1] < pt[1] - 0.2 and abs(h[0] - pt[0]) < 0.05     # H above
    for k in ("grey", "red", "white"):
        assert abs(screen[k][0] - snap[k][0]) < 0.02, k
        assert abs(screen[k][1] - snap[k][1]) < 0.02, k


@pytest.mark.parametrize("view", ["tilted", "side", "hero"])
def test_adsorbate_above_slab_on_screen(view):
    """O sits on the top surface, so on SCREEN it must be above the Pt centroid."""
    with capture.scene_page(FIXTURE.read_text(), width=400, height=300,
                            background="#000000", replicate=[1, 1], view=view,
                            ambientOcclusion=False) as page:
        screen, snap = _screen_and_snapshot(page)
    margin = {"tilted": 0.05, "side": 0.1, "hero": 0.05}[view]
    assert screen["red"][1] < screen["grey"][1] - margin
    assert abs(screen["red"][1] - snap["red"][1]) < 0.02
    assert abs(screen["grey"][1] - snap["grey"][1]) < 0.02


# ------------------------------------------------ page styles, touch, camera
GREEN_JS = """async (src) => {
  const img = new Image(); img.src = src; await img.decode();
  const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
  const g = c.getContext('2d'); g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, c.width, c.height).data;
  let green = 0, grey = 0;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], gg = d[i+1], b = d[i+2];
    if (gg > 100 && gg > r + 50 && gg > b + 50) green++;
    else if (r > 60 && Math.abs(r - gg) < 20 && b >= r && r < 215) grey++; }
  return {green: green / (d.length / 4), grey: grey / (d.length / 4)}; }"""

RESTYLE_GREEN_PT = """() => window.__scene.setRestyle((viewer, model) => {
  viewer.setStyle({elem: 'Pt'}, {sphere: {radius: 1.39, color: '#00c000'}}); })"""


def _element_colours(page):
    shot = page.locator("#scene").screenshot()
    return page.evaluate(GREEN_JS, "data:image/png;base64," + base64.b64encode(shot).decode())


def test_restyle_survives_height_resize_frames_and_playback():
    """A page style given through restyle stays on screen after every rebuild."""
    with capture.scene_page(FIXTURE.read_text(), width=320, height=240,
                            background="#000000") as page:
        page.evaluate(RESTYLE_GREEN_PT)
        page.set_viewport_size({"width": 320, "height": 420})   # height-only resize
        page.wait_for_timeout(600)                              # debounced refit runs
        page.evaluate("() => { window.__scene.setFrame(5); window.__scene.setFrame(20); }")
        after_frames = _element_colours(page)
        page.evaluate("""async () => { const s = window.__scene; s.play({force: true});
                          await new Promise(r => setTimeout(r, 400)); s.pause(); }""")
        after_play = _element_colours(page)
        page.evaluate("() => window.__scene.setRestyle(null)")
        cleared = _element_colours(page)
    for c in (after_frames, after_play):
        assert c["green"] > 0.2 and c["grey"] < 0.02, c
    assert cleared["green"] < 0.01 and cleared["grey"] > 0.2        # control


def _swipe_up(page, x, y0, y1):
    cdp = page.context.new_cdp_session(page)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart",
                                          "touchPoints": [{"x": x, "y": y0}]})
    steps = 12
    for k in range(1, steps + 1):
        y = y0 + (y1 - y0) * k / steps
        cdp.send("Input.dispatchTouchEvent", {"type": "touchMove",
                                              "touchPoints": [{"x": x, "y": y}]})
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    page.wait_for_timeout(500)
    return page.evaluate("() => window.scrollY")


def test_touch_swipe_on_scene_scrolls_page():
    """On a phone (coarse pointer) a swipe that starts on the scene scrolls the page."""
    with capture.scene_page(FIXTURE.read_text(), width=360, height=300,
                            has_touch=True, is_mobile=True, below_px=2000,
                            interactive=True) as page:   # 3Dmol touch handlers on
        assert page.evaluate("() => matchMedia('(pointer: coarse)').matches") is True
        scrolled = _swipe_up(page, 180, 250, 60)
        assert scrolled > 50
        # Control: with touch rotation switched on, the same swipe rotates instead.
        page.evaluate("() => { window.scrollTo(0, 0); window.__scene.setTouchRotate(true); }")
        page.wait_for_timeout(200)
        assert _swipe_up(page, 180, 250, 60) == 0


def test_fixed_camera_skips_autofit_and_matches_framing():
    """getCamera() from a fitted load, passed back as camera, gives the same picture."""
    xyz = FIXTURE.read_text()
    with capture.scene_page(xyz, width=320, height=240) as page:
        cam = page.evaluate("() => window.__scene.getCamera()")
        ref = page.evaluate(BBOX_JS)
    with capture.scene_page(xyz, width=320, height=240, camera=cam) as page:
        got = page.evaluate(BBOX_JS)
    assert len(cam) == 8
    assert all(abs(a - b) < 0.02 for a, b in zip(ref, got)), (ref, got)


def test_capture_timeout_is_a_parameter():
    import inspect
    assert inspect.signature(capture.scene_page).parameters["timeout_s"].default == 120.0
    assert inspect.signature(capture.render_frames).parameters["timeout_s"].default == 120.0
    with pytest.raises(SystemExit):           # --timeout is a known flag (help exits 0)
        capture.main(["--help"])


def test_frame_plan_bounce_and_hold():
    plan = capture.frame_plan(n_src=40, n_frames=10, bounce=True, hold=2)
    fwd = [0, 4, 9, 13, 17, 22, 26, 30, 35, 39]     # 10 evenly spaced of 40
    assert plan == fwd + [39, 39] + fwd[-2:0:-1] + [0, 0]


def _count_frames(path):
    lines = path.read_text().splitlines()
    i = n = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        i += int(lines[i]) + 2
        n += 1
    return n


@pytest.mark.parametrize("name", ["relax.xyz", "neb_band.xyz"], ids=["relax", "nebband"])
def test_real_pack_loads_through_loadscene(name):
    """The real data-pack trajectories parse in the viewer with every frame."""
    path = PACK / name
    assert path.exists(), f"missing data-pack file {path}"
    expected = _count_frames(path)
    with capture.scene_page(path.read_text(), width=320, height=240) as page:
        n = page.evaluate("() => window.__scene.nFrames")
    assert n == expected
