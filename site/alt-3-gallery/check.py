"""Playwright checks for the gallery front page (alternative 3).

Run outside the sandbox (Chromium cannot start inside it):
    site/.venv/bin/python site/alt-3-gallery/check.py [--no-shots]

Asserts:
  1. no console errors or page errors (desktop, light);
  2. no horizontal scroll at 375 px (light and dark);
  3. with prefers-reduced-motion: no scene plays, the video is paused, frames
     do not change over 2 s, and no CSS/Web animation is running;
  4. every <img>/<video>/<source> in README.md resolves to an existing file,
     each media file <= 2 MB, README media total <= 10 MB;
  5. landing page weight (all responses) <= 15 MB;
  6. performance: with all five panels on screen, the median frame interval
     stays <= 50 ms and every looping scene advances;
  9. panel c: the sigma colouring is still on screen after a height-only
     resize and after a full-page capture;
 10. panel e: the recording is idle by default, plays enlarged on request,
     and stops when the panel closes;
  7. enlarge: clicking a panel's enlarge button makes it fill most of the
     viewport, the full command is visible, Escape restores it;
  8. orientation on screen: in a screenshot of the MD panel the water O
     atoms sit above the Pt slab (guards the screen orientation fixed in
     site/shared/viewer.js at dee6bbe).
Chromium is launched on the GPU (ANGLE Metal) when possible, else SwiftShader.
Screenshots go to shots/ (gitignored).
"""
from __future__ import annotations

import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
sys.path.insert(0, str(SITE / "shared"))
from capture import _launch, _chromium_candidates, _threedmol_bytes, THREEDMOL_URL  # noqa: E402

GPU_ARGS = ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-webgl"]


def launch_gpu(pw):
    """Chromium on the machine's GPU (closer to what a visitor sees than
    SwiftShader). Falls back to capture.py's SwiftShader launch."""
    try:
        return pw.chromium.launch(args=GPU_ARGS), "gpu"
    except Exception:
        for cand in _chromium_candidates():
            try:
                return pw.chromium.launch(executable_path=cand, args=GPU_ARGS), "gpu"
            except Exception:
                continue
    return _launch(pw), "swiftshader"

SHOTS = HERE / "shots"
FAILS: list[str] = []


def ok(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def route_3dmol(page):
    lib = _threedmol_bytes()
    if lib is None:
        return
    page.route(THREEDMOL_URL, lambda r: r.fulfill(body=lib, content_type="application/javascript"))


def wait_ready(page):
    page.wait_for_function("() => window.__gallery && window.__gallery.ready", timeout=90000)
    page.wait_for_timeout(600)


def check_readme():
    md = (HERE / "README.md").read_text()
    refs = re.findall(r'<(?:img|video|source)[^>]*?(?:src|srcset)="([^"]+)"', md)
    refs += re.findall(r'!\[[^\]]*\]\(([^)\s]+)', md)
    refs = [r for r in refs if not r.startswith(("http://", "https://"))]
    ok(len(refs) > 0, f"README has local media references ({len(refs)})")
    missing = [r for r in refs if not (HERE / r).is_file()]
    ok(not missing, f"README media resolve to files (missing: {missing})")
    files = {r for r in refs if (HERE / r).is_file()}
    sizes = {r: (HERE / r).stat().st_size for r in files}
    big = {r: s for r, s in sizes.items() if s > 2 * 1024 * 1024}
    total = sum(sizes.values())
    ok(not big, f"each README media file <= 2 MB (over: {big})")
    ok(total <= 10 * 1024 * 1024, f"README media total {total / 1e6:.2f} MB <= 10 MB")
    ok("<script" not in md and "<iframe" not in md, "README has no script or iframe")


def main(argv):
    shots = "--no-shots" not in argv
    SHOTS.mkdir(exist_ok=True)
    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                           cwd=SITE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/alt-3-gallery/index.html"
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser, kind = launch_gpu(pw)
            print(f"     browser: {browser.version}, launch mode {kind}")
            try:
                run_page_checks(browser, url, shots)
            finally:
                browser.close()
        check_readme()
    finally:
        srv.terminate()
        srv.wait(timeout=5)
    print()
    print("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED")
    return 1 if FAILS else 0


DETAIL_JS = """(id) => { const t = document.getElementById(id), d = t.querySelector('.detail');
  const r = d.getBoundingClientRect(), cs = getComputedStyle(d);
  return {vis: cs.visibility, sh: d.scrollHeight, ch: d.clientHeight, top: r.top, bottom: r.bottom, vh: innerHeight, ov: cs.overflowY}; }"""


def details_ok(page, tag):
    bad = []
    for tid in ("neb", "relax", "committee", "md", "webapp"):
        page.evaluate(f"() => document.querySelector('#{tid} .tile-foot').scrollIntoView({{block: 'end'}})")
        page.wait_for_timeout(150)
        page.hover(f"#{tid} .tile-foot .cmd")
        page.wait_for_timeout(250)
        r = page.evaluate(DETAIL_JS, tid)
        fits = r["sh"] <= r["ch"] + 1
        onscreen = r["top"] >= -1 and r["bottom"] <= r["vh"] + 1
        if not (r["vis"] == "visible" and onscreen and (fits or r["ov"] in ("auto", "scroll"))):
            bad.append((tid, r))
        page.mouse.move(1, 1)
        page.wait_for_timeout(150)
    ok(not bad, f"{tag}: each panel's detail opens complete (or scrollable) and fully on screen (bad: {bad})")


def rings_follow(page, tag, width, drag=True):
    """Drag the band (rotates it), then change only the viewport height (as a
    phone URL bar does): the viewer re-fits, and the drawn rings must still sit
    on the projected hollows."""
    page.locator("#neb").scroll_into_view_if_needed()
    box = page.locator("#scene-neb").bounding_box()
    if drag:
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(cx, cy); page.mouse.down()
        page.mouse.move(cx + 60, cy + 25, steps=8); page.mouse.up()
        page.wait_for_timeout(400)
    h = page.viewport_size["height"]
    page.set_viewport_size({"width": width, "height": h - 150})
    page.wait_for_timeout(1500)
    r = page.evaluate("() => window.__gallery.siteCheck()")
    page.set_viewport_size({"width": width, "height": h})
    page.wait_for_timeout(800)
    err = max(abs(r["expected"][k][a] - r["drawn"][k][a]) for k in ("fcc", "hcp") for a in ("x", "y"))
    what = "drag + height-only resize" if drag else "height-only resize (touch: no drag-rotate)"
    ok(err <= 3, f"{tag}: after {what} the rings sit on the hollows (max error {err:.1f} px)")


def run_page_checks(browser, url, shots):
    # ---- 1, 5: console errors and weight, desktop light
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme="light")
    page = ctx.new_page()
    errors, total = [], [0]
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    failed = []
    page.on("requestfailed", lambda r: failed.append(f"{r.url} ({r.failure})"))

    def on_resp(r):
        try:
            total[0] += len(r.body())
        except Exception:
            pass
    page.on("response", on_resp)
    route_3dmol(page)
    page.goto(url)
    wait_ready(page)
    page.wait_for_timeout(1500)
    # scroll through so lazy media (video) loads before the weight is summed
    page.mouse.wheel(0, 3000)
    page.wait_for_timeout(1500)
    page.mouse.wheel(0, -3000)
    page.wait_for_timeout(500)
    # A Google Fonts request that this machine's network drops shows up as a
    # generic "Failed to load resource" error. It is reported, not counted: the
    # page declares fallback stacks, and it is not a page bug.
    font_fail = [f for f in failed if "fonts.googleapis.com" in f or "fonts.gstatic.com" in f]
    other_fail = [f for f in failed if f not in font_fail]
    if font_fail:
        print(f"WARN Google Fonts request failed on this network: {font_fail[:2]}")
        errors = [e for e in errors if not e.startswith("Failed to load resource")] if not other_fail else errors
    ok(not errors and not other_fail, f"no console errors (got {errors[:3]}; failed requests {other_fail[:3]})")
    n_scenes = page.evaluate("() => Object.values(window.__gallery.scenes).filter(Boolean).length")
    ok(n_scenes == 5, f"five 3D scenes loaded ({n_scenes})")
    ok(total[0] <= 15 * 1024 * 1024, f"landing page weight {total[0] / 1e6:.2f} MB <= 15 MB")
    if shots:
        page.screenshot(path=str(SHOTS / "desktop-light.png"))
        page.screenshot(path=str(SHOTS / "desktop-light-full.png"), full_page=True)
        # hover reveal
        page.hover("#relax .tile-foot")
        page.wait_for_timeout(300)
        page.screenshot(path=str(SHOTS / "desktop-hover.png"))

    # ---- 8: orientation on SCREEN (water bilayer must sit above the slab).
    # Element screenshot = what the compositor shows; red pixels are water O,
    # grey pixels are Pt. Water above the slab => red centroid higher (smaller y).
    import base64
    page.mouse.move(2, 2)
    page.locator("#md").scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    png = base64.b64encode(page.locator("#scene-md").screenshot()).decode()
    ys = page.evaluate("""async (b) => {
      const i = new Image(); i.src = 'data:image/png;base64,' + b; await i.decode();
      const c = document.createElement('canvas'); c.width = i.width; c.height = i.height;
      const g = c.getContext('2d'); g.drawImage(i, 0, 0); const d = g.getImageData(0, 0, c.width, c.height).data;
      let ry = 0, rn = 0, gy = 0, gn = 0;
      for (let k = 0; k < d.length; k += 4) { const y = Math.floor(k / 4 / c.width), r = d[k], gg = d[k+1], bb = d[k+2];
        if (r > 150 && gg < 90 && bb < 90) { ry += y; rn++; }
        else if (Math.abs(r - gg) < 14 && Math.abs(gg - bb) < 20 && r > 90 && r < 200) { gy += y; gn++; } }
      return [ry / Math.max(rn, 1), gy / Math.max(gn, 1), rn, gn]; }""", png)
    ok(ys[2] > 20 and ys[0] < ys[1], f"on screen the water O sits above the Pt slab (O mean y {ys[0]:.0f} px, Pt mean y {ys[1]:.0f} px)")

    # ---- 9: committee sigma colouring survives a height-only resize and a
    # full-page capture (the viewer rebuilds the model on every refit; the
    # colouring must come back through its restyle hook).
    SIG_JS = """async (b) => {
      const i = new Image(); i.src = 'data:image/png;base64,' + b; await i.decode();
      const c = document.createElement('canvas'); c.width = i.width; c.height = i.height;
      const g = c.getContext('2d'); g.drawImage(i, 0, 0); const d = g.getImageData(0, 0, c.width, c.height).data;
      let n = 0, tot = 0;
      for (let k = 0; k < d.length; k += 4) { const r = d[k], gg = d[k+1], bb = d[k+2];
        tot++; if (bb > r + 40 && bb > gg + 30) n++; }   // blue-violet ramp pixels
      return n / tot; }"""
    def sigma_share():
        page.locator("#committee").scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        return page.evaluate(SIG_JS, base64.b64encode(page.locator("#scene-committee").screenshot()).decode())
    s0 = sigma_share()
    page.set_viewport_size({"width": 1440, "height": 700})   # height only
    page.wait_for_timeout(1200)
    s1 = sigma_share()
    page.screenshot(path=str(SHOTS / "after-resize-full.png"), full_page=True)
    page.wait_for_timeout(800)
    s2 = sigma_share()
    page.set_viewport_size({"width": 1440, "height": 900})
    page.wait_for_timeout(1000)
    ok(s0 > 0.05 and s1 > 0.5 * s0 and s2 > 0.5 * s0,
       f"sigma colours stay on screen after a height-only resize and a full-page capture "
       f"(share of ramp pixels {s0:.3f} -> {s1:.3f} -> {s2:.3f})")

    # ---- 10: web app recording plays on request (stills by default)
    page.locator("#webapp").scroll_into_view_if_needed()
    paused0 = page.evaluate("() => document.getElementById('webapp-video').paused")
    page.click("#vplay")
    page.wait_for_timeout(1500)
    vid = page.evaluate("() => { const v = document.getElementById('webapp-video'); return [!v.paused, v.currentTime, document.getElementById('webapp').classList.contains('expanded')]; }")
    if shots:
        page.screenshot(path=str(SHOTS / "desktop-webapp-video.png"))
    ok(paused0 and vid[0] and vid[1] > 0 and vid[2], f"recording idle by default, plays enlarged on request ({vid})")
    page.keyboard.press("Escape")
    page.wait_for_timeout(400)
    ok(page.evaluate("() => document.getElementById('webapp-video').paused"), "closing the panel stops the recording")

    # ---- 11: fcc / hcp labels: each label nearer its own hollow than the other
    def sites_ok(tag):
        st = page.evaluate("() => window.__gallery.sites")
        def d(a, b): return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
        res = []
        for k, other in (("fcc", "hcp"), ("hcp", "fcc")):
            lab = st[k]["label"]
            res.append(d(lab, st[k]["site"]) < d(lab, st[other]["site"]))
        ok(all(res), f"{tag}: fcc and hcp labels each sit nearer their own hollow ({st})")
    sites_ok("1440 px")
    rings_follow(page, "1440 px", 1440)
    if shots:
        page.locator("#neb").scroll_into_view_if_needed()
        for fr in (0, 8):
            page.evaluate(f"() => window.__gallery.scenes['scene-neb'].setFrame({fr})")
            page.wait_for_timeout(300)
            page.locator("#scene-neb").screenshot(path=str(SHOTS / f"neb-frame{fr}-1440.png"))
        page.evaluate("() => window.__gallery.scenes['scene-neb'].play({force: true})")

    # ---- 12: panel a hover: command and its detail visible together
    page.locator("#neb .tile-foot").scroll_into_view_if_needed()
    page.hover("#neb .tile-foot .cmd")
    page.wait_for_timeout(300)
    r = page.evaluate("() => { const d = document.querySelector('#neb .detail').getBoundingClientRect(); const c = document.querySelector('#neb .cmd').getBoundingClientRect(); return [d.top, d.bottom, c.top, c.bottom, innerHeight, getComputedStyle(document.querySelector('#neb .detail')).visibility]; }")
    ok(r[5] == "visible" and r[0] >= 0 and r[3] <= r[4], f"panel a: hovering the command shows its detail on screen with it ({r})")
    if shots:
        page.screenshot(path=str(SHOTS / "desktop-hover-neb.png"))
    page.mouse.move(2, 2)

    details_ok(page, "1440 px")

    # ---- 7: enlarge
    page.mouse.move(5, 5)
    page.click("#neb .grow")
    page.wait_for_timeout(900)
    box = page.evaluate("() => { const r = document.getElementById('neb').getBoundingClientRect(); return [r.width, r.height]; }")
    full_vis = page.evaluate("() => getComputedStyle(document.querySelector('#neb .detail')).visibility")
    ok(box[0] > 1440 * 0.8 and box[1] > 900 * 0.8, f"enlarged panel fills the viewport ({box})")
    ok(full_vis == "visible", "full command visible when enlarged")
    if shots:
        page.screenshot(path=str(SHOTS / "desktop-enlarged.png"))
    page.keyboard.press("Escape")
    page.wait_for_timeout(400)
    closed = page.evaluate("() => !document.getElementById('neb').classList.contains('expanded')")
    ok(closed, "Escape restores the panel")
    ctx.close()

    # ---- 6: performance with all panels on screen
    ctx = browser.new_context(viewport={"width": 1440, "height": 1700}, color_scheme="light")
    page = ctx.new_page()
    route_3dmol(page)
    page.goto(url)
    wait_ready(page)
    page.wait_for_timeout(1000)
    renderer = page.evaluate("""() => { const c = document.createElement('canvas').getContext('webgl');
      const d = c && c.getExtension('WEBGL_debug_renderer_info');
      return d ? c.getParameter(d.UNMASKED_RENDERER_WEBGL) : (c ? c.getParameter(c.RENDERER) : 'none'); }""")
    print(f"     WebGL renderer: {renderer}")
    perf = page.evaluate("""async () => {
      const g = window.__gallery.scenes;
      const ids = Object.keys(g).filter(k => g[k] && g[k].loops);
      const seen = Object.fromEntries(ids.map(k => [k, new Set()]));
      const dts = []; let last = performance.now(); const t0 = last;
      await new Promise(res => { const f = (t) => { dts.push(t - last); last = t;
        ids.forEach(k => seen[k].add(g[k].frame));
        if (t - t0 < 4000) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
      dts.sort((a, b) => a - b);
      return { median: dts[Math.floor(dts.length / 2)], p95: dts[Math.floor(dts.length * 0.95)],
               n: dts.length, frames: Object.fromEntries(ids.map(k => [k, seen[k].size])) };
    }""")
    print(f"     perf: median frame {perf['median']:.1f} ms, p95 {perf['p95']:.1f} ms over {perf['n']} frames "
          f"distinct frames per scene {perf['frames']}")
    ok(perf["median"] <= 50, f"median frame interval {perf['median']:.1f} ms <= 50 ms with all panels visible")
    ok(all(v > 1 for v in perf["frames"].values()), "every looping scene advances")
    ctx.close()

    # ---- 2: mobile, light and dark
    for scheme in ("light", "dark"):
        ctx = browser.new_context(viewport={"width": 375, "height": 812}, color_scheme=scheme,
                                  device_scale_factor=2, is_mobile=True, has_touch=True)
        page = ctx.new_page()
        route_3dmol(page)
        page.goto(url)
        wait_ready(page)
        sw = page.evaluate("() => [document.documentElement.scrollWidth, window.innerWidth]")
        ok(sw[0] <= sw[1], f"no horizontal scroll at 375 px, {scheme} (scrollWidth {sw[0]}, viewport {sw[1]})")
        if scheme == "light":
            rings_follow(page, "375 px", 375, drag=False)
            details_ok(page, "375 px")
            sites_ok_m = page.evaluate("() => window.__gallery.sites")
            def dd(a, b): return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
            good = all(dd(sites_ok_m[k]["label"], sites_ok_m[k]["site"]) < dd(sites_ok_m[k]["label"], sites_ok_m[o]["site"])
                       for k, o in (("fcc", "hcp"), ("hcp", "fcc")))
            ok(good, f"375 px: fcc and hcp labels each sit nearer their own hollow ({sites_ok_m})")
            if shots:
                page.locator("#neb").scroll_into_view_if_needed()
                for fr in (0, 8):
                    page.evaluate(f"() => window.__gallery.scenes['scene-neb'].setFrame({fr})")
                    page.wait_for_timeout(300)
                    page.locator("#scene-neb").screenshot(path=str(SHOTS / f"neb-frame{fr}-375.png"))
        if shots:
            page.screenshot(path=str(SHOTS / f"mobile-{scheme}.png"))
            page.screenshot(path=str(SHOTS / f"mobile-{scheme}-full.png"), full_page=True)
        ctx.close()
    if shots:
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme="dark")
        page = ctx.new_page()
        route_3dmol(page)
        page.goto(url)
        wait_ready(page)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(SHOTS / "desktop-dark.png"))
        page.screenshot(path=str(SHOTS / "desktop-dark-full.png"), full_page=True)
        ctx.close()

    # ---- 3: reduced motion
    ctx = browser.new_context(viewport={"width": 1440, "height": 1700}, reduced_motion="reduce")
    page = ctx.new_page()
    route_3dmol(page)
    page.goto(url)
    wait_ready(page)
    a = page.evaluate("() => Object.fromEntries(Object.entries(window.__gallery.scenes).filter(e => e[1]).map(([k, c]) => [k, [c.frame, c.playing]]))")
    page.wait_for_timeout(2000)
    b = page.evaluate("() => Object.fromEntries(Object.entries(window.__gallery.scenes).filter(e => e[1]).map(([k, c]) => [k, [c.frame, c.playing]]))")
    vid = page.evaluate("() => { const v = document.getElementById('webapp-video'); return [v.paused, v.currentTime]; }")
    anims = page.evaluate("() => document.getAnimations().filter(a => a.playState === 'running').length")
    ok(all(not v[1] for v in b.values()), f"reduced motion: no scene playing ({b})")
    ok(a == b, "reduced motion: frames unchanged over 2 s")
    ok(vid[0] and vid[1] == 0, f"reduced motion: recording paused at 0 s ({vid})")
    ok(anims == 0, f"reduced motion: no running CSS/Web animations ({anims})")
    ctx.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
