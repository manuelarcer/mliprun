"""Checks for alternative 5 (cinematic). Run outside the sandbox:

    site/.venv/bin/python site/alt-5-cinematic/check.py

Asserts: no console errors (desktop and phone); no horizontal scroll at 375 px
in any chapter; with prefers-reduced-motion no animation runs (3D scene, CSS,
video); every <img>/<video>/<source> in README.md resolves to a file; landing
page weight <= 15 MB. Saves screenshots to shots/ (gitignored).

No web server: a Playwright route handler serves site/ (so ../shared/viewer.js
resolves as on GitHub Pages) and 3Dmol.js from site/.cache when present.
"""
from __future__ import annotations

import re
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from capture import LAST_CHROMIUM, _launch  # noqa: E402

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
SHOTS = HERE / "shots"
DPR = 1
ORIGIN = "http://mliprun.check"
PAGE = f"{ORIGIN}/{HERE.name}/index.html"
THREEDMOL_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"
THREEDMOL_CACHE = SITE / ".cache" / "3Dmol-2.4.2-min.js"
CHAPTERS = ["relax", "committee", "neb", "md", "webapp"]
TYPES = {".html": "text/html", ".js": "application/javascript", ".css": "text/css",
         ".json": "application/json", ".xyz": "text/plain", ".mp4": "video/mp4",
         ".gif": "image/gif", ".png": "image/png", ".jpg": "image/jpeg"}

failures: list[str] = []
GL_WARNINGS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def new_page(browser, w, h, reduced=False, weight=None):
    ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=DPR,
                              reduced_motion="reduce" if reduced else "no-preference",
                              color_scheme="dark")
    page = ctx.new_page()
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error"
            else (GL_WARNINGS.append(m.text) if m.type == "warning" else None))
    page.on("pageerror", lambda e: errors.append(str(e)))

    def handle(route):
        url = route.request.url
        if url == THREEDMOL_URL and THREEDMOL_CACHE.exists():
            body = THREEDMOL_CACHE.read_bytes()
            if weight is not None:
                weight[url] = len(body)
            return route.fulfill(body=body, content_type="application/javascript")
        if url.startswith(ORIGIN):
            rel = url[len(ORIGIN) + 1:].split("?")[0].split("#")[0]
            f = (SITE / rel).resolve()
            if SITE in f.parents and f.is_file():
                body = f.read_bytes()
                if weight is not None:
                    weight[url] = len(body)
                ctype = TYPES.get(f.suffix, "application/octet-stream")
                rng = route.request.headers.get("range")
                m = re.match(r"bytes=(\d+)-(\d*)", rng or "")
                if m:  # media elements ask for byte ranges
                    a = int(m.group(1))
                    b = int(m.group(2)) if m.group(2) else len(body) - 1
                    return route.fulfill(status=206, body=body[a:b + 1], headers={
                        "Content-Type": ctype, "Accept-Ranges": "bytes",
                        "Content-Range": f"bytes {a}-{b}/{len(body)}"})
                return route.fulfill(body=body, content_type=ctype)
            return route.fulfill(status=404, body="not found")
        if url.startswith(("https://fonts.googleapis.com/", "https://fonts.gstatic.com/")):
            # Chromium's own network stack cannot reach Google Fonts here;
            # fetch with Python so the page gets its real fonts.
            body, ctype = fetch_cached(url)
            if weight is not None:
                weight[url] = len(body)
            return route.fulfill(body=body, content_type=ctype)
        return route.continue_()

    page.route("**/*", handle)
    if weight is not None:
        def on_response(resp):
            if not resp.url.startswith(ORIGIN) and resp.url != THREEDMOL_URL:
                try:
                    weight[resp.url] = len(resp.body())
                except Exception:
                    pass
        page.on("response", on_response)
    return ctx, page, errors


_FETCHED: dict[str, tuple[bytes, str]] = {}


def fetch_cached(url):
    if url not in _FETCHED:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Chrome/156.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            _FETCHED[url] = (r.read(), r.headers.get("Content-Type", "application/octet-stream"))
    return _FETCHED[url]


def wait_scene(page, key, timeout=60000):
    page.wait_for_function(
        "(k) => window.__cinematicKey === k && window.__cinematic && window.__cinematic.ready"
        " && !document.querySelector('#stage').classList.contains('is-swapping')",
        arg=key, timeout=timeout)
    page.wait_for_timeout(400)


def go(page, chapter):
    page.evaluate("""(id) => {
      document.documentElement.style.scrollBehavior = 'auto';
      const s = document.getElementById(id);
      const card = s.querySelector('.card') || s;
      const r = card.getBoundingClientRect();
      const narrow = matchMedia('(max-width: 900px)').matches;
      const y = narrow ? s.getBoundingClientRect().top + scrollY
                       : r.top + scrollY - Math.max(24, (innerHeight - r.height) / 2);
      scrollTo(0, y);
    }""", chapter)


def no_hscroll(page):
    return page.evaluate("() => document.documentElement.scrollWidth <= innerWidth + 0.5"
                         " && document.body.scrollWidth <= innerWidth + 0.5")


CONTRAST_JS = r"""() => {
  const lum = (c) => { const v = c.map((x) => { x /= 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; });
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]; };
  const parse = (s) => (s.match(/[\d.]+/g) || []).map(Number);
  const over = (c, a, under) => c.map((x, k) => x * a + under[k] * (1 - a));
  const ratio = (a, b) => { const [h, l] = [lum(a), lum(b)].sort((x, y) => y - x); return (h + 0.05) / (l + 0.05); };
  const WHITE = [255, 255, 255];  // worst case: the brightest scene pixel under a panel
  const out = {};
  const rail = document.querySelector('.chapters');
  const rb = parse(getComputedStyle(rail).backgroundColor);
  const railBg = over(rb.slice(0, 3), rb.length > 3 ? rb[3] : 1, WHITE);
  out.rail = Math.min(...[...rail.querySelectorAll('a')].map((a) => ratio(parse(getComputedStyle(a).color).slice(0, 3), railBg)));
  const cap = document.querySelector('.scene-text');
  const cb = parse(getComputedStyle(cap).backgroundColor);
  const capBg = over(cb.slice(0, 3), cb.length > 3 ? cb[3] : 1, WHITE);
  out.caption = ratio(parse(getComputedStyle(document.querySelector('.scene-caption')).color).slice(0, 3), capBg);
  // hero note over the scrim: scrim alpha at the note's right edge, interpolated from the gradient stops
  const note = document.querySelector('.hero-note').getBoundingClientRect();
  const stops = [...getComputedStyle(document.querySelector('.scrim')).backgroundImage
    .matchAll(/rgba\(14, 23, 36, ([\d.]+)\) ([\d.]+)%/g)].map((m) => [+m[2] / 100, +m[1]]);
  const xf = note.right / innerWidth;
  let a = 0;
  for (let k = 0; k + 1 < stops.length; k++) if (xf >= stops[k][0] && xf <= stops[k + 1][0])
    a = stops[k][1] + (stops[k + 1][1] - stops[k][1]) * (xf - stops[k][0]) / (stops[k + 1][0] - stops[k][0]);
  out.heroNote = ratio(parse(getComputedStyle(document.querySelector('.hero-note')).color).slice(0, 3), over([14, 23, 36], a, WHITE));
  out.scrimAlpha = a;
  return out;
}"""


def readme_media():
    if not (HERE / "README.md").is_file():
        check(False, "README.md exists")
        return
    text = (HERE / "README.md").read_text()
    refs = re.findall(r'<(?:img|video|source)\b[^>]*?\b(?:src|srcset)="([^"]+)"', text)
    refs += re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text)
    missing = [r for r in refs if not r.startswith("http") and not (HERE / r).is_file()]
    check(bool(refs), f"README.md references media ({len(refs)} refs)")
    check(not missing, f"every README media ref resolves (missing: {missing})")
    total = 0
    for r in set(refs):
        p = HERE / r
        if p.is_file():
            total += p.stat().st_size
            check(p.stat().st_size <= 5 * 1024 * 1024, f"{r} <= 5 MB ({p.stat().st_size / 1e6:.2f} MB)")
    check(total <= 10 * 1024 * 1024, f"README media total <= 10 MB ({total / 1e6:.2f} MB)")


def main() -> int:
    SHOTS.mkdir(exist_ok=True)
    readme_media()
    with sync_playwright() as pw:
        _launch(pw).close()  # shared launcher: finds a usable (cached) Chromium build
        # Hardware WebGL (ANGLE Metal on macOS): SwiftShader needs minutes per
        # scene for the replicated hero slab.
        browser = pw.chromium.launch(executable_path=LAST_CHROMIUM["executable"],
                                     args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
        try:
            # desktop
            weight: dict[str, int] = {}
            ctx, page, errors = new_page(browser, 1440, 900, weight=weight)
            t0 = time.time()
            page.goto(PAGE, wait_until="domcontentloaded")
            wait_scene(page, "hero")
            ready = time.time() - t0 - 0.4
            check(ready < 15, f"hero scene ready in < 15 s with hardware WebGL ({ready:.1f} s)")
            page.wait_for_timeout(900)
            page.screenshot(path=str(SHOTS / "desktop-hero.png"))
            c = page.evaluate(CONTRAST_JS)
            check(c["rail"] >= 4.5, f"chapter rail text contrast >= 4.5:1 over a white worst case ({c['rail']:.2f})")
            check(c["caption"] >= 4.5, f"scene caption contrast >= 4.5:1 over a white worst case ({c['caption']:.2f})")
            check(c["heroNote"] >= 4.5, f"hero note contrast >= 4.5:1 over scrim alpha {c['scrimAlpha']:.2f} on white ({c['heroNote']:.2f})")
            heights = page.evaluate("() => [...document.querySelectorAll('.card')].map(c => [c.id, c.offsetHeight])")
            tall = [h for h in heights if h[1] > 900]
            check(not tall, f"every card fits a 900 px viewport ({heights})")
            for ch in CHAPTERS:
                go(page, ch)
                wait_scene(page, ch)
                if ch == "webapp":
                    page.wait_for_timeout(1200)
                    t1 = page.evaluate("() => document.getElementById('webapp-video').currentTime")
                    check(t1 >= 20, f"web app video opens on the result (currentTime {t1:.1f} s ~1 s after activation)")
                    page.evaluate("() => { const v = document.getElementById('webapp-video'); v.currentTime = v.duration - 0.3; }")
                    page.wait_for_timeout(1200)
                    t2 = page.evaluate("() => document.getElementById('webapp-video').currentTime")
                    check(t2 < 5, f"web app video loops back to the start ({t2:.1f} s after the end)")
                    page.evaluate("() => { document.getElementById('webapp-video').currentTime = 21; }")
                    page.wait_for_timeout(300)
                page.screenshot(path=str(SHOTS / f"desktop-{ch}.png"))
            go(page, "neb")
            wait_scene(page, "neb")
            page.click("#btn-mode")
            wait_scene(page, "ts")
            page.screenshot(path=str(SHOTS / "desktop-neb-ts.png"))
            page.evaluate("() => document.getElementById('install').scrollIntoView()")
            page.wait_for_timeout(600)
            page.screenshot(path=str(SHOTS / "desktop-coda.png"))
            total = sum(weight.values())
            check(total <= 15 * 1024 * 1024, f"landing page weight <= 15 MB ({total / 1e6:.2f} MB, {len(weight)} requests)")
            check(not errors, f"no console errors at 1440x900 ({errors[:3]})")
            ctx.close()

            # phone
            ctx, page, errors = new_page(browser, 375, 812)
            page.goto(PAGE, wait_until="domcontentloaded")
            wait_scene(page, "hero")
            page.wait_for_timeout(600)
            check(no_hscroll(page), "no horizontal scroll at 375 px, hero")
            page.screenshot(path=str(SHOTS / "phone-hero.png"))
            for ch in CHAPTERS:
                go(page, ch)
                wait_scene(page, ch)
                check(no_hscroll(page), f"no horizontal scroll at 375 px, {ch}")
                page.screenshot(path=str(SHOTS / f"phone-{ch}.png"))
            page.evaluate("() => document.getElementById('install').scrollIntoView()")
            page.wait_for_timeout(500)
            check(no_hscroll(page), "no horizontal scroll at 375 px, install")
            page.screenshot(path=str(SHOTS / "phone-coda.png"))
            check(not errors, f"no console errors at 375x812 ({errors[:3]})")
            ctx.close()

            # reduced motion
            ctx, page, errors = new_page(browser, 1440, 900, reduced=True)
            page.goto(PAGE, wait_until="domcontentloaded")
            for ch in ["hero"] + CHAPTERS:
                if ch != "hero":
                    go(page, ch)
                wait_scene(page, ch)
                f0 = page.evaluate("() => window.__cinematic.frame")
                page.wait_for_timeout(1200)
                st = page.evaluate("""() => ({
                  playing: window.__cinematic.playing,
                  frame: window.__cinematic.frame,
                  anims: document.getAnimations().filter(a => a.playState === 'running').length,
                  video: (() => { const v = document.getElementById('webapp-video'); return v ? v.paused : true; })(),
                })""")
                check(not st["playing"] and st["frame"] == f0 and st["anims"] == 0 and st["video"],
                      f"reduced motion, {ch}: nothing animates ({st})")
            check(not errors, f"no console errors with reduced motion ({errors[:3]})")
            ctx.close()
        finally:
            browser.close()
        # software WebGL: the hero must fall back to the still picture quickly
        sw = pw.chromium.launch(executable_path=LAST_CHROMIUM["executable"],
                                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        try:
            ctx, page, errors = new_page(sw, 1440, 900)
            t0 = time.time()
            page.goto(PAGE, wait_until="domcontentloaded")
            page.wait_for_function("() => window.__cinematicKey === 'poster'", timeout=30000)
            check(time.time() - t0 < 10, f"software WebGL: hero still shown in < 10 s ({time.time() - t0:.1f} s)")
            page.wait_for_timeout(500)
            page.screenshot(path=str(SHOTS / "desktop-hero-softgl.png"))
            check(not errors, f"no console errors with software WebGL ({errors[:3]})")
            ctx.close()
        finally:
            sw.close()
    gl = [w for w in GL_WARNINGS if "GL" in w or "ramebuffer" in w]
    print(f"INFO console warnings: {len(GL_WARNINGS)} total, {len(gl)} WebGL ({gl[:2]})")
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    t = time.time()
    rc = main()
    print(f"{time.time() - t:.0f} s")
    sys.exit(rc)
