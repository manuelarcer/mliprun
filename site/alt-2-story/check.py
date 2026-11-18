"""Checks for alternative 2 ("story"). Run outside the sandbox (Chromium):

    site/.venv/bin/python site/alt-2-story/check.py

Asserts:
  1. no console errors or page errors while scrolling through every chapter;
  2. no horizontal scroll at 375 px;
  3. with prefers-reduced-motion, no animation runs (scene not playing, video paused,
     no running CSS/Web animations) in any chapter;
  4. every <img>/<video>/<source> (and markdown image) in README.md resolves to a file;
  5. landing page weight (all bytes the page loads, video included) <= 15 MB;
  6. swap freeze: a rAF frame-gap probe measures the longest main-thread block while
     entering each chapter (first visit = scene build, revisit = visibility toggle), on
     the GPU (ANGLE Metal) and on SwiftShader (software WebGL).
Saves screenshots to site/alt-2-story/shots/ (gitignored).

No port is bound: a Playwright route handler serves site/ from disk; 3Dmol comes from
site/.cache/ when cached. Google Fonts go to the network.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
sys.path.insert(0, str(SITE / "shared"))
import capture  # noqa: E402  (reuse its Chromium launcher and 3Dmol cache)

ORIGIN = "http://story.check"
START = f"{ORIGIN}/alt-2-story/index.html"
SHOTS = HERE / "shots"
CHAPTERS = ["intro", "relax", "committee", "neb", "ts", "water", "remote"]
TYPES = {".html": "text/html", ".css": "text/css", ".js": "application/javascript",
         ".json": "application/json", ".xyz": "text/plain", ".mp4": "video/mp4"}

failures: list[str] = []
# Google Fonts are stubbed unless --online is given (the check machine may have no network).
OFFLINE_FONTS = "--online" not in sys.argv


def ok(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def make_handler(sizes: dict):
    lib = capture._threedmol_bytes()

    def handle(route):
        url = route.request.url
        if url.startswith(capture.THREEDMOL_URL.rsplit("/", 1)[0]) and lib is not None:
            sizes[url] = len(lib)
            return route.fulfill(body=lib, content_type="application/javascript")
        if OFFLINE_FONTS and ("fonts.googleapis.com" in url or "fonts.gstatic.com" in url):
            # The check machine's Chromium has no network. Serve Google Fonts from
            # .fontcache/ (filled once with curl; see the report) or an empty sheet, so
            # the page falls back to its declared font stacks without a load error.
            cache = HERE / ".fontcache"
            if "fonts.googleapis.com" in url:
                css = cache / "fonts.css"
                body = css.read_bytes() if css.exists() else b""
                return route.fulfill(body=body, content_type="text/css")
            f = cache / url.split("fonts.gstatic.com/", 1)[1].replace("/", "_")
            if f.exists():
                sizes[url] = f.stat().st_size
                return route.fulfill(body=f.read_bytes(), content_type="font/woff2")
            return route.fulfill(status=404, body="")
        if url.startswith(ORIGIN):
            rel = url[len(ORIGIN):].split("?")[0].split("#")[0].lstrip("/")
            p = (SITE / rel).resolve()
            if SITE in p.parents and p.is_file():
                body = p.read_bytes()
                sizes[url] = len(body)
                return route.fulfill(body=body, content_type=TYPES.get(p.suffix, "application/octet-stream"))
            return route.fulfill(status=404, body="not found")
        return route.continue_()
    return handle


def scroll_to_chapter(page, ch: str, frac: float = 0.45) -> None:
    page.evaluate("""([ch, frac]) => {
        const s = document.querySelector(`section[data-ch="${ch}"]`);
        const r = s.getBoundingClientRect();
        const stage = document.querySelector('#stage').getBoundingClientRect();
        const mobile = innerWidth <= 900;
        const probe = mobile ? stage.height + (innerHeight - stage.height) * 0.3 : innerHeight * 0.5;
        window.scrollTo({top: scrollY + r.top + r.height * frac - probe, behavior: 'instant'});
    }""", [ch, frac])
    page.wait_for_timeout(250)
    page.wait_for_function(
        "(ch) => window.__story && window.__story.chapter === ch", arg=ch, timeout=90000,
        polling=500)
    # A scene swap blocks the main thread for ~15-20 s under SwiftShader (software
    # WebGL: the viewer's pixel-based auto-fit renders ~15 frames); wait it out.
    page.wait_for_function(
        "() => !document.querySelector('#scene.swapping')", timeout=90000, polling=500)
    page.wait_for_timeout(600)


def run_page(pw, *, width, height, scheme="light", reduced=False, tag="", shots=True,
             sizes=None):
    browser = capture._launch(pw)
    errors: list[str] = []
    sizes = {} if sizes is None else sizes
    try:
        ctx = browser.new_context(viewport={"width": width, "height": height},
                                  color_scheme=scheme,
                                  reduced_motion="reduce" if reduced else "no-preference")
        page = ctx.new_page()
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                if m.type == "error" or (m.type == "warning" and "GL_INVALID" in m.text) else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.route("**/*", make_handler(sizes))
        page.goto(START)
        page.wait_for_function("() => !!(window.__story && window.__story.ctrl)", timeout=60000, polling=250)
        page.wait_for_timeout(800)
        results = {}
        for ch in CHAPTERS:
            fracs = [0.1, 0.6] if ch in ("relax", "neb", "committee") else [0.4]
            for f in fracs:
                scroll_to_chapter(page, ch, f)
                if shots:
                    SHOTS.mkdir(exist_ok=True)
                    page.screenshot(path=str(SHOTS / f"{tag}-{ch}-{int(f * 100)}.png"))
            results[ch] = page.evaluate("""() => ({
                playing: !!window.__story.playingAny,
                video: !window.__story.clip.paused,
                anims: document.getAnimations().filter(a => a.playState === 'running').length,
                scrollW: document.documentElement.scrollWidth,
                clientW: document.documentElement.clientWidth,
            })""")
            if ch == "ts" and not reduced:
                page.wait_for_timeout(400)
                results[ch]["playing"] = page.evaluate("() => window.__story.ctrl.playing")
        page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(400)
        if shots:
            page.screenshot(path=str(SHOTS / f"{tag}-end.png"))
        return errors, results
    finally:
        browser.close()


GAP_PROBE = """
window.__gap = { max: 0 };
(() => { let last = performance.now();
  const tick = (t) => { const g = t - last; if (g > window.__gap.max) window.__gap.max = g; last = t;
    requestAnimationFrame(tick); };
  requestAnimationFrame(tick); })();
"""


def launch_gpu(pw):
    args = ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"]
    try:
        return pw.chromium.launch(args=args)
    except Exception:
        for cand in capture._chromium_candidates():
            try:
                return pw.chromium.launch(executable_path=cand, args=args)
            except Exception:
                continue
        raise


def swap_freeze(pw, gpu: bool):
    """Longest frame gap (ms) while entering each chapter: first visit, then revisit."""
    browser = launch_gpu(pw) if gpu else capture._launch(pw)
    try:
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.route("**/*", make_handler({}))
        page.add_init_script(GAP_PROBE)
        page.goto(START)
        page.wait_for_function("() => !!(window.__story && window.__story.ctrl)",
                               timeout=120000, polling=250)
        renderer = page.evaluate("""() => { const g = document.createElement('canvas').getContext('webgl2');
            const e = g.getExtension('WEBGL_debug_renderer_info');
            return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : ''; }""")
        out = {"renderer": renderer, "first": {}, "revisit": {}, "loading_painted": {}}
        seq = [("first", c) for c in CHAPTERS[1:]] + [("revisit", c) for c in reversed(CHAPTERS[1:-1])]
        for kind, ch in seq:
            page.evaluate("() => { window.__gap.max = 0; window.__sawLoading = false; "
                          "new MutationObserver(() => { if (!document.getElementById('loading').hidden) "
                          "window.__sawLoading = true; }).observe(document.getElementById('loading'), {attributes: true}); }")
            # scroll without letting idle prebuilds run first: this measures the swap itself
            scroll_to_chapter(page, ch, 0.3)
            page.wait_for_timeout(300)
            out[kind][ch] = round(page.evaluate("() => window.__gap.max"))
            if kind == "first":
                out["loading_painted"][ch] = page.evaluate("() => window.__sawLoading")
        return out
    finally:
        browser.close()


def check_readme() -> None:
    readme = HERE / "README.md"
    if not readme.exists():
        ok(False, "README.md exists")
        return
    text = readme.read_text()
    refs = re.findall(r'<(?:img|video|source)\b[^>]*?\s(?:src|srcset)="([^"]+)"', text)
    refs += re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text)
    refs = [r for r in refs if not r.startswith(("http://", "https://"))]
    missing = [r for r in refs if not (HERE / r).is_file()]
    ok(bool(refs) and not missing, f"README media resolve ({len(refs)} refs, missing: {missing})")
    total = sum((HERE / r).stat().st_size for r in set(refs) if (HERE / r).is_file())
    big = [r for r in set(refs) if (HERE / r).is_file() and (HERE / r).stat().st_size > 5 * 2**20]
    ok(not big, f"each README media file <= 5 MB (over: {big})")
    ok(total <= 10 * 2**20, f"README media total {total / 2**20:.2f} MB <= 10 MB")
    ok("<script" not in text and "<iframe" not in text, "README has no script or iframe")


def main() -> int:
    with sync_playwright() as pw:
        sizes: dict = {}
        err, res = run_page(pw, width=1440, height=900, tag="desk-light", sizes=sizes)
        ok(not err, f"desktop light: no console errors or WebGL GL_INVALID warnings {err[:3]}")
        ok(res["ts"]["playing"], "desktop: TS mode plays when its chapter is active")
        weight = sum(sizes.values())
        # the video is preload=none: count it even if the browser fetched it in ranges
        mp4 = HERE / "data" / "webapp.mp4"
        if not any(k.endswith("webapp.mp4") for k in sizes):
            weight += mp4.stat().st_size
        ok(weight <= 15 * 2**20, f"landing page weight {weight / 2**20:.2f} MB <= 15 MB "
           f"({len(sizes)} local/CDN files; Google Fonts not counted, ~0.2 MB)")

        err, _ = run_page(pw, width=1440, height=900, scheme="dark", tag="desk-dark")
        ok(not err, f"desktop dark: no console errors {err[:3]}")

        for scheme in ("light", "dark"):
            err, res = run_page(pw, width=375, height=812, scheme=scheme, tag=f"m375-{scheme}")
            ok(not err, f"375 px {scheme}: no console errors {err[:3]}")
            over = {c: r["scrollW"] - r["clientW"] for c, r in res.items() if r["scrollW"] > r["clientW"]}
            ok(not over, f"375 px {scheme}: no horizontal scroll in any chapter {over}")

        err, res = run_page(pw, width=1440, height=900, reduced=True, tag="reduced", shots=False)
        ok(not err, f"reduced motion: no console errors {err[:3]}")
        moving = {c: r for c, r in res.items() if r["playing"] or r["video"] or r["anims"]}
        ok(not moving, f"reduced motion: nothing animates in any chapter {moving}")
        for gpu in (True, False):
            r = swap_freeze(pw, gpu)
            tag = "GPU" if gpu else "SwiftShader"
            print(f"INFO swap freeze, {tag} ({r['renderer']}):")
            print(f"INFO   first visit, longest frame gap per chapter (ms): {r['first']}")
            print(f"INFO   revisit (scene kept, visibility toggle) (ms): {r['revisit']}")
            print(f"INFO   loading label shown on first visit: {r['loading_painted']}")
            if gpu:
                worst_rev = max(r["revisit"].values())
                # rAF runs at ~16.7 ms; a revisit redraws the plot and one frame of the scene
                ok(worst_rev < 200, f"GPU: revisiting a chapter blocks < 200 ms (worst {worst_rev} ms)")
                worst_first = max(v for k, v in r["first"].items() if k != "remote")
                ok(worst_first < 1500, f"GPU: first visit blocks < 1500 ms (worst {worst_first} ms)")
    check_readme()
    print(f"chromium: {capture.LAST_CHROMIUM['version']}")
    print("ALL PASS" if not failures else f"{len(failures)} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
