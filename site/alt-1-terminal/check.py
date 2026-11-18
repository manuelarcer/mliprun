"""Checks for alternative 1 (terminal). Run outside the Claude Code sandbox (Chromium):

    site/.venv/bin/python site/alt-1-terminal/check.py

Asserts:
  1. no console errors or page errors (light, dark, 375 px, reduced motion);
  2. no horizontal scroll at 375 px;
  3. with prefers-reduced-motion: no CSS/Web animation, the transcript is at its final
     state at once, the video is paused, and the 3D scenes do not change over 1.5 s;
  4. every <img>/<video>/<source> in README.md resolves to an existing file, each <= 5 MB,
     total README media <= 10 MB;
  5. landing page weight (all bytes the page loads, CDN included) <= 15 MB.
Saves screenshots to shots/ (1440x900 and 375x812, light and dark, plus reduced motion).

No port is bound: a Playwright route handler serves site/ under http://mliprun.local/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from capture import _launch  # noqa: E402  (falls back to a cached Chromium build)

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
CACHE = SITE / ".cache" / "3Dmol-2.4.2-min.js"
ORIGIN = "http://mliprun.local"
URL = f"{ORIGIN}/{HERE.name}/index.html"
SHOTS = HERE / "shots"
TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript",
         ".json": "application/json", ".xyz": "text/plain", ".txt": "text/plain",
         ".log": "text/plain", ".md": "text/plain", ".mp4": "video/mp4",
         ".svg": "image/svg+xml", ".gif": "image/gif", ".png": "image/png"}

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


SERVED: dict[str, int] = {}  # bytes the route handler served, per URL


def route(r):
    url = r.request.url
    if url.startswith(ORIGIN):
        p = SITE / url[len(ORIGIN) + 1:].split("?")[0]
        if p.is_file():
            SERVED[url] = p.stat().st_size
            return r.fulfill(status=200, body=p.read_bytes(),
                             headers={"content-type": TYPES.get(p.suffix, "application/octet-stream")})
        return r.fulfill(status=404, body=b"not found")
    if "3dmol@2.4.2" in url and CACHE.is_file():
        SERVED[url] = CACHE.stat().st_size
        return r.fulfill(status=200, body=CACHE.read_bytes(),
                         headers={"content-type": "text/javascript"})
    return r.continue_()


def open_page(browser, width, height, scheme="light", reduced="no-preference"):
    ctx = browser.new_context(viewport={"width": width, "height": height},
                              color_scheme=scheme, reduced_motion=reduced)
    ctx.route("**/*", route)
    page = ctx.new_page()
    errs: list[str] = []
    def on_console(m):
        if m.type != "error":
            return
        # A third-party host the network could not reach (Google Fonts) is reported, not
        # counted: it is this machine's network, not the page. Local 404s still fail.
        if m.text.startswith("Failed to load resource: net::") and page.ext_fail:
            print("  WARN external resource unreachable:", page.ext_fail[-1][:80])
            return
        errs.append(m.text)
    page.on("console", on_console)
    page.on("pageerror", lambda e: errs.append(str(e)))
    ext_fail: list[str] = []

    def on_fail(q):
        print("  request failed:", q.url[:100], q.failure)
        if not q.url.startswith(ORIGIN):
            ext_fail.append(q.url)
    page.on("requestfailed", on_fail)
    page.ext_fail = ext_fail
    sizes: dict[str, int] = {}

    def on_resp(resp):
        # Never call resp.body() here: on an aborted media request it can block forever.
        # Local and cached files are counted from SERVED; external ones from content-length.
        try:
            n = SERVED.get(resp.url) or int(resp.headers.get("content-length", 0))
            sizes[resp.url] = n
        except Exception:
            pass
    page.on("response", on_resp)
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)  # a slow font host must not fail the page check
    try:
        page.wait_for_selector("html[data-ready='1']", state="attached", timeout=180000)
        page.wait_for_selector("html[data-scenes='1']", state="attached", timeout=240000)  # SwiftShader auto-fit is slow
        page.wait_for_selector("html[data-fonts='1']", state="attached", timeout=120000)
    except Exception:
        print("page did not get ready; console errors:", errs)
        raise
    return ctx, page, errs, sizes


CLIP_JS = """() => [...document.querySelectorAll('.cmdlines, pre.out, .term-body, .full pre')]
    .filter(e => e.offsetParent !== null && e.scrollWidth > e.clientWidth + 1)
    .map(e => (e.id || e.className) + ' ' + e.scrollWidth + '>' + e.clientWidth)"""


def no_clipping(page, where: str) -> None:
    """After fonts.ready: no command box or output block is wider than its box."""
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(400)  # the debounced re-wrap
    bad = page.evaluate(CLIP_JS)
    check(not bad, f"no command or output box clips or scrolls sideways, {where} {bad}")
    fam = page.evaluate("getComputedStyle(document.querySelector('.cmdlines')).fontFamily")
    loaded = page.evaluate("document.fonts.check('12px \\'Martian Mono\\'')")
    print(f"  (mono font: {fam.split(',')[0]}, Martian Mono loaded: {loaded})")


def shot(page, path=None, **kw):
    """Screenshot with the 3D scenes paused: under SwiftShader, three playing scenes keep the
    main thread too busy for Playwright's capture (it timed out). Frames on screen are real."""
    page.evaluate("Object.values(window.__mliprunScenes || {}).forEach(c => c.pause())")
    return page.screenshot(path=path, timeout=120000, **kw)


def readme_media() -> None:
    text = (HERE / "README.md").read_text()
    refs = re.findall(r'<(?:img|video|source)[^>]*?(?:src|srcset)="([^"]+)"', text)
    refs += re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text)
    local = [r for r in refs if not r.startswith("http")]
    check(len(local) > 0, f"README references {len(local)} local media files")
    total = 0
    for r in local:
        p = (HERE / r).resolve()
        ok = p.is_file()
        check(ok, f"README media exists: {r}")
        if ok:
            total += p.stat().st_size
            check(p.stat().st_size <= 5 * 1024 * 1024, f"  {r} <= 5 MB ({p.stat().st_size / 1e6:.2f} MB)")
    for linked in re.findall(r'<a href="(media/[^"]+)"', text):
        check((HERE / linked).is_file(), f"README link target exists: {linked}")
    media_total = sum(f.stat().st_size for f in (HERE / "media").iterdir())
    check(media_total <= 10 * 1024 * 1024, f"README media folder total <= 10 MB ({media_total / 1e6:.2f} MB)")


def main() -> int:
    SHOTS.mkdir(exist_ok=True)
    readme_media()
    with sync_playwright() as pw:
        browser = _launch(pw)

        # desktop light: let the replay finish, then screenshot
        ctx, page, errs, sizes = open_page(browser, 1440, 900, "light")
        page.wait_for_selector("#term-state.done", timeout=180000)
        page.wait_for_timeout(2200)
        shot(page, path=SHOTS / "desktop-light.png")
        shot(page, path=SHOTS / "desktop-light-full.png", full_page=True)
        no_clipping(page, "1440 px light")
        page.evaluate("document.querySelector('video').pause()")
        weight = sum(sizes.values())
        check(weight <= 15 * 1024 * 1024, f"landing page weight <= 15 MB ({weight / 1e6:.2f} MB, {len(sizes)} responses)")
        check(not errs, f"no console errors, desktop light {errs[:3]}")
        ctx.close()

        ctx, page, errs, _ = open_page(browser, 1440, 900, "dark")
        page.wait_for_selector("#term-state.done", timeout=180000)
        page.wait_for_timeout(2200)
        shot(page, path=SHOTS / "desktop-dark.png")
        shot(page, path=SHOTS / "desktop-dark-full.png", full_page=True)
        check(not errs, f"no console errors, desktop dark {errs[:3]}")
        ctx.close()

        # mid-replay frame (shows the live barrier line)
        ctx, page, errs, _ = open_page(browser, 1440, 900, "light")
        page.wait_for_function("document.querySelectorAll('#term-body .l.step').length > 20", timeout=30000)
        shot(page, path=SHOTS / "desktop-light-midreplay.png")
        ctx.close()

        for scheme in ("light", "dark"):
            ctx, page, errs, _ = open_page(browser, 375, 812, scheme)
            page.wait_for_selector("#term-state.done", timeout=180000)
            page.wait_for_timeout(1500)
            sw = page.evaluate("document.documentElement.scrollWidth")
            cw = page.evaluate("document.documentElement.clientWidth")
            check(sw <= cw, f"no horizontal scroll at 375 px, {scheme} (scrollWidth {sw}, clientWidth {cw})")
            wide = page.evaluate("""() => [...document.querySelectorAll('body *')]
                .filter(e => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1
                         && !e.closest('pre, .term-body'))
                .slice(0, 5).map(e => e.tagName + '.' + e.className)""")
            check(not wide, f"no element overflows the 375 px viewport, {scheme} {wide}")
            no_clipping(page, f"375 px {scheme}")
            shot(page, path=SHOTS / f"mobile-{scheme}.png")
            h = page.evaluate("document.documentElement.scrollHeight")
            for k, y in enumerate(range(0, h, 1600)):
                page.evaluate(f"window.scrollTo(0, {y})")
                page.wait_for_timeout(250)
                shot(page, path=SHOTS / f"mobile-{scheme}-part{k}.png",
                                clip={"x": 0, "y": y, "width": 375, "height": min(1600, h - y)},
                                full_page=True)
            page.evaluate("window.scrollTo(0, 0)")
            check(not errs, f"no console errors, mobile {scheme} {errs[:3]}")
            ctx.close()

        # reduced motion
        ctx, page, errs, _ = open_page(browser, 1440, 900, "light", reduced="reduce")
        page.wait_for_timeout(300)
        state = page.text_content("#term-state")
        check("done" in state, f"reduced motion: transcript at final state at once ({state!r})")
        a1 = page.evaluate("document.getAnimations().filter(a => a.playState === 'running').length")
        check(a1 == 0, f"reduced motion: no running CSS/Web animations ({a1})")
        paused = page.evaluate("document.querySelector('video').paused")
        check(paused, "reduced motion: video does not autoplay")
        js = "() => Object.entries(window.__mliprunScenes).map(([k, c]) => [k, c.frame, c.playing])"
        f1 = page.evaluate(js)
        page.mouse.wheel(0, 3000)  # bring the relax scene on screen too
        page.wait_for_timeout(1500)
        f2 = page.evaluate(js)
        check(len(f1) == 3 and f1 == f2 and not any(p for _, _, p in f2),
              f"reduced motion: 3D scenes rest on one frame over 1.5 s ({f1} -> {f2})")
        page.mouse.wheel(0, -3000)
        n_lines = page.evaluate("document.querySelectorAll('#term-body .l').length")
        check(n_lines > 50, f"reduced motion: full transcript shown ({n_lines} lines)")
        shot(page, path=SHOTS / "desktop-reduced.png")
        check(not errs, f"no console errors, reduced motion {errs[:3]}")
        ctx.close()
        browser.close()
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
