"""Checks for alternative 4 (handbook). Run outside the sandbox:

    site/.venv/bin/python site/alt-4-handbook/check.py

Asserts: no console errors; no horizontal scroll at 375 px; with reduced motion no
animation runs; every <img>/<video>/<source> (and markdown image) in README.md
resolves to an existing file; landing page weight <= 15 MB. Saves screenshots at
1440x900 and 375x812, light and dark, to shots/.
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
import capture  # noqa: E402  (only for its Chromium launcher and 3Dmol cache)

from playwright.sync_api import sync_playwright  # noqa: E402

SHOTS = HERE / "shots"
LIMIT_MB = 15.0
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


VAR = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")
DEF = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=", re.M)


def undefined_vars(blocks: list[str]) -> set[str]:
    """Every $NAME used in a copyable block must be exported in some block."""
    defined = {n for b in blocks for n in DEF.findall(b)}
    used = {n for b in blocks for n in VAR.findall(b)}
    return used - defined


def readme_vars() -> None:
    text = (HERE / "README.md").read_text()
    blocks = re.findall(r"```bash\n(.*?)```", text, re.S)
    # "Exactly as run" blocks reproduce the original line and are not recipes
    recipes = [b for b in blocks if not b.lstrip().startswith("/")]
    bad = undefined_vars(recipes)
    check(not bad, f"README recipes: no undefined $VAR ({len(recipes)} bash blocks) {sorted(bad) or ''}")


def root_readme() -> None:
    root = HERE.parents[1]
    p = HERE / "README.root.md"
    if not p.exists():
        check(False, "README.root.md exists (build_readme.py --root)")
        return
    text = p.read_text()
    refs = re.findall(r'<(?:img|source)[^>]*?\s(?:src|srcset)="([^"]+)"', text)
    refs += re.findall(r"\]\(([^)#\s]+)", text)
    local = [r for r in refs if not re.match(r"https?://", r)]
    missing = [r for r in local if not (root / r).exists()]
    # docs/INSTALL.md is created in Phase 5 (ruling AB): allowed to be missing now
    missing = [r for r in missing if r != "docs/INSTALL.md"]
    check(not missing, f"README.root.md: {len(local)} root-relative refs resolve (docs/INSTALL.md is Phase 5) {missing or ''}")


def readme_media() -> None:
    text = (HERE / "README.md").read_text()
    refs = re.findall(r'<(?:img|video|source)[^>]*?\s(?:src|srcset)="([^"]+)"', text)
    refs += re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text)
    local = [r for r in refs if not re.match(r"https?://", r)]
    missing = [r for r in local if not (HERE / r).is_file()]
    check(bool(local) and not missing, f"README media: {len(local)} local refs, missing {missing}")
    sizes = {r: (HERE / r).stat().st_size / 1e6 for r in set(local) if (HERE / r).is_file()}
    big = {r: round(s, 2) for r, s in sizes.items() if s > 5}
    check(not big, f"README media each <= 5 MB (largest {max(sizes.values(), default=0):.2f} MB) {big or ''}")
    total = sum(sizes.values())
    check(total <= 10, f"README media total {total:.2f} MB <= 10 MB")
    check("<script" not in text and "<iframe" not in text, "README has no script or iframe")


FONT_CACHE: dict[str, tuple[bytes, str]] = {}
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")


def prefetch_fonts() -> None:
    """Download the page's Google Fonts stylesheet and font files once, in Python."""
    import urllib.request
    html = (HERE / "index.html").read_text()
    m = re.search(r'href="(https://fonts.googleapis.com/css2[^"]+)"', html)
    if not m:
        return
    url = m.group(1).replace("&amp;", "&")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        css = urllib.request.urlopen(req, timeout=20).read()
        FONT_CACHE[url] = (css, "text/css")
        for f in set(re.findall(rb"url\((https://fonts.gstatic.com/[^)]+)\)", css)):
            u = f.decode()
            FONT_CACHE[u] = (urllib.request.urlopen(u, timeout=20).read(), "font/woff2")
        print(f"fonts prefetched: {len(FONT_CACHE) - 1} files")
    except Exception as err:
        print(f"fonts not prefetched ({err}); the page falls back to system fonts")


def font_route(route):
    hit = FONT_CACHE.get(route.request.url)
    if hit:
        route.fulfill(body=hit[0], content_type=hit[1], headers={"access-control-allow-origin": "*"})
    else:
        route.fulfill(body="", content_type="text/css")  # not prefetched: system fonts


def scroll_shots(page, prefix: str, limit: int = 14) -> None:
    """Viewport screenshots down the page (a full-page capture of a WebGL page fails)."""
    h = page.evaluate("innerHeight")
    total = page.evaluate("document.documentElement.scrollHeight")
    for n, y in enumerate(range(0, total, h)):
        if n >= limit:
            break
        page.evaluate(f"scrollTo(0, {y})")
        page.wait_for_timeout(700)
        page.screenshot(path=str(SHOTS / f"{prefix}-{n:02d}.png"))
    page.evaluate("scrollTo(0, 0)")


def wait_for(page, expr: str, timeout: float = 90.0) -> None:
    """Poll a page expression from Python (page.wait_for_function stalled on this page)."""
    t0 = time.time()
    while not page.evaluate(expr):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"timed out waiting for {expr}")
        page.wait_for_timeout(500)


def main() -> int:
    SHOTS.mkdir(exist_ok=True)
    readme_media()
    readme_vars()
    root_readme()
    prefetch_fonts()
    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                           cwd=SITE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/alt-4-handbook/index.html"
    lib = capture._threedmol_bytes()
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        with sync_playwright() as pw:
            browser = capture._launch_any(pw)

            def new_page(w, h, scheme="light", motion="no-preference"):
                ctx = browser.new_context(viewport={"width": w, "height": h}, color_scheme=scheme,
                                          reduced_motion=motion)
                # Google Fonts are served from a Python prefetch: Chromium's own sockets to the
                # font hosts drop intermittently on this machine (ERR_SOCKET_NOT_CONNECTED),
                # which is network noise, not a page error.
                ctx.route("https://fonts.googleapis.com/**", font_route)
                ctx.route("https://fonts.gstatic.com/**", font_route)
                if lib:
                    ctx.route(capture.THREEDMOL_URL, lambda r: r.fulfill(body=lib, content_type="application/javascript"))
                page = ctx.new_page()
                page.set_default_timeout(90000)
                errs: list[str] = []
                page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
                page.on("pageerror", lambda e: errs.append(str(e)))
                page.on("requestfailed", lambda r: errs.append(f"request failed: {r.url} {r.failure}"))
                return ctx, page, errs

            # weight + console + desktop shots
            for scheme in ("light", "dark"):
                ctx, page, errs = new_page(1440, 900, scheme)
                resps: list = []
                page.on("response", lambda r, resps=resps: resps.append(r))
                page.goto(url, wait_until="domcontentloaded")
                wait_for(page, "!!(window.__handbook && window.__handbook.scenes.neb && window.__handbook.scenes.mode)")
                page.wait_for_timeout(2500)
                page.screenshot(path=str(SHOTS / f"desktop-{scheme}-top.png"))
                scroll_shots(page, f"desktop-{scheme}")
                sizes: dict[str, int] = {}
                for r in resps:
                    try:
                        sizes[r.url] = len(r.body())
                    except Exception:
                        sizes.setdefault(r.url, int(r.headers.get("content-length", 0) or 0))
                if scheme == "light":
                    for sel in ("#setup", "#relax", "#committee", "#neb", "#md", "#remote"):
                        page.locator(sel).scroll_into_view_if_needed()
                        page.wait_for_timeout(600)
                        page.locator(sel).screenshot(path=str(SHOTS / f"section-{sel[1:]}.png"))
                    mb = sum(sizes.values()) / 1e6
                    # The video may stream with range requests: count its full file size.
                    vid = (HERE / "data/webapp.mp4").stat().st_size / 1e6
                    mb = max(mb, mb + vid - sum(v for k, v in sizes.items() if k.endswith(".mp4")) / 1e6)
                    check(mb <= LIMIT_MB, f"landing page weight {mb:.2f} MB <= {LIMIT_MB} MB ({len(sizes)} responses)")
                if scheme == "light":
                    blocks = page.evaluate("[...document.querySelectorAll('pre[data-copy]')].map(p => p.dataset.text || p.querySelector('code').textContent)")
                    bad = undefined_vars(blocks)
                    check(not bad, f"page copy blocks: no undefined $VAR ({len(blocks)} blocks) {sorted(bad) or ''}")
                check(not errs, f"no console errors ({scheme}) {errs[:3] if errs else ''}")
                ctx.close()

            # mobile
            for scheme in ("light", "dark"):
                ctx, page, errs = new_page(375, 812, scheme)
                page.goto(url, wait_until="domcontentloaded")
                wait_for(page, "!!(window.__handbook && window.__handbook.scenes.neb)")
                page.wait_for_timeout(1500)
                sw = page.evaluate("[document.documentElement.scrollWidth, document.documentElement.clientWidth]")
                check(sw[0] <= sw[1], f"no horizontal scroll at 375 px ({scheme}): scrollWidth {sw[0]}, clientWidth {sw[1]}")
                page.screenshot(path=str(SHOTS / f"mobile-{scheme}-top.png"))
                scroll_shots(page, f"mobile-{scheme}")
                check(not errs, f"no console errors (mobile {scheme}) {errs[:3] if errs else ''}")
                ctx.close()

            # every <details> opened: no horizontal scroll, no squeezed text (desktop and 375 px)
            for (w, h, tag) in ((1440, 900, "desktop"), (375, 812, "mobile")):
                ctx, page, errs = new_page(w, h, "light")
                page.goto(url, wait_until="domcontentloaded")
                wait_for(page, "!!(window.__handbook && window.__handbook.scenes.neb)")
                n = page.evaluate("(() => { const d = [...document.querySelectorAll('details')]; d.forEach(x => x.open = true); return d.length; })()")
                page.wait_for_timeout(1200)
                sw = page.evaluate("[document.documentElement.scrollWidth, document.documentElement.clientWidth]")
                check(sw[0] <= sw[1], f"{tag}: all {n} <details> open, no horizontal scroll: scrollWidth {sw[0]}, clientWidth {sw[1]}")
                narrow = page.evaluate("""[...document.querySelectorAll('main dd, main td, main p, main li, main figcaption, main caption')]
                    .filter(e => e.offsetParent && e.textContent.trim().length > 3 && e.getBoundingClientRect().width < 40)
                    .map(e => e.tagName + ':' + e.textContent.trim().slice(0, 30))""")
                check(not narrow, f"{tag}: no text element narrower than 40 px with details open {narrow[:5] or ''}")
                for sel in ("#machine", "#env", "#relax", "#neb", "#md"):
                    loc = page.locator(sel).locator("xpath=ancestor-or-self::*[self::article or self::section][1]")
                    page.locator(sel).scroll_into_view_if_needed()
                    page.wait_for_timeout(400)
                    page.screenshot(path=str(SHOTS / f"open-{tag}-{sel[1:]}.png"))
                check(not errs, f"no console errors ({tag}, details open) {errs[:3] if errs else ''}")
                ctx.close()

            # 3D library unavailable: the GIF fallback shows, with its caption inside the page
            ctx, page, errs = new_page(375, 812, "dark")
            ctx.route(capture.THREEDMOL_URL, lambda r: r.abort())
            page.goto(url, wait_until="domcontentloaded")
            wait_for(page, "document.querySelectorAll('.gif-fallback').length >= 3")
            page.evaluate("document.getElementById('scene-relax').scrollIntoView()")
            page.wait_for_timeout(800)
            page.screenshot(path=str(SHOTS / "fallback-mobile-dark-relax.png"))
            page.evaluate("scrollTo(0, 0)")
            page.wait_for_timeout(400)
            page.screenshot(path=str(SHOTS / "fallback-mobile-dark-top.png"))
            clipped = page.evaluate("""[...document.querySelectorAll('.gif-fallback figcaption')].filter(c => {
                const b = c.closest('.scene-box, .preview-stage').getBoundingClientRect(), r = c.getBoundingClientRect();
                return r.bottom > b.bottom + 1 || r.right > b.right + 1; }).length""")
            check(clipped == 0, f"GIF fallback captions inside their box at 375 px ({clipped} clipped)")
            ctx.close()

            # reduced motion
            ctx, page, errs = new_page(1440, 900, "light", "reduce")
            page.goto(url, wait_until="domcontentloaded")
            wait_for(page, "!!(window.__handbook && window.__handbook.scenes.neb && window.__handbook.scenes.mode && window.__handbook.scenes.relax)")
            page.wait_for_timeout(1000)
            snap = "Object.fromEntries(Object.entries(window.__handbook.scenes).map(([k,s]) => [k, [s.frame, s.playing]]))"
            a = page.evaluate(snap)
            page.wait_for_timeout(2000)
            b = page.evaluate(snap)
            vids = page.evaluate("window.__handbook.videos.map(v => [v.paused, v.currentTime])")
            check(a == b and not any(p for _, p in b.values()),
                  f"reduced motion: scenes still and not playing {b}")
            check(all(p and t == 0 for p, t in vids), f"reduced motion: videos paused at 0 s {vids}")
            anims = page.evaluate("document.getAnimations().filter(a => a.playState === 'running').length")
            check(anims == 0, f"reduced motion: no running CSS animations ({anims})")
            ctx.close()
            browser.close()
    finally:
        srv.terminate()
        srv.wait(timeout=5)
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
