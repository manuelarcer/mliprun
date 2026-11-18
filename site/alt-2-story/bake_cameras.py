"""Bake the framing camera of every scene, per layout, into data/cameras.json.

    site/.venv/bin/python site/alt-2-story/bake_cameras.py      (outside the sandbox)

The page gives each scene a fixed aspect ratio per layout (main.js ASPECT: desktop and
phone). A camera from viewer.js ctrl.getCamera() depends only on the view, the
replicate and that aspect ratio, so one auto-fit per scene and layout, done here, lets
the page load every scene with fit:false + camera and skip the pixel auto-fit.
Rerun after changing a scene's view, replicate, fill, crop or ASPECT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check  # noqa: E402  (route handler and GPU launcher)

LAYOUTS = {"desk": (1440, 900), "mobile": (375, 812)}


def main() -> int:
    out = {}
    with sync_playwright() as pw:
        browser = check.launch_gpu(pw)
        try:
            for key, (w, h) in LAYOUTS.items():
                page = browser.new_page(viewport={"width": w, "height": h})
                page.route("**/*", check.make_handler({}))
                page.goto(check.START + "?bake")
                page.wait_for_function("() => !!window.__cameras", timeout=180000, polling=500)
                got = page.evaluate("() => window.__cameras")
                if key not in got:
                    raise RuntimeError(f"layout {key}: page reported {list(got)}")
                out[key] = got[key]
                page.close()
        finally:
            browser.close()
    path = HERE / "data" / "cameras.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}: " + ", ".join(f"{k}: {sorted(v)}" for k, v in out.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
