"""Browser test (PLAN 12, Phase 3): headless Chromium via Playwright.

Starts the real server (uvicorn in a thread, real worker, EMT), uploads a Cu
POSCAR through ``upload-input``, runs an optimization and checks that the live
plot grows and the result shows; then cancels an MD job from the detail view.
Skips unless playwright is importable, a Chromium launches (bundled, then
/usr/bin/chromium) and the CDN is reachable.
"""
from __future__ import annotations

import re
import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
sync_api = pytest.importorskip("playwright.sync_api",
                               reason="playwright not installed")

import ase.io  # noqa: E402
from ase.build import bulk  # noqa: E402

CDN = ["https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js",
       "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"]
DETAIL = "[data-testid=job-detail]"


def _cdn_reachable():
    try:
        for url in CDN:
            urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=10)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def browser():
    pw = sync_api.sync_playwright().start()
    b, errors = None, []
    for kw in ({}, {"executable_path": "/usr/bin/chromium"}):
        try:
            b = pw.chromium.launch(headless=True, **kw)
            break
        except Exception as exc:
            errors.append(repr(exc)[:200])
    if b is None:
        pw.stop()
        pytest.skip(f"no Chromium could be launched: {errors}")
    if not _cdn_reachable():
        b.close()
        pw.stop()
        pytest.skip("CDN (cdn.jsdelivr.net) not reachable; plots/viewer need it")
    yield b
    b.close()
    pw.stop()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import uvicorn
    from mliprun.webapp.server import create_app
    runs = tmp_path_factory.mktemp("runs_browser")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(runs, allowed_hosts=[f"127.0.0.1:{port}", f"localhost:{port}"],
                     cancel_grace_s=3.0, poll_interval_s=0.1)
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    deadline = time.monotonic() + 30
    while not srv.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(timeout=30)


@pytest.fixture
def page(browser, server):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server)
    pg.wait_for_selector("[data-testid=calc-type-select] option[value=optimize]",
                         state="attached", timeout=30000)
    yield pg
    ctx.close()
    assert not errors, f"JavaScript errors on the page: {errors}"


def _upload(page, path: Path):
    page.set_input_files("[data-testid=upload-input]", str(path))
    item = page.locator("[data-testid=structure-item][data-ok=true]").first
    item.wait_for(timeout=30000)
    return item.get_attribute("data-upload-id")


def _choose(page, calc_type):
    page.select_option("[data-testid=calc-type-select]", calc_type)
    page.select_option("[data-testid=calculator-select]", "emt")


def _ensure_only_checked(page, upload_id):
    for item in page.locator("[data-testid=structure-item]").all():
        box = item.locator("[data-testid=structure-select]")
        want = item.get_attribute("data-upload-id") == upload_id
        if box.count() and box.is_checked() != want:
            box.set_checked(want)


def _set_field(page, name, value):
    wrapper = page.locator(f"[data-testid=field-{name}]")
    if not wrapper.is_visible():
        toggle = page.locator("[data-testid=advanced-toggle]")
        if toggle.count():
            toggle.first.click()
    ctl = wrapper.locator("input, select").first
    tag = ctl.evaluate("e => e.tagName")
    if tag == "SELECT":
        ctl.select_option(str(value))
    else:
        ctl.fill(str(value))
        ctl.dispatch_event("change")


def _detail_job_id(page, timeout=30000):
    page.wait_for_function(
        "() => { const d = document.querySelector('[data-testid=job-detail]');"
        " return d && d.getAttribute('data-job-id'); }", timeout=timeout)
    return page.get_attribute(DETAIL, "data-job-id")


PLOT_POINTS_JS = """
() => {
  const el = document.querySelector('[data-testid=plot-main] .js-plotly-plot')
           || document.querySelector('[data-testid=plot-main].js-plotly-plot');
  if (!el || !el.data) return -1;
  return Math.max(0, ...el.data.map(t => (t.x || t.y || []).length));
}
"""


def test_browser_optimize_live_plot_and_result(page, server, tmp_path):
    a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(4)       # 256 atoms
    a.rattle(0.08, seed=3)
    p = tmp_path / "POSCAR"
    ase.io.write(str(p), a, format="vasp", direct=True)
    uid = _upload(page, p)
    _choose(page, "optimize")
    _ensure_only_checked(page, uid)
    _set_field(page, "optimizer", "fire")
    _set_field(page, "fmax", "0.005")
    _set_field(page, "max_steps", "400")
    page.click("[data-testid=submit-button]")
    jid = _detail_job_id(page)
    counts = []
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        counts.append(page.evaluate(PLOT_POINTS_JS))
        status = page.get_attribute(DETAIL, "data-status")
        if status in ("done", "failed", "cancelled"):
            break
        time.sleep(0.25)
    page.wait_for_selector(f"{DETAIL}[data-status=done]", timeout=60000)
    page.wait_for_timeout(1000)
    final_count = page.evaluate(PLOT_POINTS_JS)
    live = [c for c in counts if c > 0]
    assert len(set(live)) >= 2, f"plot did not update live: {counts}"
    assert all(b >= a for a, b in zip(live, live[1:])), counts
    # the plot ends with one point per opt_step event
    import json
    evs = json.loads(urllib.request.urlopen(f"{server}/api/jobs/{jid}/events?limit=5000")
                     .read())["events"]
    n_steps = sum(e["type"] == "opt_step" for e in evs)
    assert final_count == n_steps
    detail = json.loads(urllib.request.urlopen(f"{server}/api/jobs/{jid}").read())
    energy = detail["result"]["final_energy_eV"]
    assert detail["result"]["converged"] is True
    text = page.inner_text("[data-testid=result-panel]")
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", text)]
    assert any(abs(x - energy) <= 1e-4 * max(1.0, abs(energy)) for x in nums), text
    assert page.locator(f"{DETAIL} [data-testid=job-status]").inner_text().strip().lower() \
        .startswith("done")
    assert page.locator("[data-testid=files-list]").inner_text().count("summary.json")
    row = page.locator(f"[data-testid=job-row][data-job-id='{jid}']")
    assert row.get_attribute("data-status") == "done"


def test_browser_md_cancel(page, server, tmp_path):
    a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    p = tmp_path / "md.vasp"
    ase.io.write(str(p), a, format="vasp")
    uid = _upload(page, p)
    _choose(page, "md")
    _ensure_only_checked(page, uid)
    _set_field(page, "steps", "1000000")
    page.click("[data-testid=submit-button]")
    jid = _detail_job_id(page)
    page.wait_for_selector(f"{DETAIL}[data-job-id='{jid}'][data-status=running]",
                           timeout=60000)
    page.once("dialog", lambda d: d.accept())
    page.click(f"{DETAIL} [data-testid=cancel-button]")
    page.wait_for_selector(f"{DETAIL}[data-status=cancelled]", timeout=30000)
    row = page.locator(f"[data-testid=job-row][data-job-id='{jid}']")
    row.wait_for()
    page.wait_for_function(
        f"() => document.querySelector(\"[data-testid=job-row][data-job-id='{jid}']\")"
        ".getAttribute('data-status') === 'cancelled'", timeout=15000)
