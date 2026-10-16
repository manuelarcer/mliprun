"""Revision 2 browser tests (PLAN 15.1, 15.2, 11.2): headless Chromium via Playwright.

Job name / location / folder inputs, the multi-structure ``name-preview``
table in natural filename order, Submit blocked on an existing folder, inputs
cleared after submit; NEB band-evolution plot, ``viewer-title`` and the
hidden phase on finished jobs. Real server (uvicorn thread), real worker, EMT.
"""
from __future__ import annotations

import json
import re
import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

import ase.io  # noqa: E402
from ase.build import add_adsorbate, bulk, fcc100  # noqa: E402
from ase.constraints import FixAtoms  # noqa: E402

CDN = ["https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js",
       "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"]
DETAIL = "[data-testid=job-detail]"
TITLE_RE = re.compile(r"^Image with highest energy - (\d{2,3})$")


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
    for kw in ({"executable_path": "/usr/bin/chromium"}, {}):
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


@pytest.fixture
def server(tmp_path):
    """A fresh server per test so the structure list holds only this test's uploads."""
    import uvicorn
    from mliprun.webapp.server import create_app
    runs = tmp_path / "runs"
    runs.mkdir()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert port != 8765
    app = create_app(runs, allowed_hosts=[f"127.0.0.1:{port}", f"localhost:{port}"],
                     cancel_grace_s=3.0, poll_interval_s=0.1)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
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


def api(server, path):
    return json.loads(urllib.request.urlopen(f"{server}{path}").read())


def upload_files(page, paths, fmt="vasp"):
    n_before = page.locator("[data-testid=structure-item][data-ok=true]").count()
    page.select_option("[data-testid=upload-format]", fmt)
    page.set_input_files("[data-testid=upload-input]", [str(p) for p in paths])
    page.wait_for_function(
        f"() => document.querySelectorAll('[data-testid=structure-item][data-ok=true]').length"
        f" >= {n_before + len(paths)}", timeout=30000)


def select_only(page, upload_ids):
    for item in page.locator("[data-testid=structure-item]").all():
        box = item.locator("[data-testid=structure-select]")
        want = item.get_attribute("data-upload-id") in upload_ids
        if box.count() and box.is_checked() != want:
            box.set_checked(want)


def choose(page, calc_type):
    page.select_option("[data-testid=calc-type-select]", calc_type)
    page.select_option("[data-testid=calculator-select]", "emt")


def fill_job(page, which, value):
    ctl = page.locator(f"[data-testid=field-job-{which}] [data-testid=job-{which}]")
    ctl.fill(value)
    ctl.dispatch_event("input")
    ctl.dispatch_event("change")


def set_field(page, name, value):
    wrapper = page.locator(f"[data-testid=field-{name}]")
    if not wrapper.is_visible():
        toggle = page.locator("[data-testid=advanced-toggle]")
        if toggle.count():
            toggle.first.click()
    ctl = wrapper.locator("input, select").first
    if ctl.evaluate("e => e.tagName") == "SELECT":
        ctl.select_option(str(value))
    else:
        ctl.fill(str(value))
        ctl.dispatch_event("change")


def uploads_by_name(server):
    return {u["filename"]: u["id"] for u in api(server, "/api/uploads")["uploads"]}


def preview_rows(page):
    return page.locator("[data-testid=name-preview] [data-conflict]")


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------

NAMES = ["cu_10", "Cu_3", "cu_1", "cu_2"]
NATURAL = ["cu_1", "cu_2", "Cu_3", "cu_10"]


def test_common_name_preview_natural_order_and_submit(page, server, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for n in NAMES:
        p = src / n
        ase.io.write(str(p), bulk("Cu", "fcc", a=3.6), format="vasp")
        paths.append(p)
    upload_files(page, paths)
    ids = uploads_by_name(server)
    assert set(NAMES) <= set(ids)
    choose(page, "singlepoint")
    select_only(page, {ids[n] for n in NAMES})
    loc = tmp_path / "calcs" / "ch3"
    fill_job(page, "location", str(loc))
    fill_job(page, "name", "ch3-is-proposal")
    page.wait_for_function(
        "() => document.querySelectorAll('[data-testid=name-preview] [data-conflict]').length === 4",
        timeout=10000)
    page.wait_for_timeout(800)          # debounced live check
    rows = preview_rows(page)
    texts = [rows.nth(i).inner_text() for i in range(4)]
    for k, (t, fname) in enumerate(zip(texts, NATURAL)):
        assert re.search(rf"(^|\s){re.escape(fname)}(\s|$)", t), (k, texts)
        assert f"ch3-is-proposal-{k:02d}" in t, (k, texts)
        assert str(loc / f"ch3-is-proposal-{k:02d}") in t, (k, texts)
        assert rows.nth(k).get_attribute("data-conflict") == "none"
    assert page.locator("[data-testid=submit-button]").is_enabled()
    page.click("[data-testid=submit-button]")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (loc / "ch3-is-proposal-03").is_dir():
        time.sleep(0.1)
    assert sorted(p.name for p in loc.iterdir()) == [f"ch3-is-proposal-{k:02d}" for k in range(4)]
    for k, fname in enumerate(NATURAL):
        jj = json.loads((loc / f"ch3-is-proposal-{k:02d}" / "job.json").read_text())
        assert ids[fname] in json.dumps(jj), (k, fname)
    labels = {j["label"] for j in api(server, "/api/jobs")["jobs"]}
    assert labels == {f"ch3-is-proposal-{k:02d}" for k in range(4)}
    # name and folder cleared, location kept
    page.wait_for_function(
        "() => document.querySelector('[data-testid=job-name]').value === ''", timeout=10000)
    assert page.input_value("[data-testid=job-folder]") == ""
    assert page.input_value("[data-testid=job-location]") == str(loc)


def test_submit_blocked_when_folder_exists(page, server, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for n in ("a1", "a2"):
        p = src / n
        ase.io.write(str(p), bulk("Cu", "fcc", a=3.6), format="vasp")
        paths.append(p)
    upload_files(page, paths)
    ids = uploads_by_name(server)
    loc = tmp_path / "L"
    (loc / "blocked").mkdir(parents=True)
    (loc / "nm-01").mkdir()
    choose(page, "singlepoint")
    # single structure, folder exists
    select_only(page, {ids["a1"]})
    fill_job(page, "location", str(loc))
    fill_job(page, "folder", "blocked")
    page.wait_for_function(
        "() => document.querySelector('[data-testid=submit-button]').disabled", timeout=10000)
    reason = page.locator("[data-testid=submit-blocked-reason]")
    reason.wait_for(state="visible", timeout=5000)
    assert reason.inner_text().strip()
    # multi: one row flagged exists
    fill_job(page, "folder", "")
    select_only(page, {ids["a1"], ids["a2"]})
    fill_job(page, "name", "nm")
    page.wait_for_function(
        "() => [...document.querySelectorAll('[data-testid=name-preview] [data-conflict]')]"
        ".map(r => r.getAttribute('data-conflict')).join() === 'none,exists'", timeout=10000)
    assert page.locator("[data-testid=submit-button]").is_disabled()
    assert page.locator("[data-testid=submit-blocked-reason]").inner_text().strip()
    # resolve the conflict: Submit enabled again
    fill_job(page, "name", "nm2")
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=submit-button]').disabled", timeout=10000)
    assert sorted(p.name for p in loc.iterdir()) == ["blocked", "nm-01"]


def test_invalid_location_blocks_submit(page, server, tmp_path):
    p = tmp_path / "b1"
    ase.io.write(str(p), bulk("Cu", "fcc", a=3.6), format="vasp")
    upload_files(page, [p])
    ids = uploads_by_name(server)
    choose(page, "singlepoint")
    select_only(page, {ids["b1"]})
    fill_job(page, "location", "relative/dir")
    page.wait_for_function(
        "() => document.querySelector('[data-testid=submit-button]').disabled", timeout=10000)
    assert page.locator("[data-testid=submit-blocked-reason]").inner_text().strip()


# ---------------------------------------------------------------------------
# NEB live view
# ---------------------------------------------------------------------------

BAND_TRACES_JS = """
() => {
  const w = document.querySelector('[data-testid=plot-main] [data-testid=plot-neb-band]');
  if (!w) return -2;
  const el = w.classList.contains('js-plotly-plot') ? w : w.querySelector('.js-plotly-plot');
  if (!el || !el.data) return -1;
  return el.data.filter(t => (t.y || []).length > 1).length;
}
"""

HEADER_TEXT_JS = """
() => {
  const d = document.querySelector('[data-testid=job-detail]').cloneNode(true);
  for (const s of ['log-panel', 'files-list', 'reproduce-panel', 'result-panel', 'viewer',
                   'plot-main', 'plot-secondary'])
    d.querySelectorAll(`[data-testid=${s}]`).forEach(e => e.remove());
  return d.innerText;
}
"""


def neb_endpoints():
    slab = fcc100("Cu", size=(3, 3, 3), vacuum=6.0)
    add_adsorbate(slab, "Cu", 1.7, "hollow")
    slab.set_constraint(FixAtoms(indices=[a.index for a in slab if a.tag == 3]))
    final = slab.copy()
    final.positions[-1, 0] += bulk("Cu", "fcc").cell.cellpar()[0]
    return slab, final


def _select_by_id(page, testid, upload_id, filename):
    sel = page.locator(f"[data-testid={testid}]")
    try:
        sel.select_option(value=upload_id, timeout=3000)
    except Exception:
        opts = sel.locator("option").all()
        for o in opts:
            if filename in o.inner_text():
                sel.select_option(value=o.get_attribute("value"))
                return
        raise


def test_neb_band_plot_viewer_title_and_phase_hidden(page, server, tmp_path):
    a, b = neb_endpoints()
    pi, pf = tmp_path / "ini_hop", tmp_path / "fin_hop"
    ase.io.write(str(pi), a, format="vasp", direct=True)
    ase.io.write(str(pf), b, format="vasp", direct=True)
    upload_files(page, [pi, pf])
    ids = uploads_by_name(server)
    choose(page, "neb")
    page.locator("[data-testid=neb-mode-pair]").check()
    _select_by_id(page, "neb-initial-select", ids["ini_hop"], "ini_hop")
    _select_by_id(page, "neb-final-select", ids["fin_hop"], "fin_hop")
    set_field(page, "n_images", "5")
    set_field(page, "fmax", "0.02")
    se = page.locator("[data-testid=structure-every] input, input[data-testid=structure-every]")
    if se.count():
        se.first.fill("1")
        se.first.dispatch_event("change")
    fill_job(page, "name", "hop neb")
    page.click("[data-testid=submit-button]")
    page.wait_for_function(
        "() => { const d = document.querySelector('[data-testid=job-detail]');"
        " return d && d.getAttribute('data-job-id'); }", timeout=30000)
    jid = page.get_attribute(DETAIL, "data-job-id")
    counts, titles, phase_seen_running = [], [], False
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        counts.append(page.evaluate(BAND_TRACES_JS))
        t = page.locator("[data-testid=viewer-title]")
        if t.count():
            titles.append(t.first.inner_text().strip())
        status = page.get_attribute(DETAIL, "data-status")
        if status == "running" and "phase:" in page.evaluate(HEADER_TEXT_JS):
            phase_seen_running = True
        if status in ("done", "failed", "cancelled"):
            break
        time.sleep(0.2)
    page.wait_for_selector(f"{DETAIL}[data-status=done]", timeout=60000)
    page.wait_for_timeout(1500)
    detail = api(server, f"/api/jobs/{jid}")
    evs = api(server, f"/api/jobs/{jid}/events?limit=5000")["events"]
    iters = [e for e in evs if e["type"] == "neb_iter"]
    assert len(iters) >= 3, len(iters)
    live = [c for c in counts if c > 0]
    assert len(set(live)) >= 2, f"band plot did not gain traces: {counts}"
    assert all(y >= x for x, y in zip(live, live[1:])) or max(live) >= 40, counts
    final = page.evaluate(BAND_TRACES_JS)
    assert final >= min(len(iters), 2)
    # viewer title: live values well-formed, final one = last event = result
    assert all(TITLE_RE.match(t) for t in titles if t), titles
    title = page.locator("[data-testid=viewer-title]").inner_text().strip()
    m = TITLE_RE.match(title)
    assert m, title
    nn = int(m.group(1))
    assert m.group(1) == f"{nn:02d}"
    assert nn == iters[-1]["highest_image"]
    assert nn == detail["result"]["ts_image_index"]
    # phase hidden once terminal
    assert "phase:" not in page.evaluate(HEADER_TEXT_JS)
    if not phase_seen_running:
        pytest.xfail("never observed 'phase:' while running (job may have been too fast); "
                     "terminal check passed")


def test_natural_order_tie_break_and_reload(page, server, tmp_path):
    """Equal filenames: older upload first; the mapping survives a page reload."""
    d1, d2 = tmp_path / "u1", tmp_path / "u2"
    d1.mkdir(), d2.mkdir()
    for d in (d1, d2):
        ase.io.write(str(d / "same"), bulk("Cu", "fcc", a=3.6), format="vasp")
    ase.io.write(str(d1 / "x_9"), bulk("Cu", "fcc", a=3.6), format="vasp")
    upload_files(page, [d1 / "same", d1 / "x_9"])
    time.sleep(1.1)
    upload_files(page, [d2 / "same"])
    ups = api(server, "/api/uploads")["uploads"]
    same = sorted([u for u in ups if u["filename"] == "same"], key=lambda u: u["created_at"])
    old_id, new_id = same[0]["id"], same[1]["id"]
    x9 = next(u["id"] for u in ups if u["filename"] == "x_9")
    page.reload()
    page.wait_for_selector("[data-testid=calc-type-select] option[value=optimize]",
                           state="attached", timeout=30000)
    page.wait_for_function(
        "() => document.querySelectorAll('[data-testid=structure-item][data-ok=true]').length >= 3",
        timeout=30000)
    choose(page, "singlepoint")
    select_only(page, {old_id, new_id, x9})
    loc = tmp_path / "T"
    fill_job(page, "location", str(loc))
    fill_job(page, "name", "tie")
    page.wait_for_function(
        "() => document.querySelectorAll('[data-testid=name-preview] [data-conflict]').length === 3",
        timeout=10000)
    texts = [preview_rows(page).nth(i).inner_text() for i in range(3)]
    assert "same" in texts[0] and "same" in texts[1] and "x_9" in texts[2], texts
    page.click("[data-testid=submit-button]")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (loc / "tie-02").is_dir():
        time.sleep(0.1)
    j0 = (loc / "tie-00" / "job.json").read_text()
    j1 = (loc / "tie-01" / "job.json").read_text()
    j2 = (loc / "tie-02" / "job.json").read_text()
    assert old_id in j0 and new_id in j1 and x9 in j2
