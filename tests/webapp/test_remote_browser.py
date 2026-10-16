"""Remote machines in the browser (REMOTE_PLAN section 9): headless Chromium + fake ssh.

The "remote machine" is a temp directory on this computer reached through
``tests/webapp/fake_ssh.py``; the worker is the real one (EMT). Skip rules are
identical to ``test_rev2_browser.py`` (no Chromium or no CDN -> skipped).
"""
from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

import ase.io  # noqa: E402
from ase.build import bulk  # noqa: E402
from remote_helpers import FAKE_SSH_CMD  # noqa: E402

CDN = ["https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js",
       "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"]
DETAIL = "[data-testid=job-detail]"
PLOT_POINTS_JS = """
() => {
  const el = document.querySelector('[data-testid=plot-main] .js-plotly-plot')
           || document.querySelector('[data-testid=plot-main].js-plotly-plot');
  if (!el || !el.data) return -1;
  return Math.max(0, ...el.data.map(t => (t.x || t.y || []).length));
}
"""
NEW_CARD = "[data-testid=machine-card][data-machine-id='']"


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
def start_server(tmp_path):
    """``start_server(machines_file=None) -> url``: a fresh server with the fake ssh."""
    import uvicorn

    from mliprun.webapp.server import create_app

    started = []

    def factory(machines_file=None):
        runs = tmp_path / "runs"
        runs.mkdir(exist_ok=True)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        app = create_app(runs, allowed_hosts=[f"127.0.0.1:{port}", f"localhost:{port}"],
                         cancel_grace_s=3.0, poll_interval_s=0.1,
                         machines_file=machines_file or tmp_path / "machines.json",
                         ssh_command=FAKE_SSH_CMD, remote_backoff_base=0.05)
        srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        th = threading.Thread(target=srv.run, daemon=True)
        th.start()
        deadline = time.monotonic() + 30
        while not srv.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.05)
        started.append((srv, th))
        return f"http://127.0.0.1:{port}"

    yield factory
    for srv, th in started:
        srv.should_exit = True
        th.join(timeout=30)


@pytest.fixture
def new_page(browser):
    """``new_page(url) -> page`` with a JS-error check at teardown."""
    contexts, errors = [], []

    def factory(url):
        ctx = browser.new_context(viewport={"width": 1280, "height": 1000})
        contexts.append(ctx)
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(url)
        pg.wait_for_selector("[data-testid=calc-type-select] option[value=optimize]",
                             state="attached", timeout=30000)
        return pg

    yield factory
    for c in contexts:
        c.close()
    assert not errors, f"JavaScript errors on the page: {errors}"


def get(url):
    return json.loads(urllib.request.urlopen(url).read())


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"X-Mliprun": "1", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def fill(page, selector, value):
    ctl = page.locator(selector)
    ctl.fill(value)
    ctl.dispatch_event("input")
    ctl.dispatch_event("change")


def expected_env_name(machine, python):
    p = Path(python)
    venv = p.parent.parent.name if p.parent.name == "bin" and p.parent.parent.name else p.parent.name
    return f"{machine}-{re.sub(r'[^A-Za-z0-9._+-]', '-', venv) or 'python'}"


def test_remote_flow_machine_env_run_files(new_page, start_server, tmp_path):
    remote_loc = tmp_path / "remote-box"
    remote_loc.mkdir()
    url = start_server()
    page = new_page(url)
    # No machine yet: no "Run on" choice is offered, the form is the local one.
    assert page.locator("[data-testid=field-run-on]").is_hidden()

    # ---- add a machine through the UI
    page.click("[data-testid=machines-button]")
    page.wait_for_selector(NEW_CARD, state="visible")
    fill(page, f"{NEW_CARD} [data-testid=machine-name]", "box")
    fill(page, f"{NEW_CARD} [data-testid=machine-host]", "fakehost")
    fill(page, f"{NEW_CARD} [data-testid=machine-default_location]", str(remote_loc))
    fill(page, f"{NEW_CARD} [data-testid=machine-poll_interval_s]", "0.5")
    fill(page, f"{NEW_CARD} [data-testid=machine-copy_back_max_mb]", "0")
    # a bad value shows a per-field error next to the field
    fill(page, f"{NEW_CARD} [data-testid=machine-host]", "bad host!")
    page.click(f"{NEW_CARD} [data-testid=machine-save-button]")
    err = page.locator(f"{NEW_CARD} [data-testid=machine-field-error]")
    err.first.wait_for(state="visible", timeout=10000)
    assert err.count() == 1 and err.first.inner_text().strip()
    assert get(f"{url}/api/machines")["machines"] == []
    fill(page, f"{NEW_CARD} [data-testid=machine-host]", "fakehost")
    page.click(f"{NEW_CARD} [data-testid=machine-save-button]")
    page.wait_for_selector("[data-testid=machine-card]:not([data-machine-id=''])", timeout=15000)
    machines = get(f"{url}/api/machines")["machines"]
    assert len(machines) == 1
    m = machines[0]
    assert (m["name"], m["host"], m["default_location"]) == ("box", "fakehost", str(remote_loc))
    assert m["poll_interval_s"] == 0.5 and m["copy_back_max_mb"] == 0
    card = f"[data-testid=machine-card][data-machine-id='{m['id']}']"

    # ---- test connection (no env yet: hostname only)
    page.click(f"{card} [data-testid=machine-test-button]")
    page.wait_for_selector(f"{card} [data-testid=machine-test-hostname]", timeout=30000)
    expect_host = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
    assert page.inner_text(f"{card} [data-testid=machine-test-hostname]") == expect_host

    # ---- add an env: the name is pre-filled from the python path and stays editable
    add = f"{card} [data-testid=env-add]"
    name_in = f"{add} [data-testid=env-name]"
    fill(page, f"{add} [data-testid=env-python]", sys.executable)
    want = expected_env_name("box", sys.executable)
    assert page.input_value(name_in) == want
    fill(page, name_in, "box-emt")                       # the user edits the name ...
    fill(page, f"{add} [data-testid=env-python]", sys.executable + " ")   # ... and typing on keeps it
    assert page.input_value(name_in) == "box-emt"
    fill(page, f"{add} [data-testid=env-python]", sys.executable)
    page.click(f"{add} [data-testid=env-add-button]")
    row = f"{card} [data-testid=env-row]"
    page.wait_for_selector(f"{row}[data-env-name='box-emt'] [data-testid=env-probe-status][data-ok=true]",
                           timeout=120000)
    status = page.inner_text(f"{row} [data-testid=env-probe-status]")
    envs = get(f"{url}/api/machines")["machines"][0]["envs"]
    assert len(envs) == 1 and envs[0]["name"] == "box-emt" and envs[0]["probe"]["ok"] is True
    env_id = envs[0]["id"]
    assert f"mliprun {envs[0]['probe']['mliprun_version']}" in status
    assert "CUDA" in status and "emt" in status
    # explicit probe button gives the same ok result
    page.click(f"{row} [data-testid=env-probe-button]")
    page.wait_for_function(
        "() => document.querySelector('[data-testid=env-probe-button]').textContent === 'Probe'",
        timeout=120000)
    assert page.get_attribute(f"{row} [data-testid=env-probe-status]", "data-ok") == "true"
    page.click("[data-testid=machines-close]")

    # ---- the Run on select appeared without a page reload
    run_on = page.locator("[data-testid=run-on-select]")
    assert run_on.is_visible()
    labels = run_on.locator("option").all_inner_texts()
    assert labels == ["This computer", "box-emt (box)"], labels
    run_on.select_option(env_id)
    page.wait_for_function(
        "() => document.querySelector('label[for=job-location]').textContent === 'Remote location'",
        timeout=30000)
    assert page.input_value("[data-testid=job-location]") == str(remote_loc)
    runs_dir = get(f"{url}/api/health")["runs_dir"]
    assert page.input_value("[data-testid=job-local-location]") == runs_dir
    page.select_option("[data-testid=calc-type-select]", "optimize")
    page.select_option("[data-testid=calculator-select]", "emt")
    page.select_option("[data-testid=device-select]", "cpu")

    # ---- upload, choose, submit
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.rattle(0.15, seed=2)
    poscar = tmp_path / "POSCAR"
    ase.io.write(str(poscar), atoms, format="vasp", direct=True)
    page.select_option("[data-testid=upload-format]", "vasp")
    page.set_input_files("[data-testid=upload-input]", [str(poscar)])
    page.wait_for_selector("[data-testid=structure-item][data-ok=true]", timeout=30000)
    box = page.locator("[data-testid=structure-item] [data-testid=structure-select]").first
    if not box.is_checked():
        box.set_checked(True)
    fill(page, "[data-testid=job-folder]", "rjob")
    page.wait_for_function(
        "(t) => document.querySelector('[data-testid=folder-status]').textContent.includes(t)",
        arg=f"Remote folder: {remote_loc}/rjob", timeout=15000)
    fs = page.inner_text("[data-testid=folder-status]")
    assert f"Local copy: {runs_dir}/rjob" in fs, fs
    assert page.locator("[data-testid=submit-button]").is_enabled()
    page.click("[data-testid=submit-button]")
    page.wait_for_selector(f"{DETAIL}[data-status=done]", timeout=180000)

    jobs = get(f"{url}/api/jobs")["jobs"]
    assert len(jobs) == 1
    jid = jobs[0]["job_id"]
    d = get(f"{url}/api/jobs/{jid}")
    assert d["target"]["env_name"] == "box-emt" and d["target"]["machine_name"] == "box"
    assert d["remote_run_dir"] == str(remote_loc / "rjob")
    assert Path(d["run_dir"]) == Path(runs_dir) / "rjob"
    n_steps = sum(e["type"] == "opt_step"
                  for e in get(f"{url}/api/jobs/{jid}/events?limit=5000")["events"])
    assert n_steps >= 1
    page.wait_for_function(f"() => ({PLOT_POINTS_JS})() === {n_steps}", timeout=15000)
    # the live plot has one point per opt_step event, and the energy shown is the API result
    energy = d["result"]["final_energy_eV"]
    text = page.inner_text("[data-testid=result-panel]")
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", text)]
    assert any(abs(x - energy) <= 1e-4 * max(1.0, abs(energy)) for x in nums), (energy, text)

    # ---- job list tag and job page lines
    row = page.locator(f"[data-testid=job-row][data-job-id='{jid}']")
    assert row.locator("[data-testid=job-target]").inner_text().strip() == "box-emt"
    assert page.inner_text(f"{DETAIL} [data-testid=job-runs-on]").strip() == "Runs on: box-emt (box)"
    assert page.inner_text(f"{DETAIL} [data-testid=job-remote-folder]").strip() == \
        f"Remote folder: {remote_loc / 'rjob'}"
    assert page.locator("[data-testid=job-remote-error]").is_hidden()

    # ---- copy_back_max_mb=0: files stay on the remote, listed as such, and download
    page.wait_for_selector("[data-testid=file-item][data-remote-only=true]", timeout=30000)
    entries = [f for f in get(f"{url}/api/jobs/{jid}/files")["files"] if f.get("remote_only")]
    assert entries
    rel = entries[0]["path"]
    item = page.locator(f"[data-testid=file-item][data-path='{rel}']")
    assert "not copied" in item.inner_text() and "on box" in item.inner_text()
    assert not (Path(d["run_dir"]) / rel).exists()
    remote_bytes = (remote_loc / "rjob" / rel).read_bytes()
    assert len(remote_bytes) == entries[0]["size"] > 0
    href = item.locator("a").first.get_attribute("href")
    assert href == f"/api/jobs/{jid}/files/{rel}"
    assert urllib.request.urlopen(url + href).read() == remote_bytes
    with page.expect_download() as dl:
        item.locator("a").first.click()
    assert Path(dl.value.path()).read_bytes() == remote_bytes

    # ---- the remote choice is remembered across a reload; local payloads carry no target
    page.reload()
    page.wait_for_selector("[data-testid=run-on-select] option[value='%s']" % env_id,
                           state="attached", timeout=30000)
    page.wait_for_function(
        "(id) => document.querySelector('[data-testid=run-on-select]').value === id", arg=env_id,
        timeout=30000)
    assert page.input_value("[data-testid=job-location]") == str(remote_loc)
    page.select_option("[data-testid=run-on-select]", "local")
    page.wait_for_function(
        "() => document.querySelector('label[for=job-location]').textContent === 'Location'")
    assert page.locator("[data-testid=field-job-local-location]").is_hidden()
    assert page.input_value("[data-testid=job-location]") == runs_dir


def test_remote_unreachable_blocks_submit(new_page, start_server, remote_setup, tmp_path):
    setup = remote_setup(host="down")          # probed ok earlier; the host is now down
    url = start_server(setup["machines_file"])
    page = new_page(url)
    run_on = page.locator("[data-testid=run-on-select]")
    run_on.wait_for(state="visible", timeout=15000)
    run_on.select_option(setup["env_id"])
    page.wait_for_function(
        "() => document.querySelector('label[for=job-location]').textContent === 'Remote location'",
        timeout=30000)
    assert page.input_value("[data-testid=job-location]") == str(setup["location"])
    poscar = tmp_path / "POSCAR"
    ase.io.write(str(poscar), bulk("Cu", "fcc", a=3.6, cubic=True), format="vasp", direct=True)
    page.select_option("[data-testid=upload-format]", "vasp")
    page.set_input_files("[data-testid=upload-input]", [str(poscar)])
    page.wait_for_selector("[data-testid=structure-item][data-ok=true]", timeout=30000)
    box = page.locator("[data-testid=structure-item] [data-testid=structure-select]").first
    if not box.is_checked():
        box.set_checked(True)
    fill(page, "[data-testid=job-folder]", "never")
    expected = post(f"{url}/api/paths/check",
                    {"location": str(setup["location"]), "folders": ["never"],
                     "target": setup["env_id"]})["error"]
    assert expected
    page.wait_for_function(
        "() => document.querySelector('[data-testid=location-status]').textContent.length > 0",
        timeout=30000)
    assert page.inner_text("[data-testid=location-status]").strip() == expected
    page.wait_for_function(
        "() => document.querySelector('[data-testid=submit-button]').disabled", timeout=10000)
    reason = page.inner_text("[data-testid=submit-blocked-reason]")
    assert "Remote location" in reason and expected in reason
    assert get(f"{url}/api/jobs")["jobs"] == []
    assert not (setup["location"] / "never").exists()
