"""Revision 2 (PLAN 15.1, 15.2, 15.3): job name, location, folder naming.

Independent tester, written against webapp/docs/PLAN.md section 15 only.
API-level tests use the scripted fake worker (conftest ``make_app``) so they
are fast; the NEB/AutoNEB event tests and the zip/files checks on finished
jobs use the real worker with ASE EMT.
"""
from __future__ import annotations

import gc
import io
import json
import os
import re
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import ase.io  # noqa: E402
from ase.build import add_adsorbate, bulk, fcc100  # noqa: E402
from ase.constraints import FixAtoms  # noqa: E402

H = {"X-Mliprun": "1"}
EMT_CPU = {"name": "emt", "device": "cpu"}
TERMINAL = ("done", "failed", "cancelled")
SLUG_CHARS = re.compile(r"^[A-Za-z0-9._+-]+$")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def tc(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def up(client, tmp, name, atoms=None, fmt="vasp"):
    atoms = atoms if atoms is not None else bulk("Cu", "fcc", a=3.6)
    p = tmp / f"_src_{name}"
    ase.io.write(str(p), atoms, format=fmt)
    r = client.post("/api/uploads", headers=H, data={"format": fmt},
                    files=[("files", (name, p.read_bytes()))])
    assert r.status_code == 200, r.text
    u = r.json()["uploads"][0]
    assert u["ok"], u
    return u


def body(ids, calc_type="singlepoint", **kw):
    b = {"calc_type": calc_type, "calculator": dict(EMT_CPU), "params": {},
         "structure_ids": list(ids)}
    b.update(kw)
    return b


def post(client, b):
    return client.post("/api/jobs", headers=H, json=b)


def details(r):
    assert r.status_code == 422, (r.status_code, r.text)
    err = r.json()["error"]
    assert err["code"] == "validation_error"
    assert err["details"]
    return err["details"]


def has(dets, field, code):
    return any(d.get("field") == field and d.get("code") == code for d in dets), dets


def wait(client, jid, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        d = client.get(f"/api/jobs/{jid}").json()
        if d["status"] in TERMINAL:
            return d
        time.sleep(0.05)
    raise AssertionError(f"{jid} not terminal: {d['status']}")


def snapshot(*dirs):
    """Set of all paths under the given directories (to prove nothing was created)."""
    out = set()
    for d in dirs:
        d = Path(d)
        if d.exists():
            out |= {str(p) for p in d.rglob("*")}
            out.add(str(d))
    return out


def check(client, location, folders):
    r = client.post("/api/paths/check", headers=H,
                    json={"location": location, "folders": folders})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def paused(make_app):
    """No queue runner: jobs stay pending, so submissions are cheap."""
    with tc(make_app(start_runner=False)) as c:
        yield c


@pytest.fixture
def live(make_app):
    with tc(make_app()) as c:
        yield c


# ---------------------------------------------------------------------------
# single job: label / location / folder
# ---------------------------------------------------------------------------

def test_single_default_name_location_folder(paused, runs_dir, tmp_path):
    u = up(paused, tmp_path, "POSCAR")
    r = post(paused, body([u["id"]]))
    assert r.status_code == 201, r.text
    j = r.json()["jobs"][0]
    jid = j["job_id"]
    assert j["label"] == "POSCAR"
    assert j["folder"] == jid
    assert Path(j["run_dir"]).is_absolute()
    assert Path(j["run_dir"]) == (runs_dir / jid).resolve()
    assert (runs_dir / jid / "job.json").is_file()
    d = paused.get(f"/api/jobs/{jid}").json()
    assert d["folder"] == jid and Path(d["run_dir"]) == (runs_dir / jid).resolve()
    jj = json.loads((runs_dir / jid / "job.json").read_text())
    assert jj["folder"] == jid
    assert "location" in jj
    idx = json.loads((runs_dir / "_jobs" / f"{jid}.json").read_text())
    assert idx["job_id"] == jid and Path(idx["run_dir"]) == (runs_dir / jid).resolve()


def test_single_custom_label_gives_slug_folder(paused, runs_dir, tmp_path):
    u = up(paused, tmp_path, "POSCAR")
    r = post(paused, body([u["id"]], label="My Run #1"))
    assert r.status_code == 201, r.text
    j = r.json()["jobs"][0]
    assert j["label"] == "My Run #1"
    assert j["folder"] == "My-Run-1"
    assert (runs_dir / "My-Run-1" / "job.json").is_file()


def test_single_custom_location_created_and_folder(paused, tmp_path):
    u = up(paused, tmp_path, "POSCAR")
    loc = tmp_path / "elsewhere" / "deep" / "calcs"
    assert not loc.exists()
    r = post(paused, body([u["id"]], label="ch3 IS", location=str(loc), folder="my_fold"))
    assert r.status_code == 201, r.text
    j = r.json()["jobs"][0]
    assert j["label"] == "ch3 IS" and j["folder"] == "my_fold"
    assert Path(j["run_dir"]) == (loc / "my_fold").resolve()
    assert (loc / "my_fold" / "job.json").is_file()
    jj = json.loads((loc / "my_fold" / "job.json").read_text())
    assert jj["folder"] == "my_fold"
    assert Path(jj["location"]).resolve() == loc.resolve()


def test_tilde_location_expanded(paused, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    u = up(paused, tmp_path, "POSCAR")
    r = post(paused, body([u["id"]], location="~/calcs/ch3", folder="x"))
    assert r.status_code == 201, r.text
    assert Path(r.json()["jobs"][0]["run_dir"]) == (tmp_path / "home/calcs/ch3/x").resolve()


def test_empty_strings_mean_default(paused, runs_dir, tmp_path):
    u = up(paused, tmp_path, "POSCAR")
    r = post(paused, body([u["id"]], label="", location="", folder=""))
    assert r.status_code == 201, r.text
    j = r.json()["jobs"][0]
    assert j["label"] == "POSCAR" and j["folder"] == j["job_id"]
    assert Path(j["run_dir"]).parent == runs_dir.resolve()


def test_label_too_long_rejected(paused, tmp_path):
    u = up(paused, tmp_path, "POSCAR")
    r = post(paused, body([u["id"]], label="a" * 81))
    dets = details(r)
    assert any(d["field"] == "label" for d in dets), dets


# ---------------------------------------------------------------------------
# multi-structure batches
# ---------------------------------------------------------------------------

def test_batch_of_10_common_name(paused, runs_dir, tmp_path):
    loc = tmp_path / "batch"
    ups = [up(paused, tmp_path, f"cu_{i}") for i in range(10)]
    ids = [u["id"] for u in ups]
    # the API keeps structure_ids order as given: submit in a scrambled order
    order = [3, 0, 9, 1, 7, 2, 8, 4, 6, 5]
    r = post(paused, body([ids[i] for i in order], label="ch3-is-proposal", location=str(loc)))
    assert r.status_code == 201, r.text
    jobs = r.json()["jobs"]
    assert len(jobs) == 10
    for k, (j, i) in enumerate(zip(jobs, order)):
        assert j["label"] == f"ch3-is-proposal-{k:02d}"
        assert j["folder"] == f"ch3-is-proposal-{k:02d}"
        assert Path(j["run_dir"]) == (loc / f"ch3-is-proposal-{k:02d}").resolve()
        jj = json.loads((loc / j["folder"] / "job.json").read_text())
        # job k is the k-th given structure id
        inputs_ids = json.dumps(jj)
        assert ids[i] in inputs_ids
    on_disk = sorted(p.name for p in loc.iterdir())
    assert on_disk == [f"ch3-is-proposal-{k:02d}" for k in range(10)]
    # every job is in the _jobs index
    assert {p.stem for p in (runs_dir / "_jobs").glob("*.json")} >= {j["job_id"] for j in jobs}


def test_batch_label_null_defaults_no_suffix(paused, runs_dir, tmp_path):
    ups = [up(paused, tmp_path, n) for n in ("a.vasp", "b.vasp")]
    r = post(paused, body([u["id"] for u in ups]))
    assert r.status_code == 201, r.text
    jobs = r.json()["jobs"]
    assert [j["label"] for j in jobs] == ["a.vasp", "b.vasp"]
    assert [j["folder"] for j in jobs] == [j["job_id"] for j in jobs]


def test_batch_folder_without_label(paused, tmp_path):
    ups = [up(paused, tmp_path, n) for n in ("a.vasp", "b.vasp", "c.vasp")]
    r = post(paused, body([u["id"] for u in ups], folder="base dir", location=str(tmp_path / "L")))
    assert r.status_code == 201, r.text
    jobs = r.json()["jobs"]
    assert [j["label"] for j in jobs] == ["a.vasp", "b.vasp", "c.vasp"]
    assert [j["folder"] for j in jobs] == ["base-dir-00", "base-dir-01", "base-dir-02"]


def test_batch_label_and_folder(paused, tmp_path):
    ups = [up(paused, tmp_path, n) for n in ("a.vasp", "b.vasp")]
    r = post(paused, body([u["id"] for u in ups], label="nice name", folder="fold",
                          location=str(tmp_path / "L")))
    assert r.status_code == 201, r.text
    jobs = r.json()["jobs"]
    assert [j["label"] for j in jobs] == ["nice name-00", "nice name-01"]
    assert [j["folder"] for j in jobs] == ["fold-00", "fold-01"]


def test_batch_101_width_3(paused, tmp_path):
    loc = tmp_path / "w3"
    u = up(paused, tmp_path, "one", atoms=bulk("Cu", "fcc", a=3.6))
    r = post(paused, body([u["id"]] * 101, label="w"
                          , location=str(loc)))
    assert r.status_code == 201, r.text
    jobs = r.json()["jobs"]
    assert len(jobs) == 101
    assert [j["folder"] for j in jobs] == [f"w-{k:03d}" for k in range(101)]
    assert [j["label"] for j in jobs] == [f"w-{k:03d}" for k in range(101)]
    assert sorted(p.name for p in loc.iterdir()) == [f"w-{k:03d}" for k in range(101)]


def test_batch_100_still_width_2(paused, tmp_path):
    u = up(paused, tmp_path, "one")
    r = post(paused, body([u["id"]] * 100, folder="w", location=str(tmp_path / "w2")))
    assert r.status_code == 201, r.text
    folders = [j["folder"] for j in r.json()["jobs"]]
    assert folders[0] == "w-00" and folders[-1] == "w-99"


def test_batch_long_base_truncated_to_fit(paused, tmp_path):
    ups = [up(paused, tmp_path, n) for n in ("a.vasp", "b.vasp")]
    r = post(paused, body([u["id"] for u in ups], folder="x" * 150, location=str(tmp_path / "L")))
    assert r.status_code == 201, r.text
    folders = [j["folder"] for j in r.json()["jobs"]]
    assert all(len(f) <= 100 for f in folders), folders
    assert folders[0].endswith("-00") and folders[1].endswith("-01")
    assert folders[0][:-3] == "x" * (len(folders[0]) - 3)


def test_neb_one_job_no_suffix(paused, tmp_path):
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    b = a.copy()
    b.positions[0] += 0.1
    ui, uf = up(paused, tmp_path, "ini", a), up(paused, tmp_path, "fin", b)
    r = post(paused, {"calc_type": "neb", "calculator": EMT_CPU, "params": {},
                      "initial_id": ui["id"], "final_id": uf["id"], "label": "hop ts",
                      "location": str(tmp_path / "N")})
    assert r.status_code == 201, r.text
    j = r.json()["jobs"][0]
    assert j["label"] == "hop ts" and j["folder"] == "hop-ts"
    assert (tmp_path / "N" / "hop-ts").is_dir()


# ---------------------------------------------------------------------------
# slug rule (via /api/paths/check and via submission)
# ---------------------------------------------------------------------------

SLUG_CASES = [
    ("abc", "abc"),
    ("a b/c", "a-b-c"),
    ("a   b", "a-b"),
    ("a--b", "a-b"),
    ("a.b_c+d-e", "a.b_c+d-e"),
    ("..hidden", "hidden"),
    ("--x", "x"),
    (".-.x", "x"),
    ("ab-", "ab-"),
    ("café", "caf-"),
    ("ch3 -> ch2", "ch3-ch2"),
    ("y" * 150, "y" * 100),
]


@pytest.mark.parametrize("raw,want", SLUG_CASES)
def test_slug_rule_paths_check(paused, tmp_path, raw, want):
    res = check(paused, str(tmp_path), [raw])
    f = res["folders"][0]
    assert f["input"] == raw
    assert f["folder"] == want
    assert Path(f["path"]) == (tmp_path / want).resolve()
    assert f["exists"] is False


@pytest.mark.parametrize("raw,want", SLUG_CASES[:6])
def test_slug_rule_on_submission(paused, tmp_path, raw, want):
    u = up(paused, tmp_path, "POSCAR")
    r = post(paused, body([u["id"]], folder=raw, location=str(tmp_path / "S")))
    assert r.status_code == 201, r.text
    assert r.json()["jobs"][0]["folder"] == want
    assert (tmp_path / "S" / want).is_dir()


@pytest.mark.parametrize("bad", ["_uploads", "_jobs", "///", "...", "---"])
def test_unusable_user_folder_rejected(paused, tmp_path, bad):
    u = up(paused, tmp_path, "POSCAR")
    before = snapshot(tmp_path / "S")
    r = post(paused, body([u["id"]], folder=bad, location=str(tmp_path / "S")))
    ok, dets = has(details(r), "folder", "folder_invalid")
    assert ok, dets
    assert snapshot(tmp_path / "S") <= before | {str(tmp_path / "S")}
    assert not list((tmp_path / "S").glob("*")) if (tmp_path / "S").exists() else True


@pytest.mark.parametrize("label", ["_uploads", "_jobs", "///"])
def test_unusable_label_derived_folder_falls_back_to_job_id(paused, runs_dir, tmp_path, label):
    u = up(paused, tmp_path, "POSCAR")
    r = post(paused, body([u["id"]], label=label))
    assert r.status_code == 201, r.text
    j = r.json()["jobs"][0]
    assert j["label"] == label
    assert j["folder"] == j["job_id"]


def test_reserved_folders_in_default_location(paused, runs_dir, tmp_path):
    u = up(paused, tmp_path, "POSCAR")
    for bad in ("_uploads", "_jobs"):
        r = post(paused, body([u["id"]], folder=bad))
        ok, dets = has(details(r), "folder", "folder_invalid")
        assert ok, dets


def test_paths_check_unusable_folder_entry(paused, tmp_path):
    res = check(paused, str(tmp_path), ["_uploads", "///", "fine"])
    f0, f1, f2 = res["folders"]
    assert f0["error"] and f0["path"] is None
    assert f1["error"] and f1["path"] is None
    assert f2["error"] is None and f2["path"]


# ---------------------------------------------------------------------------
# never overwrite, duplicates, all-or-nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["empty_dir", "file", "nonempty_dir"])
def test_never_overwrite_single(paused, tmp_path, kind):
    loc = tmp_path / "L"
    loc.mkdir()
    target = loc / "taken"
    if kind == "empty_dir":
        target.mkdir()
    elif kind == "file":
        target.write_text("precious")
    else:
        target.mkdir()
        (target / "OUTCAR").write_text("precious")
    before = snapshot(loc)
    u = up(paused, tmp_path, "POSCAR")
    r = post(paused, body([u["id"]], folder="taken", location=str(loc)))
    ok, dets = has(details(r), "folder", "folder_exists")
    assert ok, dets
    assert snapshot(loc) == before
    if kind == "file":
        assert target.read_text() == "precious"
    if kind == "nonempty_dir":
        assert (target / "OUTCAR").read_text() == "precious"
    assert paused.get("/api/jobs").json()["jobs"] == []


def test_never_overwrite_label_derived(paused, runs_dir, tmp_path):
    (runs_dir / "my-run").mkdir()
    u = up(paused, tmp_path, "POSCAR")
    r = post(paused, body([u["id"]], label="my run"))
    ok, dets = has(details(r), "folder", "folder_exists")
    assert ok, dets


def test_never_overwrite_same_name_twice(paused, tmp_path):
    u = up(paused, tmp_path, "POSCAR")
    b = body([u["id"]], label="again", location=str(tmp_path / "L"))
    assert post(paused, b).status_code == 201
    ok, dets = has(details(post(paused, b)), "folder", "folder_exists")
    assert ok, dets
    assert len(paused.get("/api/jobs").json()["jobs"]) == 1


def test_never_overwrite_batch_all_or_nothing(paused, runs_dir, tmp_path):
    loc = tmp_path / "L"
    loc.mkdir()
    (loc / "nm-02").mkdir()
    before = snapshot(loc) | snapshot(runs_dir / "_jobs")
    ups = [up(paused, tmp_path, f"s{i}") for i in range(4)]
    r = post(paused, body([u["id"] for u in ups], label="nm", location=str(loc)))
    dets = details(r)
    ok, _ = has(dets, "structure_ids[2].folder", "folder_exists")
    assert ok, dets
    assert snapshot(loc) | snapshot(runs_dir / "_jobs") == before
    assert paused.get("/api/jobs").json()["jobs"] == []


def test_all_or_nothing_param_error_creates_no_folders(paused, runs_dir, tmp_path):
    loc = tmp_path / "L"
    loc.mkdir()
    ups = [up(paused, tmp_path, f"s{i}") for i in range(3)]
    b = body([u["id"] for u in ups], calc_type="optimize", label="aon", location=str(loc),
             params={"fmax": -1})
    r = post(paused, b)
    assert details(r)
    assert list(loc.iterdir()) == []
    assert not (runs_dir / "_jobs").exists() or list((runs_dir / "_jobs").iterdir()) == []


def test_all_or_nothing_new_location_not_created_on_error(paused, tmp_path):
    """No folders created on any error: a missing location is not created either."""
    loc = tmp_path / "fresh" / "loc"
    u = up(paused, tmp_path, "POSCAR")
    r = post(paused, body([u["id"]], calc_type="optimize", location=str(loc),
                          params={"fmax": -1}))
    assert details(r)
    assert not (tmp_path / "fresh").exists()


def test_duplicate_folder_within_submission_unreachable(paused, tmp_path):
    """Suffixes k are unique per batch, so a duplicate cannot arise through the API;
    the check is a defensive one. Same structure id twice still gets distinct folders."""
    u = up(paused, tmp_path, "POSCAR")
    r = post(paused, body([u["id"], u["id"]], folder="ab-", location=str(tmp_path / "D")))
    assert r.status_code == 201, r.text
    folders = [j["folder"] for j in r.json()["jobs"]]
    assert len(set(folders)) == 2, folders


# ---------------------------------------------------------------------------
# location validation
# ---------------------------------------------------------------------------

def _loc_error(r):
    dets = details(r)
    assert any(d["field"] == "location" and d.get("code") in
               ("location_invalid", "location_not_writable") for d in dets), dets
    return {d.get("code") for d in dets if d["field"] == "location"}


def test_relative_location_invalid(paused, tmp_path):
    u = up(paused, tmp_path, "POSCAR")
    cwd_before = set(os.listdir("."))
    codes = _loc_error(post(paused, body([u["id"]], location="relative/dir")))
    assert codes == {"location_invalid"}
    assert set(os.listdir(".")) == cwd_before


def test_location_is_a_file_invalid(paused, tmp_path):
    f = tmp_path / "afile"
    f.write_text("x")
    u = up(paused, tmp_path, "POSCAR")
    _loc_error(post(paused, body([u["id"]], location=str(f))))
    assert f.read_text() == "x"


def test_location_inside_reserved_dirs_invalid(paused, runs_dir, tmp_path):
    u = up(paused, tmp_path, "POSCAR")
    for loc in (runs_dir / "_uploads", runs_dir / "_uploads" / "sub", runs_dir / "_jobs"):
        codes = _loc_error(post(paused, body([u["id"]], location=str(loc))))
        assert codes == {"location_invalid"}, (loc, codes)
    assert not (runs_dir / "_uploads" / "sub").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_location_not_writable(paused, tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        u = up(paused, tmp_path, "POSCAR")
        codes = _loc_error(post(paused, body([u["id"]], location=str(ro))))
        assert codes == {"location_not_writable"}
        # missing location below a read-only parent cannot be created
        _loc_error(post(paused, body([u["id"]], location=str(ro / "sub"))))
        assert list(ro.iterdir()) == []
        res = check(paused, str(ro), ["x"])
        assert res["exists"] is True and res["writable"] is False
    finally:
        ro.chmod(0o700)


# ---------------------------------------------------------------------------
# /api/paths/check
# ---------------------------------------------------------------------------

def test_paths_check_requires_header(paused, tmp_path):
    r = paused.post("/api/paths/check", json={"location": str(tmp_path), "folders": []})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"


def test_paths_check_never_creates(paused, runs_dir, tmp_path):
    loc = tmp_path / "not" / "yet"
    before = snapshot(tmp_path)
    res = check(paused, str(loc), ["a b", "c"])
    assert Path(res["location"]) == loc.resolve()
    assert res["exists"] is False
    assert res["writable"] is None
    assert res["error"] is None
    assert [f["folder"] for f in res["folders"]] == ["a-b", "c"]
    assert all(f["exists"] is False for f in res["folders"])
    assert Path(res["folders"][0]["path"]) == (loc / "a-b").resolve()
    assert snapshot(tmp_path) == before


def test_paths_check_existing(paused, runs_dir, tmp_path):
    (tmp_path / "taken").mkdir()
    (tmp_path / "file").write_text("x")
    res = check(paused, str(tmp_path), ["taken", "file", "free"])
    assert res["exists"] is True and res["writable"] is True and res["error"] is None
    assert [f["exists"] for f in res["folders"]] == [True, True, False]


def test_paths_check_default_location_and_errors(paused, runs_dir, tmp_path):
    res = check(paused, None, ["x"])
    assert Path(res["location"]) == runs_dir.resolve()
    assert Path(paused.get("/api/health").json()["runs_dir"]).resolve() == runs_dir.resolve()
    res = check(paused, "relative/path", ["x"])
    assert res["error"]
    assert not Path("relative").exists()
    f = tmp_path / "f"
    f.write_text("x")
    assert check(paused, str(f), [])["error"]


# ---------------------------------------------------------------------------
# registry, restart recovery
# ---------------------------------------------------------------------------

def test_restart_lists_jobs_outside_runs_dir(make_app, runs_dir, tmp_path):
    loc = tmp_path / "outside"
    with tc(make_app()) as c:
        u = up(c, tmp_path, "POSCAR")
        r = post(c, body([u["id"], u["id"]], label="keep", location=str(loc)))
        assert r.status_code == 201, r.text
        jobs = r.json()["jobs"]
        for j in jobs:
            assert wait(c, j["job_id"])["status"] == "done"
        # one in the default location too (old + new paths side by side)
        r = post(c, body([u["id"]]))
        jdef = r.json()["jobs"][0]["job_id"]
        wait(c, jdef)
    for j in jobs:
        idx = json.loads((runs_dir / "_jobs" / f"{j['job_id']}.json").read_text())
        assert Path(idx["run_dir"]) == Path(j["run_dir"])
    with tc(make_app()) as c:
        listed = {j["job_id"]: j for j in c.get("/api/jobs").json()["jobs"]}
        for j in jobs:
            assert j["job_id"] in listed
            assert listed[j["job_id"]]["status"] == "done"
            assert listed[j["job_id"]]["folder"] == j["folder"]
            assert listed[j["job_id"]]["run_dir"] == j["run_dir"]
            d = c.get(f"/api/jobs/{j['job_id']}").json()
            assert d["result"] is not None
            assert c.get(f"/api/jobs/{j['job_id']}/files/summary.json").status_code == 200
        assert jdef in listed


def test_restart_skips_missing_run_dir_and_loads_old_layout(make_app, runs_dir, tmp_path):
    import shutil
    loc = tmp_path / "outside"
    with tc(make_app()) as c:
        u = up(c, tmp_path, "POSCAR")
        j_out = post(c, body([u["id"]], folder="gone", location=str(loc))).json()["jobs"][0]
        j_old = post(c, body([u["id"]])).json()["jobs"][0]
        j_nostate = post(c, body([u["id"]], folder="nostate", location=str(loc))).json()["jobs"][0]
        for j in (j_out, j_old, j_nostate):
            wait(c, j["job_id"])
    shutil.rmtree(loc / "gone")
    (loc / "nostate" / "state.json").unlink()
    (runs_dir / "_jobs" / f"{j_old['job_id']}.json").unlink()   # old layout: no index entry
    with tc(make_app()) as c:
        assert c.get("/api/health").status_code == 200
        ids = {j["job_id"] for j in c.get("/api/jobs").json()["jobs"]}
        assert j_out["job_id"] not in ids
        assert j_nostate["job_id"] not in ids
        assert j_old["job_id"] in ids


# ---------------------------------------------------------------------------
# zip and file endpoints for runs outside runs-dir
# ---------------------------------------------------------------------------

def test_zip_named_after_folder(live, tmp_path):
    u = up(live, tmp_path, "POSCAR")
    j = post(live, body([u["id"]], label="zip me", location=str(tmp_path / "Z"))).json()["jobs"][0]
    wait(live, j["job_id"])
    r = live.get(f"/api/jobs/{j['job_id']}/zip")
    assert r.status_code == 200
    assert 'filename="zip-me.zip"' in r.headers["content-disposition"]
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert names and all(n.startswith("zip-me/") for n in names), names
    assert "zip-me/job.json" in names


def test_files_refuse_escape_outside_runs_dir(live, tmp_path):
    loc = tmp_path / "Out"
    u = up(live, tmp_path, "POSCAR")
    j = post(live, body([u["id"]], folder="f", location=str(loc))).json()["jobs"][0]
    jid = j["job_id"]
    wait(live, jid)
    secret = loc / "secret.txt"
    secret.write_text("TOPSECRET")
    (loc / "f2").mkdir()
    (loc / "f2" / "x").write_text("TOPSECRET")
    for bad in ("..%2Fsecret.txt", "%2E%2E/secret.txt", "inputs/..%2F..%2Fsecret.txt",
                "..%2Ff2%2Fx", "%2Fetc%2Fpasswd"):
        r = live.get(f"/api/jobs/{jid}/files/{bad}")
        assert r.status_code in (400, 404), (bad, r.status_code)
        assert "TOPSECRET" not in r.text and "root:" not in r.text
    r = live.get(f"/api/jobs/{jid}/files/..%2Fsecret.txt")
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"
    (loc / "f" / "link").symlink_to(secret)
    r = live.get(f"/api/jobs/{jid}/files/link")
    assert r.status_code == 400 and "TOPSECRET" not in r.text
    (loc / "f" / "dlink").symlink_to(loc / "f2", target_is_directory=True)
    r = live.get(f"/api/jobs/{jid}/files/dlink/x")
    assert r.status_code == 400 and "TOPSECRET" not in r.text
    zr = live.get(f"/api/jobs/{jid}/zip")
    assert b"TOPSECRET" not in zr.content
    assert live.get(f"/api/jobs/{jid}/files/summary.json").status_code == 200


# ---------------------------------------------------------------------------
# NEB / AutoNEB events with the real worker (PLAN 15.2)
# ---------------------------------------------------------------------------

def neb_endpoints():
    slab = fcc100("Cu", size=(2, 2, 3), vacuum=6.0)
    add_adsorbate(slab, "Cu", 1.7, "hollow")
    slab.set_constraint(FixAtoms(indices=[a.index for a in slab if a.tag == 3]))
    final = slab.copy()
    final.positions[-1, 0] += bulk("Cu", "fcc").cell.cellpar()[0]
    return slab, final


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    from fastapi.testclient import TestClient
    from mliprun.webapp.server import create_app
    runs = tmp_path_factory.mktemp("runs_rev2_real")
    with TestClient(create_app(runs, cancel_grace_s=3.0, poll_interval_s=0.05)) as c:
        yield c


def _check_neb_events(events, result, autoneb):
    iters = [e for e in events if e["type"] == "neb_iter"]
    assert iters
    for e in iters:
        rel = e["rel_energies"]
        assert len(rel) == e["n_images"] == len(e["energies"])
        assert rel[0] == pytest.approx(0.0, abs=1e-12)
        E0 = e["energies"][0]
        for En, r in zip(e["energies"], rel):
            if En is not None:
                assert r == pytest.approx(En - E0, abs=1e-9)
        vals = [(r if r is not None else -np.inf) for r in rel]
        hi = e["highest_image"]
        assert 0 <= hi < e["n_images"]
        assert vals[hi] == max(vals), (hi, rel)
        # PLAN 6.3: barrier = max(E) - E0
        assert e["barrier"] == pytest.approx(max(vals), abs=1e-9)
        if autoneb:
            assert "sub_band" in e["autoneb"] and len(e["autoneb"]["sub_band"]) == 2
            lo, hi_b = e["autoneb"]["sub_band"]
            assert 0 <= lo < hi_b < e["n_images"]
    last = iters[-1]
    assert len(result["relative_energies_eV"]) == len(result["energies_eV"])
    assert result["relative_energies_eV"][0] == pytest.approx(0.0, abs=1e-12)
    ts = result["ts_image_index"]
    assert result["relative_energies_eV"][ts] == pytest.approx(
        max(result["relative_energies_eV"]), abs=1e-9)
    if not autoneb:
        # the final band is the band of the last iteration
        assert last["n_images"] == len(result["energies_eV"])
        assert last["energies"] == pytest.approx(result["energies_eV"], abs=1e-6)
        assert last["highest_image"] == ts
        assert result["forward_barrier_eV"] == pytest.approx(last["barrier"], abs=1e-6)
    else:
        assert last["n_images"] == len(result["energies_eV"])
        assert last["highest_image"] == ts
    return iters


def test_neb_pair_events_rel_energies_and_highest(real, tmp_path):
    a, b = neb_endpoints()
    ui, uf = up(real, tmp_path, "ini", a), up(real, tmp_path, "fin", b)
    r = post(real, {"calc_type": "neb", "calculator": EMT_CPU,
                    "params": {"n_images": 3, "fmax": 0.1},
                    "initial_id": ui["id"], "final_id": uf["id"], "label": "neb pair"})
    assert r.status_code == 201, r.text
    j = r.json()["jobs"][0]
    assert j["folder"] == "neb-pair"
    d = wait(real, j["job_id"], 600)
    assert d["status"] == "done", d["error"]
    ev = real.get(f"/api/jobs/{j['job_id']}/events", params={"limit": 5000}).json()["events"]
    iters = _check_neb_events(ev, d["result"], autoneb=False)
    assert all(e["n_images"] == 5 for e in iters)


def test_autoneb_events_full_band(real, tmp_path):
    a, b = neb_endpoints()
    ui, uf = up(real, tmp_path, "ini", a), up(real, tmp_path, "fin", b)
    r = post(real, {"calc_type": "autoneb", "calculator": EMT_CPU,
                    "params": {"n_max": 5, "fmax": 0.1},
                    "initial_id": ui["id"], "final_id": uf["id"]})
    assert r.status_code == 201, r.text
    jid = r.json()["jobs"][0]["job_id"]
    d = wait(real, jid, 900)
    assert d["status"] == "done", d["error"]
    ev = real.get(f"/api/jobs/{jid}/events", params={"limit": 5000}).json()["events"]
    iters = _check_neb_events(ev, d["result"], autoneb=True)
    # the band grows towards n_max
    assert iters[-1]["n_images"] == 5
    assert min(e["n_images"] for e in iters) <= iters[-1]["n_images"]


# ---------------------------------------------------------------------------
# Core 15.3: optimizer trajectory closed
# ---------------------------------------------------------------------------

def test_run_optimization_closes_trajectory(tmp_path):
    from ase.calculators.emt import EMT
    from mliprun.core.optimize import run_optimization
    import inspect
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    a.rattle(0.05, seed=2)
    a.calc = EMT()
    sig = inspect.signature(run_optimization).parameters
    kw = {k: v for k, v in dict(optimizer="bfgs", fmax=0.05, max_steps=50,
                                trajectory="opt.traj", logfile="opt.log",
                                output_dir=tmp_path, plot=False).items() if k in sig}
    gc.collect()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        caught = []
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always", ResourceWarning)
            run_optimization(a, **kw)
            gc.collect()
            caught = [w for w in rec if issubclass(w.category, ResourceWarning)]
    assert not caught, [str(w.message) for w in caught]
    assert (tmp_path / "opt.traj").is_file()
    assert len(ase.io.read(str(tmp_path / "opt.traj"), index=":")) >= 1


def test_run_optimization_closes_trajectory_on_exception(tmp_path):
    from ase.calculators.emt import EMT
    from mliprun.core.optimize import run_optimization
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    a.rattle(0.05, seed=2)
    a.calc = EMT()

    def boom(payload):
        if payload["step"] >= 1:
            raise RuntimeError("stop here")

    gc.collect()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always", ResourceWarning)
        with pytest.raises(RuntimeError, match="stop here"):
            run_optimization(a, fmax=0.001, max_steps=50, output_dir=tmp_path,
                             verbose=False, observer=boom)
        gc.collect()
    caught = [w for w in rec if issubclass(w.category, ResourceWarning)]
    assert not caught, [str(w.message) for w in caught]
