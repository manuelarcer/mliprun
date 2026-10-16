"""PLAN 15.1: job name (label), location and folder; never-overwrite; job index."""
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest
from ase.build import bulk

H = {"X-Mliprun": "1"}


@pytest.fixture
def idle_client(make_app):
    """A client whose queue runner is off: submitted jobs stay pending."""
    from fastapi.testclient import TestClient

    with TestClient(make_app(start_runner=False)) as c:
        yield c


@pytest.fixture
def cu(idle_client, upload_atoms):
    return upload_atoms(idle_client, bulk("Cu", "fcc", a=3.6, cubic=True), "POSCAR", fmt="vasp")


def _details(r):
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "validation_error"
    return err["details"]


def _nothing_created(client, runs_dir):
    assert client.get("/api/jobs").json()["jobs"] == []
    jobs_dir = runs_dir / "_jobs"
    assert not jobs_dir.exists() or not list(jobs_dir.iterdir())


# --------------------------------------------------------------------------- #
# slug / width helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw, slug", [
    ("ch3-is-proposal", "ch3-is-proposal"),
    ("CH3 is  proposal", "CH3-is-proposal"),
    ("a/b\\c", "a-b-c"),
    ("../../etc", "etc"),
    ("...hidden", "hidden"),
    ("-.-x", "x"),
    ("Ångström run", "ngstr-m-run"),
    ("é", ""),
    ("ok_1+2.v3", "ok_1+2.v3"),
    ("x" * 150, "x" * 100),
    ("", ""),
])
def test_slugify_folder(raw, slug):
    from mliprun.webapp.jobs import slugify_folder

    assert slugify_folder(raw) == slug


def test_suffix_width_and_suffixed_folder():
    from mliprun.webapp.jobs import folder_problem, suffix_width, suffixed_folder

    assert [suffix_width(n) for n in (1, 2, 10, 100, 101, 1001)] == [2, 2, 2, 2, 3, 4]
    assert suffixed_folder("run", 7, 2) == "run-07"
    assert suffixed_folder("run-", 7, 3) == "run-007"       # dash runs collapse
    long = suffixed_folder("y" * 100, 3, 2)
    assert len(long) == 100 and long.endswith("-03")
    assert folder_problem("") and folder_problem("_uploads") and folder_problem("_jobs")
    assert folder_problem("fine") is None


# --------------------------------------------------------------------------- #
# single job
# --------------------------------------------------------------------------- #
def test_single_default_folder_is_job_id(idle_client, cu, submit, runs_dir):
    r = submit(idle_client, [cu["id"]])
    assert r.status_code == 201, r.text
    (job,) = r.json()["jobs"]
    jid = job["job_id"]
    assert job["label"] == "POSCAR" and job["folder"] == jid
    assert job["run_dir"] == str((runs_dir / jid).resolve())
    spec = json.loads((runs_dir / jid / "job.json").read_text())
    assert spec["folder"] == jid and spec["location"] == str(runs_dir.resolve())
    index = json.loads((runs_dir / "_jobs" / f"{jid}.json").read_text())
    assert index == {"job_id": jid, "run_dir": str((runs_dir / jid).resolve())}
    d = idle_client.get(f"/api/jobs/{jid}").json()
    assert d["folder"] == jid and d["run_dir"] == job["run_dir"]


def test_single_custom_name_derives_folder(idle_client, cu, submit, runs_dir):
    job = submit(idle_client, [cu["id"]], label="CH3 is proposal").json()["jobs"][0]
    assert job["label"] == "CH3 is proposal" and job["folder"] == "CH3-is-proposal"
    assert Path(job["run_dir"]) == (runs_dir / "CH3-is-proposal").resolve()
    assert (runs_dir / "CH3-is-proposal" / "job.json").is_file()


def test_single_custom_folder_and_location(idle_client, cu, submit, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    r = submit(idle_client, [cu["id"]], label="name", folder="my run/v1",
               location="~/calcs/ch3")
    assert r.status_code == 201, r.text
    job = r.json()["jobs"][0]
    loc = (tmp_path / "home" / "calcs" / "ch3").resolve()
    assert job["folder"] == "my-run-v1" and job["label"] == "name"
    assert job["run_dir"] == str(loc / "my-run-v1")
    spec = json.loads((loc / "my-run-v1" / "job.json").read_text())
    assert spec["folder"] == "my-run-v1" and spec["location"] == str(loc)
    assert (loc / "my-run-v1" / "inputs/structure/POSCAR").is_file()


def test_unusable_label_slug_falls_back_to_job_id(idle_client, cu, submit):
    job = submit(idle_client, [cu["id"]], label="é").json()["jobs"][0]
    assert job["label"] == "é" and job["folder"] == job["job_id"]


def test_given_folder_that_slugs_to_nothing_is_422(idle_client, cu, submit, runs_dir):
    for bad in ("...", "_uploads", "_jobs", "é"):
        (d,) = _details(submit(idle_client, [cu["id"]], folder=bad))
        assert d["field"] == "folder" and d["code"] == "folder_invalid"
    _nothing_created(idle_client, runs_dir)


def test_pair_job_has_no_suffix(idle_client, upload_atoms, runs_dir):
    from ase.build import fcc100

    slab = fcc100("Cu", (2, 2, 3), vacuum=5.0)
    a = upload_atoms(idle_client, slab, "ini.vasp", fmt="vasp")
    b = upload_atoms(idle_client, slab, "fin.vasp", fmt="vasp")
    r = idle_client.post("/api/jobs", headers=H, json={
        "calc_type": "neb", "calculator": {"name": "emt", "device": "cpu"}, "params": {},
        "initial_id": a["id"], "final_id": b["id"], "label": "hop A"})
    assert r.status_code == 201, r.text
    job = r.json()["jobs"][0]
    assert job["label"] == "hop A" and job["folder"] == "hop-A"


# --------------------------------------------------------------------------- #
# multi-structure submissions
# --------------------------------------------------------------------------- #
def test_batch_of_ten_common_name(idle_client, cu, submit, runs_dir):
    r = submit(idle_client, [cu["id"]] * 10, label="ch3-is-proposal")
    assert r.status_code == 201, r.text
    jobs = r.json()["jobs"]
    expected = [f"ch3-is-proposal-{k:02d}" for k in range(10)]
    assert [j["label"] for j in jobs] == expected
    assert [j["folder"] for j in jobs] == expected
    for j in jobs:
        assert Path(j["run_dir"]) == (runs_dir / j["folder"]).resolve()
        assert json.loads((Path(j["run_dir"]) / "job.json").read_text())["label"] == j["label"]


def test_batch_width_three_for_101(idle_client, cu, submit):
    r = submit(idle_client, [cu["id"]] * 101, label="big")
    assert r.status_code == 201, r.text
    folders = [j["folder"] for j in r.json()["jobs"]]
    assert folders[0] == "big-000" and folders[100] == "big-100" and len(set(folders)) == 101


def test_batch_with_folder_base_and_no_label(idle_client, cu, submit):
    jobs = submit(idle_client, [cu["id"]] * 2, folder="base dir").json()["jobs"]
    assert [j["folder"] for j in jobs] == ["base-dir-00", "base-dir-01"]
    assert [j["label"] for j in jobs] == ["POSCAR", "POSCAR"]


def test_batch_without_name_or_folder_uses_job_ids(idle_client, cu, submit):
    jobs = submit(idle_client, [cu["id"]] * 3).json()["jobs"]
    assert [j["folder"] for j in jobs] == [j["job_id"] for j in jobs]
    assert [j["label"] for j in jobs] == ["POSCAR"] * 3


# --------------------------------------------------------------------------- #
# never overwrite, duplicates, all-or-nothing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["empty_dir", "file", "broken_symlink"])
def test_existing_folder_is_rejected(idle_client, cu, submit, runs_dir, tmp_path, kind):
    loc = tmp_path / "loc"
    loc.mkdir()
    target = loc / "taken"
    if kind == "empty_dir":
        target.mkdir()
    elif kind == "file":
        target.write_text("keep me")
    else:
        target.symlink_to(tmp_path / "nowhere")
    (d,) = _details(submit(idle_client, [cu["id"]], folder="taken", location=str(loc)))
    assert d["field"] == "folder" and d["code"] == "folder_exists"
    _nothing_created(idle_client, runs_dir)
    if kind == "file":
        assert target.read_text() == "keep me"
    assert sorted(p.name for p in loc.iterdir()) == ["taken"]


def test_existing_folder_in_batch_names_the_structure(idle_client, cu, submit, runs_dir):
    (runs_dir / "run-01").mkdir()
    (d,) = _details(submit(idle_client, [cu["id"]] * 3, label="run"))
    assert d["field"] == "structure_ids[1].folder" and d["code"] == "folder_exists"
    _nothing_created(idle_client, runs_dir)
    assert not (runs_dir / "run-00").exists() and not (runs_dir / "run-02").exists()


def test_resubmitting_the_same_name_is_rejected(idle_client, cu, submit):
    assert submit(idle_client, [cu["id"]], label="once").status_code == 201
    (d,) = _details(submit(idle_client, [cu["id"]], label="once"))
    assert d["code"] == "folder_exists"
    assert len(idle_client.get("/api/jobs").json()["jobs"]) == 1


def test_duplicate_folders_within_submission(tmp_path):
    from mliprun.webapp.errors import ApiError
    from mliprun.webapp.server import plan_folders

    plans = [{}, {}, {}]
    with pytest.raises(ApiError) as ei:
        plan_folders(plans, tmp_path, "same", False, 2)
    assert ei.value.status == 422
    assert [(d["field"], d["code"]) for d in ei.value.details] == \
        [("folder", "folder_duplicate")] * 2          # every repeat after the first
    assert list(tmp_path.iterdir()) == []


def test_race_on_folder_creation_rolls_back(idle_client, cu, submit, runs_dir, monkeypatch):
    """A folder that appears between the check and mkdir -> 422, earlier jobs removed."""
    import mliprun.webapp.server as server

    real = server.plan_folders

    def plan_then_squat(plans, location, base, multi, width):
        real(plans, location, base, multi, width)
        (location / plans[1]["folder"]).mkdir()

    monkeypatch.setattr(server, "plan_folders", plan_then_squat)
    (d,) = _details(submit(idle_client, [cu["id"]] * 2, label="race"))
    assert d["field"] == "structure_ids[1].folder" and d["code"] == "folder_exists"
    _nothing_created(idle_client, runs_dir)
    assert not (runs_dir / "race-00").exists() and (runs_dir / "race-01").is_dir()


# --------------------------------------------------------------------------- #
# location validation
# --------------------------------------------------------------------------- #
def test_invalid_locations(idle_client, cu, submit, runs_dir, tmp_path):
    (tmp_path / "afile").write_text("x")
    cases = [("relative/dir", "location_invalid"),
             (str(tmp_path / "afile"), "location_invalid"),
             (str(tmp_path / "afile" / "sub"), "location_invalid"),
             (str(runs_dir / "_uploads"), "location_invalid"),
             (str(runs_dir / "_jobs" / "x"), "location_invalid")]
    for loc, code in cases:
        details = _details(submit(idle_client, [cu["id"]], location=loc))
        assert [(d["field"], d["code"]) for d in details] == [("location", code)], loc
    _nothing_created(idle_client, runs_dir)
    assert not (runs_dir / "relative").exists()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write")
def test_non_writable_location(idle_client, cu, submit, runs_dir, tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        for loc in (ro, ro / "new" / "deeper"):
            (d,) = _details(submit(idle_client, [cu["id"]], location=str(loc)))
            assert d["field"] == "location" and d["code"] == "location_not_writable"
        assert list(ro.iterdir()) == []
    finally:
        ro.chmod(0o700)
    _nothing_created(idle_client, runs_dir)


def test_location_is_created_with_parents(idle_client, cu, submit, tmp_path):
    loc = tmp_path / "a" / "b" / "c"
    job = submit(idle_client, [cu["id"]], location=str(loc)).json()["jobs"][0]
    assert Path(job["run_dir"]) == loc.resolve() / job["job_id"]


def test_bad_types_for_new_keys(idle_client, cu, submit):
    fields = {d["field"] for d in _details(submit(idle_client, [cu["id"]], location=3,
                                                   folder=["x"]))}
    assert fields == {"location", "folder"}


# --------------------------------------------------------------------------- #
# POST /api/paths/check
# --------------------------------------------------------------------------- #
def test_paths_check(idle_client, runs_dir, tmp_path):
    assert idle_client.post("/api/paths/check", json={"folders": []}).status_code == 403
    (runs_dir / "taken").mkdir()
    r = idle_client.post("/api/paths/check", headers=H,
                         json={"location": None, "folders": ["taken", "new one", "..."]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["location"] == str(runs_dir.resolve())
    assert out["exists"] is True and out["writable"] is True and out["error"] is None
    f0, f1, f2 = out["folders"]
    assert f0 == {"input": "taken", "folder": "taken", "exists": True, "error": None,
                  "path": str(runs_dir.resolve() / "taken")}
    assert f1["folder"] == "new-one" and f1["exists"] is False
    assert f1["path"] == str(runs_dir.resolve() / "new-one")
    assert f2["folder"] == "" and f2["path"] is None and f2["error"]

    # a location that does not exist yet but can be created; nothing is created
    new = tmp_path / "not" / "yet"
    out = idle_client.post("/api/paths/check", headers=H,
                           json={"location": str(new), "folders": ["x"]}).json()
    assert out["location"] == str(new.resolve()) and out["exists"] is False
    assert out["writable"] is None and out["error"] is None
    assert out["folders"][0]["exists"] is False and not (tmp_path / "not").exists()

    # invalid locations
    out = idle_client.post("/api/paths/check", headers=H,
                           json={"location": "rel/dir", "folders": ["x"]}).json()
    assert out["error"] and out["folders"][0]["path"] is None
    (tmp_path / "afile").write_text("x")
    out = idle_client.post("/api/paths/check", headers=H,
                           json={"location": str(tmp_path / "afile"), "folders": []}).json()
    assert out["exists"] is True and out["writable"] is False and out["error"]

    # malformed requests
    r = idle_client.post("/api/paths/check", headers=H, json={"folders": "x"})
    assert r.status_code == 422 and r.json()["error"]["details"][0]["field"] == "folders"
    r = idle_client.post("/api/paths/check", headers=H, json={"location": 1, "bogus": 2})
    assert {d["field"] for d in r.json()["error"]["details"]} == {"location", "bogus"}
    assert idle_client.post("/api/paths/check", headers=H, content=b"[").status_code == 400


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write")
def test_paths_check_not_writable(idle_client, tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        out = idle_client.post("/api/paths/check", headers=H,
                               json={"location": str(ro / "sub"), "folders": []}).json()
        assert out["exists"] is False and out["writable"] is False and out["error"]
        out = idle_client.post("/api/paths/check", headers=H,
                               json={"location": str(ro), "folders": []}).json()
        assert out["exists"] is True and out["writable"] is False and out["error"]
    finally:
        ro.chmod(0o700)


# --------------------------------------------------------------------------- #
# restart recovery from the index (and the old layout)
# --------------------------------------------------------------------------- #
def test_restart_recovers_job_outside_runs_dir(make_app, upload_atoms, submit, runs_dir,
                                               tmp_path):
    from fastapi.testclient import TestClient

    elsewhere = tmp_path / "elsewhere"
    with TestClient(make_app(start_runner=False)) as c:
        up = upload_atoms(c, bulk("Cu", "fcc", a=3.6, cubic=True), "POSCAR", fmt="vasp")
        outside = submit(c, [up["id"]], label="outside", location=str(elsewhere)).json()["jobs"]
        legacy = submit(c, [up["id"]]).json()["jobs"]
        gone = submit(c, [up["id"]], label="gone", location=str(elsewhere)).json()["jobs"]
    (outside_id,), (legacy_id,), (gone_id,) = [[j["job_id"] for j in x]
                                               for x in (outside, legacy, gone)]
    # old layout: no index entry; deleted run dir: index entry left dangling
    (runs_dir / "_jobs" / f"{legacy_id}.json").unlink()
    import shutil
    shutil.rmtree(elsewhere / "gone")
    (runs_dir / "_jobs" / "garbage.json").write_text("{not json")

    with TestClient(make_app(start_runner=False)) as c:
        ids = {j["job_id"] for j in c.get("/api/jobs").json()["jobs"]}
        assert ids == {outside_id, legacy_id}
        d = c.get(f"/api/jobs/{outside_id}").json()
        assert d["status"] == "pending" and d["folder"] == "outside"
        assert d["run_dir"] == str((elsewhere / "outside").resolve())
        assert d["queue_position"] == 1
        assert c.get(f"/api/jobs/{legacy_id}").json()["folder"] == legacy_id
        assert c.get(f"/api/jobs/{gone_id}").status_code == 404
        assert c.get(f"/api/jobs/{outside_id}/files/job.json").status_code == 200


def test_restart_marks_running_job_outside_runs_dir_failed(make_app, upload_atoms, submit,
                                                           runs_dir, tmp_path):
    from fastapi.testclient import TestClient

    with TestClient(make_app(start_runner=False)) as c:
        up = upload_atoms(c, bulk("Cu", "fcc", a=3.6, cubic=True), "POSCAR", fmt="vasp")
        job = submit(c, [up["id"]], label="x", location=str(tmp_path / "far")).json()["jobs"][0]
    run_dir = Path(job["run_dir"])
    st = json.loads((run_dir / "state.json").read_text())
    st["status"] = "running"
    (run_dir / "state.json").write_text(json.dumps(st))
    with TestClient(make_app(start_runner=False)) as c:
        d = c.get(f"/api/jobs/{job['job_id']}").json()
        assert d["status"] == "failed" and "restarted" in d["error"]
    assert json.loads((run_dir / "summary.json").read_text())["status"] == "failed"


# --------------------------------------------------------------------------- #
# zip and files for a custom folder
# --------------------------------------------------------------------------- #
def test_zip_prefix_and_name_follow_folder(idle_client, cu, submit, tmp_path):
    job = submit(idle_client, [cu["id"]], folder="my-run",
                 location=str(tmp_path / "loc")).json()["jobs"][0]
    run_dir = Path(job["run_dir"])
    (tmp_path / "secret.txt").write_text("top secret")
    (run_dir / "link.txt").symlink_to(tmp_path / "secret.txt")
    (run_dir / "linkdir").symlink_to(tmp_path, target_is_directory=True)
    r = idle_client.get(f"/api/jobs/{job['job_id']}/zip")
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="my-run.zip"'
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "my-run/job.json" in names and all(n.startswith("my-run/") for n in names)
    assert "my-run/link.txt" not in names and not any("secret" in n for n in names)
    # files endpoint: symlink out of the run dir refused, traversal refused
    assert idle_client.get(f"/api/jobs/{job['job_id']}/files/link.txt").status_code == 400
    r = idle_client.get(f"/api/jobs/{job['job_id']}/files/..%2Fsecret.txt")
    assert r.status_code == 400 and "top secret" not in r.text
    listed = [f["path"] for f in idle_client.get(f"/api/jobs/{job['job_id']}/files")
              .json()["files"]]
    assert "job.json" in listed and "link.txt" not in listed


# --------------------------------------------------------------------------- #
# real worker in a custom location
# --------------------------------------------------------------------------- #
def test_real_emt_job_in_custom_location(make_app, upload_atoms, submit, wait_for_status,
                                         tmp_path):
    from fastapi.testclient import TestClient

    cmd = [sys.executable, "-m", "mliprun.webapp.worker"]
    loc = tmp_path / "custom" / "place"
    with TestClient(make_app(worker_cmd=cmd)) as c:
        up = upload_atoms(c, bulk("Cu", "fcc", a=3.6, cubic=True), "POSCAR", fmt="vasp")
        r = submit(c, [up["id"]], calc_type="singlepoint", label="Cu sp",
                   location=str(loc))
        assert r.status_code == 201, r.text
        job = r.json()["jobs"][0]
        d = wait_for_status(c, job["job_id"], ("done", "failed"), timeout=120)
        assert d["status"] == "done", (d["error"], (loc / "Cu-sp" / "worker.log").read_text())
    run_dir = loc.resolve() / "Cu-sp"
    assert d["run_dir"] == str(run_dir) and d["folder"] == "Cu-sp"
    assert json.loads((run_dir / "summary.json").read_text())["status"] == "done"
    assert d["result"] is not None
