"""Shared path helpers (REMOTE_PLAN section 4) and tar safety, no server needed."""
import os
import stat
import tarfile

import pytest
from remote_helpers import make_tar

from mliprun.webapp import jobs, paths


def test_jobs_reexports_the_moved_names():
    for name in ("slugify_folder", "folder_problem", "suffix_width", "suffixed_folder",
                 "inspect_location", "MAX_FOLDER_LEN", "RESERVED_FOLDERS"):
        assert getattr(jobs, name) is getattr(paths, name)
    assert paths.MAX_FOLDER_LEN == 100 and paths.RESERVED_FOLDERS == {"_uploads", "_jobs"}
    assert paths.folder_problem("_jobs") is not None
    assert paths.folder_problem("_jobs", reserved=()) is None


def test_inspect_location_rejections(tmp_path):
    default = tmp_path / "d"
    assert paths.inspect_location("a\0b", default)["code"] == "location_invalid"
    assert "NUL" in paths.inspect_location("/a\0b", default)["error"]
    assert "absolute" in paths.inspect_location("rel/x", default)["error"]
    f = tmp_path / "file"
    f.write_text("x")
    out = paths.inspect_location(str(f), default)
    assert out["code"] == "location_invalid" and out["exists"] is True
    out = paths.inspect_location(str(f / "child"), default)       # parent is a file
    assert out["code"] == "location_invalid" and out["writable"] is False
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(ro, os.W_OK):
            pytest.skip("running as a user that ignores directory permissions")
        out = paths.inspect_location(str(ro), default)
        assert out["code"] == "location_not_writable" and out["writable"] is False
        out = paths.inspect_location(str(ro / "new"), default)
        assert out["code"] == "location_not_writable" and "cannot be created" in out["error"]
    finally:
        ro.chmod(stat.S_IRWXU)
    res = paths.inspect_location(str(tmp_path / "_uploads" / "x"), default,
                                 reserved=[tmp_path / "_uploads"])
    assert res["code"] == "location_invalid" and "reserved" in res["error"]
    ok = paths.inspect_location(None, tmp_path)
    assert ok["path"] == tmp_path and ok["exists"] and ok["writable"] and ok["error"] is None


def test_check_paths_reports_existing_folders(tmp_path):
    (tmp_path / "taken").mkdir()
    out = paths.check_paths(str(tmp_path), ["taken", "free", "!!!", "_uploads"], tmp_path,
                            reserved_dirs=None)
    by = {f["input"]: f for f in out["folders"]}
    assert by["taken"]["exists"] is True and by["free"]["exists"] is False
    assert by["!!!"]["folder"] == "" and "empty" in by["!!!"]["error"]
    assert by["_uploads"]["error"] == "'_uploads' is a reserved name"


def test_safe_extract_refuses_to_write_through_a_symlink_in_the_destination(tmp_path):
    dest = tmp_path / "dest"
    outside = tmp_path / "outside"
    dest.mkdir()
    outside.mkdir()
    (dest / "d").symlink_to(outside)
    with pytest.raises(paths.UnsafeArchive):
        paths.safe_extract_tar(make_tar([("d/x.txt", b"pwned")]), dest)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("name", ["", "a\0b", "C:evil"], ids=["empty", "nul", "drive"])
def test_odd_member_names_are_rejected(tmp_path, name):
    info = tarfile.TarInfo(name)
    with pytest.raises(paths.UnsafeArchive):
        paths._check_member(info)


def test_copy_back_rules():
    assert paths.is_local_authoritative("job.json") and paths.is_local_authoritative(
        "inputs/structure/POSCAR")
    assert not paths.is_local_authoritative("worker.log")
    assert paths.copy_back_eligible("a.txt", 100, 100) is True
    assert paths.copy_back_eligible("a.txt", 101, 100) is False
    assert paths.copy_back_eligible("a.txt", 0, 0) is False          # limit 0: nothing copied
    assert paths.copy_back_eligible("job.json", 1, 100) is False     # local is the master


def test_list_files_skips_unreadable_and_outside_links(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "a").write_text("1")
    (tmp_path / "o").write_text("secret")
    (run / "link").symlink_to(tmp_path / "o")
    (run / "dangling").symlink_to(tmp_path / "missing")
    assert [f["path"] for f in paths.list_files(run)] == ["a"]
    assert paths.inside(run.resolve(), run / "a") is True
    assert paths.inside(run.resolve(), run / "link") is False
