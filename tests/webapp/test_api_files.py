"""Run-directory files, downloads, path traversal, zip, static files."""
import io
import json
import zipfile

import pytest

H = {"X-Mliprun": "1"}


@pytest.fixture
def done_job(client, cu_upload, submit, wait_for_status):
    job_id = submit(client, [cu_upload["id"]]).json()["jobs"][0]["job_id"]
    wait_for_status(client, job_id, "done")
    return job_id


def test_file_listing(client, done_job, runs_dir):
    files = client.get(f"/api/jobs/{done_job}/files").json()["files"]
    paths = [f["path"] for f in files]
    assert paths == sorted(paths)
    for name in ("events.jsonl", "job.json", "state.json", "summary.json", "worker.log",
                 "inputs/structure/POSCAR", "reproduce.json"):
        assert name in paths
    entry = next(f for f in files if f["path"] == "job.json")
    assert entry["size"] == (runs_dir / done_job / "job.json").stat().st_size
    assert entry["mtime"].endswith("Z")


def test_file_download_inline_and_errors(client, done_job, runs_dir):
    r = client.get(f"/api/jobs/{done_job}/files/job.json")
    assert r.status_code == 200
    assert json.loads(r.content)["job_id"] == done_job
    assert "attachment" in r.headers["content-disposition"]
    r = client.get(f"/api/jobs/{done_job}/files/inputs/structure/POSCAR?inline=1")
    assert r.headers["content-type"] == "text/plain; charset=utf-8"
    assert r.text == (runs_dir / done_job / "inputs/structure/POSCAR").read_text()
    big = runs_dir / done_job / "big.txt"
    big.write_bytes(b"a" * (5 * 1024 * 1024 + 1))
    r = client.get(f"/api/jobs/{done_job}/files/big.txt?inline=1")
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"
    assert client.get(f"/api/jobs/{done_job}/files/big.txt").status_code == 200
    assert client.get(f"/api/jobs/{done_job}/files/nope.txt").status_code == 404
    assert client.get(f"/api/jobs/{done_job}/files/inputs").status_code == 404   # a directory


@pytest.mark.parametrize("path", ["..%2F..%2Fsecret.txt", "%2E%2E/secret.txt",
                                  "inputs/..%2F..%2F..%2Fsecret.txt", "%2Fetc%2Fpasswd"],
                         ids=["dotdot", "encoded_dots", "nested", "absolute"])
def test_path_traversal_is_400(client, done_job, runs_dir, path):
    (runs_dir / "secret.txt").write_text("top secret")
    r = client.get(f"/api/jobs/{done_job}/files/{path}")
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "bad_request"


def test_symlink_out_of_run_dir_is_400(client, done_job, runs_dir):
    (runs_dir / "secret.txt").write_text("top secret")
    (runs_dir / done_job / "link.txt").symlink_to(runs_dir / "secret.txt")
    assert client.get(f"/api/jobs/{done_job}/files/link.txt").status_code == 400


def test_zip_contains_every_file_under_job_prefix(client, done_job, runs_dir):
    r = client.get(f"/api/jobs/{done_job}/zip")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == f'attachment; filename="{done_job}.zip"'
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = sorted(zf.namelist())
    listed = sorted(f"{done_job}/{f['path']}" for f in
                    client.get(f"/api/jobs/{done_job}/files").json()["files"])
    assert names == listed
    assert zf.read(f"{done_job}/job.json") == (runs_dir / done_job / "job.json").read_bytes()
    assert client.get("/api/jobs/20260101-000000-md-abcdef/zip").status_code == 404


def test_unknown_route_uses_error_format(client):
    r = client.get("/api/nothing-here")
    assert r.status_code == 404
    assert r.json() == {"error": {"code": "not_found", "message": "not found: /api/nothing-here",
                                  "details": []}}


def test_static_files(client):
    from mliprun.webapp.server import STATIC_DIR

    r = client.get("/static/../server.py")
    assert r.status_code == 404
    js = sorted((STATIC_DIR / "js").glob("*.js")) if (STATIC_DIR / "js").is_dir() else []
    if js:
        r = client.get(f"/static/js/{js[0].name}")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/javascript")
    r = client.get("/")
    if (STATIC_DIR / "index.html").is_file():
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    else:
        assert r.status_code == 404
