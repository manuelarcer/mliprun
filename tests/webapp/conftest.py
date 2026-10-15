"""Shared fixtures for the web app tests (PLAN 12, Phase 2A).

fastapi is imported inside fixtures only, so collecting this directory never
fails when the ``webapp`` extra is missing: tests that need it skip.

Helpers are exposed as fixtures (``H``, ``upload_atoms``, ``wait_for_status``,
``submit``) that return the constant / function: two ``conftest.py`` files
exist (``tests/`` and ``tests/webapp/``), so ``from conftest import ...`` would
be ambiguous.
"""
import sys
import time
from pathlib import Path

import pytest

H = {"X-Mliprun": "1"}
FAKE_WORKER = Path(__file__).resolve().parent / "fake_worker.py"
FAKE_WORKER_CMD = [sys.executable, str(FAKE_WORKER)]


@pytest.fixture
def runs_dir(tmp_path):
    """A fresh runs directory."""
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture
def make_app(runs_dir):
    """Factory ``make_app(**kw)`` -> FastAPI app on ``runs_dir``.

    Defaults for tests: the scripted fake worker, a 2 s cancel grace and a
    fast 0.05 s tail interval. Any ``create_app`` keyword can be overridden.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from mliprun.webapp.server import create_app

    def factory(**kw):
        kw.setdefault("worker_cmd", list(FAKE_WORKER_CMD))
        kw.setdefault("cancel_grace_s", 2.0)
        kw.setdefault("poll_interval_s", 0.05)
        kw.setdefault("allowed_hosts", None)
        return create_app(kw.pop("runs_dir", runs_dir), **kw)

    return factory


@pytest.fixture
def client(make_app):
    """TestClient (context manager: queue runner running), no Host check."""
    from fastapi.testclient import TestClient

    with TestClient(make_app()) as c:
        yield c


def _upload_atoms(client, atoms, filename, fmt=None, tmp_dir=None):
    """Write ``atoms`` with ``ase.io.write`` and POST it; return the UploadInfo."""
    import tempfile

    import ase.io

    d = Path(tmp_dir or tempfile.mkdtemp())
    path = d / filename
    ase.io.write(str(path), atoms, format=fmt)
    data = {"format": fmt} if fmt else {}
    with open(path, "rb") as fh:
        r = client.post("/api/uploads", headers=H, data=data,
                        files=[("files", (filename, fh.read()))])
    assert r.status_code == 200, r.text
    return r.json()["uploads"][0]


def _wait_for_status(client, job_id, statuses, timeout=60):
    """Poll ``GET /api/jobs/<id>`` until its status is in ``statuses``; return JobDetail."""
    if isinstance(statuses, str):
        statuses = (statuses,)
    deadline = time.monotonic() + timeout
    detail = None
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200, r.text
        detail = r.json()
        if detail["status"] in statuses:
            return detail
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach {statuses} in {timeout} s; last: "
                         f"{detail and detail['status']}")


@pytest.fixture
def cu_upload(client, tmp_path):
    """A Cu fcc bulk POSCAR upload (EMT-compatible)."""
    from ase.build import bulk

    return _upload_atoms(client, bulk("Cu", "fcc", a=3.6, cubic=True), "POSCAR", fmt="vasp",
                         tmp_dir=tmp_path)


def _submit(client, upload_ids, label=None, calc_type="optimize", params=None, **extra):
    """POST /api/jobs for single-input calc types; returns the response."""
    body = {"calc_type": calc_type, "calculator": {"name": "emt", "device": "cpu"},
            "params": params or {}, "structure_ids": list(upload_ids)}
    if label is not None:
        body["label"] = label
    body.update(extra)
    return client.post("/api/jobs", headers=H, json=body)


@pytest.fixture(name="H")
def header_fixture():
    """The ``X-Mliprun: 1`` header dict required on POST requests."""
    return dict(H)


@pytest.fixture(name="upload_atoms")
def upload_atoms_fixture(tmp_path):
    """``upload_atoms(client, atoms, filename, fmt=None) -> UploadInfo``."""
    def upload(client, atoms, filename, fmt=None):
        return _upload_atoms(client, atoms, filename, fmt=fmt, tmp_dir=tmp_path)
    return upload


@pytest.fixture(name="wait_for_status")
def wait_for_status_fixture():
    """``wait_for_status(client, job_id, statuses, timeout=60) -> JobDetail``."""
    return _wait_for_status


@pytest.fixture(name="submit")
def submit_fixture():
    """``submit(client, upload_ids, label=None, calc_type="optimize", params=None, **extra)``."""
    return _submit
