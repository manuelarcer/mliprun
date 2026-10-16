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
        # never read the user's real machines file
        kw.setdefault("machines_file", runs_dir.parent / "machines.json")
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


# --------------------------------------------------------------------------- #
# Remote machines (fake ssh): shared by the test_remote_*.py files
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def real_probe(tmp_path_factory):
    """One real ``probe`` of this interpreter through the fake ssh (cached)."""
    from remote_helpers import FAKE_SSH_CMD

    from mliprun.webapp import remote

    transport = remote.SSHTransport(FAKE_SSH_CMD, ctl_dir=tmp_path_factory.mktemp("ctl"))
    target = remote.RemoteTarget({"host": "localhost", "name": "probe"},
                                 {"python": sys.executable, "setup": ""})
    probe = remote.probe_target(transport, target)
    assert probe["ok"], probe
    return probe


@pytest.fixture
def remote_setup(tmp_path, real_probe):
    """``remote_setup(**machine_overrides) -> dict`` with ids; writes the machines file.

    Keys: ``machines_file``, ``machine_id``, ``env_id``, ``location`` (the
    remote location, an empty tmp dir), ``machine``, ``env``. Call it before
    ``make_app(machines_file=...)``. ``name`` makes several machines possible
    in one file (pass the same ``machines_file`` back as ``store_path``).
    """
    from remote_helpers import FAKE_SSH_CMD  # noqa: F401 - imported for path setup

    from mliprun.webapp.remote import MachineStore

    def factory(name="box", host="box", store_path=None, **overrides):
        path = Path(store_path) if store_path else tmp_path / "machines.json"
        store = MachineStore(path)
        location = tmp_path / f"remote-{name}"
        location.mkdir(exist_ok=True)
        data = {"name": name, "host": host, "default_location": str(location),
                "poll_interval_s": 0.5}
        data.update(overrides)
        machine = store.add_machine(data)
        env = store.add_env(machine["id"], {"python": sys.executable, "name": f"{name}-env"},
                            probe=real_probe)
        return {"machines_file": path, "machine_id": machine["id"], "env_id": env["id"],
                "location": location, "machine": machine, "env": env}

    return factory


@pytest.fixture
def make_client(make_app):
    """``make_client(**create_app_kw)`` -> started TestClient (fake ssh), closed on teardown.

    Defaults: fake ssh, queue runner off. Use ``start_runner=True`` for jobs.
    """
    from fastapi.testclient import TestClient
    from remote_helpers import FAKE_SSH_CMD

    opened = []

    def factory(**kw):
        kw.setdefault("ssh_command", FAKE_SSH_CMD)
        kw.setdefault("start_runner", False)
        kw.setdefault("remote_backoff_base", 0.05)
        c = TestClient(make_app(**kw))
        c.__enter__()
        opened.append(c)
        return c

    yield factory
    for c in reversed(opened):
        c.__exit__(None, None, None)
