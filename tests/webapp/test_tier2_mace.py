"""Tier 2 smoke tests (PLAN 12, Phase 3): calculator ``mace`` (MACE-MP-0 medium, D11).

One short end-to-end job per calc type on a tiny Cu system, device ``cpu``.
Auto-skipped without mace-torch (``mace`` marker, tests/conftest.py); skipped
with a message when the model cannot be built (e.g. no download possible).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.mace, pytest.mark.slow]

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import ase.io  # noqa: E402
from ase.build import add_adsorbate, bulk, fcc100  # noqa: E402

H = {"X-Mliprun": "1"}
MACE_CPU = {"name": "mace", "device": "cpu"}
TERMINAL = ("done", "failed", "cancelled")


@pytest.fixture(scope="module")
def mace_calc():
    pytest.importorskip("mace", reason="mace-torch not importable")
    try:
        from mliprun.cli.utils import build_calculator
        return build_calculator("mace", device="cpu")
    except BaseException as exc:  # typer.Exit, download errors, ...
        pytest.skip(f"MACE-MP-0 medium could not be built (model download?): {exc!r}")


@pytest.fixture(scope="module")
def client(tmp_path_factory, mace_calc):
    from fastapi.testclient import TestClient
    from mliprun.webapp.server import create_app
    runs = tmp_path_factory.mktemp("runs_tier2")
    with TestClient(create_app(runs, cancel_grace_s=5.0, poll_interval_s=0.1)) as c:
        names = [x["name"] for x in c.get("/api/calculators").json()["calculators"]]
        if "mace" not in names:
            pytest.skip(f"server does not list calculator 'mace': {names}")
        yield c


def upload(client, path: Path, fmt="vasp"):
    r = client.post("/api/uploads", headers=H, data={"format": fmt},
                    files=[("files", (path.name, path.read_bytes()))])
    assert r.status_code == 200, r.text
    u = r.json()["uploads"][0]
    assert u["ok"], u
    return u


def wait(client, jid, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        d = client.get(f"/api/jobs/{jid}").json()
        if d["status"] in TERMINAL:
            return d
        time.sleep(0.2)
    raise AssertionError(f"{jid} not finished in {timeout} s")


def check_reproduce(detail, expect="exact"):
    import typer
    from mliprun.cli.main import app
    run_dir = Path(detail["run_dir"])
    rep = json.loads((run_dir / "reproduce.json").read_text())
    assert rep["status"] == expect, rep
    assert detail["reproduce"]["status"] == expect
    assert "--mlip" in rep["argv"] and rep["argv"][rep["argv"].index("--mlip") + 1] == "mace"
    typer.main.get_command(app).make_context("mlip", list(rep["argv"]))
    sh = (run_dir / "reproduce.sh").read_text()
    assert "\nmlip " in sh and "REPLACE_WITH_AN_MLIP" not in sh


def run_single(client, tmp, atoms, calc_type, params):
    p = tmp / "POSCAR"
    ase.io.write(str(p), atoms, format="vasp", direct=True)
    u = upload(client, p)
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": calc_type, "calculator": MACE_CPU, "params": params,
        "structure_ids": [u["id"]]})
    assert r.status_code == 201, r.text
    d = wait(client, r.json()["jobs"][0]["job_id"])
    assert d["status"] == "done", d["error"]
    assert d["calculator"]["name"] == "mace"
    summary = json.loads((Path(d["run_dir"]) / "summary.json").read_text())
    assert summary["status"] == "done"
    return d, ase.io.read(str(p))


def test_t2_singlepoint_matches_direct(client, tmp_path, mace_calc):
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    a.rattle(0.03, seed=2)
    d, back = run_single(client, tmp_path, a, "singlepoint", {})
    back.calc = mace_calc
    assert abs(d["result"]["energy_eV"] - back.get_potential_energy()) < 1e-5
    check_reproduce(d)


def test_t2_optimize_3_steps(client, tmp_path):
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    a.rattle(0.05, seed=1)
    d, _ = run_single(client, tmp_path, a, "optimize", {"max_steps": 3, "fmax": 0.001})
    assert d["result"]["steps"] <= 3
    check_reproduce(d)


def test_t2_freq(client, tmp_path):
    d, _ = run_single(client, tmp_path, bulk("Cu", "fcc", a=3.6), "freq", {})
    assert d["result"]["n_modes"] == 3
    check_reproduce(d)


def test_t2_md_nve(client, tmp_path):
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    a.rattle(0.05, seed=1)
    d, _ = run_single(client, tmp_path, a, "md",
                      {"ensemble": "nve", "steps": 5, "log_interval": 1})
    assert d["result"]["steps"] == 5
    check_reproduce(d)


def _pair(tmp):
    slab = fcc100("Cu", (2, 2, 2), vacuum=5.0)
    add_adsorbate(slab, "Cu", 1.7, "hollow")
    from ase.constraints import FixAtoms
    slab.set_constraint(FixAtoms(indices=[x.index for x in slab if x.tag == 2]))
    fin = slab.copy()
    fin.positions[-1, 0] += bulk("Cu", "fcc").cell.cellpar()[0]
    pi, pf = tmp / "initial.vasp", tmp / "final.vasp"
    ase.io.write(str(pi), slab, format="vasp", direct=True)
    ase.io.write(str(pf), fin, format="vasp", direct=True)
    return pi, pf


@pytest.mark.parametrize("calc_type,params", [
    ("neb", {"n_images": 1, "interp_steps": 5, "neb_max_steps": 3,
             "endpoint_max_steps": 2}),
    ("autoneb", {"n_max": 3, "maxsteps": 3, "endpoint_max_steps": 2}),
], ids=["band_pair", "auto_pair"])
def test_t2_neb(client, tmp_path, calc_type, params):
    pi, pf = _pair(tmp_path)
    ui, uf = upload(client, pi), upload(client, pf)
    r = client.post("/api/jobs", headers=H, json={
        "calc_type": calc_type, "calculator": MACE_CPU, "params": params,
        "initial_id": ui["id"], "final_id": uf["id"]})
    assert r.status_code == 201, r.text
    d = wait(client, r.json()["jobs"][0]["job_id"], timeout=1800)
    assert d["status"] == "done", d["error"]
    assert d["result"]["energies_eV"]
    check_reproduce(d)
