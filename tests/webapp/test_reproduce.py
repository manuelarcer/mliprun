"""``reproduce.sh`` / ``reproduce.json`` (plan section 8.4).

The generated argv must parse with the real ``mlip`` CLI (resolved down to
the subcommand, so every option is checked); nothing is run.
"""
from __future__ import annotations

import json
import os
import stat

import pytest
import typer.main

from mliprun.cli.main import app
from mliprun.webapp import schema
from mliprun.webapp.reproduce import build_reproduce, write_reproduce

CALC_TYPES = ["singlepoint", "optimize", "freq", "md", "neb", "autoneb"]


def meta(n=4):
    return {"natoms": n, "symbols": ["Cu"] * n, "n_frames": 1,
            "pbc": [True, True, True], "frames_same_symbols": True}


def make_job(calc_type, params=None, calculator=None, band=False, fmt="vasp"):
    kind = schema.get_calc_type(calc_type)["inputs"]
    if kind == "single":
        metas = {"structure": meta()}
        inputs = {"structure": {"upload_id": "0123456789ab", "filename": "POSCAR",
                                "path": "inputs/structure/POSCAR", "format": fmt}}
    elif band:
        metas = {"band": dict(meta(), n_frames=5)}
        inputs = {"band": {"upload_id": "0123456789ab", "filename": "band.traj",
                           "path": "inputs/band/band.traj", "format": "traj"}}
    else:
        metas = {"initial": meta(), "final": meta()}
        inputs = {r: {"upload_id": "0123456789ab", "filename": f"{r}.vasp",
                      "path": f"inputs/{r}/{r}.vasp", "format": fmt}
                  for r in ("initial", "final")}
    v = schema.validate_submission(calc_type, {"name": "emt"}, params or {}, metas)
    calc = dict(v["calculator"])
    if calculator:
        calc.update(calculator)
    return {"schema_version": 1, "job_id": f"20261015-142301-{calc_type}-a1b2c3",
            "calc_type": calc_type, "label": "x", "created_at": "2026-10-15T14:23:01.123Z",
            "calculator": calc, "params": v["params"], "param_sources": v["param_sources"],
            "structure_every": 5, "inputs": inputs}


def parse_with_cli(argv):
    """Resolve ``mlip <argv>`` down to the leaf command; raise on any bad option."""
    cmd = typer.main.get_command(app)
    ctx = cmd.make_context("mlip", list(argv))
    leaf = ctx
    while hasattr(leaf.command, "resolve_command"):
        args = [*leaf._protected_args, *leaf.args] if hasattr(leaf, "_protected_args") \
            else [*leaf.protected_args, *leaf.args]
        name, sub, rest = leaf.command.resolve_command(leaf, args)
        leaf = sub.make_context(name, rest, parent=leaf)
    assert leaf.args == [], leaf.args
    return leaf


MACE = {"name": "mace", "device": "cpu"}
VARIANTS = [
    ("singlepoint", {}), ("singlepoint", {"stress": "on"}), ("singlepoint", {"stress": "off"}),
    ("optimize", {}), ("optimize", {"relax_cell": True, "plot": True, "optimizer": "fire"}),
    ("freq", {}), ("freq", {"indices": "0-1, 3", "expect_fmax": 0.05, "nfree": 4}),
    ("md", {"ensemble": "nve"}), ("md", {"ensemble": "nvt", "thermostat": "berendsen"}),
    ("md", {"ensemble": "npt", "barostat": "npt", "pfactor": 1e6, "barostat_mask": "0,0,1"}),
    ("md", {"ensemble": "npt", "barostat": "berendsen", "compressibility": 0.3}),
    ("neb", {}), ("neb", {"relax_atoms": "0-2", "dyneb": True, "scale_fmax": 0.5,
                          "optimize_endpoints": False, "climb": False}),
    ("autoneb", {}), ("autoneb", {"optimize_endpoints": False, "climb": False}),
]


@pytest.mark.parametrize("calc_type,params", VARIANTS,
                         ids=[f"{c}-{i}" for i, (c, _) in enumerate(VARIANTS)])
def test_argv_parses_with_the_cli(calc_type, params):
    job = make_job(calc_type, params, calculator=MACE)
    built = build_reproduce(job, device_resolved="cpu")
    leaf = parse_with_cli(built["argv"])
    assert leaf.params["mlip"] == "mace"
    p = job["params"]
    if calc_type == "optimize":
        assert leaf.params["fmax"] == p["fmax"]
        assert leaf.params["relax_cell"] == p["relax_cell"]
        assert leaf.params["plot"] == p["plot"]
    if calc_type == "md":
        assert leaf.params["ensemble"] == p["ensemble"]
        assert leaf.params["pfactor"] == p.get("pfactor")
        if "barostat_mask" in p:
            assert leaf.params["barostat_mask"].replace(" ", "") == p["barostat_mask"]
    if calc_type == "freq" and p.get("indices"):
        assert leaf.params["indices"] == "0-1,3"
    if calc_type == "neb":
        assert leaf.params["num_images"] == p["n_images"]
        assert leaf.params["climb"] == p["climb"]
        if p.get("relax_atoms"):
            assert leaf.params["relax_atoms"] == "0,1,2"
    if calc_type == "singlepoint":
        assert leaf.params["stress"] == {"auto": None, "on": True, "off": False}[p["stress"]]
    assert built["status"] in ("exact", "approximate")


def test_status_rules():
    assert build_reproduce(make_job("optimize", calculator=MACE))["status"] == "exact"
    md = build_reproduce(make_job("md", {"ensemble": "nvt"}, calculator=MACE))
    assert md["status"] == "approximate" and "random" in md["notes"][0]
    assert build_reproduce(make_job("md", {"ensemble": "nve"},
                                    calculator=MACE))["status"] == "exact"
    lin = build_reproduce(make_job("neb", {"interpolation": "linear"}, calculator=MACE),
                          device_resolved="cpu")
    assert lin["status"] == "approximate"
    band = build_reproduce(make_job("neb", calculator=MACE, band=True),
                           device_resolved="cpu", band_frames=5)
    assert band["status"] == "approximate"
    assert band["argv"][band["argv"].index("--num-images") + 1] == "3"
    parse_with_cli(band["argv"])
    auto = build_reproduce(make_job("autoneb", calculator={"name": "mace", "device": "cuda"}),
                           device_resolved="cuda")
    assert auto["status"] == "approximate" and "--device" not in auto["argv"]
    xyz = build_reproduce(make_job("neb", calculator=MACE, fmt="extxyz"),
                          device_resolved="cpu")
    assert xyz["status"] == "approximate"
    uma = build_reproduce(make_job("singlepoint",
                                   calculator={"name": "uma-s-1p2", "uma_task": "omat"}))
    assert uma["argv"][uma["argv"].index("--uma-task") + 1] == "omat"
    parse_with_cli(uma["argv"])


def test_neb_device_is_resolved():
    built = build_reproduce(make_job("neb", calculator={"name": "mace", "device": "auto"}),
                            device_resolved="cpu")
    assert built["argv"][built["argv"].index("--device") + 1] == "cpu"


@pytest.mark.parametrize("calc_type", CALC_TYPES)
def test_emt_is_not_reproducible(calc_type, tmp_path):
    job = make_job(calc_type)
    out = write_reproduce(tmp_path, job, device_resolved="cpu")
    assert out["status"] == "no"
    assert "REPLACE_WITH_AN_MLIP" in out["argv"]
    parse_with_cli(out["argv"])
    script = (tmp_path / "reproduce.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash\n")
    assert "# Reproducible: no" in script
    assert "\n# mlip " in script and script.rstrip().endswith("exit 1")
    assert os.stat(tmp_path / "reproduce.sh").st_mode & stat.S_IXUSR
    assert json.loads((tmp_path / "reproduce.json").read_text()) == out


def test_script_runs_the_command_for_an_mlip(tmp_path):
    out = write_reproduce(tmp_path, make_job("optimize", calculator=MACE))
    script = (tmp_path / "reproduce.sh").read_text()
    assert out["status"] == "exact"
    assert "cp inputs/structure/POSCAR reproduce/" in script
    assert "\nmlip optimize run --structure POSCAR --mlip mace --device cpu" in script
