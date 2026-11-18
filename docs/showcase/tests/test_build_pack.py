"""Tests for docs/showcase/scripts/build_pack.py (web data pack, no MLIP).

Run with any Python that has ASE and pytest:
    pytest docs/showcase/tests/test_build_pack.py -v

The first five tests use small synthetic fixtures. The last ones read the real
pack in docs/showcase/data/pack/ and skip when it has not been built.
"""
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

SHOWCASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOWCASE / "scripts"))
import build_pack as bp  # noqa: E402

PACK = SHOWCASE / "data" / "pack"
RUNS = SHOWCASE / "data" / "runs"


# --------------------------------------------------------------- fixtures --
def _opt_dir(tmp_path, fmaxes, threshold=0.02, converged_line="True"):
    """Fake optimize run dir: opt.log (ASE BFGS format) + opt_params.txt."""
    d = tmp_path / "run"
    d.mkdir()
    lines = ["      Step     Time          Energy          fmax"]
    for i, f in enumerate(fmaxes):
        lines.append(f"BFGS: {i:>4d} 22:52:55     -192.{215645 + i:06d}        {f:.6f}")
    (d / "opt.log").write_text("\n".join(lines) + "\n")
    (d / "opt_params.txt").write_text(
        "Geometry Optimization Parameters\n"
        f"fmax (eV/Å):       {threshold}\n"
        f"Converged:         {converged_line}\n")
    return d


def _slab_with_o(dz):
    """Pt(111) 2x2x3 slab with one O placed dz above the top-layer mean z."""
    slab = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    top = bp.top_layer_z(slab)
    slab += Atoms("O", positions=[[1.4, 0.8, top + dz]])
    return slab


# ------------------------------------------------------------ brief tests --
def test_not_converged_is_labeled(tmp_path):
    # Log ends at fmax 0.03 > 0.02. opt_params claims True: the log wins and the
    # disagreement itself is flagged.
    d = _opt_dir(tmp_path, [1.2, 0.4, 0.03])
    converged, flags = bp.optimize_converged(d)
    assert converged is False
    assert "not_converged" in flags
    assert "convergence_record_mismatch" in flags


def test_converged_when_last_fmax_below(tmp_path):
    d = _opt_dir(tmp_path, [1.2, 0.4, 0.013])
    converged, flags = bp.optimize_converged(d)
    assert converged is True
    assert flags == []


def test_frames_capped():
    frames = [Atoms("H", positions=[[k, 0, 0]], cell=[5, 5, 5], pbc=True)
              for k in range(1000)]
    out = bp.cap_frames(frames, 300)
    assert len(out) == 300
    assert out[0].positions[0, 0] == 0.0
    assert out[-1].positions[0, 0] == 999.0
    assert bp.cap_frames(frames[:13], 300) == frames[:13]   # short input untouched


def test_schema():
    good = bp.make_record(
        use_case="relax", command="mlip optimize run --fmax 0.02", converged=True,
        flags=[], key_numbers=[bp.kn("final fmax", 0.013, "eV/Å")],
        series={"fmax_eV_per_A": [1.0, 0.013]}, frames="relax.xyz")
    assert set(bp.SCHEMA_KEYS) <= set(good)
    bp.validate_record(good)                      # does not raise
    bad = dict(good, key_numbers=[{"label": "x", "value": 1.0, "unit": ""}])
    with pytest.raises(ValueError, match="unit"):
        bp.validate_record(bad)
    missing = {k: v for k, v in good.items() if k != "series"}
    with pytest.raises(ValueError, match="series"):
        bp.validate_record(missing)
    with pytest.raises(ValueError):
        bp.validate_record(dict(good, key_numbers=[{"label": "x", "value": float("nan"),
                                                     "unit": "eV"}]))


def test_geometry_flag():
    assert "o_subsurface" in bp.geometry_flags(_slab_with_o(-0.8))
    assert "o_subsurface" not in bp.geometry_flags(_slab_with_o(1.2))

    slab = fcc111("Pt", size=(3, 3, 3), vacuum=8.0)
    top = bp.top_layer_z(slab)
    w = molecule("H2O")
    w.positions += [4.0, 2.5, top + 3.0] - w.positions[0]
    intact = slab + w
    assert "water_dissociated" not in bp.geometry_flags(intact)
    assert "o_subsurface" not in bp.geometry_flags(intact)
    broken = intact.copy()
    o = [a.index for a in broken if a.symbol == "O"][0]
    h = [a.index for a in broken if a.symbol == "H"][0]
    vec = broken.positions[h] - broken.positions[o]
    broken.positions[h] = broken.positions[o] + 1.5 * vec / np.linalg.norm(vec)
    assert "water_dissociated" in bp.geometry_flags(broken)


def test_command_is_verbatim(tmp_path):
    # 1. A run record that carries argv: command is exactly that argv.
    argv = ["mlip", "optimize", "run", "--structure", "POSCAR", "--fmax", "0.02"]
    d = tmp_path / "rec"
    d.mkdir()
    (d / "mliprun_run.json").write_text(json.dumps({"argv": argv}))
    assert bp.command_from_record(d) == shlex.join(argv)

    # 2. No argv: the command is the run script's own mlip line, variables
    #    expanded from that script (and the env file it sources).
    env = tmp_path / "cluster.env"
    env.write_text("export REMOTE=/r\nexport UMA_BIN=/env/bin\n")
    script = tmp_path / "run.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f"source {env}\n"
        'OUT="$REMOTE/01"\n'
        "COMMON=(--mlip uma-s-1p2 --uma-task oc20\n"
        "        --fmax 0.02)\n"
        "relax() {\n"
        '    local dir="$1"; shift\n'
        '    "$UMA_BIN/mlip" optimize run --structure "$dir/POSCAR" "${COMMON[@]}" "$@" \\\n'
        '        > "$dir/out.txt" 2>&1\n'
        "}\n")
    cmd = bp.command_from_script(script, "optimize run", env_file=env,
                                 local={"dir": "/r/01/o_fcc"}, extra_args=[])
    assert cmd == ("/env/bin/mlip optimize run --structure /r/01/o_fcc/POSCAR "
                   "--mlip uma-s-1p2 --uma-task oc20 --fmax 0.02")

    # 3. The command is cross-checked against the run record: every parameter
    #    the record marks as user-given must appear with its value.
    record = {"parameters": {"fmax": {"value": 0.02, "source": "user"},
                             "uma_task": {"value": "oc20", "source": "user"},
                             "plot": {"value": False, "source": "default"}}}
    assert bp.command_mismatches(cmd, record) == []
    record["parameters"]["optimizer"] = {"value": "bfgs", "source": "user"}
    assert bp.command_mismatches(cmd, record) == ["--optimizer bfgs"]


# ------------------------------------------------------ real pack (skips) --
USE_CASES = ["relax", "committee", "neb", "md"]


def _load(name):
    p = PACK / f"{name}.json"
    if not p.exists():
        pytest.skip(f"{p} not built")
    return json.loads(p.read_text())


@pytest.mark.parametrize("name", USE_CASES)
def test_real_pack_schema(name):
    rec = _load(name)
    bp.validate_record(rec)
    assert rec["use_case"] == name


@pytest.mark.parametrize("name", ["relax", "committee", "neb", "md"])
def test_real_pack_command_matches_run_record(name):
    rec = _load(name)
    run_json = RUNS / rec["run_dir"] / "mliprun_run.json"
    assert bp.command_mismatches(rec["command"], json.loads(run_json.read_text())) == []


def test_command_provenance_guard():
    record = {"provenance": {"mlip_model": "uma-s-1p2", "uma_task": "oc20",
                             "device_requested": "cuda"}}
    ok = "mlip optimize run --mlip uma-s-1p2 --uma-task oc20 --device cuda"
    assert bp.command_mismatches(ok, record) == []
    bad = "mlip optimize run --mlip mace-mh-1 --uma-task omat --device cpu"
    assert len(bp.command_mismatches(bad, record)) == 3


def _key(rec, start):
    return next(k["value"] for k in rec["key_numbers"] if k["label"].startswith(start))


def test_real_headline_numbers():
    # The five site copies show these numbers; a rebuild must not change them silently.
    neb = _load("neb")
    assert _key(neb, "forward barrier") == pytest.approx(0.5442, abs=5e-5)
    assert _key(neb, "reaction energy") == pytest.approx(0.3681, abs=5e-5)
    assert _key(_load("relax"), "Pt lattice constant a") == pytest.approx(3.9898, abs=5e-5)
    assert _key(_load("committee"), "sigma_max_free / fmax") == pytest.approx(3.677, abs=5e-4)


def test_real_md_is_not_md():
    rec = _load("md")
    assert rec["converged"] is False
    assert rec["frames"] is None
    assert rec["status"] == "pending"
    assert {"md_not_run", "pending_scientist_decision"} <= set(rec["flags"])
    assert rec["series"] == {}
    assert not (PACK / "md.xyz").exists()
    assert all("relaxation" in k["label"] for k in rec["key_numbers"])
    from ase.io import read
    assert set(rec["structures"]) == {"built", "relaxed_oc20"}
    assert "MD start" not in rec["structures"]["built"]["label"]   # run_04_md.sh starts MD from the relaxed bilayer
    assert "MD start" in rec["structures"]["relaxed_oc20"]["label"]
    for entry in rec["structures"].values():
        assert entry["label"]
        f = PACK / entry["file"]
        assert f.exists(), f
        frames = read(f, index=":")
        assert len(frames) == 1, f
        sym = frames[0].get_chemical_symbols()
        assert (sym.count("O"), sym.count("H"), sym.count("Pt")) == (6, 12, 36), f
        assert "Lattice=" in f.read_text().splitlines()[1], f


def test_real_frame_files_capped():
    from ase.io import read
    files = sorted(PACK.glob("*.xyz"))
    if not files:
        pytest.skip("pack not built")
    for f in files:
        frames = read(f, index=":")
        assert 1 <= len(frames) <= 300, f
        assert frames[0].cell.volume > 0 and frames[0].pbc.all(), f
        assert "Lattice=" in f.read_text().splitlines()[1], f
