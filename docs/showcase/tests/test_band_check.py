"""Tests for docs/showcase/scripts/band_check.py (NEB band explosion check, no MLIP).

Run with any Python that has ASE and pytest:
    pytest docs/showcase/tests/test_band_check.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from band_check import band_check, main  # noqa: E402

CELL = [10.0, 10.0, 10.0]


def _system(mover_x):
    """Three atoms: atom 0 is the mover (travels along x), atoms 1-2 stay put."""
    return Atoms(
        "OPtPt",
        positions=[[mover_x, 5.0, 5.0], [2.0, 2.0, 2.0], [8.0, 8.0, 2.0]],
        cell=CELL,
        pbc=True,
    )


def _straight_band(travel, n_interior):
    """Linear band from mover x=2 to x=2+travel with n_interior interior images."""
    xs = np.linspace(2.0, 2.0 + travel, n_interior + 2)
    return [_system(x) for x in xs]


def test_healthy_band():
    seeded = _straight_band(1.6, 7)
    final_band = [a.copy() for a in seeded]
    for img in final_band[1:-1]:
        img.positions[0] += [0.0, 0.3, 0.0]  # 0.3 A off the seeded path
    res = band_check(seeded[0], seeded[-1], seeded, final_band)
    assert res["exploded"] is False
    assert res["worst"]["atom"] == 0
    assert np.isclose(res["worst"]["drift"], 0.3)
    assert res["jump_warning"] is False


def test_exploded_band():
    seeded = _straight_band(1.6, 7)
    final_band = [a.copy() for a in seeded]
    final_band[4].positions[1] += [0.0, 0.0, 1.2]  # atom 1 never travels
    res = band_check(seeded[0], seeded[-1], seeded, final_band)
    assert res["exploded"] is True
    assert res["worst"]["image"] == 4
    assert res["worst"]["atom"] == 1
    assert np.isclose(res["worst"]["drift"], 1.2)
    assert res["worst"]["allowance"] == 1.0


def test_allowance_scales_with_travel():
    seeded = _straight_band(1.6, 7)
    ok = [a.copy() for a in seeded]
    ok[3].positions[0] += [0.0, 0.0, 2.5]
    res = band_check(seeded[0], seeded[-1], seeded, ok)
    assert res["exploded"] is False
    assert np.isclose(res["worst"]["allowance"], 2.6)

    bad = [a.copy() for a in seeded]
    bad[3].positions[0] += [0.0, 0.0, 2.7]
    res = band_check(seeded[0], seeded[-1], seeded, bad)
    assert res["exploded"] is True
    assert res["worst"]["atom"] == 0
    assert np.isclose(res["worst"]["allowance"], 2.6)


def test_jump_warning():
    seeded = _straight_band(3.3, 2)  # consecutive images 1.1 A apart
    final_band = [a.copy() for a in seeded]
    res = band_check(seeded[0], seeded[-1], seeded, final_band)
    assert np.isclose(res["max_interimage_jump"], 1.1)
    assert res["jump_warning"] is True
    assert res["exploded"] is False


def test_drift_uses_minimum_image():
    seeded = _straight_band(1.6, 7)
    final_band = [a.copy() for a in seeded]
    # Atom 2 wraps across the x boundary: a 0.2 A real move, 9.8 A naive.
    final_band[2].positions[2] += [-9.8, 0.0, 0.0]
    res = band_check(seeded[0], seeded[-1], seeded, final_band)
    assert res["exploded"] is False
    assert res["max_interimage_jump"] < 1.0


def test_travel_uses_minimum_image():
    # Mover goes from x=9.5 to x=0.5 across the boundary: travel 1.0 A, not 9.0.
    initial, final = _system(9.5), _system(0.5)
    mid = _system(10.0)
    seeded = [initial, mid, final]
    final_band = [a.copy() for a in seeded]
    final_band[1].positions[0] += [0.0, 1.9, 0.0]  # allowance 2.0 -> healthy
    res = band_check(initial, final, seeded, final_band)
    assert res["exploded"] is False
    assert np.isclose(res["worst"]["allowance"], 2.0)


def test_cli_reads_seeded_band_from_full_traj(tmp_path, capsys):
    seeded = _straight_band(1.6, 7)
    final_band = [a.copy() for a in seeded]
    final_band[4].positions[1] += [0.0, 0.0, 1.2]
    # A2B_full.traj: seeded band first, then a later iteration; A2B.traj: final band.
    write(tmp_path / "A2B_full.traj", seeded + final_band)
    write(tmp_path / "A2B.traj", final_band)
    assert main([str(tmp_path)]) == 1   # an exploded band fails the script loudly
    out = json.loads((tmp_path / "band_check.json").read_text())
    assert out["exploded"] is True
    assert out["worst"]["atom"] == 1
    assert json.loads(capsys.readouterr().out)["exploded"] is True
