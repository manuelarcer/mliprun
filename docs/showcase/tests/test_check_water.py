"""Tests for docs/showcase/scripts/check_water.py (water geometry checks, no MLIP).

Run with any Python that has ASE and pytest:
    pytest docs/showcase/tests/test_check_water.py -v
"""
import sys
from pathlib import Path

from ase.build import fcc111

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_structures import make_bilayer, make_slab  # noqa: E402
from check_water import check_frame  # noqa: E402

A = 3.99


def _bilayer():
    return make_bilayer(make_slab(A))


def test_built_bilayer_passes():
    res = check_frame(_bilayer(), max_oh=1.3, max_height=4.0)
    assert res["ok"], res
    assert res["n_water"] == 6
    assert res["max_oh"] < 1.1
    assert 3.0 < res["max_height"] < 3.2      # H-down O built at 3.10 A
    orient = sorted(w["orientation"] for w in res["waters"])
    assert orient == ["H-down"] * 3 + ["flat"] * 3


def test_stretched_oh_fails():
    atoms = _bilayer()
    h = [a.index for a in atoms if a.symbol == "H"][0]
    atoms.positions[h, 2] += 1.5
    res = check_frame(atoms, max_oh=1.3, max_height=4.0)
    assert not res["ok"]
    assert res["max_oh"] > 1.3


def test_desorbed_water_fails():
    atoms = _bilayer()
    o = [a.index for a in atoms if a.symbol == "O"][0]
    h_near = sorted((i for i in range(len(atoms)) if atoms[i].symbol == "H"),
                    key=lambda i: atoms.get_distance(o, i, mic=True))[:2]
    for i in [o] + h_near:
        atoms.positions[i, 2] += 3.0
    res = check_frame(atoms, max_oh=1.3, max_height=4.0)
    assert not res["ok"]
    assert res["max_height"] > 4.0
    assert res["max_oh"] < 1.3


def test_clean_slab_has_no_water():
    slab = fcc111("Pt", size=(3, 3, 4), a=A, vacuum=7.5)
    res = check_frame(slab, max_oh=1.3, max_height=4.0)
    assert res["n_water"] == 0
