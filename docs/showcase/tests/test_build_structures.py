"""Tests for docs/showcase/scripts/build_structures.py (structure building only, no MLIP).

Run with any Python that has ASE and pytest:
    pytest docs/showcase/tests/test_build_structures.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.constraints import FixAtoms
from ase.geometry import find_mic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_structures import add_o, is_fcc, make_bilayer, make_slab  # noqa: E402

A = 3.97


def _layers(atoms, symbol="Pt", tol=0.5):
    """Indices of `symbol` atoms grouped by z, top layer first."""
    idx = [a.index for a in atoms if a.symbol == symbol]
    idx.sort(key=lambda i: -atoms.positions[i, 2])
    layers, current = [], [idx[0]]
    for i in idx[1:]:
        if atoms.positions[current[-1], 2] - atoms.positions[i, 2] > tol:
            layers.append(current)
            current = [i]
        else:
            current.append(i)
    layers.append(current)
    return layers


def _lateral(atoms, p, q):
    """Minimum-image in-plane distance between points p and q."""
    d = np.array(q, float) - np.array(p, float)
    d[2] = 0.0
    _, dist = find_mic(d, atoms.cell, pbc=[True, True, False])
    return float(dist)


def _fixed(atoms):
    fixed = set()
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            fixed.update(int(i) for i in c.get_indices())
    return fixed


def test_is_fcc_true_for_prototype():
    assert is_fcc(bulk("Pt", "fcc", a=A))


def test_is_fcc_false_when_strained():
    atoms = bulk("Pt", "fcc", a=A)
    atoms.set_cell(atoms.cell[:] * [1, 1, 1.001], scale_atoms=True)
    assert not is_fcc(atoms)


def test_slab_geometry():
    slab = make_slab(A)
    assert len(slab) == 36
    assert np.linalg.norm(slab.cell[0]) >= 7.5
    assert np.linalg.norm(slab.cell[1]) >= 7.5
    z = slab.positions[:, 2]
    gap = slab.cell[2, 2] - (z.max() - z.min())
    assert abs(gap - 15.0) <= 0.01
    fixed = _fixed(slab)
    assert len(fixed) == 18
    layers = _layers(slab)
    assert len(layers) == 4
    assert fixed == set(layers[2]) | set(layers[3])


@pytest.mark.parametrize("site", ["fcc", "hcp"])
def test_add_o_sites(site):
    slab = make_slab(A)
    ads = add_o(slab, site)
    assert len(ads) == 37
    o = [a.index for a in ads if a.symbol == "O"]
    assert len(o) == 1
    o_pos = ads.positions[o[0]]
    layers = _layers(ads)
    second = layers[1]
    dmin = min(_lateral(ads, o_pos, ads.positions[i]) for i in second)
    if site == "fcc":
        assert dmin > 1.0  # above no second-layer atom
    else:
        assert dmin < 0.1  # directly above a second-layer atom
    top_z = ads.positions[layers[0], 2].mean()
    assert 1.0 <= o_pos[2] - top_z <= 1.4
    assert _fixed(ads) == _fixed(slab)


def test_bilayer():
    slab = make_slab(A)
    wet = make_bilayer(slab)
    added = wet[len(slab):]
    assert added.get_chemical_symbols().count("O") == 6
    assert added.get_chemical_symbols().count("H") == 12
    assert _fixed(wet) == _fixed(slab)

    # Group each H with its nearest O (minimum image).
    o_idx = [a.index for a in wet if a.symbol == "O"]
    h_idx = [a.index for a in wet if a.symbol == "H"]
    waters = {i: [] for i in o_idx}
    for h in h_idx:
        d = [np.linalg.norm(find_mic(wet.positions[h] - wet.positions[o],
                                     wet.cell, pbc=[True, True, False])[0])
             for o in o_idx]
        waters[o_idx[int(np.argmin(d))]].append(h)
    assert all(len(v) == 2 for v in waters.values())

    n_flat = n_down = 0
    for o, (h1, h2) in waters.items():
        v1 = find_mic(wet.positions[h1] - wet.positions[o], wet.cell,
                      pbc=[True, True, False])[0]
        v2 = find_mic(wet.positions[h2] - wet.positions[o], wet.cell,
                      pbc=[True, True, False])[0]
        assert abs(np.linalg.norm(v1) - 0.97) < 0.02
        normal = np.cross(v1, v2)
        tilt = np.degrees(np.arccos(abs(normal[2]) / np.linalg.norm(normal)))
        if tilt <= 15.0:
            n_flat += 1
        elif min(v1[2], v2[2]) < 0:
            n_down += 1
    # (3x3) cell = three sqrt(3) cells; one flat and one H-down water each.
    assert n_flat == 3
    assert n_down == 3

    # No H-H closer than 1.4 A between different waters.
    owner = {h: o for o, hs in waters.items() for h in hs}
    for i, h in enumerate(h_idx):
        for g in h_idx[i + 1:]:
            if owner[h] == owner[g]:
                continue
            d = np.linalg.norm(find_mic(wet.positions[g] - wet.positions[h],
                                        wet.cell, pbc=[True, True, False])[0])
            assert d >= 1.4
