"""Structure builders for the mliprun front-page showcase runs.

Settings are fixed by docs/showcase/CANON.md:
- Pt bulk from the fcc prototype (B1); after the cell relaxation the cell must
  still be fcc within 1e-4 A and 0.01 deg (B2).
- Slab: fcc111("Pt", size=(3, 3, 4), a=<relaxed a>, vacuum=7.5), bottom two
  layers frozen with FixAtoms (S1, S2, S5, S8).
- One O at the fcc or hcp hollow (A1); 6 H2O as an H-down sqrt(3) bilayer (A2).

Command line (all outputs are VASP POSCAR files):
    python build_structures.py bulk OUT
    python build_structures.py check-fcc RELAXED_BULK      # exit 1 if not fcc
    python build_structures.py slab RELAXED_BULK OUT
    python build_structures.py adsorb SLAB {fcc,hcp} OUT
    python build_structures.py bilayer SLAB OUT
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc111
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.io import read, write

O_HEIGHT = 1.2          # initial O height above the top-layer mean z (A)
FLAT_HEIGHT = 2.35      # flat water O height above the top layer (A)
DOWN_HEIGHT = 3.10      # H-down water O height above the top layer (A);
                        # higher than the flat water so the down H stays >= 2 A from Pt
OH = 0.97               # O-H bond length (A)
HOH = 104.5             # H-O-H angle (deg)
N_FIXED = 18            # two bottom layers of the (3x3) slab


# --------------------------------------------------------------------- bulk
def make_bulk() -> Atoms:
    """Primitive fcc Pt cell (B1), ASE's reference a (3.92 A) as the start."""
    return bulk("Pt", "fcc")


def is_fcc(atoms: Atoms, tol_len: float = 1e-4, tol_ang: float = 0.01) -> bool:
    """True when the cell is a primitive fcc cell: equal lengths, 60 deg angles."""
    par = atoms.cell.cellpar()
    lengths, angles = par[:3], par[3:]
    return bool(np.ptp(lengths) <= tol_len
                and np.all(np.abs(angles - 60.0) <= tol_ang))


def lattice_constant(atoms: Atoms) -> float:
    """Cubic lattice constant a of a primitive fcc cell (a = sqrt(2) * |a_i|)."""
    return float(np.sqrt(2.0) * atoms.cell.lengths().mean())


# --------------------------------------------------------------------- slab
def _layers(atoms: Atoms, symbol: str = "Pt", tol: float = 0.5):
    """Indices of `symbol` atoms grouped into z layers, top layer first."""
    idx = sorted((a.index for a in atoms if a.symbol == symbol),
                 key=lambda i: -atoms.positions[i, 2])
    layers, current = [], [idx[0]]
    for i in idx[1:]:
        if atoms.positions[current[-1], 2] - atoms.positions[i, 2] > tol:
            layers.append(current)
            current = [i]
        else:
            current.append(i)
    layers.append(current)
    return layers


def make_slab(a: float) -> Atoms:
    """(3x3) Pt(111), 4 layers, 15 A vacuum, bottom two layers fixed."""
    slab = fcc111("Pt", size=(3, 3, 4), a=a, vacuum=7.5)
    # fcc111 tags layers 1 (top) .. 4 (bottom).
    slab.set_constraint(FixAtoms(mask=slab.get_tags() >= 3))
    return slab


def _fixed_indices(atoms: Atoms) -> set:
    fixed = set()
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            fixed.update(int(i) for i in c.get_indices())
    return fixed


def _lateral_dist(atoms: Atoms, p, q) -> float:
    d = np.asarray(q, float) - np.asarray(p, float)
    d[2] = 0.0
    return float(find_mic(d, atoms.cell, pbc=[True, True, False])[1])


# ---------------------------------------------------------------- adsorbate
def hollow_sites(slab: Atoms):
    """Lateral (x, y) of one hcp and the adjacent fcc hollow.

    hcp hollow = above a second-layer atom (the one nearest the cell centre);
    fcc hollow = above the third-layer atom laterally nearest that hcp site.
    Works on relaxed slabs, which have lost ASE's adsorbate_info.
    """
    layers = _layers(slab)
    centre = 0.5 * (slab.cell[0] + slab.cell[1])
    hcp_atom = min(layers[1],
                   key=lambda i: _lateral_dist(slab, centre, slab.positions[i]))
    hcp_xy = slab.positions[hcp_atom, :2].copy()
    fcc_atom = min(layers[2],
                   key=lambda i: _lateral_dist(slab, slab.positions[hcp_atom],
                                               slab.positions[i]))
    # Unwrap the fcc site next to the hcp site so both sit in the same image.
    d = slab.positions[fcc_atom] - slab.positions[hcp_atom]
    d[2] = 0.0
    d_mic = find_mic(d, slab.cell, pbc=[True, True, False])[0]
    fcc_xy = hcp_xy + d_mic[:2]
    return {"hcp": hcp_xy, "fcc": fcc_xy}


def add_o(slab: Atoms, site: str) -> Atoms:
    """Slab plus one O at the 'fcc' or 'hcp' hollow, O_HEIGHT above the top layer."""
    if site not in ("fcc", "hcp"):
        raise ValueError(f"site must be 'fcc' or 'hcp', not {site!r}")
    xy = hollow_sites(slab)[site]
    top_z = slab.positions[_layers(slab)[0], 2].mean()
    out = slab.copy()
    out += Atoms("O", positions=[[xy[0], xy[1], top_z + O_HEIGHT]])
    out.set_constraint(FixAtoms(indices=sorted(_fixed_indices(slab))))
    return out


# ------------------------------------------------------------------ bilayer
def make_bilayer(slab: Atoms) -> Atoms:
    """Slab plus 6 H2O as an H-down sqrt(3)-type bilayer (2 H2O per sqrt(3) cell).

    The 9 top-layer Pt split into three sqrt(3) sub-lattices by
    (i - j) mod 3 in surface-lattice coordinates. Sub-lattice 0 carries flat
    waters, sub-lattice 1 H-down waters (both on top sites); sub-lattice 2 is
    the empty hexagon centre. Each honeycomb edge carries exactly one H:
    a flat water donates along v0 and v1, an H-down water donates one in-plane
    H along -v2 and points the other H at the surface.
    """
    top = _layers(slab)[0]
    top_z = slab.positions[top, 2].mean()
    a1 = slab.cell[0].copy() / 3.0
    a2 = slab.cell[1].copy() / 3.0
    a1[2] = a2[2] = 0.0
    v2 = a2 - a1                      # v0 = a1, v1 = -a2, v0 + v1 + v2 = 0
    u = -v2 / np.linalg.norm(v2)      # in-plane unit vector along -v2
    perp = np.cross([0.0, 0.0, 1.0], u)
    zhat = np.array([0.0, 0.0, 1.0])

    frac = slab.get_scaled_positions(wrap=False)
    ref = frac[top[0]]
    labels = {}
    for i in top:
        ij = np.rint((frac[i, :2] - ref[:2]) * 3.0).astype(int)
        labels[i] = int((ij[0] - ij[1]) % 3)
    counts = [list(labels.values()).count(k) for k in range(3)]
    if counts != [3, 3, 3]:
        raise RuntimeError(f"unexpected sqrt(3) sub-lattice counts {counts}")

    half = np.radians(HOH / 2.0)
    oxygens, hydrogens = [], []
    for i in top:
        xy = slab.positions[i].copy()
        if labels[i] == 0:            # flat water
            o = xy + zhat * (top_z - xy[2] + FLAT_HEIGHT)
            for s in (+1, -1):
                hydrogens.append(o + OH * (np.cos(half) * u + s * np.sin(half) * perp))
            oxygens.append(o)
        elif labels[i] == 1:          # H-down water
            o = xy + zhat * (top_z - xy[2] + DOWN_HEIGHT)
            theta = np.radians(HOH)
            hydrogens.append(o + OH * u)
            hydrogens.append(o + OH * (np.cos(theta) * u - np.sin(theta) * zhat))
            oxygens.append(o)

    out = slab.copy()
    out += Atoms("O%d" % len(oxygens), positions=oxygens)
    out += Atoms("H%d" % len(hydrogens), positions=hydrogens)
    out.set_constraint(FixAtoms(indices=sorted(_fixed_indices(slab))))
    return out


# ---------------------------------------------------------------------- CLI
def _write(path: str, atoms: Atoms) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write(path, atoms, format="vasp", direct=True)
    print(f"wrote {path} ({len(atoms)} atoms, {atoms.get_chemical_formula()})")


def _read_slab(path: str) -> Atoms:
    slab = read(path, format="vasp")
    n = len(_fixed_indices(slab))
    if n != N_FIXED:
        sys.exit(f"ERROR: {path} has {n} fixed atoms, expected {N_FIXED}")
    return slab


def _report_cell(atoms: Atoms) -> None:
    par = atoms.cell.cellpar()
    print("cell lengths (A): %.6f %.6f %.6f" % tuple(par[:3]))
    print("cell angles (deg): %.6f %.6f %.6f" % tuple(par[3:]))
    print("length spread (A): %.2e (tol 1e-4)" % np.ptp(par[:3]))
    print("max |angle - 60| (deg): %.2e (tol 0.01)" % np.abs(par[3:] - 60).max())
    print("lattice constant a (A): %.6f" % lattice_constant(atoms))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("bulk"); s.add_argument("out")
    s = sub.add_parser("check-fcc"); s.add_argument("relaxed_bulk")
    s = sub.add_parser("slab"); s.add_argument("relaxed_bulk"); s.add_argument("out")
    s = sub.add_parser("adsorb"); s.add_argument("slab")
    s.add_argument("site", choices=["fcc", "hcp"]); s.add_argument("out")
    s = sub.add_parser("bilayer"); s.add_argument("slab"); s.add_argument("out")
    args = p.parse_args(argv)

    if args.cmd == "bulk":
        _write(args.out, make_bulk())
    elif args.cmd == "check-fcc":
        atoms = read(args.relaxed_bulk, format="vasp")
        _report_cell(atoms)
        ok = is_fcc(atoms)
        print("fcc check:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    elif args.cmd == "slab":
        atoms = read(args.relaxed_bulk, format="vasp")
        if not is_fcc(atoms):
            sys.exit("ERROR: relaxed bulk is not fcc; run check-fcc")
        a = lattice_constant(atoms)
        print(f"slab from relaxed a = {a:.6f} A")
        _write(args.out, make_slab(a))
    elif args.cmd == "adsorb":
        _write(args.out, add_o(_read_slab(args.slab), args.site))
    elif args.cmd == "bilayer":
        _write(args.out, make_bilayer(_read_slab(args.slab)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
