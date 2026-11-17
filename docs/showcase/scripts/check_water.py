"""Water geometry checks for the showcase bilayer (no MLIP).

For each water O, the two H atoms assigned to it (each H goes to its nearest O,
minimum image) give the O-H lengths. The O height is measured from the mean z
of the top Pt layer. A frame passes when every H sits within `max_oh` of an O,
every O has exactly two H, and every O is at most `max_height` above the top
Pt layer.

Orientation per water, from the z offsets of its two H relative to O:
  H-down  one H at least 0.5 A below O
  H-up    one H at least 0.5 A above O
  flat    otherwise

Usage:
    python check_water.py STRUCTURE [--max-oh 1.3] [--max-height 4.0] [--index -1]

Exit code 0 when the frame passes, 1 otherwise.
"""
import argparse
import sys

import numpy as np
from ase import Atoms
from ase.io import read

ORIENT_DZ = 0.5   # A; |z_H - z_O| above which an H counts as pointing up/down


def top_layer_z(atoms: Atoms, symbol: str = "Pt", tol: float = 0.5) -> float:
    z = atoms.positions[[a.index for a in atoms if a.symbol == symbol], 2]
    return float(z[z > z.max() - tol].mean())


def _orientation(dz_h) -> str:
    if min(dz_h) <= -ORIENT_DZ:
        return "H-down"
    if max(dz_h) >= ORIENT_DZ:
        return "H-up"
    return "flat"


def check_frame(atoms: Atoms, max_oh: float = 1.3, max_height: float = 4.0) -> dict:
    o_idx = [a.index for a in atoms if a.symbol == "O"]
    h_idx = [a.index for a in atoms if a.symbol == "H"]
    res = {"n_water": len(o_idx), "max_oh": 0.0, "max_height": 0.0,
           "waters": [], "ok": True, "problems": []}
    if not o_idx:
        res["ok"] = len(h_idx) == 0
        return res

    top_z = top_layer_z(atoms)
    owner = {o: [] for o in o_idx}
    for h in h_idx:
        d = atoms.get_distances(h, o_idx, mic=True)
        k = int(np.argmin(d))
        owner[o_idx[k]].append((h, float(d[k])))
        if d[k] > max_oh:
            res["problems"].append(f"H{h} is {d[k]:.3f} A from nearest O{o_idx[k]}")
    all_oh = [d for hs in owner.values() for _, d in hs]
    res["max_oh"] = max(all_oh) if all_oh else 0.0

    for o in o_idx:
        height = float(atoms.positions[o, 2] - top_z)
        hs = owner[o]
        dz = [float(atoms.positions[h, 2] - atoms.positions[o, 2]) for h, _ in hs]
        res["waters"].append({
            "O": o, "H": [h for h, _ in hs], "height": height,
            "oh": [d for _, d in hs],
            "orientation": _orientation(dz) if dz else "no-H",
        })
        if len(hs) != 2:
            res["problems"].append(f"O{o} has {len(hs)} H")
        if height > max_height:
            res["problems"].append(f"O{o} is {height:.3f} A above the top Pt layer")
    res["max_height"] = max(w["height"] for w in res["waters"])
    res["ok"] = not res["problems"]
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("structure")
    p.add_argument("--max-oh", type=float, default=1.3)
    p.add_argument("--max-height", type=float, default=4.0)
    p.add_argument("--index", default="-1", help="frame index for trajectories")
    args = p.parse_args(argv)

    atoms = read(args.structure, index=args.index)
    res = check_frame(atoms, args.max_oh, args.max_height)
    print(f"structure: {args.structure}")
    print(f"limits: O-H <= {args.max_oh} A, O height <= {args.max_height} A")
    print(f"waters: {res['n_water']}")
    print(f"{'O':>4} {'height(A)':>10} {'O-H1(A)':>8} {'O-H2(A)':>8}  orientation")
    for w in res["waters"]:
        oh = (w["oh"] + [float("nan")] * 2)[:2]
        print(f"{w['O']:>4} {w['height']:>10.3f} {oh[0]:>8.3f} {oh[1]:>8.3f}  "
              f"{w['orientation']}")
    print(f"max O-H (A): {res['max_oh']:.3f}")
    print(f"max O height (A): {res['max_height']:.3f}")
    for msg in res["problems"]:
        print("PROBLEM:", msg)
    print("water check:", "PASS" if res["ok"] else "FAIL")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
