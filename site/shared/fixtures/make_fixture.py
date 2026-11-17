"""Write a SYNTHETIC multi-frame extended-XYZ fixture for the shared viewer tests.

This is NOT a calculation. It only gives the viewer and capture tests a
realistic-looking input until the real data pack (relax.xyz, neb_band.xyz,
ts_mode.xyz, md.xyz) exists. Every frame carries `synthetic=T` in its comment
line so it cannot be mistaken for a run.

Geometry: Pt(111) (3x3)x4 slab from build_structures.make_slab at
a = 3.989816 A, one O at the fcc hollow. Over N_FRAMES frames the O height
above the top-layer mean falls from 2.0 A to 1.13 A with an exponential-like
approach (as an optimizer would), and the free (top two) Pt layers get a small
random jitter that decays to zero.

Run (needs ASE):
    /Users/juar/venv/fairchem/bin/python site/shared/fixtures/make_fixture.py
"""
import sys
from pathlib import Path

import numpy as np
from ase.io import write

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "docs" / "showcase" / "scripts"))
import build_structures as bs  # noqa: E402  (import only; never modified)

A_PT = 3.989816        # lattice constant used for the fixture slab (A)
N_FRAMES = 40
H_START, H_END = 2.0, 1.13   # O height above the top-layer mean (A)
JITTER = 0.03          # initial Pt jitter amplitude (A), decays to 0
OUT = HERE / "relax_fixture.xyz"


def main() -> None:
    slab = bs.make_slab(A_PT)
    base = bs.add_o(slab, "fcc")
    top = bs._layers(slab)[0]
    top_z = slab.positions[top, 2].mean()
    o_idx = len(base) - 1
    fixed = bs._fixed_indices(base)
    free_pt = [i for i in range(o_idx) if i not in fixed]

    rng = np.random.default_rng(20261117)
    frames = []
    for k in range(N_FRAMES):
        t = k / (N_FRAMES - 1)
        # 1 - exp decay, normalised so frame 0 = H_START and last = H_END.
        s = (1 - np.exp(-4.0 * t)) / (1 - np.exp(-4.0))
        h = H_START + (H_END - H_START) * s
        at = base.copy()
        at.positions[o_idx, 2] = top_z + h
        amp = JITTER * (1 - s)
        at.positions[free_pt] += rng.normal(0.0, amp, size=(len(free_pt), 3))
        at.info = {"step": k, "synthetic": True, "o_height": round(float(h), 4)}
        frames.append(at)
    write(OUT, frames, format="extxyz")
    print(f"wrote {OUT} ({len(frames)} frames, {len(base)} atoms)")


if __name__ == "__main__":
    main()
