"""Band explosion check for a finished NEB (docs/showcase/CANON.md, C7).

mliprun has no built-in explosion check, so this script compares the final band
with the seeded (IDPP-interpolated) band, image by image and atom by atom.

Rule (C7): on every interior image, each atom may drift from its seeded
position by at most

    allowance = floor + factor * travel,   floor = 1.0 A, factor = 1.0,

where travel is that atom's initial-state -> final-state displacement. Any atom
beyond its allowance means the band exploded. All displacements use the
minimum-image convention under the cell's periodic boundaries.

Also reported, as a warning only: the largest single-atom displacement between
consecutive images of the final band (warning above 1.0 A).

Usage (on the cluster, after `mlip neb run`):
    python band_check.py NEB_DIR

NEB_DIR must hold A2B.traj (final band) and A2B_full.traj (all iterations; its
first len(A2B.traj) frames are the seeded band, written before the first
optimizer step). Prints the result as JSON and writes NEB_DIR/band_check.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read

JUMP_WARNING_A = 1.0


def _mic_norms(a, b, atoms):
    """Per-atom minimum-image distance between position arrays a and b."""
    _, lengths = find_mic(b - a, atoms.cell, atoms.pbc)
    return lengths


def band_check(initial, final, seeded, final_band, floor=1.0, factor=1.0):
    """Check a final NEB band against its seeded band.

    Returns a dict with keys ``exploded`` (bool), ``worst`` ({image, atom,
    drift, allowance}: the interior-image atom with the largest
    drift / allowance ratio), ``max_interimage_jump`` (A) and ``jump_warning``
    (bool, jump > 1.0 A).
    """
    if len(seeded) != len(final_band):
        raise ValueError(f"seeded band has {len(seeded)} images, "
                         f"final band has {len(final_band)}")
    if len(final_band) < 3:
        raise ValueError("band needs at least one interior image")
    natoms = len(initial)
    for img in (final, *seeded, *final_band):
        if len(img) != natoms:
            raise ValueError("all images must have the same number of atoms")

    travel = _mic_norms(initial.positions, final.positions, initial)
    allowance = floor + factor * travel

    exploded = False
    worst = None
    worst_ratio = -1.0
    for i in range(1, len(final_band) - 1):
        drift = _mic_norms(seeded[i].positions, final_band[i].positions,
                           final_band[i])
        ratio = drift / allowance
        if np.any(drift > allowance):
            exploded = True
        j = int(np.argmax(ratio))
        if ratio[j] > worst_ratio:
            worst_ratio = float(ratio[j])
            worst = {"image": i, "atom": j, "drift": float(drift[j]),
                     "allowance": float(allowance[j])}

    max_jump = 0.0
    for a, b in zip(final_band[:-1], final_band[1:]):
        max_jump = max(max_jump,
                       float(np.max(_mic_norms(a.positions, b.positions, a))))

    return {
        "exploded": bool(exploded),
        "worst": worst,
        "max_interimage_jump": max_jump,
        "jump_warning": bool(max_jump > JUMP_WARNING_A),
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: band_check.py NEB_DIR", file=sys.stderr)
        return 2
    neb_dir = Path(argv[0])
    final_band = read(neb_dir / "A2B.traj", index=":")
    n = len(final_band)
    full = read(neb_dir / "A2B_full.traj", index=":")
    if len(full) < n:
        print(f"A2B_full.traj has {len(full)} frames, fewer than one band "
              f"({n}); no seeded band available", file=sys.stderr)
        return 1
    seeded = full[:n]
    # The NEB never moves its endpoints, so the seeded endpoints are the
    # (endpoint-optimized) initial and final states.
    result = band_check(seeded[0], seeded[-1], seeded, final_band)
    result.update({
        "floor_A": 1.0,
        "factor": 1.0,
        "n_images_total": n,
        "seeded_source": "A2B_full.traj frames 0..%d" % (n - 1),
        "final_source": "A2B.traj",
    })
    text = json.dumps(result, indent=2)
    (neb_dir / "band_check.json").write_text(text + "\n")
    print(text)
    return 1 if result["exploded"] else 0


if __name__ == "__main__":
    sys.exit(main())
