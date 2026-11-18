"""Thin an MD trajectory to N evenly spaced frames, always keeping first and last.

Runs on the cluster, next to md.traj (the raw trajectory never leaves it).

Usage:
    python subsample.py md.traj md_300frames.xyz --frames 300
"""
import argparse
import sys

import numpy as np
from ase.io import write
from ase.io.trajectory import Trajectory


def pick_indices(n_total: int, n_frames: int) -> np.ndarray:
    """Evenly spaced indices into n_total frames; first and last always kept."""
    if n_frames >= n_total:
        return np.arange(n_total)
    if n_frames < 2:
        raise ValueError("need at least 2 frames to keep first and last")
    return np.rint(np.linspace(0, n_total - 1, n_frames)).astype(int)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traj")
    p.add_argument("out")
    p.add_argument("--frames", type=int, required=True)
    args = p.parse_args(argv)

    traj = Trajectory(args.traj, "r")
    n_total = len(traj)
    idx = pick_indices(n_total, args.frames)
    frames = [traj[int(i)] for i in idx]
    traj.close()
    write(args.out, frames, format="extxyz")
    print(f"wrote {args.out}: {len(frames)} of {n_total} frames "
          f"(indices {idx[0]}..{idx[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
