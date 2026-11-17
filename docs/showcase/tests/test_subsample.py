"""Tests for docs/showcase/scripts/subsample.py (trajectory thinning, no MLIP).

Run with any Python that has ASE and pytest:
    pytest docs/showcase/tests/test_subsample.py -v
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.trajectory import Trajectory

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from subsample import pick_indices  # noqa: E402


def _synthetic_traj(path: Path, n: int) -> None:
    """n frames of a 3-atom system; frame k has atom 0 at x = k (frame tag)."""
    with Trajectory(str(path), "w") as traj:
        for k in range(n):
            traj.write(Atoms("H3", positions=[[k, 0, 0], [0, 1, 0], [0, 0, 1]],
                             cell=[10, 10, 10], pbc=True))


def test_pick_indices_keeps_first_and_last():
    idx = pick_indices(1000, 300)
    assert len(idx) == 300
    assert idx[0] == 0 and idx[-1] == 999
    assert np.all(np.diff(idx) > 0)          # strictly increasing, no duplicates
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1      # evenly spaced up to rounding


def test_pick_indices_fewer_frames_than_requested():
    assert list(pick_indices(5, 300)) == [0, 1, 2, 3, 4]


def test_cli_1000_frames_to_300(tmp_path):
    traj = tmp_path / "md.traj"
    out = tmp_path / "md_300frames.xyz"
    _synthetic_traj(traj, 1000)
    res = subprocess.run(
        [sys.executable, str(SCRIPTS / "subsample.py"), str(traj), str(out),
         "--frames", "300"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    frames = read(out, index=":")
    assert len(frames) == 300
    assert frames[0].positions[0, 0] == 0.0      # first frame kept
    assert frames[-1].positions[0, 0] == 999.0   # last frame kept
