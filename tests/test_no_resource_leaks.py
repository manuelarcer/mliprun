"""Core runners must close every file they open.

A handle left for the garbage collector surfaces as a ResourceWarning at
an arbitrary later moment -- e.g. inside another test's
``warnings.catch_warnings(record=True)`` block, which then fails
intermittently. Each test here runs one runner on a tiny EMT Cu system,
forces a collection, and asserts no ResourceWarning was emitted, then checks
the trajectory it wrote is complete and readable.
"""
import gc
import warnings
from contextlib import contextmanager

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read

from mliprun.core.md import run_md
from mliprun.core.neb import CustomNEB
from mliprun.core.optimize import run_optimization


@contextmanager
def no_resource_warnings():
    gc.collect()  # don't blame this test for garbage left by earlier ones
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        yield
        gc.collect()
    leaks = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert not leaks, [str(w.message) for w in leaks]


def _cu():
    atoms = bulk("Cu", "fcc", a=3.7) * (2, 1, 1)
    atoms.rattle(0.05, seed=0)
    atoms.calc = EMT()
    return atoms


@pytest.mark.parametrize("verbose", [False, True])
def test_run_optimization_closes_files(tmp_path, verbose):
    steps = []
    with no_resource_warnings():
        run_optimization(_cu(), optimizer="bfgs", fmax=0.05, max_steps=20,
                         output_dir=tmp_path, verbose=verbose,
                         observer=lambda d: steps.append(d["step"]))
    frames = read(tmp_path / "opt.traj", index=":")
    assert len(frames) == len(steps) > 1  # one frame per observed step
    assert frames[-1].get_potential_energy() is not None


def test_run_optimization_closes_files_when_optimizer_raises(tmp_path):
    def boom(data):
        if data["step"] == 2:
            raise RuntimeError("boom")

    with no_resource_warnings():
        with pytest.raises(RuntimeError, match="boom"):
            run_optimization(_cu(), optimizer="fire", fmax=1e-6, max_steps=20,
                             output_dir=tmp_path, verbose=False, observer=boom)
    # Frames written before the failure are intact.
    assert len(read(tmp_path / "opt.traj", index=":")) == 3


def test_run_md_closes_files(tmp_path):
    with no_resource_warnings():
        run_md(_cu(), ensemble="nve", steps=4, log_interval=1, traj_interval=1,
               output_dir=tmp_path, log_path=str(tmp_path / "md.log"))
    assert len(read(tmp_path / "md.traj", index=":")) == 5
    # header + one line per logged step (0..4)
    assert len((tmp_path / "md.log").read_text().splitlines()) == 6


def _neb(tmp_path, monkeypatch):
    initial = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    final = initial.copy()
    pos = final.get_positions()
    pos[0] += np.array([0.3, 0.3, 0.0])
    final.set_positions(pos)
    neb = CustomNEB(initial=initial, final=final, num_images=3,
                    mlip="test", output_dir=tmp_path)
    monkeypatch.setattr(neb, "setup_calculator", lambda: EMT())
    return neb


def test_neb_endpoints_and_band_close_files(tmp_path, monkeypatch):
    neb = _neb(tmp_path, monkeypatch)
    with no_resource_warnings():
        neb.optimize_endpoints(endpoint_fmax=0.05, max_steps=5)
        neb.run_neb(max_steps=2)
    assert len(read(tmp_path / "initial_opt.traj", index=":")) >= 1
    assert len(read(tmp_path / "final_opt.traj", index=":")) >= 1
    assert len(read(tmp_path / "A2B.traj", index=":")) == 5
    assert len(read(tmp_path / "A2B_full.traj", index=":")) % 5 == 0
