from pathlib import Path

import pytest
from ase.io import read

try:
    from mace.calculators import mace_mp
    mace_available = True
except ImportError:
    mace_available = False

from mliprun.core.neb import CustomNEB

@pytest.mark.skipif(not mace_available, reason="MACE not installed")
def test_neb_run_mace(tmp_path):
    # Absolute fixture paths, and outputs into tmp_path: CustomNEB writes into
    # the current directory by default, which left neb.log, A2B.traj and
    # mliprun_run.json in the repository root after a full-suite run.
    structures = Path(__file__).parent / "fixtures" / "structures"
    initial_path = structures / "fragment_initial.vasp"
    final_path = structures / "fragment_final.vasp"

    initial = read(initial_path, format="vasp")
    final = read(final_path, format="vasp")

    neb = CustomNEB(
        initial=initial,
        final=final,
        num_images=5,
        interp_fmax=0.1,
        interp_steps=100,
        fmax=0.05,
        mlip="mace",
        output_dir=tmp_path,
    )

    assert len(neb.images) == 7  # num_images=5 means 5 intermediate + 1 initial + 1 final = 7 total

    neb.interpolate_idpp()

    for img in neb.images[1:-1]:
        assert img.get_positions().shape == initial.get_positions().shape
        assert not any(p is None for p in img.get_positions().flatten())

    neb.run_neb()

    for img in neb.images:
        energy = img.get_potential_energy()
        assert isinstance(energy, float)
