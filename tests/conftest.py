"""Shared fixtures and configuration for mliprun tests."""
import sys

import pytest
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.calculators.emt import EMT
from ase.build import bulk


# ---------------------------------------------------------------------------
# Marker registration helpers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "uma: requires UMA (fairchem-core) to be installed")
    config.addinivalue_line("markers", "mace: requires MACE to be installed")
    config.addinivalue_line("markers", "sevenn: requires SevenNet to be installed")
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


# ---------------------------------------------------------------------------
# MLIP availability detection
# ---------------------------------------------------------------------------

def _is_fairchem_available() -> bool:
    try:
        from importlib.metadata import distribution
        distribution("fairchem-core")
        return True
    except Exception:
        return False


def _is_mace_available() -> bool:
    try:
        from importlib.metadata import distribution
        distribution("mace-torch")
        return True
    except Exception:
        return False


def _is_sevenn_available() -> bool:
    try:
        from importlib.metadata import distribution
        distribution("sevenn")
        return True
    except Exception:
        return False


FAIRCHEM_AVAILABLE = _is_fairchem_available()
MACE_AVAILABLE = _is_mace_available()
SEVENN_AVAILABLE = _is_sevenn_available()

# Auto-skip markers based on availability
def pytest_collection_modifyitems(items):
    """Auto-skip tests based on MLIP availability."""
    for item in items:
        if "uma" in item.keywords and not FAIRCHEM_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="fairchem-core not installed"))
        if "mace" in item.keywords and not MACE_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="mace-torch not installed"))
        if "sevenn" in item.keywords and not SEVENN_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="sevenn not installed"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_atoms():
    """Cu FCC bulk structure with EMT calculator (no MLIP needed)."""
    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def simple_atoms_no_calc():
    """Cu FCC bulk structure without calculator."""
    return bulk("Cu", "fcc", a=3.6)


@pytest.fixture
def initial_final_pair():
    """Two Cu structures for NEB testing with EMT.

    Returns (initial, final) where the final structure has one atom
    slightly displaced to create a simple migration path.
    """
    # 2x2x2 supercell for more realistic NEB
    initial = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    initial.calc = EMT()

    final = initial.copy()
    # Displace one atom to create a simple pathway
    positions = final.get_positions()
    positions[0] += np.array([0.5, 0.5, 0.0])
    final.set_positions(positions)
    final.calc = EMT()

    return initial, final


@pytest.fixture
def mock_calculator():
    """EMT calculator fixture (no MLIP needed)."""
    return EMT()


@pytest.fixture
def tmp_workdir(tmp_path):
    """Temporary directory for output files, returns Path object."""
    return tmp_path


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures" / "structures"


@pytest.fixture
def fake_committee_file(tmp_path):
    """A two-member committee.yaml pointing at the current interpreter.

    Both members are the reserved ``emt`` tag, so the file is usable end to
    end with no MLIP installed. Returns ``(path, CommitteeConfig)``.
    """
    from mliprun.core.committee.config import load_committee

    env = Path(sys.executable).parents[1]
    path = tmp_path / "committee.yaml"
    path.write_text(
        "members:\n"
        f"  - {{env: {env}, mlip: emt, name: member_a}}\n"
        f"  - {{env: {env}, mlip: emt, name: member_b}}\n",
        encoding="utf-8")
    return path, load_committee(path)
