"""CLI surface of `mlip singlepoint run`."""
import csv
from pathlib import Path

import pytest
from ase.build import fcc111
from ase.io import write
from typer.testing import CliRunner

from mliprun.cli.commands.singlepoint import app

runner = CliRunner()


@pytest.fixture
def structure(tmp_path):
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    path = tmp_path / "POSCAR"
    write(path, atoms, format="vasp")
    return path


def test_help_lists_the_run_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout


def test_a_run_writes_the_forces_csv_and_the_record(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    assert result.exit_code == 0, result.stdout
    out = structure.parent
    assert (out / "singlepoint_forces.csv").exists()
    assert (out / "mliprun_run.json").exists()
    rows = list(csv.DictReader((out / "singlepoint_forces.csv").open()))
    assert len(rows) == 12


def test_the_echo_names_both_fmax_populations(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    assert "fmax_free" in result.stdout
    assert "fmax_all" in result.stdout


def test_committee_with_an_explicit_mlip_is_rejected(structure, tmp_path):
    committee_file = tmp_path / "committee.yaml"
    committee_file.write_text("members: []\n")
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(committee_file), "--mlip", "mace"])
    assert result.exit_code == 1
    assert "--mlip" in result.stdout
    assert "--committee" in result.stdout


def test_a_typed_auto_mlip_is_still_rejected(structure, tmp_path):
    """Checked by parameter source, not by value: --mlip defaults to 'auto',
    so comparing values would miss a typed one."""
    committee_file = tmp_path / "committee.yaml"
    committee_file.write_text("members: []\n")
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(committee_file), "--mlip", "auto"])
    assert result.exit_code == 1
    assert "--mlip" in result.stdout


def _use_emt(monkeypatch):
    """Attach ASE's EMT wherever the command would attach an MLIP.

    No MLIP package is installed in the unit environment, so the calculator
    setup is the one thing that cannot run for real.
    """
    from ase.calculators.emt import EMT

    def fake_setup(atoms, *args, **kwargs):
        atoms.calc = EMT()
        return atoms

    monkeypatch.setattr(
        "mliprun.cli.commands.singlepoint.setup_calculator", fake_setup)
    monkeypatch.setattr(
        "mliprun.cli.commands.singlepoint.detect_mlip", lambda: "emt")
    monkeypatch.setattr(
        "mliprun.cli.commands.singlepoint.validate_mlip",
        lambda *args, **kwargs: None)
