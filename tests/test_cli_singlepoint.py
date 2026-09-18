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


@pytest.fixture
def fully_fixed_structure(tmp_path):
    """Every atom held, so there is no free atom to be the worst one."""
    from ase.constraints import FixAtoms

    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    atoms.set_constraint(FixAtoms(indices=list(range(len(atoms)))))
    path = tmp_path / "POSCAR"
    write(path, atoms, format="vasp")
    return path


def test_help_lists_the_run_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout


def test_a_fully_fixed_structure_says_so_instead_of_printing_none(
        fully_fixed_structure, monkeypatch):
    """`worst_force_atom_free` is null there, so the echo must not render it
    as "None #None", which reads as a bug in the report rather than as the
    absence it is."""
    _use_emt(monkeypatch)
    result = runner.invoke(
        app, ["run", "--structure", str(fully_fixed_structure)])
    assert result.exit_code == 0, result.stdout
    assert "over 0 free atoms (no free atom)" in result.stdout
    assert "None" not in result.stdout
    # The all-atom population still names a real atom.
    assert "worst: Pt #" in result.stdout


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


def test_every_listed_output_file_actually_exists(structure, monkeypatch):
    """The terminal lists what it wrote. A listing that names a missing file
    is worse than a short listing."""
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    listed = [line.split("📄")[1].strip()
              for line in result.stdout.splitlines() if "📄" in line]
    assert listed
    for path in listed:
        assert Path(path).exists(), f"listed but missing: {path}"


def test_the_selected_task_is_echoed(structure, monkeypatch):
    """The head/task actually used must be visible on the terminal, not just
    recorded in the run record -- it is an explicit decision that must never
    be silently assumed."""
    _use_emt(monkeypatch)
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--mlip", "uma-s-1p2", "--uma-task", "oc20"])
    assert result.exit_code == 0, result.stdout
    assert "UMA task: oc20" in result.stdout


def test_output_dir_redirects_every_output(structure, monkeypatch, tmp_path):
    """The whole point: outputs land elsewhere, so a run does not clobber
    the record of the optimization that produced the structure."""
    _use_emt(monkeypatch)
    out = tmp_path / "sp"
    result = runner.invoke(app, [
        "run", "--structure", str(structure), "--output-dir", str(out)])
    assert result.exit_code == 0, result.stdout
    assert (out / "singlepoint_forces.csv").exists()
    assert (out / "mliprun_run.json").exists()
    # and nothing was written beside the structure
    assert not (structure.parent / "singlepoint_forces.csv").exists()
    assert not (structure.parent / "mliprun_run.json").exists()


def test_output_dir_is_created_when_missing(structure, monkeypatch, tmp_path):
    _use_emt(monkeypatch)
    out = tmp_path / "does" / "not" / "exist"
    result = runner.invoke(app, [
        "run", "--structure", str(structure), "--output-dir", str(out)])
    assert result.exit_code == 0, result.stdout
    assert (out / "singlepoint_forces.csv").exists()


def test_the_default_still_writes_beside_the_structure(structure, monkeypatch):
    """Unchanged behaviour without the flag -- this is a regression guard on
    everyone's existing scripts."""
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    assert result.exit_code == 0, result.stdout
    assert (structure.parent / "singlepoint_forces.csv").exists()


def test_a_prior_record_in_the_structure_directory_survives(
        structure, monkeypatch, tmp_path):
    """The reason this flag exists. A record already beside the structure is
    untouched when output goes elsewhere."""
    _use_emt(monkeypatch)
    prior = structure.parent / "mliprun_run.json"
    prior.write_text('{"schema_version": 5, "command": "optimize", '
                     '"stages": [{"index": 0, "kind": "optimize", '
                     '"status": "converged"}]}')
    before = prior.read_text()
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--output-dir", str(tmp_path / "sp")])
    assert result.exit_code == 0, result.stdout
    assert prior.read_text() == before


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
