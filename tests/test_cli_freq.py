"""CLI surface of `mlip freq run`."""
import json
from pathlib import Path

import pytest
from ase.build import molecule
from ase.io import write
from typer.testing import CliRunner

from mliprun.cli.commands.freq import app

runner = CliRunner()


@pytest.fixture
def structure(tmp_path):
    atoms = molecule("N2")
    atoms.center(vacuum=5.0)
    path = tmp_path / "POSCAR"
    write(path, atoms, format="vasp")
    return path


def _use_emt(monkeypatch):
    from ase.calculators.emt import EMT

    def fake_setup(atoms, *args, **kwargs):
        atoms.calc = EMT()
        return atoms

    monkeypatch.setattr("mliprun.cli.commands.freq.setup_calculator",
                        fake_setup)
    monkeypatch.setattr("mliprun.cli.commands.freq.detect_mlip",
                        lambda: "emt")
    monkeypatch.setattr("mliprun.cli.commands.freq.validate_mlip",
                        lambda *a, **k: None)


def test_help_lists_the_run_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout


def test_a_run_writes_every_expected_output(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    assert result.exit_code == 0, result.stdout
    out = structure.parent
    for name in ("freq_frequencies.csv", "freq_summary.txt",
                 "freq_vibrations.json", "mliprun_run.json"):
        assert (out / name).exists(), name
    assert (out / "freq").is_dir()          # ASE's displacement cache


def test_every_listed_output_file_actually_exists(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    listed = [line.split("📄")[1].strip()
              for line in result.stdout.splitlines() if "📄" in line]
    assert listed
    for path in listed:
        assert Path(path).exists(), f"listed but missing: {path}"


@pytest.mark.parametrize("value", ["3", "0", "-1"],
                         ids=["nfree_3", "nfree_0", "nfree_negative"])
def test_an_invalid_nfree_is_rejected_with_the_allowed_values(
        structure, monkeypatch, value):
    _use_emt(monkeypatch)
    result = runner.invoke(app, [
        "run", "--structure", str(structure), "--nfree", value])
    assert result.exit_code == 1
    assert "2" in result.stdout and "4" in result.stdout


def test_an_unknown_direction_is_rejected(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, [
        "run", "--structure", str(structure), "--direction", "sideways"])
    assert result.exit_code == 1
    assert "central" in result.stdout


def test_an_unknown_write_modes_is_rejected(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, [
        "run", "--structure", str(structure), "--write-modes", "some"])
    assert result.exit_code == 1
    assert "imaginary" in result.stdout


def test_a_bad_indices_string_is_rejected_before_any_force_call(
        structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, [
        "run", "--structure", str(structure), "--indices", "0,99"])
    assert result.exit_code == 1
    assert "out of range" in result.stdout
    assert not (structure.parent / "freq_frequencies.csv").exists()


def test_the_echo_reports_fmax_at_the_input_geometry(structure, monkeypatch):
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure)])
    assert "fmax at the input geometry" in result.stdout
    # Both populations, each named -- see C1.
    assert "over the free components" in result.stdout
    assert "over all" in result.stdout


def test_the_stationary_point_warning_names_which_fmax_it_compared(
        structure, monkeypatch):
    """The line above the warning carries two numbers, so "That is above
    the ..." had no unambiguous antecedent. The warning must say which of
    the two it measured.

    `--expect-fmax 1e-9` is below anything N2 at ASE's tabulated geometry
    can reach, so the warning is guaranteed to fire.
    """
    _use_emt(monkeypatch)
    result = runner.invoke(app, ["run", "--structure", str(structure),
                                 "--expect-fmax", "1e-9"])
    assert result.exit_code == 0, result.stdout
    assert "The free-component value is above" in result.stdout
    assert "That is above" not in result.stdout


def test_committee_with_an_explicit_mlip_is_rejected(structure, tmp_path):
    committee_file = tmp_path / "committee.yaml"
    committee_file.write_text("members: []\n")
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(committee_file), "--mlip", "mace"])
    assert result.exit_code == 1
    assert "--committee" in result.stdout


def test_output_dir_redirects_the_outputs(structure, monkeypatch, tmp_path):
    _use_emt(monkeypatch)
    out = tmp_path / "freqrun"
    result = runner.invoke(app, [
        "run", "--structure", str(structure), "--output-dir", str(out)])
    assert result.exit_code == 0, result.stdout
    assert (out / "freq_frequencies.csv").exists()
    assert not (structure.parent / "freq_frequencies.csv").exists()


def test_output_dir_does_not_break_the_fmax_lookup(
        structure, monkeypatch, tmp_path):
    """The lookup reads the STRUCTURE's directory, not the output one. If it
    followed --output-dir it would find nothing and the warning would
    silently stop firing."""
    _use_emt(monkeypatch)
    (structure.parent / "mliprun_run.json").write_text(
        '{"schema_version": 5, "command": "optimize", '
        '"parameters": {"fmax": {"value": 0.01, "source": "user"}}, '
        '"stages": [{"index": 0, "kind": "optimize", "status": "converged"}]}')
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--output-dir", str(tmp_path / "freqrun")])
    assert result.exit_code == 0, result.stdout
    record = json.loads(
        (tmp_path / "freqrun" / "mliprun_run.json").read_text())
    results = record["stages"][0]["results"]
    assert results["fmax_expectation_source"] == "run_record"
    assert results["fmax_expectation"] == pytest.approx(0.01)
