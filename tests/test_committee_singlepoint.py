"""Committee evaluation of one configuration, no relaxation."""
import csv
import json
import sys
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


def _committee_file(tmp_path):
    """Two EMT members in this interpreter's own env.

    Matches the shape of ``fake_committee_file`` in ``tests/conftest.py``:
    ``env`` is the venv root (``python_for_env`` looks for ``bin/python``
    inside it), not the interpreter path itself.
    """
    env = Path(sys.executable).parents[1]
    path = tmp_path / "committee.yaml"
    path.write_text(
        "members:\n"
        f"  - {{env: {env}, mlip: emt, name: member_a}}\n"
        f"  - {{env: {env}, mlip: emt, name: member_b}}\n",
        encoding="utf-8")
    return path


def test_two_identical_members_agree_exactly(structure, tmp_path):
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    assert result.exit_code == 0, result.stdout
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    summary = record["stages"][0]["results"]["committee_uncertainty"]
    assert summary["sigma_max_free_final_eV_per_A"] == pytest.approx(
        0.0, abs=1e-12)


def test_the_peratom_sigma_csv_is_written(structure, tmp_path):
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    path = structure.parent / "singlepoint_committee_peratom.csv"
    assert path.exists()
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 12


def test_no_threshold_means_no_verdict(structure, tmp_path):
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    summary = record["stages"][0]["results"]["committee_uncertainty"]
    assert summary["flagged"] is None
    assert summary["threshold_source"] == "none"


def test_an_explicit_threshold_reaches_the_record(structure, tmp_path):
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path)),
        "--uncertainty-threshold", "0.05"])
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    summary = record["stages"][0]["results"]["committee_uncertainty"]
    assert summary["threshold_eV_per_A"] == pytest.approx(0.05)
    assert summary["threshold_source"] == "explicit"
    assert summary["flagged"] is False     # identical members, sigma is 0


def test_the_record_names_both_members(structure, tmp_path):
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    members = record["provenance"]["committee"]["members"]
    assert [m["name"] for m in members] == ["member_a", "member_b"]


def test_the_echo_does_not_claim_a_final_geometry(structure, tmp_path):
    """`report_committee_uncertainty` is shared with `optimize` and `freq`.
    A singlepoint has one configuration and no relaxation, so "at the final
    geometry" was false here."""
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "Committee disagreement at the evaluated configuration" in (
        result.output)
    assert "final geometry" not in result.output


def test_a_tripped_threshold_does_not_claim_a_located_minimum(structure,
                                                               tmp_path):
    """Two identical EMT members give sigma exactly 0, so -1 is the only
    threshold this harness can exceed."""
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path)),
        "--uncertainty-threshold", "-1"])
    assert result.exit_code == 0, result.output
    assert "deserves a DFT check" in result.output
    assert "The located minimum sits inside" not in result.output
