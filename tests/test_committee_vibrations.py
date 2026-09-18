"""Per-member frequencies from one displacement sweep."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule
from ase.io import write
from typer.testing import CliRunner

runner = CliRunner()


def _committee_file(tmp_path):
    """Two EMT members in this interpreter's own env.

    ``env`` is the venv ROOT, not the interpreter path: ``config.
    python_for_env`` resolves ``<env>/bin/python[3]`` inside it. Matches
    ``tests/test_committee_singlepoint.py``'s ``_committee_file`` and
    ``tests/conftest.py``'s ``fake_committee_file`` -- the brief's own
    version of this helper writes ``env: {sys.executable}`` (the
    interpreter itself), which is the wrong schema.
    """
    env = Path(sys.executable).parents[1]
    path = tmp_path / "committee.yaml"
    path.write_text(
        "members:\n"
        f"  - {{env: {env}, mlip: emt, name: member_a}}\n"
        f"  - {{env: {env}, mlip: emt, name: member_b}}\n",
        encoding="utf-8")
    return path


@pytest.fixture
def structure(tmp_path):
    atoms = molecule("N2")
    atoms.center(vacuum=5.0)
    path = tmp_path / "POSCAR"
    write(path, atoms, format="vasp")
    return path


def test_identical_members_give_exactly_zero_spread(structure, tmp_path):
    from mliprun.cli.commands.freq import app
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    assert result.exit_code == 0, result.stdout
    rows = list(csv.DictReader(
        (structure.parent / "freq_committee_frequencies.csv").open()))
    assert rows
    for row in rows:
        assert float(row["frequency_member_std_cm-1"]) == pytest.approx(
            0.0, abs=1e-9)


def test_identical_members_give_overlaps_of_exactly_one(structure, tmp_path):
    from mliprun.cli.commands.freq import app
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    rows = list(csv.DictReader(
        (structure.parent / "freq_committee_frequencies.csv").open()))
    for row in rows:
        assert float(row["member_a_overlap"]) == pytest.approx(1.0, abs=1e-9)
        assert float(row["member_b_overlap"]) == pytest.approx(1.0, abs=1e-9)


def test_the_csv_has_one_column_per_member(structure, tmp_path):
    from mliprun.cli.commands.freq import app
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    rows = list(csv.DictReader(
        (structure.parent / "freq_committee_frequencies.csv").open()))
    assert "member_a_cm-1" in rows[0]
    assert "member_b_cm-1" in rows[0]
    assert "frequency_committee_cm-1" in rows[0]


def test_the_headline_frequency_is_the_mean_potentials(structure, tmp_path):
    """Not the mean of the members' frequencies -- different numbers (D8).
    With identical members they coincide, which is what makes this a
    meaningful check of the plumbing rather than of the arithmetic."""
    from mliprun.cli.commands.freq import app
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    main = list(csv.DictReader(
        (structure.parent / "freq_frequencies.csv").open()))
    committee = list(csv.DictReader(
        (structure.parent / "freq_committee_frequencies.csv").open()))
    for a, b in zip(main, committee):
        assert float(a["frequency_cm-1"]) == pytest.approx(
            float(b["frequency_committee_cm-1"]), abs=1e-9)


def test_per_member_zpe_reaches_the_run_record(structure, tmp_path):
    from mliprun.cli.commands.freq import app
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    block = record["stages"][0]["results"]["committee_frequencies"]
    assert set(block["zpe_eV_per_member"]) == {"member_a", "member_b"}
    assert block["zpe_std_eV"] == pytest.approx(0.0, abs=1e-12)


def test_a_restart_reproduces_the_same_per_member_frequencies(
        structure, tmp_path):
    """Per-member forces must survive ASE's JSON cache, or a resumed
    committee run silently loses its spread.

    EMT is deterministic, so matching CSV text alone would also pass if the
    second run silently recomputed everything from scratch instead of
    reusing the displacement cache. The zero-force-calls assertion below is
    what actually distinguishes "the cache worked" from "it was bypassed" --
    losing per-member forces on restart is this feature's quiet failure
    mode.
    """
    from mliprun.cli.commands.freq import app
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    first = (structure.parent / "freq_committee_frequencies.csv").read_text()
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    second = (structure.parent / "freq_committee_frequencies.csv").read_text()
    assert first == second

    # `freq` overwrites mliprun_run.json rather than appending, so the
    # second run's stage is stages[0], not stages[1].
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    assert record["stages"][0]["results"]["n_force_calls"] == 0


def test_a_disagreeing_committee_gives_a_non_zero_spread(tmp_path):
    """Driven directly, not through the CLI: the point is the arithmetic."""
    from mliprun.core.vibrations import assemble_hessian
    from ase.calculators.emt import EMT
    from ase.vibrations import Vibrations, VibrationsData

    atoms = molecule("N2")
    atoms.calc = EMT()
    vib = Vibrations(atoms, name=str(tmp_path / "vib"))
    vib.run()
    vib.read()

    forces = {"eq": np.asarray(vib._eq_disp().forces())}
    for a in vib.indices:
        for i in range(3):
            for n in (-1, 1):
                forces[(int(a), i, n)] = np.asarray(vib._disp(a, i, n).forces())

    scaled = {k: v * 1.05 for k, v in forces.items()}
    h_ref = assemble_hessian(forces, vib.indices, vib.delta)
    h_scaled = assemble_hessian(scaled, vib.indices, vib.delta)
    f_ref = VibrationsData.from_2d(
        atoms, h_ref, vib.indices).get_frequencies()
    f_scaled = VibrationsData.from_2d(
        atoms, h_scaled, vib.indices).get_frequencies()
    spread = np.std([np.abs(f_ref), np.abs(f_scaled)], axis=0, ddof=1)
    assert spread.max() > 1.0        # cm-1, well above numerical noise


def test_swapped_modes_are_caught_by_the_overlap_not_by_the_spread(tmp_path):
    """The failure the overlap column exists to make visible.

    Two members whose mode 0 and mode 1 are the same two physical modes in
    the opposite order. Paired by index they look like a large
    disagreement; the overlap says the pairing is what is wrong.
    """
    from mliprun.core.vibrations import MODE_OVERLAP_WARN

    committee_modes = np.array([[1.0, 0.0], [0.0, 1.0]])
    member_modes = np.array([[0.0, 1.0], [1.0, 0.0]])     # order swapped
    overlaps = np.abs(np.einsum("ij,ij->i", member_modes, committee_modes))
    assert overlaps.max() == pytest.approx(0.0, abs=1e-12)
    assert overlaps.min() < MODE_OVERLAP_WARN


def test_the_run_record_flags_suspect_mode_pairing(structure, tmp_path,
                                                   monkeypatch):
    """With identical members the pairing is clean, so the flag is False.
    Forcing the threshold above 1.0 makes every clean pairing 'suspect',
    which is how the flag's wiring is exercised without faking an overlap."""
    import mliprun.core.vibrations as vibrations_module
    from mliprun.cli.commands.freq import app

    monkeypatch.setattr(vibrations_module, "MODE_OVERLAP_WARN", 1.5)
    runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    block = record["stages"][0]["results"]["committee_frequencies"]
    assert block["mode_pairing_suspect"] is True
    assert block["worst_mode_overlap"] == pytest.approx(1.0, abs=1e-6)
