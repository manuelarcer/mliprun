"""Per-member frequencies from one displacement sweep."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
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


def test_a_restart_keeps_the_uncertainty_block_it_reports(structure,
                                                           tmp_path):
    """A fully cached committee restart must still report its disagreement.

    `committee_uncertainty` does not come from the cache -- it comes from one
    explicit evaluation of the input geometry after the sweep -- so a restart
    that made zero force calls must still carry it, correctly. The
    zero-force-calls assertion is what makes this a restart and not a second
    fresh sweep.
    """
    from mliprun.cli.commands.freq import app
    first = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path)),
        "--uncertainty-threshold", "0.05"])
    assert first.exit_code == 0, first.stdout

    second = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path)),
        "--uncertainty-threshold", "0.05"])
    assert second.exit_code == 0, second.stdout

    record = json.loads((structure.parent / "mliprun_run.json").read_text())
    results = record["stages"][0]["results"]
    assert results["n_force_calls"] == 0          # the cache was reused
    block = results["committee_uncertainty"]
    # Two identical EMT members: sigma is exactly zero, and with a threshold
    # of 0.05 eV/A nothing is flagged. Both are values, not mere presence.
    assert block["sigma_max_free_final_eV_per_A"] == pytest.approx(
        0.0, abs=1e-12)
    assert block["threshold_source"] == "explicit"
    assert block["threshold_eV_per_A"] == pytest.approx(0.05)
    assert block["flagged"] is False
    assert block["n_free_atoms"] == 2             # N2, nothing constrained


# -- crossing a single-model cache with a committee run -----------------

def test_a_single_model_cache_is_refused_by_a_committee_run(structure,
                                                             tmp_path):
    """`freq` then `freq --committee` in one directory.

    A single-model sweep writes entries whose only force key is `forces`, so
    reading `forces_per_member` out of them used to raise a bare
    `KeyError: 'forces_per_member'` -- after the run record was opened and
    before it was completed, leaving it saying `status: "running"`, which
    docs/OUTPUTS.md defines as "the job died without reporting back".
    """
    from ase.calculators.emt import EMT
    from ase.io import read

    from mliprun.cli.commands.freq import app
    from mliprun.core.vibrations import run_frequencies

    single = read(structure)
    single.calc = EMT()
    first = run_frequencies(single, output_dir=structure.parent,
                            model_name="emt")
    assert first["n_force_calls"] == 1 + 6 * 2    # a real single-model sweep

    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])

    assert result.exit_code == 1
    assert str(structure.parent / "freq") in result.stdout
    assert "--prefix" in result.stdout
    record = json.loads((structure.parent / "mliprun_run.json").read_text())
    assert record["status"] == "failed"
    assert record["stages"][-1]["status"] == "failed"
    assert "per-member forces" in record["stages"][-1]["results"]["error"]


def test_a_sidecar_that_lies_about_per_member_forces_is_still_caught(
        structure, tmp_path):
    """The post-sweep committee guard is the backstop to the pre-sweep
    identity check, which now catches this crossover first. Reaching the
    backstop means making the recorded identity agree with the committee run
    while the cache entries do not -- a hand-edited sidecar here, a future
    code path that writes one some other way in reality.

    Without this, the backstop would be a `raise` that has never executed.
    """
    from ase.calculators.emt import EMT
    from ase.io import read

    from mliprun.cli.commands.freq import app
    from mliprun.core.vibrations import run_frequencies

    single = read(structure)
    single.calc = EMT()
    run_frequencies(single, output_dir=structure.parent, model_name="emt")

    # Claim the cache is what a committee run would have written.
    sidecar = structure.parent / "freq_cache.json"
    recorded = json.loads(sidecar.read_text())
    recorded["model"] = "committee"
    recorded["per_member_forces"] = True
    sidecar.write_text(json.dumps(recorded), encoding="utf-8")

    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])

    assert result.exit_code == 1
    assert "single-model run" in result.stdout
    assert "per-member forces" in result.stdout
    record = json.loads((structure.parent / "mliprun_run.json").read_text())
    assert record["stages"][-1]["status"] == "failed"


def test_member_frequencies_refuses_a_single_model_cache_directly(tmp_path):
    """`member_frequencies` is a public entry point, so the backstop inside
    `_member_forces` is reachable without going through `run_frequencies`.
    A library caller handing it a single-model cache gets the named error
    rather than a bare `KeyError`."""
    from ase.calculators.emt import EMT

    from mliprun.core.vibrations import (
        CountingVibrations,
        FrequencyCacheError,
        member_frequencies,
    )

    atoms = molecule("N2")
    atoms.calc = EMT()
    vib = CountingVibrations(atoms, name=str(tmp_path / "vib"), delta=0.01,
                             nfree=2)
    vib.run()
    vib.read()
    data = vib.get_vibrations()
    modes = np.asarray(data.get_modes()).reshape(
        len(data.get_frequencies()), -1)

    with pytest.raises(FrequencyCacheError) as caught:
        member_frequencies(vib, atoms, vib.indices, vib.delta, 2, "central",
                           "standard", ["member_a"], modes)
    assert "per-member forces" in str(caught.value)


def test_a_different_prefix_lets_a_committee_run_beside_a_single_model_one(
        structure, tmp_path):
    """The remedy the refusal names has to actually work."""
    from ase.calculators.emt import EMT
    from ase.io import read

    from mliprun.cli.commands.freq import app
    from mliprun.core.vibrations import run_frequencies

    single = read(structure)
    single.calc = EMT()
    run_frequencies(single, output_dir=structure.parent, model_name="emt")

    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path)),
        "--prefix", "freq_committee"])
    assert result.exit_code == 0, result.stdout
    rows = list(csv.DictReader(
        (structure.parent
         / "freq_committee_committee_frequencies.csv").open()))
    assert rows
    for row in rows:
        assert float(row["member_a_overlap"]) == pytest.approx(1.0, abs=1e-9)


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


class _AxisSplittingFakeCommittee(Calculator):
    """Two members that stiffen a near-degenerate pair along different axes.

    Purpose-built to produce a genuinely low mode overlap through the real
    committee machinery, rather than asserting arithmetic inline.

    ``member_a`` adds a harmonic restoring force along **x** and
    ``member_b`` the same along **y**, both measured from one fixed
    reference geometry. Two consequences follow, and they are the whole
    point:

    1. The consensus (their mean) is stiffened equally along x and y, so its
       transverse modes stay **degenerate** -- and an eigenvector basis
       inside a degenerate subspace is arbitrary, exactly the situation the
       overlap column exists for.
    2. Each member's own basis is pinned to its own axis, and the two
       members are mirror images of each other, so their sorted spectra are
       **identical**. The per-mode spread across members is therefore
       exactly zero even though every member's basis is rotated away from
       the committee's.

    Whatever basis LAPACK returns for the committee's degenerate subspace,
    it cannot agree with both members at once, so the worst overlap is low
    by construction rather than by luck.

    Reuses the real ``committee_statistics`` / ``free_component_mask``, like
    ``_DisplacementLeakingFakeCommittee`` below, so only the mode pairing is
    under test here and not the statistics.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, reference_positions, stiffness=5.0):
        super().__init__()
        self.member_names = ["member_a", "member_b"]
        self.members = [
            type("Member", (), {"name": name})() for name in self.member_names]
        self._reference = np.asarray(reference_positions, dtype=float).copy()
        self._stiffness = float(stiffness)
        self.latest = None
        self.latest_uncertainty_summary = None

    def preflight(self, atoms):
        return self._evaluate(atoms)

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        stats = self._evaluate(self.atoms)
        self.results["energy"] = stats["energy_mean"]
        self.results["free_energy"] = stats["energy_mean"]
        self.results["forces"] = stats["forces_mean"]

    def _evaluate(self, atoms):
        from ase.calculators.emt import EMT

        from mliprun.core.committee.calculator import (
            committee_statistics,
            free_component_mask,
        )

        reference = atoms.copy()
        reference.calc = EMT()
        base = np.asarray(reference.get_forces(), dtype=float)
        energy = float(reference.get_potential_energy())
        offset = atoms.get_positions() - self._reference

        f_a = base.copy()
        f_a[:, 0] -= self._stiffness * offset[:, 0]      # stiff along x
        f_b = base.copy()
        f_b[:, 1] -= self._stiffness * offset[:, 1]      # stiff along y
        stacked = np.stack([f_a, f_b])

        free_mask, unhandled = free_component_mask(atoms)
        stats = committee_statistics([energy, energy], stacked,
                                     free_mask=free_mask)
        stats["free_mask"] = free_mask
        stats["unhandled_constraints"] = unhandled
        stats["energies"] = {"member_a": energy, "member_b": energy}
        stats["forces_per_member"] = stacked
        self.latest = stats
        return stats


def test_a_rotated_mode_basis_is_caught_by_the_overlap_not_by_the_spread(
        tmp_path, caplog):
    """The failure the overlap column exists to make visible, through the
    real code: ``member_frequencies``, the real Hessian per member, the real
    unit-row renormalization and the real index pairing.

    The two members' sorted spectra are identical, so the per-mode spread is
    exactly zero and says "the members agree". The overlap says the pairing
    is what is wrong. If the spread were the only diagnostic, this run would
    look clean.
    """
    import logging

    from mliprun.core.vibrations import MODE_OVERLAP_WARN, run_frequencies

    atoms = molecule("N2")
    atoms.center(vacuum=5.0)
    committee = _AxisSplittingFakeCommittee(atoms.get_positions().copy())
    atoms.calc = committee

    with caplog.at_level(logging.WARNING, logger="mliprun.core.vibrations"):
        results = run_frequencies(atoms, output_dir=tmp_path,
                                  committee=committee)

    block = results["committee_frequencies"]
    # The spread sees nothing: mirror-image members, identical spectra. The
    # residue is eigenvalue noise on the near-zero modes (measured 4.2e-6
    # cm^-1), seven orders below the >50 cm^-1 mispairing asserted at the
    # end of this test.
    assert max(block["frequency_member_std_cm-1"]) < 1e-4
    # The overlap does. Low by construction: the committee's transverse
    # modes are degenerate, so its basis there is arbitrary and cannot match
    # both members' axis-aligned bases at once.
    assert block["worst_mode_overlap"] < MODE_OVERLAP_WARN
    assert block["mode_pairing_suspect"] is True
    assert any("lowest mode overlap" in record.message
               for record in caplog.records)

    # At least one member's frequency genuinely disagrees with the
    # committee's for the mode it is paired against -- the consequence of
    # the mispairing, and what makes the diagnostic worth having.
    rows = list(csv.DictReader(
        (tmp_path / "freq_committee_frequencies.csv").open()))
    assert any(abs(float(row["member_a_cm-1"])
                   - float(row["frequency_committee_cm-1"])) > 50.0
               for row in rows)


def test_the_run_record_flags_suspect_mode_pairing(structure, tmp_path,
                                                   monkeypatch):
    """With identical members the pairing is clean, so the flag is False.
    Forcing the threshold above 1.0 makes every clean pairing 'suspect',
    which is how the flag's wiring is exercised without faking an overlap."""
    import mliprun.core.vibrations as vibrations_module
    from mliprun.cli.commands.freq import app

    monkeypatch.setattr(vibrations_module, "MODE_OVERLAP_WARN", 1.5)
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    assert result.exit_code == 0, result.stdout
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    block = record["stages"][0]["results"]["committee_frequencies"]
    assert block["mode_pairing_suspect"] is True
    assert block["worst_mode_overlap"] == pytest.approx(1.0, abs=1e-6)


# -- committee_uncertainty on `freq run` --------------------------------

def test_identical_members_uncertainty_sigma_is_exactly_zero(structure,
                                                              tmp_path):
    from mliprun.cli.commands.freq import app
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    assert result.exit_code == 0, result.stdout
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    block = record["stages"][0]["results"]["committee_uncertainty"]
    assert block["sigma_max_free_final_eV_per_A"] == pytest.approx(
        0.0, abs=1e-12)


def test_an_explicit_threshold_with_identical_members_is_not_flagged(
        structure, tmp_path):
    from mliprun.cli.commands.freq import app
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path)),
        "--uncertainty-threshold", "0.05"])
    assert result.exit_code == 0, result.stdout
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    block = record["stages"][0]["results"]["committee_uncertainty"]
    assert block["threshold_source"] == "explicit"
    assert block["threshold_eV_per_A"] == pytest.approx(0.05)
    assert block["flagged"] is False       # identical members, sigma is 0


def test_the_echo_names_the_input_geometry_not_a_final_one(structure,
                                                            tmp_path):
    """`report_committee_uncertainty` is shared with `optimize` and
    `singlepoint`. `freq` displaces the structure but reports its
    disagreement at the INPUT geometry, so "at the final geometry" was false
    here."""
    from mliprun.cli.commands.freq import app
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    assert result.exit_code == 0, result.stdout
    assert "Committee disagreement at the input geometry" in result.stdout
    assert "final geometry" not in result.stdout


def test_a_tripped_threshold_does_not_claim_a_located_minimum(structure,
                                                               tmp_path):
    """`freq` locates nothing: it reports the geometry it was handed. Two
    identical EMT members give sigma exactly 0, so -1 is the only threshold
    this harness can exceed."""
    from mliprun.cli.commands.freq import app
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path)),
        "--uncertainty-threshold", "-1"])
    assert result.exit_code == 0, result.stdout
    assert "deserves a DFT check" in result.stdout
    assert "The located minimum sits inside" not in result.stdout


def test_no_threshold_means_no_uncertainty_verdict(structure, tmp_path):
    from mliprun.cli.commands.freq import app
    result = runner.invoke(app, [
        "run", "--structure", str(structure),
        "--committee", str(_committee_file(tmp_path))])
    assert result.exit_code == 0, result.stdout
    record = json.loads(
        (structure.parent / "mliprun_run.json").read_text())
    block = record["stages"][0]["results"]["committee_uncertainty"]
    assert block["threshold_source"] == "none"
    assert block["flagged"] is None


class _DisplacementLeakingFakeCommittee(Calculator):
    """A committee whose disagreement is proportional to displacement from
    one fixed reference geometry -- not to which member is asked.

    Purpose-built for one test: it makes the geometry at which
    :func:`~mliprun.core.vibrations.run_frequencies` evaluates the committee
    observable. Two real EMT members (as used everywhere else in this file)
    agree everywhere, so they cannot distinguish "evaluated at the input
    geometry" from "evaluated at whatever displacement ran last" -- both
    give sigma == 0. Here, member_b's force is member_a's EMT force
    plus ``coeff * (current_position - reference_position)``, so sigma is
    EXACTLY 0 only at the reference geometry and of order
    ``coeff * delta`` (a fraction of an eV/A, not numerical noise) at any
    displaced one. ``reference_position`` is fixed at construction time, to
    the atoms' geometry before any displacement runs.

    Reuses the real ``committee_statistics``/``free_component_mask`` (Task
    4/singlepoint) rather than reimplementing the sigma reduction, so this
    only tests WHICH geometry gets evaluated, not the arithmetic.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, reference_positions, coeff=200.0):
        super().__init__()
        self.member_names = ["member_a", "member_b"]
        self.members = [
            type("Member", (), {"name": name})() for name in self.member_names]
        self._reference = np.asarray(reference_positions, dtype=float).copy()
        self._coeff = coeff
        self.latest = None
        self.latest_uncertainty_summary = None

    def preflight(self, atoms):
        return self._evaluate(atoms)

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        stats = self._evaluate(self.atoms)
        self.results["energy"] = stats["energy_mean"]
        self.results["free_energy"] = stats["energy_mean"]
        self.results["forces"] = stats["forces_mean"]

    def _evaluate(self, atoms):
        from ase.calculators.emt import EMT

        from mliprun.core.committee.calculator import (
            committee_statistics,
            free_component_mask,
        )

        reference = atoms.copy()
        reference.calc = EMT()
        f_a = np.asarray(reference.get_forces(), dtype=float)
        e_a = float(reference.get_potential_energy())
        leak = atoms.get_positions() - self._reference
        f_b = f_a + self._coeff * leak
        stacked = np.stack([f_a, f_b])

        free_mask, unhandled = free_component_mask(atoms)
        stats = committee_statistics([e_a, e_a], stacked, free_mask=free_mask)
        stats["free_mask"] = free_mask
        stats["unhandled_constraints"] = unhandled
        stats["energies"] = {"member_a": e_a, "member_b": e_a}
        stats["forces_per_member"] = stacked
        self.latest = stats
        return stats


def test_committee_uncertainty_describes_the_input_geometry_not_a_displaced_one(
        tmp_path):
    """The gap this whole round is about: reading whatever
    ``committee.latest`` happened to hold after the sweep would describe the
    LAST displaced geometry, not the input one. ``coeff=200`` on a
    ``delta=0.01`` A displacement would leak roughly 2 eV/A of spurious
    sigma if that bug were reintroduced -- five orders of magnitude above
    the 1e-9 tolerance below, so this is not a coin flip."""
    from mliprun.core.vibrations import run_frequencies

    atoms = molecule("N2")
    atoms.center(vacuum=5.0)
    reference_positions = atoms.get_positions().copy()
    committee = _DisplacementLeakingFakeCommittee(reference_positions)
    atoms.calc = committee

    results = run_frequencies(atoms, output_dir=tmp_path, committee=committee)

    assert results["committee_uncertainty"][
        "sigma_max_free_final_eV_per_A"] == pytest.approx(0.0, abs=1e-9)
    # The atoms object itself is left at the input geometry too -- the fact
    # that makes the fix possible in the first place.
    assert atoms.get_positions() == pytest.approx(
        reference_positions, abs=1e-12)
