"""run_optimization driven by a committee."""
import csv
import json
import math
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from mliprun.core.committee.calculator import CommitteeCalculator, CommitteeError
from mliprun.core.committee.config import LEVEL_TABLE_VERSION
from mliprun.core.committee.remote import RemoteMember
from mliprun.core.optimize import run_optimization

STUB_DIR = Path(__file__).parent / "committee_stubs"

#: Must match tests/committee_stubs/biased_worker.py.
BIAS_EV_PER_A = 2.0


def _rattled():
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 1)
    atoms.rattle(stdev=0.05, seed=7)
    return atoms


def _committee(tmp_path, n=2, argv=None, names=None):
    names = list(names) if names else [f"member_{i}" for i in range(n)]
    members = [
        RemoteMember(name, sys.executable, mlip="emt",
                     log_path=tmp_path / f"committee_{name}.log",
                     timeout=60.0, argv=argv)
        for name in names
    ]
    return CommitteeCalculator(members, mixed_theory=True,
                               levels=("unknown",))


def _disagreeing_committee(tmp_path):
    """One plain-EMT member and one with a fixed force bias.

    Two members with a constant offset on one force component give
    sigma_per_atom = BIAS / sqrt(2) exactly (``ddof=1``) on every atom, so the
    uncertainty a run reports is a known, large, non-zero number rather than
    the identically-zero sigma of an all-EMT committee. That is what makes a
    stale value copied from a previous structure unmistakable.
    """
    argvs = [None, [sys.executable, str(STUB_DIR / "biased_worker.py")]]
    members = [
        RemoteMember(name, sys.executable, mlip="emt",
                     log_path=tmp_path / f"committee_{name}.log",
                     timeout=60.0, argv=argv)
        for name, argv in zip(("plain", "biased"), argvs)
    ]
    return CommitteeCalculator(members, mixed_theory=False,
                               levels=("unknown",))


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _record(tmp_path):
    return json.loads((tmp_path / "mliprun_run.json").read_text())


class TestCommitteeRun:
    def test_it_writes_both_committee_files(self, tmp_path):
        atoms = _rattled()
        committee = _committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=30,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee)
        finally:
            committee.close()

        trace = _read_csv(tmp_path / "opt_committee.csv")
        peratom = _read_csv(tmp_path / "opt_committee_peratom.csv")
        assert len(trace) >= 1
        assert len(peratom) == len(atoms)
        assert [int(r["atom_index"]) for r in peratom] == list(
            range(len(atoms)))

    def test_identical_members_reproduce_the_single_model_relaxation(
            self, tmp_path):
        """The committee must not perturb the trajectory when its members
        agree: same final positions, same energy, sigma identically zero."""
        atoms = _rattled()
        committee = _committee(tmp_path, n=3)
        committee.start()
        atoms.calc = committee
        try:
            converged = run_optimization(atoms, fmax=0.05, max_steps=50,
                                         output_dir=tmp_path,
                                         model_name="committee",
                                         verbose=False, committee=committee)
        finally:
            committee.close()

        reference = _rattled()
        reference.calc = EMT()
        BFGS(reference, logfile=str(tmp_path / "ref.log")).run(fmax=0.05,
                                                               steps=50)
        # `bool(...)`, not `is True`: this ASE/numpy combination returns
        # `np.True_` from `opt.run()`, which is truthy but not identical to
        # Python's `True` singleton.
        assert bool(converged) is True
        assert atoms.get_positions() == pytest.approx(
            reference.get_positions(), abs=1e-8)

        trace = _read_csv(tmp_path / "opt_committee.csv")
        assert all(float(r["sigma_max_eV_per_A"]) == pytest.approx(0.0,
                                                                   abs=1e-12)
                   for r in trace)
        assert all(float(r["energy_spread_aligned_eV"]) == pytest.approx(
            0.0, abs=1e-12) for r in trace)

    def test_the_trace_has_one_row_per_optimizer_step(self, tmp_path):
        atoms = _rattled()
        committee = _committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=30,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee)
        finally:
            committee.close()

        trace = _read_csv(tmp_path / "opt_committee.csv")
        convergence = _read_csv(tmp_path / "opt_convergence.csv")
        assert len(trace) == len(convergence)
        assert [int(r["step"]) for r in trace] == [int(r["step"])
                                                   for r in convergence]
        for committee_row, convergence_row in zip(trace, convergence):
            assert float(committee_row["fmax_eV_per_A"]) == pytest.approx(
                float(convergence_row["fmax(eV/A)"]), abs=1e-12)
            assert float(committee_row["energy_mean_eV"]) == pytest.approx(
                float(convergence_row["energy(eV)"]), abs=1e-10)

    def test_the_mixed_theory_flag_reaches_every_row(self, tmp_path):
        atoms = _rattled()
        committee = _committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=10,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee)
        finally:
            committee.close()
        assert all(r["mixed_theory"] == "True"
                   for r in _read_csv(tmp_path / "opt_committee.csv"))


class TestRunRecord:
    def test_the_record_carries_members_and_uncertainty(self, tmp_path,
                                                        fake_committee_file):
        path, config = fake_committee_file
        atoms = _rattled()
        committee = _committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=30,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee,
                             committee_config=config)
        finally:
            committee.close()

        record = _record(tmp_path)
        assert record["schema_version"] == 4
        assert len(record["provenance"]["committee"]["members"]) == 2
        assert record["provenance"]["committee_config_sha256"] == config.sha256

        uncertainty = record["stages"][0]["results"]["committee_uncertainty"]
        assert uncertainty["threshold_source"] == "fmax"
        assert uncertainty["threshold_eV_per_A"] == pytest.approx(0.05,
                                                                  abs=1e-12)
        assert uncertainty["sigma_max_final_eV_per_A"] == pytest.approx(
            0.0, abs=1e-12)
        assert uncertainty["flagged"] is False

    def test_an_explicit_threshold_is_recorded_as_explicit(self, tmp_path):
        atoms = _rattled()
        committee = _committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=10,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee,
                             uncertainty_threshold=0.2)
        finally:
            committee.close()
        uncertainty = _record(tmp_path)["stages"][0]["results"][
            "committee_uncertainty"]
        assert uncertainty["threshold_source"] == "explicit"
        assert uncertainty["threshold_eV_per_A"] == pytest.approx(0.2,
                                                                  abs=1e-12)


class TestPartialResults:
    def test_a_member_dying_mid_run_keeps_the_steps_already_written(
            self, tmp_path, monkeypatch):
        """A committee run that dies at step N keeps its first N steps."""
        monkeypatch.setenv("MLIPRUN_STUB_DIE_AFTER", "4")
        atoms = _rattled()
        committee = _committee(
            tmp_path, argv=[sys.executable, str(STUB_DIR / "dying_worker.py")])
        committee.start()
        atoms.calc = committee
        try:
            with pytest.raises(CommitteeError):
                run_optimization(atoms, fmax=0.001, max_steps=100,
                                 output_dir=tmp_path, model_name="committee",
                                 verbose=False, committee=committee)
        finally:
            committee.close()

        trace = _read_csv(tmp_path / "opt_committee.csv")
        assert 1 <= len(trace) <= 4
        assert _record(tmp_path)["status"] == "failed"
        assert "committee incomplete" in (
            _record(tmp_path)["stages"][0]["results"]["error"])


class TestReusedCommittee:
    """A committee may be reused across structures; its statistics may not.

    ``run_optimization``'s own docstring invites reuse ("an API caller may
    reuse one loaded committee across many structures"), and until this was
    fixed the calculator's ``latest`` still held the PREVIOUS structure's
    evaluation. A structure that failed before its first successful
    evaluation therefore had the previous structure's sigma, atom index and
    flag written into its own persisted record, wearing its own element
    symbols.
    """

    def test_a_failed_run_records_no_uncertainty_from_the_previous_structure(
            self, tmp_path):
        first_dir = tmp_path / "cu"
        second_dir = tmp_path / "fe"
        committee = _disagreeing_committee(tmp_path)
        committee.start()
        try:
            copper = _rattled()
            copper.calc = committee
            run_optimization(copper, fmax=0.05, max_steps=3,
                             output_dir=first_dir, model_name="committee",
                             verbose=False, committee=committee)

            # Iron has no EMT potential, so EVERY member rejects the geometry
            # on the first evaluation: this run never produces a statistic of
            # its own.
            iron = bulk("Fe", "bcc", a=2.87)
            iron.calc = committee
            with pytest.raises(CommitteeError):
                run_optimization(iron, fmax=0.05, max_steps=3,
                                 output_dir=second_dir,
                                 model_name="committee", verbose=False,
                                 committee=committee)
        finally:
            committee.close()

        # The numbers the leak would have copied forward: real, large, and
        # attached to a Cu atom.
        first = _record(first_dir)["stages"][0]["results"][
            "committee_uncertainty"]
        assert first["sigma_max_final_eV_per_A"] == pytest.approx(
            BIAS_EV_PER_A / math.sqrt(2.0), rel=1e-9)
        assert first["worst_atom_symbol"] == "Cu"
        assert first["flagged"] is True

        second_record = _record(second_dir)
        assert second_record["status"] == "failed"
        second = second_record["stages"][0]["results"][
            "committee_uncertainty"]
        assert second["n_steps"] == 0
        assert second["sigma_max_final_eV_per_A"] is None
        assert second["sigma_mean_final_eV_per_A"] is None
        assert second["sigma_max_peak_eV_per_A"] is None
        assert second["peak_step"] is None
        assert second["worst_atom"] is None
        assert second["worst_atom_symbol"] is None
        assert second["energy_spread_aligned_final_eV"] is None
        assert second["flagged"] is False
        # The trace file exists but holds only its header: zero steps ran.
        assert _read_csv(second_dir / "opt_committee.csv") == []

    def test_the_flag_echo_is_cleared_too(self, tmp_path):
        """``latest_uncertainty_summary`` is what the CLI echoes.

        Left stale, a failed run would print the previous structure's
        high-disagreement warning as if it were this structure's.
        """
        committee = _disagreeing_committee(tmp_path)
        committee.start()
        try:
            copper = _rattled()
            copper.calc = committee
            run_optimization(copper, fmax=0.05, max_steps=3,
                             output_dir=tmp_path / "cu",
                             model_name="committee", verbose=False,
                             committee=committee)
            assert committee.latest_uncertainty_summary["flagged"] is True

            iron = bulk("Fe", "bcc", a=2.87)
            iron.calc = committee
            with pytest.raises(CommitteeError):
                run_optimization(iron, fmax=0.05, max_steps=3,
                                 output_dir=tmp_path / "fe",
                                 model_name="committee", verbose=False,
                                 committee=committee)
        finally:
            committee.close()

        assert committee.latest is None
        assert committee.latest_uncertainty_summary["flagged"] is False
        assert (committee.latest_uncertainty_summary[
            "sigma_max_final_eV_per_A"] is None)


class TestMeasuredProvenance:
    """What actually loaded, not only what committee.yaml declared."""

    def test_each_member_records_the_versions_its_own_env_reported(
            self, tmp_path, fake_committee_file):
        _path, config = fake_committee_file
        names = [m.name for m in config.members]
        atoms = _rattled()
        committee = _committee(tmp_path, names=names)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=5,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee,
                             committee_config=config)
        finally:
            committee.close()

        members = _record(tmp_path)["provenance"]["committee"]["members"]
        assert [m["name"] for m in members] == names
        for block, spec in zip(members, config.members):
            measured = block["measured"]
            # Measured: reported back over the bridge from inside the member's
            # own interpreter. These tests run every member on this same
            # interpreter, so the values are exactly checkable.
            assert measured["python"] == platform.python_version()
            assert measured["executable"] == sys.executable
            assert measured["ase"] == version("ase")
            assert measured["mliprun"] == version("mliprun")
            assert "torch" in measured
            # 'emt' is ASE's built-in, not an MLIP distribution.
            assert measured["package"] is None
            assert measured["package_version"] is None
            # Declared: what the YAML asked for, in its own namespace. The
            # declared 'python' is an interpreter PATH, the measured one a
            # version string; they must not be conflated.
            assert block["python"] == spec.python_exe
            assert block["env"] == spec.env

    def test_the_level_table_version_is_stamped_into_the_record(
            self, tmp_path, fake_committee_file):
        _path, config = fake_committee_file
        atoms = _rattled()
        committee = _committee(tmp_path,
                               names=[m.name for m in config.members])
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=3,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee,
                             committee_config=config)
        finally:
            committee.close()

        block = _record(tmp_path)["provenance"]["committee"]
        assert block["level_table_version"] == LEVEL_TABLE_VERSION

    def test_measured_is_null_when_no_member_ever_started(
            self, tmp_path, fake_committee_file):
        """A record built from the config alone says "unknown", not "empty"."""
        _path, config = fake_committee_file
        block = config.as_provenance()
        assert [m["measured"] for m in block["members"]] == [None, None]


class TestDeviceOnACommitteeRun:
    def test_both_device_fields_say_committee(self, tmp_path):
        """The driver env has no torch by design (ADR 0001), so its resolved
        device is "cpu" even when every member sits on its own GPU. The record
        must not make that claim."""
        atoms = _rattled()
        committee = _committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.05, max_steps=3,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee,
                             device_requested="auto", device_resolved="cpu")
        finally:
            committee.close()

        provenance = _record(tmp_path)["provenance"]
        assert provenance["device_requested"] == "committee"
        assert provenance["device_resolved"] == "committee"


class TestSingleModelUnchanged:
    def test_no_committee_files_without_a_committee(self, tmp_path):
        atoms = _rattled()
        atoms.calc = EMT()
        run_optimization(atoms, fmax=0.05, max_steps=30, output_dir=tmp_path,
                         model_name="emt", verbose=False)
        assert not (tmp_path / "opt_committee.csv").exists()
        assert not (tmp_path / "opt_committee_peratom.csv").exists()
        assert "committee" not in _record(tmp_path)["provenance"]

    def test_the_device_fields_are_untouched_without_a_committee(
            self, tmp_path):
        """The committee override must not reach the single-model path."""
        atoms = _rattled()
        atoms.calc = EMT()
        run_optimization(atoms, fmax=0.05, max_steps=5, output_dir=tmp_path,
                         model_name="emt", verbose=False,
                         device_requested="auto", device_resolved="cpu")
        provenance = _record(tmp_path)["provenance"]
        assert provenance["device_requested"] == "auto"
        assert provenance["device_resolved"] == "cpu"
