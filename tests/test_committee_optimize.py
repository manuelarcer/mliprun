"""run_optimization driven by a committee."""
import csv
import json
import sys
from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from mliprun.core.committee.calculator import CommitteeCalculator, CommitteeError
from mliprun.core.committee.remote import RemoteMember
from mliprun.core.optimize import run_optimization

STUB_DIR = Path(__file__).parent / "committee_stubs"


def _rattled():
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 1)
    atoms.rattle(stdev=0.05, seed=7)
    return atoms


def _committee(tmp_path, n=2, argv=None):
    members = [
        RemoteMember(f"member_{i}", sys.executable, mlip="emt",
                     log_path=tmp_path / f"committee_member_{i}.log",
                     timeout=60.0, argv=argv)
        for i in range(n)
    ]
    return CommitteeCalculator(members, mixed_theory=True,
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


class TestSingleModelUnchanged:
    def test_no_committee_files_without_a_committee(self, tmp_path):
        atoms = _rattled()
        atoms.calc = EMT()
        run_optimization(atoms, fmax=0.05, max_steps=30, output_dir=tmp_path,
                         model_name="emt", verbose=False)
        assert not (tmp_path / "opt_committee.csv").exists()
        assert not (tmp_path / "opt_committee_peratom.csv").exists()
        assert "committee" not in _record(tmp_path)["provenance"]
