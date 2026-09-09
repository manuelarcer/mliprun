"""CommitteeCalculator: fan-out, abort behaviour, and use as an ASE calculator."""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixBondLengths
from ase.optimize import BFGS

from mliprun.core.committee.calculator import CommitteeCalculator, CommitteeError
from mliprun.core.committee.remote import RemoteMember

STUB_DIR = Path(__file__).parent / "committee_stubs"


class FakeMember:
    """A member that answers from a fixed table. No subprocess."""

    def __init__(self, name, energy, forces):
        self.name = name
        self.energy = energy
        self.forces = forces
        self.versions = {"ase": "fake"}
        self.started = False
        self.closed = False
        self.n_calls = 0

    def start(self):
        self.started = True
        return self.versions

    def calculate(self, numbers, positions, cell, pbc):
        self.n_calls += 1
        return self.energy, self.forces

    def close(self, timeout=5.0):
        self.closed = True


def _rattled():
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 1)
    atoms.rattle(stdev=0.05, seed=7)
    return atoms


def _emt_committee(tmp_path, n=2, argv=None):
    members = [
        RemoteMember(f"member_{i}", sys.executable, mlip="emt",
                     log_path=tmp_path / f"member_{i}.log", timeout=60.0,
                     argv=argv)
        for i in range(n)
    ]
    return CommitteeCalculator(members)


class TestArithmeticWithFakeMembers:
    def test_results_are_the_committee_mean(self):
        f_a = np.zeros((2, 3)); f_a[0, 0] = 1.0
        f_b = np.zeros((2, 3)); f_b[0, 0] = 3.0
        committee = CommitteeCalculator(
            [FakeMember("member_a", -1.0, f_a.tolist()),
             FakeMember("member_b", -3.0, f_b.tolist())])
        committee.start()
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        atoms.calc = committee

        assert atoms.get_potential_energy() == pytest.approx(-2.0, abs=1e-12)
        assert atoms.get_forces()[0, 0] == pytest.approx(2.0, abs=1e-12)
        assert committee.latest["sigma_max_free"] == pytest.approx(
            2.0 / np.sqrt(2.0), abs=1e-12)
        assert committee.latest["energies"] == {"member_a": -1.0,
                                                "member_b": -3.0}
        committee.close()

    def test_one_member_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="at least two"):
            CommitteeCalculator([FakeMember("member_a", -1.0,
                                            np.zeros((2, 3)).tolist())])

    def test_a_wrong_force_shape_aborts_and_tears_down(self):
        good = FakeMember("member_a", -1.0, np.zeros((2, 3)).tolist())
        bad = FakeMember("member_b", -1.0, np.zeros((5, 3)).tolist())
        committee = CommitteeCalculator([good, bad])
        committee.start()
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        atoms.calc = committee
        with pytest.raises(CommitteeError, match="member_b"):
            atoms.get_potential_energy()
        assert good.closed and bad.closed

    def test_a_non_finite_force_aborts_and_names_the_member(self):
        forces = np.zeros((2, 3)); forces[1, 1] = np.inf
        committee = CommitteeCalculator(
            [FakeMember("member_a", -1.0, np.zeros((2, 3)).tolist()),
             FakeMember("member_b", -1.0, forces.tolist())])
        committee.start()
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        atoms.calc = committee
        with pytest.raises(CommitteeError, match="member_b"):
            atoms.get_potential_energy()

    def test_a_failing_member_aborts_the_whole_run(self):
        """Never continue with fewer members: the mean and sigma would
        silently change definition mid-trajectory."""
        class Broken(FakeMember):
            def calculate(self, *args):
                raise RuntimeError("[Errno 32] Broken pipe")

        good = FakeMember("member_a", -1.0, np.zeros((2, 3)).tolist())
        broken = Broken("member_b", -1.0, np.zeros((2, 3)).tolist())
        committee = CommitteeCalculator([good, broken])
        committee.start()
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        atoms.calc = committee
        with pytest.raises(CommitteeError, match="committee incomplete"):
            atoms.get_potential_energy()
        assert good.closed and broken.closed

    def test_every_member_is_asked_once_per_evaluation(self):
        members = [FakeMember("member_a", -1.0, np.zeros((2, 3)).tolist()),
                   FakeMember("member_b", -1.0, np.zeros((2, 3)).tolist())]
        committee = CommitteeCalculator(members)
        committee.start()
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        atoms.calc = committee
        atoms.get_potential_energy()
        atoms.get_forces()          # cached: same geometry, no new calls
        assert [m.n_calls for m in members] == [1, 1]
        assert committee.n_evaluations == 1
        committee.close()

    def test_start_keeps_the_measured_versions_it_returns(self):
        """Every call site used to discard `start()`'s return value, so the
        versions each member measured in its own env never reached the run
        record. They are kept on the calculator now."""
        members = [FakeMember("member_a", -1.0, np.zeros((2, 3)).tolist()),
                   FakeMember("member_b", -1.0, np.zeros((2, 3)).tolist())]
        committee = CommitteeCalculator(members)
        assert committee.member_versions == {}
        returned = committee.start()
        assert returned == {"member_a": {"ase": "fake"},
                            "member_b": {"ase": "fake"}}
        assert committee.member_versions == returned
        committee.close()


class TestRealSubprocessRoundTrip:
    def test_identical_members_give_zero_sigma_and_the_single_model_energy(
            self, tmp_path):
        """N identical workers must reproduce the single-model result exactly
        and report no disagreement at all."""
        atoms = _rattled()
        committee = _emt_committee(tmp_path, n=3)
        committee.start()
        atoms.calc = committee
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
        finally:
            committee.close()

        reference = _rattled()
        reference.calc = EMT()
        assert energy == pytest.approx(reference.get_potential_energy(),
                                       abs=1e-10)
        assert forces == pytest.approx(reference.get_forces(), abs=1e-10)
        assert committee.latest["sigma_max_free"] == pytest.approx(0.0, abs=1e-12)

    def test_a_stock_optimizer_runs_through_the_committee_unchanged(
            self, tmp_path):
        """The whole architecture rests on this: downstream engines only ever
        touch atoms.calc, so nothing needs restructuring."""
        atoms = _rattled()
        committee = _emt_committee(tmp_path, n=2)
        committee.start()
        atoms.calc = committee
        try:
            BFGS(atoms, logfile=str(tmp_path / "opt.log")).run(fmax=0.05,
                                                               steps=50)
            positions = atoms.get_positions()
        finally:
            committee.close()

        reference = _rattled()
        reference.calc = EMT()
        BFGS(reference, logfile=str(tmp_path / "ref.log")).run(fmax=0.05,
                                                               steps=50)
        assert positions == pytest.approx(reference.get_positions(), abs=1e-8)

    def test_preflight_surfaces_a_bad_geometry_before_the_optimizer(
            self, tmp_path):
        """Finding 7: fairchem's UMA calculator rejects a pbc=(T,T,F) slab
        that MACE, SevenNet and CHGNet accept. The stand-in here is a worker
        that dies on its first calc."""
        committee = _emt_committee(
            tmp_path, n=2,
            argv=[sys.executable, str(STUB_DIR / "dying_worker.py")])
        committee.start()
        try:
            with pytest.raises(CommitteeError, match="committee incomplete"):
                committee.preflight(_rattled())
        finally:
            committee.close()

    def test_start_failure_tears_down_the_members_that_did_start(self,
                                                                 tmp_path):
        from mliprun.core.committee.remote import MemberError
        good = RemoteMember("member_a", sys.executable, mlip="emt",
                            log_path=tmp_path / "a.log")
        bad = RemoteMember("member_b", sys.executable,
                           mlip="not-a-real-model",
                           log_path=tmp_path / "b.log")
        committee = CommitteeCalculator([good, bad])
        with pytest.raises(MemberError):
            committee.start()
        assert not good.is_alive
        assert not bad.is_alive

    def test_context_manager_closes_every_member(self, tmp_path):
        with _emt_committee(tmp_path, n=2) as committee:
            committee.start()
            pids = [m.pid for m in committee.members]
            atoms = _rattled()
            atoms.calc = committee
            atoms.get_potential_energy()
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)


class TestConstraintsReachTheStatistic:
    """FakeMember answers from a fixed table, so the disagreement is exact
    and independent of geometry."""

    def _members(self, sigma_0, sigma_1):
        return [
            FakeMember("a", -1.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            FakeMember("b", -1.0, [[sigma_0 * np.sqrt(2), 0.0, 0.0],
                                   [sigma_1 * np.sqrt(2), 0.0, 0.0]]),
        ]

    def _atoms(self, fixed=()):
        atoms = Atoms("H2", positions=[(0, 0, 0), (1.0, 0, 0)])
        if len(fixed):
            atoms.set_constraint(FixAtoms(indices=list(fixed)))
        return atoms

    def test_a_fixed_atom_is_excluded_from_the_reported_sigma(self):
        """The disagreement lives on the fixed atom, so the free-atom sigma
        must be far smaller than the all-atom one."""
        atoms = self._atoms(fixed=[0])
        with CommitteeCalculator(self._members(0.9, 0.1)) as calc:
            calc.start()
            atoms.calc = calc
            atoms.get_potential_energy()
        assert calc.latest["sigma_max_free"] == pytest.approx(0.1, abs=1e-9)
        assert calc.latest["sigma_max_all"] == pytest.approx(0.9, abs=1e-9)
        assert calc.latest["n_free_atoms"] == 1

    def test_constraints_survive_ase_copying_the_atoms(self):
        """ASE's Calculator.calculate stores atoms.copy(); the mask is built
        from that copy, so this asserts the copy keeps the constraint."""
        atoms = self._atoms(fixed=[0])
        with CommitteeCalculator(self._members(1.0, 0.0)) as calc:
            calc.start()
            atoms.calc = calc
            atoms.get_potential_energy()
        assert calc.latest["n_free_atoms"] == 1
        assert calc.latest["free_mask"][0].tolist() == [False, False, False]

    def test_an_unconstrained_system_reports_every_atom_free(self):
        atoms = self._atoms()
        with CommitteeCalculator(self._members(1.0, 0.0)) as calc:
            calc.start()
            atoms.calc = calc
            atoms.get_potential_energy()
        assert calc.latest["n_free_atoms"] == 2
        assert calc.latest["unhandled_constraints"] == []

    def test_preflight_masks_the_same_way_calculate_does(self):
        atoms = self._atoms(fixed=[0])
        with CommitteeCalculator(self._members(0.9, 0.1)) as calc:
            calc.start()
            stats = calc.preflight(atoms)
        assert stats["sigma_max_free"] == pytest.approx(0.1, abs=1e-9)

    def test_an_unhandled_constraint_leaves_atoms_free_and_warns_once(
            self, caplog):
        """FixBondLengths projects rather than masks, so its atoms fall back
        to counted-as-free -- sigma over-reported, the safe direction. The
        warning must not repeat every evaluation: a 400-step relaxation
        would otherwise print it 400 times and bury the thing it says."""
        atoms = self._atoms()
        atoms.set_constraint(FixBondLengths([(0, 1)]))
        with CommitteeCalculator(self._members(1.0, 0.0)) as calc:
            calc.start()
            with caplog.at_level(logging.WARNING):
                calc.preflight(atoms)
                calc.preflight(atoms)
        assert calc.latest["unhandled_constraints"] == ["FixBondLengths"]
        assert calc.latest["n_free_atoms"] == 2
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
