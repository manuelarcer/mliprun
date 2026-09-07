"""Committee calculator: mean forces and per-configuration uncertainty.

Driver-side only. Nothing here runs inside an MLIP env.

Design: docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

logger = logging.getLogger(__name__)


def committee_statistics(energies, forces) -> dict:
    """Reduce the members' energies and forces to a consensus and a spread.

    The uncertainty metric is the per-atom force disagreement

        sigma_i = || std_across_members( F_i ) ||

    -- the vector norm of the per-component standard deviation across
    members, ``ddof=1``. It behaves as an extrapolation detector: in the
    2026-09-04 probe it sat at 0.25-0.62 eV/A across the physical range of a
    CO height scan and rose to 3.48 eV/A at a strained geometry.

    The committee is a *proper potential*: the mean force is exactly the
    negative gradient of the mean energy, so the effective PES is well
    defined and line-search optimizers (``bfgsls``) behave correctly through
    it. Per-model constant offsets shift the mean energy by a constant
    without changing its shape.

    Parameters
    ----------
    energies : array-like, shape (M,)
        One energy per member, in the members' own order.
    forces : array-like, shape (M, N, 3)
        One force array per member, same order.

    Returns
    -------
    dict
        ``energy_mean`` (float), ``forces_mean`` (N, 3), ``sigma_per_atom``
        (N,), ``sigma_max`` (float), ``sigma_mean`` (float), ``worst_atom``
        (int).

    Raises
    ------
    ValueError
        If fewer than two members are given, or the shapes disagree. Two is
        the floor because ``ddof=1`` is undefined for one sample -- and
        because a committee of one has no disagreement to report.
    """
    energies = np.asarray(energies, dtype=float)
    forces = np.asarray(forces, dtype=float)
    if energies.ndim != 1:
        raise ValueError(f"energies must be 1-D, got shape {energies.shape}")
    if forces.ndim != 3 or forces.shape[2] != 3:
        raise ValueError(
            f"forces must have shape (n_members, n_atoms, 3), got {forces.shape}")
    if forces.shape[0] != energies.shape[0]:
        raise ValueError(
            f"{energies.shape[0]} energies but {forces.shape[0]} force arrays")
    if energies.shape[0] < 2:
        raise ValueError("a committee needs at least two members")

    sigma_components = forces.std(axis=0, ddof=1)          # (N, 3)
    sigma_per_atom = np.linalg.norm(sigma_components, axis=1)   # (N,)
    worst_atom = int(np.argmax(sigma_per_atom))
    return {
        "energy_mean": float(energies.mean()),
        "forces_mean": forces.mean(axis=0),
        "sigma_per_atom": sigma_per_atom,
        "sigma_max": float(sigma_per_atom[worst_atom]),
        "sigma_mean": float(sigma_per_atom.mean()),
        "worst_atom": worst_atom,
    }


def aligned_energy_spread(energies: dict, baseline: dict) -> float:
    """Spread of the members' energies after removing their own offsets.

    Raw energies are not comparable across packages -- the four-member probe
    saw a 4.81 eV spread at one fixed geometry, essentially all of it
    per-model constant offset. Subtracting each member's own step-0 energy
    collapsed that to 0.09 eV. Composition is fixed during a relaxation, so
    the offset cancels exactly and what remains is a real energy uncertainty.

    Returned as a standard deviation across members (``ddof=1``), matching
    the force metric -- not a max-minus-min range.

    Raises
    ------
    KeyError
        If a member present in ``energies`` has no baseline.
    ValueError
        If fewer than two members are given.
    """
    if len(energies) < 2:
        raise ValueError("a committee needs at least two members")
    deltas = [energies[name] - baseline[name] for name in energies]
    return float(np.std(np.asarray(deltas, dtype=float), ddof=1))


class CommitteeError(RuntimeError):
    """The committee could not produce a consensus for this geometry."""


class CommitteeCalculator(Calculator):
    """ASE calculator backed by N MLIPs in N separate environments.

    Everything downstream -- ``optimize`` today, MD and NEB later -- only
    ever touches ``atoms.calc``, so no engine needs restructuring to gain a
    committee. The 2026-09-04 probe confirmed a stock ASE ``BFGS`` relaxes
    through this unchanged, in a driver process that imports no torch,
    fairchem, mace, sevenn or chgnet.

    Cost is ``max(member)``, not ``sum(member)``: members are queried
    concurrently. The probe measured 211.7 ms for a four-member step against
    a slowest member of 206.7 ms -- 5.0 ms (2.4%) of inter-process overhead
    on a 992-byte payload, roughly constant in system size.

    Parameters
    ----------
    members : sequence
        Member handles. Anything with ``name``, ``start()``, ``calculate()``
        and ``close()`` works; in production these are
        :class:`~mliprun.core.committee.remote.RemoteMember`.
    mixed_theory : bool
        Whether the members span more than one level of theory (or any
        unrecognised one). Carried, not enforced: mixed levels warn, they do
        not refuse (Juan's call, 2026-09-04). A mixed committee's spread is a
        functional comparison, not an error bar -- the probe measured 0.363
        eV across RPBE and PBE members against 0.001-0.019 eV within a level.
    levels : sequence of str
        The distinct level-of-theory labels present, for the run record.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, members, *, mixed_theory: bool = False, levels=(),
                 **kwargs):
        super().__init__(**kwargs)
        members = list(members)
        if len(members) < 2:
            raise ValueError(
                f"a committee needs at least two members; got {len(members)}")
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate member names: {names}")
        self.members = members
        self.mixed_theory = bool(mixed_theory)
        self.levels = tuple(levels)
        #: Statistics from the most recent evaluation, plus a per-member
        #: ``energies`` dict. The caller reads this to build its trace.
        self.latest = None
        self.n_evaluations = 0
        self._pool = None

    @property
    def member_names(self) -> list:
        return [m.name for m in self.members]

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> dict:
        """Load every member's model. Returns ``{name: versions}``.

        Sequential, not concurrent: loads are one-time (22.8 s for UMA in the
        probe) and serialising them keeps peak GPU memory predictable when
        several members share a device. A member that fails to load closes
        every member -- including the one that just failed -- so a failed
        startup leaves no orphans holding CUDA contexts.
        """
        started = []
        try:
            for member in self.members:
                member.start()
                started.append(member)
        except Exception:
            self.close()
            raise
        self._pool = ThreadPoolExecutor(max_workers=len(self.members),
                                        thread_name_prefix="committee")
        return {m.name: getattr(m, "versions", {}) for m in self.members}

    def close(self) -> None:
        """Tear every member down. Idempotent.

        The pool is shut down with ``wait=False`` first, so a thread blocked
        reading from a hung member does not hold teardown up: closing the
        member closes the pipe, which unblocks that read.
        """
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        for member in self.members:
            try:
                member.close()
            except Exception as exc:  # noqa: BLE001 -- keep closing the rest
                logger.warning("could not close committee member '%s': %s",
                               member.name, exc)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    # -- evaluation --------------------------------------------------------

    def preflight(self, atoms) -> dict:
        """Evaluate one geometry on every member before the optimizer starts.

        Members disagree about what input is valid -- fairchem's UMA
        calculator raises ``MixedPBCError`` on a ``pbc=(True, True, False)``
        slab that MACE, SevenNet and CHGNet all accept. Surfacing that in the
        first seconds beats discovering it on step 400 of an overnight run.
        """
        return self._evaluate(atoms)

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        stats = self._evaluate(self.atoms)
        self.results["energy"] = stats["energy_mean"]
        # ASE asks for free_energy on some paths; for these potentials it is
        # the same number.
        self.results["free_energy"] = stats["energy_mean"]
        self.results["forces"] = stats["forces_mean"]

    def _evaluate(self, atoms) -> dict:
        if self._pool is None:
            raise CommitteeError(
                "committee has not been started: call start() before use")

        numbers = [int(z) for z in atoms.get_atomic_numbers()]
        positions = atoms.get_positions().tolist()
        cell = np.asarray(atoms.get_cell()).tolist()
        pbc = [bool(p) for p in atoms.get_pbc()]

        futures = {m.name: self._pool.submit(m.calculate, numbers, positions,
                                             cell, pbc)
                   for m in self.members}
        energies, forces, failures = {}, {}, []
        for name, future in futures.items():
            try:
                energies[name], forces[name] = future.result()
            except Exception as exc:  # noqa: BLE001 -- collect them all
                failures.append(exc)

        if failures:
            detail = "; ".join(str(exc) for exc in failures)
            self.close()
            raise CommitteeError(f"committee incomplete: {detail}")

        try:
            stacked = self._validate(energies, forces, len(numbers))
        except CommitteeError:
            self.close()
            raise

        ordered = [energies[m.name] for m in self.members]
        stats = committee_statistics(ordered, stacked)
        stats["energies"] = dict(energies)
        self.latest = stats
        self.n_evaluations += 1
        return stats

    def _validate(self, energies: dict, forces: dict, n_atoms: int):
        """Check every member's reply before it enters the mean.

        A NaN or a wrong-shaped array must be caught and *named* here: once
        averaged, a non-finite value is anonymous and poisons every number
        downstream.
        """
        stacked = []
        for member in self.members:
            array = np.asarray(forces[member.name], dtype=float)
            if array.shape != (n_atoms, 3):
                raise CommitteeError(
                    f"member '{member.name}' returned forces of shape "
                    f"{array.shape}, expected {(n_atoms, 3)}")
            if not np.isfinite(array).all():
                raise CommitteeError(
                    f"member '{member.name}' returned a non-finite force")
            if not np.isfinite(energies[member.name]):
                raise CommitteeError(
                    f"member '{member.name}' returned a non-finite energy")
            stacked.append(array)
        return np.stack(stacked)
