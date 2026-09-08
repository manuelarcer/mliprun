"""Committee calculator: mean forces and per-configuration uncertainty.

Driver-side only. Nothing here runs inside an MLIP env.

Design: docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md
"""
from __future__ import annotations

import csv
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

logger = logging.getLogger(__name__)


def free_component_mask(atoms):
    """Which force components are free to move, as an ``(N, 3)`` bool array.

    Only ``FixAtoms`` and ``FixCartesian`` are masked. They are the stock ASE
    constraints whose ``adjust_forces`` is a pure component mask -- verified
    against ase 3.26.0, ``forces[index] = 0.0`` and
    ``forces[index] *= ~mask[None, :]`` respectively -- so the component they
    hold is exactly zero and dropping it from the statistic is unambiguous.

    Everything else in ``ase.constraints`` (``FixScaled``, ``FixedPlane``,
    ``FixedLine``, ``FixBondLength``) projects rather than masks, and a
    projection does not carry over to a standard deviation, which is not a
    vector. Those atoms stay free: sigma is over-reported rather than
    under-reported, and the type names are returned so the fallback reaches
    the run record instead of being silent.

    Returns
    -------
    (numpy.ndarray, list of str)
        The ``(N, 3)`` mask -- True where the component is free -- and the
        sorted names of the constraint types that were not masked.
    """
    mask = np.ones((len(atoms), 3), dtype=bool)
    unhandled = set()
    for constraint in getattr(atoms, "constraints", ()) or ():
        kind = type(constraint).__name__
        if kind == "FixAtoms":
            mask[np.asarray(constraint.index, dtype=int)] = False
        elif kind == "FixCartesian":
            mask[np.asarray(constraint.index, dtype=int)] &= ~np.asarray(
                constraint.mask, dtype=bool)
        else:
            unhandled.add(kind)
    return mask, sorted(unhandled)


def committee_statistics(energies, forces, free_mask=None) -> dict:
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

    ``sigma_max``/``sigma_mean`` are taken over free components only, using
    ``free_mask`` (see :func:`free_component_mask`). A constrained atom
    cannot move regardless of how much the members disagree about its force,
    and the convergence criterion this is compared against (ASE's ``fmax``)
    only ever looks at free atoms -- so including constrained atoms in the
    reduction would compare against the wrong population. Design note:
    docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md.

    Parameters
    ----------
    energies : array-like, shape (M,)
        One energy per member, in the members' own order.
    forces : array-like, shape (M, N, 3)
        One force array per member, same order.
    free_mask : array-like of bool, shape (N, 3), optional
        True where a force component is free to move. Defaults to all-free,
        which reproduces the previous (unmasked) behaviour exactly.

    Returns
    -------
    dict
        ``energy_mean`` (float), ``forces_mean`` (N, 3), ``sigma_per_atom``
        (N,, all-component), ``sigma_per_atom_free`` (N,, masked),
        ``sigma_max`` (float, free components only), ``sigma_mean`` (float,
        free components only), ``worst_atom`` (int, free components only),
        ``sigma_max_all`` (float, unmasked), ``sigma_mean_all`` (float,
        unmasked), ``worst_atom_all`` (int, unmasked), ``n_free_atoms``
        (int, atoms with at least one free component), ``all_constrained``
        (bool, True when no atom has a free component -- ``sigma_max`` and
        ``sigma_mean`` then fall back to the unmasked numbers).

    Raises
    ------
    ValueError
        If fewer than two members are given, the shapes disagree, or
        ``free_mask`` is not shaped ``(n_atoms, 3)``. Two members is the
        floor because ``ddof=1`` is undefined for one sample -- and because a
        committee of one has no disagreement to report.
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

    sigma_components = forces.std(axis=0, ddof=1)                 # (N, 3)
    sigma_per_atom = np.linalg.norm(sigma_components, axis=1)     # (N,)
    worst_all = int(np.argmax(sigma_per_atom))

    if free_mask is None:
        free_mask = np.ones(sigma_components.shape, dtype=bool)
    else:
        free_mask = np.asarray(free_mask, dtype=bool)
        if free_mask.shape != sigma_components.shape:
            raise ValueError(
                f"free_mask must have shape {sigma_components.shape}, got "
                f"{free_mask.shape}")

    sigma_per_atom_free = np.linalg.norm(sigma_components * free_mask, axis=1)
    movable = np.flatnonzero(free_mask.any(axis=1))

    if movable.size:
        worst = int(movable[np.argmax(sigma_per_atom_free[movable])])
        sigma_max = float(sigma_per_atom_free[worst])
        sigma_mean = float(sigma_per_atom_free[movable].mean())
        all_constrained = False
    else:
        worst = worst_all
        sigma_max = float(sigma_per_atom[worst_all])
        sigma_mean = float(sigma_per_atom.mean())
        all_constrained = True

    return {
        "energy_mean": float(energies.mean()),
        "forces_mean": forces.mean(axis=0),
        "sigma_per_atom": sigma_per_atom,
        "sigma_per_atom_free": sigma_per_atom_free,
        "sigma_max": sigma_max,
        "sigma_mean": sigma_mean,
        "worst_atom": worst,
        "sigma_max_all": float(sigma_per_atom[worst_all]),
        "sigma_mean_all": float(sigma_per_atom.mean()),
        "worst_atom_all": worst_all,
        "n_free_atoms": int(movable.size),
        "all_constrained": all_constrained,
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
        #: The ``uncertainty_summary(...)`` dict for the last relaxation
        #: driven through this committee, set by ``run_optimization`` once it
        #: finishes. A caller (the CLI's flagged-uncertainty echo) reads this
        #: back instead of recomputing it, so the printed number can never
        #: diverge from what the run record stored.
        self.latest_uncertainty_summary = None
        #: ``{member name: versions dict}`` as MEASURED inside each member's
        #: own env by ``worker._versions`` -- the interpreter, ASE, torch and
        #: MLIP package that actually loaded, which is not the same fact as
        #: what committee.yaml declared. Populated by :meth:`start`; empty
        #: until then. ``run_optimization`` merges it into the run record.
        self.member_versions: dict = {}
        self.n_evaluations = 0
        self._pool = None
        #: Distinct ``unhandled_constraints`` tuples already logged, so the
        #: warning fires once per kind of unmasked constraint rather than
        #: once per force evaluation (``_evaluate`` runs every optimizer
        #: step).
        self._warned_unhandled: set = set()

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

        The returned block is also stored on :attr:`member_versions`, because
        it is what the run record needs and every caller of this method used
        to discard the return value.
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
        self.member_versions = {m.name: dict(getattr(m, "versions", {}) or {})
                                for m in self.members}
        return self.member_versions

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

        free_mask, unhandled = free_component_mask(atoms)
        ordered = [energies[m.name] for m in self.members]
        stats = committee_statistics(ordered, stacked, free_mask=free_mask)
        # Stored, not recomputed downstream: the per-atom CSV must describe
        # the same mask the statistic used, and `atoms` has moved on by the
        # time the run writes it.
        stats["free_mask"] = free_mask
        stats["unhandled_constraints"] = unhandled
        # Once per distinct set, not once per evaluation: `_evaluate` runs on
        # every force call, so an unconditional warning would repeat itself
        # several hundred times in one relaxation and bury the thing it is
        # trying to say.
        if unhandled and tuple(unhandled) not in self._warned_unhandled:
            self._warned_unhandled.add(tuple(unhandled))
            logger.warning(
                "committee sigma: constraint type(s) %s are not masked, so "
                "their atoms are counted as free and sigma is over-reported",
                ", ".join(unhandled))
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


class CommitteeTraceWriter:
    """Append-as-you-go writer for ``<stem>_committee.csv``.

    One row per optimizer step, flushed as it is written: a committee run
    that dies at step 300 keeps its first 300 steps of disagreement data.

    Column meanings
    ---------------
    energy_mean_eV
        Mean of the members' raw energies -- the optimizer's objective. Its
        absolute value is meaningless across packages (the four-member probe
        saw a 4.81 eV spread at one fixed geometry, almost all of it
        per-model offset); differences along a trajectory are not.
    energy_spread_aligned_eV
        Standard deviation (``ddof=1``, not a range) of the members' energies
        after each member's own step-0 energy is removed. Composition is
        fixed during a relaxation, so the offset cancels exactly and this is
        a real energy uncertainty. Zero by construction on the first row.
    E_<member>_eV
        Each member's raw energy.
    sigma_max_eV_per_A, sigma_mean_eV_per_A, worst_atom
        Per-atom force disagreement, reduced -- free components only.
    sigma_max_all_eV_per_A, n_free_atoms
        The unmasked maximum (all components, constrained or not) and how
        many atoms had at least one free component, so a reader can see how
        much of the disagreement sat in a frozen region without recomputing
        anything.
    mixed_theory
        The warn-don't-refuse flag, repeated on every row so downstream
        analysis can filter on it without having read the terminal.
    """

    def __init__(self, path, member_names, mixed_theory: bool):
        member_names = list(member_names)
        for name in member_names:
            if "," in name or any(c.isspace() for c in name):
                raise ValueError(
                    f"member name {name!r} is not usable as a CSV column "
                    f"header")
        self.path = str(path)
        self.member_names = member_names
        self.mixed_theory = bool(mixed_theory)
        self.rows: list = []
        self._baseline: dict = {}
        self._fieldnames = (
            ["step", "energy_mean_eV", "energy_spread_aligned_eV"]
            + [f"E_{name}_eV" for name in member_names]
            + ["fmax_eV_per_A", "sigma_max_eV_per_A", "sigma_mean_eV_per_A",
               "worst_atom", "sigma_max_all_eV_per_A", "n_free_atoms",
               "mixed_theory"]
        )
        self._handle = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle,
                                      fieldnames=self._fieldnames)
        self._writer.writeheader()
        self._handle.flush()

    def write_step(self, step: int, latest: dict, fmax_value: float) -> dict:
        """Append one optimizer step. Returns the row it wrote."""
        energies = latest["energies"]
        if not self._baseline:
            self._baseline = dict(energies)
        row = {
            "step": int(step),
            "energy_mean_eV": float(latest["energy_mean"]),
            "energy_spread_aligned_eV": aligned_energy_spread(
                energies, self._baseline),
            "fmax_eV_per_A": float(fmax_value),
            "sigma_max_eV_per_A": float(latest["sigma_max"]),
            "sigma_mean_eV_per_A": float(latest["sigma_mean"]),
            "worst_atom": int(latest["worst_atom"]),
            "sigma_max_all_eV_per_A": float(latest["sigma_max_all"]),
            "n_free_atoms": int(latest["n_free_atoms"]),
            "mixed_theory": self.mixed_theory,
        }
        for name in self.member_names:
            row[f"E_{name}_eV"] = float(energies[name])
        self._writer.writerow(row)
        self._handle.flush()
        self.rows.append(row)
        return row

    def close(self) -> None:
        """Close the file. Idempotent."""
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None


def write_peratom_sigma(path, symbols, sigma_per_atom, sigma_free=None,
                        free_mask=None) -> None:
    """Write the final geometry's per-atom force disagreement.

    Final geometry only: a per-atom field at every step would be a large file
    for little gain, and ``worst_atom`` already traces where the disagreement
    lives during the run. This file is the most diagnostically useful output
    -- it says *which* atoms the models disagree about, which is usually the
    adsorbate or the reacting bond.

    Constrained atoms keep their row. ``sigma_eV_per_A`` is the unmasked
    value and ``sigma_free_eV_per_A`` drops the held components, so a reader
    can see for themselves how much of the disagreement sits in a region that
    cannot move. ``free_components`` is 0 for a fully fixed atom.
    """
    symbols = list(symbols)
    sigma_per_atom = np.asarray(sigma_per_atom, dtype=float).reshape(-1)
    if len(symbols) != sigma_per_atom.size:
        raise ValueError(
            f"{len(symbols)} symbols but {sigma_per_atom.size} sigma values")
    if sigma_free is None:
        sigma_free = sigma_per_atom
    sigma_free = np.asarray(sigma_free, dtype=float).reshape(-1)
    if sigma_free.size != sigma_per_atom.size:
        raise ValueError(
            f"{sigma_per_atom.size} sigma values but {sigma_free.size} "
            f"free-component sigma values")
    if free_mask is None:
        free_mask = np.ones((sigma_per_atom.size, 3), dtype=bool)
    free_mask = np.asarray(free_mask, dtype=bool)
    if free_mask.shape != (sigma_per_atom.size, 3):
        raise ValueError(
            f"free_mask must have shape {(sigma_per_atom.size, 3)}, got "
            f"{free_mask.shape}")
    n_free = free_mask.sum(axis=1)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["atom_index", "symbol", "sigma_eV_per_A",
                         "sigma_free_eV_per_A", "free_components"])
        for index, symbol in enumerate(symbols):
            writer.writerow([index, symbol, float(sigma_per_atom[index]),
                             float(sigma_free[index]), int(n_free[index])])


def uncertainty_summary(rows, latest, *, threshold: float,
                        threshold_source: str, symbols=None) -> dict:
    """Reduce a committee run to the block the run record stores.

    The flagging rule: a configuration is flagged when ``sigma_max`` at the
    **final** geometry exceeds ``threshold``. Self-scaling and physically
    motivated -- if the models disagree about the forces by more than the
    convergence tolerance, the located minimum sits inside the committee's
    own noise and the geometry is not resolved. A path that passed through a
    strained geometry but converged to a well-constrained minimum is not
    flagged, which is why the peak is reported separately.

    **The default threshold is uncalibrated.** The 2026-09-04 probe measured
    sigma_F only across four mixed-level members, so there is no same-level
    number yet. Calibrating it is a natural first use of the feature; until
    then the threshold and its source travel with the flag so a later reader
    knows what was applied.

    Parameters
    ----------
    rows : list of dict
        The trace rows, as written by :class:`CommitteeTraceWriter`.
    latest : dict or None
        The final evaluation's statistics. ``None`` when the run died before
        evaluating anything.
    threshold : float
        The sigma_max above which the configuration is flagged.
    threshold_source : {"fmax", "explicit"}
        Where the threshold came from.
    symbols : sequence of str, optional
        Chemical symbols, used to name the worst atom.
    """
    summary = {
        "n_steps": len(rows),
        "threshold_eV_per_A": float(threshold),
        "threshold_source": threshold_source,
        "sigma_max_final_eV_per_A": None,
        "sigma_mean_final_eV_per_A": None,
        "sigma_max_peak_eV_per_A": None,
        "peak_step": None,
        "worst_atom": None,
        "worst_atom_symbol": None,
        "energy_spread_aligned_final_eV": None,
        "flagged": False,
    }
    if rows:
        peak = max(rows, key=lambda r: r["sigma_max_eV_per_A"])
        summary["sigma_max_peak_eV_per_A"] = float(peak["sigma_max_eV_per_A"])
        summary["peak_step"] = int(peak["step"])
        summary["energy_spread_aligned_final_eV"] = float(
            rows[-1]["energy_spread_aligned_eV"])
    if latest is not None:
        sigma_max = float(latest["sigma_max"])
        worst = int(latest["worst_atom"])
        summary["sigma_max_final_eV_per_A"] = sigma_max
        summary["sigma_mean_final_eV_per_A"] = float(latest["sigma_mean"])
        summary["worst_atom"] = worst
        if symbols is not None and worst < len(symbols):
            summary["worst_atom_symbol"] = symbols[worst]
        summary["flagged"] = bool(sigma_max > threshold)
    return summary
