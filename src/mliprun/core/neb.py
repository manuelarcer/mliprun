"""NEB calculation engine using ASE."""
import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp
from ase.mep import NEB
from ase.mep.autoneb import AutoNEB
from ase.mep.dyneb import DyNEB
from ase.mep.neb import idpp_interpolate
from ase.optimize import FIRE, MDMin
from scipy.interpolate import make_interp_spline

from mliprun.core.run_record import RunContext, RunRecord, collect_provenance
from mliprun.core.utils import calc_fmax, resolve_device

logger = logging.getLogger(__name__)


def summarize_neb_path(energies) -> dict:
    """Barrier metrics for one converged NEB path.

    Both directions are reported: the existing convergence log records only
    ``max(E) - E[0]``, which is half the answer for any reaction you might
    want to run backwards.

    ``ts_at_endpoint`` flags a maximum sitting on the first or last image,
    which means no saddle was bracketed -- the path is monotonic and the
    "barrier" is not a transition state.
    """
    energies = [float(e) for e in energies]
    n = len(energies)
    if n == 0:
        return {"n_images": 0, "forward_barrier_eV": 0.0,
                "reverse_barrier_eV": 0.0, "reaction_energy_eV": 0.0,
                "ts_image_index": None, "ts_at_endpoint": False}
    peak = max(energies)
    idx = energies.index(peak)
    return {
        "n_images": n,
        "forward_barrier_eV": peak - energies[0],
        "reverse_barrier_eV": peak - energies[-1],
        "reaction_energy_eV": energies[-1] - energies[0],
        "ts_image_index": idx,
        "ts_at_endpoint": n > 1 and idx in (0, n - 1),
    }


class CustomNEB:
    """Custom NEB wrapper around ASE's NEB implementation.

    Parameters
    ----------
    initial : ase.Atoms
        Initial structure.
    final : ase.Atoms
        Final structure.
    num_images : int
        Number of INTERMEDIATE images (excluding initial and final).
        Total images = num_images + 2.
    interp_fmax : float
        Force threshold for IDPP interpolation.
    interp_steps : int
        Maximum steps for IDPP interpolation.
    fmax : float
        Force convergence threshold for NEB optimization.
    mlip : str
        MLIP model name.
    uma_task : str
        Task name for UMA models.
    sevennet_task : str, optional
        SevenNet inference task ("modal"). Required for multi-task SevenNet
        models; there is no default, because the tasks are independent
        fine-tunes with independent energy zeros (CANON C1/C3).
    mace_head : str
        Head name for multi-head MACE foundation models (mace-mh-*).
    output_dir : str or Path
        Output directory for results.
    relax_atoms : list[int] or None
        Atom indices to relax.  All other atoms are fixed (highly-constrained mode).
        IDPP is skipped when this is set.
    logfile : str
        Name of the log file for NEB iteration logging.
    device : str
        Compute device for the MLIP calculator ('cpu' or 'cuda').
    """

    def __init__(
        self,
        initial,
        final,
        num_images: int = 9,
        interp_fmax: float = 0.1,
        interp_steps: int = 1000,
        fmax: float = 0.05,
        mlip: str = "7net-mf-ompa",
        uma_task: Optional[str] = None,
        mace_head: Optional[str] = None,
        sevennet_task: Optional[str] = None,
        output_dir: str | Path = ".",
        relax_atoms: Optional[list[int]] = None,
        logfile: str = "neb.log",
        device: str = "cpu",
    ) -> None:
        self.initial = initial
        final.set_cell(initial.get_cell(), scale_atoms=True)
        self.final = final
        self.num_images = num_images
        self.interp_fmax = interp_fmax
        self.interp_steps = interp_steps
        self.fmax = fmax
        self.mlip = mlip
        self.uma_task = uma_task
        self.mace_head = mace_head
        self.sevennet_task = sevennet_task
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.relax_atoms = relax_atoms
        self.logfile = logfile

        if self.relax_atoms is not None:
            all_indices = set(range(len(self.initial)))
            fixed_indices = list(all_indices - set(self.relax_atoms))
            self.initial.set_constraint(FixAtoms(indices=fixed_indices))
            self.final.set_constraint(FixAtoms(indices=fixed_indices))

        self.images = self.setup_neb()

    # ------------------------------------------------------------------
    # Parameter file parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_parameters_file(params_path: Path) -> dict:
        """Parse neb_parameters.txt file into a parameter dictionary.

        Parameters
        ----------
        params_path : Path
            Path to neb_parameters.txt.

        Returns
        -------
        dict
            Parsed parameters.

        Raises
        ------
        ValueError
            If required fields are missing.
        """
        params: dict = {}

        key_parsers = {
            "MLIP model": ("mlip", str),
            "UMA task": ("uma_task", str),
            "MACE head": ("mace_head", str),
            "SevenNet task": ("sevennet_task", str),
            "Device": ("device", str),
            "Initial": ("initial", str),
            "Final": ("final", str),
            "Intermediate images": ("num_images", int),
            "Total images": ("total_images", int),
            "IDPP fmax": ("interp_fmax", lambda v: None if v == "None" else float(v)),
            "IDPP steps": ("interp_steps", lambda v: None if v == "None" else int(v)),
            "Final fmax": ("fmax", float),
            "Spring constant (k)": ("k", lambda v: None if v == "None" else float(v)),
            "Climb": ("climb", lambda v: v.lower() == "true"),
            "Dynamic NEB": ("dyneb", lambda v: v.lower() == "true"),
            "Scale fmax": ("scale_fmax", float),
            "NEB optimizer": ("neb_optimizer", lambda v: v.lower() if v != "None" else None),
            "NEB max steps": ("neb_max_steps", lambda v: None if v == "None" else int(v)),
            "Optimize endpoints": ("optimize_endpoints", lambda v: v.lower() == "true"),
            "Endpoint fmax": ("endpoint_fmax", lambda v: None if v == "None" else float(v)),
            "Endpoint optimizer": ("endpoint_optimizer", lambda v: v.lower() if v != "None" else None),
            "Endpoint max steps": ("endpoint_max_steps", lambda v: None if v == "None" else int(v)),
            "Log file": ("log", str),
            "Output dir": ("output_dir", str),
        }

        with open(params_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("=") or line.startswith("NEB Run"):
                    continue
                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if key in key_parsers:
                    param_name, converter = key_parsers[key]
                    params[param_name] = converter(value)
                elif key == "Relax atoms":
                    value = value.strip("[]")
                    if value:
                        params["relax_atoms"] = [int(x.strip()) for x in value.split(",")]
                    else:
                        params["relax_atoms"] = None

        required = ["mlip", "num_images", "fmax"]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(
                f"neb_parameters.txt is missing required fields: {missing}\n"
                "The file may be from an older version or corrupted."
            )

        return params

    # ------------------------------------------------------------------
    # Restart
    # ------------------------------------------------------------------

    @classmethod
    def load_from_restart(
        cls,
        output_dir: str | Path = ".",
        mlip: Optional[str] = None,
        uma_task: Optional[str] = None,
        mace_head: Optional[str] = None,
        sevennet_task: Optional[str] = None,
        fmax: Optional[float] = None,
        logfile: Optional[str] = None,
        k: Optional[float] = None,
        climb: Optional[bool] = None,
        neb_optimizer: Optional[str] = None,
        neb_max_steps: Optional[int] = None,
        device: Optional[str] = None,
    ) -> tuple["CustomNEB", dict]:
        """Load a CustomNEB instance from existing restart files.

        Parameters
        ----------
        output_dir : str or Path
            Directory containing A2B_full.traj and neb_parameters.txt.
        mlip, uma_task, mace_head, sevennet_task, fmax, logfile, k, climb, neb_optimizer, neb_max_steps
            Optional overrides for the corresponding saved parameters.

        Returns
        -------
        tuple[CustomNEB, dict]
            Reconstructed NEB instance and loaded parameters dict.
        """
        output_dir = Path(output_dir)

        full_traj_path = output_dir / "A2B_full.traj"
        params_path = output_dir / "neb_parameters.txt"

        if not full_traj_path.exists():
            raise FileNotFoundError(
                f"Cannot restart: A2B_full.traj not found in {output_dir}\n"
                "This file is required to load the previous NEB state."
            )
        if not params_path.exists():
            raise FileNotFoundError(
                f"Cannot restart: neb_parameters.txt not found in {output_dir}\n"
                "This file contains the structural parameters needed for restart."
            )

        params = cls._parse_parameters_file(params_path)
        num_images = params["num_images"]
        total_images = num_images + 2

        try:
            images = read(str(full_traj_path), index=f"-{total_images}:")
            if len(images) != total_images:
                raise ValueError(
                    f"Expected {total_images} images but got {len(images)} "
                    f"from A2B_full.traj."
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load images from A2B_full.traj: {e}\n"
                "The trajectory file may be corrupted or incomplete."
            ) from e

        original_mlip = params["mlip"]
        if mlip is not None and mlip != original_mlip:
            logger.warning(
                "MLIP changed from '%s' to '%s'. This will cause energy/force "
                "discontinuity in the NEB path.", original_mlip, mlip,
            )
        else:
            mlip = original_mlip

        # No "<default>" fallbacks: see the sevennet_task line below.
        uma_task = uma_task or params.get("uma_task")
        mace_head = mace_head or params.get("mace_head")
        # No `or "<default>"` fallback here, unlike the two above: a SevenNet
        # task has no default, and inventing one on restart would resume the
        # band on a different energy zero (CANON C3).
        sevennet_task = sevennet_task or params.get("sevennet_task")
        fmax = fmax if fmax is not None else params["fmax"]
        logfile = logfile or params.get("log", "neb.log")
        device = device if device is not None else params.get("device", "cpu")

        # Build instance without calling __init__
        instance = cls.__new__(cls)
        instance.initial = images[0]
        instance.final = images[-1]
        instance.num_images = num_images
        instance.interp_fmax = params.get("interp_fmax", 0.1)
        instance.interp_steps = params.get("interp_steps", 100)
        instance.fmax = fmax
        instance.mlip = mlip
        instance.uma_task = uma_task
        instance.mace_head = mace_head
        instance.sevennet_task = sevennet_task
        instance.device = device
        instance.output_dir = output_dir
        instance.logfile = logfile
        instance.relax_atoms = params.get("relax_atoms")
        instance._fix_atoms_constraint = None
        instance.images = images

        if instance.relax_atoms is not None:
            all_indices = set(range(len(instance.initial)))
            fixed_indices = list(all_indices - set(instance.relax_atoms))
            constraint = FixAtoms(indices=fixed_indices)
            for img in instance.images:
                img.set_constraint(constraint)
            instance._fix_atoms_constraint = constraint

        return instance, params

    # ------------------------------------------------------------------
    # Calculator
    # ------------------------------------------------------------------

    def setup_calculator(self, model: Optional[str] = None, uma_task: Optional[str] = None,
                         mace_head: Optional[str] = None,
                         sevennet_task: Optional[str] = None):
        """Create and return an MLIP calculator.

        Parameters
        ----------
        model : str, optional
            Model name (default: ``self.mlip``).
        uma_task : str, optional
            UMA task name (default: ``self.uma_task``).
        mace_head : str, optional
            Head name for multi-head MACE foundation models (mace-mh-*)
            (default: ``self.mace_head``).
        sevennet_task : str, optional
            SevenNet inference task (default: ``self.sevennet_task``).

        Returns
        -------
        ASE calculator instance.

        Raises
        ------
        ValueError
            If a multi-task SevenNet model is requested with no task.
        """
        model = model or self.mlip
        uma_task = uma_task or self.uma_task
        mace_head = mace_head or getattr(self, "mace_head", None)
        sevennet_task = sevennet_task or getattr(self, "sevennet_task", None)

        device = getattr(self, "device", "cpu")

        if model.startswith("7net"):
            # Table lookup and the guard come first, so the error is raised
            # at the boundary even when sevenn is not installed.
            from mliprun.cli.utils import _SEVENNET_MODELS

            tasks = _SEVENNET_MODELS.get(model, ())
            if tasks and sevennet_task is None:
                # Caught here rather than left to SevenNet: this class wires
                # its own calculators, and the library default mlip is still
                # a multi-task tag, so a direct API caller would otherwise
                # fail deep inside SevenNet with no mention of the task.
                raise ValueError(
                    f"{model} is a multi-task SevenNet model and needs an "
                    f"explicit task: its tasks are independent fine-tunes "
                    f"with independent energy zeros, so one cannot be "
                    f"guessed. Pass sevennet_task=... (CLI: --sevennet-task). "
                    f"Valid tasks: {', '.join(tasks)}."
                )

            from sevenn.calculator import SevenNetCalculator

            if sevennet_task is None:
                return SevenNetCalculator(model, device=device)
            return SevenNetCalculator(model, modal=sevennet_task, device=device)
        elif model == "mace":
            from mace.calculators import mace_mp
            return mace_mp(model="medium", device=device)
        elif model.startswith("mace-mh-"):
            if mace_head is None:
                raise ValueError(
                    f"{model} is a multi-head MACE model and needs an "
                    f"explicit head: the heads are independent fine-tunes "
                    f"with independent energy zeros, so one cannot be "
                    f"guessed. Pass mace_head=... (CLI: --mace-head)."
                )
            from mace.calculators import MACECalculator
            from mliprun.cli.utils import _ensure_mace_foundation_checkpoint
            ckpt = _ensure_mace_foundation_checkpoint(model)
            return MACECalculator(model_paths=ckpt, device=device, head=mace_head)
        elif model.startswith("uma-"):
            if uma_task is None:
                raise ValueError(
                    f"{model} is a multi-head UMA model and needs an "
                    f"explicit task: the heads are independent fine-tunes "
                    f"with independent energy zeros, so one cannot be "
                    f"guessed. Pass uma_task=... (CLI: --uma-task)."
                )
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            predictor = pretrained_mlip.get_predict_unit(model, device=device)
            return FAIRChemCalculator(predictor, task_name=uma_task)
        elif model == "chgnet":
            from chgnet.model.dynamics import CHGNetCalculator
            return CHGNetCalculator(use_device=device)
        else:
            raise ValueError(f"Unknown model: {model}")

    # ------------------------------------------------------------------
    # Endpoint optimization
    # ------------------------------------------------------------------

    def optimize_endpoints(
        self,
        endpoint_fmax: float = 0.01,
        optimizer: str = "BFGS",
        max_steps: int = 200,
    ) -> dict:
        """Optimize initial and final structures before NEB.

        Parameters
        ----------
        endpoint_fmax : float
            Force convergence threshold (eV/Ang).
        optimizer : str
            Optimizer name: ``'BFGS'``, ``'LBFGS'``, ``'FIRE'``.
        max_steps : int
            Maximum optimization steps.

        Returns
        -------
        dict
            Results with ``'initial'``, ``'final'``, ``'reaction_energy'``,
            and ``'similarity'`` keys.
        """
        from ase.optimize import BFGS, FIRE, LBFGS

        opt_map = {"bfgs": BFGS, "lbfgs": LBFGS, "fire": FIRE}
        opt_class = opt_map.get(optimizer.lower(), BFGS)

        logger.info("Optimizing endpoints (fmax=%.4f eV/Ang, optimizer=%s)", endpoint_fmax, optimizer)

        results: dict = {}

        for label, atoms, traj_name, log_name in [
            ("initial", self.initial, "initial_opt.traj", "initial_opt.log"),
            ("final", self.final, "final_opt.traj", "final_opt.log"),
        ]:
            logger.info("   Optimizing %s structure...", label)
            atoms.calc = self.setup_calculator()
            energy_before = atoms.get_potential_energy()

            opt = opt_class(
                atoms,
                trajectory=str(self.output_dir / traj_name),
                logfile=str(self.output_dir / log_name),
            )
            opt.run(fmax=endpoint_fmax, steps=max_steps)

            energy_after = atoms.get_potential_energy()
            fmax_val = calc_fmax(atoms.get_forces())
            converged = fmax_val <= endpoint_fmax

            results[label] = {
                "converged": converged,
                "energy_before": energy_before,
                "energy_after": energy_after,
                "energy_change": energy_after - energy_before,
                "steps": opt.nsteps,
            }
            logger.info(
                "      %s: %.6f -> %.6f eV (dE=%.6f, %d steps, %s)",
                label, energy_before, energy_after,
                energy_after - energy_before, opt.nsteps,
                "converged" if converged else "NOT converged",
            )

        reaction_energy = results["final"]["energy_after"] - results["initial"]["energy_after"]
        logger.info("   Reaction energy: %.6f eV", reaction_energy)
        results["reaction_energy"] = reaction_energy

        if not results["initial"]["converged"] or not results["final"]["converged"]:
            logger.warning("One or both endpoints did not converge!")

        results["similarity"] = self._check_endpoint_similarity()
        return results

    def _check_endpoint_similarity(
        self,
        displacement_threshold: float = 0.5,
        energy_threshold: float = 0.02,
    ) -> dict:
        """Check if initial and final structures are too similar after optimization.

        Parameters
        ----------
        displacement_threshold : float
            Max displacement threshold (Ang).
        energy_threshold : float
            Energy difference threshold (eV).

        Returns
        -------
        dict
            Displacement statistics and warning flags.
        """
        logger.info("Checking endpoint similarity...")

        disp = self.final.get_positions() - self.initial.get_positions()
        mic_disp, mic_dist = find_mic(disp, self.initial.cell, pbc=True)

        avg_displacement = float(mic_dist.mean())
        max_displacement = float(mic_dist.max())
        max_disp_atom = int(mic_dist.argmax())
        min_displacement = float(mic_dist.min())
        energy_diff = abs(self.final.get_potential_energy() - self.initial.get_potential_energy())

        logger.info("   Avg displacement: %.3f Ang", avg_displacement)
        logger.info("   Max displacement: %.3f Ang (atom %d)", max_displacement, max_disp_atom)
        logger.info("   Energy difference: %.6f eV", energy_diff)

        warning_reasons = []
        if energy_diff < energy_threshold:
            warning_reasons.append(
                f"energy difference ({energy_diff:.6f} eV) < {energy_threshold} eV"
            )
        if max_displacement < displacement_threshold:
            warning_reasons.append(
                f"max displacement ({max_displacement:.3f} Ang) < {displacement_threshold} Ang"
            )

        is_similar = bool(warning_reasons)
        if is_similar:
            logger.warning("Endpoints may be too similar: %s", "; ".join(warning_reasons))
        else:
            logger.info("   Endpoints are sufficiently different")

        return {
            "avg_displacement": avg_displacement,
            "max_displacement": max_displacement,
            "max_disp_atom": max_disp_atom,
            "min_displacement": min_displacement,
            "energy_diff": energy_diff,
            "is_similar": is_similar,
            "warning_reasons": warning_reasons,
        }

    # ------------------------------------------------------------------
    # NEB setup & interpolation
    # ------------------------------------------------------------------

    def setup_neb(self) -> list:
        """Set up NEB image list with linear interpolation.

        Returns
        -------
        list[ase.Atoms]
            List of images (initial + intermediates + final).
        """
        images = [self.initial]
        images += [self.initial.copy() for _ in range(self.num_images)]
        images += [self.final]

        # Store FixAtoms constraint from initial (to restore after IDPP)
        self._fix_atoms_constraint = None
        for constraint in self.initial.constraints:
            if isinstance(constraint, FixAtoms):
                self._fix_atoms_constraint = constraint
                break

        # Remove FixAtoms from intermediate images to allow interpolation
        for img in images[1:-1]:
            non_fix = [c for c in img.constraints if not isinstance(c, FixAtoms)]
            img.set_constraint(non_fix)

        neb_temp = NEB(images)
        neb_temp.interpolate(method="linear", mic=True)

        # Re-apply FixAtoms for highly-constrained mode
        if self.relax_atoms is not None:
            fix_indices = [i for i in range(len(self.initial)) if i not in self.relax_atoms]
            constraint = FixAtoms(indices=fix_indices)
            for img in images[1:-1]:
                current = list(img.constraints)
                current.append(constraint)
                img.set_constraint(current)

        return images

    def interpolate_idpp(self) -> None:
        """Run IDPP interpolation and restore FixAtoms constraints after.

        Also checks for unreasonable atomic distances after interpolation
        using asetools if available.
        """
        if self.relax_atoms is not None:
            logger.info("Skipping IDPP interpolation (highly-constrained mode).")
            return

        traj_path = self.output_dir / "idpp.traj"
        log_path = self.output_dir / "idpp.log"
        idpp_interpolate(
            self.images, traj=str(traj_path), log=str(log_path),
            fmax=self.interp_fmax, mic=True, steps=self.interp_steps,
        )

        # Restore FixAtoms constraints to intermediate images after IDPP
        if self._fix_atoms_constraint is not None:
            for img in self.images[1:-1]:
                current = list(img.constraints)
                current.append(self._fix_atoms_constraint)
                img.set_constraint(current)

        # Check atomic distances after interpolation (if asetools is available)
        self._check_interpolation_sanity()

    def _check_interpolation_sanity(self) -> None:
        """Check for unreasonable atomic distances in interpolated images."""
        try:
            from asetools.pathways.neb import check_atomic_distances
        except ImportError:
            return

        for image_index, img in enumerate(self.images[1:-1], start=1):
            close_pairs = check_atomic_distances(img)
            if close_pairs:
                logger.warning(
                    "Image %d has %d atom pair(s) too close after interpolation:",
                    image_index, len(close_pairs),
                )
                for atom_i, atom_j, dist, threshold in close_pairs:
                    logger.warning(
                        "   atoms %d-%d: %.3f Ang (threshold: %.3f Ang)",
                        atom_i, atom_j, dist, threshold,
                    )

    # ------------------------------------------------------------------
    # NEB run
    # ------------------------------------------------------------------

    def run_neb(
        self,
        optimizer=FIRE,
        trajectory: str = "A2B.traj",
        full_traj: str = "A2B_full.traj",
        climb: bool = False,
        max_steps: int = 600,
        k: Optional[float] = None,
        dyneb: bool = False,
        scale_fmax: float = 0.0,
        plot: bool = False,
        run_context: Optional[RunContext] = None,
        append: bool = False,
    ) -> list:
        """Run NEB optimization.

        Parameters
        ----------
        optimizer : ASE optimizer class
            Optimizer to use.
        trajectory : str
            Name of the final trajectory file.
        full_traj : str
            Name of the full trajectory (all steps).
        climb : bool
            Enable climbing image NEB.
        max_steps : int
            Maximum number of optimization steps.
        k : float, optional
            NEB spring constant, forwarded to ``ase.mep.NEB``. Left ``None``
            uses ASE's own default (0.1 eV/Ang^2, same as this project's CLI
            default), so omitting it reproduces the prior behaviour exactly.
        dyneb : bool
            Use ``ase.mep.dyneb.DyNEB`` instead of plain NEB: images whose
            forces are already below ``fmax`` are frozen (not recomputed)
            until a neighbouring image unconverges them, saving force calls
            in this serial loop. Same tangent method and climb handling.
        scale_fmax : float
            DyNEB only. Loosens the per-image convergence criterion with
            distance from the highest-energy image (0 = one uniform
            criterion, ASE's default).
        plot : bool
            If True, write neb_convergence.png. Defaults to False -- plotting is
            opt-in; neb_convergence.csv is always written.
        run_context : RunContext, optional
            Declares the command and where each parameter value came from.
            When omitted the record still gets written, with every parameter
            tagged ``unspecified``.
        append : bool
            If True, append a ``neb-restart`` stage to an existing
            ``mliprun_run.json`` in ``output_dir`` instead of starting a new
            record -- a plain-then-CI-NEB workflow is two invocations in the
            same directory, and each is one stage.

        Returns
        -------
        list[ase.Atoms]
            Optimized NEB images.
        """
        record = RunRecord.begin(
            self.output_dir,
            command="neb",
            stage_kind="neb-restart" if append else "neb",
            parameters={"num_images": self.num_images,
                        "uma_task": self.uma_task,
                        "sevennet_task": getattr(self, "sevennet_task", None),
                        "interp_fmax": self.interp_fmax,
                        "interp_steps": self.interp_steps},
            inputs={"n_images": len(self.images),
                    "n_atoms": len(self.images[0]) if self.images else 0},
            provenance=collect_provenance(
                mlip_model=self.mlip,
                device_requested=self.device,
                device_resolved=resolve_device(self.device),
                uma_task=self.uma_task,
                # Default None, unlike setup_calculator's "omat_pbe": the
                # calculator needs *a* usable head to run at all, but the
                # record must never invent one it cannot verify was used
                # (CANON C1 -- the head is an explicit decision, never
                # inferred). An unset attribute is recorded as unknown.
                mace_head=getattr(self, "mace_head", None),
                sevennet_task=getattr(self, "sevennet_task", None),
            ),
            run_context=run_context,
            # k, climb, max_steps, the optimizer, and fmax are arguments of
            # this call (fmax via self.fmax), not fixed properties of the
            # directory, so they belong to the stage: a CI-NEB restart
            # changes them without changing the directory. fmax in
            # particular must be per-stage -- RunRecord.begin never rewrites
            # top-level `parameters` on append, so a restart at a different
            # fmax would otherwise leave the new value unrecorded.
            stage_parameters={"climb": climb, "max_steps": max_steps, "k": k,
                              "optimizer": getattr(optimizer, "__name__", str(optimizer)),
                              "fmax": self.fmax,
                              "dyneb": dyneb, "scale_fmax": scale_fmax},
            append=append,
        )

        neb_kwargs = {"climb": climb, "method": "improvedtangent"}
        if k is not None:
            neb_kwargs["k"] = k
        if dyneb:
            # DyNEB requires its own fmax to equal the fmax handed to the
            # optimizer's run() below; both come from self.fmax.
            neb = DyNEB(self.images, fmax=self.fmax, dynamic_relaxation=True,
                        scale_fmax=scale_fmax, **neb_kwargs)
        else:
            neb = NEB(self.images, **neb_kwargs)
        for image in self.images:
            image.calc = self.setup_calculator()

        full_traj_path = self.output_dir / full_traj
        traj_writer = Trajectory(str(full_traj_path), "w")

        log_data = {"step": [], "fmax(eV/A)": [], "barrier(eV)": []}

        log_path = self.output_dir / self.logfile
        with open(log_path, "w") as log_file:

            def log_iteration():
                step = opt.nsteps
                neb_forces = neb.get_forces()
                fmax_val = calc_fmax(neb_forces)

                energies = [img.get_potential_energy() for img in neb.images]
                barrier = max(energies) - energies[0]

                log_file.write(
                    f"Step {step:4d}  Fmax: {fmax_val:.6f} eV/A  Barrier: {barrier:.6f} eV\n"
                )
                log_file.flush()

                log_data["step"].append(step)
                log_data["fmax(eV/A)"].append(fmax_val)
                log_data["barrier(eV)"].append(barrier)

                for img in neb.images:
                    traj_writer.write(img)

            opt = optimizer(neb, logfile=log_file)
            opt.attach(log_iteration, interval=1)
            try:
                converged = opt.run(fmax=self.fmax, steps=max_steps)
            except Exception as exc:
                traj_writer.close()
                record.complete(status="failed", results={"error": str(exc)})
                raise

        traj_writer.close()

        # Save convergence data
        csv_path = self.output_dir / "neb_convergence.csv"
        df_conv = pd.DataFrame(log_data)
        df_conv.to_csv(csv_path, index=False)
        if plot:
            self._plot_convergence(df_conv)

        # Ensure final energies are computed
        for image in self.images:
            image.get_potential_energy()

        traj_path = self.output_dir / trajectory
        with Trajectory(str(traj_path), "w") as traj:
            for img in self.images:
                traj.write(img)

        final_energies = [img.get_potential_energy() for img in self.images]
        results = summarize_neb_path(final_energies)
        results["final_fmax_eV_per_A"] = (
            float(log_data["fmax(eV/A)"][-1]) if log_data["fmax(eV/A)"] else None
        )
        record.complete(
            status="converged" if converged else "not_converged",
            steps=int(opt.nsteps),
            results=results,
        )

        return self.images

    # ------------------------------------------------------------------
    # AutoNEB
    # ------------------------------------------------------------------

    def run_autoneb(
        self,
        n_simul: int = 1,
        n_max: int = 9,
        k: float = 0.1,
        climb: bool = True,
        optimizer=FIRE,
        space_energy_ratio: float = 0.5,
        interpolate_method: str = "idpp",
        maxsteps: int = 10000,
        prefix: str = "autoneb",
        run_context: Optional[RunContext] = None,
    ) -> None:
        """Run AutoNEB calculation.

        Parameters
        ----------
        n_simul : int
            Number of parallel relaxations.
        n_max : int
            Target number of images including endpoints.
        k : float
            Spring constant.
        climb : bool
            Enable climbing image.
        optimizer : ASE optimizer class
            Optimizer to use.
        space_energy_ratio : float
            Preference for geometric (1.0) vs energy (0.0) gaps.
        interpolate_method : str
            ``'linear'`` or ``'idpp'``.
        maxsteps : int
            Maximum steps per relaxation.
        prefix : str
            Prefix for output files.
        run_context : RunContext, optional
            Declares the command and where each parameter value came from.
            When omitted the record still gets written, with every parameter
            tagged ``unspecified``.
        """
        if self.relax_atoms is not None:
            logger.warning("AutoNEB with relax_atoms (highly-constrained mode) may not work as expected.")

        # Opened before the chdir below: self.output_dir may be a relative
        # path, and RunRecord.begin() resolves it against the current
        # working directory at the moment it is called.
        record = RunRecord.begin(
            self.output_dir,
            command="autoneb",
            stage_kind="autoneb",
            parameters={"n_max": n_max, "climb": climb, "fmax": self.fmax,
                        "k": k, "maxsteps": maxsteps,
                        "uma_task": self.uma_task,
                        "sevennet_task": getattr(self, "sevennet_task", None)},
            inputs={"n_images": len(self.images),
                    "n_atoms": len(self.images[0]) if self.images else 0},
            provenance=collect_provenance(
                mlip_model=self.mlip,
                device_requested=self.device,
                device_resolved=resolve_device(self.device),
                uma_task=self.uma_task,
                # Default None, unlike setup_calculator's "omat_pbe" -- see
                # the same call in run_neb: a record must not invent a head.
                mace_head=getattr(self, "mace_head", None),
                sevennet_task=getattr(self, "sevennet_task", None),
            ),
            run_context=run_context,
        )

        original_cwd = os.getcwd()
        os.chdir(self.output_dir)

        # Clean up previous AutoNEB files
        for old_file in glob.glob(f"{prefix}*.traj"):
            os.remove(old_file)
        if os.path.exists("AutoNEB_iter"):
            shutil.rmtree("AutoNEB_iter")

        try:
            def attach_calculators(images):
                for image in images:
                    image.calc = self.setup_calculator()

            # Wrap final structure for MIC
            final_wrapped = self.final.copy()
            disp = final_wrapped.get_positions() - self.initial.get_positions()
            mic_disp, mic_dist = find_mic(disp, self.initial.cell, pbc=True)
            final_wrapped.set_positions(self.initial.get_positions() + mic_disp)

            max_diff = np.max(np.abs(disp - mic_disp))
            if max_diff > 0.1:
                logger.info("Applied MIC wrapping (max correction: %.3f Ang)", max_diff)

            initial_images = [self.initial, final_wrapped]
            for img in initial_images:
                img.calc = self.setup_calculator()
                img.get_potential_energy()

            for i, img in enumerate(initial_images):
                write(f"{prefix}{i:03d}.traj", img)

            logger.info("Starting AutoNEB (n_max=%d, fmax=%.4f, climb=%s)", n_max, self.fmax, climb)

            autoneb = AutoNEB(
                attach_calculators=attach_calculators, prefix=prefix,
                n_simul=n_simul, n_max=n_max, fmax=self.fmax,
                maxsteps=maxsteps, k=k, climb=climb, optimizer=optimizer,
                space_energy_ratio=space_energy_ratio,
                interpolate_method=interpolate_method,
            )
            try:
                final_images = autoneb.run()
            except Exception as exc:
                # Restore cwd first: record.path was built from
                # self.output_dir while cwd was still original_cwd (see the
                # comment above), so it must be written back from the same
                # cwd or a relative output_dir would resolve to the wrong
                # place.
                os.chdir(original_cwd)
                record.complete(status="failed", results={"error": str(exc)})
                raise

            logger.info("AutoNEB calculation complete")
        finally:
            os.chdir(original_cwd)

        # AutoNEB.run() returns the converged path (self.all_images, each
        # image carrying a SinglePointCalculator snapshot of its final
        # energy) -- NOT self.images, which is CustomNEB's own placeholder
        # list built from a dummy num_images in __init__ and never touched
        # by AutoNEB.
        final_energies = [img.get_potential_energy() for img in final_images]
        record.complete(status="converged", results=summarize_neb_path(final_energies))

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _plot_convergence(self, df: pd.DataFrame) -> None:
        """Plot NEB convergence (force and barrier vs steps)."""
        fig_path = self.output_dir / "neb_convergence.png"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        ax1.plot(df["step"], df["fmax(eV/A)"], marker="o", markersize=4, linewidth=1.5, color="orange")
        ax1.axhline(y=self.fmax, color="r", linestyle="--", label=f"fmax target = {self.fmax}")
        ax1.set_xlabel("NEB Step")
        ax1.set_ylabel("Max Force (eV/Ang)")
        ax1.set_title("NEB Force Convergence")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        ax2.plot(df["step"], df["barrier(eV)"], marker="o", markersize=4, linewidth=1.5)
        ax2.set_xlabel("NEB Step")
        ax2.set_ylabel("Barrier Height (eV)")
        ax2.set_title("Barrier Height Evolution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

    def process_results(self) -> pd.DataFrame:
        """Extract energies from NEB images and save to CSV.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``'image_index'``, ``'energy'``, ``'relative_energy'``.
        """
        results = {"image_index": [], "energy": []}
        for image_index, image in enumerate(self.images):
            results["image_index"].append(image_index)
            results["energy"].append(image.get_potential_energy())

        df = pd.DataFrame(results)
        df["relative_energy"] = df["energy"] - df["energy"].iloc[0]
        df.to_csv(self.output_dir / "neb_data.csv", index=False)
        return df

    def plot_results(self, df: pd.DataFrame) -> None:
        """Plot NEB energy profile.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame from :meth:`process_results`.
        """
        fig_path = self.output_dir / "neb_energy.png"
        x = df["image_index"]
        y = df["relative_energy"]

        x_smooth = np.linspace(x.min(), x.max(), 200)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_smooth)

        plt.figure(figsize=(6, 4))
        plt.plot(x_smooth, y_smooth, label="NEB Path", linewidth=2)
        plt.scatter(x, y, color="blue", zorder=5)
        plt.xlabel("Reaction Coordinate (Image Index)")
        plt.ylabel("Relative Energy (eV)")
        plt.title(f"NEB Energy Profile ({self.mlip})")
        plt.grid(True)

        barrier = y.max()
        barrier_idx = y.idxmax()
        plt.annotate(
            f"Barrier: {barrier:.3f} eV",
            xy=(x[barrier_idx], barrier),
            xytext=(x[barrier_idx], barrier + 0.1),
            arrowprops=dict(arrowstyle="->"),
            fontsize=9,
        )

        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

    def export_poscars(self) -> None:
        """Export NEB images as VASP POSCAR files in numbered directories."""
        for image_index, image in enumerate(self.images):
            folder = self.output_dir / f"{image_index:02d}"
            folder.mkdir(exist_ok=True)
            write_vasp(folder / "POSCAR", image, direct=True, vasp5=True, sort=True)
