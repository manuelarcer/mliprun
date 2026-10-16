"""Parameter schema for the web app's calculation types, and its validator.

One table drives three things: the form the browser generates
(``GET /api/calc-types``), the server-side check of a submission
(:func:`validate_submission`), and the CLI flags written to
``reproduce.sh``. Defaults equal the ``mlip`` CLI defaults.
"""
from __future__ import annotations

import copy
import json
import math
from typing import Optional

SCHEMA_VERSION = 1

_ANG = "Å"


class ValidationFailed(Exception):
    """A submission broke one or more rules.

    Parameters
    ----------
    errors : list of dict
        Every problem found, each ``{"field": str, "message": str}``.
    """

    def __init__(self, errors: list):
        self.errors = list(errors)
        super().__init__(f"{len(self.errors)} problem(s) with the submission")


def _field(name, label, type, default, *, unit=None, help="", min=None,  # noqa: A002
           max=None, min_exclusive=False, max_exclusive=False, choices=None,
           nullable=False, advanced=False, visible_if=None, input_modes=None,
           cli_flag="auto") -> dict:
    if cli_flag == "auto":
        cli_flag = "--" + name.replace("_", "-")
    return {"name": name, "label": label, "type": type, "default": default,
            "unit": unit, "help": help, "min": min, "max": max,
            "min_exclusive": min_exclusive, "max_exclusive": max_exclusive,
            "choices": choices, "nullable": nullable, "advanced": advanced,
            "visible_if": visible_if, "input_modes": input_modes,
            "cli_flag": cli_flag}


_PLOT_HELP = "Also write PNG plots (CSV data is always written)."

_SINGLEPOINT = [
    _field("stress", "Stress", "choice", "auto", choices=["auto", "on", "off"],
           help="Stress tensor. auto = only when the cell is periodic in all "
                "three directions; on = always ask the calculator; off = skip.",
           cli_flag="--stress/--no-stress"),
]

_OPTIMIZE = [
    _field("optimizer", "Optimizer", "choice", "bfgs",
           choices=["bfgs", "lbfgs", "fire", "bfgsls", "gpmin", "mdmin"],
           help="ASE optimizer algorithm."),
    _field("fmax", "Force threshold", "float", 0.05, unit=f"eV/{_ANG}", min=0,
           max=10, min_exclusive=True,
           help="Converged when the largest atomic force is below this "
                "(includes the cell virials when relax_cell is on)."),
    _field("max_steps", "Max steps", "int", 200, min=0, max=100000,
           help="Stop after this many optimizer steps even if not converged."),
    _field("relax_cell", "Relax cell", "bool", False,
           help="Also relax the cell vectors (FrechetCellFilter), like VASP ISIF=3."),
    _field("plot", "Plot", "bool", False, advanced=True, help=_PLOT_HELP,
           cli_flag="--plot/--no-plot"),
]

_FREQ = [
    _field("indices", "Atoms to displace", "indices", None, nullable=True,
           help="Atom indices such as '0,1,5', '12-30' or a mix (0-based, "
                "ranges inclusive). Empty = every atom not fixed by FixAtoms."),
    _field("delta", "Displacement", "float", 0.01, unit=_ANG, min=0, max=0.1,
           min_exclusive=True, help="Finite-difference displacement."),
    _field("nfree", "Points per coordinate", "choice", 2, choices=[2, 4],
           help="2 = three-point stencil; 4 = five-point (twice the cost)."),
    _field("direction", "Direction", "choice", "central",
           choices=["central", "forward", "backward"], advanced=True,
           help="Finite-difference direction."),
    _field("method", "Hessian method", "choice", "standard",
           choices=["standard", "frederiksen"], advanced=True,
           help="How the Hessian is assembled from the displaced forces."),
    _field("write_modes", "Write mode animations", "choice", "imaginary",
           choices=["none", "imaginary", "all"],
           help="Write freq.<n>.traj animations for no modes, the imaginary "
                "ones, or all."),
    _field("imaginary_floor", "Imaginary floor", "float", 10.0,
           unit="cm⁻¹", min=0, max=1000, advanced=True,
           help="Imaginary modes smaller than this in magnitude are reported "
                "but not counted (numerical noise near zero)."),
    _field("expect_fmax", "Expected fmax", "float", None, unit=f"eV/{_ANG}",
           min=0, max=10, min_exclusive=True, nullable=True, advanced=True,
           help="Warn if the input geometry's largest free force exceeds "
                "this (a frequency run needs a relaxed structure). Never refuses."),
]

_MD = [
    _field("ensemble", "Ensemble", "choice", "nvt", choices=["nve", "nvt", "npt"],
           help="NVE starts from rest: no velocities are drawn (Maxwell-"
                "Boltzmann velocities are drawn only for NVT and NPT)."),
    _field("steps", "Steps", "int", 1000, min=1, max=10000000,
           help="Number of MD steps."),
    _field("timestep", "Timestep", "float", 1.0, unit="fs", min=0, max=10,
           min_exclusive=True, help="Integration timestep."),
    _field("temperature", "Temperature", "float", 300.0, unit="K", min=0,
           max=100000, min_exclusive=True,
           visible_if=[{"ensemble": ["nvt", "npt"]}],
           help="Target temperature; also sets the initial velocities."),
    _field("thermostat", "Thermostat", "choice", "langevin",
           choices=["langevin", "nose-hoover", "berendsen"],
           visible_if=[{"ensemble": ["nvt"]}], help="NVT thermostat."),
    # Before friction/ttime/taut: their visible_if reads it, and a controlling
    # field must precede the fields it controls.
    _field("barostat", "Barostat", "choice", "npt", choices=["npt", "berendsen"],
           visible_if=[{"ensemble": ["npt"]}],
           help="npt = ASE NPT (Melchionna); berendsen = Berendsen barostat."),
    _field("friction", "Friction", "float", 0.01, unit="1/fs", min=0, max=10,
           min_exclusive=True, advanced=True,
           visible_if=[{"ensemble": ["nvt"], "thermostat": ["langevin"]}],
           help="Langevin friction coefficient."),
    _field("ttime", "Thermostat time", "float", 25.0, unit="fs", min=0,
           max=100000, min_exclusive=True, advanced=True,
           visible_if=[{"ensemble": ["nvt"], "thermostat": ["nose-hoover"]},
                       {"ensemble": ["npt"], "barostat": ["npt"]}],
           help="Nosé-Hoover / ASE NPT thermostat time constant."),
    _field("taut", "Tau T", "float", 100.0, unit="fs", min=0, max=100000,
           min_exclusive=True, advanced=True,
           visible_if=[{"ensemble": ["nvt"], "thermostat": ["berendsen"]},
                       {"ensemble": ["npt"], "barostat": ["berendsen"]}],
           help="Berendsen temperature coupling time."),
    _field("pressure", "Pressure", "float", 0.0, unit="GPa", min=-1000, max=1000,
           visible_if=[{"ensemble": ["npt"]}], help="Target pressure."),
    _field("taup", "Tau P", "float", 1000.0, unit="fs", min=0, max=1e6,
           min_exclusive=True, advanced=True,
           visible_if=[{"ensemble": ["npt"], "barostat": ["berendsen"]}],
           help="Berendsen pressure coupling time."),
    _field("compressibility", "Compressibility", "float", 0.457, unit="1/GPa",
           min=0, max=100, min_exclusive=True, advanced=True,
           visible_if=[{"ensemble": ["npt"], "barostat": ["berendsen"]}],
           help="Berendsen barostat compressibility."),
    _field("pfactor", "pfactor", "float", None, unit="GPa·fs²", min=0,
           max=1e12, min_exclusive=True, nullable=True, advanced=True,
           visible_if=[{"ensemble": ["npt"], "barostat": ["npt"]}],
           help="ASE NPT barostat mass factor. Empty = automatic, "
                "(75 fs)² × 100 GPa."),
    _field("barostat_mask", "Barostat mask", "mask3", "1,1,1", advanced=True,
           visible_if=[{"ensemble": ["npt"]}],
           help="'x,y,z' of 0/1: which cell axes the barostat may scale."),
    _field("log_interval", "Log interval", "int", 10, min=1, max=100000,
           help="One md_step event and one md_energy.csv row every this many steps."),
    _field("traj_interval", "Trajectory interval", "int", 100, min=1, max=100000,
           help="Write a frame to md.traj every this many steps."),
    _field("plot", "Plot", "bool", False, advanced=True, help=_PLOT_HELP,
           cli_flag="--plot/--no-plot"),
]

_PAIR = ["pair"]
_IDPP = [{"interpolation": ["idpp"]}]
_ENDPOINTS = [{"optimize_endpoints": [True]}]

_NEB = [
    _field("n_images", "Intermediate images", "int", 5, min=1, max=30,
           input_modes=_PAIR, cli_flag="--num-images",
           help="Number of intermediate images (total images = n + 2)."),
    _field("interpolation", "Interpolation", "choice", "idpp",
           choices=["idpp", "linear"], input_modes=_PAIR, cli_flag=None,
           help="Initial band: IDPP (image-dependent pair potential) or "
                "plain linear interpolation."),
    _field("interp_fmax", "IDPP fmax", "float", 0.1, min=0, max=10,
           min_exclusive=True, advanced=True, visible_if=_IDPP,
           input_modes=_PAIR, help="Force threshold of the IDPP interpolation."),
    _field("interp_steps", "IDPP steps", "int", 100, min=1, max=100000,
           advanced=True, visible_if=_IDPP, input_modes=_PAIR,
           help="Step limit of the IDPP interpolation."),
    _field("fmax", "Force threshold", "float", 0.05, unit=f"eV/{_ANG}", min=0,
           max=10, min_exclusive=True, help="NEB convergence threshold."),
    _field("k", "Spring constant", "float", 0.1, unit=f"eV/{_ANG}²", min=0,
           max=100, min_exclusive=True, help="NEB spring constant."),
    _field("climb", "Climbing image", "bool", True, cli_flag="--climb/--no-climb",
           help="Climbing-image NEB (the highest image climbs to the saddle)."),
    _field("neb_optimizer", "NEB optimizer", "choice", "fire",
           choices=["fire", "mdmin", "bfgs", "lbfgs"], help="Band optimizer."),
    _field("neb_max_steps", "NEB max steps", "int", 600, min=1, max=100000,
           help="Step limit of the band optimization."),
    _field("dyneb", "DyNEB", "bool", False, advanced=True,
           cli_flag="--dyneb/--no-dyneb",
           help="Dynamic NEB: freeze images already below fmax to save force calls."),
    _field("scale_fmax", "Scale fmax", "float", 0.0, min=0, max=100,
           advanced=True, visible_if=[{"dyneb": [True]}],
           help="DyNEB only: loosen the per-image criterion with distance "
                "from the highest image (0 = uniform)."),
    _field("optimize_endpoints", "Optimize endpoints", "bool", True,
           input_modes=_PAIR,
           cli_flag="--optimize-endpoints/--no-optimize-endpoints",
           help="Relax the initial and final structures before the NEB."),
    _field("endpoint_fmax", "Endpoint fmax", "float", 0.01, unit=f"eV/{_ANG}",
           min=0, max=10, min_exclusive=True, visible_if=_ENDPOINTS,
           input_modes=_PAIR, help="Force threshold for the endpoint relaxations."),
    _field("endpoint_optimizer", "Endpoint optimizer", "choice", "bfgs",
           choices=["bfgs", "lbfgs", "fire"], visible_if=_ENDPOINTS,
           input_modes=_PAIR, help="Optimizer for the endpoint relaxations."),
    _field("endpoint_max_steps", "Endpoint max steps", "int", 200, min=1,
           max=100000, visible_if=_ENDPOINTS, input_modes=_PAIR,
           help="Step limit for each endpoint relaxation."),
    _field("relax_atoms", "Relax only atoms", "indices", None, nullable=True,
           advanced=True, input_modes=_PAIR,
           help="Only these atoms move; all others are fixed, and IDPP is "
                "skipped (highly constrained mode)."),
    _field("plot", "Plot", "bool", False, advanced=True, help=_PLOT_HELP,
           cli_flag="--plot/--no-plot"),
]

_AUTONEB = [
    _field("n_max", "Max images", "int", 9, min=3, max=50,
           help="Target number of images, endpoints included."),
    _field("fmax", "Force threshold", "float", 0.05, unit=f"eV/{_ANG}", min=0,
           max=10, min_exclusive=True, help="Convergence threshold."),
    _field("k", "Spring constant", "float", 0.1, unit=f"eV/{_ANG}²", min=0,
           max=100, min_exclusive=True, help="NEB spring constant."),
    _field("climb", "Climbing image", "bool", True, cli_flag="--climb/--no-climb",
           help="Climbing-image NEB."),
    _field("interpolate_method", "Interpolation", "choice", "idpp",
           choices=["idpp", "linear"], help="How new images are interpolated."),
    _field("space_energy_ratio", "Space/energy ratio", "float", 0.5, min=0,
           max=1, advanced=True,
           help="Where to insert images: 0 = largest energy gaps, 1 = largest "
                "geometric gaps."),
    _field("maxsteps", "Max steps per relaxation", "int", 10000, min=1,
           max=1000000, advanced=True, help="Step limit for each sub-relaxation."),
    _field("optimize_endpoints", "Optimize endpoints", "bool", True,
           cli_flag="--optimize-endpoints/--no-optimize-endpoints",
           help="Relax the initial and final structures first."),
    _field("endpoint_fmax", "Endpoint fmax", "float", 0.01, unit=f"eV/{_ANG}",
           min=0, max=10, min_exclusive=True, visible_if=_ENDPOINTS,
           help="Force threshold for the endpoint relaxations."),
    _field("endpoint_optimizer", "Endpoint optimizer", "choice", "bfgs",
           choices=["bfgs", "lbfgs", "fire"], visible_if=_ENDPOINTS,
           help="Optimizer for the endpoint relaxations."),
    _field("endpoint_max_steps", "Endpoint max steps", "int", 200, min=1,
           max=100000, visible_if=_ENDPOINTS,
           help="Step limit for each endpoint relaxation."),
]

CALC_TYPES = [
    {"name": "singlepoint", "label": "Single point",
     "description": "Energy, forces and (optionally) stress of one structure.",
     "inputs": "single", "cli_command": "singlepoint run", "fields": _SINGLEPOINT},
    {"name": "optimize", "label": "Geometry optimization",
     "description": "Relax atomic positions (optionally the cell) to a force threshold.",
     "inputs": "single", "cli_command": "optimize run", "fields": _OPTIMIZE},
    {"name": "freq", "label": "Vibrational frequencies",
     "description": "Harmonic frequencies by finite differences of forces.",
     "inputs": "single", "cli_command": "freq run", "fields": _FREQ},
    {"name": "md", "label": "Molecular dynamics",
     "description": "NVE, NVT or NPT molecular dynamics.",
     "inputs": "single", "cli_command": "md run", "fields": _MD},
    {"name": "neb", "label": "NEB",
     "description": "Nudged elastic band between two structures, or from an "
                    "uploaded band (in band mode the frames are the starting "
                    "band: no interpolation, no endpoint optimization).",
     "inputs": "pair_or_band", "cli_command": "neb run", "fields": _NEB},
    {"name": "autoneb", "label": "AutoNEB",
     "description": "NEB that inserts images automatically up to n_max.",
     "inputs": "pair", "cli_command": "autoneb run", "fields": _AUTONEB},
]

_BY_NAME = {ct["name"]: ct for ct in CALC_TYPES}


def get_calc_types() -> dict:
    """The ``GET /api/calc-types`` payload (plan section 3.1), deep-copied."""
    return {"schema_version": SCHEMA_VERSION,
            "calc_types": copy.deepcopy(CALC_TYPES)}


def get_calc_type(name: str) -> Optional[dict]:
    """The calc-type entry for ``name`` (not copied), or ``None``."""
    return _BY_NAME.get(name)


def _fmt(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _matches(value, allowed: list) -> bool:
    return any(value == a and type(value) is type(a) for a in allowed)


def _clause_holds(clause: dict, eff: dict) -> bool:
    return all(_matches(eff.get(k), allowed) for k, allowed in clause.items())


def field_applies(field: dict, eff: dict, mode: Optional[str]) -> tuple:
    """Whether ``field`` applies; when not, why (``(False, "x=y")``)."""
    modes = field.get("input_modes")
    if modes is not None and mode not in modes:
        return False, f"input mode={mode}"
    clauses = field.get("visible_if")
    if not clauses:
        return True, None
    for clause in clauses:
        if _clause_holds(clause, eff):
            return True, None
    for clause in clauses:
        for key, allowed in clause.items():
            if not _matches(eff.get(key), allowed):
                return False, f"{key}={_fmt(eff.get(key))}"
    return False, "the controlling fields' values"  # pragma: no cover


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_bounds(field: dict, value) -> Optional[str]:
    lo, hi = field.get("min"), field.get("max")
    if lo is not None:
        if field.get("min_exclusive") and not value > lo:
            return f"must be > {_fmt(lo)}"
        if not field.get("min_exclusive") and not value >= lo:
            return f"must be >= {_fmt(lo)}"
    if hi is not None:
        if field.get("max_exclusive") and not value < hi:
            return f"must be < {_fmt(hi)}"
        if not field.get("max_exclusive") and not value <= hi:
            return f"must be <= {_fmt(hi)}"
    return None


def parse_mask3(value) -> tuple:
    """Parse ``"x,y,z"`` of 0/1 into a 3-tuple of ints (raises ``ValueError``)."""
    from mliprun.core.md import normalize_barostat_mask

    if not isinstance(value, str):
        raise ValueError("must be a string 'x,y,z' of 0/1")
    parts = [p.strip() for p in value.split(",")]
    try:
        ints = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"could not parse {value!r}; expected 'x,y,z' of 0/1") from None
    return normalize_barostat_mask(ints)


def parse_index_list(text: str, natoms: int) -> list:
    """Parse an ``indices`` field value with the package's own parser."""
    from mliprun.core.vibrations import parse_indices

    return parse_indices(text, natoms)


def check_value(field: dict, value, natoms: Optional[int] = None):
    """Validate one value against its FieldSpec.

    Returns
    -------
    (normalized, error)
        ``error`` is ``None`` on success, else a message.
    """
    ftype = field["type"]
    if value is None:
        if field.get("nullable"):
            return None, None
        return None, "must not be null"
    if ftype == "float":
        if not _is_number(value):
            return None, "must be a number"
        value = float(value)
        if not math.isfinite(value):
            return None, "must be finite"
        err = _check_bounds(field, value)
        return (None, err) if err else (value, None)
    if ftype == "int":
        if isinstance(value, float) and not isinstance(value, bool) and value.is_integer():
            value = int(value)
        if not isinstance(value, int) or isinstance(value, bool):
            return None, "must be an integer"
        err = _check_bounds(field, value)
        return (None, err) if err else (value, None)
    if ftype == "bool":
        if not isinstance(value, bool):
            return None, "must be true or false"
        return value, None
    if ftype == "choice":
        for choice in field["choices"]:
            if value == choice and type(value) is type(choice):
                return choice, None
        return None, "must be one of " + ", ".join(_fmt(c) for c in field["choices"])
    if ftype == "indices":
        if not isinstance(value, str):
            return None, "must be a string such as '0,1,5' or '12-30'"
        text = value.strip()
        if natoms is None:
            return None, "cannot be checked without a structure"
        try:
            parsed = parse_index_list(text, natoms)
        except ValueError as exc:
            return None, str(exc)
        if not parsed:
            return None, "must list at least one atom index (omit the field for the default)"
        return text, None
    if ftype == "mask3":
        try:
            mask = parse_mask3(value)
        except ValueError as exc:
            return None, str(exc)
        return ",".join(str(v) for v in mask), None
    raise AssertionError(f"unknown field type {ftype}")  # pragma: no cover


def _input_mode(kind: str, inputs, errors: list) -> Optional[str]:
    if not isinstance(inputs, dict):
        errors.append({"field": "inputs", "message": "must be an object"})
        return None
    roles = set(inputs)
    if kind == "single":
        if roles == {"structure"}:
            return "single"
        errors.append({"field": "inputs", "message": "needs exactly one 'structure' input"})
        return None
    if roles == {"initial", "final"}:
        return "pair"
    if kind == "pair_or_band" and roles == {"band"}:
        return "band"
    if kind == "pair":
        errors.append({"field": "inputs", "message": "needs 'initial' and 'final' inputs"})
    else:
        errors.append({"field": "inputs",
                       "message": "needs either 'initial' + 'final' or 'band'"})
    return None


def _check_structures(mode: str, inputs: dict, errors: list) -> None:
    if mode == "pair":
        ini, fin = inputs["initial"], inputs["final"]
        if ini.get("natoms") != fin.get("natoms"):
            errors.append({"field": "inputs.final", "message":
                           f"has {fin.get('natoms')} atoms but initial has "
                           f"{ini.get('natoms')}; NEB needs the same atoms"})
        elif list(ini.get("symbols") or []) != list(fin.get("symbols") or []):
            errors.append({"field": "inputs.final", "message":
                           "atom species/order differ from initial; NEB needs "
                           "the same atoms in the same order"})
    elif mode == "band":
        band = inputs["band"]
        n = band.get("n_frames")
        if not isinstance(n, int) or n < 3:
            errors.append({"field": "inputs.band", "message":
                           f"a band needs at least 3 frames (got {n})"})
        elif n > 52:
            errors.append({"field": "inputs.band", "message":
                           f"a band may have at most 52 frames (got {n})"})
        if not band.get("frames_same_symbols", False):
            errors.append({"field": "inputs.band", "message":
                           "all frames must have the same atoms in the same order"})


def validate_submission(calc_type: str, calculator: dict, params: dict,
                        inputs: dict, catalog: Optional[dict] = None) -> dict:
    """Validate and normalize one job submission (plan section 3.3).

    Parameters
    ----------
    calc_type : str
        One of the calc-type names.
    calculator : dict
        Calculator block as submitted.
    params : dict
        Submitted parameters (hidden fields must be absent).
    inputs : dict
        Role -> ``{"natoms", "symbols", "n_frames", "pbc"}`` (plus
        ``"frames_same_symbols"`` for ``band``).
    catalog : dict, optional
        Calculator catalog of the target environment (see
        ``calculators.validate_calculator``); ``None`` checks against this
        process's own install.

    Returns
    -------
    dict
        ``{"calculator": dict, "params": dict, "param_sources": dict}``.

    Raises
    ------
    ValidationFailed
        With every problem found.
    """
    errors: list = []
    ct = _BY_NAME.get(calc_type) if isinstance(calc_type, str) else None
    if ct is None:
        raise ValidationFailed([{"field": "calc_type", "message":
                                 "must be one of " + ", ".join(_BY_NAME)}])

    mode = _input_mode(ct["inputs"], inputs, errors)
    natoms = None
    symbols: set = set()
    if mode is not None:
        first = inputs["structure"] if mode == "single" else (
            inputs["initial"] if mode == "pair" else inputs["band"])
        natoms = first.get("natoms")
        for meta in inputs.values():
            symbols.update(meta.get("symbols") or [])
        _check_structures(mode, inputs, errors)

    if params is None:
        params = {}
    norm_params: dict = {}
    sources: dict = {}
    if not isinstance(params, dict):
        errors.append({"field": "params", "message": "must be an object"})
        params = {}
    known = {f["name"] for f in ct["fields"]}
    for key in params:
        if key not in known:
            errors.append({"field": f"params.{key}", "message": "unknown parameter"})

    eff: dict = {}
    for field in ct["fields"]:
        name = field["name"]
        applies, why = field_applies(field, eff, mode)
        if not applies:
            eff[name] = field["default"]
            if name in params:
                errors.append({"field": f"params.{name}",
                               "message": f"does not apply when {why}"})
            continue
        if name in params:
            value, err = check_value(field, params[name], natoms)
            if err:
                errors.append({"field": f"params.{name}", "message": err})
                eff[name] = params[name]
                continue
            norm_params[name] = value
            sources[name] = "user"
        else:
            norm_params[name] = copy.deepcopy(field["default"])
            sources[name] = "default"
        eff[name] = norm_params[name]

    from mliprun.webapp.calculators import validate_calculator

    norm_calc, calc_errors = validate_calculator(calculator, symbols, catalog)
    errors.extend(calc_errors)

    if errors:
        raise ValidationFailed(errors)
    return {"calculator": norm_calc, "params": norm_params,
            "param_sources": sources}


def applicable_fields(calc_type: str, params: dict, mode: str) -> list:
    """The FieldSpecs that apply to a normalized ``params`` in input ``mode``."""
    out, eff = [], {}
    for field in _BY_NAME[calc_type]["fields"]:
        applies, _ = field_applies(field, eff, mode)
        eff[field["name"]] = params.get(field["name"], field["default"]) if applies \
            else field["default"]
        if applies:
            out.append(field)
    return out
