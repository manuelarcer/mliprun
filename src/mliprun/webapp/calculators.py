"""Which calculators the web app offers, and how the worker builds them.

The list comes from the package's own install detection
(``mliprun.cli.utils``), which reads package metadata only and never imports
an MLIP or torch. ``emt`` (ASE's built-in EMT potential) is always offered,
for demos and tests.

Device detection (is CUDA usable?) needs torch, so it runs in a child
process: the server must never import torch itself.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from typing import Optional

#: Elements ASE's EMT potential has parameters for.
EMT_ELEMENTS = ("Ag", "Al", "Au", "C", "Cu", "H", "N", "Ni", "O", "Pd", "Pt")

DEVICE_CHOICES = ("auto", "cpu", "cuda")
TASK_KEYS = ("uma_task", "mace_head", "sevennet_task")

_UMA_LABELS = {
    "uma-s-1p2": "UMA small 1.2",
    "uma-s-1p1": "UMA small 1.1",
    "uma-m-1p1": "UMA medium 1.1",
}

_DEVICE_LOCK = threading.Lock()
_DEVICE_CACHE: Optional[dict] = None


def _utils():
    from mliprun.cli import utils
    return utils


def _torch_installed() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def detect_devices(timeout: float = 60.0) -> dict:
    """Find out whether torch is installed and CUDA usable, once per process.

    torch is probed in a child interpreter (``sys.executable``) so this
    process never imports it. The result is cached for the process lifetime.

    Parameters
    ----------
    timeout : float
        Seconds to wait for the child.

    Returns
    -------
    dict
        ``{"torch_installed": bool, "cuda_available": bool | None}``;
        ``cuda_available`` is ``None`` when the probe failed.
    """
    global _DEVICE_CACHE
    with _DEVICE_LOCK:
        if _DEVICE_CACHE is not None:
            return dict(_DEVICE_CACHE)
        if not _torch_installed():
            result = {"torch_installed": False, "cuda_available": False}
        else:
            cuda = None
            try:
                proc = subprocess.run(
                    [sys.executable, "-c",
                     "import torch, json; print(json.dumps(torch.cuda.is_available()))"],
                    capture_output=True, text=True, timeout=timeout)
                if proc.returncode == 0:
                    lines = proc.stdout.strip().splitlines()
                    value = json.loads(lines[-1]) if lines else None
                    cuda = value if isinstance(value, bool) else None
            except (OSError, subprocess.SubprocessError, ValueError):
                cuda = None
            result = {"torch_installed": True, "cuda_available": cuda}
        _DEVICE_CACHE = result
        return dict(result)


def device_status() -> dict:
    """The cached device facts, without blocking (``cuda_available`` may be ``None``)."""
    cached = _DEVICE_CACHE
    if cached is not None:
        return dict(cached)
    installed = _torch_installed()
    return {"torch_installed": installed,
            "cuda_available": None if installed else False}


def _calculator_entries() -> list:
    u = _utils()
    entries = []
    if u.FAIRCHEM_AVAILABLE:
        for tag in u._KNOWN_UMA_MODELS:
            entries.append({
                "name": tag,
                "label": f"{_UMA_LABELS.get(tag, tag)} (requires Hugging Face access)",
                "family": "uma", "task_field": "uma_task",
                "task_choices": list(u._UMA_TASKS), "task_required": True,
                "elements": None, "cli_reproducible": True,
                "note": "UMA is gated: request access on Hugging Face first "
                        "(docs/UMA_USAGE_GUIDE.md). The task is required: "
                        "tasks are independent fine-tunes with their own "
                        "energy zero."})
    if u.MACE_AVAILABLE:
        entries.append({
            "name": "mace", "label": "MACE-MP-0 medium", "family": "mace",
            "task_field": None, "task_choices": [], "task_required": False,
            "elements": None, "cli_reproducible": True,
            "note": "Weights download to ~/.cache/mace on first use."})
        for tag in ("mace-mh-0", "mace-mh-1"):
            entries.append({
                "name": tag, "label": f"MACE multi-head ({tag})",
                "family": "mace", "task_field": "mace_head",
                "task_choices": list(u._MACE_MH_HEADS), "task_required": True,
                "elements": None, "cli_reproducible": True,
                "note": "The head is required: heads are independent "
                        "fine-tunes with their own energy zero."})
    if u.SEVENN_AVAILABLE:
        for tag, tasks in u._SEVENNET_MODELS.items():
            entries.append({
                "name": tag, "label": f"SevenNet {tag}", "family": "sevenn",
                "task_field": "sevennet_task", "task_choices": list(tasks),
                "task_required": bool(tasks), "elements": None,
                "cli_reproducible": True,
                "note": ("Multi-task model: the task is required (case-sensitive)."
                         if tasks else "Single-task model: no task to choose.")})
    if u.CHGNET_AVAILABLE:
        entries.append({
            "name": "chgnet", "label": "CHGNet", "family": "chgnet",
            "task_field": None, "task_choices": [], "task_required": False,
            "elements": None, "cli_reproducible": True, "note": ""})
    entries.append({
        "name": "emt", "label": "EMT (ASE built-in; demo/testing only)",
        "family": "emt", "task_field": None, "task_choices": [],
        "task_required": False, "elements": list(EMT_ELEMENTS),
        "cli_reproducible": False, "note": "Not an mlip CLI model."})
    return entries


def exit_exception_types() -> tuple:
    """The exception classes the package raises to stop a command.

    ``typer.Exit`` is its own ``RuntimeError`` subclass in current typer (it
    vendors click), not ``click.exceptions.Exit``, so both are caught.
    """
    import typer

    types = [typer.Exit]
    try:
        import click

        types.append(click.exceptions.Exit)
    except ImportError:  # pragma: no cover - click ships with typer
        pass
    return tuple(types)


def default_calculator() -> str:
    """``detect_mlip()`` when an MLIP is installed, else ``"emt"``."""
    try:
        return _utils().detect_mlip()
    except exit_exception_types():
        return "emt"


def list_calculators() -> dict:
    """The ``GET /api/calculators`` payload (plan section 7.1). Never blocks."""
    devices = device_status()
    return {
        "calculators": _calculator_entries(),
        "default": default_calculator(),
        "devices": {"choices": list(DEVICE_CHOICES), "default": "auto",
                    "torch_installed": devices["torch_installed"],
                    "cuda_available": devices["cuda_available"]},
        "note": "Restart the server after installing an MLIP package.",
    }


def get_calculator_entry(name: str) -> Optional[dict]:
    """The listed entry for ``name``, or ``None`` when it is not offered."""
    for entry in _calculator_entries():
        if entry["name"] == name:
            return entry
    return None


def _known_family_label(name: str) -> Optional[str]:
    """Label for a known tag (installed or not), for the install hint."""
    u = _utils()
    if name in u._KNOWN_UMA_MODELS:
        return "UMA"
    if name in ("mace", "mace-mh-0", "mace-mh-1"):
        return "MACE"
    if name in u._SEVENNET_MODELS:
        return "SevenNet"
    if name == "chgnet":
        return "CHGNet"
    return None


def validate_calculator(calculator, symbols: Optional[set] = None,
                        catalog: Optional[dict] = None):
    """Check a submitted calculator block (plan section 7.2).

    Parameters
    ----------
    calculator : dict
        ``{"name", "device"?, "uma_task"?, "mace_head"?, "sevennet_task"?}``.
    symbols : set of str, optional
        Every element in the job's inputs, for the EMT element check.
    catalog : dict, optional
        A :func:`list_calculators`-shaped payload (a remote environment's
        probe result, plus optional ``"env_name"`` for messages). When given,
        the offered entries and the CUDA fact come from it instead of this
        process; ``None`` keeps the local behaviour exactly.

    Returns
    -------
    (dict, list)
        The normalized calculator (all five keys) and a list of
        ``{"field", "message"}`` errors.
    """
    errors = []
    norm = {"name": None, "device": "auto", "uma_task": None,
            "mace_head": None, "sevennet_task": None}
    if not isinstance(calculator, dict):
        return norm, [{"field": "calculator", "message": "must be an object"}]
    for key in calculator:
        if key not in norm:
            errors.append({"field": f"calculator.{key}", "message": "unknown key"})

    name = calculator.get("name")
    if not isinstance(name, str) or not name:
        errors.append({"field": "calculator.name", "message": "required"})
        return norm, errors
    norm["name"] = name
    if catalog is not None:
        entries = list(catalog.get("calculators") or [])
        entry = next((e for e in entries if e.get("name") == name), None)
        cuda_available = (catalog.get("devices") or {}).get("cuda_available")
    else:
        entry = get_calculator_entry(name)
        cuda_available = None
    if entry is None:
        family = _known_family_label(name) if catalog is None else None
        if catalog is not None:
            env_name = catalog.get("env_name") or "this environment"
            offered = ", ".join(str(e.get("name")) for e in entries)
            msg = f"{name} is not installed in environment {env_name}; offered there: {offered}"
        elif family is not None:
            msg = _utils()._install_message(family, name)
        else:
            offered = ", ".join(e["name"] for e in _calculator_entries())
            msg = f"unknown calculator {name!r}; offered here: {offered}"
        errors.append({"field": "calculator.name", "message": msg})
        return norm, errors

    device = calculator.get("device", "auto")
    if device is None:
        device = "auto"
    if device not in DEVICE_CHOICES:
        errors.append({"field": "calculator.device",
                       "message": f"must be one of {', '.join(DEVICE_CHOICES)}"})
    else:
        norm["device"] = device
        if catalog is None:
            cuda_available = device_status()["cuda_available"]
        if device == "cuda" and name != "emt" and cuda_available is False:
            errors.append({"field": "calculator.device",
                           "message": "CUDA is not available on this machine"})

    task_field = entry["task_field"]
    for key in TASK_KEYS:
        value = calculator.get(key)
        if key == task_field and entry["task_choices"]:
            if value is None:
                if entry["task_required"]:
                    errors.append({"field": f"calculator.{key}", "message":
                                   f"required for {name}; one of "
                                   f"{', '.join(entry['task_choices'])}"})
            elif value not in entry["task_choices"]:
                errors.append({"field": f"calculator.{key}", "message":
                               f"must be one of {', '.join(entry['task_choices'])} "
                               f"(case-sensitive)"})
            else:
                norm[key] = value
        elif value is not None:
            why = ("is a single-task model" if key == task_field
                   else "takes no " + key.replace("_", " "))
            errors.append({"field": f"calculator.{key}",
                           "message": f"does not apply: {name} {why}"})

    if name == "emt" and symbols:
        bad = sorted(set(symbols) - set(EMT_ELEMENTS))
        if bad:
            errors.append({"field": "calculator.name", "message":
                           f"EMT has no parameters for {', '.join(bad)}; it "
                           f"supports only {', '.join(EMT_ELEMENTS)}"})
    return norm, errors


def resolved_device(spec: dict) -> str:
    """The device actually used: ``cpu`` for EMT, else ``resolve_device``."""
    if spec.get("name") == "emt":
        return "cpu"
    from mliprun.core.utils import resolve_device

    return resolve_device(spec.get("device") or "auto")


def _exit_message(exc) -> str:
    code = getattr(exc, "exit_code", None)
    if isinstance(code, str):
        return code
    return f"exited with code {code}"


def check_mlip(spec: dict) -> None:
    """Run the package's ``validate_mlip`` for ``spec``; raise ``RuntimeError``.

    No-op for ``emt``.
    """
    if spec.get("name") == "emt":
        return
    try:
        _utils().validate_mlip(spec["name"], sevennet_task=spec.get("sevennet_task"),
                               uma_task=spec.get("uma_task"),
                               mace_head=spec.get("mace_head"))
    except exit_exception_types() as exc:
        raise RuntimeError(_exit_message(exc)) from None


def make_calculator(spec: dict):
    """Build the ASE calculator for a normalized calculator spec (worker side).

    Parameters
    ----------
    spec : dict
        Normalized calculator block from ``job.json``.

    Returns
    -------
    ase.calculators.calculator.Calculator

    Raises
    ------
    RuntimeError
        When the package refuses the model (its ``typer.Exit`` converted).
    """
    if spec.get("name") == "emt":
        from ase.calculators.emt import EMT

        return EMT()
    check_mlip(spec)
    try:
        calc = _utils().build_calculator(
            spec["name"], uma_task=spec.get("uma_task"),
            device=resolved_device(spec), mace_head=spec.get("mace_head"),
            sevennet_task=spec.get("sevennet_task"))
    except exit_exception_types() as exc:
        raise RuntimeError(_exit_message(exc)) from None
    if calc is None:
        raise RuntimeError(f"unknown MLIP tag {spec['name']!r}")
    return calc
