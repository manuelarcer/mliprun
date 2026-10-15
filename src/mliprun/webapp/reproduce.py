"""``reproduce.sh`` / ``reproduce.json``: the ``mlip`` CLI call equivalent to a job.

Every applicable parameter is passed explicitly, defaults included, so the
script does not depend on future CLI default changes. The script copies the
inputs into ``reproduce/`` and runs there, so the web app's own outputs in
the run directory are never overwritten.
"""
from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Optional

from mliprun.webapp import schema

PLACEHOLDER_MLIP = "REPLACE_WITH_AN_MLIP"

_CLI_COMMAND = {
    "singlepoint": ["singlepoint", "run"],
    "optimize": ["optimize", "run"],
    "freq": ["freq", "run"],
    "md": ["md", "run"],
    "neb": ["neb", "run"],
    "autoneb": ["autoneb", "run"],
}


def input_mode(job: dict) -> str:
    """``single``, ``pair`` or ``band``, from the roles in ``job["inputs"]``."""
    roles = set(job.get("inputs") or {})
    if "band" in roles:
        return "band"
    if {"initial", "final"} <= roles:
        return "pair"
    return "single"


def _format_value(field: dict, value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _param_args(job: dict, mode: str) -> list:
    calc_type = job["calc_type"]
    params = job.get("params") or {}
    args: list = []
    for field in schema.applicable_fields(calc_type, params, mode):
        flag = field["cli_flag"]
        name = field["name"]
        if flag is None or name not in params:
            continue
        value = params[name]
        if calc_type == "singlepoint" and name == "stress":
            if value == "on":
                args.append("--stress")
            elif value == "off":
                args.append("--no-stress")
            continue
        if value is None:
            continue
        if field["type"] == "bool":
            if "/" in flag:
                on, off = flag.split("/", 1)
                args.append(on if value else off)
            elif value:
                args.append(flag)
            continue
        if field["type"] == "indices":
            if name == "relax_atoms":
                # The neb CLI parses a plain comma list only, no ranges.
                natoms = 10 ** 9  # range-checked at submission already
                value = ",".join(str(i) for i in schema.parse_index_list(value, natoms))
            else:
                value = "".join(str(value).split())
        args += [flag, _format_value(field, value)]
    return args


def build_reproduce(job: dict, device_resolved: Optional[str] = None,
                    band_frames: Optional[int] = None,
                    run_dir: Optional[Path] = None) -> dict:
    """Build the reproduce command for a job.

    Parameters
    ----------
    job : dict
        Parsed ``job.json``.
    device_resolved : str, optional
        The device the worker resolved ``auto`` to. Used for ``neb`` (whose
        CLI forwards the device unresolved) and for the ``autoneb`` note.
    band_frames : int, optional
        Number of frames of a NEB band input.
    run_dir : Path, optional
        Where the inputs live, to check that the CLI's format detection
        reads a single-structure input the same way.

    Returns
    -------
    dict
        ``{"status", "notes", "argv", "script"}``; ``argv`` excludes the
        leading ``mlip``.
    """
    calc_type = job["calc_type"]
    calc = job.get("calculator") or {}
    name = calc.get("name")
    device = calc.get("device") or "auto"
    mode = input_mode(job)
    notes: list = []
    approximate = False

    argv = list(_CLI_COMMAND[calc_type])
    copies: list = []
    if mode == "single":
        meta = job["inputs"]["structure"]
        path = meta["path"]
        filename = Path(path).name
        copies.append(path)
        argv += ["--structure", filename]
        if run_dir is not None:
            try:
                from ase.io.formats import filetype

                detected = filetype(str(Path(run_dir) / path), read=True)
            except Exception:  # noqa: BLE001 - any failure means "CLI cannot detect it"
                detected = None
            if meta.get("format") and detected != meta.get("format"):
                approximate = True
                notes.append(f"The input was read as format '{meta.get('format')}', "
                             f"but the CLI detects formats from the file name "
                             f"and may read {filename} differently or not at all.")
    else:
        copies += ["inputs/initial.vasp", "inputs/final.vasp"]
        argv += ["--initial", "initial.vasp", "--final", "final.vasp"]

    if name == "emt":
        argv += ["--mlip", PLACEHOLDER_MLIP]
    else:
        argv += ["--mlip", name]
        for key in ("uma_task", "mace_head", "sevennet_task"):
            if calc.get(key) is not None:
                argv += ["--" + key.replace("_", "-"), calc[key]]

    if calc_type == "neb":
        argv += ["--device", device_resolved or (device if device != "auto" else "cpu")]
    elif calc_type != "autoneb":
        argv += ["--device", device]

    if calc_type == "neb" and mode == "band" and band_frames:
        # Closest CLI equivalent: same image count, endpoints as uploaded.
        argv += ["--num-images", str(max(band_frames - 2, 1)),
                 "--no-optimize-endpoints"]
    argv += _param_args(job, mode)

    params = job.get("params") or {}
    if calc_type == "neb" and mode == "band":
        approximate = True
        notes.append("The job started from an uploaded band; the CLI cannot, so "
                     "this command interpolates a new band (IDPP) between the "
                     "first and last frames.")
    if (calc_type == "neb" and mode == "pair" and params.get("interpolation") == "linear"
            and not params.get("relax_atoms")):
        approximate = True
        notes.append("The job used linear interpolation; the CLI always runs IDPP.")
    if calc_type == "autoneb" and name != "emt":
        used = device_resolved or device
        if used != "cpu":
            approximate = True
            notes.append(f"The job ran on device '{used}'; the autoneb CLI has no "
                         f"--device option and always uses the CPU.")
    if calc_type == "md" and params.get("ensemble") in ("nvt", "npt"):
        approximate = True
        notes.append("Initial velocities are drawn at random (no seed), so the "
                     "trajectory agrees statistically, not step by step.")
    if calc_type in ("neb", "autoneb"):
        fmts = {role: (meta or {}).get("format") for role, meta in job["inputs"].items()}
        non_vasp = sorted({f for f in fmts.values() if f and f != "vasp"})
        if non_vasp:
            approximate = True
            notes.append(f"Inputs were read as {', '.join(non_vasp)} and converted "
                         f"to VASP for the CLI, which reads only VASP files; "
                         f"anything beyond species, positions, cell and "
                         f"constraints is not carried over.")

    if name == "emt":
        status = "no"
        notes.insert(0, "Calculator 'emt' (ASE EMT) has no mlip CLI equivalent; "
                        f"replace {PLACEHOLDER_MLIP} with an MLIP tag to rerun "
                        "with a real model.")
    elif approximate:
        status = "approximate"
    else:
        status = "exact"

    lines = ["#!/usr/bin/env bash",
             f"# mliprun webapp job {job.get('job_id')} ({calc_type}), "
             f"created {job.get('created_at')}",
             f"# Reproducible: {status}"]
    lines += [f"# Note: {n}" for n in notes]
    lines += ["set -euo pipefail", 'cd "$(dirname "$0")"', "mkdir -p reproduce",
              "cp " + " ".join(shlex.quote(c) for c in copies) + " reproduce/",
              "cd reproduce"]
    command = "mlip " + " ".join(shlex.quote(a) for a in argv)
    if status == "no":
        lines.append("# " + command)
        lines.append("echo " + shlex.quote(notes[0]))
        lines.append("exit 1")
    else:
        lines.append(command)
    script = "\n".join(lines) + "\n"
    return {"status": status, "notes": notes, "argv": argv, "script": script}


def write_reproduce(run_dir: Path, job: dict, device_resolved: Optional[str] = None,
                    band_frames: Optional[int] = None) -> dict:
    """Write ``reproduce.sh`` (mode 0755) and ``reproduce.json`` into ``run_dir``.

    Returns
    -------
    dict
        ``{"status", "notes", "argv"}`` as written to ``reproduce.json``.
    """
    run_dir = Path(run_dir)
    built = build_reproduce(job, device_resolved=device_resolved,
                            band_frames=band_frames, run_dir=run_dir)
    sh = run_dir / "reproduce.sh"
    sh.write_text(built["script"], encoding="utf-8")
    os.chmod(sh, 0o755)
    payload = {"status": built["status"], "notes": built["notes"],
               "argv": built["argv"]}
    (run_dir / "reproduce.json").write_text(json.dumps(payload, indent=2) + "\n",
                                            encoding="utf-8")
    return payload
