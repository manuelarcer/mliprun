"""Build the web data pack for the front-page showcase (no MLIP, no calculation).

Reads the run outputs pulled back from cos-cluster into docs/showcase/data/runs/
and writes docs/showcase/data/pack/:

    relax.json      + relax.xyz       O at the Pt(111) fcc hollow, single model
    committee.json                    same start, three-model committee
    neb.json        + neb_band.xyz    O fcc -> hcp CI-NEB, final band
                    + neb_seed.xyz    the IDPP seed band
                    + ts_mode.xyz     TS +/- 0.3 A along the imaginary mode
    md.json                           MD did NOT run; labelled as such, no frames

Every number is read from a run output (run record, log, CSV, structure) or is
geometry computed from a stored structure; nothing is recomputed with an MLIP.
JSON schema (one file per use case):

    {"use_case": str, "command": str, "converged": bool, "flags": [str],
     "key_numbers": [{"label": str, "value": float, "unit": str}],
     "series": {name: [float]}, "frames": str | null}

plus extra keys "run_dir" (relative to data/runs), "notes", "cwd" (where a
command with relative paths ran) and, for the NEB, "ts_image_index" and
"frames_extra". Series names carry their unit as a suffix, as in mliprun's CSVs.

Usage:
    python build_pack.py [--runs DIR] [--out DIR]
"""
import argparse
import csv
import json
import math
import re
import shlex
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_water import check_frame, top_layer_z  # noqa: E402,F401  (re-exported)
from subsample import pick_indices  # noqa: E402

SHOWCASE = Path(__file__).resolve().parents[1]
SCRIPTS = SHOWCASE / "scripts"
ENV_FILE = SHOWCASE / "cluster.env"

SCHEMA_KEYS = ("use_case", "command", "converged", "flags", "key_numbers",
               "series", "frames")
MAX_FRAMES = 300
MAX_OH = 1.3            # A; longer O-H means a dissociated water (same as check_water)
TS_MODE_FRAMES = 30
TS_MODE_AMPLITUDE = 0.3  # A; peak displacement of the most-displaced atom

# ----------------------------------------------------------------- schema --


def kn(label: str, value, unit: str) -> dict:
    """One key number. The value is kept at full precision."""
    return {"label": label, "value": float(value), "unit": unit}


def make_record(use_case, command, converged, flags, key_numbers, series, frames,
                **extra) -> dict:
    rec = {"use_case": use_case, "command": command, "converged": bool(converged),
           "flags": list(flags), "key_numbers": list(key_numbers),
           "series": {k: [float(x) for x in v] for k, v in series.items()},
           "frames": frames}
    rec.update(extra)
    return rec


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_record(rec: dict) -> None:
    """Raise ValueError when rec breaks the pack schema."""
    for k in SCHEMA_KEYS:
        if k not in rec:
            raise ValueError(f"missing key: {k}")
    if not isinstance(rec["use_case"], str) or not rec["use_case"]:
        raise ValueError("use_case must be a non-empty string")
    if not isinstance(rec["command"], str) or not rec["command"]:
        raise ValueError("command must be a non-empty string")
    if not isinstance(rec["converged"], bool):
        raise ValueError("converged must be a bool")
    if not all(isinstance(f, str) and f for f in rec["flags"]):
        raise ValueError("flags must be non-empty strings")
    for item in rec["key_numbers"]:
        if set(item) != {"label", "value", "unit"}:
            raise ValueError(f"key number needs label/value/unit: {item}")
        if not isinstance(item["label"], str) or not item["label"]:
            raise ValueError(f"key number without label: {item}")
        if not isinstance(item["unit"], str) or not item["unit"].strip():
            raise ValueError(f"key number without unit: {item['label']}")
        if not _finite(item["value"]):
            raise ValueError(f"key number not a finite number: {item['label']}")
    if not isinstance(rec["series"], dict):
        raise ValueError("series must be a dict")
    for name, vals in rec["series"].items():
        if not all(_finite(v) for v in vals):
            raise ValueError(f"series {name} has a non-finite value")
    if rec["frames"] is not None and not isinstance(rec["frames"], str):
        raise ValueError("frames must be a file name or null")


# ------------------------------------------------------------ convergence --

_LOG_LINE = re.compile(r"^\s*\w+:\s+(\d+)\s+\S+\s+(-?\d+\.\d+)\*?\s+(\d+\.\d+)\s*$")


def read_opt_log(path: Path) -> list:
    """(step, energy_eV, fmax_eV_per_A) rows of an ASE optimizer log."""
    rows = []
    for line in Path(path).read_text().splitlines():
        m = _LOG_LINE.match(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    if not rows:
        raise ValueError(f"no optimizer steps in {path}")
    return rows


def _params_txt(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^\s*fmax[^:]*:\s*([0-9.eE+-]+)\s*$", line)
        if m:
            out["fmax"] = float(m.group(1))
        m = re.match(r"^\s*Converged:\s*(\w+)", line)
        if m:
            out["converged"] = m.group(1) == "True"
    return out


def optimize_converged(run_dir: Path):
    """Converged when the last fmax in opt.log is below the run's fmax target.

    The log decides. A disagreement with opt_params.txt or with the run record
    is flagged as "convergence_record_mismatch".
    """
    run_dir = Path(run_dir)
    params = _params_txt(run_dir / "opt_params.txt")
    last_fmax = read_opt_log(run_dir / "opt.log")[-1][2]
    converged = last_fmax < params["fmax"]
    flags = [] if converged else ["not_converged"]
    claims = []
    if "converged" in params:
        claims.append(params["converged"])
    rec_path = run_dir / "mliprun_run.json"
    if rec_path.exists():
        for st in json.loads(rec_path.read_text()).get("stages", []):
            if "converged" in st.get("results", {}):
                claims.append(bool(st["results"]["converged"]))
    if any(c != converged for c in claims):
        flags.append("convergence_record_mismatch")
    return converged, flags


# ----------------------------------------------------------------- frames --


def cap_frames(frames: list, n: int = MAX_FRAMES) -> list:
    """At most n evenly spaced frames, first and last kept (subsample.py rule)."""
    if len(frames) <= n:
        return frames
    return [frames[int(i)] for i in pick_indices(len(frames), n)]


def write_frames(path: Path, frames: list) -> int:
    frames = cap_frames(frames)
    write(str(path), frames, format="extxyz")
    return len(frames)


# --------------------------------------------------------------- geometry --


def o_heights(atoms: Atoms) -> list:
    """Height (A) of every O above the mean z of the top Pt layer."""
    top = top_layer_z(atoms)
    return [float(a.position[2] - top) for a in atoms if a.symbol == "O"]


def geometry_flags(atoms: Atoms) -> list:
    """o_subsurface: an O below the top Pt layer. water_dissociated: an O-H
    longer than 1.3 A, or an O without exactly two H (H present only)."""
    flags = []
    if any(a.symbol == "Pt" for a in atoms) and any(h < 0.0 for h in o_heights(atoms)):
        flags.append("o_subsurface")
    if any(a.symbol == "H" for a in atoms):
        res = check_frame(atoms, max_oh=MAX_OH, max_height=float("inf"))
        if res["max_oh"] > MAX_OH or any(len(w["H"]) != 2 for w in res["waters"]):
            flags.append("water_dissociated")
    return flags


def union_flags(structures) -> list:
    out = []
    for at in structures:
        for f in geometry_flags(at):
            if f not in out:
                out.append(f)
    return out


def shortest_distance(atoms: Atoms, sym_a: str, sym_b: str) -> float:
    ia = [a.index for a in atoms if a.symbol == sym_a]
    ib = [a.index for a in atoms if a.symbol == sym_b]
    return float(min(atoms.get_distances(i, ib, mic=True).min() for i in ia))


# ---------------------------------------------------------------- command --


def command_from_record(run_dir: Path):
    """shlex-joined argv when the run record stores one, else None."""
    rec = json.loads((Path(run_dir) / "mliprun_run.json").read_text())
    argv = rec.get("argv") or rec.get("provenance", {}).get("argv")
    return shlex.join(argv) if argv else None


_VAR = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _expand(text: str, scalars: dict) -> str:
    def sub(m):
        name = m.group(1) or m.group(2)
        if name not in scalars:
            raise KeyError(f"variable ${name} has no value in the script")
        return _expand(scalars[name], scalars)
    return _VAR.sub(sub, text)


def command_from_script(script: Path, subcommand: str, env_file: Path = ENV_FILE,
                        local: dict = None, extra_args: list = ()) -> str:
    """The first `.../mlip <subcommand>` line of a bash run script, verbatim,
    with its variables expanded to the values that script (and the env file it
    sources) assigns. Output redirections are dropped.

    local: values of function-local variables (e.g. {"dir": "$OUT/o_fcc"});
    extra_args: what "$@" stood for at that call.
    """
    scalars, arrays = {}, {}
    for line in Path(env_file).read_text().splitlines():
        m = re.match(r"^\s*export\s+(\w+)=(.*)$", line)
        if m:
            scalars[m.group(1)] = shlex.split(m.group(2))[0] if m.group(2) else ""
    text = Path(script).read_text()
    for m in re.finditer(r"^(\w+)=\((.*?)\)", text, flags=re.M | re.S):
        body = "\n".join(l.split("#")[0] for l in m.group(2).splitlines())
        arrays[m.group(1)] = shlex.split(body)
    for m in re.finditer(r"^(\w+)=([^(\n].*)$", text, flags=re.M):
        scalars[m.group(1)] = shlex.split(m.group(2))[0]
    scalars.update(local or {})

    joined = text.replace("\\\n", " ")
    pat = re.compile(r'^\s*"?\$\{?\w+\}?/mlip"?\s+' + re.escape(subcommand) + r"\b.*$",
                     flags=re.M)
    m = pat.search(joined)
    if not m:
        raise ValueError(f"no 'mlip {subcommand}' line in {script}")
    tokens = []
    for tok in shlex.split(m.group(0)):
        if tok in (">", ">>", "||", "&&", "|", ";") or re.match(r"^\d?>", tok):
            break
        am = re.fullmatch(r"\$\{(\w+)\[@\]\}", tok)
        if am:
            tokens.extend(_expand(t, scalars) for t in arrays[am.group(1)])
        elif tok == "$@":
            tokens.extend(extra_args)
        else:
            tokens.append(_expand(tok, scalars))
    return shlex.join(tokens)


def _user_params(record: dict) -> dict:
    out = {}
    blocks = [record.get("parameters", {})]
    blocks += [st.get("parameters", {}) for st in record.get("stages", [])]
    for block in blocks:
        for name, p in block.items():
            if isinstance(p, dict) and p.get("source") == "user":
                out[name] = p["value"]
    return out


def command_mismatches(command: str, record: dict) -> list:
    """Every parameter the run record marks as user-given must appear in the
    command as `--flag value` (or a bare `--flag` for True). Returns the
    missing ones as "--flag value" strings."""
    toks = shlex.split(command)
    missing = []
    for name, value in _user_params(record).items():
        flag = "--" + name.replace("_", "-")
        if value is True:
            if flag not in toks:
                missing.append(flag)
            continue
        ok = False
        for i, t in enumerate(toks[:-1]):
            if t == flag:
                nxt = toks[i + 1]
                if nxt == str(value):
                    ok = True
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    try:
                        ok = math.isclose(float(nxt), float(value))
                    except ValueError:
                        ok = False
        if not ok:
            missing.append(f"{flag} {value}")
    return missing


def checked_command(command: str, run_dir: Path) -> str:
    record = json.loads((Path(run_dir) / "mliprun_run.json").read_text())
    bad = command_mismatches(command, record)
    if bad:
        raise ValueError(f"command disagrees with {run_dir}/mliprun_run.json: {bad}")
    return command


def _command(run_dir, script, subcommand, local=None):
    return checked_command(command_from_record(run_dir)
                           or command_from_script(SCRIPTS / script, subcommand,
                                                  local=local),
                           run_dir)


# -------------------------------------------------------------- utilities --


def read_csv_columns(path: Path) -> dict:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    return cols


def numeric(cols: dict) -> dict:
    out = {}
    for k, v in cols.items():
        try:
            out[k] = [float(x) for x in v]
        except ValueError:
            continue
    return out


def _record(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "mliprun_run.json").read_text())


def _results(run_dir: Path, kind: str) -> dict:
    for st in _record(run_dir)["stages"]:
        if st["kind"] == kind:
            return st
    raise KeyError(f"no stage {kind} in {run_dir}")


def _close(a, b, tol, what):
    if abs(a - b) > tol:
        raise ValueError(f"{what}: {a} vs {b} differ by more than {tol}")


# ------------------------------------------------------------- use cases --


def build_relax(runs: Path, out: Path) -> dict:
    rd = runs / "01_relax"
    converged, flags = optimize_converged(rd / "o_fcc")
    for name in ("bulk", "slab", "o_hcp"):
        ok, _ = optimize_converged(rd / name)
        if not ok:
            flags.append(f"{name}_not_converged")

    bulk = read(rd / "bulk" / "opt_final.vasp")
    a_lat = float(np.mean(bulk.cell.lengths()) * math.sqrt(2.0))
    m = re.search(r"lattice constant a \(A\):\s*([0-9.]+)",
                  (rd / "bulk" / "fcc_check.txt").read_text())
    _close(a_lat, float(m.group(1)), 1e-6, "lattice constant vs fcc_check.txt")

    e_fcc = _results(rd / "o_fcc", "optimize")
    e_hcp = _results(rd / "o_hcp", "optimize")
    fin_fcc = read(rd / "o_fcc" / "opt_final.vasp")
    fin_hcp = read(rd / "o_hcp" / "opt_final.vasp")
    log = read_opt_log(rd / "o_fcc" / "opt.log")
    fmax_target = _params_txt(rd / "o_fcc" / "opt_params.txt")["fmax"]

    frames = read(rd / "o_fcc" / "opt_traj.xyz", index=":")
    if len(frames) != len(log):
        raise ValueError("opt_traj.xyz frame count differs from opt.log steps")
    n = write_frames(out / "relax.xyz", frames)
    flags += union_flags(frames + [fin_fcc, fin_hcp])

    cols = numeric(read_csv_columns(rd / "o_fcc" / "opt_convergence.csv"))
    key = [
        kn("Pt lattice constant a (relaxed bulk, uma-s-1p2/oc20)", a_lat, "Å"),
        kn("total energy, O at fcc hollow (relaxed)", e_fcc["results"]["final_energy_eV"], "eV"),
        kn("total energy, O at hcp hollow (relaxed)", e_hcp["results"]["final_energy_eV"], "eV"),
        kn("E(hcp) − E(fcc)", e_hcp["results"]["final_energy_eV"]
           - e_fcc["results"]["final_energy_eV"], "eV"),
        kn("final fmax, O at fcc hollow", e_fcc["results"]["final_fmax_eV_per_A"], "eV/Å"),
        kn("fmax target", fmax_target, "eV/Å"),
        kn("BFGS steps, O at fcc hollow", e_fcc["steps"], "steps"),
        kn("O height above top Pt layer, fcc (relaxed)", o_heights(fin_fcc)[0], "Å"),
        kn("shortest O–Pt distance, fcc (relaxed)", shortest_distance(fin_fcc, "O", "Pt"), "Å"),
    ]
    return make_record(
        "relax", _command(rd / "o_fcc", "run_01_relax.sh", "optimize run",
                          local={"dir": "$OUT/o_fcc"}),
        converged, flags, key,
        {"step": cols["step"], "energy_eV": cols["energy(eV)"],
         "fmax_eV_per_A": cols["fmax(eV/A)"]},
        "relax.xyz", run_dir="01_relax/o_fcc",
        notes=[f"relax.xyz: all {n} optimizer steps of 01_relax/o_fcc/opt.traj "
               "(converted on the cluster with subsample.py).",
               "Energies are total energies of the 37-atom cell at the "
               "uma-s-1p2/oc20 level; only differences are meaningful.",
               "Slab: Pt(111) 3x3, 4 layers, bottom 2 fixed, cut from the relaxed a."])


def build_committee(runs: Path, out: Path) -> dict:
    rd = runs / "02_committee"
    converged, flags = optimize_converged(rd)
    st = _results(rd, "optimize")
    cu = st["results"]["committee_uncertainty"]
    cols = read_csv_columns(rd / "opt_committee.csv")
    num = numeric(cols)
    _close(num["energy_spread_aligned_eV"][-1], cu["energy_spread_aligned_final_eV"],
           1e-9, "final energy spread CSV vs run record")
    _close(num["sigma_max_free_eV_per_A"][-1], cu["sigma_max_free_final_eV_per_A"],
           1e-9, "final sigma_max_free CSV vs run record")
    if any(v != "False" for v in cols["mixed_theory"]):
        flags.append("mixed_theory")
    members = [k for k in num if k.startswith("E_") and k.endswith("_eV")]
    fin = read(rd / "opt_final.vasp")
    flags += union_flags([fin])

    key = [kn(f"final energy, member {m[2:-3]} (absolute; offsets differ between models)",
              num[m][-1], "eV") for m in members]
    key += [
        kn("energy spread across members, final step (standard deviation, ddof=1, "
           "each member's step-0 energy subtracted)", num["energy_spread_aligned_eV"][-1], "eV"),
        kn("sigma_max_free, final step (largest per-atom force disagreement, free atoms)",
           cu["sigma_max_free_final_eV_per_A"], "eV/Å"),
        kn("sigma_mean_free, final step (mean per-atom force disagreement, free atoms)",
           cu["sigma_mean_free_final_eV_per_A"], "eV/Å"),
        kn("sigma_max_free, peak over the run", cu["sigma_max_free_peak_eV_per_A"], "eV/Å"),
        kn("sigma_max_free / fmax, final step", cu["sigma_max_free_over_fmax_final"],
           "dimensionless"),
        kn("final fmax of the committee-mean force", st["results"]["final_fmax_eV_per_A"], "eV/Å"),
        kn("fmax target", _params_txt(rd / "opt_params.txt")["fmax"], "eV/Å"),
        kn("BFGS steps", st["steps"], "steps"),
        kn("committee members", len(members), "models"),
        kn("free atoms (sigma_*_free population)", cu["n_free_atoms"], "atoms"),
    ]
    series = {k: num[k] for k in
              ["step", "energy_mean_eV", "energy_spread_aligned_eV", *members,
               "fmax_eV_per_A", "sigma_max_free_eV_per_A", "sigma_mean_free_eV_per_A"]}
    pa = numeric(read_csv_columns(rd / "opt_committee_peratom.csv"))
    series["peratom_final_atom_index"] = pa["atom_index"]
    series["peratom_final_sigma_free_eV_per_A"] = pa["sigma_free_eV_per_A"]
    series["peratom_final_sigma_all_eV_per_A"] = pa["sigma_all_eV_per_A"]
    return make_record(
        "committee", _command(rd, "run_02_committee.sh", "optimize run"),
        converged, flags, key, series, None, run_dir="02_committee",
        cwd=str(Path(_record(rd)["inputs"]["structure_abspath"]).parent),
        notes=["Members: uma-s-1p2@oc20, 7net-omni@oc20, mace-mh-1@oc20_usemppbe, all at the "
               "OC20 (RPBE) level (mixed_theory False on every row).",
               "Start: the unrelaxed 01_relax/o_fcc/POSCAR; the relaxation follows the "
               "members' mean force.",
               "energy_spread_aligned_eV is a standard deviation (ddof=1), not a range "
               "(docs/OUTPUTS.md).",
               f"Worst free atom at the final step: {cu['worst_atom_free_symbol']} "
               f"#{cu['worst_atom_free']}.",
               "No frame file: the committee trace is per-step numbers only."])


def ts_mode_frames(vib_json: Path, ts_vasp: Path, expected_indices,
                   n_frames=TS_MODE_FRAMES, amplitude=TS_MODE_AMPLITUDE):
    """TS displaced along its imaginary mode, sinusoid over n_frames, peak
    displacement `amplitude` (A) on the most-displaced atom."""
    from ase.vibrations import VibrationsData
    vib = VibrationsData.read(str(vib_json))
    if list(vib.get_indices()) != list(expected_indices):
        raise ValueError("vibration indices differ from ts_verify displaced_indices")
    energies, modes = vib.get_energies_and_modes(all_atoms=True)
    imag = [i for i, e in enumerate(energies) if abs(e.imag) > abs(e.real)]
    if len(imag) != 1:
        raise ValueError(f"expected one imaginary mode, found {len(imag)}")
    mode = np.real(modes[imag[0]])
    mode = mode / np.linalg.norm(mode, axis=1).max()
    ts = read(ts_vasp)
    _close(float(np.abs(ts.positions - vib.get_atoms().positions).max()), 0.0, 1e-6,
           "ts_structure.vasp vs vibration reference positions")
    frames = []
    for k in range(n_frames):
        at = ts.copy()
        at.set_constraint()
        at.positions = ts.positions + amplitude * math.sin(2 * math.pi * k / n_frames) * mode
        frames.append(at)
    freq = float(np.abs(vib.get_frequencies()[imag[0]]))
    moving = [int(i) for i in np.nonzero(np.linalg.norm(mode, axis=1) > 0)[0]]
    return frames, freq, moving


def reaction_coordinate(band) -> list:
    s = [0.0]
    for a, b in zip(band[:-1], band[1:]):
        d = b.positions - a.positions
        cell = a.cell
        frac = np.linalg.solve(cell.T, d.T).T
        frac -= np.round(frac) * a.pbc
        s.append(s[-1] + float(np.linalg.norm(frac @ cell)))
    return s


def build_neb(runs: Path, out: Path) -> dict:
    rd = runs / "03_neb"
    rec = _record(rd)
    st = _results(rd, "neb")
    res = st["results"]
    conv = numeric(read_csv_columns(rd / "neb_convergence.csv"))
    fmax_target = st["parameters"]["fmax"]["value"]
    converged = rec["status"] == "converged" and conv["fmax(eV/A)"][-1] < fmax_target
    flags = [] if converged else ["not_converged"]

    data = numeric(read_csv_columns(rd / "neb_data.csv"))
    ts_i = res["ts_image_index"]
    _close(data["relative_energy"][ts_i], res["forward_barrier_eV"], 1e-9,
           "neb_data.csv TS energy vs run record barrier")

    band = read(rd / "A2B_band.xyz", index=":")
    seed = read(rd / "idpp_seed.xyz", index=":")
    if len(band) != res["n_images"] or len(seed) != res["n_images"]:
        raise ValueError("band or seed frame count differs from n_images")
    write_frames(out / "neb_band.xyz", band)
    write_frames(out / "neb_seed.xyz", seed)

    bc = json.loads((rd / "band_check.json").read_text())
    if bc["exploded"]:
        flags.append("band_exploded")
    if bc["jump_warning"]:
        flags.append("band_jump_warning")

    tv_dir = rd / "ts_verify"
    tv = _results(tv_dir, "ts-verify")["results"]
    flags.append("ts_verified" if tv["verdict"] == "verified" else "ts_not_verified")
    relax = tv["relax"]
    notes = []
    if max(relax["initial_fmax_eV_per_A"], relax["final_fmax_eV_per_A"]) > relax["fmax_eV_per_A"]:
        flags.append("ts_verify_endpoint_fmax_above_relax_fmax")
        notes.append(
            "ts_verify reported endpoint fmax "
            f"{relax['initial_fmax_eV_per_A']:.4f} (IS) / {relax['final_fmax_eV_per_A']:.4f} "
            f"(FS) eV/Å, above its internal displace-and-relax fmax of "
            f"{relax['fmax_eV_per_A']} eV/Å (the endpoints were relaxed to 0.02 eV/Å). "
            "The verdict is unaffected: both displaced relaxations matched their endpoint "
            f"(result: {relax['result']}).")

    mode_frames, freq, moving = ts_mode_frames(
        tv_dir / "ts_freq_vibrations.json", tv_dir / "ts_structure.vasp",
        tv["displaced_indices"])
    _close(freq, tv["imaginary_frequencies_cm-1"][0], 1e-3,
           "imaginary frequency from Hessian vs run record")
    write_frames(out / "ts_mode.xyz", mode_frames)
    flags += union_flags(band + seed + mode_frames)

    ts = read(tv_dir / "ts_structure.vasp")
    key = [
        kn("forward barrier, fcc → hcp", res["forward_barrier_eV"], "eV"),
        kn("reverse barrier, hcp → fcc", res["reverse_barrier_eV"], "eV"),
        kn("reaction energy, E(hcp) − E(fcc)", res["reaction_energy_eV"], "eV"),
        kn("final band fmax", res["final_fmax_eV_per_A"], "eV/Å"),
        kn("fmax target (band and endpoints)", fmax_target, "eV/Å"),
        kn("NEB optimizer steps", st["steps"], "steps"),
        kn("TS imaginary frequency (magnitude; the mode is imaginary)",
           tv["imaginary_frequencies_cm-1"][0], "cm⁻¹"),
        kn("overlap of imaginary mode with NEB tangent", tv["overlap"]["value"], "dimensionless"),
        kn("overlap threshold", tv["overlap"]["threshold"], "dimensionless"),
        kn("TS fmax (free components)", tv["ts_fmax_free_eV_per_A"], "eV/Å"),
        kn("O height above top Pt layer at TS", o_heights(ts)[0], "Å"),
        kn("band check: largest drift from seeded band", bc["worst"]["drift"], "Å"),
        kn("band check: allowance for that atom", bc["worst"]["allowance"], "Å"),
        kn("band check: largest inter-image jump", bc["max_interimage_jump"], "Å"),
        kn("ts_verify endpoint fmax, IS", relax["initial_fmax_eV_per_A"], "eV/Å"),
        kn("ts_verify endpoint fmax, FS", relax["final_fmax_eV_per_A"], "eV/Å"),
        kn("ts_verify internal relax fmax", relax["fmax_eV_per_A"], "eV/Å"),
        kn("TS mode animation peak displacement", TS_MODE_AMPLITUDE, "Å"),
    ]
    series = {
        "image_index": data["image_index"],
        "energy_eV": data["energy"],
        "relative_energy_eV": data["relative_energy"],
        "reaction_coordinate_A": reaction_coordinate(band),
        "seed_reaction_coordinate_A": reaction_coordinate(seed),
        "neb_step": conv["step"],
        "neb_fmax_eV_per_A": conv["fmax(eV/A)"],
        "neb_barrier_eV": conv["barrier(eV)"],
    }
    notes += [
        f"TS is image {ts_i} of {res['n_images']} (climbing image).",
        "neb_band.xyz: final band (A2B.traj). neb_seed.xyz: IDPP seed band (idpp.traj; "
        "identical to A2B_full.traj frames 0..8, max position difference 0.0 Å, checked "
        "on the cluster).",
        f"ts_mode.xyz: {TS_MODE_FRAMES} frames, TS + {TS_MODE_AMPLITUDE} Å · sin(2πk/"
        f"{TS_MODE_FRAMES}) · mode, the mode scaled so its most-displaced atom moves "
        f"{TS_MODE_AMPLITUDE} Å; nonzero only on the {len(moving)} atoms ts_verify displaced "
        f"(indices {moving[0]}..{moving[-1]}). An illustration of the mode, not dynamics.",
        "Band explosion allowance: 1.0 Å + 1.0 × each atom's IS→FS travel; not exploded.",
    ]
    return make_record(
        "neb", _command(rd, "run_03_neb.sh", "neb run"), converged, flags, key, series,
        "neb_band.xyz", run_dir="03_neb",
        cwd=re.search(r"^Output dir:\s*(\S+)", (rd / "neb_parameters.txt").read_text(),
                      flags=re.M).group(1),
        frames_extra={"seed": "neb_seed.xyz", "ts_mode": "ts_mode.xyz"},
        ts_image_index=ts_i, notes=notes)


def build_md(runs: Path, out: Path) -> dict:
    rd = runs / "04_md"
    md_dir = rd / "md"
    if md_dir.exists() and any(md_dir.iterdir()):
        raise ValueError(f"{md_dir} is not empty: the MD state changed, review md.json by hand")
    if (out / "md.xyz").exists():
        raise ValueError("md.xyz exists but the MD never ran; remove it")
    br = rd / "bilayer_relax"
    relax_ok, relax_flags = optimize_converged(br)
    fin = read(br / "opt_final.vasp")
    res = check_frame(fin, max_oh=MAX_OH, max_height=4.0)
    heights = [w["height"] for w in res["waters"]]
    flags = ["md_not_run", "pending_decision_water_underbinding"]
    flags += [f"bilayer_relaxation_{f}" for f in relax_flags]
    flags += union_flags([fin])
    key = [
        kn("bilayer relaxation (not MD): highest water O above top Pt layer", max(heights), "Å"),
        kn("bilayer relaxation (not MD): lowest water O above top Pt layer", min(heights), "Å"),
        kn("bilayer relaxation (not MD): longest O–H bond", res["max_oh"], "Å"),
        kn("bilayer relaxation (not MD): water-check height limit", 4.0, "Å"),
        kn("bilayer relaxation (not MD): final fmax",
           _results(br, "optimize")["results"]["final_fmax_eV_per_A"], "eV/Å"),
    ]
    return make_record(
        "md", _command(br, "run_04_md.sh", "optimize run"), False, flags, key, {}, None,
        run_dir="04_md/bilayer_relax",
        notes=["The MD did NOT run. The 6-H2O bilayer relaxation converged "
               f"({'converged' if relax_ok else 'NOT converged'}), but the relaxed water "
               "layer sits 4.7–5.2 Å above the top Pt layer with uma-s-1p2/oc20, so the "
               "water check (O height ≤ 4.0 Å) failed and the run script stopped before MD.",
               "Likely underbinding of water at the OC20 (RPBE) level; how to proceed is "
               "Juan's decision.",
               "command is the bilayer relaxation that ran, not an MD command.",
               "converged refers to the MD (false: it never ran); frames and series are empty."])


BUILDERS = {"relax": build_relax, "committee": build_committee, "neb": build_neb,
            "md": build_md}


def build(runs: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, fn in BUILDERS.items():
        rec = fn(runs, out)
        validate_record(rec)
        (out / f"{name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        records[name] = rec
    return records


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=SHOWCASE / "data" / "runs")
    p.add_argument("--out", type=Path, default=SHOWCASE / "data" / "pack")
    args = p.parse_args(argv)
    for name, rec in build(args.runs, args.out).items():
        print(f"{name}: converged={rec['converged']} flags={rec['flags']} "
              f"frames={rec['frames']} key_numbers={len(rec['key_numbers'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
