"""Upload store: saves uploaded structure files and parses them with ASE.

Layout (PLAN section 2)::

    <runs-dir>/_uploads/<upload_id>/<sanitized filename>   the bytes, unchanged
    <runs-dir>/_uploads/<upload_id>/meta.json              UploadInfo minus xyz/frames_xyz

Parsing (``ase.io.read``) is CPU work, so the server calls the ``process_*``
and ``describe_*`` methods from a threadpool, never on the event loop.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

logger = logging.getLogger(__name__)

UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{12}$")
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_FILES_PER_REQUEST = 50
MAX_FRAMES_XYZ = 500

# Keys of UploadInfo (PLAN 4.3), in order. Anything else in meta.json is internal.
UPLOAD_INFO_KEYS = ("id", "filename", "ok", "format", "created_at", "formula",
                    "natoms", "n_frames", "symbols", "cell", "pbc",
                    "has_constraints", "xyz", "frames_xyz")

_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._+-]")


def iso_utc(dt: Optional[datetime] = None) -> str:
    """ISO-8601 UTC with milliseconds and ``Z`` (``2026-10-15T14:23:01.123Z``)."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def sanitize_filename(name: Optional[str]) -> str:
    """Make an uploaded filename safe to store.

    Takes the basename (``/`` and ``\\`` both count as separators), replaces
    every character outside ``[A-Za-z0-9._+-]`` with ``_``, strips leading
    dots; an empty result becomes ``structure``. ``POSCAR`` stays ``POSCAR``.
    """
    base = re.split(r"[/\\]", name or "")[-1]
    base = _SAFE_CHARS_RE.sub("_", base).lstrip(".")
    return base or "structure"


def frame_xyz(atoms) -> str:
    """One extended-XYZ block for ``atoms`` (PLAN 4.3)."""
    cell = atoms.cell.array.reshape(-1)
    lattice = " ".join(f"{float(v):.8f}" for v in cell)
    pbc = " ".join("T" if p else "F" for p in atoms.pbc)
    lines = [str(len(atoms)),
             f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="{pbc}"']
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        lines.append("%s %.6f %.6f %.6f" % (sym, pos[0], pos[1], pos[2]))
    return "\n".join(lines) + "\n"


def read_frames(path: Path, fmt: Optional[str]) -> list:
    """All frames of ``path`` as a list of Atoms (raises on any failure)."""
    import ase.io

    frames = ase.io.read(str(path), index=":", format=fmt)
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        raise ValueError("no structures found in the file")
    return frames


def describe_frames(frames: list) -> Dict[str, Any]:
    """Structure fields of UploadInfo for a list of frames (last frame describes)."""
    last = frames[-1]
    first_symbols = frames[0].get_chemical_symbols()
    same = all(f.get_chemical_symbols() == first_symbols for f in frames)
    n = len(frames)
    return {
        "formula": last.get_chemical_formula(),
        "natoms": len(last),
        "n_frames": n,
        "symbols": last.get_chemical_symbols(),
        "cell": [[float(v) for v in row] for row in last.cell.array],
        "pbc": [bool(p) for p in last.pbc],
        "has_constraints": bool(last.constraints),
        "xyz": frame_xyz(last),
        "frames_xyz": ("".join(frame_xyz(f) for f in frames)
                       if 2 <= n <= MAX_FRAMES_XYZ else None),
        # internal (not part of UploadInfo): used to build band StructureMeta
        "frames_same_symbols": same,
    }


def describe_file(path: Path, fmt: Optional[str]) -> Dict[str, Any]:
    """Parse ``path`` and return :func:`describe_frames` of it."""
    return describe_frames(read_frames(path, fmt))


def public_info(meta: Dict[str, Any], full: bool) -> Dict[str, Any]:
    """Project internal metadata onto UploadInfo (without xyz unless ``full``)."""
    out = {k: meta.get(k) for k in UPLOAD_INFO_KEYS}
    if not full:
        out.pop("xyz", None)
        out.pop("frames_xyz", None)
    return out


class UploadStore:
    """Uploaded structures under ``<runs-dir>/_uploads``.

    Parameters
    ----------
    runs_dir : Path
        The web app's runs directory.
    """

    def __init__(self, runs_dir: Path):
        self.root = Path(runs_dir) / "_uploads"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._metas: Dict[str, Dict[str, Any]] = {}
        self._xyz_cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ----------------------------------------------------------------- loading
    def _load(self) -> None:
        for d in self.root.iterdir():
            if not d.is_dir() or not UPLOAD_ID_RE.match(d.name):
                continue
            try:
                meta = json.loads((d / "meta.json").read_text())
                if meta.get("id") != d.name:
                    raise ValueError("meta.json id does not match its directory")
            except Exception as exc:  # noqa: BLE001 - unreadable upload: skip it
                logger.warning("ignoring upload %s: %s", d, exc)
                continue
            self._metas[d.name] = meta

    # ----------------------------------------------------------------- queries
    def get(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Internal metadata of an upload, or ``None``."""
        if not isinstance(upload_id, str) or not UPLOAD_ID_RE.match(upload_id):
            return None
        with self._lock:
            return self._metas.get(upload_id)

    def path_of(self, upload_id: str) -> Path:
        """Path of the stored file of an existing upload."""
        meta = self._metas[upload_id]
        return self.root / upload_id / meta["filename"]

    def list(self) -> List[Dict[str, Any]]:
        """UploadInfo (without xyz) of all uploads, newest first."""
        with self._lock:
            metas = list(self._metas.values())
        metas.sort(key=lambda m: (m.get("created_at") or "", m.get("id") or ""),
                   reverse=True)
        return [public_info(m, full=False) for m in metas]

    def full_info(self, upload_id: str) -> Dict[str, Any]:
        """Full UploadInfo (with xyz). Parses the file when not cached. Blocking."""
        meta = self._metas[upload_id]
        cached = self._xyz_cache.get(upload_id)
        if cached is None:
            desc = describe_file(self.path_of(upload_id), meta.get("format"))
            cached = {"xyz": desc["xyz"], "frames_xyz": desc["frames_xyz"]}
            self._xyz_cache[upload_id] = cached
        info = public_info(meta, full=True)
        info.update(cached)
        return info

    def structure_meta(self, upload_id: str, role: str) -> Dict[str, Any]:
        """StructureMeta for ``validate_submission`` (PLAN 3.3)."""
        meta = self._metas[upload_id]
        out = {"natoms": meta["natoms"], "symbols": list(meta["symbols"]),
               "n_frames": meta["n_frames"], "pbc": list(meta["pbc"])}
        if role == "band":
            out["frames_same_symbols"] = bool(meta.get("frames_same_symbols", True))
        return out

    # -------------------------------------------------------------- processing
    def process_upload(self, filename: Optional[str], fileobj: BinaryIO,
                       fmt: Optional[str]) -> Dict[str, Any]:
        """Store and parse one uploaded file. Blocking (call from a threadpool).

        Returns a full UploadInfo on success, or a failure entry
        ``{"id": None, "filename", "ok": False, "error"}``.
        """
        name = sanitize_filename(filename)
        fmt = fmt or None
        tmpdir = Path(tempfile.mkdtemp(prefix=".incoming-", dir=str(self.root)))
        try:
            tmp_path = tmpdir / name
            size = 0
            too_big = False
            with open(tmp_path, "wb") as out:
                while True:
                    chunk = fileobj.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_FILE_BYTES:
                        too_big = True
                        break
                    out.write(chunk)
            if too_big:
                return _failure(name, f"file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
            if size == 0:
                return _failure(name, "file is empty")
            try:
                if fmt is None:
                    from ase.io.formats import filetype

                    fmt = filetype(str(tmp_path))
                frames = read_frames(tmp_path, fmt)
                desc = describe_frames(frames)
            except Exception as exc:  # noqa: BLE001 - any parse failure is per-file
                msg = str(exc) or type(exc).__name__
                return _failure(name, f"could not read the file as a structure: {msg}")

            with self._lock:
                upload_id = secrets.token_hex(6)
                while upload_id in self._metas or (self.root / upload_id).exists():
                    upload_id = secrets.token_hex(6)
                final_dir = self.root / upload_id
                final_dir.mkdir()
            shutil.move(str(tmp_path), str(final_dir / name))
            meta = {"id": upload_id, "filename": name, "ok": True, "format": fmt,
                    "created_at": iso_utc()}
            meta.update({k: v for k, v in desc.items() if k not in ("xyz", "frames_xyz")})
            tmp_meta = final_dir / "meta.json.tmp"
            tmp_meta.write_text(json.dumps(meta, indent=1))
            os.replace(tmp_meta, final_dir / "meta.json")
            with self._lock:
                self._metas[upload_id] = meta
                self._xyz_cache[upload_id] = {"xyz": desc["xyz"],
                                              "frames_xyz": desc["frames_xyz"]}
            info = public_info(meta, full=True)
            info["xyz"] = desc["xyz"]
            info["frames_xyz"] = desc["frames_xyz"]
            return info
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _failure(filename: str, message: str) -> Dict[str, Any]:
    return {"id": None, "filename": filename, "ok": False, "error": message}
