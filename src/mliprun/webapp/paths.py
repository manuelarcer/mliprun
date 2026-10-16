"""Run-folder naming, location checks and safe tar extraction (stdlib only).

Shared by the server (local locations) and by the remote agent (remote
locations), so both sides apply exactly the same rules. ``jobs.py``
re-exports the names that used to live there.
"""
from __future__ import annotations

import io
import os
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Union

JOBS_INDEX_NAME = "_jobs"
RESERVED_FOLDERS = frozenset({"_uploads", JOBS_INDEX_NAME})
MAX_FOLDER_LEN = 100
_SLUG_BAD = re.compile(r"[^A-Za-z0-9._+-]")
_SLUG_DASHES = re.compile(r"-{2,}")


# --------------------------------------------------------------------------- #
# Run folder naming and locations (PLAN 15.1)
# --------------------------------------------------------------------------- #
def slugify_folder(name: Any) -> str:
    """Folder slug: chars outside ``[A-Za-z0-9._+-]`` -> ``-``, dash runs collapsed,
    leading ``.``/``-`` stripped, at most 100 chars. May return ``""``."""
    text = _SLUG_DASHES.sub("-", _SLUG_BAD.sub("-", str(name or "")))
    return text.lstrip(".-")[:MAX_FOLDER_LEN]


def folder_problem(folder: str, reserved: Iterable[str] = RESERVED_FOLDERS) -> Optional[str]:
    """Why ``folder`` (already slugged) is not usable, or None."""
    if not folder:
        return "the folder name is empty after removing unsupported characters"
    if folder in reserved:
        return f"{folder!r} is a reserved name"
    return None


def suffix_width(n_jobs: int) -> int:
    """Zero-pad width of the ``-<kk>`` suffix for ``n_jobs`` jobs."""
    return max(2, len(str(max(n_jobs, 1) - 1)))


def suffixed_folder(base: str, k: int, width: int) -> str:
    """``"<slug(base)>-<kk>"`` with the base shortened so the result fits 100 chars."""
    suffix = f"-{k:0{width}d}"
    return slugify_folder(slugify_folder(base)[:MAX_FOLDER_LEN - len(suffix)] + suffix)


def _writable_dir(path: Path) -> bool:
    return os.access(str(path), os.W_OK | os.X_OK)


def inspect_location(raw: Optional[str], default: Path,
                     reserved: Optional[List[Path]] = None) -> dict:
    """Resolve and check a run location without creating anything.

    Returns ``{"path": Path|None, "display": str, "exists": bool,
    "writable": bool|None, "code": str|None, "error": str|None}``;
    ``writable`` is None when the directory does not exist yet but can be
    created (its nearest existing parent is a writable directory).
    """
    out: Dict[str, Any] = {"path": None, "display": "", "exists": False, "writable": None,
                           "code": None, "error": None}

    def fail(code: str, message: str) -> dict:
        out["code"], out["error"] = code, message
        return out

    if raw is None or not str(raw).strip():
        path = Path(default)
    else:
        text = str(raw).strip()
        try:
            expanded = os.path.expanduser(text)
        except Exception:  # noqa: BLE001
            expanded = text
        out["display"] = expanded
        if "\0" in expanded:
            return fail("location_invalid", "the location contains a NUL character")
        if not os.path.isabs(expanded):
            return fail("location_invalid",
                        "the location must be an absolute path (or start with ~)")
        try:
            path = Path(expanded).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            return fail("location_invalid", f"cannot resolve the location: {exc}")
    out["path"] = path
    out["display"] = str(path)
    for r in reserved or ():
        r = Path(r)
        if path == r or r in path.parents:
            return fail("location_invalid", f"{r.name} is reserved for the web app")
    try:
        if path.exists():
            out["exists"] = True
            if not path.is_dir():
                out["writable"] = False
                return fail("location_invalid", "the location exists but is not a directory")
            out["writable"] = _writable_dir(path)
            if not out["writable"]:
                return fail("location_not_writable", "the location is not writable")
            return out
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir():
            out["writable"] = False
            return fail("location_invalid", f"{parent} is not a directory")
        if not _writable_dir(parent):
            out["writable"] = False
            return fail("location_not_writable",
                        f"the location does not exist and cannot be created "
                        f"({parent} is not writable)")
    except OSError as exc:
        out["writable"] = False
        return fail("location_invalid", f"cannot inspect the location: {exc}")
    return out


def check_paths(location: Optional[str], folders: List[str], default: Path,
                reserved_dirs: Optional[List[Path]] = None,
                reserved_names: Iterable[str] = RESERVED_FOLDERS) -> dict:
    """The ``POST /api/paths/check`` payload for one location (creates nothing)."""
    info = inspect_location(location, default, reserved_dirs)
    out_folders = []
    for raw in folders:
        slug = slugify_folder(raw)
        problem = folder_problem(slug, reserved_names)
        entry = {"input": raw, "folder": slug, "path": None, "exists": False, "error": problem}
        if info["path"] is not None and problem is None:
            path = info["path"] / slug
            entry["path"] = str(path)
            entry["exists"] = os.path.lexists(str(path))
        out_folders.append(entry)
    return {"location": info["display"], "exists": info["exists"],
            "writable": info["writable"], "error": info["error"], "folders": out_folders}


# --------------------------------------------------------------------------- #
# Safe tar handling
# --------------------------------------------------------------------------- #
class UnsafeArchive(Exception):
    """A tar member would escape the destination or is not a plain file/dir."""


def _check_member(member: tarfile.TarInfo) -> None:
    name = member.name
    if not name or "\0" in name:
        raise UnsafeArchive(f"unsafe tar member name {name!r}")
    if name.startswith("/") or PurePosixPath(name).is_absolute() or re.match(r"^[A-Za-z]:", name):
        raise UnsafeArchive(f"absolute path in tar: {name!r}")
    if ".." in PurePosixPath(name).parts or ".." in name.replace("\\", "/").split("/"):
        raise UnsafeArchive(f"'..' in tar member name: {name!r}")
    if member.issym() or member.islnk():
        raise UnsafeArchive(f"link in tar: {name!r}")
    if not (member.isfile() or member.isdir()):
        raise UnsafeArchive(f"unsupported member type in tar: {name!r}")


def safe_extract_tar(source: Union[bytes, BinaryIO], dest: Path) -> int:
    """Extract a tar stream into ``dest`` refusing anything unsafe.

    Rejects absolute names, ``..`` components, links and device files
    (:class:`UnsafeArchive`, nothing after the offending member is written),
    and uses ``tarfile``'s ``filter="data"`` when the interpreter has it.

    Parameters
    ----------
    source : bytes or binary file object
        The tar data (a pipe is fine: it is read as a stream).
    dest : Path
        Existing directory to extract into.

    Returns
    -------
    int
        Number of regular files extracted.
    """
    dest = Path(dest)
    fileobj = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    root = dest.resolve()
    n_files = 0
    with tarfile.open(fileobj=fileobj, mode="r|*") as tar:
        for member in tar:
            _check_member(member)
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise UnsafeArchive(f"tar member escapes the destination: {member.name!r}")
            if hasattr(tarfile, "data_filter"):
                tar.extract(member, str(dest), filter="data")
            else:  # pragma: no cover - interpreters older than the filter backport
                tar.extract(member, str(dest))
            if member.isfile():
                n_files += 1
    return n_files


def write_tar(fileobj: BinaryIO, base: Path, relpaths: Iterable[str]) -> int:
    """Write the regular files ``base/<rel>`` as a tar stream to ``fileobj``.

    Parameters
    ----------
    fileobj : binary file object
        Receives the tar stream (a pipe is fine).
    base : Path
        Directory the relative paths are under.
    relpaths : iterable of str
        POSIX relative paths; each becomes the archive member name.

    Returns
    -------
    int
        Number of files written.
    """
    n = 0
    with tarfile.open(fileobj=fileobj, mode="w|") as tar:
        for rel in relpaths:
            tar.add(str(Path(base) / rel), arcname=rel, recursive=False)
            n += 1
    return n


def tar_bytes(base: Path, relpaths: Iterable[str]) -> bytes:
    """:func:`write_tar` into memory."""
    buf = io.BytesIO()
    write_tar(buf, base, relpaths)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Run directory listing
# --------------------------------------------------------------------------- #
MAX_LISTED_FILES = 10000


def _iso_mtime(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def inside(root: Path, p: Path) -> bool:
    """True if ``p`` resolves (symlinks followed) to a path inside ``root``."""
    try:
        target = p.resolve()
    except (OSError, RuntimeError):
        return False
    return target == root or root in target.parents


def list_files(run_dir: Path) -> List[dict]:
    """``[{"path", "size", "mtime"}]`` of every file under ``run_dir``, sorted.

    Files whose real path is outside ``run_dir`` (symlinks) are skipped.
    """
    files: List[dict] = []
    root = Path(run_dir).resolve()
    for dirpath, dirnames, filenames in os.walk(run_dir):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if not inside(root, p):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            files.append({"path": p.relative_to(run_dir).as_posix(), "size": st.st_size,
                          "mtime": _iso_mtime(st.st_mtime)})
            if len(files) >= MAX_LISTED_FILES:
                break
        if len(files) >= MAX_LISTED_FILES:
            break
    files.sort(key=lambda f: f["path"])
    return files


# --------------------------------------------------------------------------- #
# Copy-back rules for remote jobs
# --------------------------------------------------------------------------- #
LOCAL_AUTHORITATIVE = frozenset({"job.json", "state.json", "events.jsonl"})


def is_local_authoritative(rel: str) -> bool:
    """True for files whose local copy is the master (never fetched from the remote)."""
    return rel in LOCAL_AUTHORITATIVE or rel.startswith("inputs/")


def copy_back_eligible(rel: str, size: int, max_bytes: int) -> bool:
    """Should the remote file ``rel`` of ``size`` bytes be copied back?

    A limit of 0 (or less) means nothing is copied.
    """
    return (not is_local_authoritative(rel)) and max_bytes > 0 and size <= max_bytes
