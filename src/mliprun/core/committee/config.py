"""Committee declaration: the level-of-theory table and the YAML file.

Lives in core and imports nothing from the CLI layer. The per-tag task and
head *requirements* (which tags need a task, which reject one) are not
duplicated here: they are enforced by ``build_calculator`` inside each
worker, so a missing task surfaces as a member start failure before the
optimizer runs, with the same message a single-model run would print.

Design: docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

#: A tag/task combination this table does not recognise. Treated as
#: *possibly* mixed: it warns rather than passing silently. Being wrong in
#: this direction costs a spurious warning; being wrong the other way would
#: label an RPBE-vs-PBE comparison as an error bar.
UNKNOWN_LEVEL = "unknown"

# ---------------------------------------------------------------------------
# Level-of-theory table
#
# DELIBERATELY INCOMPLETE. A combination is listed only where the underlying
# dataset's level of theory is unambiguous; everything else resolves to
# UNKNOWN_LEVEL and warns. Adding a row is a deliberate act that changes
# whether a committee is reported as same-level, so it needs the same care as
# a CANON entry -- do not add one to silence a warning.
#
# Labels are compared for equality, nothing more. Their exact spelling is a
# display detail; what matters is that two members at the same level produce
# the same string.
# ---------------------------------------------------------------------------

#: UMA task heads (fairchem-core).
_UMA_LEVELS = {
    "oc20": "RPBE/OC20",
    "oc22": "PBE+U/OC22",
    "omat": "PBE/OMat24",
}

#: SevenNet inference tasks ("modals").
_SEVENNET_LEVELS = {
    "oc20": "RPBE/OC20",
    "oc22": "PBE+U/OC22",
    "omat24": "PBE/OMat24",
    "matpes_pbe": "PBE/MatPES",
    "matpes_r2scan": "r2SCAN/MatPES",
}

#: Heads of the multi-head MACE foundation checkpoints (mace-mh-*).
_MACE_MH_LEVELS = {
    "oc20_usemppbe": "RPBE/OC20",
    "omat_pbe": "PBE/OMat24",
    "matpes_r2scan": "r2SCAN/MatPES",
}

#: Single-task tags whose level is fixed by the tag alone.
_FIXED_LEVELS = {
    "mace": "PBE/MPtrj",
    "chgnet": "PBE+U/MPtrj",
    "7net-omat": "PBE/OMat24",
}


def resolve_level_of_theory(mlip: str, *, uma_task=None, mace_head=None,
                            sevennet_task=None) -> str:
    """Resolve one member's ``(tag, task-or-head)`` to a level-of-theory label.

    Matching is exact and case-sensitive: ``7net-mf-0`` names its tasks in
    uppercase (``PBE``, ``R2SCAN``) where every other model uses lowercase,
    so case-folding would resolve a task no checkpoint has.

    Returns
    -------
    str
        A label, or :data:`UNKNOWN_LEVEL` for anything this table does not
        recognise -- including an unregistered tag, which is reachable in
        production because ``cli/utils`` deliberately forwards unknown
        ``uma-*`` and ``7net-*`` tags through to their packages unchanged.
    """
    if not isinstance(mlip, str):
        return UNKNOWN_LEVEL
    if mlip in _FIXED_LEVELS:
        return _FIXED_LEVELS[mlip]
    if mlip.startswith("uma-"):
        return _UMA_LEVELS.get(uma_task or "", UNKNOWN_LEVEL)
    if mlip.startswith("mace-mh-"):
        return _MACE_MH_LEVELS.get(mace_head or "", UNKNOWN_LEVEL)
    if mlip.startswith("7net"):
        return _SEVENNET_LEVELS.get(sevennet_task or "", UNKNOWN_LEVEL)
    return UNKNOWN_LEVEL


def is_mixed_theory(levels) -> bool:
    """Whether these members span more than one level of theory.

    Any :data:`UNKNOWN_LEVEL` makes the committee mixed, because an
    unrecognised member *might* sit at a different level and there is no way
    to tell. Mixed does not refuse -- it warns (Juan's call, 2026-09-04) --
    but the flag is stamped into the run record and every CSV row so
    downstream analysis can filter on it without anyone having read the
    terminal.
    """
    levels = set(levels)
    if not levels:
        return False
    return len(levels) > 1 or UNKNOWN_LEVEL in levels


class CommitteeConfigError(ValueError):
    """The committee file is missing, malformed, or internally inconsistent."""


#: Keys a member may carry. Anything else is a typo and is rejected: a
#: silently ignored ``gpus:`` would send every member to the same device.
_MEMBER_KEYS = frozenset({
    "name", "env", "mlip", "uma_task", "mace_head", "sevennet_task",
    "device", "gpu",
})

#: Member names become CSV column headers and worker-log filenames.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+-]*$")


@dataclass(frozen=True)
class MemberSpec:
    """One declared committee member."""

    name: str
    env: str
    python_exe: str
    mlip: str
    uma_task: str | None
    mace_head: str | None
    sevennet_task: str | None
    device: str
    gpu: int | None
    level_of_theory: str

    def as_provenance(self) -> dict:
        return {
            "name": self.name,
            "mlip": self.mlip,
            "uma_task": self.uma_task,
            "mace_head": self.mace_head,
            "sevennet_task": self.sevennet_task,
            "env": self.env,
            "python": self.python_exe,
            "device": self.device,
            "gpu": self.gpu,
            "level_of_theory": self.level_of_theory,
        }


@dataclass(frozen=True)
class CommitteeConfig:
    """A parsed, validated committee declaration."""

    members: tuple
    levels: tuple
    mixed_theory: bool
    sha256: str
    source_path: str

    def as_provenance(self) -> dict:
        """The block the run record stores under ``provenance.committee``."""
        return {
            "members": [m.as_provenance() for m in self.members],
            "levels": list(self.levels),
            "mixed_theory": self.mixed_theory,
            "config_sha256": self.sha256,
            "config_path": self.source_path,
        }

    def mixed_theory_warning(self) -> str:
        """The warn-don't-refuse message. Empty when the levels agree."""
        if not self.mixed_theory:
            return ""
        rows = "\n".join(f"    {m.name}: {m.level_of_theory}"
                         for m in self.members)
        return (
            "This committee spans more than one level of theory:\n"
            f"{rows}\n"
            "  The spread it reports is a comparison BETWEEN levels of "
            "theory, not an error bar. In the 2026-09-04 probe the "
            "across-level difference was 0.363 eV against 0.001-0.019 eV "
            "within a level. The 'unknown' label means this tag/task is not "
            "in mliprun's level-of-theory table, not that the members "
            "disagree. Every CSV row carries mixed_theory=True so downstream "
            "analysis can filter on it."
        )


def python_for_env(env) -> Path:
    """Return the interpreter inside a member env.

    Existence only -- not executability. A path that exists but cannot be
    executed fails at member start with the exec error attached, which is a
    better message than anything guessed here.
    """
    root = Path(os.path.expandvars(str(env))).expanduser()
    for candidate in (root / "bin" / "python", root / "bin" / "python3"):
        if candidate.exists():
            return candidate
    raise CommitteeConfigError(
        f"no interpreter found in env '{env}': expected "
        f"{root / 'bin' / 'python'} or {root / 'bin' / 'python3'}. "
        f"Each committee member needs its own MLIP env with mliprun "
        f"installed in it -- see docs/install/README.md.")


def _member_name(entry: dict, mlip: str) -> str:
    task = (entry.get("uma_task") or entry.get("mace_head")
            or entry.get("sevennet_task"))
    return f"{mlip}@{task}" if task else mlip


def _parse_member(entry, index: int) -> dict:
    if not isinstance(entry, dict):
        raise CommitteeConfigError(
            f"member {index} is not a mapping (got {type(entry).__name__})")
    unknown = sorted(set(entry) - _MEMBER_KEYS)
    if unknown:
        raise CommitteeConfigError(
            f"unknown key(s) {unknown} in member {index}; valid keys are "
            f"{sorted(_MEMBER_KEYS)}")
    for required in ("env", "mlip"):
        if not entry.get(required):
            raise CommitteeConfigError(
                f"member {index} is missing required key '{required}'")
    gpu = entry.get("gpu")
    if isinstance(gpu, bool):
        # bool is a subclass of int -- `gpu: true` must not silently become
        # device index 1.
        raise CommitteeConfigError(
            f"member {index}: gpu must be an integer device index, got "
            f"{gpu!r}")
    if gpu is not None and not isinstance(gpu, int):
        raise CommitteeConfigError(
            f"member {index}: gpu must be an integer device index, got "
            f"{gpu!r}")
    return entry


def load_committee(path) -> CommitteeConfig:
    """Parse and validate a ``committee.yaml``.

    Structural validation only. Whether a tag *requires* a task or head is
    not re-checked here: ``build_calculator`` enforces that inside the
    member's own env, so a missing ``--uma-task`` equivalent fails at member
    start with exactly the message a single-model run would print (CANON C1).

    Raises
    ------
    CommitteeConfigError
        With a message naming the offending member by index.
    """
    import yaml

    path = Path(path)
    if not path.is_file():
        raise CommitteeConfigError(f"committee file not found: {path}")
    try:
        raw = path.read_bytes()
        document = yaml.safe_load(raw.decode("utf-8"))
    except OSError as exc:
        # is_file() only proves the path is a regular file -- a
        # permission-denied (or otherwise unreadable) committee.yaml must
        # not propagate a raw OSError past this module.
        raise CommitteeConfigError(f"could not read {path}: {exc}") from exc
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise CommitteeConfigError(
            f"could not parse {path}: {exc}") from exc

    if not isinstance(document, dict):
        raise CommitteeConfigError(
            f"{path}: top level must be a mapping with a 'members' key")
    entries = document.get("members")
    if not isinstance(entries, list):
        raise CommitteeConfigError(
            f"{path}: 'members' must be a list, got "
            f"{type(entries).__name__}")
    if len(entries) < 2:
        raise CommitteeConfigError(
            f"{path}: a committee needs at least two members; got "
            f"{len(entries)}. A committee of one has no disagreement to "
            f"report.")

    members = []
    used_names = set()
    for index, entry in enumerate(entries):
        entry = _parse_member(entry, index)
        mlip = entry["mlip"]
        explicit = entry.get("name")
        name = explicit or _member_name(entry, mlip)
        if not _NAME_RE.match(str(name)):
            raise CommitteeConfigError(
                f"member {index}: name {name!r} must match "
                f"{_NAME_RE.pattern} -- it becomes a CSV column header and a "
                f"log filename")
        if name in used_names:
            if explicit:
                raise CommitteeConfigError(
                    f"member {index}: duplicate member name {name!r}")
            suffix = 2
            while f"{name}-{suffix}" in used_names:
                suffix += 1
            name = f"{name}-{suffix}"
        used_names.add(name)

        env = str(Path(os.path.expandvars(str(entry["env"]))).expanduser())
        try:
            python_exe = str(python_for_env(env))
        except CommitteeConfigError as exc:
            # python_for_env's own message names only the env path; with
            # several members pointing at similarly-named env directories
            # the reader needs the member index too. Not changed on
            # python_for_env itself -- it is a public interface exercised
            # directly by TestPythonForEnv.
            raise CommitteeConfigError(f"member {index}: {exc}") from exc
        members.append(MemberSpec(
            name=name,
            env=env,
            python_exe=python_exe,
            mlip=mlip,
            uma_task=entry.get("uma_task"),
            mace_head=entry.get("mace_head"),
            sevennet_task=entry.get("sevennet_task"),
            device=entry.get("device", "auto"),
            gpu=entry.get("gpu"),
            level_of_theory=resolve_level_of_theory(
                mlip,
                uma_task=entry.get("uma_task"),
                mace_head=entry.get("mace_head"),
                sevennet_task=entry.get("sevennet_task"),
            ),
        ))

    levels = tuple(sorted({m.level_of_theory for m in members}))
    return CommitteeConfig(
        members=tuple(members),
        levels=levels,
        mixed_theory=is_mixed_theory(levels),
        sha256=hashlib.sha256(raw).hexdigest(),
        source_path=str(path.resolve()),
    )
