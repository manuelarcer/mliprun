"""Committee declaration: the level-of-theory table and the YAML file.

Per-tag task and head requirements (which tags need a task, which reject one,
which values are valid) ARE checked here, against the same tables the CLI's
``validate_mlip`` uses -- see :func:`_validate_head_task`. An earlier version
of this module claimed ``build_calculator`` enforced them inside the worker.
That was false: it guards only a *missing* head or task on the multi-head
tags, so ``mlip: mace`` with ``mace_head: oc20_usemppbe`` ran MACE-MP-0 while
the member's name, its CSV column and its provenance all described an
RPBE/OC20 head, and ``7net-omat`` with ``sevennet_task: oc20`` resolved its
level label from the tag while the calculator was handed ``modal="oc20"``.

Design: docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

# The reserved non-MLIP tag, imported rather than repeated so the parser and
# the worker can never disagree about what "emt" means. `worker` pulls in only
# stdlib plus `protocol` at import time, so this costs nothing.
from mliprun.core.committee.worker import EMT_TAG

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

#: Version of the level-of-theory table below. BUMP IT ON EVERY EDIT -- a row
#: added, removed, or relabelled.
#:
#: Stamped into ``provenance.committee.level_table_version`` so a record can be
#: re-judged later. The table's incompleteness is self-announcing (an unlisted
#: combination resolves to ``unknown``, which flags the committee as mixed),
#: but its one silent failure mode is a WRONG row: two entries carrying the
#: same label for genuinely different datasets read as same-level, with no
#: signal anywhere and nothing able to detect it at runtime. A stored version
#: number is what lets a reader ask "was this record written under the table
#: that had the bad row?" instead of trusting it blindly.
LEVEL_TABLE_VERSION = 2

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
#:
#: ``mace`` (MACE-MP-0 medium) and ``chgnet`` share one label because they
#: share one training set: MPtrj, the Materials Project relaxation
#: trajectories. The label names the dataset and signals the mixing rather
#: than asserting a functional, because MPtrj applies Hubbard U to specific
#: transition metals in oxides and fluorides only -- so an MPtrj-trained
#: model is effectively plain PBE for a metallic slab and PBE+U for an oxide,
#: and no static label can settle that. Juan's ruling, 2026-09-08.
#:
#: ``7net-0``, ``7net-0_22may2024`` and ``7net-l3i5`` are BELIEVED to be MPtrj
#: too, and are still absent from this table on purpose. The belief comes from
#: SevenNet's documentation; the rule for this table is that a row is added
#: only once the training set is confirmed from the installed package
#: (``sevenn cp <tag>``), which nobody has done for those three. The
#: asymmetry is deliberate, not an oversight: adding an unconfirmed row is
#: exactly the *wrong* row this table's one silent failure mode is about --
#: two entries carrying the same label for genuinely different datasets read
#: as same-level with no signal anywhere. Leaving them out costs a spurious
#: mixed-theory warning; putting them in could silently turn a functional
#: comparison into an error bar.
_FIXED_LEVELS = {
    "mace": "PBE(+U)/MPtrj",
    "chgnet": "PBE(+U)/MPtrj",
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

    def as_provenance(self, measured=None) -> dict:
        """The member's block in the run record.

        Every top-level key here is DECLARED: it is what ``committee.yaml``
        asked for. ``measured`` is what the member's own env reported back
        over the bridge once it had loaded (``worker._versions``): the
        interpreter, ASE, torch and MLIP package that actually ran. The two
        are kept in separate namespaces on purpose, because the difference
        between them is the entire point of recording either.

        ``measured`` is ``None`` when no member ever started -- a committee
        built but not run, or a record written from the config alone. Note
        that the declared ``python`` is a path to an interpreter while
        ``measured["python"]`` is a version string.
        """
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
            "measured": dict(measured) if measured else None,
        }


@dataclass(frozen=True)
class CommitteeConfig:
    """A parsed, validated committee declaration."""

    members: tuple
    levels: tuple
    mixed_theory: bool
    sha256: str
    source_path: str

    def as_provenance(self, measured_versions=None) -> dict:
        """The block the run record stores under ``provenance.committee``.

        Parameters
        ----------
        measured_versions : dict, optional
            ``{member name: versions dict}`` as reported by each member's own
            env at load time -- ``CommitteeCalculator.member_versions``. A
            member with no entry records ``measured: null`` rather than an
            empty dict, so "never started" is distinguishable from "started
            and reported nothing".
        """
        measured_versions = measured_versions or {}
        return {
            "members": [m.as_provenance(measured_versions.get(m.name))
                        for m in self.members],
            "levels": list(self.levels),
            "mixed_theory": self.mixed_theory,
            # Which revision of this module's level-of-theory table produced
            # the labels above. A later correction makes this record
            # re-judgeable instead of silently trusted.
            "level_table_version": LEVEL_TABLE_VERSION,
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


def _validate_head_task(index: int, mlip, uma_task, mace_head,
                        sevennet_task) -> None:
    """Apply the CLI's head/task rules to one declared member.

    This is the same policy ``mliprun.cli.utils.validate_mlip`` enforces on
    ``mlip optimize run``, and it is deliberately not that function: for a
    committee, ``validate_mlip``'s *availability* half is wrong. It would
    reject ``mlip: mace`` because MACE is not importable in the DRIVER
    env -- which is the whole premise of the committee bridge (ADR 0001).
    Only the head/task half applies here; whether the package exists is
    answered inside the member's own env, at member start.

    The tables are imported from the CLI module rather than copied, so a new
    UMA task or SevenNet modal cannot be valid on the command line and
    invalid in a ``committee.yaml``. The import is local because it pulls in
    typer and four ``importlib.metadata`` lookups, which a caller that only
    parses a file should not pay for at import time.

    Unrecognised ``uma-*`` and ``7net-*`` tags pass through unchecked, exactly
    as the CLI forwards them to their packages: a newer checkpoint may carry
    heads these tables have not seen, and rejecting a valid one would be worse
    than not checking it. Nothing is silent about it -- an unlisted
    combination resolves to :data:`UNKNOWN_LEVEL`, which flags the committee
    as mixed theory and warns at startup.

    Raises
    ------
    CommitteeConfigError
        Naming the member index and the offending key.
    """
    if not isinstance(mlip, str):
        # Mirrors resolve_level_of_theory's guard. A non-string tag already
        # fails loudly at member start (build_calculator has no branch for
        # it); turning that into a parse-time rejection would be a new policy
        # this fix has no mandate for.
        return

    # The reserved bridge tag builds ASE's EMT and takes no head or task.
    if mlip == EMT_TAG:
        if uma_task or mace_head or sevennet_task:
            raise CommitteeConfigError(
                f"member {index}: '{EMT_TAG}' is ASE's built-in EMT "
                f"calculator, not an MLIP: it has no selectable task or head. "
                f"Remove the task/head key.")
        return

    from mliprun.cli.utils import (
        _KNOWN_UMA_MODELS,
        _MACE_MH_HEADS,
        _SEVENNET_MODELS,
        _UMA_TASKS,
    )

    if mlip.startswith("7net"):
        if mlip not in _SEVENNET_MODELS:
            return          # unknown tag: forwarded to SevenNet unchanged
        tasks = _SEVENNET_MODELS[mlip]
        if not tasks:
            if sevennet_task is not None:
                raise CommitteeConfigError(
                    f"member {index}: '{mlip}' is a single-task SevenNet "
                    f"model and has no selectable task, but sevennet_task: "
                    f"{sevennet_task!r} was given. Remove the key -- it is "
                    f"ignored by the calculator, so leaving it would record a "
                    f"task this member never used.")
            return
        if sevennet_task is None:
            raise CommitteeConfigError(
                f"member {index}: '{mlip}' is a multi-task SevenNet model, so "
                f"sevennet_task is required and has no default: its tasks are "
                f"independent fine-tunes with independent energy zeros. Valid "
                f"tasks: {', '.join(tasks)}.")
        if sevennet_task not in tasks:
            raise CommitteeConfigError(
                f"member {index}: unknown sevennet_task "
                f"{sevennet_task!r} for '{mlip}'. Task names are matched "
                f"exactly, so case matters. Valid tasks: {', '.join(tasks)}.")
        return

    if mlip.startswith("uma-"):
        if uma_task is None:
            raise CommitteeConfigError(
                f"member {index}: '{mlip}' is a multi-head UMA model, so "
                f"uma_task is required and has no default: the heads are "
                f"independent fine-tunes with independent energy zeros. Valid "
                f"tasks: {', '.join(_UMA_TASKS)}.")
        if mlip in _KNOWN_UMA_MODELS and uma_task not in _UMA_TASKS:
            raise CommitteeConfigError(
                f"member {index}: unknown uma_task {uma_task!r} for "
                f"'{mlip}'. Valid tasks: {', '.join(_UMA_TASKS)}.")
        return

    if mlip.startswith("mace-mh-"):
        if mace_head is None:
            raise CommitteeConfigError(
                f"member {index}: '{mlip}' is a multi-head MACE model, so "
                f"mace_head is required and has no default: the heads are "
                f"independent fine-tunes with independent energy zeros. Valid "
                f"heads: {', '.join(_MACE_MH_HEADS)}.")
        if mace_head not in _MACE_MH_HEADS:
            raise CommitteeConfigError(
                f"member {index}: unknown mace_head {mace_head!r} for "
                f"'{mlip}'. Valid heads: {', '.join(_MACE_MH_HEADS)}.")
        return

    if mlip == "mace" and mace_head is not None:
        raise CommitteeConfigError(
            f"member {index}: 'mace' (MACE-MP-0 medium) is single-head and "
            f"has no selectable head, but mace_head: {mace_head!r} was given. "
            f"Remove the key, or use a 'mace-mh-*' tag if you meant a "
            f"multi-head checkpoint -- the head is ignored by the calculator, "
            f"so leaving it would name this member, its CSV column and its "
            f"provenance after a head that never ran.")


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
    _validate_head_task(index, entry["mlip"], entry.get("uma_task"),
                        entry.get("mace_head"), entry.get("sevennet_task"))
    return entry


def load_committee(path) -> CommitteeConfig:
    """Parse and validate a ``committee.yaml``.

    Structure, plus each member's head/task combination against the same
    tables the CLI checks (:func:`_validate_head_task`). What is NOT checked
    here is whether the member's MLIP package is installed: that question is
    only answerable inside the member's own env, and it is answered there, at
    member start.

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
