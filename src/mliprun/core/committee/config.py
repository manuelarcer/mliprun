"""Committee declaration: the level-of-theory table and the YAML file.

Lives in core and imports nothing from the CLI layer. The per-tag task and
head *requirements* (which tags need a task, which reject one) are not
duplicated here: they are enforced by ``build_calculator`` inside each
worker, so a missing task surfaces as a member start failure before the
optimizer runs, with the same message a single-model run would print.

Design: docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md
"""
from __future__ import annotations

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
