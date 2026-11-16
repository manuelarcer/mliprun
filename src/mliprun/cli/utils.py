"""Shared utilities for CLI commands."""
import functools
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path
from typing import Optional

import typer

from mliprun.core.utils import resolve_device as _resolve_device


# ---------------------------------------------------------------------------
# CLI help strings (single source of truth — imported by every command)
# ---------------------------------------------------------------------------

MLIP_HELP = (
    "MLIP model. Use 'auto' (default) to auto-detect, or pass a tag explicitly: "
    "any 'uma-*' name (e.g. 'uma-s-1p2', the current default), 'mace' "
    "(MACE-MP-0 medium), 'mace-mh-1' (multi-head foundation; requires "
    "--mace-head), any '7net-*' tag (e.g. '7net-omni', which requires "
    "--sevennet-task), 'chgnet', 'aqcat25-ev2-midfilm' or "
    "'aqcat25-ev2-infilm' (SandboxAQ AQCat25-EV2; require --aqcat-spin, "
    "never picked by 'auto', singlepoint/optimize/neb only), or the path to "
    "a trained MACE model file "
    "ending in '.model'. Any tag starting with 'uma-' is forwarded "
    "to FAIRChem unchanged, and any unrecognised '7net-*' tag is forwarded to "
    "SevenNet unchanged."
)

AQCAT_SPIN_HELP = (
    "Spin conditioning for the AQCat25-EV2 models (aqcat25-ev2-*): 'on' "
    "(spin-polarized DFT reference) or 'off' (non-spin-polarized). Required "
    "for those models, with no default: spin on versus off changes the "
    "predicted energy by up to several eV, so it cannot be guessed. Rejected "
    "for every other model."
)

AQCAT_FIDELITY_HELP = (
    "Fidelity of the DFT settings the AQCat25-EV2 models are conditioned on: "
    "'high' (default; the AQCat25 settings, RPBE at 500 eV) or 'low' (the "
    "OC20 settings, RPBE at 350 eV). Rejected for every other model."
)

MACE_HEAD_HELP = (
    "Head selection for multi-head MACE foundation models (mace-mh-*). "
    "Required for those models, with no default: the heads are independent "
    "fine-tunes with independent energy zeros. One of 'omat_pbe' (PBE bulk "
    "inorganic), 'oc20_usemppbe' (catalysis on surfaces), 'matpes_r2scan' "
    "(r2SCAN materials), 'mp_pbe_refit_add', 'omol' (molecules), "
    "'spice_wB97M' (organic). Rejected for plain 'mace', which is "
    "single-head, and ignored for non-MACE models."
)

UMA_TASK_HELP = (
    "UMA task head. Required for every 'uma-*' model, with no default: the "
    "heads are independent fine-tunes with independent energy zeros, so a "
    "guessed head silently changes the level of theory. One of 'omat' (bulk "
    "inorganic materials), 'omc' (molecular crystals), 'omol' (molecules), "
    "'oc20' (catalysis on surfaces), 'oc22', 'oc25', or 'odac' (ODAC "
    "dataset). Ignored for non-UMA models."
)

SEVENNET_TASK_HELP = (
    "SevenNet inference task (the 'modal' in SevenNet's own API). Required "
    "for multi-task models and rejected for single-task ones. There is no "
    "default: tasks are independent fine-tunes with independent energy zeros, "
    "so a guessed task silently changes the level of theory. "
    "7net-omni / -i8 / -i12: 'mpa' (PBE+U, SevenNet's recommended general "
    "default), 'oc20' (RPBE, catalysis on surfaces), 'oc22', 'omat24', "
    "'matpes_pbe', 'odac23', 'omol25_low', 'omol25_high', 'spice', 'qcml', "
    "'pet_mad', 'mp_r2scan', 'matpes_r2scan'. 7net-mf-ompa: 'omat24' or "
    "'mpa'. 7net-mf-0: 'PBE' or 'R2SCAN' (uppercase). Names are matched "
    "exactly, so case matters. Ignored for non-SevenNet models."
)

DEVICE_HELP = (
    "Compute device: 'auto' (default; cuda if available, else cpu), "
    "'cuda' (force GPU), or 'cpu' (force CPU). For multi-GPU nodes, set "
    "CUDA_VISIBLE_DEVICES to pick which GPU."
)

PLOT_HELP = (
    "Write PNG plots of the results. Off by default (plotting is opt-in): the "
    "figure/save is IO that dominates short runs, so pass --plot only when you "
    "want the figures. The CSV data is always written and can be plotted later."
)


# ---------------------------------------------------------------------------
# Package availability checks (lightweight, no heavy imports)
# ---------------------------------------------------------------------------

def _major_version(version) -> Optional[int]:
    """Leading integer of a version string, or None if it does not parse."""
    try:
        return int(str(version).split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _fairchem_major() -> Optional[int]:
    """Major version of the installed fairchem-core, or None.

    One distribution name serves two families: fairchem-core 2.x is UMA,
    and fairchem-core 1.x (patched by SandboxAQ) is AQCat25-EV2. An absent
    distribution and an unparseable version both return None, which counts
    as neither family.
    """
    try:
        return _major_version(distribution("fairchem-core").version)
    except PackageNotFoundError:
        return None


@functools.lru_cache(maxsize=1)
def _check_fairchem() -> bool:
    """Check if fairchem-core 2.x (UMA) is available without importing it."""
    major = _fairchem_major()
    return major is not None and major >= 2


@functools.lru_cache(maxsize=1)
def _check_aqcat() -> bool:
    """Check if fairchem-core 1.x (the AQCat25-EV2 line) is available.

    Only the distribution's version is read. Whether the SandboxAQ patch is
    applied is checked when the calculator is loaded (`_load_aqcat_calc`).
    """
    return _fairchem_major() == 1


@functools.lru_cache(maxsize=1)
def _check_sevenn() -> bool:
    """Check if sevenn is available without importing it."""
    try:
        distribution("sevenn")
        return True
    except PackageNotFoundError:
        return False


@functools.lru_cache(maxsize=1)
def _check_mace() -> bool:
    """Check if mace is available without importing it."""
    try:
        distribution("mace-torch")
        return True
    except PackageNotFoundError:
        return False


@functools.lru_cache(maxsize=1)
def _check_chgnet() -> bool:
    """Check if chgnet is available without importing it."""
    try:
        distribution("chgnet")
        return True
    except PackageNotFoundError:
        return False


# Module-level convenience flags (still evaluated once at import)
FAIRCHEM_AVAILABLE = _check_fairchem()
SEVENN_AVAILABLE = _check_sevenn()
MACE_AVAILABLE = _check_mace()
CHGNET_AVAILABLE = _check_chgnet()
AQCAT_AVAILABLE = _check_aqcat()


# ---------------------------------------------------------------------------
# MLIP detection / validation
# ---------------------------------------------------------------------------

# Base URL for the install recipes on GitHub. Update if the repo moves.
_INSTALL_DOCS_BASE = (
    "https://github.com/manuelarcer/mliprun/blob/main/docs/install"
)

# MLIP tag → recipe path (filename, optionally with #anchor). Unknown
# `uma-*` and `mace-mh-*` tags fall through to the package-level recipe
# without an anchor — see `_recipe_for_tag`.
_TAG_TO_RECIPE = {
    "mace": "mace.md#mace",
    "mace-mh-0": "mace.md#mace-mh-0",
    "mace-mh-1": "mace.md#mace-mh-1",
    "uma-s-1p2": "uma.md#uma-s-1p2",
    "7net-omni": "sevenn.md#7net-omni",
    "7net-omni-i8": "sevenn.md#7net-omni-i8",
    "7net-omni-i12": "sevenn.md#7net-omni-i12",
    "7net-mf-ompa": "sevenn.md#7net-mf-ompa",
    "7net-mf-0": "sevenn.md#7net-mf-0",
    "7net-omat": "sevenn.md#7net-omat",
    "7net-l3i5": "sevenn.md#7net-l3i5",
    "7net-0": "sevenn.md#7net-0",
    "7net-0_22may2024": "sevenn.md#7net-0",
    "chgnet": "chgnet.md",
    "aqcat25-ev2-midfilm": "aqcat25.md#aqcat25-ev2-midfilm",
    "aqcat25-ev2-infilm": "aqcat25.md#aqcat25-ev2-infilm",
}

#: AQCat25-EV2 checkpoints (SandboxAQ EquiformerV2 with FiLM conditioning;
#: Allam et al., npj Comput. Mater. 12, 226 (2026)), on the gated Hugging Face
#: repo below. The revision is pinned, so a tag always names the same file:
#: that is what fixes the model identity, and why no checkpoint hash goes
#: into the run record. mliprun ships none of these files (CC BY-NC-SA 4.0).
_AQCAT_REPO_ID = "SandboxAQ/aqcat25-ev2"
_AQCAT_REVISION = "2a7d5574aa7f0addc0678080eb7621a47a892abd"
_AQCAT_MODELS = {
    # Authors' recommended model: trained from scratch on AQCat25 + 20M OC20.
    "aqcat25-ev2-midfilm":
        "checkpoints_aqcat_ev2/ev2-in+midFiLM-AQCat25+OC20-20M_20251008_223220.pt",
    # Cotuned from the OC20 EV2-31M model.
    "aqcat25-ev2-infilm":
        "checkpoints_aqcat_ev2/ev2-inFiLM-OC20-ft-AQCat25+OC20-20M_20251008_223143.pt",
}
_AQCAT_SPIN_CHOICES = ("on", "off")
_AQCAT_FIDELITY_CHOICES = ("high", "low")
_AQCAT_DEFAULT_FIDELITY = "high"

#: Shared refusal text for every entry point that does not support the tags.
AQCAT_UNSUPPORTED_MESSAGE = (
    "AQCat25-EV2 models are supported in singlepoint, optimize and neb only "
    "for now"
)

#: Printed whenever a supported command runs with an AQCat tag.
AQCAT_OUTPUT_NOTE = (
    "Note: the energy is an OC20-style referenced adsorption energy, not a "
    "DFT total energy, and the forces are a direct model output rather than "
    "the energy gradient."
)

#: Optimizers that assume the forces are the gradient of the energy (a line
#: search; a Gaussian-process fit of energy and forces together).
_AQCAT_GRADIENT_OPTIMIZERS = ("bfgsls", "gpmin")

#: SevenNet tag -> selectable inference tasks ("modals" in the SevenNet API).
#: An empty tuple marks a single-task model, which rejects --sevennet-task.
#:
#: Read from the checkpoints themselves on 2026-09-01 (sevenn 0.13.0): the tag
#: list from ``sevenn.util.get_available_pretrained_models()``, the task lists
#: from the ``Modality`` line of ``sevenn cp <tag>``. Note that ``7net-mf-0``
#: names its tasks in UPPERCASE where every other model uses lowercase, so
#: task comparison is exact and never case-folded -- lowercasing user input
#: would send SevenNet a task name its checkpoint does not have.
#:
#: The documented ``7net-nano-*`` tags are absent from the 0.13.0 registry and
#: are deliberately not listed; if a later release adds them, the unknown-tag
#: passthrough in `_validate_sevennet_task` already carries them.
_SEVENNET_OMNI_TASKS = (
    "omat24", "mpa", "omol25_low", "omol25_high", "matpes_pbe",
    "matpes_r2scan", "mp_r2scan", "oc20", "oc22", "spice", "qcml",
    "odac23", "pet_mad",
)

_SEVENNET_MODELS: dict[str, tuple[str, ...]] = {
    "7net-omni": _SEVENNET_OMNI_TASKS,
    "7net-omni-i8": _SEVENNET_OMNI_TASKS,
    "7net-omni-i12": _SEVENNET_OMNI_TASKS,
    "7net-mf-ompa": ("omat24", "mpa"),
    "7net-mf-0": ("PBE", "R2SCAN"),
    "7net-omat": (),
    "7net-l3i5": (),
    "7net-0": (),
    "7net-0_22may2024": (),
}

#: UMA task heads, read from uma-s-1p2's own task registry on 2026-09-01
#: (fairchem-core 2.19.0): the set of datasets across `predict_unit.tasks`.
#: The previous --uma-task help listed only omat/oc20/omol/odac; the model
#: carries three more. Like SevenNet tasks and MACE heads, these are
#: independent fine-tunes with independent energy zeros (CANON C1/C3), so
#: the task is required and never defaulted.
_UMA_TASKS = ("omat", "omc", "omol", "oc20", "oc22", "oc25", "odac")

#: UMA tags whose task list is known to match `_UMA_TASKS`. A `uma-*` tag
#: outside this set still requires a task, but its value is forwarded
#: unvalidated -- a newer checkpoint may carry heads this table has not seen.
_KNOWN_UMA_MODELS = ("uma-s-1p2", "uma-s-1p1", "uma-m-1p1")

#: Heads carried by the multi-head MACE foundation checkpoints, read from
#: mace-mh-1's `heads` attribute on 2026-09-01 (mace-torch 0.3.15).
_MACE_MH_HEADS = ("omat_pbe", "mp_pbe_refit_add", "matpes_r2scan",
                  "oc20_usemppbe", "omol", "spice_wB97M")


def _recipe_for_tag(mlip: str) -> str:
    """Return the recipe path for an MLIP tag (file, optionally with anchor).

    Returns an empty string for unrecognised tags.
    """
    if mlip in _TAG_TO_RECIPE:
        return _TAG_TO_RECIPE[mlip]
    if mlip.startswith("uma-"):
        return "uma.md"
    if mlip.startswith("mace-mh-"):
        return "mace.md"
    if mlip.startswith("7net"):
        return "sevenn.md"
    if is_aqcat_tag(mlip):
        return "aqcat25.md"
    return ""


def is_aqcat_tag(mlip) -> bool:
    """Whether ``mlip`` names an AQCat25-EV2 model (any ``aqcat25-*`` tag)."""
    return isinstance(mlip, str) and mlip.startswith("aqcat25-")


def reject_aqcat_unsupported(mlip, command_name: str) -> None:
    """Stop a command that does not support the AQCat25-EV2 models.

    Raises
    ------
    typer.Exit
        If ``mlip`` is an AQCat tag.
    """
    if is_aqcat_tag(mlip):
        raise typer.Exit(
            f"{AQCAT_UNSUPPORTED_MESSAGE}; '{command_name}' cannot run "
            f"'{mlip}'.")


def resolve_aqcat_fidelity(aqcat_fidelity: Optional[str]) -> str:
    """The fidelity actually used: ``high`` when none was given."""
    return _AQCAT_DEFAULT_FIDELITY if aqcat_fidelity is None else aqcat_fidelity


def echo_aqcat_settings(mlip, aqcat_spin: Optional[str],
                        aqcat_fidelity: Optional[str]) -> None:
    """Echo the spin, the resolved fidelity and the output caveat.

    Silent for every non-AQCat tag.
    """
    if not is_aqcat_tag(mlip):
        return
    typer.echo(f"   AQCat spin: {aqcat_spin}")
    typer.echo(f"   AQCat fidelity: {resolve_aqcat_fidelity(aqcat_fidelity)}")
    typer.echo(f"   {AQCAT_OUTPUT_NOTE}")


def check_aqcat_optimize(mlip, optimizer: str, relax_cell: bool) -> None:
    """Refuse an optimization the AQCat25-EV2 models cannot support.

    Called by the CLI before any model is loaded and again by
    :func:`mliprun.core.optimize.run_optimization`, so API callers hit it too.

    Raises
    ------
    ValueError
        For ``relax_cell`` (the models predict no stress) or for an
        optimizer that assumes the forces are the energy gradient (the
        forces come from a direct output head).
    """
    if not is_aqcat_tag(mlip):
        return
    if relax_cell:
        raise ValueError(
            f"{mlip} predicts no stress, so the cell cannot be relaxed "
            f"(--relax-cell / relax_cell=True). Relax positions only.")
    if str(optimizer).lower() in _AQCAT_GRADIENT_OPTIMIZERS:
        raise ValueError(
            f"Optimizer '{optimizer}' cannot be used with {mlip}: it assumes "
            f"the forces are the gradient of the energy, and this model's "
            f"forces come from a direct output head, so they are not. Use "
            f"fire, bfgs, lbfgs or mdmin.")


def _aqcat_recipe_url(mlip: str = "") -> str:
    recipe = _recipe_for_tag(mlip) if mlip else "aqcat25.md"
    return f"{_INSTALL_DOCS_BASE}/{recipe or 'aqcat25.md'}"


def _install_message(label: str, mlip: str) -> str:
    """Format a 'not available' error pointing at the install recipe.

    Always prints the absolute GitHub URL. Additionally prints the
    repo-relative local path if the recipe file is on disk (source
    checkouts / editable installs).
    """
    recipe = _recipe_for_tag(mlip)
    if not recipe:
        return (
            f"{label} not available, and no install recipe is registered for "
            f"tag '{mlip}'. Browse recipes: {_INSTALL_DOCS_BASE}/README.md"
        )

    recipe_file = recipe.split("#", 1)[0]
    url = f"{_INSTALL_DOCS_BASE}/{recipe}"
    repo_root = Path(__file__).resolve().parents[3]
    on_disk = (repo_root / "docs" / "install" / recipe_file).exists()

    msg = f"{label} not available. See install recipe:\n  Online: {url}"
    if on_disk:
        msg += f"\n  Local:  docs/install/{recipe}"
    return msg


def _no_mlip_message() -> str:
    """Message printed by `detect_mlip` when nothing is installed."""
    url = f"{_INSTALL_DOCS_BASE}/README.md"
    repo_root = Path(__file__).resolve().parents[3]
    on_disk = (repo_root / "docs" / "install" / "README.md").exists()

    msg = (
        "No supported MLIP installed. Pick one and follow its install recipe "
        "(each MLIP needs its own env — see ADR 0001):\n"
        f"  Online: {url}"
    )
    if on_disk:
        msg += "\n  Local:  docs/install/README.md"
    return msg


def _aqcat_only_message() -> str:
    """Message printed by `detect_mlip` in an env holding only AQCat25-EV2."""
    tags = " or ".join(f"--mlip {tag}" for tag in _AQCAT_MODELS)
    return (
        "--mlip auto does not resolve in this environment: its only MLIP is "
        "AQCat25-EV2 (fairchem-core 1.x), and those models are explicit-only. "
        f"Pass {tags} (the authors recommend aqcat25-ev2-midfilm), together "
        "with --aqcat-spin on|off. Supported commands: singlepoint, optimize "
        "and neb."
    )


def detect_mlip() -> str:
    """Detect available MLIP model in order of preference: UMA > MACE > SevenNet > CHGNet.

    UMA is preferred when installed (the most accurate foundation model
    available here), but it is gated on Hugging Face and unusable without first
    requesting access. MACE is therefore placed ahead of SevenNet/CHGNet as the
    readily-usable fallback: a fresh environment with ``pip install mace-torch``
    lands on a working model without any access request.

    The SevenNet pick is ``7net-omni``, that family's recommended model and the
    only one of them carrying surface heads (``oc20``/``oc22``). It is
    multi-task, so in a SevenNet-only environment ``--mlip auto`` resolves here
    and then stops in :func:`validate_mlip` for a missing ``--sevennet-task``.
    That is deliberate: CANON C1 makes the task an explicit decision, and a
    loud stop is recoverable where a wrong energy zero is not.

    The AQCat25-EV2 models are never picked: they are explicit-only. In an
    env where they are the only MLIP, the exit message names the tags and
    ``--aqcat-spin`` instead of claiming nothing is installed.

    Returns
    -------
    str
        Name of detected MLIP model.

    Raises
    ------
    typer.Exit
        If no supported MLIP is found.
    """
    if FAIRCHEM_AVAILABLE:
        return "uma-s-1p2"
    elif MACE_AVAILABLE:
        return "mace"
    elif SEVENN_AVAILABLE:
        return "7net-omni"
    elif CHGNET_AVAILABLE:
        return "chgnet"
    elif AQCAT_AVAILABLE:
        raise typer.Exit(_aqcat_only_message())
    else:
        raise typer.Exit(_no_mlip_message())


def _validate_sevennet_task(mlip: str, sevennet_task: Optional[str]) -> None:
    """Check a SevenNet tag against its task list.

    SevenNet tasks ("modals") are independent fine-tunes with independent
    energy zeros, so CANON C1 makes the task an explicit decision and C3
    forbids mixing tasks inside one energy formula. There is therefore no
    default: a multi-task model with no task stops the run rather than
    picking one.

    Unknown ``7net-*`` tags are forwarded to SevenNet unchanged so that new
    checkpoints work without a code change here. The trade-off -- neither the
    tag nor the task can be checked -- is stated on stdout rather than
    hidden.

    Parameters
    ----------
    mlip : str
        SevenNet tag, already known to start with ``7net``.
    sevennet_task : str or None
        The requested task, or ``None`` if the flag was not given.

    Raises
    ------
    typer.Exit
        If the task is missing, unknown for this model, or given to a
        single-task model.
    """
    if mlip not in _SEVENNET_MODELS:
        typer.echo(
            f"Warning: '{mlip}' is not a SevenNet tag mliprun knows. It will "
            f"be passed to SevenNet unchanged; neither it nor the task "
            f"'{sevennet_task}' can be validated here. Known tags: "
            f"{', '.join(sorted(_SEVENNET_MODELS))}."
        )
        return

    tasks = _SEVENNET_MODELS[mlip]

    if not tasks:
        if sevennet_task is not None:
            raise typer.Exit(
                f"{mlip} is a single-task model: it has no selectable task, "
                f"but --sevennet-task {sevennet_task} was given. Drop the flag."
            )
        return

    if sevennet_task is None:
        raise typer.Exit(
            f"{mlip} is a multi-task model, so --sevennet-task is required "
            f"and has no default: its tasks are independent fine-tunes with "
            f"independent energy zeros, and guessing one would silently "
            f"change the level of theory. Valid tasks: {', '.join(tasks)}."
        )

    if sevennet_task not in tasks:
        raise typer.Exit(
            f"Unknown SevenNet task '{sevennet_task}' for {mlip}. Task names "
            f"are matched exactly, so case matters. "
            f"Valid tasks: {', '.join(tasks)}."
        )


def _validate_uma_task(mlip: str, uma_task: Optional[str]) -> None:
    """Check a UMA tag against its task list.

    Every UMA model is multi-head, so the task is always required. The value
    is only checked for tags whose head list has been verified; a newer
    checkpoint may carry heads this table has not seen, and rejecting a valid
    one would be worse than not checking it.
    """
    if uma_task is None:
        raise typer.Exit(
            f"{mlip} is a multi-head model, so --uma-task is required and "
            f"has no default: the heads are independent fine-tunes with "
            f"independent energy zeros, and guessing one would silently "
            f"change the level of theory. Valid tasks: "
            f"{', '.join(_UMA_TASKS)}."
        )

    if mlip not in _KNOWN_UMA_MODELS:
        typer.echo(
            f"Warning: '{mlip}' is not a UMA tag mliprun has verified, so "
            f"the task '{uma_task}' can be validated only by FAIRChem "
            f"itself. Verified tags: {', '.join(_KNOWN_UMA_MODELS)}."
        )
        return

    if uma_task not in _UMA_TASKS:
        raise typer.Exit(
            f"Unknown UMA task '{uma_task}' for {mlip}. "
            f"Valid tasks: {', '.join(_UMA_TASKS)}."
        )


def is_mace_model_file(mlip: str) -> bool:
    """Whether ``mlip`` names a MACE model file rather than a tag.

    A user-trained MACE potential is passed by path, e.g.
    ``--mlip /path/to/MACE_model.model``. The ``.model`` suffix is what
    ``mace_run_train`` writes for a finished model (the ``*_epoch-N.pt``
    checkpoints are not loadable here), so the suffix alone decides.
    """
    return mlip.endswith(".model")


def _validate_mace_head(mlip: str, mace_head: Optional[str]) -> None:
    """Check a MACE tag against its head list.

    Plain ``mace`` (MACE-MP-0) is single-head and rejects the flag, the same
    way a single-task SevenNet tag rejects ``--sevennet-task``.
    """
    if not mlip.startswith("mace-mh-"):
        if mace_head is not None:
            raise typer.Exit(
                f"{mlip} is a single-head model: it has no selectable head, "
                f"but --mace-head {mace_head} was given. Drop the flag."
            )
        return

    if mace_head is None:
        raise typer.Exit(
            f"{mlip} is a multi-head model, so --mace-head is required and "
            f"has no default: the heads are independent fine-tunes with "
            f"independent energy zeros, and guessing one would silently "
            f"change the level of theory. Valid heads: "
            f"{', '.join(_MACE_MH_HEADS)}."
        )

    if mace_head not in _MACE_MH_HEADS:
        raise typer.Exit(
            f"Unknown MACE head '{mace_head}' for {mlip}. "
            f"Valid heads: {', '.join(_MACE_MH_HEADS)}."
        )


def _validate_aqcat_options(mlip: str, aqcat_spin: Optional[str],
                            aqcat_fidelity: Optional[str]) -> None:
    """Check an AQCat25-EV2 tag and its spin / fidelity options.

    Spin has no default: upstream would guess it from the elements, and spin
    on versus off moves the predicted energy by up to several eV. Fidelity
    defaults to ``high`` (upstream's own default) when not given.
    """
    if mlip not in _AQCAT_MODELS:
        raise typer.Exit(
            f"Unknown AQCat25-EV2 tag '{mlip}'. Known tags: "
            f"{', '.join(_AQCAT_MODELS)}.")
    if aqcat_spin is None:
        raise typer.Exit(
            f"{mlip} needs --aqcat-spin on|off, and it has no default: spin "
            f"on versus off changes the predicted energy by up to several "
            f"eV, so it cannot be guessed.")
    if aqcat_spin not in _AQCAT_SPIN_CHOICES:
        raise typer.Exit(
            f"Unknown --aqcat-spin value '{aqcat_spin}'. Valid values: "
            f"{', '.join(_AQCAT_SPIN_CHOICES)}.")
    if (aqcat_fidelity is not None
            and aqcat_fidelity not in _AQCAT_FIDELITY_CHOICES):
        raise typer.Exit(
            f"Unknown --aqcat-fidelity value '{aqcat_fidelity}'. Valid "
            f"values: {', '.join(_AQCAT_FIDELITY_CHOICES)}.")


def validate_mlip(mlip: str, sevennet_task: Optional[str] = None,
                  uma_task: Optional[str] = None,
                  mace_head: Optional[str] = None,
                  aqcat_spin: Optional[str] = None,
                  aqcat_fidelity: Optional[str] = None) -> None:
    """Validate that the specified MLIP -- and its task, if any -- is usable.

    Parameters
    ----------
    mlip : str
        MLIP model name to validate.
    sevennet_task : str, optional
        SevenNet inference task. Required for multi-task SevenNet tags,
        rejected for single-task ones, ignored for every other MLIP.
    uma_task : str, optional
        UMA task head. Required for every ``uma-*`` tag.
    mace_head : str, optional
        MACE head. Required for ``mace-mh-*`` tags, rejected for plain
        ``mace``.
    aqcat_spin : str, optional
        ``'on'`` or ``'off'``. Required for ``aqcat25-*`` tags, rejected for
        every other model.
    aqcat_fidelity : str, optional
        ``'high'`` (used when None) or ``'low'``. Only for ``aqcat25-*``
        tags, rejected for every other model.

    Raises
    ------
    typer.Exit
        If the specified MLIP is not available or unknown, or if its
        SevenNet task is missing or invalid.
    """
    if mlip == "auto":
        return

    if is_aqcat_tag(mlip):
        if not AQCAT_AVAILABLE:
            raise typer.Exit(_install_message("AQCat25-EV2", mlip))
        _validate_aqcat_options(mlip, aqcat_spin, aqcat_fidelity)
        return

    given = [flag for flag, value in (("--aqcat-spin", aqcat_spin),
                                      ("--aqcat-fidelity", aqcat_fidelity))
             if value is not None]
    if given:
        raise typer.Exit(
            f"{' and '.join(given)} apply only to the AQCat25-EV2 models "
            f"({', '.join(_AQCAT_MODELS)}), but --mlip is '{mlip}'. Drop "
            f"the flag.")

    if mlip.startswith("7net"):
        if not SEVENN_AVAILABLE:
            raise typer.Exit(_install_message("SevenNet", mlip))
        _validate_sevennet_task(mlip, sevennet_task)
        return

    if mlip.startswith("uma-"):
        if not FAIRCHEM_AVAILABLE:
            raise typer.Exit(_install_message("UMA", mlip))
        _validate_uma_task(mlip, uma_task)
        return

    if mlip == "mace" or mlip.startswith("mace-mh-"):
        if not MACE_AVAILABLE:
            raise typer.Exit(_install_message("MACE", mlip))
        _validate_mace_head(mlip, mace_head)
        return

    if is_mace_model_file(mlip):
        if not MACE_AVAILABLE:
            raise typer.Exit(_install_message("MACE", "mace"))
        if not Path(mlip).is_file():
            raise typer.Exit(f"MACE model file not found: {mlip}")
        # Treated as single-head: a trained single-dataset model has no
        # selectable head, so --mace-head is rejected rather than ignored.
        _validate_mace_head(mlip, mace_head)
        return

    if mlip == "chgnet" and not CHGNET_AVAILABLE:
        raise typer.Exit(_install_message("CHGNet", mlip))
    elif not (mlip in ["mace", "chgnet"]
              or mlip.startswith("uma-")
              or mlip.startswith("mace-mh-")):
        raise typer.Exit(
            f"Unknown MLIP: {mlip}. Use any 'uma-*' tag (e.g. 'uma-s-1p2'), "
            f"'mace', 'mace-mh-1', any '7net-*' tag (e.g. '7net-omni'), "
            f"'chgnet', or 'aqcat25-ev2-midfilm' / 'aqcat25-ev2-infilm' "
            f"(with --aqcat-spin). "
            f"See {_INSTALL_DOCS_BASE}/README.md for install recipes."
        )


def resolve_mlip(mlip: str, sevennet_task: Optional[str] = None,
                 uma_task: Optional[str] = None,
                 mace_head: Optional[str] = None,
                 aqcat_spin: Optional[str] = None,
                 aqcat_fidelity: Optional[str] = None) -> str:
    """Detect or validate MLIP and echo the result.

    Combines detect + validate + user echo into a single helper so every
    CLI command doesn't repeat the same pattern.

    Parameters
    ----------
    mlip : str
        MLIP model name or ``"auto"`` for auto-detection.
    sevennet_task, uma_task, mace_head, aqcat_spin, aqcat_fidelity : str, optional
        Head/task selections, forwarded to :func:`validate_mlip`.

    Returns
    -------
    str
        Resolved MLIP model name.
    """
    if mlip == "auto":
        mlip = detect_mlip()
        typer.echo(f"Auto-detected MLIP: {mlip}")
        # An auto-detected tag still has to satisfy its own task rules: the
        # SevenNet pick is multi-task, and skipping this would let it through
        # to a far worse error inside SevenNet itself.
        validate_mlip(mlip, sevennet_task, uma_task, mace_head,
                      aqcat_spin=aqcat_spin, aqcat_fidelity=aqcat_fidelity)
    else:
        validate_mlip(mlip, sevennet_task, uma_task, mace_head,
                      aqcat_spin=aqcat_spin, aqcat_fidelity=aqcat_fidelity)
        typer.echo(f"Using MLIP: {mlip}")
    return mlip


# ---------------------------------------------------------------------------
# Relax-atoms parsing
# ---------------------------------------------------------------------------

def parse_relax_atoms(relax_atoms_str: str, num_atoms: int) -> list[int]:
    """Parse a comma-separated atom index string and validate.

    Parameters
    ----------
    relax_atoms_str : str
        Comma-separated list of atom indices (e.g. ``"0,1,5"``).
    num_atoms : int
        Total number of atoms (for bounds checking).

    Returns
    -------
    list[int]
        Validated list of atom indices.

    Raises
    ------
    typer.Exit
        On invalid format or out-of-range indices.
    """
    try:
        indices = [int(i.strip()) for i in relax_atoms_str.split(",")]
    except ValueError:
        typer.echo("Error: --relax-atoms must be a comma-separated list of integers.")
        raise typer.Exit(code=1)

    invalid = [i for i in indices if i < 0 or i >= num_atoms]
    if invalid:
        typer.echo(f"Error: Invalid atom indices {invalid}. Must be between 0 and {num_atoms - 1}.")
        raise typer.Exit(code=1)

    return indices


# ---------------------------------------------------------------------------
# Calculator setup (lazy imports via lru_cache)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_mace_mp():
    from mace.calculators import mace_mp
    return mace_mp


@functools.lru_cache(maxsize=1)
def _load_sevenn_calculator():
    from sevenn.calculator import SevenNetCalculator
    return SevenNetCalculator


@functools.lru_cache(maxsize=1)
def _load_fairchem():
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    return pretrained_mlip, FAIRChemCalculator


@functools.lru_cache(maxsize=1)
def _load_chgnet_calculator():
    from chgnet.model.dynamics import CHGNetCalculator
    return CHGNetCalculator


@functools.lru_cache(maxsize=1)
def _load_aqcat_calc():
    """Return SandboxAQ's ``patched_calc`` factory from the patched fairchem.

    ``patched_calc`` exists only after the SandboxAQ ``ase_utils.py`` has been
    copied over fairchem-core 1.10.0's own, so an ImportError here means the
    env has fairchem-core 1.x without the patch.
    """
    try:
        from fairchem.core.common.relaxation.ase_utils import patched_calc
    except ImportError as exc:
        raise ImportError(
            "This environment has fairchem-core 1.x but not the SandboxAQ "
            "AQCat25-EV2 patch (fairchem.core.common.relaxation.ase_utils "
            "has no patched_calc). Apply the two patched files as the "
            f"install recipe describes: {_aqcat_recipe_url()}"
        ) from exc
    return patched_calc


# MACE foundation-model tag → release URL. Values are stable GitHub release
# assets owned by ACEsuit/mace-foundations.
_MACE_FOUNDATION_URLS = {
    "mace-mh-1": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
    "mace-mh-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model",
}


def _ensure_mace_foundation_checkpoint(tag: str) -> str:
    """Return a local path to the requested MACE foundation checkpoint,
    downloading it into ``~/.cache/mace/`` if it is not already present.
    """
    import os
    import urllib.request

    cache_dir = os.path.expanduser("~/.cache/mace")
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, f"{tag}.model")
    if os.path.exists(target):
        return target

    url = _MACE_FOUNDATION_URLS.get(tag)
    if url is None:
        raise typer.Exit(f"Unknown MACE foundation tag: {tag}")

    typer.echo(f"Downloading {tag} from {url} → {target}")
    urllib.request.urlretrieve(url, target)
    return target


def _is_hf_access_error(exc: BaseException) -> bool:
    """Whether a Hugging Face download failed on access (gated / no login).

    Matched by class name and HTTP status rather than by importing
    ``huggingface_hub.errors``, whose location has moved between releases.
    An unauthenticated request to a gated repo can surface as any of these.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & {"GatedRepoError", "RepositoryNotFoundError",
                "LocalEntryNotFoundError"}:
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403)


def _ensure_aqcat_checkpoint(tag: str) -> str:
    """Return a local path to an AQCat25-EV2 checkpoint.

    Downloads it into the Hugging Face cache on first use, from the pinned
    revision of the gated repo; later calls reuse the cached file.

    Raises
    ------
    ValueError
        For a tag with no checkpoint.
    ImportError
        If huggingface_hub is missing from the env.
    RuntimeError
        On an access failure (no login, or access not granted), pointing at
        the install recipe.
    """
    filename = _AQCAT_MODELS.get(tag)
    if filename is None:
        raise ValueError(
            f"Unknown AQCat25-EV2 tag: {tag}. Known tags: "
            f"{', '.join(_AQCAT_MODELS)}.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is needed to fetch the AQCat25-EV2 checkpoint "
            f"and is missing from this env. See {_aqcat_recipe_url(tag)}"
        ) from exc
    try:
        return hf_hub_download(repo_id=_AQCAT_REPO_ID, filename=filename,
                               revision=_AQCAT_REVISION)
    except Exception as exc:  # noqa: BLE001 -- re-raised unless access failed
        if not _is_hf_access_error(exc):
            raise
        raise RuntimeError(
            f"Could not download {tag} from the gated Hugging Face repo "
            f"{_AQCAT_REPO_ID} ({type(exc).__name__}: {exc}). Request access "
            f"on the repo page and run `hf auth login` in this env; on a node "
            f"without network access, pre-fetch the checkpoint first. See the "
            f"install recipe: {_aqcat_recipe_url(tag)}"
        ) from exc


#: (spin, fidelity) of the first AQCat25-EV2 calculator built in this
#: process. Upstream's ``patched_calc`` patches ``AtomsToGraphs.convert`` at
#: class level on every call, so the most recent flags apply to every AQCat
#: calculator in the process. Holding one pair per process keeps a second,
#: different calculator from silently changing the first one's energies.
_AQCAT_FLAGS_IN_PROCESS: Optional[tuple] = None


def _build_aqcat_calculator(mlip: str, aqcat_spin: Optional[str],
                            aqcat_fidelity: Optional[str], device: str):
    """Build an AQCat25-EV2 calculator: the one path for every caller.

    Used by :func:`build_calculator` and by
    :meth:`mliprun.core.neb.CustomNEB.setup_calculator`. Both flags are
    always passed to upstream explicitly, so it never guesses the spin from
    the elements.

    Raises
    ------
    ValueError
        For an unknown tag, a missing or invalid spin, an invalid fidelity,
        a resolved device other than ``cpu``/``cuda``, or a (spin, fidelity)
        pair that differs from the one already built in this process.
    """
    global _AQCAT_FLAGS_IN_PROCESS
    if mlip not in _AQCAT_MODELS:
        raise ValueError(
            f"Unknown AQCat25-EV2 tag '{mlip}'. Known tags: "
            f"{', '.join(_AQCAT_MODELS)}.")
    if aqcat_spin is None:
        raise ValueError(
            f"{mlip} needs an explicit spin setting: spin on versus off "
            f"changes the predicted energy by up to several eV, so it cannot "
            f"be guessed. Pass aqcat_spin='on' or 'off' (CLI: --aqcat-spin).")
    if aqcat_spin not in _AQCAT_SPIN_CHOICES:
        raise ValueError(
            f"Unknown aqcat_spin '{aqcat_spin}'. Valid values: "
            f"{', '.join(_AQCAT_SPIN_CHOICES)}.")
    fidelity = resolve_aqcat_fidelity(aqcat_fidelity)
    if fidelity not in _AQCAT_FIDELITY_CHOICES:
        raise ValueError(
            f"Unknown aqcat_fidelity '{aqcat_fidelity}'. Valid values: "
            f"{', '.join(_AQCAT_FIDELITY_CHOICES)}.")
    resolved = _resolve_device(device)
    if resolved not in ("cpu", "cuda"):
        raise ValueError(
            f"{mlip} runs on 'cpu' or 'cuda' only, but the device resolved "
            f"to '{resolved}'.")
    pair = (aqcat_spin, fidelity)
    if _AQCAT_FLAGS_IN_PROCESS is not None and pair != _AQCAT_FLAGS_IN_PROCESS:
        held_spin, held_fidelity = _AQCAT_FLAGS_IN_PROCESS
        raise ValueError(
            f"This process already built an AQCat25-EV2 calculator with "
            f"spin={held_spin}, fidelity={held_fidelity}; it cannot build one "
            f"with spin={aqcat_spin}, fidelity={fidelity}. Upstream applies "
            f"the flags by patching a class attribute on every build, so the "
            f"newest flags would silently apply to every AQCat calculator in "
            f"the process. Run the other setting in a separate process.")
    patched_calc = _load_aqcat_calc()
    checkpoint = _ensure_aqcat_checkpoint(mlip)
    # Held before the call: patched_calc applies its class-level patch as it
    # runs, so a failure part-way through may already have changed it.
    _AQCAT_FLAGS_IN_PROCESS = pair
    return patched_calc(checkpoint, is_spin_off=(aqcat_spin == "off"),
                        is_low_fi=(fidelity == "low"),
                        cpu=(resolved == "cpu"))


def build_calculator(mlip: str, uma_task: Optional[str] = None,
                     device: str = "auto",
                     mace_head: Optional[str] = None,
                     sevennet_task: Optional[str] = None,
                     aqcat_spin: Optional[str] = None,
                     aqcat_fidelity: Optional[str] = None):
    """Build and return an ASE calculator for the given MLIP choice.

    This is the expensive step: it loads the model weights into memory (and,
    for GPU runs, onto the device). Building it once and reusing the returned
    object across many structures avoids paying the load cost repeatedly --
    see :func:`setup_calculator`, which builds and attaches in one call.

    Parameters
    ----------
    mlip : str
        MLIP model name.
    uma_task : str, optional
        Task name for UMA models (default: ``"omat"``).
    device : str, optional
        Compute device: ``"auto"`` (default; cuda if available else cpu),
        ``"cuda"``, or ``"cpu"``.
    mace_head : str, optional
        Head name for multi-head MACE foundation models (mace-mh-*). Only
        used when ``mlip`` matches one of those tags. Default ``"omat_pbe"``
        (PBE bulk inorganic). Use ``"oc20_usemppbe"`` for catalysis on
        surfaces.
    sevennet_task : str, optional
        Inference task ("modal") for multi-task SevenNet models. Only used
        for ``7net*`` tags, and deliberately has no default -- see
        :func:`validate_mlip`, which rejects a multi-task tag without one.
        ``None`` omits the argument entirely, which is what a single-task
        checkpoint requires.
    aqcat_spin : str, optional
        ``'on'`` or ``'off'`` for the ``aqcat25-*`` tags. Required for them,
        with no default; ignored for every other model.
    aqcat_fidelity : str, optional
        ``'high'`` (used when None) or ``'low'`` for the ``aqcat25-*`` tags.

    Returns
    -------
    ase.calculators.calculator.Calculator
        The ready-to-use ASE calculator. Assign it to ``atoms.calc``.

    Raises
    ------
    ValueError
        If a multi-head model is requested with no head, or an AQCat25-EV2
        model with no spin. `validate_mlip` catches this earlier on the CLI
        path, but direct API callers never reach it, so the guard is
        repeated here rather than left to the upstream package to fail on.
        Also for the AQCat limits in :func:`_build_aqcat_calculator`.
    """
    device = _resolve_device(device)

    if mlip == "mace":
        mace_mp = _load_mace_mp()
        return mace_mp(model="medium", device=device)

    elif is_mace_model_file(mlip):
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=mlip, device=device)

    elif mlip.startswith("mace-mh-"):
        if mace_head is None:
            raise ValueError(
                f"{mlip} is a multi-head MACE model and needs an explicit "
                f"head: the heads are independent fine-tunes with independent "
                f"energy zeros, so one cannot be guessed. Pass mace_head=... "
                f"(CLI: --mace-head). Valid heads: "
                f"{', '.join(_MACE_MH_HEADS)}."
            )
        from mace.calculators import MACECalculator
        ckpt = _ensure_mace_foundation_checkpoint(mlip)
        return MACECalculator(model_paths=ckpt, device=device, head=mace_head)

    elif mlip.startswith("7net"):
        if sevennet_task is None and _SEVENNET_MODELS.get(mlip, ()):
            raise ValueError(
                f"{mlip} is a multi-task SevenNet model and needs an explicit "
                f"task: its tasks are independent fine-tunes with independent "
                f"energy zeros, so one cannot be guessed. Pass "
                f"sevennet_task=... (CLI: --sevennet-task). Valid tasks: "
                f"{', '.join(_SEVENNET_MODELS[mlip])}."
            )
        SevenNetCalculator = _load_sevenn_calculator()
        if sevennet_task is None:
            # Single-task checkpoints reject `modal`; omit it rather than
            # passing None.
            return SevenNetCalculator(mlip, device=device)
        return SevenNetCalculator(mlip, modal=sevennet_task, device=device)

    elif mlip.startswith("uma-"):
        if uma_task is None:
            raise ValueError(
                f"{mlip} is a multi-head UMA model and needs an explicit "
                f"task: the heads are independent fine-tunes with independent "
                f"energy zeros, so one cannot be guessed. Pass uma_task=... "
                f"(CLI: --uma-task). Valid tasks: {', '.join(_UMA_TASKS)}."
            )
        pretrained_mlip, FAIRChemCalculator = _load_fairchem()
        predictor = pretrained_mlip.get_predict_unit(mlip, device=device)
        return FAIRChemCalculator(predictor, task_name=uma_task)

    elif mlip == "chgnet":
        CHGNetCalculator = _load_chgnet_calculator()
        return CHGNetCalculator(use_device=device)

    elif is_aqcat_tag(mlip):
        return _build_aqcat_calculator(mlip, aqcat_spin, aqcat_fidelity,
                                       device)

    return None


def setup_calculator(atoms, mlip: str, uma_task: Optional[str] = None,
                     device: str = "auto",
                     mace_head: Optional[str] = None,
                     sevennet_task: Optional[str] = None,
                     aqcat_spin: Optional[str] = None,
                     aqcat_fidelity: Optional[str] = None):
    """Attach a freshly built calculator to ``atoms`` based on MLIP choice.

    Convenience wrapper that builds the calculator (see
    :func:`build_calculator`) and attaches it. For a series of structures
    sharing one model, call :func:`build_calculator` once and assign the
    returned object to each ``atoms.calc`` instead, to load the model only
    once.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object to attach calculator to.
    mlip : str
        MLIP model name.
    uma_task : str, optional
        Task name for UMA models (default: ``"omat"``).
    device : str, optional
        Compute device: ``"auto"`` (default; cuda if available else cpu),
        ``"cuda"``, or ``"cpu"``.
    mace_head : str, optional
        Head name for multi-head MACE foundation models (mace-mh-*).
    sevennet_task : str, optional
        Inference task ("modal") for multi-task SevenNet models. No default;
        see :func:`build_calculator`.
    aqcat_spin, aqcat_fidelity : str, optional
        Spin and fidelity for the ``aqcat25-*`` tags; see
        :func:`build_calculator`.

    Returns
    -------
    ase.Atoms
        Atoms object with calculator attached.
    """
    atoms.calc = build_calculator(mlip, uma_task, device=device,
                                  mace_head=mace_head,
                                  sevennet_task=sevennet_task,
                                  aqcat_spin=aqcat_spin,
                                  aqcat_fidelity=aqcat_fidelity)
    return atoms


# ---------------------------------------------------------------------------
# Parameter provenance (for the run record)
# ---------------------------------------------------------------------------

#: Click's provenance vocabulary -> the record's, keyed by the ParameterSource
#: enum member NAME (a plain string) rather than the member object. Typer >=0.16
#: vendors its own copy of click (``typer._click``), so a command's ``ctx`` yields
#: ``typer._click.core.ParameterSource`` members -- a DIFFERENT enum class than
#: top-level ``click.core.ParameterSource``, which never compares or hashes equal
#: to it. Matching on ``.name`` ("COMMANDLINE", "DEFAULT", ...) is identical across
#: both and survives future click/typer splits. DEFAULT_MAP means a config file
#: supplied the value; we report it as "user" because a human wrote it.
_SOURCE_LABELS = {
    "COMMANDLINE": "user",
    "ENVIRONMENT": "env",
    "PROMPT": "prompt",
    "DEFAULT": "default",
    "DEFAULT_MAP": "user",
}


def param_sources_from_ctx(ctx) -> dict:
    """Map each CLI parameter name to where its value came from.

    Returns an empty dict when no context is available, which the record
    module then reports as ``unspecified`` rather than guessing.
    """
    if ctx is None:
        return {}
    sources = {}
    for name in getattr(ctx, "params", {}):
        try:
            src = ctx.get_parameter_source(name)
        except Exception:  # noqa: BLE001 -- provenance is best-effort
            continue
        # Match on the enum member's NAME, not the member object: typer vendors
        # its own click, so ctx yields typer._click ParameterSource members that
        # do not hash-equal the top-level click enum. ``.name`` is identical
        # across both; None guards a missing source or an unexpected return.
        key = getattr(src, "name", None)
        label = _SOURCE_LABELS.get(key)
        if label is None:
            continue
        sources[name] = label
    return sources
