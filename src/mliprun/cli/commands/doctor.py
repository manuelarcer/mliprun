"""`mlip doctor` — environment self-check.

Reports what a fresh install actually got: package versions, whether the
optional asetools extra is present (and whether it is the real GitHub
package or the unrelated PyPI impostor), which MLIP packages are present,
what `--mlip auto` would resolve to, and torch/CUDA status. Exits 0 iff at
least one MLIP package is installed; exits 1 otherwise with a fix hint, so
scripts and CI can assert on it.
"""
import importlib.util
import platform
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

import typer

# (distribution, MLIP tag reported for `--mlip auto`) in detect_mlip()'s
# preference order: UMA > MACE > SevenNet > CHGNet.
#
# This duplicates detect_mlip()'s tag choices and has drifted from them
# before. tests/test_cli_doctor.py asserts the two agree for every entry --
# keep that test passing rather than editing this table alone.
_MLIP_PACKAGES = [
    ("fairchem-core", "uma-s-1p2"),
    ("mace-torch", "mace"),
    ("sevenn", "7net-omni"),
    ("chgnet", "chgnet"),
]

_INSTALL_DOCS = "docs/install/README.md"


def _package_version(dist_name):
    """Installed version of a distribution, or None if not installed."""
    try:
        return _dist_version(dist_name)
    except PackageNotFoundError:
        return None


def _asetools_status():
    """'ok' | 'wrong-package' | 'missing'.

    PyPI's "asetools" is an unrelated project (Aseprite tooling) that
    shadows the real dependency (github.com/manuelarcer/asetools). The two
    are told apart by the `asetools.pathways` subpackage, which only the
    real one has.
    """
    try:
        if importlib.util.find_spec("asetools") is None:
            return "missing"
        if importlib.util.find_spec("asetools.pathways") is None:
            return "wrong-package"
    except Exception:
        # importable-but-broken behaves like the wrong package
        return "wrong-package"
    return "ok"


def _torch_info():
    """(torch version or None, human-readable CUDA status)."""
    torch_version = _package_version("torch")
    if torch_version is None:
        return None, "n/a (torch not installed)"
    try:
        import torch
        if torch.cuda.is_available():
            cuda = f"available ({torch.cuda.device_count()} device(s))"
        else:
            cuda = "not available (CPU only)"
    except Exception as exc:  # a broken torch install should not crash doctor
        cuda = f"torch import failed: {exc}"
    return torch_version, cuda


def doctor():
    """Check this environment: MLIP packages, asetools health, torch/CUDA."""
    problems = []

    typer.echo("mliprun environment doctor")
    typer.echo("=" * 32)
    typer.echo(f"Python          {platform.python_version()} ({sys.platform})")
    for dist in ("mliprun", "ase"):
        typer.echo(f"{dist:<15} {_package_version(dist) or 'not installed'}")

    status = _asetools_status()
    if status == "ok":
        typer.echo("asetools        OK (real package; NEB sanity checks active)")
    elif status == "wrong-package":
        typer.echo(
            "asetools        WARN: wrong package — PyPI's 'asetools' is an "
            "unrelated project (Aseprite tooling).\n"
            "                Fix: pip uninstall asetools, then "
            'pip install -e ".[neb]" from the repo root'
        )
    else:
        typer.echo(
            "asetools        not installed (optional — NEB interpolation "
            "sanity check disabled).\n"
            '                Enable: pip install -e ".[neb]" from the repo root'
        )

    torch_version, cuda = _torch_info()
    typer.echo(f"torch           {torch_version or 'not installed'}")
    typer.echo(f"CUDA            {cuda}")

    typer.echo("")
    typer.echo("MLIP packages (one env per MLIP — see ADR 0001):")
    installed = []
    for dist, tag in _MLIP_PACKAGES:
        dist_version = _package_version(dist)
        typer.echo(f"  {dist:<14} {dist_version or 'not installed'}")
        if dist_version is not None:
            installed.append(tag)

    if len(installed) > 1:
        typer.echo(
            "  WARN: multiple MLIP packages in one env — their torch/e3nn "
            "pins collide; at least one may be silently broken."
        )

    typer.echo("")
    if installed:
        typer.echo(f"--mlip auto resolves to: {installed[0]}")
    else:
        typer.echo(
            "FAIL: no supported MLIP installed. Pick one and follow its "
            f"install recipe in {_INSTALL_DOCS}."
        )
        problems.append("no MLIP")

    typer.echo("")
    if problems:
        typer.echo(f"Environment NOT usable ({', '.join(problems)}).")
        raise typer.Exit(code=1)
    typer.echo("Environment OK.")
