"""``mlip webapp``: launch the local web app.

Only typer is imported at module level, so ``mlip --help`` works without the
``webapp`` extra; fastapi/uvicorn are imported inside the command.
"""
from __future__ import annotations

import threading
import webbrowser
from typing import List, Optional

import typer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_RUNS_DIR = "~/.mliprun/webapp-runs"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

INSTALL_HINT = ('The web app needs extra packages: pip install "mliprun[webapp]"\n'
                '(from a source checkout: pip install -e ".[webapp]")')


def allowed_hosts_for(host: str, port: int) -> Optional[List[str]]:
    """Host-header allow-list for a bind address; ``None`` (no check) off loopback."""
    if host in LOOPBACK_HOSTS:
        return [f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"]
    return None


def url_for(host: str, port: int) -> str:
    """The URL to open in a browser for a bind address."""
    shown = host
    if host in ("0.0.0.0", "::", ""):
        shown = "127.0.0.1"
    if ":" in shown:
        shown = f"[{shown}]"
    return f"http://{shown}:{port}/"


def _missing_extra() -> Optional[str]:
    """Name of the first missing web app dependency, or ``None``."""
    import importlib.util

    for module in ("fastapi", "uvicorn"):
        if importlib.util.find_spec(module) is None:
            return module
    if (importlib.util.find_spec("python_multipart") is None
            and importlib.util.find_spec("multipart") is None):
        return "python-multipart"
    return None


def webapp_command(
    host: str = typer.Option(DEFAULT_HOST, "--host",
                             help="Address to bind. Keep the default (this computer only) "
                                  "unless you understand the risk: the app has no login."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="TCP port to listen on."),
    runs_dir: str = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir",
                                 help="Folder for uploads and job run directories."),
    no_browser: bool = typer.Option(False, "--no-browser",
                                    help="Do not open a browser window."),
) -> None:
    """Start the local web app (browser front end for mliprun calculations)."""
    missing = _missing_extra()
    if missing is not None:
        typer.echo(INSTALL_HINT, err=True)
        raise typer.Exit(1)

    from pathlib import Path

    import uvicorn

    from mliprun.webapp.server import create_app

    allowed = allowed_hosts_for(host, port)
    if allowed is None:
        typer.echo(
            f"WARNING: binding to {host}. The web app has no authentication: anyone who "
            f"can reach port {port} can run jobs and read files in the runs directory.",
            err=True)
    app = create_app(Path(runs_dir).expanduser(), allowed_hosts=allowed)
    url = url_for(host, port)
    typer.echo(f"mliprun web app: {url}  (runs dir: {app.state.runs_dir})")
    typer.echo("Press Ctrl+C to stop.")
    if not no_browser:
        timer = threading.Timer(1.0, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    uvicorn.run(app, host=host, port=port, log_level="info")
