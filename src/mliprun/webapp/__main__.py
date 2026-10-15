"""``python -m mliprun.webapp``: same as ``mlip webapp``."""
from __future__ import annotations

import typer

from mliprun.webapp.cli import webapp_command


def main() -> None:
    typer.run(webapp_command)


if __name__ == "__main__":
    main()
