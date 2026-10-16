import typer

from mliprun.cli.commands import (
    autoneb,
    autoneb_results,
    benchmark,
    doctor,
    freq,
    md,
    neb,
    optimize,
    singlepoint,
)
# Only typer is imported by this module; fastapi/uvicorn load inside the command.
from mliprun.webapp.cli import webapp_command

app = typer.Typer(help="mliprun: optimization, MD, NEB, AutoNEB, and benchmarking with MLIP models.")

app.add_typer(optimize.app, name="optimize", help="Run geometry optimization on a structure")
app.add_typer(md.app, name="md", help="Run MD simulations")
app.add_typer(neb.app, name="neb", help="Run NEB interpolation and relaxation")
app.add_typer(autoneb.app, name="autoneb", help="Run AutoNEB with dynamic image insertion")
app.add_typer(autoneb_results.app, name="autoneb-results", help="Extract and visualize AutoNEB results")
app.add_typer(benchmark.app, name="benchmark", help="Run MLIP benchmark on a structure")
app.add_typer(singlepoint.app, name="singlepoint",
              help="Evaluate a structure once: energy, forces, stress")
app.add_typer(freq.app, name="freq",
              help="Vibrational frequencies by finite differences")
app.command("doctor")(doctor.doctor)
app.command("webapp", help="Start the local web app (needs the 'webapp' extra)")(webapp_command)


if __name__ == "__main__":
    app()
