# Developing mliprun

Testing, internals and design notes. Contribution rules: [CONTRIBUTING.md](../CONTRIBUTING.md).

## How it works

```mermaid
sequenceDiagram
    actor User
    participant CLI as mlip CLI
    participant Model as MLIP calculator
    participant Files as output files

    User->>CLI: optimize / md / neb run<br/>--structure POSCAR
    CLI->>CLI: pick model (--mlip auto)<br/>UMA→MACE→SevenNet→CHGNet
    CLI->>Model: load model once,<br/>attach to structure
    loop until converged
        CLI->>Model: energy + forces
        Model-->>CLI: results
        CLI->>Files: trajectory + CSV
    end
    CLI->>Files: final structure,<br/>parameter file
    CLI-->>User: summary
```


---

## Testing

Install the test dependencies first (`[dev]` brings pytest and the coverage tools; `[neb]` the optional asetools):
```bash
pip install -e ".[dev,neb]"
```

Run all unit tests (no MLIP required):
```bash
pytest -m "not uma and not mace and not sevenn"
```

Run all tests including UMA integration:
```bash
pytest
```

Run specific test categories:
```bash
pytest tests/test_core_optimize.py     # Optimization tests (EMT)
pytest tests/test_core_md.py           # MD tests (EMT)
pytest tests/test_core_neb.py          # NEB tests (EMT)
pytest tests/test_neb_restart.py       # Restart logic tests
pytest tests/test_cli_commands.py      # CLI help/argument tests
pytest -m uma                          # UMA integration tests only
```


---

## Developer notes

- CLI powered by [`typer`](https://typer.tiangolo.com/)
- Entry points defined in [pyproject.toml](../pyproject.toml) (`[project.scripts]`):
  ```toml
  [project.scripts]
  mlip = "mliprun.cli.main:app"
  md = "mliprun.cli.commands.md:app"
  neb = "mliprun.cli.commands.neb:app"
  autoneb = "mliprun.cli.commands.autoneb:app"
  autoneb-results = "mliprun.cli.commands.autoneb_results:app"
  benchmark = "mliprun.cli.commands.benchmark:app"
  optimize = "mliprun.cli.commands.optimize:app"
  singlepoint = "mliprun.cli.commands.singlepoint:app"
  freq = "mliprun.cli.commands.freq:app"
  ```
- Lazy imports for fast CLI startup (no heavy dependencies loaded until needed)
- Output locations: `optimize`/`singlepoint`/`freq`/`md` write next to the input structure (`singlepoint` and `freq` unless `--output-dir` overrides it); `neb`/`autoneb` write into the current working directory (see [OUTPUTS.md](OUTPUTS.md))
- Plots are opt-in via `--plot`; CSV data is always written
- Shared utilities in `core/utils.py` (fmax calculation, unit conversions)
- Parameter I/O in `core/params_io.py` (reduces duplication across commands)
- Optional integration with [asetools](https://github.com/manuelarcer/asetools) for NEB interpolation sanity checks (`pip install -e ".[neb]"`; the import in `core/neb.py` is guarded, so the platform runs without it)
