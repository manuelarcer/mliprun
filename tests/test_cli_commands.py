"""Tests for CLI command invocations."""
import subprocess
import pytest

from typer.testing import CliRunner

from mliprun.cli.commands.md import app as md_app


def _run_cli(cmd):
    """Run a CLI command and return the result."""
    return subprocess.run(cmd, capture_output=True, text=True)


class TestHelpCommands:
    """Test that --help works for all CLI entry points."""

    @pytest.mark.parametrize("command", [
        ["optimize", "run", "--help"],
        ["optimize", "batch", "--help"],
        ["md", "run", "--help"],
        ["neb", "run", "--help"],
        ["autoneb", "run", "--help"],
        ["autoneb-results", "results", "--help"],
        ["benchmark", "run", "--help"],
    ])
    def test_help_exits_cleanly(self, command):
        result = _run_cli(command)
        assert result.returncode == 0
        assert "Usage" in result.stdout or "Options" in result.stdout


class TestMissingArgs:
    """Test that missing required arguments produce errors."""

    def test_optimize_missing_structure(self):
        result = _run_cli(["optimize", "run", "--no-input"])
        # Should fail because --structure is required
        assert result.returncode != 0

    def test_md_missing_structure(self):
        result = _run_cli(["md", "run", "--no-input"])
        assert result.returncode != 0

    def test_neb_missing_initial_final(self):
        result = _run_cli(["neb", "run"])
        # Should fail: neither --initial/--final nor --restart provided
        assert result.returncode != 0


class TestCLIForwardsHead:
    """The head is the one level-of-theory fact the record cannot infer,
    so the CLI must hand it to the engine (CANON C1)."""

    def test_md_run_forwards_the_uma_task(self, tmp_path, monkeypatch):
        from ase.build import bulk
        from ase.io import write

        structure = tmp_path / "POSCAR"
        write(str(structure), bulk("Cu", "fcc", a=3.6))

        captured = {}
        monkeypatch.setattr("mliprun.cli.commands.md.setup_calculator",
                            lambda atoms, *a, **k: atoms)
        monkeypatch.setattr("mliprun.cli.commands.md.validate_mlip",
                            lambda *a, **k: None)
        monkeypatch.setattr("mliprun.cli.commands.md.run_md",
                            lambda **kwargs: captured.update(kwargs))

        # md_app registers a single command, so typer collapses it to a
        # top-level CLI: there is no "run" subcommand to pass (confirmed via
        # `md --help` / `md run --help` -- the latter only "works" because
        # --help is resolved eagerly, before the extra "run" argument is
        # validated).
        result = CliRunner().invoke(md_app, [
            "--structure", str(structure), "--mlip", "uma-s-1p2",
            "--uma-task", "oc25", "--steps", "1",
        ])

        assert result.exit_code == 0, result.output
        assert captured["uma_task"] == "oc25"
        # The CLI forwards its parsed --mace-head default verbatim; gating a
        # UMA run's mace_head to None is collect_provenance's job, not the
        # CLI's, and is covered by Task 1.
        assert captured["mace_head"] == "omat_pbe"
