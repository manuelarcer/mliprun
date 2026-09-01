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


class TestCLIForwardsSevenNetTask:
    """Same contract as the UMA task above, for SevenNet (CANON C1)."""

    def _structure(self, tmp_path):
        from ase.build import bulk
        from ase.io import write
        path = tmp_path / "POSCAR"
        write(str(path), bulk("Cu", "fcc", a=3.6))
        return path

    def test_md_run_forwards_the_sevennet_task(self, tmp_path, monkeypatch):
        structure = self._structure(tmp_path)
        captured = {}
        monkeypatch.setattr("mliprun.cli.commands.md.setup_calculator",
                            lambda atoms, *a, **k: atoms)
        monkeypatch.setattr("mliprun.cli.commands.md.validate_mlip",
                            lambda *a, **k: None)
        monkeypatch.setattr("mliprun.cli.commands.md.run_md",
                            lambda **kwargs: captured.update(kwargs))

        result = CliRunner().invoke(md_app, [
            "--structure", str(structure), "--mlip", "7net-omni",
            "--sevennet-task", "oc20", "--steps", "1",
        ])

        assert result.exit_code == 0, result.output
        assert captured["sevennet_task"] == "oc20"

    def test_optimize_run_rejects_a_multi_task_tag_without_a_task(self, tmp_path):
        from unittest.mock import patch
        from mliprun.cli.commands.optimize import app as optimize_app

        structure = self._structure(tmp_path)
        with patch("mliprun.cli.utils.SEVENN_AVAILABLE", True):
            result = CliRunner().invoke(optimize_app, [
                "run", "--structure", str(structure), "--mlip", "7net-omni",
            ])

        assert result.exit_code != 0
        assert "--sevennet-task is required" in result.output

    def test_auto_detected_sevennet_tag_is_also_validated(self, tmp_path):
        # `--mlip auto` resolving to a multi-task tag must still stop for a
        # missing task; the old code only validated the explicit branch, so
        # auto slipped past into a much worse error inside SevenNet.
        from unittest.mock import patch
        from mliprun.cli.commands.optimize import app as optimize_app

        structure = self._structure(tmp_path)
        with patch("mliprun.cli.commands.optimize.detect_mlip",
                   return_value="7net-omni"), \
             patch("mliprun.cli.utils.SEVENN_AVAILABLE", True):
            result = CliRunner().invoke(optimize_app, [
                "run", "--structure", str(structure), "--mlip", "auto",
            ])

        assert result.exit_code != 0
        assert "--sevennet-task is required" in result.output


def _option_names(typer_app, command_name=None):
    """Return the option strings a typer command actually declares.

    Introspection rather than grepping `--help`: MLIP_HELP mentions
    "--sevennet-task" in its prose, so a substring check on the help text
    passes whether or not the option exists.
    """
    import typer.main
    click_obj = typer.main.get_command(typer_app)
    if command_name is not None:
        click_obj = click_obj.commands[command_name]
    names = []
    for param in click_obj.params:
        names.extend(getattr(param, "opts", []))
    return names


class TestSevenNetTaskOptionIsExposed:
    @pytest.mark.parametrize("module,command_name", [
        ("mliprun.cli.commands.optimize", "run"),
        ("mliprun.cli.commands.optimize", "batch"),
        ("mliprun.cli.commands.md", None),
    ])
    def test_command_declares_the_option(self, module, command_name):
        import importlib
        app_obj = importlib.import_module(module).app
        assert "--sevennet-task" in _option_names(app_obj, command_name)


class TestOptParamsRecordsTheTask:
    def test_task_written_for_a_sevennet_run(self, tmp_path):
        from mliprun.cli.commands.optimize import _write_params
        path = tmp_path / "opt_params.txt"
        _write_params(path, "7net-omni", "omat", "omat_pbe", "cuda", False,
                      "POSCAR", "bfgs", 0.05, 200, True, tmp_path,
                      sevennet_task="oc20")
        text = path.read_text()
        assert "SevenNet task:     oc20" in text
        assert "UMA task" not in text

    def test_task_omitted_for_a_non_sevennet_run(self, tmp_path):
        from mliprun.cli.commands.optimize import _write_params
        path = tmp_path / "opt_params.txt"
        _write_params(path, "uma-s-1p2", "oc20", "omat_pbe", "cuda", False,
                      "POSCAR", "bfgs", 0.05, 200, True, tmp_path,
                      sevennet_task=None)
        text = path.read_text()
        assert "SevenNet task" not in text
        assert "UMA task:          oc20" in text
