"""Tests for `md run --barostat-mask`.

The mask decides which cell axes an NPT run is allowed to change. Getting it
wrong is invisible in the output energies and only shows up in the cell, so
the CLI has to reject a malformed mask loudly and record the resolved one.
"""
import re

import pytest
from typer.testing import CliRunner

from mliprun.cli.commands.md import app as md_app

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Strip ANSI styling from rendered CLI output before asserting on it.

    Typer formats usage errors through rich, which colorizes the flag name
    when it decides the output is a terminal -- GitHub Actions and a local
    run disagree about that. Colorized, `--barostat-mask` arrives as
    `ESC[1;36m-ESC[0mESC[1;36m-barostatESC[0mESC[1;36m-maskESC[0m`, so a
    literal substring check passes locally and fails in CI. Assert on the
    text the user reads, not on the styling.
    """
    return _ANSI.sub("", output)


def _structure(tmp_path):
    from ase.build import bulk
    from ase.io import write

    path = tmp_path / "POSCAR"
    write(str(path), bulk("Cu", "fcc", a=3.6, cubic=True))
    return path


def _stub_run(tmp_path, monkeypatch, extra_args):
    """Invoke `md run` with the MLIP layer stubbed out, capturing run_md."""
    structure = _structure(tmp_path)
    captured = {}
    monkeypatch.setattr("mliprun.cli.commands.md.setup_calculator",
                        lambda atoms, *a, **k: atoms)
    monkeypatch.setattr("mliprun.cli.commands.md.validate_mlip",
                        lambda *a, **k: None)
    monkeypatch.setattr("mliprun.cli.commands.md.run_md",
                        lambda **kwargs: captured.update(kwargs))

    # md_app registers a single command, so typer collapses it to a top-level
    # CLI: there is no "run" subcommand to pass.
    result = CliRunner().invoke(md_app, [
        "--structure", str(structure), "--mlip", "mace-mp-0",
        "--steps", "1",
    ] + extra_args)
    return result, captured, structure


class TestBarostatMaskForwarding:
    def test_mask_reaches_run_md(self, tmp_path, monkeypatch):
        result, captured, _ = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "npt", "--temperature", "300", "--pressure", "0",
            "--barostat", "berendsen", "--barostat-mask", "0,0,1",
        ])

        assert result.exit_code == 0, result.output
        assert captured["barostat_mask"] == (0, 0, 1)

    def test_default_is_isotropic(self, tmp_path, monkeypatch):
        """No flag must forward the historical isotropic mask, not None."""
        result, captured, _ = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "npt", "--temperature", "300", "--pressure", "0",
            "--barostat", "berendsen",
        ])

        assert result.exit_code == 0, result.output
        assert captured["barostat_mask"] == (1, 1, 1)

    def test_surrounding_whitespace_is_accepted(self, tmp_path, monkeypatch):
        result, captured, _ = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "npt", "--temperature", "300", "--pressure", "0",
            "--barostat", "berendsen", "--barostat-mask", " 0, 0 ,1 ",
        ])

        assert result.exit_code == 0, result.output
        assert captured["barostat_mask"] == (0, 0, 1)


class TestBarostatMaskRejection:
    """A bad mask must produce a message, not a traceback."""

    def _assert_clean_rejection(self, result):
        output = _plain(result.output)
        assert result.exit_code != 0
        assert "--barostat-mask" in output, output
        # Guard against a false green: before the option existed, click
        # rejected the flag itself with a message that also names it.
        assert "No such option" not in output, output
        # A ValueError escaping to the top would be a traceback in real use.
        assert result.exception is None or isinstance(result.exception, SystemExit), (
            f"unhandled {type(result.exception).__name__}: {result.exception}"
        )

    @pytest.mark.parametrize("bad", ["1,1", "1,1,1,1", "0,0,2", "-1,0,1",
                                     "a,b,c", "", "0,0,1.5"])
    def test_malformed_mask_is_rejected(self, tmp_path, monkeypatch, bad):
        result, captured, _ = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "npt", "--temperature", "300", "--pressure", "0",
            "--barostat", "berendsen", "--barostat-mask", bad,
        ])

        self._assert_clean_rejection(result)
        assert captured == {}, "run_md must not be reached with a bad mask"

    def test_non_default_mask_outside_npt_is_rejected(self, tmp_path, monkeypatch):
        """Silently ignoring it would leave the user believing z was free."""
        result, captured, _ = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "nvt", "--temperature", "300",
            "--barostat-mask", "0,0,1",
        ])

        assert result.exit_code != 0
        assert "npt" in _plain(result.output).lower()
        assert captured == {}, "run_md must not be reached"

    def test_default_mask_outside_npt_is_allowed(self, tmp_path, monkeypatch):
        """The default must stay silent for every ensemble."""
        result, captured, _ = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "nvt", "--temperature", "300",
        ])

        assert result.exit_code == 0, result.output
        assert captured["barostat_mask"] == (1, 1, 1)


class TestBarostatMaskIsReported:
    """`md_params.txt` alone must distinguish a masked run from an isotropic
    one, without opening the trajectory."""

    def test_params_file_and_stdout_carry_the_mask(self, tmp_path, monkeypatch):
        result, _, structure = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "npt", "--temperature", "300", "--pressure", "0",
            "--barostat", "berendsen", "--barostat-mask", "0,0,1",
        ])

        assert result.exit_code == 0, result.output
        params = (structure.parent / "md_params.txt").read_text()
        assert "Barostat mask" in params
        assert "0,0,1" in params
        assert "0,0,1" in _plain(result.output)

    def test_params_file_records_the_default_mask(self, tmp_path, monkeypatch):
        result, _, structure = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "npt", "--temperature", "300", "--pressure", "0",
            "--barostat", "berendsen",
        ])

        assert result.exit_code == 0, result.output
        params = (structure.parent / "md_params.txt").read_text()
        assert "1,1,1" in params

    def test_nvt_params_file_omits_the_mask(self, tmp_path, monkeypatch):
        """The mask is an NPT fact; an NVT block must not imply otherwise."""
        result, _, structure = _stub_run(tmp_path, monkeypatch, [
            "--ensemble", "nvt", "--temperature", "300",
        ])

        assert result.exit_code == 0, result.output
        params = (structure.parent / "md_params.txt").read_text()
        assert "Barostat mask" not in params


class TestBarostatMaskProvenanceThroughCLI:
    """The record must say whether the mask was chosen or merely defaulted.

    `param_sources_from_ctx` keys on the CLI parameter name, and the record
    keys on the `parameters` dict. The two only line up because both call it
    `barostat_mask`; a rename on either side would silently downgrade every
    mask to `unspecified`, so it is asserted rather than assumed.
    """

    def _run(self, tmp_path, monkeypatch, extra_args):
        from ase.calculators.emt import EMT

        structure = _structure(tmp_path)

        def _attach_emt(atoms, *a, **k):
            atoms.calc = EMT()
            return atoms

        monkeypatch.setattr("mliprun.cli.commands.md.setup_calculator", _attach_emt)
        monkeypatch.setattr("mliprun.cli.commands.md.validate_mlip",
                            lambda *a, **k: None)

        result = CliRunner().invoke(md_app, [
            "--structure", str(structure), "--mlip", "mace-mp-0",
            "--steps", "2", "--log-interval", "1", "--traj-interval", "1",
            "--ensemble", "npt", "--temperature", "300", "--pressure", "0",
            "--barostat", "berendsen",
        ] + extra_args)
        return result, structure.parent

    def test_an_explicit_mask_is_tagged_user(self, tmp_path, monkeypatch):
        import json

        result, out_dir = self._run(tmp_path, monkeypatch,
                                    ["--barostat-mask", "0,0,1"])

        assert result.exit_code == 0, result.output
        data = json.loads((out_dir / "mliprun_run.json").read_text())
        entry = data["parameters"]["barostat_mask"]
        assert entry["value"] == [0, 0, 1]
        assert entry["source"] == "user"

    def test_an_omitted_mask_is_tagged_default(self, tmp_path, monkeypatch):
        import json

        result, out_dir = self._run(tmp_path, monkeypatch, [])

        assert result.exit_code == 0, result.output
        entry = json.loads(
            (out_dir / "mliprun_run.json").read_text()
        )["parameters"]["barostat_mask"]
        assert entry["value"] == [1, 1, 1]
        assert entry["source"] == "default"
