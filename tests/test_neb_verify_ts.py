"""``neb run --verify-ts``: TS verification after a converged climbing-image NEB.

The system is an Al adatom hopping between two neighbouring hollow sites of a
2x2 Al(100) slab (2 layers, bottom layer fixed), described by EMT. It has a
real saddle (the bridge site), so the whole chain runs for real: NEB,
frequencies, overlap with the NEB tangent. EMT replaces the MLIP by patching
``setup_calculator`` and ``resolve_mlip``, so no model is needed.
"""
import json
import re
from pathlib import Path

import pytest
from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import write
from typer.testing import CliRunner

from mliprun.cli.commands.neb import app
from mliprun.core.neb import CustomNEB

runner = CliRunner()


def _hollow_slab(shift):
    """2x2 Al(100), 2 layers, bottom layer fixed, adatom in a hollow site.

    ``shift`` moves the adatom by that fraction of the first cell vector,
    so 0.5 puts it in the neighbouring hollow site.
    """
    slab = fcc100("Al", size=(2, 2, 2), vacuum=5.0)
    add_adsorbate(slab, "Al", 1.5, "hollow")
    slab.positions[-1] += shift * slab.cell[0]
    z = slab.positions[:, 2]
    slab.set_constraint(FixAtoms(mask=z < z.min() + 0.1))
    return slab


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A working directory holding the two endpoints, with EMT in place of the MLIP."""
    monkeypatch.chdir(tmp_path)
    write("initial.vasp", _hollow_slab(0.0), format="vasp")
    write("final.vasp", _hollow_slab(0.5), format="vasp")
    monkeypatch.setattr("mliprun.cli.commands.neb.resolve_mlip",
                        lambda mlip, *a, **k: "emt")
    monkeypatch.setattr(CustomNEB, "setup_calculator",
                        lambda self, *a, **k: EMT())
    return tmp_path


def _neb(*extra):
    return runner.invoke(app, ["--initial", "initial.vasp", "--final", "final.vasp",
                               "--num-images", "3", *extra])


def _last_stage_results(path):
    return json.loads(Path(path).read_text())["stages"][-1]["results"]


def test_a_converged_neb_is_verified(workdir):
    result = _neb("--verify-ts", "overlap")
    assert result.exit_code == 0, result.stdout
    results = _last_stage_results("ts_verify/mliprun_run.json")
    assert results["n_imaginary"] == 1 and results["overlap"]["reference"] == "neb_tangent"
    assert results["overlap"]["value"] > 0.7 and results["verdict"] == "verified"
    assert "TS verification: verified" in result.stdout
    # The NEB's own record is not touched by the verification.
    stages = json.loads(Path("mliprun_run.json").read_text())["stages"]
    assert "ts-verify" not in json.dumps(stages)


def test_the_option_is_written_and_read_back_on_restart(workdir):
    result = _neb("--verify-ts", "overlap")
    assert result.exit_code == 0, result.stdout
    assert "Verify TS:" in Path("neb_parameters.txt").read_text()
    assert CustomNEB._parse_parameters_file(Path("neb_parameters.txt"))["verify_ts"] == "overlap"
    (workdir / "ts_verify" / "ts_verify.txt").unlink()

    restarted = runner.invoke(app, ["--restart"])
    assert restarted.exit_code == 0, restarted.stdout
    assert (workdir / "ts_verify" / "ts_verify.txt").exists()
    assert "Verify TS:" in Path("neb_parameters.txt").read_text()


def test_a_value_given_on_restart_overrides_the_stored_one(workdir):
    assert _neb("--verify-ts", "overlap").exit_code == 0
    restarted = runner.invoke(app, ["--restart", "--verify-ts", "both"])
    assert restarted.exit_code == 0, restarted.stdout
    assert re.search(r"Verify TS:\s+both\n", Path("neb_parameters.txt").read_text())
    assert "relax" in _last_stage_results("ts_verify/mliprun_run.json")


def test_without_the_option_nothing_is_verified(workdir):
    result = _neb()
    assert result.exit_code == 0, result.stdout
    assert not (workdir / "ts_verify").exists()
    assert "Verify TS:" not in Path("neb_parameters.txt").read_text()


def test_an_unconverged_neb_skips_verification(workdir):
    result = _neb("--verify-ts", "overlap", "--neb-max-steps", "1")
    assert result.exit_code == 0, result.stdout
    assert "skipped" in result.stdout
    assert not (workdir / "ts_verify" / "ts_verify.txt").exists()


def test_verify_ts_without_climb_is_refused_at_start(workdir):
    result = _neb("--verify-ts", "overlap", "--no-climb")
    assert result.exit_code == 1
    assert "climbing" in result.stdout
    assert not (workdir / "neb.log").exists()


def test_an_invalid_value_is_refused(workdir):
    result = _neb("--verify-ts", "sometimes")
    assert result.exit_code == 1
    assert "overlap" in result.stdout
    assert not (workdir / "neb.log").exists()


def test_an_aqcat_tag_is_refused(workdir):
    result = _neb("--verify-ts", "overlap", "--mlip", "aqcat25-ev2-1",
                  "--aqcat-spin", "on")
    assert result.exit_code == 1
    assert not (workdir / "neb.log").exists()


def test_a_failure_inside_verification_does_not_fail_the_neb(workdir, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("mliprun.cli.commands.neb.verify_ts", boom)
    result = _neb("--verify-ts", "overlap")
    assert result.exit_code == 0, result.stdout
    assert "TS verification failed: boom" in result.stdout
    assert json.loads(Path("mliprun_run.json").read_text())["status"] == "converged"


def test_a_band_with_its_maximum_at_an_endpoint_skips_verification(tmp_path, capsys):
    from types import SimpleNamespace

    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    from mliprun.cli.commands.neb import _verify_ts_after_neb

    images = []
    for x, energy in ((0.0, 1.0), (0.5, 0.5), (1.0, 0.0)):
        atoms = Atoms("H", positions=[[x, 0.0, 0.0]])
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
        images.append(atoms)
    neb_obj = SimpleNamespace(converged=True, images=images)

    _verify_ts_after_neb(neb_obj, "overlap", tmp_path)

    assert "no interior maximum" in capsys.readouterr().out
    assert not (tmp_path / "ts_verify").exists()
