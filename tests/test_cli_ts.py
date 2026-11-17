"""CLI surface of `mlip ts verify` on a one-atom double well."""
import json
import re

import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.io.trajectory import Trajectory
from typer.testing import CliRunner

from mliprun.cli import main
from mliprun.cli.commands.ts import app
from mliprun.core.params_io import write_parameters_file
from ts_helpers import CENTRE, DoubleWell, single_atom

runner = CliRunner()
X = np.array([1.0, 0.0, 0.0])


def _plain(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture(autouse=True)
def _fake_model(monkeypatch):
    monkeypatch.setattr("mliprun.cli.commands.ts.build_calculator",
                        lambda *a, **k: DoubleWell())
    monkeypatch.setattr("mliprun.cli.commands.ts.detect_mlip", lambda: "fake")
    monkeypatch.setattr("mliprun.cli.commands.ts.validate_mlip",
                        lambda *a, **k: None)


@pytest.fixture
def three_files(tmp_path):
    folder = tmp_path / "files"
    folder.mkdir()
    paths = {}
    for name, shift in (("ts", 0.0), ("initial", -1.0), ("final", 1.0)):
        paths[name] = folder / f"{name}.vasp"
        write(paths[name], single_atom(CENTRE + shift * X), format="vasp")
    return paths


def _make_neb_dir(root, xs=(-1.0, 0.0, 1.0), climb=True, status="converged",
                  mlip="fake"):
    neb_dir = root / "neb"
    neb_dir.mkdir()
    calc = DoubleWell()
    with Trajectory(str(neb_dir / "A2B.traj"), "w") as traj:
        for x in xs:
            atoms = single_atom(CENTRE + x * X)
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            atoms.calc = SinglePointCalculator(atoms, energy=energy,
                                               forces=forces)
            traj.write(atoms)
    write_parameters_file(neb_dir / "neb_parameters.txt", "NEB Run Parameters", {
        "MLIP model:": mlip,
        "Intermediate images:": len(xs) - 2,
        "Final fmax:": 0.05,
        "Climb:": climb,
    })
    record = {"stages": [{"kind": "neb", "status": status,
                          "parameters": {"climb": {"value": climb,
                                                   "source": "default"}}}]}
    (neb_dir / "mliprun_run.json").write_text(json.dumps(record))
    return neb_dir


def _three_file_args(paths):
    return ["verify", "--ts", str(paths["ts"]),
            "--initial", str(paths["initial"]), "--final", str(paths["final"])]


def test_three_files_verify_and_exit_zero(three_files):
    result = runner.invoke(app, _three_file_args(three_files))
    assert result.exit_code == 0, result.stdout
    assert "verified" in result.stdout
    assert (three_files["ts"].parent / "ts_verify" / "ts_verify.txt").exists()


def test_a_neb_folder_uses_the_tangent_reference(tmp_path):
    neb_dir = _make_neb_dir(tmp_path)
    result = runner.invoke(app, ["verify", "--neb-dir", str(neb_dir)])
    assert result.exit_code == 0, result.stdout
    record = json.loads(
        (neb_dir / "ts_verify" / "mliprun_run.json").read_text())
    stage = record["stages"][-1]
    assert stage["results"]["overlap"]["reference"] == "neb_tangent"
    assert stage["results"]["verdict"] == "verified"
    assert stage["results"]["inputs"]["ts_image_index"] == 1
    assert stage["results"]["inputs"]["n_images"] == 3


def test_the_neb_record_is_not_modified(tmp_path):
    neb_dir = _make_neb_dir(tmp_path)
    before = (neb_dir / "mliprun_run.json").read_bytes()
    result = runner.invoke(app, ["verify", "--neb-dir", str(neb_dir)])
    assert result.exit_code == 0, result.stdout
    assert (neb_dir / "mliprun_run.json").read_bytes() == before


def test_check_both_runs_the_relax_check(tmp_path):
    neb_dir = _make_neb_dir(tmp_path)
    result = runner.invoke(app, ["verify", "--neb-dir", str(neb_dir),
                                 "--check", "both"])
    assert result.exit_code == 0, result.stdout
    record = json.loads(
        (neb_dir / "ts_verify" / "mliprun_run.json").read_text())
    assert record["stages"][-1]["results"]["relax"]["result"] == "connects"


def test_a_failed_verification_still_exits_zero(three_files):
    result = runner.invoke(
        app, _three_file_args(three_files) + ["--overlap-threshold", "1.1"])
    assert result.exit_code == 0, result.stdout
    assert "not_verified" in result.stdout


def test_neb_dir_and_ts_together_are_refused(tmp_path, three_files):
    neb_dir = _make_neb_dir(tmp_path)
    result = runner.invoke(app, ["verify", "--neb-dir", str(neb_dir),
                                 "--ts", str(three_files["ts"])])
    assert result.exit_code == 1


def test_a_partial_three_file_set_is_refused(three_files):
    result = runner.invoke(app, ["verify", "--ts", str(three_files["ts"]),
                                 "--initial", str(three_files["initial"])])
    assert result.exit_code == 1
    assert "--final" in _plain(result.stdout)


def test_no_input_at_all_is_refused():
    result = runner.invoke(app, ["verify"])
    assert result.exit_code == 1


def test_an_invalid_check_value_is_refused(three_files):
    result = runner.invoke(
        app, _three_file_args(three_files) + ["--check", "sideways"])
    assert result.exit_code == 1
    assert "--check" in _plain(result.stdout)


def test_an_invalid_nfree_is_refused(three_files):
    result = runner.invoke(
        app, _three_file_args(three_files) + ["--nfree", "3"])
    assert result.exit_code == 1


def test_an_unparsable_indices_value_is_refused(three_files):
    result = runner.invoke(
        app, _three_file_args(three_files) + ["--indices", "a-b"])
    assert result.exit_code == 1


def test_an_aqcat_tag_is_refused(three_files):
    result = runner.invoke(
        app, _three_file_args(three_files) + ["--mlip", "aqcat25-ev2-1"])
    assert result.exit_code == 1


def test_a_band_whose_maximum_is_an_endpoint_exits_nonzero(tmp_path):
    neb_dir = _make_neb_dir(tmp_path, xs=(-1.0, -0.5, 0.0))
    result = runner.invoke(app, ["verify", "--neb-dir", str(neb_dir)])
    assert result.exit_code != 0
    assert "interior maximum" in _plain(result.stdout)


def test_a_missing_trajectory_is_reported(tmp_path):
    neb_dir = _make_neb_dir(tmp_path)
    (neb_dir / "A2B.traj").unlink()
    result = runner.invoke(app, ["verify", "--neb-dir", str(neb_dir)])
    assert result.exit_code == 1
    assert "A2B.traj" in _plain(result.stdout)


def test_an_unconverged_neb_warns_and_continues(tmp_path):
    neb_dir = _make_neb_dir(tmp_path, status="not_converged")
    result = runner.invoke(app, ["verify", "--neb-dir", str(neb_dir)])
    assert result.exit_code == 0, result.stdout
    assert "not converged" in result.stdout


def test_a_non_climbing_neb_warns_and_continues(tmp_path):
    neb_dir = _make_neb_dir(tmp_path, climb=False)
    result = runner.invoke(app, ["verify", "--neb-dir", str(neb_dir)])
    assert result.exit_code == 0, result.stdout
    assert "climbing" in result.stdout


def test_a_foreign_cache_is_reported_without_a_traceback(three_files):
    args = _three_file_args(three_files)
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.stdout
    second = runner.invoke(app, args + ["--delta", "0.02"])
    assert second.exit_code == 1
    assert "Traceback" not in second.stdout
    assert second.exception is None or isinstance(second.exception, SystemExit)


def test_mlip_ts_is_registered():
    result = runner.invoke(main.app, ["ts", "--help"])
    assert result.exit_code == 0
    assert "verify" in result.stdout


def test_a_missing_parameter_file_is_reported(tmp_path):
    neb_dir = _make_neb_dir(tmp_path)
    (neb_dir / "neb_parameters.txt").unlink()
    result = runner.invoke(app, ["verify", "--neb-dir", str(neb_dir)])
    assert result.exit_code == 1
    assert "neb_parameters.txt" in _plain(result.stdout)


def test_an_unreadable_neb_record_does_not_stop_the_verification(tmp_path):
    neb_dir = _make_neb_dir(tmp_path)
    (neb_dir / "mliprun_run.json").write_text("{ not json")
    result = runner.invoke(app, ["verify", "--neb-dir", str(neb_dir)])
    assert result.exit_code == 0, result.stdout
    assert "TS verification: verified" in _plain(result.stdout)


def test_the_console_summary_prints_the_relax_note(capsys, tmp_path):
    from mliprun.cli.commands.ts import echo_ts_verification

    results = {
        "verdict": "inconclusive", "reason": "r", "n_imaginary": 1,
        "imaginary_frequencies_cm-1": [64.5], "ts_fmax_free_eV_per_A": 0.01,
        "ts_fmax_limit_eV_per_A": 0.03,
        "relax": {"result": "inconclusive", "status": "inconclusive",
                  "note": "a relaxation returned to the TS; use a larger "
                          "displacement or another optimizer"},
    }
    echo_ts_verification(results, tmp_path)
    assert "Note: a relaxation returned to the TS" in capsys.readouterr().out
