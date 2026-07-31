"""The record as produced through real core and CLI code paths."""
import csv
import json

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from typer.testing import CliRunner

from mliprun.core.optimize import run_optimization
from mliprun.core.run_record import RECORD_FILENAME, RunContext
from mliprun.cli.commands import optimize as opt_cmd
from mliprun.cli.commands.optimize import app as optimize_app

runner = CliRunner()


def _cu(a=3.7, cubic=False):
    # cubic=True gives a diagonal (triangular) cell -- required by ase.md.npt.NPT
    # (the default "npt" barostat), which raises NotImplementedError on the
    # primitive fcc cell's off-diagonal Cell matrix.
    atoms = bulk("Cu", "fcc", a=a, cubic=cubic)
    atoms.calc = EMT()
    return atoms


def _record(d):
    return json.loads((d / RECORD_FILENAME).read_text())


#: One case per MLIP family, for the head/task forwarding tests:
#: (model tag, provenance/keyword name, a **non-default** value, the sibling
#: key that must stay null because it belongs to the other family).
#: Every value here is deliberately not the CLI/engine default -- see
#: `test_run_forwards_the_head_to_the_record`.
#:
#: The ids must not be the bare words `uma`/`mace`: a parametrize id becomes
#: a pytest keyword, and conftest's `pytest_collection_modifyitems` skips
#: anything keyworded `uma`/`mace` when that package is absent -- which would
#: silently disable every case here. Model tags collide with nothing.
_HEAD_CASES = [
    pytest.param("uma-s-1p2", "uma_task", "oc25", "mace_head", id="uma-s-1p2"),
    pytest.param("mace-mh-1", "mace_head", "omol", "uma_task", id="mace-mh-1"),
]


def _flag(head: str) -> str:
    """`uma_task` -> `--uma-task`."""
    return "--" + head.replace("_", "-")


class TestOptimizeRecord:
    def test_library_caller_gets_a_complete_record(self, tmp_path):
        """A caller that bypasses the CLI still gets a record -- the defect
        that left basin_00 with no parameter file at all."""
        run_optimization(atoms=_cu(), optimizer="bfgs", fmax=0.5, max_steps=5,
                         output_dir=tmp_path, model_name="emt", verbose=False)
        data = _record(tmp_path)
        assert data["command"] == "optimize"
        assert data["status"] in {"converged", "not_converged"}
        assert data["parameters"]["fmax"]["value"] == 0.5
        assert data["parameters"]["fmax"]["source"] == "unspecified"
        assert data["stages"][0]["kind"] == "optimize"
        assert isinstance(data["stages"][0]["steps"], int)

    def test_records_outcome_numbers(self, tmp_path):
        run_optimization(atoms=_cu(), optimizer="bfgs", fmax=0.5, max_steps=5,
                         output_dir=tmp_path, model_name="emt", verbose=False)
        results = _record(tmp_path)["stages"][0]["results"]
        assert isinstance(results["converged"], bool)
        assert isinstance(results["final_energy_eV"], float)
        assert results["final_fmax_eV_per_A"] >= 0.0

    def test_sources_honoured_when_context_supplied(self, tmp_path):
        ctx = RunContext(command="optimize",
                         param_sources={"fmax": "user", "max_steps": "default"})
        run_optimization(atoms=_cu(), optimizer="bfgs", fmax=0.5, max_steps=5,
                         output_dir=tmp_path, model_name="emt", verbose=False,
                         run_context=ctx)
        params = _record(tmp_path)["parameters"]
        assert params["fmax"]["source"] == "user"
        assert params["max_steps"]["source"] == "default"

    def test_inputs_describe_the_structure(self, tmp_path):
        run_optimization(atoms=_cu(), optimizer="bfgs", fmax=0.5, max_steps=5,
                         output_dir=tmp_path, model_name="emt", verbose=False)
        inputs = _record(tmp_path)["inputs"]
        assert inputs["n_atoms"] == 1
        assert inputs["formula"] == "Cu"

    def test_failed_run_still_leaves_a_record(self, tmp_path):
        """An exception mid-run must not swallow the evidence."""
        class Exploding(EMT):
            def calculate(self, *args, **kwargs):
                raise RuntimeError("calculator exploded")

        atoms = bulk("Cu", "fcc", a=3.7)
        atoms.calc = Exploding()
        try:
            run_optimization(atoms=atoms, optimizer="bfgs", fmax=0.5,
                             max_steps=5, output_dir=tmp_path,
                             model_name="emt", verbose=False)
        except Exception:
            pass
        data = _record(tmp_path)
        assert data["status"] == "failed"
        assert data["stages"][0]["status"] == "failed"


@pytest.fixture
def emt_patched(monkeypatch):
    """Swap the real model load for EMT so the CLI runs offline."""
    monkeypatch.setattr(opt_cmd, "build_calculator", lambda *a, **k: EMT())
    monkeypatch.setattr(opt_cmd, "validate_mlip", lambda *a, **k: None)

    def fake_setup(atoms, *a, **k):
        atoms.calc = EMT()
        return atoms

    monkeypatch.setattr(opt_cmd, "setup_calculator", fake_setup)


class TestOptimizeCliRecord:
    def test_run_marks_one_off_and_tags_sources(self, tmp_path, emt_patched):
        struct = tmp_path / "init.vasp"
        write(str(struct), bulk("Cu", "fcc", a=3.7), format="vasp")

        result = runner.invoke(optimize_app, [
            "run", "--structure", str(struct), "--mlip", "mace",
            "--fmax", "0.5", "--max-steps", "5",
        ])
        assert result.exit_code == 0, result.output

        data = _record(tmp_path)
        assert data["run"]["mode"] == "one-off"
        assert data["run"]["batch"] is None
        assert data["parameters"]["fmax"]["source"] == "user"
        assert data["parameters"]["optimizer"]["source"] == "default"
        assert data["inputs"]["structure"] == "init.vasp"

    def test_batch_shares_one_id_across_subdirs(self, tmp_path, emt_patched):
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            write(str(d / "init.vasp"), bulk("Cu", "fcc", a=3.7), format="vasp")

        result = runner.invoke(optimize_app, [
            "batch", "--parent", str(tmp_path), "--mlip", "mace",
            "--fmax", "0.5", "--max-steps", "5",
        ])
        assert result.exit_code == 0, result.output

        rec_a = _record(tmp_path / "a")
        rec_b = _record(tmp_path / "b")
        assert rec_a["run"]["mode"] == "batch"
        assert rec_a["run"]["batch"]["batch_id"] == rec_b["run"]["batch"]["batch_id"]
        assert rec_a["run"]["batch"]["root"] == str(tmp_path.resolve())
        assert rec_a["inputs"]["structure"] == "init.vasp"
        assert "batch" in rec_a["run"]["batch"]["driver"]

    def test_batch_writes_no_record_into_parent(self, tmp_path, emt_patched):
        d = tmp_path / "a"
        d.mkdir()
        write(str(d / "init.vasp"), bulk("Cu", "fcc", a=3.7), format="vasp")
        runner.invoke(optimize_app, [
            "batch", "--parent", str(tmp_path), "--mlip", "mace",
            "--fmax", "0.5", "--max-steps", "5",
        ])
        assert not (tmp_path / RECORD_FILENAME).exists()

        summary_path = tmp_path / "batch_summary.csv"
        assert summary_path.exists()
        with open(summary_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["subdir", "status", "converged", "steps",
                            "energy_eV", "walltime_s", "detail"]
        assert any(row[0] == "a" for row in rows[1:])

    @pytest.mark.parametrize("mlip,head,value,other", _HEAD_CASES)
    def test_run_forwards_the_head_to_the_record(self, tmp_path, emt_patched,
                                                  mlip, head, value, other):
        """`optimize run` must hand its parsed head/task to the engine.

        A **non-default** value is required: with the default, a value the
        CLI parsed and one the engine defaulted to are indistinguishable, so
        the assertion would still pass with the kwarg deleted.
        """
        struct = tmp_path / "init.vasp"
        write(str(struct), bulk("Cu", "fcc", a=3.7), format="vasp")

        result = runner.invoke(optimize_app, [
            "run", "--structure", str(struct), "--mlip", mlip,
            _flag(head), value, "--fmax", "0.5", "--max-steps", "5",
        ])
        assert result.exit_code == 0, result.output

        prov = _record(tmp_path)["provenance"]
        assert prov[head] == value
        assert prov[other] is None  # gated off: wrong model family

    @pytest.mark.parametrize("mlip,head,value,other", _HEAD_CASES)
    def test_batch_forwards_the_head_to_every_record(self, tmp_path, emt_patched,
                                                      mlip, head, value, other):
        """Same forwarding, separate call site in `optimize batch`."""
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            write(str(d / "init.vasp"), bulk("Cu", "fcc", a=3.7), format="vasp")

        result = runner.invoke(optimize_app, [
            "batch", "--parent", str(tmp_path), "--mlip", mlip,
            _flag(head), value, "--fmax", "0.5", "--max-steps", "5",
        ])
        assert result.exit_code == 0, result.output

        for name in ("a", "b"):
            prov = _record(tmp_path / name)["provenance"]
            assert prov[head] == value, name
            assert prov[other] is None, name

    def test_txt_file_still_written(self, tmp_path, emt_patched):
        """The .txt files are retained; the JSON supplements, not replaces."""
        struct = tmp_path / "init.vasp"
        write(str(struct), bulk("Cu", "fcc", a=3.7), format="vasp")
        runner.invoke(optimize_app, [
            "run", "--structure", str(struct), "--mlip", "mace",
            "--fmax", "0.5", "--max-steps", "5",
        ])
        assert "Geometry Optimization Parameters" in (tmp_path / "opt_params.txt").read_text()


from mliprun.core.md import run_md


class TestMdRecord:
    def _run(self, tmp_path, **kwargs):
        opts = dict(atoms=_cu(), ensemble="nve", steps=20, timestep=1.0,
                    log_interval=2, traj_interval=10, output_dir=tmp_path,
                    model_name="emt", plot=False)
        opts.update(kwargs)
        return run_md(**opts)

    def test_records_raw_statistics(self, tmp_path):
        self._run(tmp_path)
        results = _record(tmp_path)["stages"][0]["results"]
        assert results["mean_temperature_K"] >= 0.0
        assert results["std_temperature_K"] >= 0.0
        assert isinstance(results["mean_total_energy_eV"], float)
        assert isinstance(results["mean_potential_energy_eV"], float)
        assert isinstance(results["total_energy_drift_eV_per_atom_per_ps"], float)
        assert len(results["decile_mean_total_energy_eV"]) == 10

    def test_completes_even_without_plotting(self, tmp_path):
        """run_md returns early when plot=False; the record must still close."""
        self._run(tmp_path, plot=False)
        assert _record(tmp_path)["status"] == "converged"
        assert _record(tmp_path)["stages"][0]["walltime_s"] is not None

    def test_resume_appends_a_stage(self, tmp_path):
        self._run(tmp_path)
        self._run(tmp_path, resume=True, steps=10)
        stages = _record(tmp_path)["stages"]
        assert len(stages) == 2
        assert stages[0]["kind"] == "md"
        assert stages[1]["kind"] == "md-resume"
        assert stages[0]["results"]["mean_temperature_K"] >= 0.0

    def test_resume_statistics_cover_only_new_rows(self, tmp_path):
        """Stage 1 must describe the resumed segment, not the whole history."""
        self._run(tmp_path, steps=20)
        self._run(tmp_path, resume=True, steps=10)
        stages = _record(tmp_path)["stages"]
        assert stages[1]["results"]["n_samples"] < stages[0]["results"]["n_samples"]

    def test_npt_records_pressure_and_volume(self, tmp_path):
        # ase.md.npt.NPT (the default "npt" barostat) requires a triangular
        # cell, so this test needs the cubic Cu cell rather than _cu()'s
        # default primitive one -- see _cu()'s docstring comment.
        self._run(tmp_path, atoms=_cu(cubic=True), ensemble="npt",
                  temperature=300, pressure=0.0, steps=10)
        results = _record(tmp_path)["stages"][0]["results"]
        assert "mean_pressure_GPa" in results
        assert "mean_volume_A3" in results


from mliprun.core.run_record import RunRecord


class TestNebBarrierMetrics:
    """The barrier arithmetic, isolated from the cost of a real NEB."""

    def test_forward_and_reverse_barriers(self):
        from mliprun.core.neb import summarize_neb_path

        # Energies along a path with a clear interior maximum.
        s = summarize_neb_path([0.0, 0.4, 0.9, 0.3, -0.35])
        assert s["forward_barrier_eV"] == pytest.approx(0.9)
        assert s["reverse_barrier_eV"] == pytest.approx(1.25)
        assert s["reaction_energy_eV"] == pytest.approx(-0.35)
        assert s["ts_image_index"] == 2
        assert s["n_images"] == 5
        assert s["ts_at_endpoint"] is False

    def test_maximum_at_final_image_is_flagged(self):
        from mliprun.core.neb import summarize_neb_path

        s = summarize_neb_path([0.0, 0.2, 0.5, 0.8])
        assert s["ts_at_endpoint"] is True
        assert s["ts_image_index"] == 3

    def test_maximum_at_first_image_is_flagged(self):
        from mliprun.core.neb import summarize_neb_path

        s = summarize_neb_path([0.5, 0.2, 0.0])
        assert s["ts_at_endpoint"] is True

    def test_single_image_degenerates_safely(self):
        from mliprun.core.neb import summarize_neb_path

        s = summarize_neb_path([1.0])
        assert s["n_images"] == 1
        assert s["forward_barrier_eV"] == 0.0


class TestNebStageAppend:
    def test_restart_appends_second_stage(self, tmp_path):
        """A plain NEB then a CI-NEB in the same directory is two stages."""
        rec = RunRecord.begin(
            tmp_path, command="neb", stage_kind="neb",
            parameters={"climb": False}, inputs={"n_images": 5},
            provenance={"mliprun_version": "test"},
            stage_parameters={"climb": False},
        )
        rec.complete(status="converged", steps=100,
                     results={"forward_barrier_eV": 0.85})

        rec2 = RunRecord.begin(
            tmp_path, command="neb", stage_kind="neb-restart",
            parameters={"climb": True}, inputs={"n_images": 5},
            provenance={"mliprun_version": "test"},
            stage_parameters={"climb": True}, append=True,
        )
        rec2.complete(status="converged", steps=40,
                      results={"forward_barrier_eV": 0.91})

        stages = _record(tmp_path)["stages"]
        assert [s["kind"] for s in stages] == ["neb", "neb-restart"]
        assert stages[0]["results"]["forward_barrier_eV"] == 0.85
        assert stages[1]["parameters"]["climb"]["value"] is True


def _neb_pair():
    """Cu initial/final pair for NEB wiring tests.

    Mirrors tests/test_core_neb.py::_make_neb_pair -- kept local rather than
    imported so this file's tests stand alone.
    """
    initial = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    final = initial.copy()
    pos = final.get_positions()
    pos[0] += np.array([0.3, 0.3, 0.0])
    final.set_positions(pos)
    return initial, final


class TestNebRunWiring:
    """Exercises the real run_neb code path with EMT calculators.

    TestNebBarrierMetrics and TestNebStageAppend above test the arithmetic
    and the RunRecord API in isolation -- neither one calls the real
    run_neb, so a bug in the wiring itself (record never opened, wrong
    stage_kind, append not honored, results never populated) would pass
    every test above while `neb run` silently wrote nothing useful. These
    tests close that gap.
    """

    def _emt_neb(self, tmp_path, monkeypatch, **kwargs):
        from mliprun.core.neb import CustomNEB

        initial, final = _neb_pair()
        # `mlip` is a default rather than fixed, so the head/task tests can
        # ask for a real model family -- collect_provenance gates uma_task /
        # mace_head on the tag prefix and would drop them under "test".
        kwargs.setdefault("mlip", "test")
        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            output_dir=tmp_path, **kwargs,
        )
        # run_neb builds a calculator per image via self.setup_calculator();
        # swap in EMT so the run works without a real MLIP installed.
        monkeypatch.setattr(neb, "setup_calculator", lambda: EMT())
        return neb

    def test_fresh_run_writes_a_neb_stage_with_barrier_results(self, tmp_path, monkeypatch):
        neb = self._emt_neb(tmp_path, monkeypatch)
        neb.run_neb(max_steps=3)

        data = _record(tmp_path)
        assert data["command"] == "neb"
        stage = data["stages"][0]
        assert stage["kind"] == "neb"
        assert stage["status"] in {"converged", "not_converged"}
        assert isinstance(stage["steps"], int)

        results = stage["results"]
        assert "forward_barrier_eV" in results
        assert "reverse_barrier_eV" in results
        assert "reaction_energy_eV" in results
        assert results["n_images"] == 5  # 3 intermediate + 2 endpoints
        assert isinstance(results["ts_at_endpoint"], bool)
        assert results["final_fmax_eV_per_A"] is not None

    def test_append_true_appends_a_neb_restart_stage(self, tmp_path, monkeypatch):
        neb = self._emt_neb(tmp_path, monkeypatch)
        neb.run_neb(max_steps=3)

        # A second CustomNEB in the same directory, as load_from_restart
        # would produce, with climb turned on -- the plain-then-CI-NEB
        # workflow the design doc calls out.
        neb2 = self._emt_neb(tmp_path, monkeypatch)
        neb2.run_neb(max_steps=3, climb=True, append=True)

        stages = _record(tmp_path)["stages"]
        assert [s["kind"] for s in stages] == ["neb", "neb-restart"]
        assert stages[1]["parameters"]["climb"]["value"] is True
        assert "forward_barrier_eV" in stages[0]["results"]
        assert "forward_barrier_eV" in stages[1]["results"]

    def test_restart_records_the_new_fmax_per_stage(self, tmp_path, monkeypatch):
        """A restart at a different fmax must record that stage's own value.

        Confirmed defect: fmax lived only in top-level `parameters`, which
        `RunRecord.begin` never rewrites on append -- so stage 1 had no
        `fmax` key at all, and a reader would (wrongly) read the first run's
        fmax as governing the restart too.
        """
        neb = self._emt_neb(tmp_path, monkeypatch, fmax=0.1)
        neb.run_neb(max_steps=3)

        neb2 = self._emt_neb(tmp_path, monkeypatch, fmax=0.03)
        neb2.run_neb(max_steps=3, climb=True, append=True)

        stages = _record(tmp_path)["stages"]
        assert stages[0]["parameters"]["fmax"]["value"] == pytest.approx(0.1)
        assert stages[1]["parameters"]["fmax"]["value"] == pytest.approx(0.03)

    def test_run_context_tags_parameter_sources(self, tmp_path, monkeypatch):
        neb = self._emt_neb(tmp_path, monkeypatch)
        ctx = RunContext(command="neb", param_sources={"fmax": "user"})
        neb.run_neb(max_steps=3, run_context=ctx)

        # fmax lives in the stage's parameters (see
        # test_restart_records_the_new_fmax_per_stage above), not the
        # top-level ones -- it is stage-scoped, not directory-fixed.
        stage_params = _record(tmp_path)["stages"][0]["parameters"]
        assert stage_params["fmax"]["value"] == pytest.approx(neb.fmax)
        assert stage_params["fmax"]["source"] == "user"

    @pytest.mark.parametrize("mlip,head,value,other", _HEAD_CASES)
    def test_neb_stage_records_the_head_the_images_ran_with(
            self, tmp_path, monkeypatch, mlip, head, value, other):
        """The record must name the head that produced the barrier.

        A NEB barrier is exactly the kind of number CANON C3 governs, and
        `run_neb` reads the head off `self` -- so a kwarg dropped from the
        `collect_provenance` call is silent. Non-default values are used
        deliberately: with the default, a forwarded value and a defaulted
        one are indistinguishable and the assertion proves nothing.
        """
        neb = self._emt_neb(tmp_path, monkeypatch, mlip=mlip, **{head: value})
        neb.run_neb(max_steps=2)

        prov = _record(tmp_path)["provenance"]
        assert prov[head] == value
        assert prov["mlip_model"] == mlip
        assert prov[other] is None  # gated off: wrong model family

    def test_neb_restart_stage_records_a_switched_task(self, tmp_path, monkeypatch):
        """Restarting under a different head is the C3 hazard the record
        exists to expose; the append path must carry the new value too."""
        neb = self._emt_neb(tmp_path, monkeypatch, mlip="uma-s-1p2",
                            uma_task="oc25")
        neb.run_neb(max_steps=2)

        neb2 = self._emt_neb(tmp_path, monkeypatch, mlip="uma-s-1p2",
                             uma_task="odac")
        neb2.run_neb(max_steps=2, climb=True, append=True)

        data = _record(tmp_path)
        assert data["provenance"]["uma_task"] == "oc25"
        assert data["stages"][1]["stage_provenance"] == {"uma_task": "odac"}

    def test_failed_neb_run_still_leaves_a_record(self, tmp_path, monkeypatch):
        """An exception mid-optimization must not swallow the evidence."""
        from mliprun.core.neb import CustomNEB

        initial, final = _neb_pair()
        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            mlip="test", output_dir=tmp_path,
        )

        class Exploding(EMT):
            def calculate(self, *args, **kwargs):
                raise RuntimeError("calculator exploded")

        monkeypatch.setattr(neb, "setup_calculator", lambda: Exploding())
        with pytest.raises(RuntimeError):
            neb.run_neb(max_steps=3)

        data = _record(tmp_path)
        assert data["status"] == "failed"
        assert data["stages"][0]["status"] == "failed"


class TestAutonebRecord:
    def test_autoneb_records_command_and_stage_kind(self, tmp_path):
        """AutoNEB shares the barrier summary but is its own command."""
        rec = RunRecord.begin(
            tmp_path, command="autoneb", stage_kind="autoneb",
            parameters={"n_max": 9, "climb": True},
            inputs={"n_images": 5},
            provenance={"mliprun_version": "test"},
        )
        from mliprun.core.neb import summarize_neb_path
        rec.complete(status="converged", steps=250,
                     results=summarize_neb_path([0.0, 0.6, 1.1, 0.2]))
        data = _record(tmp_path)
        assert data["command"] == "autoneb"
        assert data["stages"][0]["kind"] == "autoneb"
        assert data["stages"][0]["results"]["forward_barrier_eV"] == pytest.approx(1.1)
        assert data["stages"][0]["results"]["ts_at_endpoint"] is False


class TestAutonebRunWiring:
    """Exercises the real run_autoneb code path with EMT calculators.

    TestAutonebRecord above tests the command/stage_kind wiring against the
    RunRecord API directly, without going through run_autoneb -- a bug in
    the wiring itself (record never opened, wrong stage_kind, results never
    populated from the actual converged path) would pass that test while
    `autoneb run` silently wrote nothing useful. These tests close that gap.

    n_max=3, n_simul=1 is the smallest AutoNEB configuration ASE accepts:
    two fixed endpoints plus one free interior image, so a single tiny NEB
    relaxation is all that runs. climb is left off here -- ASE's own
    `assert climb_safe` inside AutoNEB.run() requires the highest-energy
    image to land on the (only) interior image, which is not guaranteed for
    these unrelaxed endpoints, and that assertion is orthogonal to what
    this test is checking.
    """

    def _emt_neb(self, tmp_path, monkeypatch, **kwargs):
        from mliprun.core.neb import CustomNEB

        initial, final = _neb_pair()
        # `mlip` is a default rather than fixed, so the head/task tests can
        # ask for a real model family -- collect_provenance gates uma_task /
        # mace_head on the tag prefix and would drop them under "test".
        kwargs.setdefault("mlip", "test")
        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            output_dir=tmp_path, **kwargs,
        )
        monkeypatch.setattr(neb, "setup_calculator", lambda: EMT())
        return neb

    def test_autoneb_run_writes_a_record_with_barrier_results(self, tmp_path, monkeypatch):
        neb = self._emt_neb(tmp_path, monkeypatch)
        neb.run_autoneb(n_simul=1, n_max=3, maxsteps=5, climb=False)

        data = _record(tmp_path)
        assert data["command"] == "autoneb"
        stage = data["stages"][0]
        assert stage["kind"] == "autoneb"
        assert stage["status"] == "converged"

        results = stage["results"]
        assert "forward_barrier_eV" in results
        assert "reverse_barrier_eV" in results
        assert "reaction_energy_eV" in results
        assert results["n_images"] == 3
        assert isinstance(results["ts_at_endpoint"], bool)

    def test_run_context_tags_parameter_sources(self, tmp_path, monkeypatch):
        neb = self._emt_neb(tmp_path, monkeypatch)
        ctx = RunContext(command="autoneb", param_sources={"n_max": "user"})
        neb.run_autoneb(n_simul=1, n_max=3, maxsteps=5, climb=False, run_context=ctx)

        params = _record(tmp_path)["parameters"]
        assert params["n_max"]["value"] == 3
        assert params["n_max"]["source"] == "user"

    @pytest.mark.parametrize("mlip,head,value,other", _HEAD_CASES)
    def test_autoneb_stage_records_the_head_the_images_ran_with(
            self, tmp_path, monkeypatch, mlip, head, value, other):
        """Same C3 exposure as the NEB stage, separate call site: AutoNEB
        opens its own record. Non-default values keep the assertion
        load-bearing (see the NEB counterpart)."""
        neb = self._emt_neb(tmp_path, monkeypatch, mlip=mlip, **{head: value})
        neb.run_autoneb(n_simul=1, n_max=3, maxsteps=5, climb=False)

        prov = _record(tmp_path)["provenance"]
        assert prov[head] == value
        assert prov["mlip_model"] == mlip
        assert prov[other] is None  # gated off: wrong model family

    def test_failed_autoneb_run_still_leaves_a_record(self, tmp_path, monkeypatch):
        """An exception inside AutoNEB's own run() must not swallow the
        evidence -- but a failure during endpoint setup (before AutoNEB.run()
        is even called) is deliberately NOT wrapped, per the design ("only
        the AutoNEB run in try/except"), so the fake calculator here must
        only start failing once AutoNEB's interior images request one."""
        from mliprun.core.neb import CustomNEB

        initial, final = _neb_pair()
        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            mlip="test", output_dir=tmp_path,
        )

        class Exploding(EMT):
            def calculate(self, *args, **kwargs):
                raise RuntimeError("calculator exploded")

        calls = {"n": 0}

        def flaky_calculator():
            calls["n"] += 1
            # The first two calls set up the initial/final endpoint
            # energies before AutoNEB.run() starts; only calculators
            # handed out afterwards (to AutoNEB's interior images) fail.
            return EMT() if calls["n"] <= 2 else Exploding()

        monkeypatch.setattr(neb, "setup_calculator", flaky_calculator)
        with pytest.raises(RuntimeError):
            neb.run_autoneb(n_simul=1, n_max=3, maxsteps=5, climb=False)

        data = _record(tmp_path)
        assert data["status"] == "failed"
        assert data["stages"][0]["status"] == "failed"
