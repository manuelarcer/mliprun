"""Unit tests for the canonical JSON run record."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from mliprun.core.run_record import (
    NEW_FIELDS_KEY,
    RECORD_FILENAME,
    SCHEMA_VERSION,
    BatchInfo,
    RunContext,
    RunRecord,
    _split_provenance,
    collect_provenance,
    new_batch_id,
)


def _read(tmp_path: Path) -> dict:
    return json.loads((tmp_path / RECORD_FILENAME).read_text())


def _begin(tmp_path, **kwargs):
    defaults = dict(
        command="optimize",
        stage_kind="optimize",
        parameters={"fmax": 0.02, "max_steps": 200},
        inputs={"structure": "init.vasp", "n_atoms": 4},
        provenance={"mliprun_version": "0.4.0"},
    )
    defaults.update(kwargs)
    return RunRecord.begin(tmp_path, **defaults)


class TestPhaseOne:
    def test_writes_running_status_before_completion(self, tmp_path):
        _begin(tmp_path)
        data = _read(tmp_path)
        assert data["status"] == "running"
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["command"] == "optimize"
        assert data["provenance"]["finished_at"] is None
        assert data["stages"][0]["status"] == "running"

    def test_unspecified_source_without_context(self, tmp_path):
        _begin(tmp_path)
        params = _read(tmp_path)["parameters"]
        assert params["fmax"] == {"value": 0.02, "source": "unspecified"}
        assert params["max_steps"] == {"value": 200, "source": "unspecified"}

    def test_sources_applied_from_context(self, tmp_path):
        ctx = RunContext(command="optimize",
                         param_sources={"fmax": "user", "max_steps": "default"})
        _begin(tmp_path, run_context=ctx)
        params = _read(tmp_path)["parameters"]
        assert params["fmax"]["source"] == "user"
        assert params["max_steps"]["source"] == "default"

    def test_extra_inputs_merged_from_context(self, tmp_path):
        ctx = RunContext(command="optimize",
                         extra_inputs={"structure": "init.vasp",
                                       "structure_abspath": "/tmp/init.vasp"})
        _begin(tmp_path, run_context=ctx)
        inputs = _read(tmp_path)["inputs"]
        assert inputs["structure"] == "init.vasp"
        assert inputs["structure_abspath"] == "/tmp/init.vasp"
        assert inputs["n_atoms"] == 4  # core-supplied fact survives the merge

    def test_one_off_has_null_batch(self, tmp_path):
        _begin(tmp_path)
        run = _read(tmp_path)["run"]
        assert run["mode"] == "one-off"
        assert run["batch"] is None

    def test_batch_fields_recorded(self, tmp_path):
        batch = BatchInfo(batch_id="20260721T000000-abc123",
                          driver="mliprun optimize batch",
                          argv=["mliprun", "optimize", "batch"],
                          root="/tmp/parent")
        ctx = RunContext(command="optimize", mode="batch", batch=batch)
        _begin(tmp_path, run_context=ctx)
        run = _read(tmp_path)["run"]
        assert run["mode"] == "batch"
        assert run["batch"]["batch_id"] == "20260721T000000-abc123"
        assert run["batch"]["root"] == "/tmp/parent"
        assert run["batch"]["config_file"] is None


class TestPhaseTwo:
    def test_complete_sets_status_and_results(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="converged", steps=47,
                     results={"converged": True, "final_energy_eV": -1.5})
        data = _read(tmp_path)
        assert data["status"] == "converged"
        assert data["provenance"]["finished_at"] is not None
        assert data["provenance"]["walltime_s"] >= 0
        stage = data["stages"][0]
        assert stage["status"] == "converged"
        assert stage["steps"] == 47
        assert stage["results"]["final_energy_eV"] == -1.5

    def test_parameters_survive_completion(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="converged", results={})
        assert _read(tmp_path)["parameters"]["fmax"]["value"] == 0.02


class TestStages:
    def test_append_adds_stage_preserving_first(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="converged", steps=10, results={"barrier_eV": 0.85})

        rec2 = _begin(tmp_path, stage_kind="neb-restart", append=True,
                      stage_parameters={"climb": True})
        rec2.complete(status="converged", steps=5, results={"barrier_eV": 0.91})

        stages = _read(tmp_path)["stages"]
        assert len(stages) == 2
        assert stages[0]["index"] == 0
        assert stages[0]["results"]["barrier_eV"] == 0.85
        assert stages[0]["status"] == "converged"
        assert stages[1]["index"] == 1
        assert stages[1]["kind"] == "neb-restart"
        assert stages[1]["parameters"]["climb"]["value"] is True

    def test_top_level_status_reflects_latest_stage(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="converged", results={})
        rec2 = _begin(tmp_path, stage_kind="neb-restart", append=True)
        rec2.complete(status="failed", results={})
        data = _read(tmp_path)
        assert data["status"] == "failed"
        assert data["stages"][0]["status"] == "converged"

    def test_append_to_missing_record_marks_unknown_history(self, tmp_path):
        rec = _begin(tmp_path, stage_kind="md-resume", append=True)
        rec.complete(status="converged", results={})
        data = _read(tmp_path)
        assert data["stages"][0]["prior_history_unknown"] is True

    def test_append_to_fresh_record_does_not_mark_unknown(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="converged", results={})
        rec2 = _begin(tmp_path, append=True)
        rec2.complete(status="converged", results={})
        stages = _read(tmp_path)["stages"]
        assert "prior_history_unknown" not in stages[1]


class TestRobustness:
    def test_corrupt_record_is_backed_up_not_fatal(self, tmp_path):
        (tmp_path / RECORD_FILENAME).write_text("{not valid json")
        rec = _begin(tmp_path, append=True)
        rec.complete(status="converged", results={})
        data = _read(tmp_path)
        assert data["status"] == "converged"
        backups = list(tmp_path.glob(f"{RECORD_FILENAME}.corrupt-*"))
        assert len(backups) == 1
        assert backups[0].read_text() == "{not valid json"

    def test_non_finite_floats_coerced_to_null(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="failed",
                     results={"final_energy_eV": float("nan"),
                              "final_fmax_eV_per_A": float("inf")})
        raw = (tmp_path / RECORD_FILENAME).read_text()
        assert "NaN" not in raw and "Infinity" not in raw
        results = _read(tmp_path)["stages"][0]["results"]
        assert results["final_energy_eV"] is None
        assert results["final_fmax_eV_per_A"] is None

    def test_numpy_and_path_values_serialized(self, tmp_path):
        rec = _begin(tmp_path, parameters={"fmax": np.float64(0.02),
                                           "steps": np.int64(7),
                                           "out": Path("/tmp/x")})
        rec.complete(status="converged", results={})
        params = _read(tmp_path)["parameters"]
        assert params["fmax"]["value"] == 0.02
        assert params["steps"]["value"] == 7
        assert params["out"]["value"] == "/tmp/x"

    def test_unwritable_directory_does_not_raise(self, tmp_path):
        target = tmp_path / "nope"
        target.mkdir()
        target.chmod(0o500)  # read+execute, no write
        try:
            rec = _begin(target)
            rec.complete(status="converged", results={})  # must not raise
        finally:
            target.chmod(0o700)

    def test_no_partial_file_left_by_failed_serialization(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="converged", results={})

        class Boom:
            def __repr__(self):
                raise RuntimeError("boom")

        # begin()'s own phase-one write legitimately changes the file (new
        # stage, status back to "running") -- that happens before the
        # failing call below, so the byte-identity check must anchor to
        # *this* state, not to the state before begin() ran.
        rec2 = _begin(tmp_path, append=True)
        before = (tmp_path / RECORD_FILENAME).read_text()
        rec2.complete(status="converged", results={"bad": Boom()})
        # A failed complete() (serialization raises before the file is ever
        # touched) must leave the record exactly as it was -- byte-for-byte,
        # never truncated, never partially overwritten.
        text = (tmp_path / RECORD_FILENAME).read_text()
        assert text == before
        assert len(list(tmp_path.glob(f"{RECORD_FILENAME}.tmp*"))) == 0


class TestHelpers:
    def test_batch_ids_are_unique(self):
        assert new_batch_id() != new_batch_id()

    def test_provenance_records_requested_and_resolved_device(self):
        prov = collect_provenance(mlip_model="uma-s-1p2",
                                  device_requested="auto",
                                  device_resolved="cuda")
        assert prov["device_requested"] == "auto"
        assert prov["device_resolved"] == "cuda"
        assert prov["mlip_model"] == "uma-s-1p2"
        assert prov["mlip_package"]["name"] == "fairchem-core"
        assert prov["mliprun_version"]
        assert prov["ase_version"]
        assert prov["hostname"]
        assert prov["started_at"] is None

    @pytest.mark.parametrize("model,expected", [
        ("uma-s-1p2", "fairchem-core"),
        ("mace", "mace-torch"),
        ("mace-mh-1", "mace-torch"),
        ("7net-mf-ompa", "sevenn"),
        ("chgnet", "chgnet"),
        ("something-else", None),
    ])
    def test_mlip_package_mapping(self, model, expected):
        prov = collect_provenance(mlip_model=model, device_requested="cpu",
                                  device_resolved="cpu")
        assert prov["mlip_package"]["name"] == expected


class TestReviewFixes:
    """Regression tests for the task-1 code review findings (C1, C2, I1, I2,
    I3, M1). M2 (unused import) and M3 (test-only) have no dedicated test
    here -- M3's fix is folded into
    TestRobustness.test_no_partial_file_left_by_failed_serialization above.
    """

    def test_begin_with_invalid_output_dir_does_not_raise(self):
        """C1: `Path(output_dir)` construction happens inside the try block,
        so a bad `output_dir` yields a dead handle instead of an uncaught
        TypeError."""
        rec = RunRecord.begin(None, command="optimize", stage_kind="optimize",
                              parameters={}, inputs={}, provenance={})
        rec.complete(status="converged", results={})  # must be a silent no-op

    def test_collect_provenance_tolerates_none_model(self):
        """C2: every fallible call in collect_provenance is individually
        guarded, including a non-string mlip_model reaching _mlip_package."""
        prov = collect_provenance(mlip_model=None, device_requested="cpu",
                                  device_resolved="cpu")
        assert prov["mlip_package"] == {"name": None, "version": None}

    def test_multistage_walltime_is_summed(self, tmp_path):
        """I1 (human decision: SUM): top-level provenance.walltime_s is the
        sum of every stage's walltime_s, not just the latest one."""
        rec = _begin(tmp_path)
        time.sleep(0.05)
        rec.complete(status="converged", results={})

        rec2 = _begin(tmp_path, stage_kind="neb-restart", append=True)
        time.sleep(0.15)
        rec2.complete(status="converged", results={})

        data = _read(tmp_path)
        stages = data["stages"]
        # Measurably different durations, per the finding's test recipe.
        assert stages[1]["walltime_s"] > stages[0]["walltime_s"]
        expected = stages[0]["walltime_s"] + stages[1]["walltime_s"]
        assert data["provenance"]["walltime_s"] == pytest.approx(expected, abs=1e-6)

    def test_append_records_only_changed_provenance_fields(self, tmp_path):
        """I2 (human decision): an appended stage's stage_provenance holds
        only the fields that differ from the run's origin provenance; the
        origin itself is never overwritten."""
        rec = _begin(tmp_path, provenance={"mliprun_version": "0.4.0",
                                           "hostname": "node-a",
                                           "device_resolved": "cpu",
                                           "mlip_model": "uma-s-1p2"})
        rec.complete(status="converged", results={})

        rec2 = _begin(tmp_path, stage_kind="neb-restart", append=True,
                      provenance={"mliprun_version": "0.4.1",
                                  "hostname": "node-b",
                                  "device_resolved": "cpu",
                                  "mlip_model": "uma-s-1p2"})
        rec2.complete(status="converged", results={})

        data = _read(tmp_path)
        assert data["provenance"]["hostname"] == "node-a"
        assert data["provenance"]["mliprun_version"] == "0.4.0"
        assert data["stages"][1]["stage_provenance"] == {
            "mliprun_version": "0.4.1",
            "hostname": "node-b",
        }
        assert "stage_provenance" not in data["stages"][0]

    def test_append_with_same_environment_writes_no_stage_provenance(self, tmp_path):
        """I2: a same-environment append omits stage_provenance entirely --
        never an empty dict."""
        prov = {"mliprun_version": "0.4.0", "hostname": "node-a",
                "device_resolved": "cpu", "mlip_model": "uma-s-1p2"}
        rec = _begin(tmp_path, provenance=dict(prov))
        rec.complete(status="converged", results={})

        rec2 = _begin(tmp_path, stage_kind="neb-restart", append=True,
                      provenance=dict(prov))
        rec2.complete(status="converged", results={})

        data = _read(tmp_path)
        assert "stage_provenance" not in data["stages"][1]

    def test_invalid_param_source_coerced_to_unspecified(self, tmp_path):
        """I3: only the five documented source tags are accepted; anything
        else is coerced to 'unspecified' rather than written verbatim."""
        ctx = RunContext(command="x", param_sources={"fmax": "bogus"})
        _begin(tmp_path, run_context=ctx)
        assert _read(tmp_path)["parameters"]["fmax"]["source"] == "unspecified"

    def test_non_dict_json_record_is_backed_up(self, tmp_path):
        """M1: a record that parses to valid JSON but is not an object (e.g.
        a bare `[]`) gets the same back-up-then-continue treatment as
        unparseable JSON, instead of being silently discarded."""
        (tmp_path / RECORD_FILENAME).write_text("[]")
        rec = _begin(tmp_path, append=True)
        rec.complete(status="converged", results={})
        backups = list(tmp_path.glob(f"{RECORD_FILENAME}.corrupt-*"))
        assert len(backups) == 1
        assert backups[0].read_text() == "[]"


class TestHeadProvenance:
    """CANON C1/C3: the head/task is its own decision and must never be
    inferred. A record naming only the model tag cannot identify the level
    of theory."""

    def test_head_fields_default_to_none(self):
        prov = collect_provenance(mlip_model="chgnet", device_requested="cpu",
                                  device_resolved="cpu")
        assert prov["uma_task"] is None
        assert prov["mace_head"] is None

    def test_records_uma_task_for_uma_model(self):
        prov = collect_provenance(mlip_model="uma-s-1p2", device_requested="cuda",
                                  device_resolved="cuda", uma_task="oc25")
        assert prov["uma_task"] == "oc25"
        assert prov["mace_head"] is None

    def test_records_mace_head_for_mace_model(self):
        prov = collect_provenance(mlip_model="mace-mh-1", device_requested="cpu",
                                  device_resolved="cpu", mace_head="omat_pbe")
        assert prov["mace_head"] == "omat_pbe"
        assert prov["uma_task"] is None

    def test_drops_head_that_does_not_match_the_model_family(self):
        """A CLI passes its parsed --uma-task unconditionally, defaults
        included. A MACE run must not inherit a UMA task it never used."""
        prov = collect_provenance(mlip_model="mace-mh-1", device_requested="cpu",
                                  device_resolved="cpu", uma_task="omat",
                                  mace_head="omat_pbe")
        assert prov["uma_task"] is None
        assert prov["mace_head"] == "omat_pbe"

    def test_tolerates_non_string_model(self):
        prov = collect_provenance(mlip_model=None, device_requested="cpu",
                                  device_resolved="cpu", uma_task="omat")
        assert prov["uma_task"] is None
        assert prov["mace_head"] is None


class TestSchemaVersion:
    def test_schema_version_is_four(self):
        """2 added uma_task/mace_head; 3 adds sevennet_task; 4 adds the
        committee block, because collect_provenance took a single
        mlip_model and a committee needs a list."""
        assert SCHEMA_VERSION == 4

    def test_written_record_carries_the_new_version(self, tmp_path):
        _begin(tmp_path)
        assert _read(tmp_path)["schema_version"] == 4


class TestHeadInStageProvenance:
    def test_append_records_a_switched_head(self, tmp_path):
        """C3 forbids mixing heads within an energy formula, so a resume
        that switches head is exactly what must not go unrecorded."""
        base = {"mliprun_version": "0.4.0", "hostname": "node-a",
                "device_resolved": "cuda", "mlip_model": "uma-s-1p2",
                "uma_task": "omat", "mace_head": None}
        rec = _begin(tmp_path, provenance=dict(base))
        rec.complete(status="converged")

        rec2 = _begin(tmp_path, provenance=dict(base, uma_task="oc25"),
                      append=True)
        rec2.complete(status="converged")

        data = _read(tmp_path)
        assert data["stages"][1]["stage_provenance"] == {"uma_task": "oc25"}
        assert data["provenance"]["uma_task"] == "omat"
        # Present-and-different is a real change, not a new field.
        assert NEW_FIELDS_KEY not in data["stages"][1]

    def test_append_with_same_head_writes_no_stage_provenance(self, tmp_path):
        base = {"mliprun_version": "0.4.0", "hostname": "node-a",
                "device_resolved": "cuda", "mlip_model": "uma-s-1p2",
                "uma_task": "omat", "mace_head": None}
        rec = _begin(tmp_path, provenance=dict(base))
        rec.complete(status="converged")
        rec2 = _begin(tmp_path, provenance=dict(base), append=True)
        rec2.complete(status="converged")
        stage = _read(tmp_path)["stages"][1]
        assert "stage_provenance" not in stage
        # Present-and-same: neither key. A null `mace_head` on both sides is
        # "no MACE head applies", not a fact worth flagging.
        assert NEW_FIELDS_KEY not in stage


#: A record as schema 1 wrote it: no `uma_task` / `mace_head` keys anywhere.
_SCHEMA_1_PROVENANCE = {
    "mliprun_version": "0.4.0",
    "ase_version": "3.23.0",
    "mlip_package": {"name": "fairchem-core", "version": "2.0.0"},
    "mlip_model": "uma-s-1p2",
    "device_requested": "cuda",
    "device_resolved": "cuda",
    "python_version": "3.11.9",
    "hostname": "node-a",
    "started_at": "2026-07-01T09:00:00+08:00",
    "finished_at": "2026-07-01T11:00:00+08:00",
    "walltime_s": 7200.0,
}


def _write_schema_1_record(tmp_path):
    """Drop a genuine schema-1 record on disk for an append to resume."""
    (tmp_path / RECORD_FILENAME).write_text(json.dumps({
        "schema_version": 1,
        "command": "neb",
        "status": "converged",
        "run": {"mode": "one-off", "batch": None},
        "inputs": {"n_images": 5, "n_atoms": 32},
        "parameters": {},
        "provenance": dict(_SCHEMA_1_PROVENANCE),
        "stages": [{
            "index": 0, "kind": "neb", "status": "converged",
            "started_at": "2026-07-01T09:00:00+08:00",
            "walltime_s": 7200.0, "steps": 120, "results": {},
        }],
    }, indent=2))


def _schema_2_provenance(**overrides):
    """The same environment, described by schema-2 collect_provenance()."""
    prov = dict(_SCHEMA_1_PROVENANCE, uma_task="omat", mace_head=None)
    prov.update(overrides)
    return prov


class TestNewFieldsMarker:
    """I4 (human ruling): a diff field the stored record has no key for is
    reported as *new*, never as *changed*.

    A schema-1 record predates `uma_task`/`mace_head`, so comparing with
    `.get()` on both sides made "never written" indistinguishable from
    "null" -- and a same-head resume of a legacy UMA run claimed the head
    had switched. That is a CANON C3 false positive in the artifact whose
    whole job is answering C3 questions.
    """

    def test_same_head_resume_of_a_legacy_record_claims_no_change(self, tmp_path):
        _write_schema_1_record(tmp_path)
        rec = _begin(tmp_path, command="neb", stage_kind="neb-restart",
                     append=True, provenance=_schema_2_provenance())
        rec.complete(status="converged")

        stage = _read(tmp_path)["stages"][1]
        # The head did NOT change, so nothing may go to stage_provenance.
        assert "stage_provenance" not in stage
        assert stage[NEW_FIELDS_KEY] == {"uma_task": "omat", "mace_head": None}

    def test_marker_carries_this_stage_s_head_not_a_guess_at_stage_0(self, tmp_path):
        """The reader must see what ran *now*; stage 0 stays unknown."""
        _write_schema_1_record(tmp_path)
        rec = _begin(tmp_path, command="neb", stage_kind="neb-restart",
                     append=True,
                     provenance=_schema_2_provenance(uma_task="oc25"))
        rec.complete(status="converged")

        data = _read(tmp_path)
        assert data["stages"][1][NEW_FIELDS_KEY]["uma_task"] == "oc25"
        # The origin provenance is never back-filled: stage 0's head is not
        # knowable from this record, and inventing it would be worse.
        assert "uma_task" not in data["provenance"]

    def test_legacy_append_leaves_schema_version_at_one(self, tmp_path):
        """The record becomes a hybrid, and says so honestly. Rewriting it
        to 2 would assert stage 0 carried fields it never did."""
        _write_schema_1_record(tmp_path)
        rec = _begin(tmp_path, command="neb", stage_kind="neb-restart",
                     append=True, provenance=_schema_2_provenance())
        rec.complete(status="converged")
        assert _read(tmp_path)["schema_version"] == 1

    def test_a_real_change_still_reaches_stage_provenance(self, tmp_path):
        """Both keys can appear at once: the hostname genuinely moved, and
        the head fields are merely new."""
        _write_schema_1_record(tmp_path)
        rec = _begin(tmp_path, command="neb", stage_kind="neb-restart",
                     append=True,
                     provenance=_schema_2_provenance(hostname="node-b"))
        rec.complete(status="converged")

        stage = _read(tmp_path)["stages"][1]
        assert stage["stage_provenance"] == {"hostname": "node-b"}
        assert stage[NEW_FIELDS_KEY] == {"uma_task": "omat", "mace_head": None}

    def test_field_absent_from_the_incoming_provenance_is_reported_nowhere(self, tmp_path):
        """A minimal caller that omits the head fields entirely has nothing
        to say about them -- neither key may appear."""
        _write_schema_1_record(tmp_path)
        rec = _begin(tmp_path, command="neb", stage_kind="neb-restart",
                     append=True, provenance=dict(_SCHEMA_1_PROVENANCE))
        rec.complete(status="converged")

        stage = _read(tmp_path)["stages"][1]
        assert "stage_provenance" not in stage
        assert NEW_FIELDS_KEY not in stage


class TestSevenNetTaskProvenance:
    """schema 3: the SevenNet task joins uma_task and mace_head.

    Without it a SevenNet record cannot say which task produced its numbers,
    so under CANON C1/C3 it cannot back any energy entering a formula.
    """

    def test_sevennet_run_carries_the_task(self):
        prov = collect_provenance(
            mlip_model="7net-omni", device_requested="auto",
            device_resolved="cuda", sevennet_task="oc20",
        )
        assert prov["sevennet_task"] == "oc20"
        assert prov["uma_task"] is None
        assert prov["mace_head"] is None

    def test_task_is_recorded_verbatim(self):
        # 7net-mf-0's tasks are uppercase; the record must not normalise them.
        prov = collect_provenance(
            mlip_model="7net-mf-0", device_requested="cpu",
            device_resolved="cpu", sevennet_task="R2SCAN",
        )
        assert prov["sevennet_task"] == "R2SCAN"

    def test_non_sevennet_run_nulls_the_task(self):
        # CLIs pass whatever their option resolved to regardless of model, so
        # the field is gated on the tag -- a MACE run must not be recorded as
        # carrying a SevenNet task it never used.
        prov = collect_provenance(
            mlip_model="mace", device_requested="cpu",
            device_resolved="cpu", sevennet_task="oc20",
        )
        assert prov["sevennet_task"] is None

    def test_key_is_always_present(self):
        prov = collect_provenance(
            mlip_model="uma-s-1p2", device_requested="cpu",
            device_resolved="cpu", uma_task="omat",
        )
        assert "sevennet_task" in prov
        assert prov["sevennet_task"] is None

    def test_task_switch_between_stages_is_reported_as_changed(self):
        top = {"mlip_model": "7net-omni", "sevennet_task": "mpa"}
        incoming = {"mlip_model": "7net-omni", "sevennet_task": "oc20"}
        changed, new = _split_provenance(incoming, top)
        assert changed == {"sevennet_task": "oc20"}
        assert new == {}

    def test_absent_key_in_a_legacy_record_is_new_not_changed(self):
        # A schema-2 record has no sevennet_task key at all. That is
        # "unknown", not "different"; reporting it as a change would be a C3
        # false positive in the very artifact that answers C3 questions.
        top = {"mlip_model": "7net-omni"}
        incoming = {"mlip_model": "7net-omni", "sevennet_task": "mpa"}
        changed, new = _split_provenance(incoming, top)
        assert changed == {}
        assert new == {"sevennet_task": "mpa"}


class TestCommitteeProvenance:
    """Schema 4: a committee run records a list of members, not one model."""

    def _committee(self):
        return {
            "members": [
                {"name": "uma-s-1p2@oc20", "mlip": "uma-s-1p2",
                 "uma_task": "oc20", "mace_head": None, "sevennet_task": None,
                 "env": "/scratchb/juar/.venv/uma",
                 "python": "/scratchb/juar/.venv/uma/bin/python",
                 "device": "auto", "gpu": 0,
                 "level_of_theory": "RPBE/OC20"},
                {"name": "7net-omni@oc20", "mlip": "7net-omni",
                 "uma_task": None, "mace_head": None,
                 "sevennet_task": "oc20",
                 "env": "/scratchb/juar/.venv/sevenn",
                 "python": "/scratchb/juar/.venv/sevenn/bin/python",
                 "device": "auto", "gpu": 1,
                 "level_of_theory": "RPBE/OC20"},
            ],
            "levels": ["RPBE/OC20"],
            "mixed_theory": False,
            "config_sha256": "a" * 64,
            "config_path": "/work/committee.yaml",
        }

    def test_the_committee_block_is_recorded_verbatim(self):
        prov = collect_provenance(mlip_model="committee",
                                  device_requested="auto",
                                  device_resolved="cuda",
                                  committee=self._committee())
        assert len(prov["committee"]["members"]) == 2
        assert prov["committee"]["members"][0]["level_of_theory"] == "RPBE/OC20"
        assert prov["committee"]["mixed_theory"] is False
        assert prov["committee_config_sha256"] == "a" * 64

    def test_the_config_hash_is_promoted_to_a_flat_field(self):
        """Flat, so an appended stage can report 'the committee changed' as
        one string rather than diffing a nested member list."""
        prov = collect_provenance(mlip_model="committee",
                                  device_requested="auto",
                                  device_resolved="cuda",
                                  committee=self._committee())
        assert prov["committee_config_sha256"] == prov["committee"][
            "config_sha256"]

    def test_a_single_model_run_gains_no_committee_keys(self):
        """Single-model runs keep writing exactly what they write today."""
        prov = collect_provenance(mlip_model="uma-s-1p2",
                                  device_requested="auto",
                                  device_resolved="cuda", uma_task="oc20")
        assert "committee" not in prov
        assert "committee_config_sha256" not in prov

    def test_head_gating_still_applies_to_a_committee_run(self):
        """model tag 'committee' is not a UMA/MACE/SevenNet tag, so the
        top-level head fields stay null -- the heads live per member."""
        prov = collect_provenance(mlip_model="committee",
                                  device_requested="auto",
                                  device_resolved="cpu", uma_task="oc20",
                                  committee=self._committee())
        assert prov["uma_task"] is None
        assert prov["mace_head"] is None
        assert prov["sevennet_task"] is None

    def test_a_malformed_committee_block_does_not_raise(self):
        """collect_provenance must be total: it runs outside RunRecord.begin's
        try, so anything it raises kills the run."""
        prov = collect_provenance(mlip_model="committee",
                                  device_requested="auto",
                                  device_resolved="cpu",
                                  committee={"members": {1, 2, 3}})
        assert "committee" in prov

    def test_a_changed_committee_shows_up_in_an_appended_stage(self, tmp_path):
        RunRecord.begin(tmp_path, command="optimize", stage_kind="optimize",
                        parameters={}, inputs={},
                        provenance=collect_provenance(
                            mlip_model="committee", device_requested="auto",
                            device_resolved="cpu",
                            committee=self._committee())).complete(
                                status="converged")
        changed = self._committee()
        changed["config_sha256"] = "b" * 64
        RunRecord.begin(tmp_path, command="optimize", stage_kind="optimize",
                        parameters={}, inputs={},
                        provenance=collect_provenance(
                            mlip_model="committee", device_requested="auto",
                            device_resolved="cpu", committee=changed),
                        append=True).complete(status="converged")

        stage = _read(tmp_path)["stages"][1]
        assert stage["stage_provenance"]["committee_config_sha256"] == "b" * 64
