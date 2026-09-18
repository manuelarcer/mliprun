"""Where the stationary-point expectation comes from (D5: warn, never refuse)."""
import json

import pytest

from mliprun.core.vibrations import resolve_fmax_expectation


def _record(tmp_path, stages):
    (tmp_path / "mliprun_run.json").write_text(
        json.dumps({"schema_version": 5, "stages": stages}))
    return tmp_path


def test_an_explicit_value_wins():
    value, source = resolve_fmax_expectation(0.02, structure_dir=None)
    assert value == pytest.approx(0.02)
    assert source == "explicit"


def test_an_explicit_value_wins_even_over_a_record(tmp_path):
    directory = _record(tmp_path, [
        {"kind": "optimize", "status": "converged",
         "parameters": {"fmax": {"value": 0.05, "source": "user"}}}])
    value, source = resolve_fmax_expectation(0.01, structure_dir=directory)
    assert value == pytest.approx(0.01)
    assert source == "explicit"


def test_a_converged_optimize_stage_supplies_the_expectation(tmp_path):
    directory = _record(tmp_path, [
        {"kind": "optimize", "status": "converged",
         "parameters": {"fmax": {"value": 0.05, "source": "user"}}}])
    value, source = resolve_fmax_expectation(None, structure_dir=directory)
    assert value == pytest.approx(0.05)
    assert source == "run_record"


def test_the_latest_converged_optimize_stage_wins(tmp_path):
    directory = _record(tmp_path, [
        {"kind": "optimize", "status": "converged",
         "parameters": {"fmax": {"value": 0.05, "source": "user"}}},
        {"kind": "optimize", "status": "converged",
         "parameters": {"fmax": {"value": 0.01, "source": "user"}}}])
    value, source = resolve_fmax_expectation(None, structure_dir=directory)
    assert value == pytest.approx(0.01)


def test_a_not_converged_stage_supplies_nothing(tmp_path):
    """The fmax it was aiming at is not a criterion it met."""
    directory = _record(tmp_path, [
        {"kind": "optimize", "status": "not_converged",
         "parameters": {"fmax": {"value": 0.05, "source": "user"}}}])
    value, source = resolve_fmax_expectation(None, structure_dir=directory)
    assert value is None
    assert source == "none"


def test_a_non_optimize_stage_supplies_nothing(tmp_path):
    directory = _record(tmp_path, [
        {"kind": "md", "status": "completed",
         "parameters": {"fmax": {"value": 0.05, "source": "user"}}}])
    value, source = resolve_fmax_expectation(None, structure_dir=directory)
    assert value is None
    assert source == "none"


def test_no_record_at_all_supplies_nothing(tmp_path):
    value, source = resolve_fmax_expectation(None, structure_dir=tmp_path)
    assert value is None
    assert source == "none"


def test_an_unreadable_record_supplies_nothing_and_does_not_raise(tmp_path):
    """A corrupt record must never cost the calculation."""
    (tmp_path / "mliprun_run.json").write_text("{not json")
    value, source = resolve_fmax_expectation(None, structure_dir=tmp_path)
    assert value is None
    assert source == "none"


def test_an_untagged_parameter_value_is_read_too(tmp_path):
    """A stage's own `parameters` block can still carry a bare value rather
    than the `{"value":, "source":}` shape `run_record._tag` produces.

    Correction from the brief: `_tag` was checked directly
    (`src/mliprun/core/run_record.py`) and it wraps *every* parameter value
    unconditionally, with or without a `RunContext` -- with none, it simply
    tags the source as "unspecified" rather than leaving the value bare. So
    a live record written through `_tag` never carries a bare value; this
    guards against one anyway, because nothing here can guarantee an
    on-disk record was written through `_tag` at all (a hand-edited file,
    or a schema variant predating it). The provenance-must-never-raise rule
    means this case has to degrade gracefully, not that it is expected in
    practice.
    """
    directory = _record(tmp_path, [
        {"kind": "optimize", "status": "converged",
         "parameters": {"fmax": 0.03}}])
    value, source = resolve_fmax_expectation(None, structure_dir=directory)
    assert value == pytest.approx(0.03)
    assert source == "run_record"


def test_a_converged_optimize_stage_with_no_stage_parameters_falls_back_to_the_record_top_level(tmp_path):
    """This is the shape `mliprun optimize` actually writes.

    Verified against `src/mliprun/core/optimize.py` and
    `tests/test_run_record_integration.py::TestOptimizeRecord::
    test_library_caller_gets_a_complete_record`: `RunRecord.begin()` is
    called there without `stage_parameters`, so the "optimize" stage in
    `stages` carries no `parameters` key of its own -- `fmax` lives only in
    the record's top-level `parameters`, tagged exactly like any other
    parameter. Only NEB's restart stages pass `stage_parameters`
    (`src/mliprun/core/neb.py`). Without this fallback,
    `resolve_fmax_expectation` would return `(None, "none")` for every
    record a real `optimize` run produces -- the function's main intended
    case.
    """
    (tmp_path / "mliprun_run.json").write_text(json.dumps({
        "schema_version": 5,
        "command": "optimize",
        "status": "converged",
        "parameters": {"fmax": {"value": 0.04, "source": "user"}},
        "stages": [
            {"index": 0, "kind": "optimize", "status": "converged"},
        ],
    }))
    value, source = resolve_fmax_expectation(None, structure_dir=tmp_path)
    assert value == pytest.approx(0.04)
    assert source == "run_record"


def test_the_top_level_fallback_is_not_borrowed_from_a_different_origin_command(tmp_path):
    """The top-level `parameters` block belongs to whichever command first
    created the record (`payload["command"]`), since `RunRecord.begin`
    writes it once, only when the record has no prior history, and never
    rewrites it on append. A stage that is kind="optimize" but not the
    record's own origin command must not borrow that unrelated block, even
    if it happens to contain a same-named key.
    """
    (tmp_path / "mliprun_run.json").write_text(json.dumps({
        "schema_version": 5,
        "command": "singlepoint",
        "status": "converged",
        "parameters": {"fmax": {"value": 0.04, "source": "user"}},
        "stages": [
            {"index": 0, "kind": "optimize", "status": "converged"},
        ],
    }))
    value, source = resolve_fmax_expectation(None, structure_dir=tmp_path)
    assert value is None
    assert source == "none"
