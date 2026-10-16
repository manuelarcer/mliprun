"""Parameter schema and ``validate_submission`` (plan section 3)."""
from __future__ import annotations

import pytest

from mliprun.cli import utils as cli_utils
from mliprun.webapp import schema
from mliprun.webapp.schema import ValidationFailed, get_calc_types, validate_submission

FIELD_KEYS = {"name", "label", "type", "default", "unit", "help", "min", "max",
              "min_exclusive", "max_exclusive", "choices", "nullable", "advanced",
              "visible_if", "input_modes", "cli_flag"}
EMT = {"name": "emt", "device": "auto"}


def meta(n=4, symbols=None, n_frames=1, same=True):
    return {"natoms": n, "symbols": symbols or ["Cu"] * n, "n_frames": n_frames,
            "pbc": [True, True, True], "frames_same_symbols": same}


def inputs_for(ct):
    kind = schema.get_calc_type(ct)["inputs"]
    if kind == "single":
        return {"structure": meta()}
    return {"initial": meta(), "final": meta()}


def errors_of(*args):
    with pytest.raises(ValidationFailed) as info:
        validate_submission(*args)
    return info.value.errors


def fields_with(errors, field):
    return [e for e in errors if e["field"] == field]


def test_payload_shape_and_deep_copy():
    data = get_calc_types()
    assert data["schema_version"] == 1
    names = [c["name"] for c in data["calc_types"]]
    assert names == ["singlepoint", "optimize", "freq", "md", "neb", "autoneb"]
    data["calc_types"][0]["fields"][0]["default"] = "mutated"
    assert get_calc_types()["calc_types"][0]["fields"][0]["default"] == "auto"


@pytest.mark.parametrize("ct", ["singlepoint", "optimize", "freq", "md", "neb", "autoneb"])
def test_every_calc_type_has_complete_fields(ct):
    entry = schema.get_calc_type(ct)
    assert entry["fields"]
    assert entry["inputs"] in ("single", "pair", "pair_or_band")
    seen = set()
    for field in entry["fields"]:
        assert set(field) == FIELD_KEYS
        assert field["type"] in ("float", "int", "bool", "choice", "indices", "mask3")
        assert field["help"]
        for clause in field["visible_if"] or []:
            for controlling in clause:
                assert controlling in seen, (ct, field["name"], controlling)
        seen.add(field["name"])
        if field["type"] == "choice":
            assert field["default"] in field["choices"]


@pytest.mark.parametrize("ct", ["singlepoint", "optimize", "freq", "md", "neb", "autoneb"])
def test_defaults_pass_their_own_validation(ct):
    out = validate_submission(ct, EMT, {}, inputs_for(ct))
    assert set(out["param_sources"].values()) == {"default"}
    for field in schema.get_calc_type(ct)["fields"]:
        if field["name"] in out["params"] and field["default"] is not None:
            _, err = schema.check_value(field, field["default"], natoms=4)
            assert err is None, (ct, field["name"], err)
    assert out["calculator"] == {"name": "emt", "device": "auto", "uma_task": None,
                                 "mace_head": None, "sevennet_task": None}


def test_explicit_defaults_round_trip_as_user():
    out = validate_submission("optimize", EMT, {"fmax": 0.02, "max_steps": 10.0},
                              inputs_for("optimize"))
    assert out["params"]["fmax"] == 0.02
    assert out["params"]["max_steps"] == 10 and isinstance(out["params"]["max_steps"], int)
    assert out["param_sources"]["fmax"] == "user"
    assert out["param_sources"]["optimizer"] == "default"


def test_md_visibility_matches_cli_rules():
    out = validate_submission("md", EMT, {"ensemble": "nve"}, inputs_for("md"))
    assert "temperature" not in out["params"] and "thermostat" not in out["params"]
    out = validate_submission("md", EMT, {"ensemble": "npt", "barostat": "berendsen"},
                              inputs_for("md"))
    assert {"taut", "taup", "compressibility", "barostat_mask"} <= set(out["params"])
    assert "pfactor" not in out["params"] and "ttime" not in out["params"]
    out = validate_submission("md", EMT, {"ensemble": "npt"}, inputs_for("md"))
    assert out["params"]["pfactor"] is None and out["params"]["ttime"] == 25.0
    assert "compressibility" not in out["params"]


def test_unknown_calc_type():
    errs = errors_of("benchmark", EMT, {}, {"structure": meta()})
    assert errs[0]["field"] == "calc_type"


def test_hidden_field_present_is_rejected():
    errs = errors_of("md", EMT, {"ensemble": "nvt", "compressibility": 0.3},
                     inputs_for("md"))
    assert fields_with(errs, "params.compressibility")[0]["message"] == \
        "does not apply when ensemble=nvt"
    errs = errors_of("neb", EMT, {"n_images": 3},
                     {"band": meta(n_frames=5)})
    assert "input mode=band" in fields_with(errs, "params.n_images")[0]["message"]
    errs = errors_of("neb", EMT, {"dyneb": False, "scale_fmax": 0.1}, inputs_for("neb"))
    assert fields_with(errs, "params.scale_fmax")[0]["message"] == \
        "does not apply when dyneb=false"


def test_unknown_parameter():
    errs = errors_of("optimize", EMT, {"fmaxx": 1}, inputs_for("optimize"))
    assert fields_with(errs, "params.fmaxx")[0]["message"] == "unknown parameter"


def test_out_of_range_and_wrong_type_all_reported():
    errs = errors_of("optimize", EMT,
                     {"fmax": 0, "max_steps": 1.5, "relax_cell": "yes", "optimizer": "BFGS"},
                     inputs_for("optimize"))
    assert fields_with(errs, "params.fmax")[0]["message"] == "must be > 0"
    assert fields_with(errs, "params.max_steps")[0]["message"] == "must be an integer"
    assert fields_with(errs, "params.relax_cell")[0]["message"] == "must be true or false"
    assert "must be one of" in fields_with(errs, "params.optimizer")[0]["message"]
    errs = errors_of("optimize", EMT, {"fmax": True, "max_steps": 100001},
                     inputs_for("optimize"))
    assert fields_with(errs, "params.fmax")[0]["message"] == "must be a number"
    assert fields_with(errs, "params.max_steps")[0]["message"] == "must be <= 100000"


def test_nullable_and_choice_types():
    errs = errors_of("optimize", EMT, {"fmax": None}, inputs_for("optimize"))
    assert fields_with(errs, "params.fmax")[0]["message"] == "must not be null"
    out = validate_submission("freq", EMT, {"expect_fmax": None, "nfree": 4},
                              inputs_for("freq"))
    assert out["params"]["expect_fmax"] is None and out["params"]["nfree"] == 4
    errs = errors_of("freq", EMT, {"nfree": "4"}, inputs_for("freq"))
    assert fields_with(errs, "params.nfree")


def test_bad_indices():
    ok = validate_submission("freq", EMT, {"indices": " 0-1, 3 "}, inputs_for("freq"))
    assert ok["params"]["indices"] == "0-1, 3"
    for bad, needle in (("0,9", "out of range"), ("a", "could not parse"),
                        (",", "at least one"), (3, "must be a string")):
        errs = errors_of("freq", EMT, {"indices": bad}, inputs_for("freq"))
        assert needle in fields_with(errs, "params.indices")[0]["message"], bad


def test_mask3():
    out = validate_submission("md", EMT, {"ensemble": "npt", "barostat_mask": "0, 0,1"},
                              inputs_for("md"))
    assert out["params"]["barostat_mask"] == "0,0,1"
    errs = errors_of("md", EMT, {"ensemble": "npt", "barostat_mask": "1,2,1"},
                     inputs_for("md"))
    assert fields_with(errs, "params.barostat_mask")


def test_input_roles_must_match_kind():
    errs = errors_of("optimize", EMT, {}, {"initial": meta(), "final": meta()})
    assert fields_with(errs, "inputs")
    errs = errors_of("autoneb", EMT, {}, {"band": meta(n_frames=5)})
    assert fields_with(errs, "inputs")


def test_neb_atom_mismatch():
    errs = errors_of("neb", EMT, {}, {"initial": meta(4), "final": meta(5)})
    assert fields_with(errs, "inputs.final")
    errs = errors_of("autoneb", EMT, {}, {"initial": meta(2, ["Cu", "Pt"]),
                                          "final": meta(2, ["Pt", "Cu"])})
    assert "order" in fields_with(errs, "inputs.final")[0]["message"]


def test_neb_band_checks():
    assert validate_submission("neb", EMT, {}, {"band": meta(n_frames=3)})
    errs = errors_of("neb", EMT, {}, {"band": meta(n_frames=2)})
    assert fields_with(errs, "inputs.band")
    errs = errors_of("neb", EMT, {}, {"band": meta(n_frames=53, same=False)})
    assert len(fields_with(errs, "inputs.band")) == 2


def test_relax_atoms_checked_against_initial():
    out = validate_submission("neb", EMT, {"relax_atoms": "0-2"}, inputs_for("neb"))
    assert out["params"]["relax_atoms"] == "0-2"
    errs = errors_of("neb", EMT, {"relax_atoms": "7"}, inputs_for("neb"))
    assert fields_with(errs, "params.relax_atoms")


def test_emt_unsupported_element():
    errs = errors_of("singlepoint", EMT, {}, {"structure": meta(2, ["Fe", "Cu"])})
    msg = fields_with(errs, "calculator.name")[0]["message"]
    assert "Fe" in msg and "Cu" not in msg.split(";")[0]


def test_missing_uma_task(monkeypatch):
    monkeypatch.setattr(cli_utils, "FAIRCHEM_AVAILABLE", True)
    errs = errors_of("singlepoint", {"name": "uma-s-1p2"}, {}, inputs_for("singlepoint"))
    assert "required" in fields_with(errs, "calculator.uma_task")[0]["message"]
    out = validate_submission("singlepoint", {"name": "uma-s-1p2", "uma_task": "omat"},
                              {}, inputs_for("singlepoint"))
    assert out["calculator"]["uma_task"] == "omat"


def test_task_on_single_task_model(monkeypatch):
    monkeypatch.setattr(cli_utils, "SEVENN_AVAILABLE", True)
    errs = errors_of("singlepoint", {"name": "7net-0", "sevennet_task": "mpa"}, {},
                     inputs_for("singlepoint"))
    assert "single-task" in fields_with(errs, "calculator.sevennet_task")[0]["message"]


def test_errors_are_all_collected():
    errs = errors_of("md", {"name": "emt", "device": "tpu"},
                     {"steps": 0, "temperature": -1, "bogus": 1},
                     {"structure": meta(2, ["Fe", "Fe"])})
    fields = {e["field"] for e in errs}
    assert {"params.steps", "params.temperature", "params.bogus",
            "calculator.device", "calculator.name"} <= fields
