"""Calculator listing, validation and construction (plan section 7)."""
from __future__ import annotations

import subprocess
import sys

import pytest
import typer
from ase.calculators.emt import EMT

from mliprun.cli import utils as cli_utils
from mliprun.webapp import calculators


@pytest.fixture
def nothing_installed(monkeypatch):
    for flag in ("FAIRCHEM_AVAILABLE", "MACE_AVAILABLE", "SEVENN_AVAILABLE",
                 "CHGNET_AVAILABLE"):
        monkeypatch.setattr(cli_utils, flag, False)


@pytest.fixture
def everything_installed(monkeypatch):
    for flag in ("FAIRCHEM_AVAILABLE", "MACE_AVAILABLE", "SEVENN_AVAILABLE",
                 "CHGNET_AVAILABLE"):
        monkeypatch.setattr(cli_utils, flag, True)


def test_emt_only_when_nothing_installed(nothing_installed):
    data = calculators.list_calculators()
    assert [c["name"] for c in data["calculators"]] == ["emt"]
    assert data["default"] == "emt"
    emt = data["calculators"][0]
    assert emt["elements"] == ["Ag", "Al", "Au", "C", "Cu", "H", "N", "Ni", "O", "Pd", "Pt"]
    assert emt["cli_reproducible"] is False
    assert data["devices"]["choices"] == ["auto", "cpu", "cuda"]


def test_all_families_listed_with_emt_last(everything_installed):
    data = calculators.list_calculators()
    names = [c["name"] for c in data["calculators"]]
    assert names[-1] == "emt"
    assert names[:3] == ["uma-s-1p2", "uma-s-1p1", "uma-m-1p1"]
    assert {"mace", "mace-mh-0", "mace-mh-1", "chgnet", "7net-omni", "7net-0"} <= set(names)
    assert data["default"] == "uma-s-1p2"
    by = {c["name"]: c for c in data["calculators"]}
    assert by["uma-s-1p2"]["task_required"] and by["uma-s-1p2"]["task_field"] == "uma_task"
    assert by["7net-0"]["task_required"] is False and by["7net-0"]["task_choices"] == []
    assert by["7net-mf-0"]["task_choices"] == ["PBE", "R2SCAN"]
    assert by["mace"]["task_field"] is None
    assert by["mace-mh-1"]["task_field"] == "mace_head"
    for entry in data["calculators"]:
        assert set(entry) == {"name", "label", "family", "task_field", "task_choices",
                              "task_required", "elements", "cli_reproducible", "note"}


def test_validate_calculator_rules(everything_installed, monkeypatch):
    norm, errs = calculators.validate_calculator({"name": "mace-mh-1", "mace_head": "omol"})
    assert errs == [] and norm["mace_head"] == "omol" and norm["device"] == "auto"
    _, errs = calculators.validate_calculator({"name": "mace", "mace_head": "omol"})
    assert errs[0]["field"] == "calculator.mace_head"
    _, errs = calculators.validate_calculator({"name": "uma-s-1p2", "uma_task": "OMAT"})
    assert errs[0]["field"] == "calculator.uma_task"
    _, errs = calculators.validate_calculator({"name": "uma-s-1p2", "uma_task": "omat",
                                               "sevennet_task": "mpa"})
    assert [e["field"] for e in errs] == ["calculator.sevennet_task"]
    _, errs = calculators.validate_calculator({"name": "mace", "colour": 1})
    assert errs[0]["field"] == "calculator.colour"
    _, errs = calculators.validate_calculator("mace")
    assert errs[0]["field"] == "calculator"
    monkeypatch.setattr(calculators, "_DEVICE_CACHE",
                        {"torch_installed": True, "cuda_available": False})
    _, errs = calculators.validate_calculator({"name": "mace", "device": "cuda"})
    assert errs[0]["field"] == "calculator.device"
    # EMT records the device only.
    _, errs = calculators.validate_calculator({"name": "emt", "device": "cuda"})
    assert errs == []


def test_uninstalled_known_tag_gets_install_hint(nothing_installed):
    _, errs = calculators.validate_calculator({"name": "mace"})
    assert errs[0]["field"] == "calculator.name"
    assert "install recipe" in errs[0]["message"]
    _, errs = calculators.validate_calculator({"name": "nonsense"})
    assert "unknown calculator" in errs[0]["message"]


def test_make_calculator_emt():
    assert isinstance(calculators.make_calculator({"name": "emt", "device": "cuda"}), EMT)
    assert calculators.resolved_device({"name": "emt", "device": "cuda"}) == "cpu"


def test_make_calculator_converts_typer_exit(monkeypatch):
    def refuse(*a, **k):
        raise typer.Exit("model refused")
    monkeypatch.setattr(cli_utils, "validate_mlip", refuse)
    with pytest.raises(RuntimeError, match="model refused"):
        calculators.make_calculator({"name": "mace", "device": "cpu"})

    def refuse_code(*a, **k):
        raise typer.Exit(code=1)
    monkeypatch.setattr(cli_utils, "validate_mlip", refuse_code)
    with pytest.raises(RuntimeError, match="exited with code 1"):
        calculators.make_calculator({"name": "mace", "device": "cpu"})


def test_make_calculator_unknown_tag(monkeypatch):
    monkeypatch.setattr(cli_utils, "validate_mlip", lambda *a, **k: None)
    monkeypatch.setattr(cli_utils, "build_calculator", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="unknown MLIP"):
        calculators.make_calculator({"name": "mace", "device": "cpu"})


def test_detect_devices_never_imports_torch_in_process():
    code = ("import sys\n"
            "from mliprun.webapp import calculators as c\n"
            "d = c.detect_devices(timeout=120)\n"
            "assert 'torch' not in sys.modules, 'torch imported'\n"
            "assert c.detect_devices() == d\n"
            "assert set(d) == {'torch_installed', 'cuda_available'}\n"
            "assert d['cuda_available'] in (True, False, None)\n"
            "c.list_calculators()\n"
            "assert 'torch' not in sys.modules\n"
            "print(d)\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=180)
    assert proc.returncode == 0, proc.stderr


def test_detect_devices_without_torch(monkeypatch):
    monkeypatch.setattr(calculators, "_DEVICE_CACHE", None)
    monkeypatch.setattr(calculators, "_torch_installed", lambda: False)
    assert calculators.device_status() == {"torch_installed": False,
                                           "cuda_available": False}
    assert calculators.detect_devices() == {"torch_installed": False,
                                            "cuda_available": False}
