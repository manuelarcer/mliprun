"""Tests for `mlip doctor` — the environment self-check command.

Contract: exit code 0 iff at least one MLIP package is installed; exit
code 1 otherwise, with a pointer at the install recipes. asetools is an
optional extra (`.[neb]`, enables the NEB interpolation sanity check), so
its absence is a hint, not a failure — but the PyPI name-collision impostor
("asetools" on PyPI is unrelated Aseprite tooling) is loudly flagged with
the recovery command.

Package availability is monkeypatched through the `_package_version` and
`_asetools_status` seams so the tests are independent of which MLIPs happen
to be installed in the test environment.
"""
import subprocess
import sys
import types

from typer.testing import CliRunner

from mliprun.cli.commands import doctor as doctor_cmd
from mliprun.cli.main import app as main_app

runner = CliRunner()

# Core (always-installed) distributions the doctor reports on. Anything
# outside this set is treated as not installed in the mocked environment.
_CORE_VERSIONS = {
    "mliprun": "0.3.0",
    "ase": "3.29.0",
}


def _fake_versions(extra):
    """Build a `_package_version` replacement: core deps + `extra` dists."""
    table = dict(_CORE_VERSIONS, **extra)

    def _package_version(dist_name):
        return table.get(dist_name)

    return _package_version


class TestDoctorHelp:
    def test_help_exits_cleanly(self):
        # invoke via the running interpreter, not the `mlip` script on PATH —
        # PATH may hold an older install that predates `doctor`
        result = subprocess.run(
            [sys.executable, "-m", "mliprun.cli.main", "doctor", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Usage" in result.stdout or "Options" in result.stdout


class TestDoctorUsableEnvironment:
    def test_real_asetools_plus_mace_exits_zero(self, monkeypatch):
        monkeypatch.setattr(
            doctor_cmd, "_package_version",
            _fake_versions({"mace-torch": "0.3.9"}),
        )
        monkeypatch.setattr(doctor_cmd, "_asetools_status", lambda: "ok")

        result = runner.invoke(main_app, ["doctor"])

        assert result.exit_code == 0
        assert "mace-torch" in result.output
        assert "0.3.9" in result.output

    def test_auto_detection_order_uma_wins_over_mace(self, monkeypatch):
        monkeypatch.setattr(
            doctor_cmd, "_package_version",
            _fake_versions({"fairchem-core": "2.0.0", "mace-torch": "0.3.9"}),
        )
        monkeypatch.setattr(doctor_cmd, "_asetools_status", lambda: "ok")

        result = runner.invoke(main_app, ["doctor"])

        assert result.exit_code == 0
        # detect_mlip prefers UMA over MACE; doctor must report the same tag
        assert "uma-s-1p2" in result.output

    def test_two_mlips_in_one_env_warns_but_exits_zero(self, monkeypatch):
        monkeypatch.setattr(
            doctor_cmd, "_package_version",
            _fake_versions({"fairchem-core": "2.0.0", "mace-torch": "0.3.9"}),
        )
        monkeypatch.setattr(doctor_cmd, "_asetools_status", lambda: "ok")

        result = runner.invoke(main_app, ["doctor"])

        assert result.exit_code == 0
        # ADR 0001: one env per MLIP; two packages in one env is a warning
        assert "WARN" in result.output


class TestDoctorBrokenEnvironment:
    def test_no_mlip_installed_exits_one_with_recipe_pointer(self, monkeypatch):
        monkeypatch.setattr(
            doctor_cmd, "_package_version", _fake_versions({}),
        )
        monkeypatch.setattr(doctor_cmd, "_asetools_status", lambda: "ok")

        result = runner.invoke(main_app, ["doctor"])

        assert result.exit_code == 1
        assert "docs/install" in result.output

    def test_wrong_asetools_warns_with_recovery_command_but_exits_zero(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            doctor_cmd, "_package_version",
            _fake_versions({"mace-torch": "0.3.9"}),
        )
        monkeypatch.setattr(
            doctor_cmd, "_asetools_status", lambda: "wrong-package"
        )

        result = runner.invoke(main_app, ["doctor"])

        # the impostor doesn't stop MLIP runs — usable env, loud warning
        assert result.exit_code == 0
        assert "WARN" in result.output
        assert "pip uninstall asetools" in result.output

    def test_missing_asetools_hints_at_neb_extra_and_exits_zero(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            doctor_cmd, "_package_version",
            _fake_versions({"mace-torch": "0.3.9"}),
        )
        monkeypatch.setattr(doctor_cmd, "_asetools_status", lambda: "missing")

        result = runner.invoke(main_app, ["doctor"])

        assert result.exit_code == 0
        assert '.[neb]' in result.output


class TestDoctorRealEnvironment:
    """Unmocked runs — exercise the real probe helpers on whatever env runs
    the tests (CI has the real asetools but no MLIP; a user env may have
    both). Assert invariants, not env-specific facts."""

    def test_real_run_exit_code_is_zero_or_one(self):
        result = runner.invoke(main_app, ["doctor"])
        assert result.exit_code in (0, 1)
        assert "mliprun" in result.output
        assert "asetools" in result.output

    def test_asetools_status_returns_known_value(self):
        assert doctor_cmd._asetools_status() in ("ok", "wrong-package", "missing")

    def test_package_version_known_and_unknown(self):
        # mliprun is installed in every test env (editable install)
        assert doctor_cmd._package_version("mliprun") is not None
        assert doctor_cmd._package_version("no-such-distribution-xyz") is None

    def test_torch_info_returns_version_and_cuda_fields(self):
        version, cuda = doctor_cmd._torch_info()
        assert version is None or isinstance(version, str)
        assert isinstance(cuda, str)


class TestTorchInfoBranches:
    """CUDA branches via a stub torch — CI has no torch installed, so these
    paths would otherwise never execute."""

    def _stub_torch(self, monkeypatch, cuda_available, device_count=0):
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: cuda_available,
            device_count=lambda: device_count,
        )
        monkeypatch.setitem(
            sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda)
        )
        monkeypatch.setattr(
            doctor_cmd, "_package_version", _fake_versions({"torch": "2.5.0"})
        )

    def test_cuda_available_reports_device_count(self, monkeypatch):
        self._stub_torch(monkeypatch, cuda_available=True, device_count=2)
        version, cuda = doctor_cmd._torch_info()
        assert version == "2.5.0"
        assert "2 device(s)" in cuda

    def test_cpu_only_torch_reported(self, monkeypatch):
        self._stub_torch(monkeypatch, cuda_available=False)
        version, cuda = doctor_cmd._torch_info()
        assert version == "2.5.0"
        assert "CPU only" in cuda


class TestDoctorAgreesWithDetectMlip:
    """doctor keeps its own (distribution, tag) table for the `--mlip auto`
    line. It drifted once already: after auto-detect moved to 7net-omni,
    doctor still reported 7net-mf-ompa, so the diagnostic tool contradicted
    the tool it diagnoses. This test ties the two together."""

    _FLAGS = ("FAIRCHEM_AVAILABLE", "MACE_AVAILABLE", "SEVENN_AVAILABLE",
              "CHGNET_AVAILABLE", "AQCAT_AVAILABLE")

    def test_every_doctor_tag_matches_what_detect_mlip_returns(self):
        # Keyed by availability flag, not distribution: fairchem-core serves
        # two families (2.x UMA, 1.x AQCat25-EV2).
        from unittest.mock import patch
        import pytest
        import typer
        from mliprun.cli.commands.doctor import _MLIP_PACKAGES
        from mliprun.cli.utils import detect_mlip

        for label, dist, flag, tag in _MLIP_PACKAGES:
            patches = {name: (name == flag) for name in self._FLAGS}
            with patch.multiple("mliprun.cli.utils", **patches):
                if tag is None:
                    # Explicit-only family: auto must refuse, naming why.
                    with pytest.raises(typer.Exit) as exc:
                        detect_mlip()
                    assert "--aqcat-spin" in str(exc.value.exit_code)
                    continue
                assert detect_mlip() == tag, (
                    f"doctor reports '{tag}' for a {label}-only env, but "
                    f"detect_mlip() returns '{detect_mlip()}'"
                )

    def test_doctor_table_covers_every_availability_flag(self):
        from mliprun.cli.commands.doctor import _MLIP_PACKAGES
        assert {flag for _, _, flag, _ in _MLIP_PACKAGES} == set(self._FLAGS)
        assert {d for _, d, _, _ in _MLIP_PACKAGES} == {
            "fairchem-core", "mace-torch", "sevenn", "chgnet"}


class TestDoctorSharedFairchemDistribution:
    """fairchem-core 2.x is UMA; 1.x is the AQCat25-EV2 env."""

    def _run(self, monkeypatch, extra):
        monkeypatch.setattr(doctor_cmd, "_package_version",
                            _fake_versions(extra))
        monkeypatch.setattr(doctor_cmd, "_asetools_status", lambda: "ok")
        return runner.invoke(main_app, ["doctor"])

    def test_aqcat_only_env_is_usable_and_auto_is_explained(self, monkeypatch):
        result = self._run(monkeypatch, {"fairchem-core": "1.10.0"})
        assert result.exit_code == 0, result.output
        assert "AQCat25-EV2" in result.output
        assert "1.10.0" in result.output
        assert "--mlip auto does not resolve" in result.output
        assert "aqcat25-ev2-midfilm" in result.output
        assert "aqcat25-ev2-infilm" in result.output
        assert "--aqcat-spin" in result.output
        assert "resolves to: uma-s-1p2" not in result.output
        assert "WARN: multiple" not in result.output

    def test_fairchem_2_is_reported_as_uma_only(self, monkeypatch):
        result = self._run(monkeypatch, {"fairchem-core": "2.19.0"})
        assert result.exit_code == 0
        assert "--mlip auto resolves to: uma-s-1p2" in result.output
        assert "--mlip auto does not resolve" not in result.output

    def test_aqcat_next_to_mace_resolves_to_mace_and_warns(self, monkeypatch):
        result = self._run(monkeypatch, {"fairchem-core": "1.10.0",
                                         "mace-torch": "0.3.15"})
        assert result.exit_code == 0
        assert "--mlip auto resolves to: mace" in result.output
        assert "WARN: multiple" in result.output
        assert "aqcat25-ev2-midfilm" in result.output

    def test_unparseable_fairchem_version_counts_as_neither(self, monkeypatch):
        result = self._run(monkeypatch, {"fairchem-core": "unknown"})
        assert result.exit_code == 1
        assert "unknown" in result.output
