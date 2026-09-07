"""Parsing and validating committee.yaml."""
import hashlib
import os
import sys
import textwrap
from pathlib import Path

import pytest

from mliprun.core.committee.config import (
    LEVEL_TABLE_VERSION,
    CommitteeConfigError,
    load_committee,
    python_for_env,
)


@pytest.fixture
def fake_env(tmp_path):
    """A directory that looks like a venv: <env>/bin/python exists."""
    def _make(name):
        env = tmp_path / name
        (env / "bin").mkdir(parents=True)
        (env / "bin" / "python").write_text("#!/bin/sh\n")
        return env
    return _make


def _write(tmp_path, text):
    path = tmp_path / "committee.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


class TestValidFile:
    def test_a_three_member_committee_parses(self, tmp_path, fake_env):
        a, b, c = fake_env("uma"), fake_env("sevenn"), fake_env("mace")
        path = _write(tmp_path, f"""
            members:
              - env:  {a}
                mlip: uma-s-1p2
                uma_task: oc20
                gpu: 0
              - env:  {b}
                mlip: 7net-omni
                sevennet_task: oc20
                gpu: 1
              - env:  {c}
                mlip: mace-mh-1
                mace_head: oc20_usemppbe
                gpu: 2
        """)
        config = load_committee(path)

        assert len(config.members) == 3
        assert [m.mlip for m in config.members] == ["uma-s-1p2", "7net-omni",
                                                    "mace-mh-1"]
        assert [m.gpu for m in config.members] == [0, 1, 2]
        assert config.levels == ("RPBE/OC20",)
        assert config.mixed_theory is False
        assert config.members[0].python_exe == str(a / "bin" / "python")

    def test_default_names_pair_the_tag_with_its_task(self, tmp_path,
                                                      fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: uma-s-1p2, uma_task: oc20}}
              - {{env: {b}, mlip: chgnet}}
        """)
        config = load_committee(path)
        assert [m.name for m in config.members] == ["uma-s-1p2@oc20", "chgnet"]

    def test_an_explicit_name_wins(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: uma-s-1p2, uma_task: oc20, name: reference}}
              - {{env: {b}, mlip: chgnet}}
        """)
        assert load_committee(path).members[0].name == "reference"

    def test_repeated_tag_and_task_get_distinct_names(self, tmp_path,
                                                      fake_env):
        """Two members can legitimately differ only by env (two builds of the
        same model), and the CSV needs one column per member."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet}}
              - {{env: {b}, mlip: chgnet}}
        """)
        names = [m.name for m in load_committee(path).members]
        assert len(set(names)) == 2
        assert names[0] == "chgnet"

    def test_sha256_matches_the_file_bytes(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
        """)
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert load_committee(path).sha256 == expected

    def test_env_paths_expand_user_and_variables(self, tmp_path, fake_env,
                                                 monkeypatch):
        a, b = fake_env("a"), fake_env("b")
        monkeypatch.setenv("MLIPRUN_TEST_ROOT", str(a))
        # A genuine `~` case: .expanduser() is a separate code path from
        # $VAR expansion and needs its own coverage, or it could be deleted
        # without any test noticing.
        monkeypatch.setenv("HOME", str(tmp_path))
        home_env = fake_env("home_env")
        path = _write(tmp_path, f"""
            members:
              - {{env: "$MLIPRUN_TEST_ROOT", mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
              - {{env: "~/home_env", mlip: chgnet}}
        """)
        config = load_committee(path)
        assert config.members[0].env == str(a)
        assert config.members[2].env == str(home_env)

    def test_provenance_block_carries_every_member(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
        """)
        provenance = load_committee(path).as_provenance()
        assert len(provenance["members"]) == 2
        assert provenance["members"][0]["mlip"] == "chgnet"
        assert provenance["members"][0]["level_of_theory"] == "PBE+U/MPtrj"
        assert provenance["mixed_theory"] is True   # PBE+U/MPtrj vs PBE/MPtrj
        assert sorted(provenance["levels"]) == ["PBE+U/MPtrj", "PBE/MPtrj"]
        assert provenance["config_sha256"] == load_committee(path).sha256


class TestMixedTheoryFlag:
    def test_two_levels_set_the_flag(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: uma-s-1p2, uma_task: oc20}}
              - {{env: {b}, mlip: mace}}
        """)
        config = load_committee(path)
        assert config.mixed_theory is True
        assert sorted(config.levels) == ["PBE/MPtrj", "RPBE/OC20"]

    def test_an_unknown_task_sets_the_flag(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: uma-s-1p2, uma_task: oc20}}
              - {{env: {b}, mlip: 7net-omni, sevennet_task: pet_mad}}
        """)
        assert load_committee(path).mixed_theory is True


class TestRejections:
    def test_a_missing_file(self, tmp_path):
        with pytest.raises(CommitteeConfigError, match="not found"):
            load_committee(tmp_path / "absent.yaml")

    def test_malformed_yaml(self, tmp_path):
        path = _write(tmp_path, "members: [unclosed\n")
        with pytest.raises(CommitteeConfigError, match="could not parse"):
            load_committee(path)

    def test_an_unreadable_file(self, tmp_path, fake_env):
        """is_file() only proves the path is a regular file. A
        permission-denied committee.yaml must not propagate a raw OSError
        past load_committee -- the CLI catches only CommitteeConfigError."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
        """)
        os.chmod(path, 0o000)
        if os.geteuid() == 0:
            pytest.skip("root ignores file permissions; chmod cannot deny "
                        "the owner")
        try:
            with pytest.raises(CommitteeConfigError, match="could not read"):
                load_committee(path)
        finally:
            os.chmod(path, 0o644)

    def test_a_top_level_list(self, tmp_path):
        path = _write(tmp_path, "- one\n- two\n")
        with pytest.raises(CommitteeConfigError, match="mapping"):
            load_committee(path)

    def test_a_missing_members_key(self, tmp_path):
        path = _write(tmp_path, "committee: []\n")
        with pytest.raises(CommitteeConfigError, match="members"):
            load_committee(path)

    def test_a_single_member(self, tmp_path, fake_env):
        a = fake_env("a")
        path = _write(tmp_path, f"members:\n  - {{env: {a}, mlip: chgnet}}\n")
        with pytest.raises(CommitteeConfigError, match="at least two"):
            load_committee(path)

    def test_an_unknown_member_key(self, tmp_path, fake_env):
        """Typo protection: a silently ignored 'gpus:' would send every
        member to the same device."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet, gpus: 0}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match="gpus"):
            load_committee(path)

    def test_a_missing_env(self, tmp_path, fake_env):
        b = fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match="env"):
            load_committee(path)

    def test_a_missing_mlip(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match="mlip"):
            load_committee(path)

    def test_an_env_without_an_interpreter(self, tmp_path, fake_env):
        b = fake_env("b")
        empty = tmp_path / "empty"
        empty.mkdir()
        path = _write(tmp_path, f"""
            members:
              - {{env: {empty}, mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match=r"member 0.*interpreter"):
            load_committee(path)

    def test_duplicate_explicit_names(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet, name: same}}
              - {{env: {b}, mlip: mace, name: same}}
        """)
        with pytest.raises(CommitteeConfigError, match="duplicate"):
            load_committee(path)

    def test_a_name_with_unsafe_characters(self, tmp_path, fake_env):
        """Names become CSV headers and log filenames."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet, name: "../escape"}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match="name"):
            load_committee(path)

    def test_a_non_integer_gpu(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet, gpu: "first"}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match="gpu"):
            load_committee(path)

    def test_a_boolean_gpu(self, tmp_path, fake_env):
        """bool is a subclass of int: `gpu: true` must not silently become
        device index 1."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet, gpu: true}}
              - {{env: {b}, mlip: mace}}
        """)
        with pytest.raises(CommitteeConfigError, match="gpu"):
            load_committee(path)


class TestMixedTheoryWarning:
    def test_a_same_level_committee_has_no_warning(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet}}
              - {{env: {b}, mlip: chgnet}}
        """)
        assert load_committee(path).mixed_theory_warning() == ""

    def test_a_mixed_level_committee_names_every_member(self, tmp_path,
                                                         fake_env):
        """Assert on the member names and resolved levels, not the prose --
        the surrounding wording may be reworded later."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
        """)
        config = load_committee(path)
        warning = config.mixed_theory_warning()
        assert warning != ""
        for member in config.members:
            assert member.name in warning
            assert member.level_of_theory in warning


class TestHeadAndTaskRules:
    """committee.yaml obeys the same head/task rules `mlip optimize run` does.

    Each rejection below is one the CLI's ``validate_mlip`` already makes.
    Before this, a committee member went straight to ``build_calculator``,
    which guards only a MISSING head or task on the multi-head tags, so the
    combinations here ran silently with the wrong model while the member's
    name, its CSV column header and its provenance block all described the
    head or task that was ignored.

    Only the head/task half of ``validate_mlip`` applies. Its availability
    half is deliberately NOT used: it asks whether the package is importable
    in the DRIVER env, which for a committee is the wrong env by design.
    """

    def test_a_head_on_single_head_mace_is_rejected(self, tmp_path, fake_env):
        """`mace` is MACE-MP-0 medium: the head is ignored by the calculator,
        so accepting it names the member, its CSV column and its provenance
        after an RPBE/OC20 head that never ran."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet}}
              - {{env: {b}, mlip: mace, mace_head: oc20_usemppbe}}
        """)
        with pytest.raises(CommitteeConfigError) as excinfo:
            load_committee(path)
        assert "member 1" in str(excinfo.value)
        assert "single-head" in str(excinfo.value)
        assert "oc20_usemppbe" in str(excinfo.value)

    def test_a_task_on_a_single_task_sevennet_tag_is_rejected(self, tmp_path,
                                                              fake_env):
        """`7net-omat` resolves its level from the TAG (PBE/OMat24) while the
        calculator would be handed modal='oc20': a well-formed file yielding a
        wrong level-of-theory label."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet}}
              - {{env: {b}, mlip: 7net-omat, sevennet_task: oc20}}
        """)
        with pytest.raises(CommitteeConfigError) as excinfo:
            load_committee(path)
        assert "member 1" in str(excinfo.value)
        assert "single-task" in str(excinfo.value)

    def test_an_unknown_head_on_a_multi_head_tag_is_rejected(self, tmp_path,
                                                             fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: mace-mh-1, mace_head: not_a_head}}
              - {{env: {b}, mlip: chgnet}}
        """)
        with pytest.raises(CommitteeConfigError) as excinfo:
            load_committee(path)
        assert "member 0" in str(excinfo.value)
        assert "unknown mace_head" in str(excinfo.value)

    def test_an_unknown_task_on_a_verified_uma_tag_is_rejected(self, tmp_path,
                                                               fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: uma-s-1p2, uma_task: not_a_task}}
              - {{env: {b}, mlip: chgnet}}
        """)
        with pytest.raises(CommitteeConfigError) as excinfo:
            load_committee(path)
        assert "member 0" in str(excinfo.value)
        assert "unknown uma_task" in str(excinfo.value)

    def test_a_missing_uma_task_is_rejected(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: uma-s-1p2}}
              - {{env: {b}, mlip: chgnet}}
        """)
        with pytest.raises(CommitteeConfigError, match="uma_task is required"):
            load_committee(path)

    def test_a_missing_mace_head_is_rejected(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: mace-mh-1}}
              - {{env: {b}, mlip: chgnet}}
        """)
        with pytest.raises(CommitteeConfigError,
                           match="mace_head is required"):
            load_committee(path)

    def test_a_missing_sevennet_task_is_rejected(self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: 7net-omni}}
              - {{env: {b}, mlip: chgnet}}
        """)
        with pytest.raises(CommitteeConfigError,
                           match="sevennet_task is required"):
            load_committee(path)

    def test_an_unknown_sevennet_task_is_rejected(self, tmp_path, fake_env):
        """SevenNet task names are matched exactly: 7net-mf-0 spells its tasks
        in uppercase where every other model uses lowercase, so a case-folded
        match would send the checkpoint a task it does not have."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: 7net-mf-0, sevennet_task: pbe}}
              - {{env: {b}, mlip: chgnet}}
        """)
        with pytest.raises(CommitteeConfigError) as excinfo:
            load_committee(path)
        assert "unknown sevennet_task" in str(excinfo.value)
        assert "PBE" in str(excinfo.value)

    def test_a_head_or_task_on_the_reserved_emt_tag_is_rejected(self, tmp_path,
                                                                fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: emt, uma_task: omat}}
              - {{env: {b}, mlip: emt}}
        """)
        with pytest.raises(CommitteeConfigError) as excinfo:
            load_committee(path)
        assert "member 0" in str(excinfo.value)
        assert "EMT" in str(excinfo.value)

    def test_the_reserved_emt_tag_still_parses_on_its_own(self, tmp_path,
                                                          fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: emt}}
              - {{env: {b}, mlip: emt}}
        """)
        config = load_committee(path)
        assert [m.mlip for m in config.members] == ["emt", "emt"]

    def test_an_unregistered_uma_tag_forwards_its_task_unchecked(
            self, tmp_path, fake_env):
        """A newer checkpoint may carry heads this table has not seen;
        rejecting a valid one would be worse than not checking it. Nothing is
        silent about it: the level resolves to 'unknown', which flags the
        committee as mixed."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: uma-x-9p9, uma_task: brand_new}}
              - {{env: {b}, mlip: chgnet}}
        """)
        config = load_committee(path)
        assert config.members[0].uma_task == "brand_new"
        assert config.members[0].level_of_theory == "unknown"
        assert config.mixed_theory is True

    def test_an_unregistered_sevennet_tag_forwards_its_task_unchecked(
            self, tmp_path, fake_env):
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: 7net-brand-new, sevennet_task: whatever}}
              - {{env: {b}, mlip: chgnet}}
        """)
        config = load_committee(path)
        assert config.members[0].sevennet_task == "whatever"
        assert config.members[0].level_of_theory == "unknown"

    def test_the_shipped_example_file_is_still_valid(self):
        """examples/committee.yaml is the file users copy. Its head/task
        combinations must survive the rule this class adds. Checked through
        the validator directly rather than load_committee, because the
        example's env paths are cos-cluster paths that do not exist here."""
        import yaml

        from mliprun.core.committee.config import _validate_head_task

        example = (Path(__file__).resolve().parents[1] / "examples"
                   / "committee.yaml")
        entries = yaml.safe_load(example.read_text(encoding="utf-8"))["members"]
        assert len(entries) == 3
        for index, entry in enumerate(entries):
            _validate_head_task(index, entry["mlip"], entry.get("uma_task"),
                                entry.get("mace_head"),
                                entry.get("sevennet_task"))


class TestLevelTableVersion:
    def test_the_table_version_reaches_provenance(self, tmp_path, fake_env):
        """A wrong ROW in the level-of-theory table fails silently: two
        entries with the same label for different datasets read as same-level
        with no signal anywhere. The stamped version is what makes a record
        written under a later-corrected table re-judgeable."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet}}
              - {{env: {b}, mlip: mace}}
        """)
        provenance = load_committee(path).as_provenance()
        assert provenance["level_table_version"] == LEVEL_TABLE_VERSION
        assert isinstance(LEVEL_TABLE_VERSION, int)
        assert LEVEL_TABLE_VERSION >= 1

    def test_measured_versions_merge_in_by_member_name(self, tmp_path,
                                                       fake_env):
        """`as_provenance` echoes what the YAML declared; the measured block
        is what each member's own env reported back. A member with no measured
        entry records null, not an empty dict, so "never started" stays
        distinguishable from "started and reported nothing"."""
        a, b = fake_env("a"), fake_env("b")
        path = _write(tmp_path, f"""
            members:
              - {{env: {a}, mlip: chgnet, name: first}}
              - {{env: {b}, mlip: mace, name: second}}
        """)
        config = load_committee(path)
        provenance = config.as_provenance(measured_versions={
            "first": {"ase": "3.29.0", "torch": "2.6.0",
                      "package": "chgnet", "package_version": "0.4.0"},
        })
        first, second = provenance["members"]
        assert first["measured"]["torch"] == "2.6.0"
        assert first["measured"]["package_version"] == "0.4.0"
        assert first["mlip"] == "chgnet"        # declared, unchanged
        assert second["measured"] is None


class TestPythonForEnv:
    def test_the_current_environment_resolves(self):
        env = Path(sys.executable).parents[1]
        assert python_for_env(env).exists()

    def test_python3_is_accepted_when_python_is_absent(self, tmp_path):
        env = tmp_path / "env"
        (env / "bin").mkdir(parents=True)
        (env / "bin" / "python3").write_text("#!/bin/sh\n")
        assert python_for_env(env).name == "python3"

    def test_a_missing_env_is_rejected(self, tmp_path):
        with pytest.raises(CommitteeConfigError, match="interpreter"):
            python_for_env(tmp_path / "nope")
