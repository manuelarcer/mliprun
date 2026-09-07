"""Parsing and validating committee.yaml."""
import hashlib
import os
import sys
import textwrap
from pathlib import Path

import pytest

from mliprun.core.committee.config import (
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
