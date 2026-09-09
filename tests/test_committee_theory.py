"""Level-of-theory resolution for committee members.

CANON C3 is load-bearing here: heads and tasks are independent fine-tunes
with independent energy zeros, so a committee spanning two levels reports
the level difference, not model error. The 2026-09-04 probe measured 0.363
eV across RPBE and PBE members against 0.001-0.019 eV within a level.
"""
import pytest

from mliprun.core.committee.config import (
    UNKNOWN_LEVEL,
    is_mixed_theory,
    resolve_level_of_theory,
)


class TestKnownLevels:
    @pytest.mark.parametrize(
        "mlip,kwargs,expected",
        [
            ("uma-s-1p2", {"uma_task": "oc20"}, "RPBE/OC20"),
            ("uma-s-1p2", {"uma_task": "oc22"}, "PBE+U/OC22"),
            ("uma-s-1p2", {"uma_task": "omat"}, "PBE/OMat24"),
            ("7net-omni", {"sevennet_task": "oc20"}, "RPBE/OC20"),
            ("7net-omni", {"sevennet_task": "omat24"}, "PBE/OMat24"),
            ("7net-mf-ompa", {"sevennet_task": "omat24"}, "PBE/OMat24"),
            ("7net-omat", {}, "PBE/OMat24"),
            ("mace-mh-1", {"mace_head": "oc20_usemppbe"}, "RPBE/OC20"),
            ("mace-mh-1", {"mace_head": "omat_pbe"}, "PBE/OMat24"),
            ("mace", {}, "PBE(+U)/MPtrj"),
            ("chgnet", {}, "PBE(+U)/MPtrj"),
        ],
        ids=["member_a", "member_b", "member_c", "member_d", "member_e",
             "member_f", "member_g", "member_h", "member_i", "member_j",
             "member_k"],
    )
    def test_table_lookups(self, mlip, kwargs, expected):
        assert resolve_level_of_theory(mlip, **kwargs) == expected

    def test_three_packages_agree_on_the_oc20_label(self):
        """The whole point: a genuinely cross-package same-level committee."""
        levels = {
            resolve_level_of_theory("uma-s-1p2", uma_task="oc20"),
            resolve_level_of_theory("7net-omni", sevennet_task="oc20"),
            resolve_level_of_theory("mace-mh-1", mace_head="oc20_usemppbe"),
        }
        assert levels == {"RPBE/OC20"}
        assert is_mixed_theory(levels) is False


class TestUnknownLevels:
    def test_an_unregistered_tag_is_unknown(self):
        """cli/utils deliberately forwards unknown uma-*/7net-* tags to their
        packages, so this path is reachable in production."""
        assert resolve_level_of_theory("uma-x-9p9",
                                       uma_task="oc20") == "RPBE/OC20"
        assert resolve_level_of_theory("not-a-model") == UNKNOWN_LEVEL

    def test_an_unregistered_task_is_unknown(self):
        assert resolve_level_of_theory(
            "uma-s-1p2", uma_task="brand-new-head") == UNKNOWN_LEVEL

    def test_a_missing_task_is_unknown(self):
        assert resolve_level_of_theory("7net-omni") == UNKNOWN_LEVEL

    def test_the_reserved_emt_tag_is_unknown(self):
        assert resolve_level_of_theory("emt") == UNKNOWN_LEVEL

    def test_task_matching_is_case_sensitive(self):
        """7net-mf-0 names its tasks in UPPERCASE where every other model uses
        lowercase, so case-folding here would resolve a task no checkpoint
        has."""
        assert resolve_level_of_theory(
            "uma-s-1p2", uma_task="OC20") == UNKNOWN_LEVEL


class TestMixedDetection:
    def test_one_level_is_not_mixed(self):
        assert is_mixed_theory(["RPBE/OC20", "RPBE/OC20"]) is False

    def test_two_levels_are_mixed(self):
        assert is_mixed_theory(["RPBE/OC20", "PBE(+U)/MPtrj"]) is True

    def test_any_unknown_counts_as_mixed(self):
        """Unknown is *possibly* mixed: it warns rather than passing
        silently."""
        assert is_mixed_theory(["RPBE/OC20", UNKNOWN_LEVEL]) is True
        assert is_mixed_theory([UNKNOWN_LEVEL, UNKNOWN_LEVEL]) is True

    def test_an_empty_set_is_not_mixed(self):
        assert is_mixed_theory([]) is False


class TestMPtrjModelsShareOneLevel:
    def test_mace_mp_0_and_chgnet_are_the_same_level(self):
        """Both are trained on MPtrj. Their disagreement is architectural at
        a fixed level of theory, which is exactly what a committee measures.
        Juan's ruling, 2026-09-08."""
        assert (resolve_level_of_theory("mace")
                == resolve_level_of_theory("chgnet"))

    def test_the_label_names_the_mixing_not_one_functional(self):
        """MPtrj applies +U to transition-metal oxides and fluorides only, so
        the effective functional depends on the system. A label asserting
        plain PBE or plain PBE+U is wrong for half the compositions."""
        assert resolve_level_of_theory("mace") == "PBE(+U)/MPtrj"

    def test_an_mptrj_committee_is_not_mixed_theory(self):
        assert is_mixed_theory([resolve_level_of_theory("mace"),
                                resolve_level_of_theory("chgnet")]) is False

    def test_omat24_is_still_a_different_level(self):
        """The ruling merges MPtrj labels only; it must not collapse
        genuinely different datasets."""
        assert is_mixed_theory([resolve_level_of_theory("7net-omat"),
                                resolve_level_of_theory("mace")]) is True

    def test_the_unverified_mptrj_sevennet_tags_stay_unknown(self):
        """7net-0 and 7net-l3i5 are believed to be MPtrj too, but that is not
        confirmed from the checkpoints, so they must not be quietly merged."""
        assert resolve_level_of_theory("7net-0") == UNKNOWN_LEVEL
        assert resolve_level_of_theory("7net-l3i5") == UNKNOWN_LEVEL
