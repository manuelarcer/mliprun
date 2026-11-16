"""Tests for mliprun.cli.utils."""
import pytest
from unittest.mock import patch

import typer

from mliprun.cli.utils import (
    _SEVENNET_MODELS,
    build_calculator,
    detect_mlip,
    validate_mlip,
    resolve_mlip,
    parse_relax_atoms,
    FAIRCHEM_AVAILABLE,
    SEVENN_AVAILABLE,
    MACE_AVAILABLE,
    CHGNET_AVAILABLE,
)


class TestDetectMlip:
    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", True)
    def test_returns_string(self):
        # With a backend available, detect_mlip returns a non-empty model tag.
        # Mocked so the result does not depend on what is installed in the env.
        result = detect_mlip()
        assert isinstance(result, str)
        assert result

    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", True)
    def test_prefers_uma(self):
        assert detect_mlip() == "uma-s-1p2"

    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", False)
    @patch("mliprun.cli.utils.MACE_AVAILABLE", True)
    def test_falls_back_to_mace(self):
        # Without UMA, MACE is the preferred fallback (it is not gated).
        assert detect_mlip() == "mace"

    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", False)
    @patch("mliprun.cli.utils.MACE_AVAILABLE", True)
    @patch("mliprun.cli.utils.SEVENN_AVAILABLE", True)
    def test_mace_preferred_over_sevenn(self):
        # When both MACE and SevenNet are installed, MACE wins (readily usable).
        assert detect_mlip() == "mace"

    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", False)
    @patch("mliprun.cli.utils.MACE_AVAILABLE", False)
    @patch("mliprun.cli.utils.SEVENN_AVAILABLE", True)
    def test_falls_back_to_sevenn(self):
        # 7net-omni is SevenNet's recommended model and the only one of its
        # family with surface heads (oc20/oc22). It is multi-task, so `--mlip
        # auto` in a SevenNet-only env resolves here and then stops for a
        # missing --sevennet-task rather than guessing a task (CANON C1).
        assert detect_mlip() == "7net-omni"

    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", False)
    @patch("mliprun.cli.utils.SEVENN_AVAILABLE", False)
    @patch("mliprun.cli.utils.MACE_AVAILABLE", False)
    @patch("mliprun.cli.utils.CHGNET_AVAILABLE", True)
    def test_falls_back_to_chgnet(self):
        assert detect_mlip() == "chgnet"

    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", False)
    @patch("mliprun.cli.utils.SEVENN_AVAILABLE", False)
    @patch("mliprun.cli.utils.MACE_AVAILABLE", False)
    @patch("mliprun.cli.utils.CHGNET_AVAILABLE", False)
    def test_none_available_raises(self):
        with pytest.raises(typer.Exit):
            detect_mlip()


class TestValidateMlip:
    def test_auto_passes(self):
        validate_mlip("auto")  # should not raise

    def test_unknown_mlip_raises(self):
        with pytest.raises(typer.Exit):
            validate_mlip("nonexistent-model")

    @patch("mliprun.cli.utils.MACE_AVAILABLE", False)
    def test_mace_unavailable_raises(self):
        with pytest.raises(typer.Exit):
            validate_mlip("mace")

    @patch("mliprun.cli.utils.SEVENN_AVAILABLE", False)
    def test_sevenn_unavailable_raises(self):
        with pytest.raises(typer.Exit):
            validate_mlip("7net-mf-ompa")

    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", False)
    def test_uma_unavailable_raises(self):
        with pytest.raises(typer.Exit):
            validate_mlip("uma-s-1p1")

    @patch("mliprun.cli.utils.CHGNET_AVAILABLE", False)
    def test_chgnet_unavailable_raises(self):
        with pytest.raises(typer.Exit):
            validate_mlip("chgnet")

    @patch("mliprun.cli.utils.CHGNET_AVAILABLE", True)
    def test_chgnet_available_passes(self):
        validate_mlip("chgnet")  # should not raise

    @patch("mliprun.cli.utils.MACE_AVAILABLE", True)
    def test_mace_model_file_passes(self, tmp_path):
        model = tmp_path / "MACE_model.model"
        model.write_bytes(b"")
        validate_mlip(str(model))  # should not raise

    @patch("mliprun.cli.utils.MACE_AVAILABLE", True)
    def test_missing_mace_model_file_raises(self, tmp_path):
        with pytest.raises(typer.Exit):
            validate_mlip(str(tmp_path / "absent.model"))

    @patch("mliprun.cli.utils.MACE_AVAILABLE", True)
    def test_mace_model_file_rejects_head(self, tmp_path):
        model = tmp_path / "MACE_model.model"
        model.write_bytes(b"")
        with pytest.raises(typer.Exit):
            validate_mlip(str(model), mace_head="omat_pbe")

    @patch("mliprun.cli.utils.MACE_AVAILABLE", False)
    def test_mace_model_file_without_mace_raises(self, tmp_path):
        model = tmp_path / "MACE_model.model"
        model.write_bytes(b"")
        with pytest.raises(typer.Exit):
            validate_mlip(str(model))


class TestResolveMlip:
    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", True)
    def test_auto_resolves(self):
        # UMA is multi-head, so auto-detection now needs the task too.
        result = resolve_mlip("auto", uma_task="omat")
        assert isinstance(result, str)
        assert result != "auto"

    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", True)
    def test_explicit_passes_through(self):
        result = resolve_mlip("uma-s-1p1", uma_task="omat")
        assert result == "uma-s-1p1"

    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", True)
    def test_auto_still_enforces_the_head(self):
        # Auto-detecting UMA must not become a way to skip the head choice.
        with pytest.raises(typer.Exit):
            resolve_mlip("auto")


class TestParseRelaxAtoms:
    def test_valid_input(self):
        result = parse_relax_atoms("0,1,5", num_atoms=10)
        assert result == [0, 1, 5]

    def test_single_atom(self):
        result = parse_relax_atoms("3", num_atoms=10)
        assert result == [3]

    def test_with_spaces(self):
        result = parse_relax_atoms("0, 1, 5", num_atoms=10)
        assert result == [0, 1, 5]

    def test_invalid_format_raises(self):
        with pytest.raises(typer.Exit):
            parse_relax_atoms("a,b,c", num_atoms=10)

    def test_out_of_range_raises(self):
        with pytest.raises(typer.Exit):
            parse_relax_atoms("0,1,100", num_atoms=10)

    def test_negative_index_raises(self):
        with pytest.raises(typer.Exit):
            parse_relax_atoms("-1,0,1", num_atoms=10)


class TestParamSourcesFromCtx:
    """Click records where each parameter value came from; we relabel it."""

    def test_maps_click_sources_to_record_vocabulary(self):
        import typer
        from typer.testing import CliRunner

        from mliprun.cli.utils import param_sources_from_ctx

        seen = {}
        app = typer.Typer()

        @app.command()
        def go(ctx: typer.Context,
               fmax: float = typer.Option(0.05),
               max_steps: int = typer.Option(200)):
            seen.update(param_sources_from_ctx(ctx))

        result = CliRunner().invoke(app, ["--fmax", "0.02"])
        assert result.exit_code == 0, result.output
        assert seen["fmax"] == "user"
        assert seen["max_steps"] == "default"

    def test_returns_empty_dict_for_none(self):
        from mliprun.cli.utils import param_sources_from_ctx

        assert param_sources_from_ctx(None) == {}

    def test_unrecognized_source_is_skipped_not_labeled_unspecified(self):
        """An out-of-vocabulary ParameterSource must SKIP the key, not fall
        back to a literal "unspecified" -- that label is the record module's
        own default for keys this function never emits (see
        run_record.py:251-252), and this function producing it directly would
        be a latent trap if a future Click version adds a 6th source.
        """
        from types import SimpleNamespace

        from mliprun.cli.utils import param_sources_from_ctx

        class FakeCtx:
            """Duck-types the two attributes param_sources_from_ctx uses;
            not a real click.Context, deliberately. The source has a ``.name``
            (like a real ParameterSource enum member) that is not in the
            vocabulary, so it exercises the label-miss skip path."""
            params = {"x": 1}

            def get_parameter_source(self, name):
                return SimpleNamespace(name="NOT_A_REAL_SOURCE")

        result = param_sources_from_ctx(FakeCtx())
        assert "x" not in result
        assert result == {}

    def test_source_labels_cover_every_current_parameter_source_member(self):
        """Belt-and-suspenders: confirms the fallback in the previous test
        is exercising a genuinely out-of-vocabulary value, not one that
        happens to already be missing from _SOURCE_LABELS today. Keyed by
        member NAME because _SOURCE_LABELS is name-keyed (typer vendors its
        own click enum; matching by member object would miss)."""
        from click.core import ParameterSource
        from mliprun.cli.utils import _SOURCE_LABELS

        for member in ParameterSource:
            assert member.name in _SOURCE_LABELS, f"{member} has no record-vocabulary label"

    def test_maps_default_map_env_and_prompt_sources(self):
        """Drives all three untested mappings end-to-end through the public
        Typer/Click path in one invocation: DEFAULT_MAP (config-file-style
        value), ENVIRONMENT (envvar-sourced), and PROMPT (interactively
        supplied)."""
        import typer
        from typer.testing import CliRunner

        from mliprun.cli.utils import param_sources_from_ctx

        seen = {}
        app = typer.Typer()

        @app.command()
        def go(ctx: typer.Context,
               fmax: float = typer.Option(0.05),
               envval: float = typer.Option(0.1, envvar="TEST_ENVVAL"),
               name: str = typer.Option(..., prompt=True)):
            seen.update(param_sources_from_ctx(ctx))

        result = CliRunner().invoke(
            app, [],
            default_map={"fmax": 0.09},
            env={"TEST_ENVVAL": "2.5"},
            input="bob\n",
        )
        assert result.exit_code == 0, result.output
        assert seen["fmax"] == "user"
        assert seen["envval"] == "env"
        assert seen["name"] == "prompt"

    def test_source_labels_dict_values_directly(self):
        """Direct check of the mapping table itself, per the review's minimum
        bar, in addition to the end-to-end test above. Keyed by member NAME
        (the table is name-keyed so it works across click and typer's vendored
        click)."""
        from click.core import ParameterSource
        from mliprun.cli.utils import _SOURCE_LABELS

        assert _SOURCE_LABELS[ParameterSource.DEFAULT_MAP.name] == "user"
        assert _SOURCE_LABELS[ParameterSource.ENVIRONMENT.name] == "env"
        assert _SOURCE_LABELS[ParameterSource.PROMPT.name] == "prompt"


class TestResolveDeviceRelocation:
    def test_explicit_device_passes_through_from_core(self):
        from mliprun.core.utils import resolve_device

        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda") == "cuda"

    def test_auto_resolves_to_a_concrete_device(self):
        from mliprun.core.utils import resolve_device

        assert resolve_device("auto") in {"cuda", "cpu"}

    def test_cli_alias_still_points_at_the_same_function(self):
        """cli/utils.py:408 still calls _resolve_device; keep it working."""
        from mliprun.cli.utils import _resolve_device
        from mliprun.core.utils import resolve_device

        assert _resolve_device is resolve_device


class TestSevenNetTable:
    """The tag -> task table is the single source of truth for SevenNet.

    Every value here was read from the checkpoints themselves on 2026-09-01
    (``sevenn`` 0.13.0): the tag list from
    ``sevenn.util.get_available_pretrained_models()``, the task lists from the
    ``Modality`` line of ``sevenn cp <tag>``.
    """

    def test_every_registry_tag_is_present(self):
        # No 7net-nano-* in this release, despite the documentation listing it.
        expected = {
            "7net-omni", "7net-omni-i8", "7net-omni-i12",
            "7net-mf-ompa", "7net-mf-0",
            "7net-omat", "7net-l3i5", "7net-0", "7net-0_22may2024",
        }
        assert set(_SEVENNET_MODELS) == expected

    def test_multi_task_models_carry_tasks(self):
        assert len(_SEVENNET_MODELS["7net-omni"]) == 13
        assert _SEVENNET_MODELS["7net-mf-ompa"] == ("omat24", "mpa")
        # oc20/oc22 are the surface heads; their presence is the reason
        # 7net-omni replaced 7net-mf-ompa as the auto-detected tag.
        assert "oc20" in _SEVENNET_MODELS["7net-omni"]
        assert "oc22" in _SEVENNET_MODELS["7net-omni"]

    def test_mf_0_tasks_are_uppercase(self):
        # 7net-mf-0 is the one tag whose checkpoint names its modalities in
        # uppercase. Storing them lowercased would send SevenNet a task its
        # checkpoint does not have.
        assert _SEVENNET_MODELS["7net-mf-0"] == ("PBE", "R2SCAN")

    def test_single_task_models_carry_no_tasks(self):
        for tag in ("7net-omat", "7net-l3i5", "7net-0", "7net-0_22may2024"):
            assert _SEVENNET_MODELS[tag] == ()

    def test_omni_variants_share_one_task_list(self):
        assert _SEVENNET_MODELS["7net-omni-i8"] == _SEVENNET_MODELS["7net-omni"]
        assert _SEVENNET_MODELS["7net-omni-i12"] == _SEVENNET_MODELS["7net-omni"]


@patch("mliprun.cli.utils.SEVENN_AVAILABLE", True)
class TestValidateSevenNetTask:
    def test_multi_task_without_task_raises(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("7net-omni")
        assert "--sevennet-task is required" in str(exc.value.exit_code)

    def test_multi_task_error_lists_the_valid_tasks(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("7net-mf-ompa")
        message = str(exc.value.exit_code)
        assert "omat24" in message and "mpa" in message

    def test_invalid_task_raises_and_names_it(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("7net-omni", sevennet_task="not_a_task")
        assert "not_a_task" in str(exc.value.exit_code)

    def test_task_on_single_task_model_raises(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("7net-0", sevennet_task="mpa")
        assert "no selectable task" in str(exc.value.exit_code)

    def test_valid_pair_passes(self):
        validate_mlip("7net-omni", sevennet_task="oc20")
        validate_mlip("7net-mf-ompa", sevennet_task="mpa")
        validate_mlip("7net-mf-0", sevennet_task="PBE")

    def test_task_matching_is_case_sensitive(self):
        # 'pbe' is not a task 7net-mf-0 has; accepting it would invent a name
        # the checkpoint does not carry.
        with pytest.raises(typer.Exit):
            validate_mlip("7net-mf-0", sevennet_task="pbe")

    def test_single_task_model_without_task_passes(self):
        validate_mlip("7net-0")
        validate_mlip("7net-omat")

    def test_unknown_7net_tag_passes_with_a_warning(self, capsys):
        validate_mlip("7net-future-model", sevennet_task="whatever")
        assert "can be validated" in capsys.readouterr().out


class TestBuildSevenNetCalculator:
    """SevenNetCalculator is patched, so these run with no sevenn installed."""

    def _build(self, *args, **kwargs):
        from unittest.mock import MagicMock
        fake_cls = MagicMock()
        with patch("mliprun.cli.utils._load_sevenn_calculator",
                   return_value=fake_cls):
            build_calculator(*args, **kwargs)
        return fake_cls

    def test_passes_tag_and_task_to_the_calculator(self):
        fake_cls = self._build("7net-omni", device="cpu", sevennet_task="oc20")
        fake_cls.assert_called_once_with("7net-omni", modal="oc20", device="cpu")

    def test_task_is_forwarded_verbatim_not_case_folded(self):
        fake_cls = self._build("7net-mf-0", device="cpu", sevennet_task="R2SCAN")
        fake_cls.assert_called_once_with("7net-mf-0", modal="R2SCAN", device="cpu")

    def test_single_task_model_gets_no_modal_argument(self):
        # Passing modal=None to a single-task checkpoint is not the same as
        # omitting it; SevenNet only accepts the argument for multi-fidelity
        # models.
        fake_cls = self._build("7net-0", device="cpu")
        fake_cls.assert_called_once_with("7net-0", device="cpu")

    def test_unknown_tag_is_forwarded_with_its_task(self):
        fake_cls = self._build("7net-future", device="cpu", sevennet_task="mpa")
        fake_cls.assert_called_once_with("7net-future", modal="mpa", device="cpu")


class TestHeadTables:
    """UMA tasks and MACE heads, read from the installed packages on
    2026-09-01: fairchem-core 2.19.0 (uma-s-1p2's own task registry) and
    mace-torch 0.3.15 (the mace-mh-1 checkpoint's `heads`)."""

    def test_uma_tasks(self):
        from mliprun.cli.utils import _UMA_TASKS
        # The old --uma-task help advertised only omat/oc20/omol/odac; the
        # model actually carries three more.
        assert _UMA_TASKS == ("omat", "omc", "omol", "oc20", "oc22",
                              "oc25", "odac")

    def test_mace_mh_heads(self):
        from mliprun.cli.utils import _MACE_MH_HEADS
        assert set(_MACE_MH_HEADS) == {
            "omat_pbe", "mp_pbe_refit_add", "matpes_r2scan",
            "oc20_usemppbe", "omol", "spice_wB97M"}


@patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", True)
class TestValidateUmaTask:
    def test_uma_without_task_raises(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("uma-s-1p2")
        assert "--uma-task is required" in str(exc.value.exit_code)

    def test_error_lists_the_valid_tasks(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("uma-s-1p2")
        assert "oc20" in str(exc.value.exit_code)

    def test_invalid_task_raises(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("uma-s-1p2", uma_task="not_a_task")
        assert "not_a_task" in str(exc.value.exit_code)

    def test_valid_task_passes(self):
        validate_mlip("uma-s-1p2", uma_task="oc20")
        validate_mlip("uma-s-1p1", uma_task="omat")

    def test_unknown_uma_tag_still_requires_a_task(self):
        with pytest.raises(typer.Exit):
            validate_mlip("uma-future-model")

    def test_unknown_uma_tag_does_not_validate_the_value(self, capsys):
        # A newer checkpoint may carry tasks this table has never seen, so
        # the value is forwarded with a warning rather than rejected.
        validate_mlip("uma-future-model", uma_task="some_new_task")
        assert "can be validated" in capsys.readouterr().out


@patch("mliprun.cli.utils.MACE_AVAILABLE", True)
class TestValidateMaceHead:
    def test_multi_head_without_head_raises(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("mace-mh-1")
        assert "--mace-head is required" in str(exc.value.exit_code)

    def test_invalid_head_raises(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("mace-mh-1", mace_head="not_a_head")
        assert "not_a_head" in str(exc.value.exit_code)

    def test_valid_head_passes(self):
        validate_mlip("mace-mh-1", mace_head="oc20_usemppbe")
        validate_mlip("mace-mh-0", mace_head="omat_pbe")

    def test_plain_mace_rejects_a_head(self):
        # MACE-MP-0 is a single-head model, like a single-task SevenNet tag.
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("mace", mace_head="omat_pbe")
        assert "no selectable head" in str(exc.value.exit_code)

    def test_plain_mace_without_a_head_passes(self):
        validate_mlip("mace")


class TestBuildCalculatorGuardsTheHead:
    """validate_mlip only runs on the CLI path. Direct API callers --
    fa2i-mtc's batch_relax.py calls setup_calculator(**calc_kwargs) and omits
    the head entirely when the flag was not given -- would otherwise reach
    FAIRChem/MACE/SevenNet with no head and fail there instead of here."""

    def test_uma_without_task_raises(self):
        with pytest.raises(ValueError) as exc:
            build_calculator("uma-s-1p2", device="cpu")
        assert "uma_task" in str(exc.value)

    def test_mace_mh_without_head_raises(self):
        with pytest.raises(ValueError) as exc:
            build_calculator("mace-mh-1", device="cpu")
        assert "mace_head" in str(exc.value)

    def test_multi_task_sevennet_without_task_raises(self):
        with pytest.raises(ValueError) as exc:
            build_calculator("7net-omni", device="cpu")
        assert "sevennet_task" in str(exc.value)
        assert "oc20" in str(exc.value)

    def test_single_task_sevennet_needs_no_task(self):
        from unittest.mock import MagicMock
        fake_cls = MagicMock()
        with patch("mliprun.cli.utils._load_sevenn_calculator",
                   return_value=fake_cls):
            build_calculator("7net-0", device="cpu")
        fake_cls.assert_called_once_with("7net-0", device="cpu")

    def test_plain_mace_needs_no_head(self):
        from unittest.mock import MagicMock
        fake = MagicMock()
        with patch("mliprun.cli.utils._load_mace_mp", return_value=fake):
            build_calculator("mace", device="cpu")
        fake.assert_called_once_with(model="medium", device="cpu")


# ---------------------------------------------------------------------------
# AQCat25-EV2 (SandboxAQ EquiformerV2 checkpoints on fairchem-core 1.x)
# ---------------------------------------------------------------------------

_MIDFILM = "aqcat25-ev2-midfilm"
_INFILM = "aqcat25-ev2-infilm"


class TestAqcatTables:
    def test_two_tags_map_to_the_pinned_checkpoints(self):
        from mliprun.cli.utils import _AQCAT_MODELS
        assert _AQCAT_MODELS == {
            _MIDFILM: "checkpoints_aqcat_ev2/"
                      "ev2-in+midFiLM-AQCat25+OC20-20M_20251008_223220.pt",
            _INFILM: "checkpoints_aqcat_ev2/"
                     "ev2-inFiLM-OC20-ft-AQCat25+OC20-20M_20251008_223143.pt",
        }

    def test_repo_and_revision_are_pinned(self):
        from mliprun.cli.utils import _AQCAT_REPO_ID, _AQCAT_REVISION
        assert _AQCAT_REPO_ID == "SandboxAQ/aqcat25-ev2"
        assert _AQCAT_REVISION == "2a7d5574aa7f0addc0678080eb7621a47a892abd"

    def test_is_aqcat_tag(self):
        from mliprun.cli.utils import is_aqcat_tag
        assert is_aqcat_tag(_MIDFILM) and is_aqcat_tag(_INFILM)
        assert not is_aqcat_tag("uma-s-1p2")
        assert not is_aqcat_tag("mace")
        assert not is_aqcat_tag(None)

    def test_recipe_points_at_the_aqcat_recipe(self):
        from mliprun.cli.utils import _recipe_for_tag
        assert _recipe_for_tag(_MIDFILM) == "aqcat25.md#aqcat25-ev2-midfilm"
        assert _recipe_for_tag(_INFILM) == "aqcat25.md#aqcat25-ev2-infilm"
        assert _recipe_for_tag("aqcat25-future") == "aqcat25.md"


class TestFairchemMajorVersion:
    """One distribution name, two families: 2.x is UMA, 1.x is AQCat."""

    @pytest.mark.parametrize("version,expected", [
        ("2.19.0", 2), ("1.10.0", 1), ("1.10.1.dev3+g1234", 1),
        ("garbage", None), ("", None), (None, None),
    ], ids=["v2", "v1", "v1_dev", "garbage", "empty", "none"])
    def test_major_version_parsing(self, version, expected):
        from mliprun.cli.utils import _major_version
        assert _major_version(version) == expected

    def _check(self, func_name, version):
        from types import SimpleNamespace
        import mliprun.cli.utils as u
        func = getattr(u, func_name)
        func.cache_clear()
        try:
            with patch("mliprun.cli.utils.distribution",
                       return_value=SimpleNamespace(version=version)):
                return func()
        finally:
            func.cache_clear()

    def test_fairchem_2_is_uma_not_aqcat(self):
        assert self._check("_check_fairchem", "2.19.0") is True
        assert self._check("_check_aqcat", "2.19.0") is False

    def test_fairchem_1_is_aqcat_not_uma(self):
        assert self._check("_check_fairchem", "1.10.0") is False
        assert self._check("_check_aqcat", "1.10.0") is True

    def test_unparseable_version_is_neither(self):
        assert self._check("_check_fairchem", "weird") is False
        assert self._check("_check_aqcat", "weird") is False

    def test_absent_distribution_is_neither(self):
        import mliprun.cli.utils as u
        from importlib.metadata import PackageNotFoundError
        for name in ("_check_fairchem", "_check_aqcat"):
            func = getattr(u, name)
            func.cache_clear()
            try:
                with patch("mliprun.cli.utils.distribution",
                           side_effect=PackageNotFoundError("fairchem-core")):
                    assert func() is False
            finally:
                func.cache_clear()


@patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", False)
@patch("mliprun.cli.utils.MACE_AVAILABLE", False)
@patch("mliprun.cli.utils.SEVENN_AVAILABLE", False)
@patch("mliprun.cli.utils.CHGNET_AVAILABLE", False)
class TestDetectMlipInAqcatOnlyEnv:
    @patch("mliprun.cli.utils.AQCAT_AVAILABLE", True)
    def test_auto_names_the_tags_and_the_spin_flag(self):
        with pytest.raises(typer.Exit) as exc:
            detect_mlip()
        msg = str(exc.value.exit_code)
        assert _MIDFILM in msg and _INFILM in msg
        assert "--aqcat-spin" in msg
        assert "No supported MLIP installed" not in msg

    @patch("mliprun.cli.utils.AQCAT_AVAILABLE", False)
    def test_nothing_installed_keeps_the_generic_message(self):
        with pytest.raises(typer.Exit) as exc:
            detect_mlip()
        assert "No supported MLIP installed" in str(exc.value.exit_code)


@patch("mliprun.cli.utils.AQCAT_AVAILABLE", True)
class TestValidateAqcatOptions:
    def test_spin_is_required_and_the_error_says_why(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip(_MIDFILM)
        msg = str(exc.value.exit_code)
        assert "--aqcat-spin" in msg
        assert "several eV" in msg

    @pytest.mark.parametrize("spin", ["on", "off"], ids=["spin_on", "spin_off"])
    def test_valid_spin_passes_with_and_without_fidelity(self, spin):
        validate_mlip(_MIDFILM, aqcat_spin=spin)
        validate_mlip(_INFILM, aqcat_spin=spin, aqcat_fidelity="low")
        validate_mlip(_INFILM, aqcat_spin=spin, aqcat_fidelity="high")

    def test_invalid_spin_is_named(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip(_MIDFILM, aqcat_spin="yes")
        assert "'yes'" in str(exc.value.exit_code)

    def test_invalid_fidelity_is_named(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip(_MIDFILM, aqcat_spin="on", aqcat_fidelity="medium")
        assert "'medium'" in str(exc.value.exit_code)

    def test_unknown_aqcat_tag_lists_the_known_ones(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("aqcat25-ev2-plain", aqcat_spin="on")
        msg = str(exc.value.exit_code)
        assert _MIDFILM in msg and _INFILM in msg

    def test_resolve_mlip_forwards_the_options(self):
        assert resolve_mlip(_MIDFILM, aqcat_spin="off") == _MIDFILM
        with pytest.raises(typer.Exit):
            resolve_mlip(_MIDFILM)


class TestAqcatOptionsOnOtherModels:
    @patch("mliprun.cli.utils.MACE_AVAILABLE", True)
    def test_spin_on_a_non_aqcat_tag_is_rejected(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("mace", aqcat_spin="on")
        assert "--aqcat-spin" in str(exc.value.exit_code)

    @patch("mliprun.cli.utils.CHGNET_AVAILABLE", True)
    def test_fidelity_on_a_non_aqcat_tag_is_rejected(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("chgnet", aqcat_fidelity="low")
        assert "--aqcat-fidelity" in str(exc.value.exit_code)

    @patch("mliprun.cli.utils.AQCAT_AVAILABLE", False)
    def test_unavailable_points_at_the_recipe(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip(_MIDFILM, aqcat_spin="on")
        assert "aqcat25.md" in str(exc.value.exit_code)

    def test_unknown_tag_message_names_the_aqcat_tags(self):
        with pytest.raises(typer.Exit) as exc:
            validate_mlip("nonexistent-model")
        assert _MIDFILM in str(exc.value.exit_code)


class TestRejectAqcatUnsupported:
    def test_rejects_with_the_shared_message(self):
        from mliprun.cli.utils import reject_aqcat_unsupported
        with pytest.raises(typer.Exit) as exc:
            reject_aqcat_unsupported(_MIDFILM, "md")
        msg = str(exc.value.exit_code)
        assert ("AQCat25-EV2 models are supported in singlepoint, optimize "
                "and neb only for now") in msg
        assert "md" in msg

    def test_other_tags_pass(self):
        from mliprun.cli.utils import reject_aqcat_unsupported
        assert reject_aqcat_unsupported("mace", "md") is None
        assert reject_aqcat_unsupported("auto", "md") is None


class TestCheckAqcatOptimize:
    def test_relax_cell_is_refused_for_lack_of_stress(self):
        from mliprun.cli.utils import check_aqcat_optimize
        with pytest.raises(ValueError) as exc:
            check_aqcat_optimize(_MIDFILM, "bfgs", relax_cell=True)
        assert "stress" in str(exc.value)

    @pytest.mark.parametrize("optimizer", ["bfgsls", "GPMin"],
                             ids=["line_search", "gaussian_process"])
    def test_gradient_assuming_optimizers_are_refused(self, optimizer):
        from mliprun.cli.utils import check_aqcat_optimize
        with pytest.raises(ValueError) as exc:
            check_aqcat_optimize(_INFILM, optimizer, relax_cell=False)
        assert "gradient" in str(exc.value)

    def test_plain_relaxation_passes_and_other_models_are_untouched(self):
        from mliprun.cli.utils import check_aqcat_optimize
        assert check_aqcat_optimize(_MIDFILM, "fire", relax_cell=False) is None
        assert check_aqcat_optimize("mace", "gpmin", relax_cell=True) is None


class TestBuildAqcatCalculator:
    """patched_calc and the checkpoint download are patched: no fairchem,
    no network."""

    def _build(self, *args, **kwargs):
        from unittest.mock import MagicMock
        fake = MagicMock(name="patched_calc")
        with patch("mliprun.cli.utils._load_aqcat_calc", return_value=fake), \
             patch("mliprun.cli.utils._ensure_aqcat_checkpoint",
                   return_value="/cache/ckpt.pt") as ensure:
            build_calculator(*args, **kwargs)
        return fake, ensure

    def test_spin_on_high_fidelity_on_cpu(self):
        fake, ensure = self._build(_MIDFILM, device="cpu", aqcat_spin="on")
        ensure.assert_called_once_with(_MIDFILM)
        fake.assert_called_once_with("/cache/ckpt.pt", is_spin_off=False,
                                     is_low_fi=False, cpu=True)

    def test_spin_off_low_fidelity_on_cuda(self):
        fake, _ = self._build(_INFILM, device="cuda", aqcat_spin="off",
                              aqcat_fidelity="low")
        fake.assert_called_once_with("/cache/ckpt.pt", is_spin_off=True,
                                     is_low_fi=True, cpu=False)

    def test_setup_calculator_forwards_the_options(self):
        from unittest.mock import MagicMock
        from ase.build import bulk
        from mliprun.cli.utils import setup_calculator
        fake = MagicMock(name="patched_calc")
        with patch("mliprun.cli.utils._load_aqcat_calc", return_value=fake), \
             patch("mliprun.cli.utils._ensure_aqcat_checkpoint",
                   return_value="/c.pt"):
            atoms = setup_calculator(bulk("Cu"), _MIDFILM, device="cpu",
                                     aqcat_spin="off")
        assert atoms.calc is fake.return_value
        fake.assert_called_once_with("/c.pt", is_spin_off=True,
                                     is_low_fi=False, cpu=True)

    def test_missing_spin_raises_before_loading_anything(self):
        from unittest.mock import MagicMock
        loader = MagicMock()
        with patch("mliprun.cli.utils._load_aqcat_calc", loader):
            with pytest.raises(ValueError) as exc:
                build_calculator(_MIDFILM, device="cpu")
        assert "aqcat_spin" in str(exc.value)
        assert "several eV" in str(exc.value)
        loader.assert_not_called()

    def test_invalid_values_raise(self):
        with pytest.raises(ValueError):
            build_calculator(_MIDFILM, device="cpu", aqcat_spin="maybe")
        with pytest.raises(ValueError):
            build_calculator(_MIDFILM, device="cpu", aqcat_spin="on",
                             aqcat_fidelity="medium")
        with pytest.raises(ValueError):
            build_calculator("aqcat25-ev2-plain", device="cpu",
                             aqcat_spin="on")

    def test_device_other_than_cpu_or_cuda_raises(self):
        with pytest.raises(ValueError) as exc:
            build_calculator(_MIDFILM, device="mps", aqcat_spin="on")
        assert "mps" in str(exc.value)

    def test_same_flag_pair_twice_is_allowed(self):
        fake, _ = self._build(_MIDFILM, device="cpu", aqcat_spin="on")
        fake2, _ = self._build(_INFILM, device="cpu", aqcat_spin="on",
                               aqcat_fidelity="high")
        assert fake.call_count == 1 and fake2.call_count == 1

    def test_a_different_flag_pair_in_one_process_raises(self):
        self._build(_MIDFILM, device="cpu", aqcat_spin="on")
        with pytest.raises(ValueError) as exc:
            self._build(_MIDFILM, device="cpu", aqcat_spin="off")
        assert "process" in str(exc.value)

    def test_dispatch_goes_through_the_shared_helper(self):
        from unittest.mock import MagicMock
        helper = MagicMock(return_value="calc")
        with patch("mliprun.cli.utils._build_aqcat_calculator", helper):
            assert build_calculator(_INFILM, device="cpu", aqcat_spin="on",
                                    aqcat_fidelity="low") == "calc"
        helper.assert_called_once_with(_INFILM, "on", "low", "cpu")


class TestAqcatLoaderAndCheckpoint:
    def test_loader_without_the_patch_points_at_the_recipe(self, monkeypatch):
        import sys
        from mliprun.cli.utils import _load_aqcat_calc
        _load_aqcat_calc.cache_clear()
        # A None entry makes `import` raise ImportError for that name.
        monkeypatch.setitem(
            sys.modules, "fairchem.core.common.relaxation.ase_utils", None)
        try:
            with pytest.raises(ImportError) as exc:
                _load_aqcat_calc()
        finally:
            _load_aqcat_calc.cache_clear()
        assert "SandboxAQ" in str(exc.value)
        assert "aqcat25.md" in str(exc.value)

    def test_loader_returns_patched_calc(self, monkeypatch):
        import sys
        import types
        from mliprun.cli.utils import _load_aqcat_calc
        module = types.ModuleType("fairchem.core.common.relaxation.ase_utils")
        module.patched_calc = object()
        monkeypatch.setitem(
            sys.modules, "fairchem.core.common.relaxation.ase_utils", module)
        _load_aqcat_calc.cache_clear()
        try:
            assert _load_aqcat_calc() is module.patched_calc
        finally:
            _load_aqcat_calc.cache_clear()

    def _fake_hub(self, monkeypatch, download):
        import sys
        import types
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = download
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    def test_checkpoint_download_uses_the_pinned_revision(self, monkeypatch):
        from unittest.mock import MagicMock
        from mliprun.cli.utils import _ensure_aqcat_checkpoint
        download = MagicMock(return_value="/hf/cache/file.pt")
        self._fake_hub(monkeypatch, download)
        assert _ensure_aqcat_checkpoint(_INFILM) == "/hf/cache/file.pt"
        download.assert_called_once_with(
            repo_id="SandboxAQ/aqcat25-ev2",
            filename="checkpoints_aqcat_ev2/"
                     "ev2-inFiLM-OC20-ft-AQCat25+OC20-20M_20251008_223143.pt",
            revision="2a7d5574aa7f0addc0678080eb7621a47a892abd")

    def test_gated_repo_error_points_at_the_recipe(self, monkeypatch):
        from mliprun.cli.utils import _ensure_aqcat_checkpoint

        class GatedRepoError(Exception):
            pass

        def download(**kwargs):
            raise GatedRepoError("401 Client Error")

        self._fake_hub(monkeypatch, download)
        with pytest.raises(RuntimeError) as exc:
            _ensure_aqcat_checkpoint(_MIDFILM)
        assert "aqcat25.md" in str(exc.value)
        assert "hf auth login" in str(exc.value)

    def test_http_403_is_treated_as_an_access_failure(self, monkeypatch):
        from types import SimpleNamespace
        from mliprun.cli.utils import _ensure_aqcat_checkpoint

        class HfHubHTTPError(Exception):
            response = SimpleNamespace(status_code=403)

        def download(**kwargs):
            raise HfHubHTTPError("forbidden")

        self._fake_hub(monkeypatch, download)
        with pytest.raises(RuntimeError) as exc:
            _ensure_aqcat_checkpoint(_MIDFILM)
        assert "aqcat25.md" in str(exc.value)

    def test_other_download_errors_propagate_unchanged(self, monkeypatch):
        from mliprun.cli.utils import _ensure_aqcat_checkpoint

        def download(**kwargs):
            raise OSError("disk full")

        self._fake_hub(monkeypatch, download)
        with pytest.raises(OSError, match="disk full"):
            _ensure_aqcat_checkpoint(_MIDFILM)

    def test_unknown_tag_raises(self):
        from mliprun.cli.utils import _ensure_aqcat_checkpoint
        with pytest.raises(ValueError):
            _ensure_aqcat_checkpoint("aqcat25-ev2-plain")

    def test_missing_huggingface_hub_points_at_the_recipe(self, monkeypatch):
        import sys
        from mliprun.cli.utils import _ensure_aqcat_checkpoint
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(ImportError) as exc:
            _ensure_aqcat_checkpoint(_MIDFILM)
        assert "aqcat25.md" in str(exc.value)


class TestEchoAqcatSettings:
    def test_echoes_spin_resolved_fidelity_and_the_caveat(self, capsys):
        from mliprun.cli.utils import echo_aqcat_settings
        echo_aqcat_settings(_MIDFILM, "on", None)
        out = capsys.readouterr().out
        assert "AQCat spin: on" in out
        assert "AQCat fidelity: high" in out
        assert "adsorption energy" in out
        assert "not the energy gradient" in out or "rather than the energy gradient" in out

    def test_silent_for_other_models(self, capsys):
        from mliprun.cli.utils import echo_aqcat_settings
        echo_aqcat_settings("mace", "on", "low")
        assert capsys.readouterr().out == ""
