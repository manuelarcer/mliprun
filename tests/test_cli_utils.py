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


class TestResolveMlip:
    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", True)
    def test_auto_resolves(self):
        result = resolve_mlip("auto")
        assert isinstance(result, str)
        assert result != "auto"

    @patch("mliprun.cli.utils.FAIRCHEM_AVAILABLE", True)
    def test_explicit_passes_through(self):
        result = resolve_mlip("uma-s-1p1")
        assert result == "uma-s-1p1"


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
