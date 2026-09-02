

class TestBenchmarkSevenNet:
    """SevenNet's recommended model is multi-task, and its tasks have
    independent energy zeros. Benchmarking it under an assumed task would
    report a number nobody chose, so it joins the auto list only when a task
    was given (CANON C1)."""

    def _flags(self, sevenn):
        from unittest.mock import patch
        return [
            patch("mliprun.cli.commands.benchmark.FAIRCHEM_AVAILABLE", False),
            patch("mliprun.cli.commands.benchmark.MACE_AVAILABLE", False),
            patch("mliprun.cli.commands.benchmark.CHGNET_AVAILABLE", False),
            patch("mliprun.cli.commands.benchmark.SEVENN_AVAILABLE", sevenn),
        ]

    def _available(self, sevenn, **kwargs):
        import contextlib
        from mliprun.cli.commands.benchmark import _available_models
        with contextlib.ExitStack() as stack:
            for p in self._flags(sevenn):
                stack.enter_context(p)
            return _available_models(**kwargs)

    def test_auto_list_skips_sevennet_without_a_task(self):
        assert self._available(True) == []

    def test_auto_list_includes_sevennet_with_a_task(self):
        assert self._available(True, sevennet_task="mpa") == ["7net-omni"]

    def test_task_does_not_conjure_sevennet_when_absent(self):
        assert self._available(False, sevennet_task="mpa") == []
