"""`mlip webapp` / `python -m mliprun.webapp`: options, lazy imports, install hint."""
import re
import subprocess
import sys

import pytest
from typer.testing import CliRunner

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text):
    """Strip ANSI styling; rich colours help output on CI (GITHUB_ACTIONS)."""
    return _ANSI.sub("", text)


def test_cli_main_does_not_import_fastapi():
    code = ("import sys, mliprun.cli.main; "
            "print(any(m in sys.modules for m in ('fastapi', 'uvicorn', 'starlette')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "False"


def test_mlip_help_lists_webapp_with_four_options():
    from mliprun.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0 and "webapp" in r.output
    r = runner.invoke(app, ["webapp", "--help"])
    assert r.exit_code == 0
    out = _plain(r.output)
    for opt in ("--host", "--port", "--runs-dir", "--no-browser"):
        assert opt in out
    assert "127.0.0.1" in out and "8765" in out


def test_python_m_mliprun_webapp_help():
    r = subprocess.run([sys.executable, "-m", "mliprun.webapp", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    out = _plain(r.stdout)
    assert "--no-browser" in out and "--runs-dir" in out


def test_missing_extra_exits_1_with_hint(monkeypatch):
    import importlib.util

    from mliprun.cli.main import app

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "fastapi" else real(name, *a))
    r = CliRunner().invoke(app, ["webapp", "--no-browser"])
    assert r.exit_code == 1
    assert 'pip install "mliprun[webapp]"' in r.output
    assert 'pip install -e ".[webapp]"' in r.output


def test_allowed_hosts_and_url():
    from mliprun.webapp.cli import allowed_hosts_for, url_for

    assert allowed_hosts_for("127.0.0.1", 9000) == ["127.0.0.1:9000", "localhost:9000",
                                                    "[::1]:9000"]
    assert allowed_hosts_for("localhost", 1) == ["127.0.0.1:1", "localhost:1", "[::1]:1"]
    assert allowed_hosts_for("::1", 2)[2] == "[::1]:2"
    assert allowed_hosts_for("0.0.0.0", 8765) is None
    assert url_for("127.0.0.1", 8765) == "http://127.0.0.1:8765/"
    assert url_for("::1", 8765) == "http://[::1]:8765/"
    assert url_for("0.0.0.0", 80) == "http://127.0.0.1:80/"


@pytest.mark.parametrize("args, host, allowed, warned, browser", [
    ([], "127.0.0.1", ["127.0.0.1:8765", "localhost:8765", "[::1]:8765"], False, True),
    (["--host", "0.0.0.0", "--port", "9999", "--no-browser"], "0.0.0.0", None, True, False),
], ids=["loopback", "public"])
def test_launch_wires_create_app_and_uvicorn(monkeypatch, tmp_path, args, host, allowed,
                                             warned, browser):
    pytest.importorskip("fastapi")
    import uvicorn

    import mliprun.webapp.cli as wcli
    import mliprun.webapp.server as server
    from mliprun.cli.main import app

    calls = {}
    real_create = server.create_app

    def fake_create(runs_dir, **kw):
        calls["create"] = (runs_dir, kw)
        return real_create(runs_dir, start_runner=False, **kw)

    monkeypatch.setattr(server, "create_app", fake_create)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.setdefault("run", kw))
    opened = []

    class FakeTimer:
        def __init__(self, delay, fn, args=()):
            self.delay, self.fn, self.args = delay, fn, args
            self.daemon = False

        def start(self):
            opened.append((self.delay, self.args))

    monkeypatch.setattr(wcli.threading, "Timer", FakeTimer)
    runs = tmp_path / "runs"
    r = CliRunner().invoke(app, ["webapp", "--runs-dir", str(runs), *args])
    assert r.exit_code == 0, r.output
    assert calls["create"][1] == {"allowed_hosts": allowed}
    assert calls["run"]["host"] == host
    assert ("no authentication" in r.output) is warned
    assert (len(opened) == 1) is browser
    if browser:
        assert opened[0][0] == pytest.approx(1.0)
        assert opened[0][1] == ("http://127.0.0.1:8765/",)
    assert runs.is_dir() and (runs / "_uploads").is_dir()
    assert "mliprun web app: http://" in r.output
