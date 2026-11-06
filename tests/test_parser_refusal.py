"""The plugin that reports a test failing on the host's parser refusal as skipped.

Each case runs a small suite in its own interpreter. Its tests make the real
check refuse by setting ``sys.platform`` to ``darwin``, as
``test_parser_process`` does, and the plugin is loaded either as a host that
refuses would load it (``sys.platform`` is ``darwin`` while it decides) or as
one that parses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ltspice_mcp.lib import parser_bootstrap
from tests.conftest import LIVENESS_S

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]

_CONFTEST = """
import sys

import pytest

from ltspice_mcp.lib import parser_process  # noqa: F401  (loaded as this host loads it)

_platform = sys.platform
sys.platform = {plugin_platform!r}
from tests import parser_refusal  # noqa: E402
sys.platform = _platform


def pytest_configure(config):
    config.pluginmanager.register(parser_refusal, "parser-refusal")


@pytest.fixture(autouse=True)
def _refusing(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
"""

_TESTS = """
import time
from pathlib import Path

import pytest

from ltspice_mcp.lib.parser_process import (
    ParserProcessError,
    ParserProcessLimits,
    run_parser_sync,
)
from tests.parser_refusal import needs_parsing

LIMITS = ParserProcessLimits(
    memory_bytes=128 * 1024 * 1024,
    request_bytes=65536,
    metadata_bytes=4096,
    error_bytes=1024,
    cleanup_grace_s=1.0,
)


def parse():
    run_parser_sync({}, work_dir=Path.cwd(), deadline=time.monotonic() + 60, limits=LIMITS)


@pytest.fixture
def parsed():
    parse()


def test_fails_on_the_refusal():
    parse()


def test_setup_fails_on_the_refusal(parsed):
    pass


@needs_parsing
def test_marked_as_needing_parsing():
    parse()


def test_expects_the_refusal():
    with pytest.raises(ParserProcessError, match="containment"):
        parse()


def test_fails_without_the_refusal():
    assert sum([1, 1]) == 3
"""


def _macos_reason(monkeypatch: pytest.MonkeyPatch) -> str:
    """The skip reason a test refused as on macOS gives: the refusal's own text."""
    with monkeypatch.context() as macos:
        macos.setattr(sys, "platform", "darwin")
        with pytest.raises(RuntimeError) as refused:
            parser_bootstrap.require_containment_platform()
    return f"result parsing is refused on this host: {refused.value}"


def _run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, plugin_platform: str):
    pytester.makeconftest(_CONFTEST.format(plugin_platform=plugin_platform))
    pytester.makepyfile(test_inner=_TESTS)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    # The inner suite's failures are intended: keep them out of the
    # annotations a CI run draws for failing tests.
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    return pytester.runpytest_subprocess("-rfs", "-p", "no:cacheprovider", timeout=LIVENESS_S)


def test_a_failure_on_the_refusal_is_skipped_where_the_host_refuses(pytester, monkeypatch):
    reason = _macos_reason(monkeypatch)
    result = _run(pytester, monkeypatch, "darwin")
    result.assert_outcomes(passed=1, failed=1, skipped=3)
    skipped = [line for line in result.stdout.lines if line.startswith("SKIPPED")]
    assert len(skipped) == 3, result.stdout.str()
    for line in skipped:
        assert line.endswith(f": {reason}"), line
    result.stdout.fnmatch_lines(["FAILED test_inner.py::test_fails_without_the_refusal*"])


def test_nothing_changes_where_the_host_parses(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, "linux")
    result.assert_outcomes(passed=1, failed=3, errors=1)
