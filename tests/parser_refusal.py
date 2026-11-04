"""Report a test that fails because this host refuses to parse results as skipped.

The server parses result files only where it can contain the parser process,
and on macOS it refuses (``parser_bootstrap.require_containment_platform``;
docs/TESTING.md, "Platforms and the release gate"). A test that reads a result
cannot pass there. No marker or fixture names those tests, because a result is
read through a handler, a fixture, a job or the Python API alike, so this
plugin watches the refusal itself.

On such a host the supervisor's check is wrapped once per process: the real
check still runs and still raises, and a refusal it raises is recorded for the
running test. A test whose setup, call or teardown fails after it met a
refusal is reported as skipped, with the refusal as the reason. A test that
meets the refusal and passes stays passed, and every failure without one stays
a failure, so a fault of the host's own still shows.

A refusal in a process the test started itself (a supervisor in a program the
test runs, or the parser's entry module started directly) is not seen here.
Such a test carries ``needs_parsing``, which skips it on such a host with the
same reason.

On a host that parses, nothing is wrapped and no report changes, even for a
test that makes the check refuse by setting ``sys.platform``.
"""

from __future__ import annotations

import threading
from collections.abc import Generator

import pytest

from ltspice_mcp.lib import parser_bootstrap, parser_process


def _host_refusal() -> str | None:
    """The message this host's parser refusal carries, or None where it parses."""
    try:
        parser_bootstrap.require_containment_platform()
    except RuntimeError as refusal:
        return str(refusal)
    return None


REFUSAL = _host_refusal()
"""Why this host refuses to parse results, or None where it parses them."""

REASON = f"result parsing is refused on this host: {REFUSAL}"
"""The reason a test skipped for the refusal gives."""

needs_parsing = pytest.mark.skipif(REFUSAL is not None, reason=REASON)
"""Skip, where the host refuses to parse, a test whose parser runs in a process
the test started itself."""

_refused = threading.Event()
"""Set once the running test has met the refusal in this process."""


def _recorded_check() -> None:
    try:
        parser_bootstrap.require_containment_platform()
    except RuntimeError:
        _refused.set()
        raise


def pytest_configure(config: pytest.Config) -> None:
    if REFUSAL is not None:
        parser_process.require_containment_platform = _recorded_check


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    _refused.clear()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if _refused.is_set() and report.failed:
        path, line, _ = item.reportinfo()
        report.outcome = "skipped"
        report.longrepr = (str(path), (line or 0) + 1, REASON)
    return report
