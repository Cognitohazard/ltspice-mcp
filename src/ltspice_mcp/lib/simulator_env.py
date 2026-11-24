"""The environment a simulator process is started with.

Every launch of a simulator, whether a run, an export, a netlist, a version
probe or a question the server asks ngspice, starts from
:func:`simulator_environment`, and spicelib's own launches are routed here by
:func:`install_spicelib_launch_environment`. The server's own environment is
never changed: the parser, the ``run_code`` worker and a detached job's owner
inherit it, and none of them is a simulator.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

from spicelib.sim import simulator as _spicelib_simulator

from ltspice_mcp.lib import wsl

logger = logging.getLogger(__name__)

#: Variables a simulator process is started without, matched without regard
#: to case, as Windows matches them.
#:
#: ``NoDefaultCurrentDirectoryInExePath``: a Windows setting that some MCP
#: clients put in the environment of the servers they start. With it set,
#: LTspice 26 on macOS, which starts through a Wine launcher, was reported to
#: exit with code 0 and write no log and no raw file; the reporter measured
#: that removing it alone restores the run. That report is the only evidence.
#: It is removed on macOS, and under Wine on Linux because LTspice runs
#: through Wine there too; no one has measured Linux. It is kept on native
#: Windows and under WSL interop, where no failure is reported.
REMOVED_VARIABLES: tuple[str, ...] = ("NoDefaultCurrentDirectoryInExePath",)


def _host_removes() -> bool:
    """Whether this host starts simulators without ``REMOVED_VARIABLES``:
    macOS, and Linux other than WSL, where LTspice runs under Wine."""
    return sys.platform != "win32" and not wsl.is_wsl()


def simulator_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment to start a simulator with: ``base`` (the server's own
    environment when None), copied, without the variables this host removes.

    The result is a new dict a caller may extend; ``os.environ`` is never
    changed.
    """
    source = os.environ if base is None else base
    if not _host_removes():
        return dict(source)
    removed = {name.upper() for name in REMOVED_VARIABLES}
    return {name: value for name, value in source.items() if name.upper() not in removed}


def run_simulator(
    command: Any,
    timeout: float | None = None,
    stdout: Any = None,
    stderr: Any = None,
    cwd: Any = None,
) -> int:
    """spicelib's ``run_function``, with the simulator's environment: start
    ``command``, wait for it, and return its exit code."""
    logger.debug("Running command: %s, with timeout: %s", command, timeout)
    result = subprocess.run(
        command,
        timeout=timeout,
        stdout=stdout,
        stderr=stderr,
        cwd=cwd,
        env=simulator_environment(),
    )
    return result.returncode


#: spicelib's own ``run_function``, the one each of its modules is given.
_SPICELIB_RUN_FUNCTION = _spicelib_simulator.run_function  # noqa: TID251 - the function replaced


def install_spicelib_launch_environment() -> None:
    """Route every spicelib launch through :func:`run_simulator`.

    spicelib starts every simulator, in ``run`` and ``create_netlist`` of each
    family, through the module-level ``run_function``, which takes no
    environment (``docs/spicelib_bugs.md``, "run_function takes no
    environment"). A subclass would have to repeat each family's command
    line to pass one, so the function itself is replaced: in
    ``spicelib.sim.simulator``, which a module loaded later imports it from,
    and in every spicelib module already loaded that holds its own reference.
    Idempotent.
    """
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "spicelib" or name.startswith("spicelib.")):
            continue
        if getattr(module, "run_function", None) is _SPICELIB_RUN_FUNCTION:
            module.run_function = run_simulator  # type: ignore[attr-defined]


install_spicelib_launch_environment()
