"""The simulator families, in one place for the modules that must not import
``simulator`` (which imports spicelib and detects programs at import time).

A family is the lowercase name a run selects (``execution.simulator``) and a
record keeps (``simulator_family``); ``DISPLAY`` is how a person reads it.
``HOSTS`` are the platforms ``simulator._platform_key`` tells apart.
"""

from __future__ import annotations

#: Every supported family, in the order the server lists them.
FAMILIES: tuple[str, ...] = ("ltspice", "ngspice", "qspice", "xyce")

#: How a person names each family, in messages and the server instructions.
DISPLAY: dict[str, str] = {
    "ltspice": "LTspice",
    "ngspice": "ngspice",
    "qspice": "QSPICE",
    "xyce": "Xyce",
}

#: The hosts a capability can be restricted to: native Windows, WSL, Linux
#: (Wine for a Windows program) and macOS.
HOSTS: tuple[str, ...] = ("windows", "wsl", "linux", "darwin")
