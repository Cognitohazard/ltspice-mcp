"""pytest plugin: run the suite as a Linux host that is not WSL.

The development box is WSL2, where ``is_wsl()`` is true and every WSL
interop branch is taken; Linux CI is not. Loading this plugin
(``PYTHONPATH=scripts uv run pytest -p nonwsl_plugin tests/``) forces the
detection off for every test that does not set it itself, so the branch CI
runs is the branch that was checked.
"""

import pytest


@pytest.fixture(autouse=True)
def _force_non_wsl(monkeypatch: pytest.MonkeyPatch) -> None:
    import ltspice_mcp.lib.wsl as wsl

    monkeypatch.setattr(wsl, "_is_wsl_cached", False)
