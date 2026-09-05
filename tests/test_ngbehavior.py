"""ngspice ngbehavior override + the sectioned-.lib failure hint.

The override changes a process-wide spicelib class attribute
(``NGspiceSimulator._compatibility_mode``), so every test saves and restores it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.simulator import (
    _SPICELIB_DEFAULT_NGBEHAVIOR,
    _apply_ngbehavior,
    current_ngbehavior,
)

INCLUDE_ERR = "Error: Could not find include file tt"


@pytest.fixture(autouse=True)
def restore_ngbehavior() -> Iterator[None]:
    saved = NGspiceSimulator._compatibility_mode
    yield
    NGspiceSimulator._compatibility_mode = saved


class TestApplyNgbehavior:
    def test_sets_class_attribute(self):
        _apply_ngbehavior(ServerConfig(ngbehavior="kia"))
        assert current_ngbehavior() == "kia"

    def test_lowercases_and_strips(self):
        _apply_ngbehavior(ServerConfig(ngbehavior="  HSA  "))
        assert current_ngbehavior() == "hsa"

    def test_default_is_kiltpsa(self):
        # Sanity: spicelib still ships the lt-containing default we guard against.
        assert _SPICELIB_DEFAULT_NGBEHAVIOR == "kiltpsa"

    @pytest.mark.parametrize("unset", [ServerConfig(ngbehavior=None), None])
    def test_unset_resets_to_default_after_override(self, unset):
        # The re-entrancy guard: a prior override must NOT leak into a later unset
        # config (or a None config). Unset RESETS to spicelib's captured default,
        # it is not a no-op.
        _apply_ngbehavior(ServerConfig(ngbehavior="hsa"))
        assert current_ngbehavior() == "hsa"
        _apply_ngbehavior(unset)
        assert current_ngbehavior() == _SPICELIB_DEFAULT_NGBEHAVIOR


def _write_deck(tmp_path: Path, body: str) -> Path:
    deck = tmp_path / "deck.cir"
    deck.write_text(body)
    return deck
