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
from ltspice_mcp.lib.services import _deck_has_sectioned_lib, ngbehavior_lib_hint
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


class TestDeckHasSectionedLib:
    def test_sectioned_lib_detected(self, tmp_path: Path):
        deck = _write_deck(tmp_path, ".title t\n.lib /pdk/sky130.lib.spice tt\nR1 a b 1k\n.end\n")
        assert _deck_has_sectioned_lib(deck) is True

    def test_section_less_lib_not_detected(self, tmp_path: Path):
        deck = _write_deck(tmp_path, ".lib /pdk/models.lib\nR1 a b 1k\n.end\n")
        assert _deck_has_sectioned_lib(deck) is False

    def test_case_insensitive(self, tmp_path: Path):
        deck = _write_deck(tmp_path, ".LIB models.lib TT\n.end\n")
        assert _deck_has_sectioned_lib(deck) is True

    def test_full_line_comment_not_detected(self, tmp_path: Path):
        deck = _write_deck(tmp_path, "* .lib models.lib tt\n.end\n")
        assert _deck_has_sectioned_lib(deck) is False

    def test_section_less_lib_with_inline_comment_not_detected(self, tmp_path: Path):
        # Inline comment tokens ($ ngspice, ; LTspice) must not count as a section.
        for body in (".lib /pdk/models.lib $ tt corner\n", ".lib /pdk/models.lib ; note\n"):
            assert _deck_has_sectioned_lib(_write_deck(tmp_path, body)) is False, body

    def test_sectioned_lib_with_trailing_comment_detected(self, tmp_path: Path):
        deck = _write_deck(tmp_path, ".lib /pdk/models.lib tt $ note\n.end\n")
        assert _deck_has_sectioned_lib(deck) is True

    def test_quoted_path_with_space_not_detected(self, tmp_path: Path):
        # A quoted path containing a space is ONE token, not file + section.
        deck = _write_deck(tmp_path, '.lib "my models.lib"\n.end\n')
        assert _deck_has_sectioned_lib(deck) is False

    def test_quoted_path_with_space_and_section_detected(self, tmp_path: Path):
        deck = _write_deck(tmp_path, '.lib "my models.lib" tt\n.end\n')
        assert _deck_has_sectioned_lib(deck) is True

    def test_missing_file_is_false(self, tmp_path: Path):
        assert _deck_has_sectioned_lib(tmp_path / "nope.cir") is False


class TestNgbehaviorLibHint:
    def _deck(self, tmp_path: Path) -> Path:
        return _write_deck(tmp_path, ".lib /pdk/sky130.lib.spice tt\nR1 a b 1k\n.end\n")

    def test_fires_on_all_conditions(self, tmp_path: Path):
        hint = ngbehavior_lib_hint(
            self._deck(tmp_path), INCLUDE_ERR, is_ngspice=True, current_mode="kiltpsa"
        )
        assert hint is not None
        # Must recommend a mode with neither lt nor ps; must NOT recommend kipsa
        # (still contains ps → reproduces the same failure).
        assert 'ngbehavior = "hsa"' in hint
        assert "kipsa" not in hint

    def test_fires_when_mode_has_ps_but_no_lt(self, tmp_path: Path):
        # kipsa dropped lt but kept ps, which ALSO splits a sectioned .lib — the
        # user is still broken, so the hint must still fire (guard covers ps).
        assert (
            ngbehavior_lib_hint(
                self._deck(tmp_path), INCLUDE_ERR, is_ngspice=True, current_mode="kipsa"
            )
            is not None
        )

    def test_no_fire_when_not_ngspice(self, tmp_path: Path):
        assert (
            ngbehavior_lib_hint(
                self._deck(tmp_path), INCLUDE_ERR, is_ngspice=False, current_mode="kiltpsa"
            )
            is None
        )

    def test_no_fire_when_mode_is_safe(self, tmp_path: Path):
        # A mode with neither lt nor ps parses the section fine — don't suggest a fix.
        assert (
            ngbehavior_lib_hint(
                self._deck(tmp_path), INCLUDE_ERR, is_ngspice=True, current_mode="hsa"
            )
            is None
        )

    def test_no_fire_without_include_error(self, tmp_path: Path):
        assert (
            ngbehavior_lib_hint(
                self._deck(tmp_path),
                "Error: singular matrix",
                is_ngspice=True,
                current_mode="kiltpsa",
            )
            is None
        )

    def test_no_fire_when_deck_has_no_sectioned_lib(self, tmp_path: Path):
        deck = _write_deck(tmp_path, ".lib /pdk/models.lib\nR1 a b 1k\n.end\n")
        assert (
            ngbehavior_lib_hint(deck, INCLUDE_ERR, is_ngspice=True, current_mode="kiltpsa") is None
        )
