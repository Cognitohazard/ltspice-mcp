"""Native source identity refuses drift before deriving a statistical sample."""

from pathlib import Path, PureWindowsPath

import pytest

from ltspice_mcp.lib.deck_staging import stage_deck
from ltspice_mcp.lib.hierarchy import SemanticProfile
from ltspice_mcp.lib.native_inputs import NativeCaseValidator, _bench_identities
from ltspice_mcp.lib.pdk_native import PROFILE, NativeCaseError, NativeRequest
from ltspice_mcp.lib.variations import (
    CircuitDeck,
    DeckFile,
    expand_variations,
    materialize_variants,
)


@pytest.mark.parametrize("changed", ["root", "include"])
def test_native_capture_refuses_changed_materialized_bytes(tmp_path: Path, changed: str):
    source = tmp_path / "source"
    source.mkdir()
    bench = source / "bench.cir"
    bench.write_text('* bench\n.include "load.inc"\nV1 in 0 1\n.op\n.end\n', encoding="utf-8")
    (source / "load.inc").write_bytes(b"* resistance in \xb5ohm\r\nR1 in 0 100k\r\n")
    staged = stage_deck(bench, tmp_path / "stage", [source], origin=bench)
    deck = CircuitDeck(
        "bench",
        staged.staged_deck,
        staged.text,
        includes=tuple(
            DeckFile(item.staged_path, item.text, item.sha256) for item in staged.includes
        ),
        semantic_profile=SemanticProfile("ngspice", "hsa"),
        record_source_lineage=True,
    )
    expanded = expand_variations([deck], [], max_cases=1)
    variant = materialize_variants(deck, expanded, staged.staged_deck.parent)[0]
    path = variant.path if changed == "root" else staged.includes[0].staged_path
    path.write_bytes(
        path.read_bytes().replace(b"100k", b"200k")
        if changed == "include"
        else b"* changed\n.op\n.end\n"
    )
    with pytest.raises(NativeCaseError) as caught:
        NativeCaseValidator(staged, tmp_path / "stage").validate(
            NativeRequest("bench", "samples", PROFILE, "nominal", 1, 0), variant, expanded[0]
        )
    assert caught.value.code == "artifact_drift"


def test_native_bench_identities_span_windows_volumes_and_survive_relocation():
    original = [PureWindowsPath("C:/circuits/bench.cir"), PureWindowsPath("D:/shared/load.inc")]
    relocated = [PureWindowsPath("Z:/work/bench.cir"), PureWindowsPath("E:/libraries/load.inc")]
    expected = ["bench/root-0/bench.cir", "bench/root-1/load.inc"]
    assert list(_bench_identities(original).values()) == expected
    assert list(_bench_identities(relocated).values()) == expected


def test_native_bench_identity_keeps_existing_single_volume_layout():
    paths = [PureWindowsPath("C:/circuits/bench.cir"), PureWindowsPath("c:/circuits/lib/load.inc")]
    assert list(_bench_identities(paths).values()) == ["bench/bench.cir", "bench/lib/load.inc"]


def test_circuit_original_cache_reuses_pins_and_cards_for_same_capture_set(tmp_path, monkeypatch):
    from ltspice_mcp.lib import pdk_native
    from ltspice_mcp.lib.native_inputs import NativeCaseValidator

    original = pdk_native.OriginalCapture("bench/main.cir", b"R1 in 0 1k\n.op\n.end\n")
    changed = pdk_native.OriginalCapture("bench/main.cir", b"R1 in 0 2k\n.op\n.end\n")
    calls = {"pins": 0, "lex": 0}
    pins = pdk_native.profile_pins
    lex = pdk_native.lex

    def counted_pins():
        calls["pins"] += 1
        return pins()

    def counted_lex(text):
        calls["lex"] += 1
        return lex(text)

    monkeypatch.setattr(pdk_native, "profile_pins", counted_pins)
    monkeypatch.setattr(pdk_native, "lex", counted_lex)
    bench = tmp_path / "bench.cir"
    bench.write_bytes(original.content)
    staged = stage_deck(bench, tmp_path / "stage", [tmp_path], origin=bench)
    validator = NativeCaseValidator(staged, tmp_path / "stage")
    first = validator._verified_originals((original,))
    assert (
        validator._verified_originals(
            (pdk_native.OriginalCapture(*(original.identity, original.content)),)
        )
        is first
    )
    second = validator._verified_originals((changed,))
    assert second is not first
    assert second.cards[(changed.identity, 1)].body.endswith("2k")
    expanded = validator._verified_originals(
        (original, pdk_native.OriginalCapture("bench/other.cir", original.content))
    )
    assert expanded is not first
    assert set(expanded.captures) == {"bench/main.cir", "bench/other.cir"}
    assert calls == {"pins": 3, "lex": 4}
