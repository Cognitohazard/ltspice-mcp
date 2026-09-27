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
