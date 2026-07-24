"""Dependency-manifest staging and platform routing."""

from __future__ import annotations

from pathlib import Path

import pytest
from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.lib import deck_staging, wsl
from ltspice_mcp.lib.deck_staging import (
    DeckStagingError,
    resolve_experiment_paths,
    stage_deck,
    verify_staged_manifest,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class TestManifestWalk:
    def test_walks_three_include_levels(self, tmp_path: Path):
        root = tmp_path / "root"
        deck = _write(root / "deck.cir", '.include "a.inc"\n.op\n.end\n')
        _write(root / "a.inc", '.include "b.inc"\n')
        _write(root / "b.inc", '.include "c.inc"\n')
        _write(root / "c.inc", ".param x=1\n")

        staged = stage_deck(deck, tmp_path / "stage", [root])

        assert {entry.path.name for entry in staged.manifest} == {
            "deck.cir",
            "a.inc",
            "b.inc",
            "c.inc",
        }

    def test_outside_allowed_root_fails_closed(self, tmp_path: Path):
        root = tmp_path / "root"
        outside = _write(tmp_path / "outside.inc", ".param x=1\n")
        deck = _write(root / "deck.cir", f'.include "{outside}"\n.op\n.end\n')

        with pytest.raises(DeckStagingError, match="outside allowed roots"):
            stage_deck(deck, tmp_path / "stage", [root])

    def test_allow_live_include_marks_manifest_and_observation(self, tmp_path: Path):
        root = tmp_path / "root"
        outside = _write(tmp_path / "outside.inc", ".param x=1\n")
        deck = _write(root / "deck.cir", f'.include "{outside}"\n.op\n.end\n')

        staged = stage_deck(
            deck,
            tmp_path / "stage",
            [root],
            allow_live_includes=True,
        )

        live = [entry for entry in staged.manifest if entry.live]
        assert len(live) == 1
        assert live[0].path == outside.resolve()
        assert any(item["code"] == "live_include" for item in staged.observations)

    def test_relative_topology_is_preserved(self, tmp_path: Path):
        root = tmp_path / "root"
        deck = _write(root / "deck.cir", '.include "models/device.lib"\n.op\n.end\n')
        _write(root / "models" / "device.lib", ".model DFAST D(Is=1e-12)\n")

        staged = stage_deck(deck, tmp_path / "stage", [root])

        assert (staged.staged_deck.parent / "models" / "device.lib").is_file()
        assert '.include "models/device.lib"' in staged.staged_deck.read_text()
        assert staged.text == staged.staged_deck.read_text()

    def test_lib_section_is_recorded(self, tmp_path: Path):
        root = tmp_path / "root"
        deck = _write(root / "deck.cir", '.lib "models.lib" TT\n.op\n.end\n')
        _write(root / "models.lib", ".lib TT\n.model DFAST D\n.endl TT\n")

        staged = stage_deck(deck, tmp_path / "stage", [root])

        assert any(
            entry.path.name == "models.lib" and entry.section == "TT" for entry in staged.manifest
        )

    def test_include_cycle_is_bounded(self, tmp_path: Path):
        root = tmp_path / "root"
        deck = _write(root / "deck.cir", '.include "a.inc"\n.op\n.end\n')
        _write(root / "a.inc", '.include "deck.cir"\n')

        staged = stage_deck(deck, tmp_path / "stage", [root])

        assert [entry.path.name for entry in staged.manifest].count("deck.cir") == 1
        assert [entry.path.name for entry in staged.manifest].count("a.inc") == 1

    def test_symlink_escape_is_rejected(self, tmp_path: Path):
        root = tmp_path / "root"
        outside = _write(tmp_path / "outside.inc", ".param x=1\n")
        root.mkdir()
        (root / "escape.inc").symlink_to(outside)
        deck = _write(root / "deck.cir", '.include "escape.inc"\n.op\n.end\n')

        with pytest.raises(DeckStagingError, match="outside allowed roots"):
            stage_deck(deck, tmp_path / "stage", [root])

    def test_duplicate_includes_copy_once(self, tmp_path: Path):
        root = tmp_path / "root"
        deck = _write(
            root / "deck.cir",
            '.include "shared.inc"\n.include "shared.inc"\n.op\n.end\n',
        )
        shared = _write(root / "shared.inc", ".param x=1\n")

        staged = stage_deck(deck, tmp_path / "stage", [root])

        assert sum(entry.path == shared.resolve() for entry in staged.manifest) == 1

    def test_quoted_and_windows_path_tokens_are_supported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        root = tmp_path / "root"
        quoted = _write(root / "models with spaces" / "quoted.lib", ".model DQ D\n")
        windows = _write(root / "windows.lib", ".model DW D\n")
        deck = _write(
            root / "deck.cir",
            '.include "models with spaces/quoted.lib"\n'
            '.include "C:\\vendor\\windows.lib"\n'
            ".op\n.end\n",
        )
        original = deck_staging._resolve_reference

        def resolve(parent: Path, raw: str) -> Path:
            if raw.startswith("C:\\"):
                return windows
            return original(parent, raw)

        monkeypatch.setattr(deck_staging, "_resolve_reference", resolve)

        staged = stage_deck(deck, tmp_path / "stage", [root])

        assert {quoted.resolve(), windows.resolve()} <= {entry.path for entry in staged.manifest}

    def test_source_change_after_manifest_does_not_change_staged_copy(self, tmp_path: Path):
        root = tmp_path / "root"
        deck = _write(root / "deck.cir", '.include "value.inc"\n.op\n.end\n')
        included = _write(root / "value.inc", ".param x=1\n")
        staged = stage_deck(deck, tmp_path / "stage", [root])
        staged_include = next(
            entry.staged_path for entry in staged.manifest if entry.path == included.resolve()
        )
        included.write_text(".param x=2\n")

        observations = verify_staged_manifest(staged.manifest)

        assert staged_include is not None
        assert staged_include.read_text() == ".param x=1\n"
        assert any(item["code"] == "source_modified_after_staging" for item in observations)


class TestPlatformRouting:
    def test_wsl_ltspice_uses_windows_native_staging_and_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        windows = tmp_path / "mnt" / "c" / "temp"
        windows.mkdir(parents=True)
        monkeypatch.setattr(wsl, "is_wsl", lambda: True)
        monkeypatch.setattr(wsl, "get_windows_output_dir", lambda: windows)

        paths = resolve_experiment_paths(tmp_path, "exp1", "dut", LTspice)

        assert paths.staging_root.is_relative_to(windows)
        assert paths.output_folder.is_relative_to(windows)

    def test_wsl_ltspice_without_windows_dir_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(wsl, "is_wsl", lambda: True)
        monkeypatch.setattr(wsl, "get_windows_output_dir", lambda: None)

        with pytest.raises(DeckStagingError, match="Windows-native"):
            resolve_experiment_paths(tmp_path, "exp1", "dut", LTspice)

    def test_wsl_ngspice_keeps_linux_staging(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(wsl, "is_wsl", lambda: True)
        monkeypatch.setattr(
            wsl,
            "get_windows_output_dir",
            lambda: pytest.fail("ngspice must not request a Windows temp directory"),
        )

        paths = resolve_experiment_paths(
            tmp_path,
            "exp1",
            "dut",
            NGspiceSimulator,
        )

        assert paths.staging_root == (
            tmp_path / ".ltspice-mcp" / "jobs" / "exp1" / "staged" / "dut"
        )
