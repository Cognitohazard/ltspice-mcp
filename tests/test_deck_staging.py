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

        staged = stage_deck(deck, tmp_path / "stage", [root], origin=deck)

        assert {entry.path.name for entry in staged.manifest} == {
            "deck.cir",
            "a.inc",
            "b.inc",
            "c.inc",
        }

    def test_origin_is_snapshotted_and_digested_beside_the_deck(self, tmp_path: Path):
        """The file the deck was generated from is a source like any other.

        A netlist exported from a schematic describes the schematic only until
        the schematic is edited, and the export on disk does not change when it
        is. Recording the origin is what lets a later reader tell the two apart.
        """
        root = tmp_path / "root"
        schematic = _write(root / "amp.asc", "Version 4\nSYMATTR Value 1k\n")
        deck = _write(root / "amp.net", "V1 in 0 1\nR1 in 0 1k\n.op\n.end\n")

        staged = stage_deck(deck, tmp_path / "stage", [root], origin=schematic)

        entry = next(item for item in staged.manifest if item.path == schematic.resolve())
        assert entry.staged and not entry.live
        assert entry.sha256 == deck_staging.sha256_file(schematic)
        assert entry.staged_path is not None
        assert entry.staged_path.read_bytes() == schematic.read_bytes()
        assert staged.origin_sha256 == entry.sha256
        # The origin is not part of the deck the simulator reads, so it must not
        # arrive as an include a variation would try to edit.
        assert schematic not in {included.source for included in staged.includes}

        schematic.write_text("Version 4\nSYMATTR Value 2k\n")
        assert [item["code"] for item in verify_staged_manifest(staged.manifest)] == [
            "source_modified_after_staging"
        ]

    def test_origin_equal_to_the_deck_records_one_entry(self, tmp_path: Path):
        """A hand-written deck is its own origin; it must not be listed twice."""
        root = tmp_path / "root"
        deck = _write(root / "deck.cir", ".op\n.end\n")

        staged = stage_deck(deck, tmp_path / "stage", [root], origin=deck)

        assert [entry.path for entry in staged.manifest] == [deck.resolve()]
        assert staged.origin_sha256 == staged.sha256

    def test_outside_allowed_root_fails_closed(self, tmp_path: Path):
        root = tmp_path / "root"
        outside = _write(tmp_path / "outside.inc", ".param x=1\n")
        deck = _write(root / "deck.cir", f'.include "{outside}"\n.op\n.end\n')

        with pytest.raises(DeckStagingError, match="outside allowed roots"):
            stage_deck(deck, tmp_path / "stage", [root], origin=deck)

    def test_allow_live_include_marks_manifest_and_observation(self, tmp_path: Path):
        root = tmp_path / "root"
        outside = _write(tmp_path / "outside.inc", ".param x=1\n")
        deck = _write(root / "deck.cir", f'.include "{outside}"\n.op\n.end\n')

        staged = stage_deck(
            deck,
            tmp_path / "stage",
            [root],
            origin=deck,
            allow_live_includes=True,
        )

        live = [entry for entry in staged.manifest if entry.live]
        assert len(live) == 1
        assert live[0].path == outside.resolve()
        assert any(item["code"] == "live_include" for item in staged.observations)

    def test_relative_topology_is_preserved(self, tmp_path: Path):
        """The staged tree mirrors the source layout, and files INSIDE it keep
        referring to each other relatively so the bundle stays relocatable.

        The root deck is the exception: it is copied out to the simulator's
        output folder without its siblings, so its own references are absolute
        (proved necessary by a run that failed with "File not found" on exactly
        this shape)."""
        root = tmp_path / "root"
        deck = _write(root / "deck.cir", '.include "models/device.lib"\n.op\n.end\n')
        _write(root / "models" / "device.lib", '.include "params.inc"\n.model DFAST D\n')
        _write(root / "models" / "params.inc", ".param x=1\n")

        staged = stage_deck(deck, tmp_path / "stage", [root], origin=deck)

        staged_lib = staged.staged_deck.parent / "models" / "device.lib"
        assert staged_lib.is_file()
        assert staged.text == staged.staged_deck.read_text()

        reference = staged.staged_deck.read_text().splitlines()[0].split()[1].strip('"')
        assert Path(reference).is_absolute()
        assert Path(reference).resolve() == staged_lib.resolve()

        # A file deeper in the bundle keeps its relative reference.
        assert '"params.inc"' in staged_lib.read_text()

    def test_lib_section_is_recorded(self, tmp_path: Path):
        root = tmp_path / "root"
        deck = _write(root / "deck.cir", '.lib "models.lib" TT\n.op\n.end\n')
        _write(root / "models.lib", ".lib TT\n.model DFAST D\n.endl TT\n")

        staged = stage_deck(deck, tmp_path / "stage", [root], origin=deck)

        assert any(
            entry.path.name == "models.lib" and entry.section == "TT" for entry in staged.manifest
        )

    def test_include_cycle_is_bounded(self, tmp_path: Path):
        root = tmp_path / "root"
        deck = _write(root / "deck.cir", '.include "a.inc"\n.op\n.end\n')
        _write(root / "a.inc", '.include "deck.cir"\n')

        staged = stage_deck(deck, tmp_path / "stage", [root], origin=deck)

        assert [entry.path.name for entry in staged.manifest].count("deck.cir") == 1
        assert [entry.path.name for entry in staged.manifest].count("a.inc") == 1

    def test_symlink_escape_is_rejected(self, tmp_path: Path):
        root = tmp_path / "root"
        outside = _write(tmp_path / "outside.inc", ".param x=1\n")
        root.mkdir()
        (root / "escape.inc").symlink_to(outside)
        deck = _write(root / "deck.cir", '.include "escape.inc"\n.op\n.end\n')

        with pytest.raises(DeckStagingError, match="outside allowed roots"):
            stage_deck(deck, tmp_path / "stage", [root], origin=deck)

    def test_duplicate_includes_copy_once(self, tmp_path: Path):
        root = tmp_path / "root"
        deck = _write(
            root / "deck.cir",
            '.include "shared.inc"\n.include "shared.inc"\n.op\n.end\n',
        )
        shared = _write(root / "shared.inc", ".param x=1\n")

        staged = stage_deck(deck, tmp_path / "stage", [root], origin=deck)

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
        original = deck_staging.resolve_reference

        def resolve(parent: Path, raw: str) -> Path:
            if raw.startswith("C:\\"):
                return windows
            return original(parent, raw)

        monkeypatch.setattr(deck_staging, "resolve_reference", resolve)

        staged = stage_deck(deck, tmp_path / "stage", [root], origin=deck)

        assert {quoted.resolve(), windows.resolve()} <= {entry.path for entry in staged.manifest}

    def test_source_change_after_manifest_does_not_change_staged_copy(self, tmp_path: Path):
        root = tmp_path / "root"
        deck = _write(root / "deck.cir", '.include "value.inc"\n.op\n.end\n')
        included = _write(root / "value.inc", ".param x=1\n")
        staged = stage_deck(deck, tmp_path / "stage", [root], origin=deck)
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


class TestFoundryPdkShapes:
    """Staging shapes that only real foundry PDKs exercise.

    Modelled on sky130/gf180: a sectioned corner library named ``*.lib.spice``,
    a device model five include levels below the deck, and the library living
    under a different allowed root than the deck.
    """

    @staticmethod
    def _pdk(tmp_path: Path) -> tuple[Path, Path]:
        pdk = tmp_path / "pdk"
        # Corner sections declared with a bare ``.lib <name>`` inside a file
        # whose final suffix is ``.spice`` — the near-universal PDK naming.
        _write(
            pdk / "models.lib.spice",
            '.lib tt\n.include "corners/tt.spice"\n.endl\n'
            '.lib ss\n.include "corners/ss.spice"\n.endl\n',
        )
        _write(pdk / "corners" / "tt.spice", '.include "../devices/nfet__tt.corner.spice"\n')
        _write(pdk / "corners" / "ss.spice", '.include "../devices/nfet__ss.corner.spice"\n')
        _write(pdk / "devices" / "nfet__tt.corner.spice", '.include "nfet.pm3.spice"\n')
        _write(pdk / "devices" / "nfet__ss.corner.spice", '.include "nfet.pm3.spice"\n')
        _write(pdk / "devices" / "nfet.pm3.spice", ".model nfet nmos level=8\n")
        return pdk, pdk / "models.lib.spice"

    def test_sectioned_library_declarations_are_not_files(self, tmp_path: Path):
        """``.lib tt`` inside a ``*.lib.spice`` declares a section. Reading it
        as a filename fails every real PDK corner selection."""
        root = tmp_path / "root"
        pdk, lib = self._pdk(tmp_path)
        deck = _write(root / "deck.cir", f'.lib "{lib}" tt\nM1 d g s b nfet\n.op\n.end\n')

        staged = stage_deck(deck, tmp_path / "stage", [root, pdk], origin=deck)

        names = {entry.path.name for entry in staged.manifest}
        assert "models.lib.spice" in names
        assert "nfet.pm3.spice" in names, "did not reach the device model five levels down"
        assert not any(entry.path.name in {"tt", "ss"} for entry in staged.manifest)

    def test_root_deck_reference_survives_relocation(self, tmp_path: Path):
        """The staged deck is handed to the simulator, which runs it from its
        own output folder — so the root deck's rewritten reference has to
        resolve from anywhere, not just from the staging directory."""
        root = tmp_path / "root"
        pdk, lib = self._pdk(tmp_path)
        deck = _write(root / "deck.cir", f'.lib "{lib}" tt\n.op\n.end\n')

        staged = stage_deck(deck, tmp_path / "stage", [root, pdk], origin=deck)

        text = staged.staged_deck.read_text()
        reference = text.splitlines()[0].split()[1].strip('"')
        assert Path(reference).is_absolute(), (
            f"root deck kept a relocatable-only path: {reference}"
        )
        assert Path(reference).exists()

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        moved = elsewhere / "run.cir"
        moved.write_text(text)
        assert (moved.parent / reference).exists(), "reference broke when the deck moved"

    def test_default_depth_reaches_pdk_device_models(self, tmp_path: Path):
        """sky130 puts a device model five levels below the deck, and a bound
        too small for it does not truncate the manifest — it refuses the deck.
        So the default has to clear that chain, and the way to see that it does
        is that the same deck under a smaller bound fails."""
        root = tmp_path / "root"
        pdk, lib = self._pdk(tmp_path)
        deck = _write(root / "deck.cir", f'.lib "{lib}" tt\n.op\n.end\n')

        staged = stage_deck(deck, tmp_path / "stage", [root, pdk], origin=deck)

        assert "nfet.pm3.spice" in {entry.path.name for entry in staged.manifest}

        with pytest.raises(DeckStagingError, match="exceeds depth 3"):
            stage_deck(deck, tmp_path / "shallow", [root, pdk], max_depth=3, origin=deck)


class TestRootDeckSurvivesTheRun:
    """The staged root deck is handed to the simulator and run from the shared
    output folder, leaving its siblings behind. Every reference it carries has
    to resolve from there — including a plain same-directory include, which is
    already correct inside the staging tree and so is easy to leave alone."""

    def test_sibling_include_is_absolute_in_the_root_deck(self, tmp_path: Path):
        root = tmp_path / "root"
        _write(root / "core.inc", "R1 in out 1k\n")
        deck = _write(root / "tb.cir", "V1 in 0 1\n.include core.inc\n.op\n.end\n")

        staged = stage_deck(deck, tmp_path / "stage", [root], origin=deck)

        reference = staged.staged_deck.read_text().splitlines()[1].split()[1].strip('"')
        assert Path(reference).is_absolute(), (
            f"root deck kept a staging-relative reference: {reference}"
        )
        assert Path(reference).exists()

        # The move the runner actually performs.
        elsewhere = tmp_path / "runs"
        elsewhere.mkdir()
        moved = elsewhere / "case_0.cir"
        moved.write_text(staged.staged_deck.read_text())
        assert (moved.parent / reference).exists(), "include broke when the deck moved"

    def test_windows_paths_render_for_a_windows_simulator(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A Windows simulator reached across the WSL boundary cannot open the
        /mnt/c spelling of the file it is handed.

        The conversion itself belongs to the WSL interop layer and depends on
        the host, so it is replaced with a known one here: what this pins is
        that the staged deck names the staged file THROUGH that conversion, and
        names the whole path — a reference that survives the deck being run from
        another directory.
        """
        monkeypatch.setattr(wsl, "to_windows_path", lambda path: f"Z:{path}".replace("/", "\\"))
        root = tmp_path / "root"
        _write(root / "core.inc", "R1 in out 1k\n")
        deck = _write(root / "tb.cir", "V1 in 0 1\n.include core.inc\n.op\n.end\n")

        staged = stage_deck(deck, tmp_path / "stage", [root], origin=deck, windows_paths=True)

        reference = staged.staged_deck.read_text().splitlines()[1].split()[1].strip('"')
        included = next(item for item in staged.includes if item.source.name == "core.inc")
        assert reference == wsl.to_windows_path(included.staged_path)
        assert Path(reference.replace("\\", "/").removeprefix("Z:")).is_file()

    def test_posix_paths_render_without_the_windows_routing(self, tmp_path: Path):
        """The default is the path the local simulator can open: the Windows
        rendering is a WSL-boundary special case, not the general spelling."""
        root = tmp_path / "root"
        _write(root / "core.inc", "R1 in out 1k\n")
        deck = _write(root / "tb.cir", "V1 in 0 1\n.include core.inc\n.op\n.end\n")

        staged = stage_deck(deck, tmp_path / "stage", [root], origin=deck)

        reference = staged.staged_deck.read_text().splitlines()[1].split()[1].strip('"')
        included = next(item for item in staged.includes if item.source.name == "core.inc")
        assert reference == included.staged_path.as_posix()
