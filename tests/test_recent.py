"""Tests for the global recent-circuits index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ltspice_mcp.lib import recent


@pytest.fixture
def recent_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the recent index to a per-test temp dir."""
    monkeypatch.setenv("LTSPICE_MCP_HOME", str(tmp_path / "ltspice-mcp-home"))
    return tmp_path / "ltspice-mcp-home"


class TestIndexPath:
    def test_env_override_respected(self, recent_home: Path) -> None:
        assert recent.index_path() == recent_home / "recent.json"


class TestTouch:
    def test_touch_circuit_file(self, tmp_path: Path, recent_home: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        recent.touch(circuit)
        data = json.loads(recent.index_path().read_text())
        assert len(data["circuits"]) == 1
        assert Path(data["circuits"][0]["path"]) == circuit.resolve()
        assert data["circuits"][0]["last_touched"]  # ISO string

    def test_touch_ignores_non_circuit(self, tmp_path: Path, recent_home: Path) -> None:
        not_a_circuit = tmp_path / "readme.txt"
        not_a_circuit.write_text("")
        recent.touch(not_a_circuit)
        assert not recent.index_path().exists()

    def test_touch_bumps_to_top(self, tmp_path: Path, recent_home: Path) -> None:
        a = tmp_path / "a.cir"
        b = tmp_path / "b.cir"
        a.write_text("")
        b.write_text("")
        recent.touch(a)
        recent.touch(b)
        recent.touch(a)

        entries = recent.load()
        # Most recent first: a, b
        paths = [Path(e["path"]) for e in entries]
        assert paths == [a.resolve(), b.resolve()]

    def test_touch_respects_cap(self, tmp_path: Path, recent_home: Path) -> None:
        for i in range(5):
            p = tmp_path / f"c{i}.cir"
            p.write_text("")
            recent.touch(p, cap=3)
        entries = recent.load()
        assert len(entries) == 3
        # Newest three: c4, c3, c2
        names = [Path(e["path"]).name for e in entries]
        assert names == ["c4.cir", "c3.cir", "c2.cir"]


class TestLoad:
    def test_empty_when_no_index(self, recent_home: Path) -> None:
        assert recent.load() == []

    def test_prune_missing_drops_deleted_files(self, tmp_path: Path, recent_home: Path) -> None:
        alive = tmp_path / "alive.cir"
        dead = tmp_path / "dead.cir"
        alive.write_text("")
        dead.write_text("")
        recent.touch(alive)
        recent.touch(dead)
        dead.unlink()

        entries = recent.load(prune_missing=True)
        assert len(entries) == 1
        assert Path(entries[0]["path"]) == alive.resolve()

        # Disk index is rewritten without the missing entry.
        on_disk = json.loads(recent.index_path().read_text())
        assert len(on_disk["circuits"]) == 1

    def test_corrupt_index_returns_empty(self, recent_home: Path) -> None:
        recent_home.mkdir(parents=True, exist_ok=True)
        (recent_home / "recent.json").write_text("{not valid")
        assert recent.load() == []


class TestIsCircuitFile:
    @pytest.mark.parametrize("name", ["lna.asc", "rc.cir", "tb.net", "a.sp", "x.spice"])
    def test_recognised_extensions(self, name: str) -> None:
        assert recent.is_circuit_file(Path(name))

    @pytest.mark.parametrize("name", ["readme.txt", "notes.md", "raw.bin", "sim.log"])
    def test_rejected_extensions(self, name: str) -> None:
        assert not recent.is_circuit_file(Path(name))


class TestSchemaVersion:
    def test_write_includes_schema_header(self, recent_home: Path, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        recent.touch(circuit)
        on_disk = json.loads(recent.index_path().read_text())
        assert on_disk["schema"] == recent.SCHEMA
        assert on_disk["schema_version"] == recent.SCHEMA_VERSION

    def test_read_rejects_versionless_file(self, recent_home: Path, tmp_path: Path) -> None:
        recent_home.mkdir(parents=True, exist_ok=True)
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        (recent_home / "recent.json").write_text(
            json.dumps({"circuits": [{"path": str(circuit.resolve()), "last_touched": None}]})
        )
        assert recent.load() == []

    def test_read_rejects_future_version(self, recent_home: Path) -> None:
        recent_home.mkdir(parents=True, exist_ok=True)
        (recent_home / "recent.json").write_text(
            json.dumps(
                {
                    "schema": recent.SCHEMA,
                    "schema_version": 999,
                    "circuits": [{"path": "/tmp/x.cir", "last_touched": None}],
                }
            )
        )
        assert recent.load() == []
