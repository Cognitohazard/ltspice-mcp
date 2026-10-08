"""inspect(kind="simulator_docs"): the reference documents LTspice installs.

The documents are the vendor's and are read from an install, so the suite
makes a small install of its own: a library directory and, beside it, a
``reference`` directory in the shape LTspice 26.1 writes (Markdown files with
``title`` and ``description`` front matter). The opt-in tier holds the reading
of a real install to what LTspice's own server lists
(``test_ltspice_integration.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import simulator_docs
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import inspect_tools
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from tests.conftest import installed_simulator
from tests.ltspice_bridge_recorder import FIXTURES, load_manifest, recorded_builds

MEAS = """---
title: Measuring Things
description: Examples of the measure statement.
version: "24+"
---

[back](README.md)

# Measuring Things

What a measurement is.

## Transient

```spice
.meas TRAN vmax MAX V(out)
## not a heading: this line is inside a code block
```

## AC

Magnitude and phase.

## Noise

Integrated noise.
"""

SHORTCUTS = """# Keys

No front matter here.

## Schematic

F2 places a component.
"""


def an_install(root: Path, *, with_reference: bool = True) -> type:
    library = root / "LTspice" / "lib"
    library.mkdir(parents=True)
    if with_reference:
        reference = root / "LTspice" / "reference"
        (reference / "images").mkdir(parents=True)
        (reference / "MEAS-REFERENCE.md").write_text(MEAS, encoding="utf-8")
        (reference / "keys.md").write_bytes(SHORTCUTS.replace("\n", "\r\n").encode("utf-8"))
        (reference / "notes.txt").write_text("not a document", encoding="utf-8")
    return installed_simulator(library)


@pytest.fixture
def state(config: ServerConfig, tmp_path_factory: pytest.TempPathFactory) -> SessionState:
    simulator = an_install(tmp_path_factory.mktemp("install"))
    return SessionState.create(config, available={"fake": simulator})


async def ask(state: SessionState, **query: Any) -> dict[str, Any]:
    result = await handle_inspect(
        InspectInput.model_validate({"queries": [{"kind": "simulator_docs", **query}]}), state
    )
    data = result.structured_content
    assert data is not None
    jsonschema.Draft202012Validator(inspect_tools._OUTPUT_SCHEMA).validate(data)
    (item,) = data["results"]
    return item


async def test_with_no_name_it_lists_the_documents(state: SessionState):
    item = await ask(state)

    assert item["ok"] is True
    data = item["data"]
    assert Path(data["source"]).name == "reference"
    # The text file and the images directory are not documents.
    assert data["docs"] == [
        {"name": "keys.md", "title": "Keys", "description": ""},
        {
            "name": "MEAS-REFERENCE.md",
            "title": "Measuring Things",
            "description": "Examples of the measure statement.",
        },
    ]
    assert (data["total"], data["returned"]) == (2, 2)
    assert "Read one with name" in data["hint"]
    assert item["next_cursor"] is None


@pytest.mark.parametrize("name", ["MEAS-REFERENCE.md", "meas-reference", " MEAS-REFERENCE.MD "])
async def test_a_document_comes_back_in_sections_cut_at_its_headings(
    state: SessionState, name: str
):
    item = await ask(state, name=name)

    assert item["ok"] is True
    data = item["data"]
    assert (data["name"], data["title"]) == ("MEAS-REFERENCE.md", "Measuring Things")
    assert [section["heading"] for section in data["sections"]] == [
        "Measuring Things",
        "Transient",
        "AC",
        "Noise",
    ]
    opening, transient, *_rest = data["sections"]
    # The front matter is what the list reports; it is not part of the text.
    assert opening["text"].startswith("[back](README.md)")
    assert "description:" not in opening["text"]
    # A line that looks like a heading inside a code block stays where it is.
    assert transient["text"].startswith("## Transient\n")
    assert "## not a heading" in transient["text"]
    assert data["sections"][-1]["text"] == "## Noise\n\nIntegrated noise."


async def test_a_document_with_windows_line_ends_and_no_front_matter_reads_the_same(
    state: SessionState,
):
    data = (await ask(state, name="keys"))["data"]

    assert data["title"] == "Keys"
    assert data["sections"] == [
        {"heading": "Keys", "text": "# Keys\n\nNo front matter here."},
        {"heading": "Schematic", "text": "## Schematic\n\nF2 places a component."},
    ]


async def test_a_long_document_is_paged_at_a_heading(
    state: SessionState, monkeypatch: pytest.MonkeyPatch
):
    whole = (await ask(state, name="MEAS-REFERENCE.md"))["data"]["sections"]
    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 3)

    first = await ask(state, name="MEAS-REFERENCE.md")
    assert first["data"]["returned"] == 3
    assert first["data"]["total"] == 4
    assert first["next_cursor"] is not None
    rest = await ask(state, name="MEAS-REFERENCE.md", cursor=first["next_cursor"])

    assert rest["next_cursor"] is None
    assert first["data"]["sections"] + rest["data"]["sections"] == whole


async def test_a_cursor_into_one_document_does_not_open_another(
    state: SessionState, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 1)
    first = await ask(state, name="MEAS-REFERENCE.md")

    item = await ask(state, name="keys.md", cursor=first["next_cursor"])

    assert item["ok"] is False
    assert item["error"]["code"] == "invalid_cursor"


async def test_a_name_that_is_not_a_document_is_refused_with_the_names_that_are(
    state: SessionState,
):
    item = await ask(state, name="../lib/standard.mos")

    assert item["ok"] is False
    assert item["error"]["code"] == "unknown_document"
    assert item["error"]["supported"] == ["keys.md", "MEAS-REFERENCE.md"]


async def test_an_install_with_no_reference_directory_says_so(
    config: ServerConfig, tmp_path_factory: pytest.TempPathFactory
):
    simulator = an_install(tmp_path_factory.mktemp("older"), with_reference=False)
    state = SessionState.create(config, available={"fake": simulator})

    item = await ask(state)

    assert item["ok"] is False
    assert item["error"]["code"] == "simulator_docs_unavailable"
    assert "LTspice 26.1 and later" in item["error"]["message"]


@pytest.mark.parametrize("build", recorded_builds())
def test_the_install_this_models_is_the_one_ltspice_was_recorded_writing(build: str):
    """The directory's name, the suffix and the two front matter keys read here
    are what an LTspice 26.1 install holds (names and keys only: the documents
    are the vendor's and are not in the recording)."""
    reference = load_manifest(FIXTURES / build)["reference"]
    assert reference["directory"] == "reference"
    documents = [name for name in reference["files"] if name.endswith(".md")]
    assert len(documents) >= 10
    assert {"title", "description"} <= set(reference["front_matter"])


def test_the_directory_is_the_one_beside_a_library_root(tmp_path: Path):
    an_install(tmp_path)
    library = tmp_path / "LTspice" / "lib"
    assert simulator_docs.reference_directory([tmp_path, library]) == library.parent / "reference"
    assert simulator_docs.reference_directory([tmp_path]) is None
    assert simulator_docs.reference_directory([]) is None
