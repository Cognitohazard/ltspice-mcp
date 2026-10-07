"""The reference documents a simulator's vendor installs with it.

From 26.1 LTspice installs a set of Markdown files written for an assistant to
read, in a ``reference`` directory beside its library: the keyboard shortcuts,
the menus, the schematic file format, ``.MEAS``, the waveform viewer,
troubleshooting. They are the vendor's own account of the program, which the
guide packaged with this server is not: the guide is about using this server
and about what goes wrong in a deck. LTspice's own MCP server hands these
files to a model, and this module is how this server does.

They are read from the install at the time of asking and nothing of them is
packaged here: they are the vendor's, and the installed copy is the one that
describes the installed build. A document is named by its file name, and a
name is only ever looked up among the files the directory lists, so it cannot
reach outside it.

A document is served in sections, cut at its second-level headings, so that a
long one (the schematic reference is about sixty thousand characters) can be
paged at a place where a reader would stop anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ltspice_mcp.lib.guide import split_front_matter

_DIRECTORY = "reference"
_SUFFIX = ".md"
_SECTION_MARK = "## "
_FENCES = ("```", "~~~")


@dataclass(frozen=True)
class Document:
    """One reference file: what it is called and what it says it is for."""

    name: str
    title: str
    description: str
    path: Path


@dataclass(frozen=True)
class Section:
    """A document's text from one second-level heading to the next.

    ``heading`` is the heading's own text, and the document's title for what
    comes before the first one. ``text`` keeps the heading line.
    """

    heading: str
    text: str


def reference_directory(library_roots: list[Path]) -> Path | None:
    """The directory of reference documents beside one of ``library_roots``.

    LTspice keeps it next to its library (``…/LTspice/lib`` and
    ``…/LTspice/reference``). None for a build that installs none: LTspice
    before 26.1, and every other simulator.
    """
    for root in library_roots:
        candidate = root.parent / _DIRECTORY
        if candidate.is_dir():
            return candidate
    return None


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8-sig", errors="replace").replace("\r\n", "\n")


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _listed(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.suffix.lower() == _SUFFIX and path.is_file()
        ),
        key=lambda path: path.name.casefold(),
    )


def _document(path: Path) -> tuple[Document, str]:
    """``path`` as a document, and its text after the front matter. One read."""
    fields, body = split_front_matter(_read(path))
    document = Document(
        name=path.name,
        title=fields.get("title") or _first_heading(body) or path.stem,
        description=fields.get("description", ""),
        path=path,
    )
    return document, body


def names(directory: Path) -> list[str]:
    """The name of every reference document in ``directory``. Reads no file."""
    return [path.name for path in _listed(directory)]


def documents(directory: Path) -> list[Document]:
    """Every reference document in ``directory``, by name. Reads each file."""
    return [_document(path)[0] for path in _listed(directory)]


def read(directory: Path, name: str) -> tuple[Document, list[Section]] | None:
    """The document called ``name`` in ``directory``, cut at its second-level headings.

    ``name`` is taken with or without its suffix and looked up only among the
    files the directory lists; None when it lists none of that name. Reads
    that one file, once. The front matter is left out, and a heading inside a
    fenced code block is code, not a heading.
    """
    wanted = name.strip().casefold()
    path = next(
        (
            listed
            for listed in _listed(directory)
            if wanted in (listed.name.casefold(), listed.stem.casefold())
        ),
        None,
    )
    if path is None:
        return None
    document, body = _document(path)
    cut: list[Section] = []
    heading = document.title
    lines: list[str] = []
    fenced = False

    def close() -> None:
        text = "\n".join(lines).strip("\n")
        if text:
            cut.append(Section(heading=heading, text=text))

    for line in body.split("\n"):
        if line.lstrip().startswith(_FENCES):
            fenced = not fenced
        if not fenced and line.startswith(_SECTION_MARK):
            close()
            heading = line[len(_SECTION_MARK) :].strip()
            lines = []
        lines.append(line)
    close()
    return document, cut
