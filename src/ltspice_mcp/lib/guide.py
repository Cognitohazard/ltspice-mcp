"""The agent-facing guide: a core, then topic sections and task playbooks.

The guide is what a model reads before it works here, so it is split by how it
is read. The core (``assets/guide/core.md``) is short enough to read every
session: how to work with the server, Python or tools, the rules that cause
silent errors. The files beside it are read when a task needs them. Each one
says in its front matter what it is: a ``topic`` section holds reference
material (a simulator's syntax, the tools' arguments), and a ``task`` playbook
walks one kind of job. The core ends with an index of both, generated from that
front matter, so a model picks what to read by its description, and a new file
is listed by adding it and its name to ``SECTION_ORDER``.

Every door serves this one module: ``inspect(kind="guide")`` over MCP,
``Api.guide()`` in Python, and the ``spice://guide`` resources. A section is
named by its file stem (``ltspice``). The Claude Code plugin's one skill only
points a session here, so there is no second copy of any of this text to drift.

Nothing here imports beyond the standard library, so
``python -m ltspice_mcp.api guide`` reads the guide without starting the
engine.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable

#: The sections, in the order the index lists them within their kind. A test
#: pins this to the files present, both ways.
SECTION_ORDER: tuple[str, ...] = (
    "python",
    "tools",
    "signals",
    "variations",
    "fundamentals",
    "ltspice",
    "ngspice",
    "schematics",
    "operating-points",
    "rf",
    "hierarchy",
    "sky130",
    "bench-craft",
)

#: What a section's front matter may name as its ``kind``, in index order, with
#: the heading the index groups it under. A section that names none is a topic.
KINDS: dict[str, str] = {
    "topic": "Topics — reference for a simulator, the tools or a technique",
    "task": "Tasks — playbooks that walk one kind of job",
}


class UnknownGuideSection(ValueError):
    """A section name the guide does not have; the message lists the ones it does."""

    def __init__(self, name: str, known: tuple[str, ...]) -> None:
        super().__init__(
            f"unknown guide section {name!r}; omit 'section' for the core and its index, "
            f"or name one of: {', '.join(known)}"
        )
        self.name = name
        self.known = known


@dataclass(frozen=True)
class GuideEntry:
    """One section of the guide: a topic or a task playbook."""

    name: str
    kind: str
    title: str
    description: str
    body: str


def _guide_dir() -> Traversable:
    return files("ltspice_mcp") / "assets" / "guide"


def _read(node: Traversable) -> str:
    return node.read_text(encoding="utf-8").replace("\r\n", "\n")


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---`` front matter from a Markdown body.

    Reads the subset the guide uses: ``key: value`` lines, and a folded
    ``key: >`` block whose indented lines join with spaces. Text with no front
    matter returns an empty mapping and the text unchanged.
    """
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    fields: dict[str, str] = {}
    key: str | None = None
    for line in text[4:end].splitlines():
        if line.startswith((" ", "\t")) and key is not None:
            fields[key] = f"{fields[key]} {line.strip()}".strip()
            continue
        name, sep, value = line.partition(":")
        if not sep:
            continue
        key = name.strip()
        value = value.strip()
        fields[key] = "" if value in (">", "|", ">-", "|-") else value
    return fields, text[end + len("\n---\n") :].lstrip("\n")


def _title(body: str, fallback: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _entry(name: str, text: str) -> GuideEntry:
    fields, body = split_front_matter(text)
    return GuideEntry(
        name=name,
        kind=fields.get("kind", "topic"),
        title=_title(body, name),
        description=fields.get("description", ""),
        body=body.rstrip("\n") + "\n",
    )


@functools.cache
def sections() -> tuple[GuideEntry, ...]:
    """Every section, in ``SECTION_ORDER``."""
    directory = _guide_dir()
    return tuple(_entry(name, _read(directory / f"{name}.md")) for name in SECTION_ORDER)


def names() -> tuple[str, ...]:
    """Every name ``read`` accepts."""
    return tuple(entry.name for entry in sections())


def _index() -> str:
    lines = ["## Index", "", "Pass a name as `section` to read it."]
    for kind, heading in KINDS.items():
        entries = [entry for entry in sections() if entry.kind == kind]
        if entries:
            lines.extend(["", f"{heading}:", ""])
            lines.extend(f"- `{entry.name}`: {entry.description}" for entry in entries)
    return "\n".join(lines) + "\n"


@functools.cache
def core() -> str:
    """The core and the generated index: what a session reads first."""
    return _read(_guide_dir() / "core.md").rstrip("\n") + "\n\n" + _index()


def _find(section: str) -> GuideEntry:
    for entry in sections():
        if entry.name == section:
            return entry
    raise UnknownGuideSection(section, names())


def read(section: str | None = None) -> str:
    """The core with its index, or one section by name.

    Raises :class:`UnknownGuideSection` for a name the guide does not have.
    """
    return core() if section is None else _find(section).body


def title_of(section: str | None) -> str:
    """The heading a read of ``section`` opens with."""
    return _title(core(), "SPICE simulation guide") if section is None else _find(section).title
