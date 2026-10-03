"""The agent-facing guide: a core, topic sections, and the task skills.

The guide is what a model reads before it works here, so it is split by how it
is read. The core (``assets/guide/core.md``) is short enough to read every
session: how to work with the server, Python or tools, the rules that cause
silent errors. The topic sections beside it, and the plugin's task skills, are
read when a task needs them; the core ends with an index of both, generated
from their own front matter, so a new section or skill is listed by adding the
file.

Every door serves this one module: ``inspect(kind="guide")`` over MCP,
``Api.guide()`` in Python, and the ``spice://guide`` resources. A section is
named by its file stem (``ltspice``); a skill by ``skill:`` and its directory
(``skill:spice-experiments``), and a further Markdown file inside a skill by its
path below that directory (``skill:spice-bench-craft/references/BENCH_NOTES.md``).

The skills live at the repository root, where the Claude Code plugin loads
them; the wheel carries a copy at ``ltspice_mcp/skills`` (a build-time
force-include), and a source checkout reads the root directly. Nothing here
imports beyond the standard library, so ``python -m ltspice_mcp.api guide``
reads the guide without starting the engine.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

#: The topic sections, in the order the index lists them. A test pins this to
#: the files present, both ways.
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
    "bench-craft",
    "rf",
    "hierarchy",
    "sky130",
)

SKILL_PREFIX = "skill:"

_SKILL_FILE = "SKILL.md"


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
    """One readable unit: a topic section or a task skill."""

    name: str
    title: str
    description: str
    body: str
    #: Further Markdown files a skill carries, as paths below its directory.
    extra_files: tuple[str, ...] = ()


def _guide_dir() -> Traversable:
    return files("ltspice_mcp") / "assets" / "guide"


def _skills_dir() -> Traversable | None:
    """The skills directory: the wheel's packaged copy, else the checkout's root."""
    packaged = files("ltspice_mcp") / "skills"
    if packaged.is_dir():
        return packaged
    # src/ltspice_mcp/lib/guide.py -> the repository root.
    checkout = Path(__file__).resolve().parents[3] / "skills"
    return checkout if checkout.is_dir() else None


def _read(node: Traversable) -> str:
    return node.read_text(encoding="utf-8").replace("\r\n", "\n")


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---`` front matter from a Markdown body.

    Reads the subset the guide and the skills use: ``key: value`` lines, and a
    folded ``key: >`` block whose indented lines join with spaces. Text with no
    front matter returns an empty mapping and the text unchanged.
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


def _entry(name: str, text: str, extra_files: tuple[str, ...] = ()) -> GuideEntry:
    fields, body = split_front_matter(text)
    return GuideEntry(
        name=name,
        title=_title(body, name),
        description=fields.get("description", ""),
        body=body.rstrip("\n") + "\n",
        extra_files=extra_files,
    )


@functools.cache
def sections() -> tuple[GuideEntry, ...]:
    """The topic sections, in index order."""
    directory = _guide_dir()
    return tuple(_entry(name, _read(directory / f"{name}.md")) for name in SECTION_ORDER)


def _skill_files(directory: Traversable, prefix: str = "") -> list[str]:
    found: list[str] = []
    for child in sorted(directory.iterdir(), key=lambda node: node.name):
        path = f"{prefix}{child.name}"
        if child.is_dir():
            found.extend(_skill_files(child, f"{path}/"))
        elif child.name.endswith(".md") and path != _SKILL_FILE:
            found.append(path)
    return found


@functools.cache
def skills() -> tuple[GuideEntry, ...]:
    """The task skills, by name; empty when no skills directory is reachable."""
    root = _skills_dir()
    if root is None:
        return ()
    entries = []
    for directory in sorted(root.iterdir(), key=lambda node: node.name):
        skill_file = directory / _SKILL_FILE
        if not directory.is_dir() or not skill_file.is_file():
            continue
        entries.append(
            _entry(
                f"{SKILL_PREFIX}{directory.name}",
                _read(skill_file),
                tuple(_skill_files(directory)),
            )
        )
    return tuple(entries)


def names() -> tuple[str, ...]:
    """Every name ``read`` accepts besides a skill's further files."""
    return tuple(entry.name for entry in (*sections(), *skills()))


def _index() -> str:
    lines = ["## Index", "", "Topic sections — pass the name as `section`:", ""]
    lines.extend(f"- `{entry.name}`: {entry.description}" for entry in sections())
    if skills():
        lines.extend(["", "Task skills — playbooks for a kind of task, named the same way:", ""])
        lines.extend(f"- `{entry.name}`: {entry.description}" for entry in skills())
    return "\n".join(lines) + "\n"


@functools.cache
def core() -> str:
    """The core and the generated index: what a session reads first."""
    return _read(_guide_dir() / "core.md").rstrip("\n") + "\n\n" + _index()


def _render(entry: GuideEntry) -> str:
    if not entry.extra_files:
        return entry.body
    listed = "\n".join(f"- `{entry.name}/{path}`" for path in entry.extra_files)
    return f"{entry.body}\nFurther files in this skill, read by the same name:\n\n{listed}\n"


def _skill_file(name: str) -> str | None:
    """A further file inside a skill, or None when ``name`` names none."""
    skill, sep, path = name.partition("/")
    if not sep:
        return None
    entry = next((entry for entry in skills() if entry.name == skill), None)
    if entry is None or path not in entry.extra_files:
        return None
    root = _skills_dir()
    assert root is not None  # a listed skill came from it
    node = root / skill.removeprefix(SKILL_PREFIX)
    for part in path.split("/"):
        node = node / part
    return _read(node)


def read(section: str | None = None) -> str:
    """The core with its index, or one section, skill, or skill file by name.

    Raises :class:`UnknownGuideSection` for a name the guide does not have.
    """
    if section is None:
        return core()
    for entry in (*sections(), *skills()):
        if entry.name == section:
            return _render(entry)
    text = _skill_file(section)
    if text is not None:
        return text
    raise UnknownGuideSection(section, names())


def title_of(section: str | None) -> str:
    """The heading a read of ``section`` opens with."""
    if section is None:
        return _title(core(), "SPICE simulation guide")
    for entry in (*sections(), *skills()):
        if entry.name == section:
            return entry.title
    return _title(read(section), section)
