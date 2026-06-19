"""Drift tests that keep docs + error messages honest about tool names.

Two classes of rot we guard against here:

1. The tool count listed in README.md / CLAUDE.md falls out of sync
   with the actual registry when someone adds or removes a tool.
2. Tool names hardcoded in error strings ("Use ltspice_foo to …") drift
   when a tool is renamed, leaving users chasing ghosts.

These tests are cheap and run on every CI pass.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import ClassVar

import pytest

from ltspice_mcp.tools import (  # noqa: F401
    advanced,
    analysis,
    circuit,
    library,
    simulation,
    status,
)
from ltspice_mcp.tools._base import registry

ROOT = Path(__file__).resolve().parents[1]


def _profile_counts() -> tuple[int, int]:
    full = sum(1 for t in registry._registered if "full" in t.profiles)
    agentic = sum(1 for t in registry._registered if "agentic" in t.profiles)
    return full, agentic


def _registered_names() -> set[str]:
    return {t.definition.name for t in registry._registered}


# (doc path, profile, count-pattern template). ``{n}`` is replaced with the
# registry's count for that profile; each pattern is the exact regex the doc
# must match (README/CLAUDE prose for full, profile-table rows otherwise).
_DOC_COUNT_CHECKS = (
    ("README.md", "full", r"All {n} tools"),
    ("README.md", "agentic", r"\|\s*`agentic`\s*\|\s*{n}\s*\|"),
    ("CLAUDE.md", "full", r"All {n}"),
    ("CLAUDE.md", "agentic", r"\|\s*`agentic`\s*\|\s*{n}\s*\|"),
    ("docs/DESIGN.md", "full", r"\|\s*`full`[^|]*\|\s*{n}\s*\|"),
    ("docs/DESIGN.md", "agentic", r"\|\s*`agentic`[^|]*\|\s*{n}\s*\|"),
)


class TestToolCountInDocs:
    @pytest.mark.parametrize(("rel", "profile", "template"), _DOC_COUNT_CHECKS)
    def test_doc_count_matches_registry(self, rel: str, profile: str, template: str) -> None:
        full, agentic = _profile_counts()
        n = full if profile == "full" else agentic
        text = (ROOT / rel).read_text()
        assert re.search(template.format(n=n), text), (
            f"{rel} must list {n} tools for the {profile} profile "
            f"(expected pattern {template.format(n=n)!r}); the registry "
            f"exposes {n} — update every place the count appears."
        )


DOC_PATHS = (
    "README.md",
    "docs/DESIGN.md",
    "skills/ltspice/SKILL.md",
    "skills/ngspice/SKILL.md",
)

# Tool names that existed before the v0.2.0 consolidation and no longer do.
# Their functionality moved into other tools (bode_metrics modes, query_value
# step addressing, simulation_summary, find_model, edit_directive); a doc that
# still names them as tools sends users chasing ghosts.
REMOVED_TOOL_NAMES = (
    "measurements",
    "model_info",
    "add_text",
    "step_get",
    "filter_metrics",
    "roll_off",
    "gain_at",
    "find_crossing",
    "get_measurements",
    "get_simulation_summary",
    "schematic_from_netlist",
)


class TestStaleToolNamesInDocs:
    def test_no_prefixed_tool_names_in_docs(self) -> None:
        """Tools were renamed from `ltspice_<name>` to bare `<name>`; no doc
        may still use the prefixed form of any registered tool."""
        registered = _registered_names()
        failures: list[str] = []
        for rel in DOC_PATHS:
            text = (ROOT / rel).read_text()
            stale = sorted(f"ltspice_{name}" for name in registered if f"ltspice_{name}" in text)
            if stale:
                failures.append(f"  {rel}: {stale}")
        assert not failures, (
            "Docs reference tools by their old ltspice_-prefixed names:\n"
            + "\n".join(failures)
            + "\nUse the bare registered names instead."
        )

    def test_no_removed_tool_references_in_docs(self) -> None:
        """No doc may reference a removed tool as a tool.

        Only backticked forms — `name` or `name(...)` — count as tool
        references; the bare words ("measurements" in prose) are fine.
        The old `ltspice_`-prefixed form counts too — the prefix check
        above only covers currently-registered names, so a removed tool's
        prefixed form would otherwise slip through both guards.
        """
        failures: list[str] = []
        for rel in DOC_PATHS:
            text = (ROOT / rel).read_text()
            stale = sorted(
                name
                for name in REMOVED_TOOL_NAMES
                if re.search(rf"`(?:ltspice_)?{name}[`(]", text)
            )
            if stale:
                failures.append(f"  {rel}: {stale}")
        assert not failures, (
            "Docs reference tools that no longer exist:\n"
            + "\n".join(failures)
            + "\nPoint at the absorbing tool instead (bode_metrics modes, "
            "query_value step_axis/step_value, simulation_summary, "
            "find_model, edit_directive)."
        )


def _ltspice_refs_in_strings(py_path: Path) -> set[str]:
    """Extract `ltspice_*` tokens that appear INSIDE string literals.

    Regexing the raw source would also match variable / parameter names
    (``ltspice_cls``, ``ltspice_lib_paths``), which aren't tool
    references. Parsing via AST restricts the scan to string values
    only — that's where tool-name rot actually hurts users.
    """
    try:
        tree = ast.parse(py_path.read_text())
    except SyntaxError:
        return set()
    pat = re.compile(r"\bltspice_[a-z][a-z_]+\b")
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.update(pat.findall(node.value))
    return found


class TestToolNamesInErrorStrings:
    # Tokens that look like ltspice_* but aren't tools:
    #   ltspice_mcp   — package name, appears in module paths and log
    #                   prefixes
    #   ltspice_event — log-record extra key used by observability
    _NON_TOOL_TOKENS: ClassVar[set[str]] = {
        "ltspice_mcp",
        "ltspice_event",
    }

    def test_every_ltspice_name_in_strings_is_registered(self) -> None:
        """Any `ltspice_*` token embedded in a string literal must resolve
        to a real registered tool.

        Catches stale tool-name references in error messages, tool
        descriptions, and cross-reference docstrings. Aggregates failures
        across all source files into a single report so a rename that
        breaks many files surfaces as one readable failure instead of
        N near-identical ones.
        """
        registered = _registered_names()
        failures: list[str] = []
        for py_file in sorted((ROOT / "src" / "ltspice_mcp").rglob("*.py")):
            refs = _ltspice_refs_in_strings(py_file) - self._NON_TOOL_TOKENS
            unknown = refs - registered
            if unknown:
                rel = py_file.relative_to(ROOT)
                failures.append(f"  {rel}: {sorted(unknown)}")
        assert not failures, (
            f"{len(failures)} file(s) reference unknown tool(s) in string "
            f"literals:\n" + "\n".join(failures) + "\n"
            f"Either these tools were renamed, or the references are typos.\n"
            f"Registered tools: {sorted(registered)}"
        )
