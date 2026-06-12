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


class TestToolCountInDocs:
    def test_readme_full_count_matches_registry(self) -> None:
        full, _ = _profile_counts()
        text = (ROOT / "README.md").read_text()
        assert f"All {full} tools" in text, (
            f"README.md must say 'All {full} tools' — registry exposes {full} "
            f"in the full profile. Update the three places that mention the "
            f"count (tool profile table, tools intro, full profile row)."
        )

    def test_readme_agentic_count_matches_registry(self) -> None:
        _, agentic = _profile_counts()
        text = (ROOT / "README.md").read_text()
        # "| `agentic` | 27 |" row in the tool-profile table.
        pattern = rf"\|\s*`agentic`\s*\|\s*{agentic}\s*\|"
        assert re.search(pattern, text), (
            f"README.md tool-profile table must list {agentic} for the agentic profile"
        )

    def test_claude_md_full_count_matches_registry(self) -> None:
        full, _ = _profile_counts()
        text = (ROOT / "CLAUDE.md").read_text()
        assert f"All {full}" in text, (
            f"CLAUDE.md must say 'All {full}' somewhere in the tool profile "
            f"table; registry exposes {full} tools in the full profile"
        )

    def test_claude_md_agentic_count_matches_registry(self) -> None:
        _, agentic = _profile_counts()
        text = (ROOT / "CLAUDE.md").read_text()
        pattern = rf"\|\s*`agentic`\s*\|\s*{agentic}\s*\|"
        assert re.search(pattern, text), (
            f"CLAUDE.md tool-profile table must list {agentic} for the agentic profile"
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
)


class TestStaleToolNamesInDocs:
    def test_no_prefixed_tool_names_in_docs(self) -> None:
        """Tools were renamed from `ltspice_<name>` to bare `<name>`; no doc
        may still use the prefixed form of any registered tool."""
        registered = {t.definition.name for t in registry._registered}
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


class TestDesignDocCounts:
    def test_design_md_full_count_matches_registry(self) -> None:
        full, _ = _profile_counts()
        text = (ROOT / "docs" / "DESIGN.md").read_text()
        pattern = rf"\|\s*`full`[^|]*\|\s*{full}\s*\|"
        assert re.search(pattern, text), (
            f"docs/DESIGN.md tool-profile table must list {full} for the full profile"
        )

    def test_design_md_agentic_count_matches_registry(self) -> None:
        _, agentic = _profile_counts()
        text = (ROOT / "docs" / "DESIGN.md").read_text()
        pattern = rf"\|\s*`agentic`[^|]*\|\s*{agentic}\s*\|"
        assert re.search(pattern, text), (
            f"docs/DESIGN.md tool-profile table must list {agentic} for the agentic profile"
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
    _NON_TOOL_TOKENS: ClassVar[set[str]] = {"ltspice_mcp", "ltspice_event"}

    def test_every_ltspice_name_in_strings_is_registered(self) -> None:
        """Any `ltspice_*` token embedded in a string literal must resolve
        to a real registered tool.

        Catches stale tool-name references in error messages, tool
        descriptions, and cross-reference docstrings. Aggregates failures
        across all source files into a single report so a rename that
        breaks many files surfaces as one readable failure instead of
        N near-identical ones.
        """
        registered = {t.definition.name for t in registry._registered}
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
