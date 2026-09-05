"""Drift tests that keep docs + error messages honest about tool names.

Two classes of rot we guard against here:

1. The tool count listed in README.md / CLAUDE.md falls out of sync
   with the actual registry when someone adds or removes a tool.
2. Tool names hardcoded in docs and error strings ("Use foo to …") drift
   when a tool is removed or renamed, leaving users chasing ghosts.

These tests are cheap and run on every CI pass.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, ClassVar

import pytest

import ltspice_mcp.tools  # noqa: F401  (imports trigger every registration)
from ltspice_mcp.tools._base import registry
from tests.conftest import TOOLS_REMOVED_IN_0_6

ROOT = Path(__file__).resolve().parents[1]


def _registered_names() -> set[str]:
    return {t.definition.name for t in registry._registered}


# (doc path, count-pattern template). ``{n}`` is replaced with the registry's
# consolidated tool count; each pattern is the exact regex the doc must match.
_DOC_COUNT_CHECKS = (
    ("README.md", r"{n} tools"),
    ("CLAUDE.md", r"{n} (?:registered )?tools"),
    ("docs/DESIGN.md", r"{n} (?:registered )?tools"),
)


class TestToolCountInDocs:
    @pytest.mark.parametrize(("rel", "template"), _DOC_COUNT_CHECKS)
    def test_doc_count_matches_registry(self, rel: str, template: str) -> None:
        n = len(registry.get_for_profile("consolidated")[0])
        text = (ROOT / rel).read_text()
        assert re.search(template.format(n=n), text), (
            f"{rel} must state the registered tool count {n} "
            f"(expected pattern {template.format(n=n)!r}) — update every place "
            "the count appears."
        )


DOC_PATHS = (
    "README.md",
    "src/ltspice_mcp/assets/spice_guide.md",
    "docs/DESIGN.md",
    "skills/ltspice/SKILL.md",
    "skills/ngspice/SKILL.md",
    "skills/spice-experiments/SKILL.md",
    "skills/spice-bench-craft/SKILL.md",
)

# Every tool name that has ever been removed from the registry: the
# pre-consolidation removals plus the 42 v0.5 tools de-registered when the
# consolidated profile became the product. Frozen history — grown, never
# shrunk by hand.
_DEAD_TOOL_NAMES: tuple[str, ...] = (
    # Pre-0.6 consolidations (bode_metrics modes, query_value step addressing,
    # simulation_summary, find_model, edit_directive absorbed them).
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
    "pulse_response",
    "disturbance_response",
    # v0.5 tools removed in 0.6.0: single-homed in conftest.
    *TOOLS_REMOVED_IN_0_6,
)


def _live_surface_vocabulary() -> set[str]:
    """Every enum/const value in the registered tools' input schemas.

    This is the live capability vocabulary — edit_schematic op kinds, recipe
    metrics, inspect kinds, jobs actions. A dead TOOL name that is also one of
    these (``wire_pins`` the op, ``signal_stats`` the metric,
    ``operating_point`` the keyed metric) names a live capability, so docs may
    reference it in backticks; forbidding it would forbid documenting the
    product. Derived from the schemas rather than hand-listed so a name that
    later stops being a live op automatically re-enters the removed-name gate.
    """
    vocab: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            enum = node.get("enum")
            if isinstance(enum, list):
                vocab.update(v for v in enum if isinstance(v, str))
            const = node.get("const")
            if isinstance(const, str):
                vocab.add(const)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    for reg in registry._registered:
        walk(reg.definition.inputSchema)
    return vocab


# The gate's operative list: dead tool names that are NOT also live-surface
# vocabulary. Docs may not reference these in backticked tool position.
REMOVED_TOOL_NAMES = tuple(sorted(set(_DEAD_TOOL_NAMES) - _live_surface_vocabulary()))


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
        prefixed form would otherwise slip through both guards. Dead names
        that survive as live-surface vocabulary (edit_schematic ops,
        analyze_results metrics — see _live_surface_vocabulary) are exempt
        by derivation, never by hand.
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
            + "\nPoint at the absorbing surface instead (run_experiments, "
            "jobs, analyze_results recipes, inspect kinds, edit_schematic "
            "ops, verify_circuit)."
        )

    def test_dead_name_list_is_history_not_surface(self) -> None:
        """A registered tool name in the dead list means the list rotted (or a
        tool was resurrected without pruning it) — either way the gate would
        forbid documenting a live tool."""
        overlap = sorted(set(_DEAD_TOOL_NAMES) & _registered_names())
        assert not overlap, f"dead-name list contains registered tools: {overlap}"


class TestConsolidatedSkillDocCoverage:
    def test_experiment_skill_doc_names_every_consolidated_tool(self) -> None:
        # Derived from the registry, not hand-copied: adding a tool to the
        # consolidated profile fails here until the skill doc teaches it (or
        # this pin is deliberately revisited).
        tool_defs, _ = registry.get_for_profile("consolidated")
        names = sorted(t.name for t in tool_defs)
        assert names, "consolidated profile registered no tools"
        text = (ROOT / "skills/spice-experiments/SKILL.md").read_text()
        missing = [name for name in names if name not in text]
        assert not missing, f"skills/spice-experiments/SKILL.md never mentions {missing}"


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


def _non_docstring_strings(py_path: Path) -> list[str]:
    """Every string literal in the file except module/class/function docstrings.

    Docstrings are developer text — they legitimately name a helper after the
    tool it once backed (":func:`handle_find_crossing`") and no client ever
    reads them. Error messages, ``Field`` descriptions, warnings and
    observation details are what a caller sees, and those are exactly the
    string constants that are NOT a docstring.
    """
    tree = ast.parse(py_path.read_text())
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


class TestRemovedToolNamesInClientReachingStrings:
    """Strings the CLIENT reads must not point at removed tools.

    A resource note that says "Use check_job(job_id)" survives tool removal
    silently — nothing imports the dead name, so only a client following the
    advice finds out, as an "Unknown tool" error. Scoped to the modules whose
    strings actually reach the wire (resources, server, prompts) plus the
    advertised tool descriptions; a src-wide scan would drown in adapters'
    own docstrings and ordinary English ("parameter", "recent"). Call-shaped
    references only (``name(``): that is how guidance names a tool, while
    prose reuse of a word like "recent" is legitimate.

    The sibling scan below covers the OTHER live source of client-visible
    prose: the error messages, warnings and observations raised in ``lib/``
    and in the analysis adapters. Those never look like a call — they say
    "use check_job (status + completion summary)" — so they need the wider,
    word-shaped match, restricted to non-docstring strings.
    """

    # Retained 0.5 adapters whose strings are rewritten or deleted with the
    # legacy layer; until then their prose is not client-reaching guidance.
    _RETAINED_ADAPTERS: ClassVar[frozenset[str]] = frozenset(
        {"tools/circuit.py", "tools/simulation.py"}
    )

    # Every module whose strings a caller can read back: ``lib/`` raises the
    # errors and builds the warnings/observations the tools relay verbatim,
    # ``tools/`` holds the tools' own prose and the analysis adapters every
    # ``analyze_results`` recipe reaches, and the client modules carry the
    # handshake instructions, prompts and resources.
    _PROSE_MODULES: ClassVar[tuple[str, ...]] = tuple(
        sorted(
            {
                *(f"lib/{p.name}" for p in (ROOT / "src" / "ltspice_mcp" / "lib").glob("*.py")),
                *(
                    f"tools/{p.name}"
                    for p in (ROOT / "src" / "ltspice_mcp" / "tools").glob("*.py")
                ),
                "resources.py",
                "server.py",
                "prompts.py",
            }
            - _RETAINED_ADAPTERS
        )
    )

    # Removed tool names that are also ordinary English, so a word-shaped
    # match over prose cannot tell a tool reference from a sentence. Both are
    # single common words with no call syntax anywhere in the scanned tree;
    # keeping them in would make the gate unusable rather than strict. Dead
    # names that survive as live surface vocabulary (the ``signal_stats``
    # recipe, the ``wire_pins`` op) are already subtracted upstream by
    # REMOVED_TOOL_NAMES — those are exempt by derivation, not by this list.
    _ENGLISH_HOMONYMS: ClassVar[frozenset[str]] = frozenset({"parameter", "recent"})

    def test_exemptions_are_subsets_of_what_they_exempt(self) -> None:
        assert set(REMOVED_TOOL_NAMES) >= self._ENGLISH_HOMONYMS
        assert {
            f"tools/{p.name}" for p in (ROOT / "src" / "ltspice_mcp" / "tools").glob("*.py")
        } >= self._RETAINED_ADAPTERS

    def test_no_removed_tool_names_in_prose_strings(self) -> None:
        names = sorted(set(REMOVED_TOOL_NAMES) - self._ENGLISH_HOMONYMS)
        pat = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b")
        failures: list[str] = []
        for rel in self._PROSE_MODULES:
            py_file = ROOT / "src" / "ltspice_mcp" / rel
            hits: set[str] = set()
            for text in _non_docstring_strings(py_file):
                hits.update(m.group(1) for m in pat.finditer(text))
            if hits:
                failures.append(f"  src/ltspice_mcp/{rel}: {sorted(hits)}")
        assert not failures, (
            "Error messages, warnings or observations name removed tools:\n"
            + "\n".join(failures)
            + "\nName the live surface instead (an analyze_results recipe, "
            "an inspect kind, a jobs action, run_experiments, verify_circuit)."
        )

    def test_no_removed_tool_names_in_advertised_descriptions(self) -> None:
        # The subtracted list, same as the doc gate: a description may say
        # "the signal_stats recipe" (live vocabulary) but not "use bode_metrics"
        # (a dead tool with no live meaning).
        pat = re.compile(r"\b(" + "|".join(map(re.escape, REMOVED_TOOL_NAMES)) + r")\b")
        failures: list[str] = []
        for reg in registry._registered:
            hits = sorted({m.group(1) for m in pat.finditer(reg.definition.description or "")})
            if hits:
                failures.append(f"  {reg.definition.name}: {hits}")
        assert not failures, "Advertised tool descriptions name removed tools:\n" + "\n".join(
            failures
        )


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
