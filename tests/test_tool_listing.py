"""The two ways the same tool surface can be served: full and compact.

``[tools] listing`` changes only how much of each definition a client is shown
at ``tools/list``. Neither mode adds or removes a capability, both are static
listings, and ``full`` — the default — must stay byte for byte what the
registry produces, because every other contract test in the suite reads it.
"""

from __future__ import annotations

from typing import Any

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import get_tools
from ltspice_mcp.tools._base import registry
from ltspice_mcp.tools._schema import strip_argument_descriptions
from tests.conftest import REGISTERED_TOOLS


def _descriptions(node: Any, path: str = "") -> dict[str, str]:
    """Every description in a schema, keyed by where it sits."""
    found: dict[str, str] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "description" and isinstance(value, str):
                found[path or "<root>"] = value
            else:
                found |= _descriptions(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found |= _descriptions(item, f"{path}[{index}]")
    return found


def _without_prose(node: Any, in_name_map: bool = False) -> Any:
    """A schema with every description removed.

    Written here rather than reused from the server so a comparison of the two
    listings has an oracle of its own.
    """
    if isinstance(node, dict):
        if in_name_map:
            return {key: _without_prose(value) for key, value in node.items()}
        return {
            key: _without_prose(value, key in {"properties", "$defs"})
            for key, value in node.items()
            if key != "description"
        }
    if isinstance(node, list):
        return [_without_prose(item) for item in node]
    return node


def _argument_free(node: Any) -> bool:
    """True for a branch whose published properties are all fixed values.

    The rule the compact listing exempts, spelled here independently of the
    server's copy: with every property a ``const`` there is no argument left
    to describe, so the node's description is everything the branch says.
    """
    properties = node.get("properties") if isinstance(node, dict) else None
    if not isinstance(properties, dict) or not properties:
        return False
    return all(isinstance(value, dict) and "const" in value for value in properties.values())


def _exempt_descriptions(node: Any, path: str = "") -> dict[str, str]:
    """The descriptions the compact listing must keep, keyed the same way."""
    found: dict[str, str] = {}
    if isinstance(node, dict):
        if _argument_free(node) and isinstance(node.get("description"), str):
            found[path or "<root>"] = node["description"]
        for key, value in node.items():
            found |= _exempt_descriptions(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found |= _exempt_descriptions(item, f"{path}[{index}]")
    return found


class TestFullListingIsUnchanged:
    def test_default_argument_is_the_registry_output(self):
        """Every other test in the suite calls get_tools() with no argument; the
        listing parameter must not have moved what that returns."""
        registry_defs, registry_dispatch = registry.get_tools()
        defs, dispatch = get_tools()
        assert [d.model_dump_json(by_alias=True) for d in defs] == [
            d.model_dump_json(by_alias=True) for d in registry_defs
        ]
        assert dispatch == registry_dispatch

    def test_full_is_the_same_as_the_default(self):
        assert [d.model_dump_json(by_alias=True) for d in get_tools("full")[0]] == [
            d.model_dump_json(by_alias=True) for d in get_tools()[0]
        ]


class TestCompactListing:
    def test_same_tools_and_same_tool_level_descriptions(self):
        full = {d.name: d for d in get_tools("full")[0]}
        compact = {d.name: d for d in get_tools("compact")[0]}
        assert set(compact) == set(REGISTERED_TOOLS)
        assert list(compact) == list(full), "compact must not reorder the surface"
        for name, definition in compact.items():
            assert definition.description == full[name].description


class TestEveryToolCarriesADisplayTitle:
    """The label a client shows a person in place of the wire name.

    A title is not an argument description, so the compact listing keeps it:
    the mode drops what an argument means, never how the tool is named.
    """

    @pytest.mark.parametrize("listing", ["full", "compact"])
    @pytest.mark.parametrize("name", REGISTERED_TOOLS)
    def test_title_is_present_and_readable(self, listing: str, name: str):
        definition = {d.name: d for d in get_tools(listing)[0]}[name]
        assert definition.title, f"{name} advertises no display title"
        assert definition.title != name, (
            f"{name}: the title repeats the wire name, so it tells a reader nothing new"
        )
        # A label, not a sentence: a client renders it inline in a tool list.
        assert len(definition.title) <= 40
        assert not definition.title.endswith(".")

    @pytest.mark.parametrize("name", REGISTERED_TOOLS)
    def test_every_argument_description_is_gone(self, name: str):
        """Except the argument-free branches — see the class below."""
        full = {d.name: d for d in get_tools("full")[0]}[name]
        compact = {d.name: d for d in get_tools("compact")[0]}[name]
        assert _descriptions(full.input_schema), f"{name} advertises no descriptions to strip"
        assert _descriptions(compact.input_schema) == _exempt_descriptions(full.input_schema)

    @pytest.mark.parametrize("name", REGISTERED_TOOLS)
    def test_nothing_but_the_descriptions_changes(self, name: str):
        """Schema equality with the prose removed: structure, enums, defaults,
        required and $defs must all survive.

        Both sides go through this module's own stripper rather than the
        server's, so the comparison holds even if ``get_tools`` stops calling
        the transform it is supposed to.
        """
        full = {d.name: d for d in get_tools("full")[0]}[name]
        compact = {d.name: d for d in get_tools("compact")[0]}[name]
        assert _without_prose(compact.input_schema) == _without_prose(full.input_schema)
        assert set(compact.input_schema.get("$defs", {})) == set(
            full.input_schema.get("$defs", {})
        )

    def test_compact_is_materially_smaller_than_full(self):
        """What the mode is for. docs/design/mcp_surface.md claims roughly 45%
        off, and the measured ratio is about 0.55; the bound sits just above it
        so that losing a large part of the saving fails here rather than
        quietly halving what the mode is worth."""

        def total(listing: str) -> int:
            return sum(
                len(d.model_dump_json(by_alias=True, exclude_none=True))
                for d in get_tools(listing)[0]
            )

        full, compact = total("full"), total("compact")
        assert compact < full * 0.60, (
            f"compact listing is {compact} chars against full's {full} — the "
            "argument-description strip is no longer paying for itself"
        )

    def test_dispatch_and_validation_are_untouched(self):
        """The published copy is filtered, never the models — a compact-mode
        client's calls are validated exactly as a full-mode client's are."""
        assert get_tools("compact")[1] == registry.get_tools()[1]

    def test_the_registry_schema_is_not_mutated_by_compaction(self):
        get_tools("compact")
        assert _descriptions(get_tools("full")[0][0].input_schema)

    def test_an_argument_literally_named_description_survives(self):
        """The filter descends structurally: inside ``properties`` the keys are
        argument names, so a field called ``description`` is data, not prose."""
        schema = {
            "type": "object",
            "description": "the tool's own prose",
            "properties": {
                "description": {"type": "string", "description": "what this field means"},
                "nested": {
                    "type": "object",
                    "properties": {"title": {"type": "string", "description": "gone"}},
                },
            },
            "$defs": {"description": {"type": "string", "description": "also gone"}},
        }
        stripped = strip_argument_descriptions(schema)
        assert "description" not in stripped
        assert set(stripped["properties"]) == {"description", "nested"}
        assert stripped["properties"]["description"] == {"type": "string"}
        assert stripped["properties"]["nested"]["properties"]["title"] == {"type": "string"}
        assert set(stripped["$defs"]) == {"description"}
        assert stripped["$defs"]["description"] == {"type": "string"}


class TestDormantBranchesKeepTheirDescription:
    """The one exemption from the strip, and why it has to exist.

    A dormant recipe branch is advertised as its discriminant and nothing
    else, so its description carries the whole branch: what it produces and
    where to read the arguments. Compact strips prose on the bet that the
    published structure still lets a client build a valid call; on these three
    there is no structure to fall back on, so a stripped branch would be a
    metric name a client could send and the server would then reject. Nothing
    declares the exemption: the branch's own shape is what earns it.
    """

    @staticmethod
    def _analyze(listing: str) -> dict[str, Any]:
        return {d.name: d for d in get_tools(listing)[0]}["analyze_results"].input_schema

    def _exempt(self, listing: str) -> dict[str, Any]:
        exempt = {
            name: body
            for name, body in self._analyze(listing)["$defs"].items()
            if _argument_free(body)
        }
        assert exempt, "no recipe branch publishes its discriminant alone"
        return exempt

    def test_the_exempt_branches_are_the_discriminant_only_ones(self):
        for name, body in self._exempt("full").items():
            assert set(body["properties"]) == {"metric"}, (
                f"{name} publishes arguments, so its description is not its whole content"
            )
            assert body["description"]

    def test_each_exempt_description_survives_compaction_verbatim(self):
        full = self._analyze("full")["$defs"]
        compact = self._analyze("compact")["$defs"]
        for name in self._exempt("full"):
            assert compact[name]["description"] == full[name]["description"]

    def test_the_description_names_the_mcp_lookup_first(self):
        """The route a client on this listing can actually take: the other two
        pointers are a Python import and a resource read."""
        for name, body in self._exempt("full").items():
            metric = body["properties"]["metric"]["const"]
            pointer = f"inspect(kind='reference', query='{metric}')"
            assert pointer in body["description"], name
            assert body["description"].index(pointer) < body["description"].index("api.reference(")
            assert "spice://guide" in body["description"]


def _state(work_dir, listing: str) -> SessionState:
    return SessionState.create(
        ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], tool_listing=listing),  # type: ignore[arg-type]
        available={},
    )


class TestSessionStateHonoursTheListing:
    @pytest.mark.parametrize("listing", ["full", "compact"])
    def test_state_serves_the_configured_listing(self, work_dir, listing: str):
        state = _state(work_dir, listing)
        names = [d.name for d in state.tool_defs]
        assert names == [d.name for d in get_tools(listing)[0]]

    @pytest.mark.parametrize("listing", ["full", "compact"])
    def test_capabilities_reports_which_listing_the_session_got(self, work_dir, listing: str):
        """spice://guide tells a caller to reach for the reference lookup when
        the listing is compact. Nothing else on the wire says which one it is:
        both modes advertise the same seven tools and the same schemas, so a
        caller could only infer it from prose that is not there."""
        from ltspice_mcp.tools.inspect_tools import _do_capabilities

        assert _do_capabilities(_state(work_dir, listing))["tool_listing"] == listing

    def test_compact_state_serves_no_argument_prose(self, work_dir):
        state = _state(work_dir, "compact")
        full = {d.name: d for d in get_tools("full")[0]}
        for definition in state.tool_defs:
            assert _descriptions(definition.input_schema) == _exempt_descriptions(
                full[definition.name].input_schema
            )
