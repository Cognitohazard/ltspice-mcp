"""The in-process door's argument catalogue: coverage, shape, and delivery."""

from __future__ import annotations

import asyncio
import inspect as inspect_mod
import subprocess
import sys

import pytest
from pydantic import ValidationError

from ltspice_mcp import cli
from ltspice_mcp.api import _reference
from ltspice_mcp.api._methods import ApiMethodsMixin
from ltspice_mcp.errors import compact_validation_error
from ltspice_mcp.lib.recipes import DISCRIMINANTS
from ltspice_mcp.tools.inspect_tools import SUPPORTED_KINDS
from ltspice_mcp.tools.schematic_edit import EditSchematicInput

OPS = ("run_experiments", "jobs", "analyze_results", "inspect", "edit_schematic", "verify_circuit")


def _spelled_fields(model) -> list[str]:
    """The names a caller writes — an alias makes some differ from the attribute."""
    return [
        _reference._field_name(model, name, field) for name, field in model.model_fields.items()
    ]


def _op_kinds() -> set[str]:
    """Every op discriminant edit_schematic accepts, read off the union."""
    from typing import get_args

    from ltspice_mcp.tools import schematic_edit

    union = get_args(get_args(schematic_edit.ConsolidatedOp)[0])
    return {value for model in union for value in get_args(model.model_fields["op"].annotation)}


class TestIndex:
    def test_index_names_all_six_operations(self):
        text = _reference.reference()
        for name in OPS:
            assert name in text

    def test_index_says_how_to_drill_in(self):
        assert "api.reference(" in _reference.reference()

    def test_op_names_is_the_six(self):
        assert set(_reference.op_names()) == set(OPS)


class TestPerOperationTree:
    @pytest.mark.parametrize("name", OPS)
    def test_every_operation_renders(self, name: str):
        text = _reference.reference(name)
        assert text.startswith(name)
        assert "arguments" in text
        assert "example" in text

    @pytest.mark.parametrize("name", OPS)
    def test_tree_carries_no_json_schema_artifacts(self, name: str):
        text = _reference.reference(name)
        for artifact in ("$ref", "anyOf", "allOf", "$defs", "propertyName"):
            assert artifact not in text, f"{name} reference leaked {artifact}"

    @pytest.mark.parametrize("name", OPS)
    def test_top_level_fields_are_all_listed(self, name: str):
        operation = _reference._find(name)
        text = _reference.reference(name)
        for spelled in _spelled_fields(operation.model):
            assert spelled in text, f"{name}.{spelled} missing from its reference"

    def test_recipe_union_is_enumerated_member_by_member(self):
        text = _reference.reference("analyze_results")
        for metric in DISCRIMINANTS:
            assert repr(metric) in text, f"recipe metric {metric} missing"
        # ...and with the per-member fields, not just the names.
        assert "level_deg" in text
        assert "at_hz" in text

    def test_inspect_query_kinds_are_enumerated(self):
        text = _reference.reference("inspect")
        for kind in SUPPORTED_KINDS:
            assert repr(kind) in text

    def test_edit_op_kinds_are_enumerated_with_their_fields(self):
        text = _reference.reference("edit_schematic")
        for kind in _op_kinds():
            assert repr(kind) in text, f"op {kind} missing"
        assert "waypoints" in text
        assert "expected_sha256" in text

    def test_nested_models_are_flattened_onto_dotted_paths(self):
        text = _reference.reference("verify_circuit")
        assert "render.mode" in text
        assert "render.scale" in text

    def test_enum_members_are_written_out_inline(self):
        text = _reference.reference("verify_circuit")
        assert "'with_checks'" in text
        assert "'only'" in text

    def test_defaults_and_requiredness_are_stated(self):
        text = _reference.reference("verify_circuit")
        assert "path" in text
        assert "[required]" in text
        assert "[default" in text

    def test_unknown_operation_names_the_six(self):
        with pytest.raises(ValueError, match="unknown operation") as excinfo:
            _reference.reference("run_simulation")
        for name in OPS:
            assert name in str(excinfo.value)

    def test_a_non_string_op_is_a_type_error(self):
        with pytest.raises(TypeError):
            _reference.reference(3)  # type: ignore[arg-type]


class TestMethodDocstrings:
    @pytest.mark.parametrize("name", OPS)
    def test_method_doc_is_non_empty_and_names_its_fields(self, name: str):
        method = getattr(ApiMethodsMixin, name)
        doc = inspect_mod.getdoc(method)
        assert doc
        operation = _reference._find(name)
        for spelled in _spelled_fields(operation.model):
            assert spelled in doc, f"{name}.{spelled} missing from its docstring"

    def test_docstring_and_reference_come_from_one_renderer(self):
        doc = inspect_mod.getdoc(ApiMethodsMixin.verify_circuit)
        assert doc is not None
        # The tree body is shared; only the header differs.
        assert "render.mode" in doc
        assert "render.mode" in _reference.reference("verify_circuit")

    def test_reference_is_reachable_without_an_engine_session(self):
        # A static method: reading the catalogue must not take this process's
        # single session lease, nor require one to have been opened.
        assert "run_experiments" in ApiMethodsMixin.reference()


class TestCliDelivery:
    """`spice-mcp reference` is the same renderer behind a print."""

    @staticmethod
    def _run(argv: list[str], capsys) -> tuple[int, str]:
        namespace = cli.parse_args(argv)
        code = asyncio.run(cli.execute(namespace))
        return code, capsys.readouterr().out

    def test_index_prints_the_six(self, capsys):
        code, out = self._run(["reference"], capsys)
        assert code == 0
        for name in OPS:
            assert name in out

    def test_hyphenated_and_underscored_names_both_work(self, capsys):
        _, hyphenated = self._run(["reference", "edit-schematic"], capsys)
        _, underscored = self._run(["reference", "edit_schematic"], capsys)
        assert hyphenated == underscored
        assert hyphenated.rstrip("\n") == _reference.reference("edit_schematic")

    def test_an_unknown_name_is_a_usage_refusal(self, capsys):
        code, _ = self._run(["reference", "run-simulation"], capsys)
        assert code == cli.EXIT_REFUSED

    def test_printing_the_catalogue_starts_no_session(self, tmp_path):
        """It is documentation, not a server start. Starting the engine writes
        an ltspice-mcp.toml into the working directory; reading the catalogue
        must leave the directory exactly as it found it."""
        probe = (
            "from ltspice_mcp.cli import main\n"
            "try:\n"
            "    main(['reference', 'verify-circuit'])\n"
            "except SystemExit:\n"
            "    pass\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )
        assert "render.mode" in proc.stdout
        assert list(tmp_path.iterdir()) == []


class TestOpsUnionErrorEnumerates:
    """A mistyped op used to produce an error per branch, truncated at
    "… and 23 more"; the caller learned neither the kinds nor their own gap."""

    def test_unknown_op_error_names_every_kind(self):
        with pytest.raises(ValidationError) as excinfo:
            EditSchematicInput.model_validate({"target": "s.asc", "ops": [{"op": "bogus"}]})
        detail = compact_validation_error(excinfo.value)
        assert "more" not in detail, detail
        for kind in _op_kinds():
            assert kind in detail, f"op {kind} missing from the union error"

    def test_a_known_op_reports_against_that_kind_alone(self):
        with pytest.raises(ValidationError) as excinfo:
            EditSchematicInput.model_validate(
                {"target": "s.asc", "ops": [{"op": "add_component", "reference": "R1"}]}
            )
        detail = compact_validation_error(excinfo.value)
        assert "symbol" in detail
        assert "from_pin" not in detail, "a sibling branch's fields leaked into the error"


class TestCatalogueMatchesTheRegistry:
    """_operations() names the six ops by hand next to the registry that
    already maps them; this pin is what turns a drift into a failure."""

    def test_operation_names_equal_the_consolidated_profile(self):
        from ltspice_mcp.tools import get_tools_for_profile

        tool_defs, _dispatch = get_tools_for_profile("consolidated")
        assert set(_reference.op_names()) == {tool.name for tool in tool_defs}
