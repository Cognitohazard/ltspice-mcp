"""One spelling per concept, on the two tools that take `render` and `compare`.

`verify_circuit` and `edit_schematic` describe the same thing — what to compare
a sheet against — so they take the same model, and only the object form. The
flat fields both tools once accepted beside it (verify's
`reference`/`compare_mode`/`anchors`/`rtol`, edit's `reference`) said nothing
the object did not, and a call carrying both was refused rather than resolved,
so a second spelling could only ever be the same call written a longer way.

Drawing the sheet belongs to `verify_circuit` alone: its policy has a pixel cap,
a delivery channel and a render-only mode, and `edit_schematic`'s did not, so an
edit that also wants a picture is one call away from the better one.

This file is the table of what each interface accepts and what it refuses,
asserted on the RESOLVED value rather than on the field.
"""

from __future__ import annotations

from types import NoneType
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from ltspice_mcp.tools._base import CompareSpec, RenderPolicy
from ltspice_mcp.tools.schematic_edit import EditSchematicInput
from ltspice_mcp.tools.verify import VerifyCircuitInput

_EDIT_BASE: dict[str, Any] = {"target": "sheet.asc", "ops": []}


def _edit(**kwargs: Any) -> EditSchematicInput:
    return EditSchematicInput.model_validate({**_EDIT_BASE, **kwargs})


def _verify(**kwargs: Any) -> VerifyCircuitInput:
    return VerifyCircuitInput.model_validate({"path": "deck.cir", **kwargs})


class TestEditSchematicDrawsNothing:
    """Rendering is verify_circuit's; edit_schematic advertises no spelling of it."""

    @pytest.mark.parametrize(
        "payload",
        [
            {"render": True},
            {"render": {"format": "svg"}},
            {"render_format": "svg"},
            {"render_scale": 2.0},
            {"return_views": ["render"]},
        ],
        ids=["bool", "policy", "flat-format", "flat-scale", "view-name"],
    )
    def test_no_render_spelling_is_accepted(self, payload: dict[str, Any]):
        with pytest.raises(ValidationError):
            _edit(**payload)

    def test_verify_circuit_is_where_a_render_is_asked_for(self):
        policy = _verify(render=True).render_policy
        assert policy is not None
        assert policy.format == "png"


class TestEditSchematicCompareSpellings:
    """edit_schematic: the `compare` object, and nothing beside it."""

    def test_no_compare_argument_means_no_comparison(self):
        assert _edit().compare_spec is None

    def test_the_object_names_the_reference(self):
        spec = _edit(compare={"reference": "golden.cir"}).compare_spec
        assert spec is not None
        assert spec.reference == "golden.cir"
        # The tolerance and anchors keep the graph engine's own defaults.
        assert spec.rtol == 1e-6
        assert spec.anchors is None

    def test_the_object_carries_the_comparison_controls(self):
        spec = _edit(compare={"reference": "g.cir", "rtol": 1e-3, "anchors": ["out"]}).compare_spec
        assert spec is not None
        assert (spec.rtol, spec.anchors) == (1e-3, ["out"])

    @pytest.mark.parametrize(
        "payload",
        [
            {"compare": {"rtol": 1e-3}},
            {"compare": {"reference": "g.cir", "mode": "structural_diff"}},
            {"reference": "golden.cir"},
        ],
        ids=["no-reference", "verify-only-mode", "flat-reference"],
    )
    def test_refused_compare_spellings(self, payload: dict[str, Any]):
        with pytest.raises(ValidationError):
            _edit(**payload)


class TestVerifyCircuitSpellings:
    """verify_circuit: the render policy, and the compare object."""

    def test_no_arguments_render_nothing_and_compare_nothing(self):
        args = _verify()
        assert args.render_policy is None
        assert args.compare_spec is None

    @pytest.mark.parametrize(
        ("payload", "field", "expected"),
        [
            ({"render": True}, "format", "png"),
            ({"render": {"format": "svg"}}, "format", "svg"),
            ({"render": {"scale": 2.0}}, "scale", 2.0),
            ({"render": {"max_pixels": 100_000}}, "max_pixels", 100_000),
            ({"render": {"mode": "only"}}, "mode", "only"),
            ({"render": {"delivery": "inline"}}, "delivery", "inline"),
        ],
    )
    def test_render_policy_spellings(self, payload: dict[str, Any], field: str, expected: Any):
        policy = _verify(**payload).render_policy
        assert policy is not None
        assert getattr(policy, field) == expected

    def test_false_and_omitted_both_draw_nothing(self):
        assert _verify(render=False).render_policy is None
        assert _verify().render_policy is None

    def test_the_compare_object_carries_every_control(self):
        spec = _verify(
            compare={
                "reference": "golden.cir",
                "mode": "structural_diff",
                "anchors": ["out", "vdd"],
                "rtol": 1e-3,
            }
        ).compare_spec
        assert spec is not None
        assert spec.reference == "golden.cir"
        assert spec.mode == "structural_diff"
        assert spec.anchors == ["out", "vdd"]
        assert spec.rtol == 1e-3

    @pytest.mark.parametrize(
        "payload",
        [
            {"render": "yes"},
            {"render": {"mode": "sideways"}},
            {"compare": {"rtol": 1e-3}},
            {"reference": "a.cir"},
            {"compare_mode": "structural_diff"},
            {"anchors": ["out"]},
            {"rtol": 1e-3},
        ],
        ids=[
            "not-a-policy",
            "unknown-mode",
            "no-reference",
            "flat-reference",
            "flat-mode",
            "flat-anchors",
            "flat-rtol",
        ],
    )
    def test_refused_spellings(self, payload: dict[str, Any]):
        with pytest.raises(ValidationError):
            _verify(**payload)


def _model_of(input_model: type, field: str) -> type:
    """The model class behind an ``X | None`` field annotation."""
    annotation = input_model.model_fields[field].annotation
    return next(
        arg for arg in get_args(annotation) if isinstance(arg, type) and arg is not NoneType
    )


@pytest.mark.parametrize(
    ("tool", "field", "shared"),
    [
        (EditSchematicInput, "compare", CompareSpec),
        (VerifyCircuitInput, "compare", CompareSpec),
        (VerifyCircuitInput, "render", RenderPolicy),
    ],
)
def test_the_two_tools_share_one_compare_model(tool: type, field: str, shared: type):
    """Not "the same fields" — literally the same class, so they cannot drift.

    A tool that needs more subclasses the shared model, so the shared half stays
    one declaration and the extra half is visibly that tool's own.
    """
    assert issubclass(_model_of(tool, field), shared)
