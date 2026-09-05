"""Every accepted spelling of `render` and `compare`, on both tools that take them.

`verify_circuit` and `edit_schematic` describe the same two things — how to draw
a sheet, and what to compare it against — so they take the same two models. Each
tool ALSO keeps the flat spelling it shipped with (verify's
`reference`/`compare_mode`/`anchors`/`rtol`, edit's
`render_format`/`render_scale`/`reference`), retained as aliases for 0.6.

This file is the table of what each door accepts and what it refuses, asserted
on the RESOLVED value rather than on the field, so an alias that stops mapping
fails here even while it still validates. A spelling removed from the accepted
half is a breaking change to a caller that has already written it.
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


class TestEditSchematicRenderSpellings:
    """edit_schematic: the flat pair, the policy object, and the bare booleans."""

    def test_no_render_argument_keeps_the_shipped_defaults(self):
        args = _edit()
        assert args.render_policy.format == "png"
        assert args.render_policy.scale == 1.5
        assert args.render_policy.max_pixels is None
        # return_views still decides whether a render happens at all.
        assert args.wants_render is False
        assert _edit(return_views=["render"]).wants_render is True

    @pytest.mark.parametrize(
        ("payload", "field", "expected"),
        [
            ({"render_format": "svg"}, "format", "svg"),
            ({"render_scale": 2.0}, "scale", 2.0),
            ({"render": {"format": "svg"}}, "format", "svg"),
            ({"render": {"scale": 2.0}}, "scale", 2.0),
            ({"render": {"max_pixels": 100_000}}, "max_pixels", 100_000),
            ({"render": True}, "format", "png"),
        ],
    )
    def test_every_render_spelling_resolves_to_one_policy(
        self, payload: dict[str, Any], field: str, expected: Any
    ):
        assert getattr(_edit(**payload).render_policy, field) == expected

    def test_the_flat_pair_maps_together(self):
        args = _edit(render_format="svg", render_scale=0.75)
        assert (args.render_policy.format, args.render_policy.scale) == ("svg", 0.75)

    def test_true_asks_for_the_render_without_naming_the_view(self):
        assert _edit(render=True).wants_render is True
        assert _edit(render={"format": "svg"}).wants_render is True

    def test_false_suppresses_the_render_the_view_list_asked_for(self):
        assert _edit(render=False, return_views=["render"]).wants_render is False

    @pytest.mark.parametrize(
        "payload",
        [
            {"render": "yes"},
            {"render": {"format": "gif"}},
            {"render": {"scale": 99.0}},
            {"render": {"mode": "only"}},
            {"render": {"delivery": "inline"}},
            {"render": {"format": "svg"}, "render_format": "png"},
            {"render": True, "render_scale": 2.0},
        ],
        ids=[
            "not-a-policy",
            "unknown-format",
            "scale-out-of-range",
            "verify-only-mode",
            "verify-only-delivery",
            "both-spellings-format",
            "both-spellings-scale",
        ],
    )
    def test_refused_render_spellings(self, payload: dict[str, Any]):
        with pytest.raises(ValidationError):
            _edit(**payload)


class TestEditSchematicCompareSpellings:
    """edit_schematic: the flat `reference`, and the `compare` object."""

    def test_no_compare_argument_means_no_comparison(self):
        assert _edit().compare_spec is None

    @pytest.mark.parametrize(
        "payload",
        [{"reference": "golden.cir"}, {"compare": {"reference": "golden.cir"}}],
        ids=["flat", "object"],
    )
    def test_both_spellings_name_the_same_reference(self, payload: dict[str, Any]):
        spec = _edit(**payload).compare_spec
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
            {"reference": "a.cir", "compare": {"reference": "b.cir"}},
        ],
        ids=["no-reference", "verify-only-mode", "both-spellings"],
    )
    def test_refused_compare_spellings(self, payload: dict[str, Any]):
        with pytest.raises(ValidationError):
            _edit(**payload)


class TestVerifyCircuitSpellings:
    """verify_circuit: its flat compare fields, and the same two objects."""

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

    def test_the_flat_compare_fields_still_map(self):
        spec = _verify(
            reference="golden.cir",
            compare_mode="structural_diff",
            anchors=["out", "vdd"],
            rtol=1e-3,
        ).compare_spec
        assert spec is not None
        assert spec.reference == "golden.cir"
        assert spec.mode == "structural_diff"
        assert spec.anchors == ["out", "vdd"]
        assert spec.rtol == 1e-3

    def test_the_compare_object_says_the_same_thing(self):
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
            {"reference": "a.cir", "compare": {"reference": "b.cir"}},
            {"compare_mode": "structural_diff", "compare": {"reference": "b.cir"}},
        ],
        ids=[
            "not-a-policy",
            "unknown-mode",
            "no-reference",
            "both-spellings-reference",
            "both-spellings-mode",
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


@pytest.mark.parametrize("tool", [EditSchematicInput, VerifyCircuitInput])
@pytest.mark.parametrize(("field", "shared"), [("render", RenderPolicy), ("compare", CompareSpec)])
def test_the_two_tools_share_one_render_model_and_one_compare_model(
    tool: type, field: str, shared: type
):
    """Not "the same fields" — literally the same classes, so they cannot drift.

    A tool that needs more subclasses the shared model, so the shared half stays
    one declaration and the extra half is visibly that tool's own.
    """
    assert issubclass(_model_of(tool, field), shared)
