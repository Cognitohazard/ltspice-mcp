"""Per-instance mismatch reached through one subcircuit level.

Fixtures are shaped like a foundry model deck — see ``tests._mismatch_fixtures``
— but self-contained, so the suite runs with no PDK installed. This module uses
the BINNED model spelling, which is what a real PDK ships.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from ltspice_mcp.lib.spice_lex import TokenKind, lex, tokenize_body
from ltspice_mcp.lib.subckt_mismatch import (
    ClosureFile,
    MismatchPlanError,
    MismatchValues,
    build_plan,
    encode_ref,
    forwarded_param,
    render,
    require_geometry,
)
from tests._mismatch_fixtures import BINNED_MINI_FET as MINI_FET
from tests._mismatch_fixtures import clone_body, instance_line

DECK = f"""\
* mini pdk-shaped deck
{MINI_FET}
XN0 dn0 g 0 0 minifet W=1 L=0.15
XN1 dn1 g 0 0 minifet W=1 L=0.15
V1 dn0 0 0.9
Vg g 0 0.9
.op
.end
"""


def closure(*texts: str, root: Path | None = None) -> list[ClosureFile]:
    base = root or Path("/deck")
    return [
        ClosureFile(
            index=i,
            path=base / ("deck.cir" if i == 0 else f"inc{i}.spice"),
            text=text,
        )
        for i, text in enumerate(texts)
    ]


class TestPlanDiscovery:
    def test_prefix_reaches_every_x_wrapped_device(self):
        plan = build_plan(closure(DECK), prefix="X")
        assert [t.ref for t in plan.targets] == ["XN0.m0", "XN1.m0"]
        assert {t.subckt for t in plan.targets} == {"minifet"}
        assert {t.model_name for t in plan.targets} == {"minifet_model"}

    def test_geometry_comes_off_the_x_line_in_metres(self):
        # No .option scale, so the numbers on the X line are already metres.
        plan = build_plan(closure(DECK), prefix="X")
        assert require_geometry(plan.targets[0]) == (1.0, 0.15)

    def test_option_scale_converts_the_x_line_geometry(self):
        # The foundry spelling: geometry in scale units, scale in metres.
        scaled = DECK.replace("* mini pdk-shaped deck", "* scaled\n.option scale=1.0u")
        plan = build_plan(closure(scaled), prefix="X")
        assert require_geometry(plan.targets[0]) == pytest.approx((1e-6, 0.15e-6))

    def test_two_different_scales_cannot_be_reconciled(self):
        scaled = DECK.replace(
            "* mini pdk-shaped deck", "* scaled\n.option scale=1.0u\n.options scale=1n"
        )
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(scaled), prefix="X")
        assert exc.value.code == "ambiguous_scale"

    def test_one_clone_per_device_type_not_per_instance(self):
        plan = build_plan(closure(DECK), prefix="X")
        assert len(plan.clones) == 1
        assert plan.clones[0].name == "minifet__mcpatch"
        assert {t.clone for t in plan.targets} == {"minifet__mcpatch"}

    def test_clone_name_avoids_a_name_the_deck_already_uses(self):
        taken = DECK.replace(
            "V1 dn0 0 0.9",
            ".subckt minifet__mcpatch a b\nR9 a b 1\n.ends\nV1 dn0 0 0.9",
        )
        plan = build_plan(closure(taken), prefix="X")
        assert plan.clones[0].name == "minifet__mcpatch_2"

    def test_prefix_ignores_x_cards_inside_library_bodies(self):
        # A wrapper's own internal X card is the library's wiring, not a device
        # the caller asked to perturb.
        wrapper = MINI_FET.replace(
            ".ends minifet",
            ".ends minifet\n.subckt wrapper d g s b\nxinner d g s b minifet\n.ends wrapper\n",
        )
        deck = DECK.replace(MINI_FET, wrapper)
        plan = build_plan(closure(deck), prefix="X")
        assert [t.x_ref for t in plan.targets] == ["XN0", "XN1"]


class TestCloneEmission:
    def test_clone_forwards_a_reserved_pair_onto_the_inner_device(self):
        files = closure(DECK)
        plan = build_plan(files, prefix="X")
        out = render(files, plan, {"XN0.m0": MismatchValues(delvto=-0.02)})
        text = clone_body(out[0], "minifet__mcpatch")
        assert ".subckt minifet__mcpatch d g s b" in text
        assert "mc_delvto__m0 = 0" in text
        assert "mc_mulu0__m0 = 1" in text
        inner = instance_line(text, "m0")
        assert inner.get_param("delvto") == "{mc_delvto__m0}"
        assert inner.get_param("mulu0") == "{mc_mulu0__m0}"
        assert ".ends minifet__mcpatch" in text.lower()

    def test_clone_keeps_the_library_definition_untouched(self):
        files = closure(DECK)
        plan = build_plan(files, prefix="X")
        out = render(files, plan, {"XN0.m0": MismatchValues(delvto=-0.02)})
        assert ".subckt minifet d g s b" in out[0]
        assert instance_line(MINI_FET, "m0").get_param("delvto") is None

    def test_clone_lands_before_the_end_directive(self):
        files = closure(DECK)
        plan = build_plan(files, prefix="X")
        out = render(files, plan, {"XN0.m0": MismatchValues(delvto=-0.02)})
        assert out[0].index("minifet__mcpatch") < out[0].lower().rindex(".end")

    def test_declares_defaults_on_the_subckt_line_when_there_is_no_param_card(self):
        header_style = (
            ".subckt minifet d g s b w=1 l=0.15\n"
            "m0 d g s b minifet_model w={w} l={l}\n"
            ".model minifet_model nmos level=49 vth0=0.7\n"
            ".ends minifet\n"
        )
        deck = DECK.replace(MINI_FET, header_style)
        files = closure(deck)
        plan = build_plan(files, prefix="X")
        out = render(files, plan, {"XN0.m0": MismatchValues(delvto=-0.02)})
        body = clone_body(out[0], "minifet__mcpatch")
        opener = next(c for c in lex(body).cards if c.kind == "subckt")
        assert "mc_delvto__m0=0" in opener.body.replace(" =", "=").replace("= ", "=")


class TestRender:
    def test_values_land_on_their_own_instance(self):
        files = closure(DECK)
        plan = build_plan(files, prefix="X")
        out = render(
            files,
            plan,
            {
                "XN0.m0": MismatchValues(delvto=-0.02, mulu0=1.05),
                "XN1.m0": MismatchValues(delvto=0.031),
            },
        )
        xn0 = instance_line(out[0], "XN0")
        xn1 = instance_line(out[0], "XN1")
        assert xn0.model == "minifet__mcpatch"
        assert xn0.get_param("mc_delvto__m0") == "-0.02"
        assert xn0.get_param("mc_mulu0__m0") == "1.05"
        assert xn1.get_param("mc_delvto__m0") == "0.031"
        # An unsupplied knob is simply not written; the clone's default holds it
        # neutral rather than the caller having to spell out "no change".
        assert xn1.get_param("mc_mulu0__m0") is None
        # W/L survive the rewrite — the clone still needs its geometry.
        assert xn0.get_param("W") == "1"
        assert xn0.get_param("L") == "0.15"

    def test_untargeted_instance_keeps_the_original_subcircuit(self):
        files = closure(DECK)
        plan = build_plan(files, prefix="X")
        out = render(files, plan, {"XN0.m0": MismatchValues(delvto=-0.02)})
        assert instance_line(out[0], "XN1").model == "minifet"

    def test_render_reports_only_changed_files(self):
        included = closure(DECK, MINI_FET)
        # The deck alone is edited when it holds both the X lines and the
        # definition, so nothing asks the caller to re-stage the include.
        plan = build_plan(included[:1], prefix="X")
        assert set(render(included[:1], plan, {"XN0.m0": MismatchValues(delvto=0.01)})) == {0}

    def test_a_value_for_an_unplanned_device_is_refused(self):
        files = closure(DECK)
        plan = build_plan(files, prefix="X")
        with pytest.raises(MismatchPlanError) as exc:
            render(files, plan, {"XN9.m0": MismatchValues(delvto=0.01)})
        assert exc.value.code == "unplanned_instance"


class TestIncludedDefinition:
    """The device library normally sits in an included file."""

    DECK_WITH_INCLUDE = """\
* deck that includes its device library
.include inc1.spice
XN0 dn0 g 0 0 minifet W=1 L=0.15
V1 dn0 0 0.9
.op
.end
"""

    def test_clone_goes_to_the_root_deck_and_the_library_is_not_rewritten(self):
        files = closure(self.DECK_WITH_INCLUDE, MINI_FET)
        plan = build_plan(files, prefix="X")
        out = render(files, plan, {"XN0.m0": MismatchValues(delvto=-0.02)})
        assert set(out) == {0}
        assert ".subckt minifet__mcpatch" in out[0]
        assert instance_line(out[0], "XN0").model == "minifet__mcpatch"

    def test_x_line_inside_an_included_file_is_rewritten_there(self):
        root = "* root\n.include inc1.spice\n.include inc2.spice\nV1 dn0 0 0.9\n.op\n.end\n"
        dut = "XN0 dn0 g 0 0 minifet W=1 L=0.15\n"
        files = closure(root, MINI_FET, dut)
        plan = build_plan(files, selectors=["XN0"])
        out = render(files, plan, {"XN0.m0": MismatchValues(delvto=-0.02)})
        assert set(out) == {0, 2}
        assert ".subckt minifet__mcpatch" in out[0]
        assert instance_line(out[2], "XN0").model == "minifet__mcpatch"


class TestLibrarySections:
    """Only the corner a caller selected is in the circuit."""

    LIB = """\
.lib tt
.subckt minifet d g s b
.param w = 1 l = 0.15
m0 d g s b tt_model w = {w} l = {l}
.model tt_model nmos level = 54 vth0 = 0.7
.ends minifet
.endl

.lib ff
.subckt minifet d g s b
.param w = 1 l = 0.15
m0 d g s b ff_model w = {w} l = {l}
.model ff_model nmos level = 54 vth0 = 0.6
.ends minifet
.endl
"""

    DECK = """\
* sectioned library
.lib inc1.spice tt
XN0 dn0 g 0 0 minifet W=1 L=0.15
V1 dn0 0 0.9
.op
.end
"""

    def test_the_selected_section_is_the_one_that_is_patched(self, tmp_path: Path):
        (tmp_path / "inc1.spice").write_text(self.LIB)
        files = closure(self.DECK, self.LIB, root=tmp_path)
        plan = build_plan(files, prefix="X")
        assert [t.model_name for t in plan.targets] == ["tt_model"]

    def test_an_unselected_second_definition_is_not_an_ambiguity(self, tmp_path: Path):
        (tmp_path / "inc1.spice").write_text(self.LIB)
        files = closure(self.DECK, self.LIB, root=tmp_path)
        out = render(
            files,
            build_plan(files, prefix="X"),
            {"XN0.m0": MismatchValues(delvto=0.01)},
        )
        assert "tt_model" in out[0]
        assert "ff_model" not in out[0]


class TestMultipleInnerDevices:
    PAIR = """\
.subckt pairfet d1 d2 g s b
.param w = 1 l = 0.15
ma d1 g s b pair_model w = {w} l = {l}
mb d2 g s b pair_model w = {w} l = {l}
.model pair_model nmos level = 54 vth0 = 0.7
.ends pairfet
"""

    DECK = f"""\
* two devices in one body
{PAIR}
XP0 da db g 0 0 pairfet W=1 L=0.15
V1 da 0 0.9
.op
.end
"""

    def test_a_whole_instance_rule_addresses_every_inner_device(self):
        plan = build_plan(closure(self.DECK), prefix="X")
        assert [t.ref for t in plan.targets] == ["XP0.ma", "XP0.mb"]

    def test_the_two_devices_take_independent_values(self):
        files = closure(self.DECK)
        plan = build_plan(files, prefix="X")
        out = render(
            files,
            plan,
            {
                "XP0.ma": MismatchValues(delvto=-0.01),
                "XP0.mb": MismatchValues(delvto=0.02),
            },
        )
        view = instance_line(out[0], "XP0")
        assert view.get_param("mc_delvto__ma") == "-0.01"
        assert view.get_param("mc_delvto__mb") == "0.02"
        clone = clone_body(out[0], "pairfet__mcpatch")
        assert instance_line(clone, "ma").get_param("delvto") == "{mc_delvto__ma}"
        assert instance_line(clone, "mb").get_param("delvto") == "{mc_delvto__mb}"

    def test_an_exact_value_must_name_which_device_it_is_for(self):
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(self.DECK), selectors=["XP0"], exact=True)
        assert exc.value.code == "ambiguous_inner_device"
        assert "XP0.<device>" in str(exc.value)

    def test_naming_the_device_resolves_it(self):
        files = closure(self.DECK)
        plan = build_plan(files, selectors=["XP0.mb"], exact=True)
        assert [t.ref for t in plan.targets] == ["XP0.mb"]
        out = render(files, plan, {"XP0.mb": MismatchValues(delvto=-0.02)})
        # The untargeted sibling gets no forwarded pair at all.
        assert "mc_delvto__ma" not in clone_body(out[0], "pairfet__mcpatch")

    def test_a_device_the_body_does_not_hold_is_named_as_missing(self):
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(self.DECK), selectors=["XP0.mz"], exact=True)
        assert exc.value.code == "inner_device_not_found"


class TestRefusals:
    def test_two_subcircuit_levels_are_refused_by_name(self):
        nested = MINI_FET + (".subckt outerfet d g s b\nxin d g s b minifet\n.ends outerfet\n")
        deck = DECK.replace(MINI_FET, nested).replace(
            "XN0 dn0 g 0 0 minifet W=1 L=0.15", "XN0 dn0 g 0 0 outerfet"
        )
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(deck), prefix="XN0")
        assert exc.value.code == "nested_fet_unsupported"

    def test_a_subcircuit_without_a_mos_device_is_a_named_skip(self):
        passive = ".subckt divider a b\nR1 a b 1k\n.ends divider\n"
        deck = DECK.replace("V1 dn0 0 0.9", f"{passive}XD0 dn0 0 divider\nV1 dn0 0 0.9")
        plan = build_plan(closure(deck), prefix="X")
        assert [(s.x_ref, s.code) for s in plan.skips] == [("XD0", "no_mos_at_depth")]
        assert [t.x_ref for t in plan.targets] == ["XN0", "XN1"]

    def test_a_level_1_inner_device_is_refused_with_the_simulator_message(self):
        level1 = MINI_FET.replace("level = 54", "level = 1")
        deck = DECK.replace(MINI_FET, level1)
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(deck), prefix="X")
        assert exc.value.code == "non_bsim_inner_device"
        assert "unknown parameter (delvto)" in str(exc.value)

    def test_a_model_with_no_level_cannot_be_confirmed(self):
        no_level = MINI_FET.replace(" level = 54", "")
        deck = DECK.replace(MINI_FET, no_level)
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(deck), prefix="X")
        assert exc.value.code == "inner_model_unresolved"

    def test_two_active_definitions_of_one_device_are_ambiguous(self):
        deck = DECK.replace(MINI_FET, MINI_FET + MINI_FET)
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(deck), prefix="X")
        assert exc.value.code == "ambiguous_subckt"

    def test_a_preexisting_value_on_the_instance_is_refused(self):
        deck = DECK.replace(
            "XN0 dn0 g 0 0 minifet W=1 L=0.15", "XN0 dn0 g 0 0 minifet W=1 L=0.15 delvto=0.01"
        )
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(deck), prefix="X")
        assert exc.value.code == "preexisting_mismatch_param"

    def test_a_preexisting_value_on_the_inner_device_is_refused(self):
        deck = DECK.replace(
            "m0 d g s b minifet_model w = {w} l = {l}",
            "m0 d g s b minifet_model w = {w} l = {l} mulu0 = 1.1",
        )
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(deck), prefix="X")
        assert exc.value.code == "preexisting_mismatch_param"

    def test_a_library_that_owns_the_knob_itself_is_refused(self):
        deck = DECK.replace(".param w = 1 l = 0.15 nf = 1", ".param w = 1 l = 0.15 delvto = 0")
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(deck), prefix="X")
        assert exc.value.code == "preexisting_mismatch_param"

    def test_the_reserved_parameter_namespace_is_collision_checked(self):
        deck = DECK.replace(
            ".param w = 1 l = 0.15 nf = 1", ".param w = 1 l = 0.15 MC_DELVTO__M0 = 0.5"
        )
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(deck), prefix="X")
        assert exc.value.code == "param_namespace_collision"

    def test_one_reference_declared_twice_is_refused(self):
        twice = "\n.subckt block a\nXN0 a g 0 0 minifet W=1 L=0.15\n.ends block\n"
        deck = DECK.replace("V1 dn0 0 0.9", f"V1 dn0 0 0.9{twice}")
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(deck), selectors=["XN0"])
        assert exc.value.code == "ambiguous_instance_ref"

    def test_a_missing_instance_is_named(self):
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(closure(DECK), selectors=["XZ9"])
        assert exc.value.code == "instance_not_found"

    def test_non_literal_geometry_refuses_a_scaled_sigma(self):
        deck = DECK.replace(
            "XN0 dn0 g 0 0 minifet W=1 L=0.15", "XN0 dn0 g 0 0 minifet W={wn} L=0.15"
        )
        plan = build_plan(closure(deck), selectors=["XN0"])
        with pytest.raises(MismatchPlanError) as exc:
            require_geometry(plan.targets[0])
        assert exc.value.code == "geometry_not_literal"


class TestForwardedParamNaming:
    def test_the_name_is_keyed_by_the_real_inner_reference(self):
        assert forwarded_param("delvto", "M0") == "mc_delvto__m0"
        # Underscores double so the escape for an illegal character stays
        # unambiguous; the reference is still legible in the result.
        assert (
            forwarded_param("mulu0", "msky130_fd_pr__nfet") == "mc_mulu0__msky130__fd__pr____nfet"
        )


class TestPatchedDeckStillValidates:
    """A patched deck has to survive the checks a hand-written one gets.

    The engine writes parameters onto X lines and a cloned subcircuit into the
    deck; a validator that read either as malformed would reject exactly the
    decks this feature produces.
    """

    def _patched(self) -> str:
        files = closure(DECK)
        plan = build_plan(files, prefix="X")
        return render(
            files,
            plan,
            {
                "XN0.m0": MismatchValues(delvto=-0.02, mulu0=1.05),
                "XN1.m0": MismatchValues(delvto=0.03),
            },
        )[0]

    def test_arity_and_reference_checks_pass(self):
        from ltspice_mcp.lib.spice_validator import (
            validate_netlist_arity,
            validate_netlist_directive_refs,
        )

        cards = lex(self._patched()).cards
        assert validate_netlist_arity(cards, simulator="ngspice") == []
        assert validate_netlist_directive_refs(cards) == []

    def test_the_patched_deck_reparses_to_the_same_text(self):
        from ltspice_mcp.lib.spice_lex import emit

        patched = self._patched()
        assert emit(lex(patched).cards) == patched


# Point LTSPICE_MCP_TEST_PDK_ROOT at an installed sky130A PDK root (the
# directory holding ``libs.ref`` and ``libs.tech``) to run the tests below.
# Unset, the path cannot exist and they skip.
PDK_ROOT = Path(os.environ.get("LTSPICE_MCP_TEST_PDK_ROOT") or "/nonexistent/sky130A")
PDK_SPICE = PDK_ROOT / "libs.ref/sky130_fd_pr/spice"


@pytest.mark.skipif(
    not (PDK_SPICE / "sky130_fd_pr__nfet_01v8__tt.pm3.spice").exists(),
    reason="set LTSPICE_MCP_TEST_PDK_ROOT to an installed sky130A PDK root to run these",
)
class TestOpenPdkDevice:
    """The engine against a real foundry device deck, end to end.

    Needs a locally installed PDK, so CI skips it. This is the measurement the
    whole mechanism rests on — two instances of one cell taking different exact
    threshold shifts — reproduced through the engine rather than a hand-patched
    file.
    """

    DECK = f"""\
* per-instance mismatch on a foundry device
.option scale=1.0u
.param mc_mm_switch=0
.param mc_pr_switch=0
.include {PDK_ROOT}/libs.tech/ngspice/parameters/lod.spice
.include {PDK_SPICE}/sky130_fd_pr__nfet_01v8__mismatch.corner.spice
.include {PDK_SPICE}/sky130_fd_pr__nfet_01v8__tt.pm3.spice
Vg g 0 0.9
XN0 dn0 g 0 0 sky130_fd_pr__nfet_01v8 W=1 L=0.15
XN1 dn1 g 0 0 sky130_fd_pr__nfet_01v8 W=1 L=0.15
Vn0 dn0 0 0.9
Vn1 dn1 0 0.9
.control
op
print @m.xn0.msky130_fd_pr__nfet_01v8[vth]
print @m.xn1.msky130_fd_pr__nfet_01v8[vth]
.endc
.end
"""

    def _patched(self, tmp_path: Path) -> str:
        from ltspice_mcp.lib.subckt_mismatch import closure_from_deck

        deck = tmp_path / "pdk.spice"
        deck.write_text(self.DECK)
        files = closure_from_deck(deck, self.DECK)
        plan = build_plan(files, prefix="X")
        assert [t.ref for t in plan.targets] == [
            "XN0.msky130_fd_pr__nfet_01v8",
            "XN1.msky130_fd_pr__nfet_01v8",
        ]
        return render(
            files,
            plan,
            {
                "XN0.msky130_fd_pr__nfet_01v8": MismatchValues(delvto=0.0),
                "XN1.msky130_fd_pr__nfet_01v8": MismatchValues(delvto=-0.020),
            },
        )[0]

    def test_the_written_deck_carries_each_shift_on_its_own_instance(self, tmp_path: Path):
        patched = self._patched(tmp_path)
        param = forwarded_param("delvto", "msky130_fd_pr__nfet_01v8")
        assert instance_line(patched, "XN0").get_param(param) == "0"
        assert instance_line(patched, "XN1").get_param(param) == "-0.02"
        assert instance_line(patched, "XN1").model == "sky130_fd_pr__nfet_01v8__mcpatch"
        # One patched copy for the device type, however many instances use it.
        # The library's own spacing survives the rename, hence the loose match.
        assert len(re.findall(r"\.subckt\s+sky130_fd_pr__nfet_01v8__mcpatch", patched)) == 1

    @pytest.mark.skipif(not __import__("shutil").which("ngspice"), reason="ngspice is not on PATH")
    def test_the_simulated_threshold_shift_equals_the_requested_one(self, tmp_path: Path):
        import shutil
        import subprocess

        run_deck = tmp_path / "run.spice"
        run_deck.write_text(self._patched(tmp_path))
        result = subprocess.run(
            [shutil.which("ngspice") or "ngspice", "-b", str(run_deck)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        printed = [
            float(match.group(1))
            for match in re.finditer(r"\[vth\]\s*=\s*([-+0-9.eE]+)", result.stdout + result.stderr)
        ]
        assert len(printed) == 2, result.stdout + result.stderr
        # delvto adds to the signed vth0, so the requested -20 mV lands as a
        # 20 mV drop in the printed threshold — exactly, not approximately.
        assert printed[1] - printed[0] == pytest.approx(-0.020, abs=1e-9)


class TestIdentifierEncoding:
    """Forwarded names have to be legal parameter identifiers, and distinct."""

    def test_ordinary_references_read_as_themselves(self):
        assert encode_ref("M0") == "m0"
        assert encode_ref("mn12") == "mn12"

    def test_case_only_differences_fold_together(self):
        # SPICE does not distinguish them, so neither does the encoding.
        assert encode_ref("M0") == encode_ref("m0")

    def test_illegal_characters_are_encoded_not_pasted(self):
        assert encode_ref("m.a") == "m_2ea"
        assert encode_ref("m-a") == "m_2da"
        assert encode_ref("m$a") == "m_24a"
        for ref in ("m.a", "m-a", "m$a", "m_a"):
            assert re.fullmatch(r"[a-z0-9_]+", encode_ref(ref)), ref

    def test_the_encoding_is_injective(self):
        refs = ["m_a", "m.a", "m-a", "ma", "m__a", "m_2ea", "m2ea", "m$a", "m_5fa"]
        assert len({encode_ref(ref) for ref in refs}) == len(refs)

    def test_a_character_with_no_representation_is_refused(self):
        with pytest.raises(MismatchPlanError) as exc:
            encode_ref("mµa")
        assert exc.value.code == "unencodable_device_ref"

    def test_an_illegal_reference_still_produces_a_legal_deck(self):
        deck = DECK.replace("m0 d g s b", "m.0 d g s b").replace(
            ".param w = 1 l = 0.15 nf = 1", ".param w = 1 l = 0.15"
        )
        files = closure(deck)
        plan = build_plan(files, prefix="X")
        out = render(files, plan, {"XN0.m.0": MismatchValues(delvto=-0.02)})
        assert instance_line(out[0], "XN0").get_param("mc_delvto__m_2e0") == "-0.02"
        assert "mc_delvto__m.0" not in out[0]


class TestTrailingComments:
    """A comment on the parameter card must not swallow the forwarded defaults."""

    def test_defaults_land_in_the_code_not_the_comment(self):
        deck = DECK.replace(
            ".param w = 1 l = 0.15 nf = 1", ".param w = 1 l = 0.15 nf = 1 ; channel width"
        )
        files = closure(deck)
        plan = build_plan(files, prefix="X")
        out = render(files, plan, {"XN0.m0": MismatchValues(delvto=-0.02)})
        text = clone_body(out[0], "minifet__mcpatch")
        param_line = next(line for line in text.splitlines() if line.startswith(".param"))
        code, _, comment = param_line.partition(";")
        assert "mc_delvto__m0 = 0" in code
        assert "mc_delvto__m0" not in comment
        assert "channel width" in comment

    def test_the_patched_clone_still_declares_what_the_inner_card_references(self):
        deck = DECK.replace(
            ".param w = 1 l = 0.15 nf = 1", ".param w = 1 l = 0.15 nf = 1 ; channel width"
        )
        files = closure(deck)
        plan = build_plan(files, prefix="X")
        out = render(files, plan, {"XN0.m0": MismatchValues(delvto=-0.02)})
        cards = lex(clone_body(out[0], "minifet__mcpatch")).cards
        declared = {
            token.key.casefold()
            for card in cards
            if card.kind == "param"
            for token in tokenize_body(card.body)[1:]
            if token.kind == TokenKind.KEY_VALUE and token.key
        }
        assert {"mc_delvto__m0", "mc_mulu0__m0"} <= declared


class TestOnePhysicalFileUnderTwoSections:
    """One library, entered twice under different corner sections.

    Sections exist so a library can carry one device per corner under one name.
    Reading every selected section as equally live turns that into two
    definitions of the same subcircuit and refuses a deck the simulator runs.
    """

    SHARED = """\
.lib tt
.subckt minifet d g s b
.param w = 1 l = 0.15
m0 d g s b tt_model w = {w} l = {l}
.model tt_model nmos level = 54 vth0 = 0.7
.ends minifet
XTT dtt g 0 0 minifet W=1 L=0.15
.endl

.lib ff
.subckt minifet d g s b
.param w = 1 l = 0.15
m0 d g s b ff_model w = {w} l = {l}
.model ff_model nmos level = 54 vth0 = 0.6
.ends minifet
XFF dff g 0 0 minifet W=1 L=0.15
.endl
"""

    DECK = """\
* both corners of one library
.lib inc1.spice tt
.lib inc1.spice ff
V1 dtt 0 0.9
.op
.end
"""

    def _files(self, tmp_path: Path) -> list[ClosureFile]:
        (tmp_path / "inc1.spice").write_text(self.SHARED)
        return closure(self.DECK, self.SHARED, root=tmp_path)

    def test_each_instance_resolves_to_the_device_beside_it(self, tmp_path: Path):
        plan = build_plan(self._files(tmp_path), prefix="X")
        assert {t.x_ref: t.model_name for t in plan.targets} == {
            "XTT": "tt_model",
            "XFF": "ff_model",
        }

    def test_both_corners_get_their_own_patched_copy(self, tmp_path: Path):
        files = self._files(tmp_path)
        plan = build_plan(files, prefix="X")
        out = render(
            files,
            plan,
            {
                "XTT.m0": MismatchValues(delvto=-0.01),
                "XFF.m0": MismatchValues(delvto=0.02),
            },
        )
        # The instances live in the library, so that is the file that changed;
        # the patched copies land in the deck the simulator is handed.
        assert set(out) == {0, 1}
        assert instance_line(out[1], "XTT").get_param("mc_delvto__m0") == "-0.01"
        assert instance_line(out[1], "XFF").get_param("mc_delvto__m0") == "0.02"
        assert instance_line(out[1], "XTT").model != instance_line(out[1], "XFF").model
        assert "tt_model" in out[0] and "ff_model" in out[0]

    def test_two_sections_defining_different_devices_are_never_ambiguous(self, tmp_path: Path):
        shared = self.SHARED.replace(
            "minifet d g s b\n.param w = 1 l = 0.15\nm0 d g s b ff",
            "fffet d g s b\n.param w = 1 l = 0.15\nm0 d g s b ff",
        ).replace(".ends minifet\nXFF dff g 0 0 minifet", ".ends fffet\nXFF dff g 0 0 fffet")
        (tmp_path / "inc1.spice").write_text(shared)
        files = closure(self.DECK, shared, root=tmp_path)
        plan = build_plan(files, prefix="X")
        assert {t.subckt for t in plan.targets} == {"minifet", "fffet"}

    def test_an_instance_outside_both_sections_is_still_ambiguous(self, tmp_path: Path):
        # Nothing says which corner it means, and the engine will not pick one.
        deck = self.DECK.replace("V1 dtt 0 0.9", "XR0 dr g 0 0 minifet W=1 L=0.15\nV1 dtt 0 0.9")
        (tmp_path / "inc1.spice").write_text(self.SHARED)
        files = closure(deck, self.SHARED, root=tmp_path)
        with pytest.raises(MismatchPlanError) as exc:
            build_plan(files, selectors=["XR0"])
        assert exc.value.code == "ambiguous_subckt"
