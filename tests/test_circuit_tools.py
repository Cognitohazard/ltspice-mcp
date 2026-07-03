"""Integration tests for circuit management tools."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ltspice_mcp.errors import NetlistError, PathSecurityError
from ltspice_mcp.lib.component_value import apply_value_to_instance
from ltspice_mcp.lib.spice_lex import emit, lex
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.circuit import (
    handle_create_netlist,
    handle_create_schematic,
    handle_diff_circuit,
    handle_edit_directive,
    handle_list_components,
    handle_parameter,
    handle_read_circuit,
    handle_set_component_value,
    handle_validate_netlist,
)


@pytest.mark.asyncio
class TestCreateNetlist:
    async def test_creates_file(self, state_no_sim: SessionState, work_dir: Path):
        result = await handle_create_netlist(
            {"name": "test", "content": "* test\nR1 1 0 1k\nV1 1 0 1\n"},
            state_no_sim,
        )
        created = work_dir / "test.cir"
        assert created.exists()
        content = created.read_text()
        assert content.startswith("* test")
        assert "R1 1 0 1k" in content
        assert "test.cir" in result.content[0].text

    async def test_appends_end_directive(self, state_no_sim: SessionState, work_dir: Path):
        await handle_create_netlist(
            {"name": "noend", "content": "* test\nR1 1 0 1k\n"},
            state_no_sim,
        )
        content = (work_dir / "noend.cir").read_text()
        assert content.strip().upper().endswith(".END")

    async def test_rejects_duplicate(self, state_no_sim: SessionState, work_dir: Path):
        await handle_create_netlist(
            {"name": "dup", "content": "* test\nR1 1 0 1k\n"},
            state_no_sim,
        )
        with pytest.raises(NetlistError, match="already exists"):
            await handle_create_netlist(
                {"name": "dup", "content": "* test\nR1 1 0 1k\n"},
                state_no_sim,
            )

    async def test_rejects_path_escape(self, state_no_sim: SessionState):
        with pytest.raises(PathSecurityError):
            await handle_create_netlist(
                {"name": "../../etc/evil", "content": "* test\n"},
                state_no_sim,
            )

    async def test_rejects_empty_content(self, state_no_sim: SessionState, work_dir: Path):
        # Regression: empty content used to reach spicelib, which surfaced a
        # cryptic 'Expected pattern "^\\*" not found'. Reject it up front with a
        # clear message and leave no file behind.
        with pytest.raises(NetlistError, match="empty"):
            await handle_create_netlist({"name": "empty", "content": ""}, state_no_sim)
        assert not (work_dir / "empty.cir").exists()

    async def test_rejects_whitespace_only_content(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        with pytest.raises(NetlistError, match="empty"):
            await handle_create_netlist({"name": "ws", "content": "  \n\t\n"}, state_no_sim)

    async def test_overwrite_replaces_existing(self, state_no_sim: SessionState, work_dir: Path):
        """``overwrite=True`` skips the FileExistsError path so iterating on
        a design doesn't force read+edit roundtrips."""
        await handle_create_netlist(
            {"name": "ow", "content": "* v1\nR1 1 0 1k\n"},
            state_no_sim,
        )
        await handle_create_netlist(
            {"name": "ow", "content": "* v2\nR1 1 0 5k\n", "overwrite": True},
            state_no_sim,
        )
        path = work_dir / "ow.cir"
        assert "v2" in path.read_text()
        assert "5k" in path.read_text()


@pytest.mark.asyncio
class TestReadCircuit:
    async def test_reads_content_and_components(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        result = await handle_read_circuit({"path": sample_netlist.name}, state_no_sim)
        text = result.content[0].text
        assert "R1" in text
        assert "C1" in text
        assert "V1" in text
        # Verify actual component values from the parsed netlist
        assert "1k" in text
        assert "100n" in text

    async def test_file_not_found(self, state_no_sim: SessionState):
        with pytest.raises(NetlistError, match="not found"):
            await handle_read_circuit({"path": "nonexistent.cir"}, state_no_sim)

    async def test_path_escape_blocked(self, state_no_sim: SessionState):
        with pytest.raises(PathSecurityError):
            await handle_read_circuit({"path": "/etc/passwd"}, state_no_sim)


@pytest.mark.asyncio
class TestListComponents:
    async def test_lists_all(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_list_components({"path": sample_netlist.name}, state_no_sim)
        text = result.content[0].text
        assert "R1" in text
        assert "C1" in text
        assert "V1" in text

    async def test_prefix_filter(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_list_components(
            {"path": sample_netlist.name, "prefix": "R"}, state_no_sim
        )
        text = result.content[0].text
        assert "R1" in text
        assert "C1" not in text

    async def test_no_match_prefix(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_list_components(
            {"path": sample_netlist.name, "prefix": "Q"}, state_no_sim
        )
        assert "No components" in result.content[0].text

    async def test_single_reference(self, state_no_sim: SessionState, sample_netlist: Path):
        """Single-component lookup via 'reference' parameter."""
        result = await handle_list_components(
            {"path": sample_netlist.name, "reference": "R1"}, state_no_sim
        )
        assert "1k" in result.content[0].text

    async def test_case_insensitive_reference(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        result = await handle_list_components(
            {"path": sample_netlist.name, "reference": "r1"}, state_no_sim
        )
        assert "1k" in result.content[0].text

    async def test_nonexistent_reference(self, state_no_sim: SessionState, sample_netlist: Path):
        with pytest.raises(NetlistError, match="not found"):
            await handle_list_components(
                {"path": sample_netlist.name, "reference": "R99"},
                state_no_sim,
            )

    async def test_b_source_does_not_crash(self, state_no_sim: SessionState, work_dir: Path):
        """A behavioural source whose value has commas inside ``if(...)``
        defeats spicelib's component-line regex. ``list_components`` used to
        return ``Internal error in ltspice_list_components``; the shared lexer
        now preserves the full function-call value."""
        cir = work_dir / "with_b.cir"
        cir.write_text(
            "* B-source torture test\n"
            "R1 a b 1k\n"
            "B1 amp 0 V = if(3.5*V(vp)>10, 10, if(3.5*V(vp)<-10, -10, 3.5*V(vp)))\n"
            "C1 b 0 100n\n"
            ".tran 0 1m\n"
            ".end\n"
        )
        result = await handle_list_components({"path": cir.name}, state_no_sim)
        text = result.content[0].text
        # All three components should appear; the B-source's value is
        # parsed as a full KEY=VALUE function call rather than truncated.
        assert "R1" in text
        assert "B1" in text
        assert "C1" in text
        assert "V=if(3.5*V(vp)>10" in text
        assert "<unparseable>" not in text

    async def test_b_source_operator_after_call_reads_intact(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """An unbraced behavioural value with an operator after a V() call
        (``V=V(in)*2``) is a valid LTspice/ngspice form. It used to read back
        as ``<unparseable>`` because the lexer left the ``*2`` as a stray
        token; the whole expression must now round-trip on read."""
        cir = work_dir / "bmul.cir"
        cir.write_text(
            "* operator-after-call B-source\n"
            "V1 in 0 1\n"
            "B1 out 0 V=V(in)*2\n"
            "R1 out 0 1k\n"
            ".op\n"
            ".end\n"
        )
        result = await handle_list_components({"path": cir.name, "reference": "B1"}, state_no_sim)
        text = result.content[0].text
        assert "V=V(in)*2" in text
        assert "<unparseable>" not in text


@pytest.mark.asyncio
class TestReadCircuitDegrades:
    async def test_b_source_degrades_gracefully(self, state_no_sim: SessionState, work_dir: Path):
        """Same bug surfaced through ``read_circuit`` (which iterates every
        component, not just the prefix-filtered subset)."""
        from ltspice_mcp.tools.circuit import handle_read_circuit

        cir = work_dir / "with_b.cir"
        cir.write_text(
            "* B-source torture test\n"
            "R1 a b 1k\n"
            "B1 amp 0 V = if(3.5*V(vp)>10, 10, if(3.5*V(vp)<-10, -10, 3.5*V(vp)))\n"
            ".tran 0 1m\n"
            ".end\n"
        )
        result = await handle_read_circuit({"path": cir.name, "format": "json"}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        refs = {c["reference"] for c in data["components"]}
        assert "R1" in refs
        assert "B1" in refs

    async def test_source_function_spec_value_intact(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A PULSE/SIN/PWL source value must read back whole, not truncated to the
        # parenthesised args with the function name dropped.
        cir = work_dir / "pulse.cir"
        cir.write_text("* p\nV1 in 0 PULSE(0 5 0 1n 1n 1m 2m)\nR1 in 0 1k\n.END\n")
        result = await handle_read_circuit({"path": cir.name, "format": "json"}, state_no_sim)
        comps = {c["reference"]: c["value"] for c in result.structuredContent["components"]}
        assert comps["V1"] == "PULSE(0 5 0 1n 1n 1m 2m)"


@pytest.mark.asyncio
class TestParameter:
    async def test_get_params(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_parameter({"path": sample_netlist.name}, state_no_sim)
        text = result.content[0].text
        assert "RVAL" in text or "Rval" in text

    async def test_read_all_preserves_source_casing(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        # Regression: spicelib's get_all_parameter_names() uppercases
        # ('Rval'->'RVAL'), but the read-all projection should echo the verbatim
        # on-disk casing recovered from the file text.
        result = await handle_parameter({"path": sample_netlist.name}, state_no_sim)
        text = result.content[0].text
        assert "Rval" in text
        assert "RVAL" not in text
        assert "Rval" in result.structuredContent["parameters"]

    async def test_no_params(self, state_no_sim: SessionState, work_dir: Path):
        p = work_dir / "noparam.cir"
        p.write_text("* test\nR1 1 0 1k\n.END\n")
        result = await handle_parameter({"path": "noparam.cir"}, state_no_sim)
        assert "No .PARAM" in result.content[0].text

    async def test_set_param(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_parameter(
            {"path": sample_netlist.name, "name": "Rval", "value": "2k"},
            state_no_sim,
        )
        assert "Rval" in result.content[0].text

        # Verify value was actually written
        params = await handle_parameter({"path": sample_netlist.name}, state_no_sim)
        assert "2k" in params.content[0].text

    async def test_set_param_does_not_leave_batch_instruction_comment(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Creating a NEW .param via spicelib appends a "; Batch instruction"
        # comment to the rendered line. The save path strips that leaked marker
        # while keeping the directive and its value intact.
        cir = work_dir / "freshparam.cir"
        cir.write_text("* fresh\nR1 in 0 1k\n.END\n")
        await handle_parameter({"path": cir.name, "name": "Gain", "value": "3"}, state_no_sim)
        text = cir.read_text()
        assert "Batch instruction" not in text
        assert ".param" in text.lower()
        assert "Gain" in text
        assert "3" in text

    async def test_set_param_preserves_user_comment_on_other_lines(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # The batch-instruction strip is scoped to .param lines, so a
        # user-authored '; note' on an element line must survive an unrelated
        # param-set untouched.
        cir = work_dir / "withnote.cir"
        cir.write_text("* note test\nR1 n1 0 1k ; my note\n.END\n")
        await handle_parameter({"path": cir.name, "name": "Gain", "value": "3"}, state_no_sim)
        text = cir.read_text()
        assert "; my note" in text
        assert "Batch instruction" not in text

    async def test_add_then_delete_param_round_trips(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Undoing an accidentally-added parameter: delete must match by name
        # even though spicelib reformatted the line on write.
        cir = work_dir / "delparam.cir"
        cir.write_text("* del\nR1 in 0 1k\n.END\n")
        await handle_parameter({"path": cir.name, "name": "Gain", "value": "3"}, state_no_sim)
        assert "gain" in cir.read_text().lower()

        result = await handle_parameter(
            {"path": cir.name, "name": "Gain", "delete": True}, state_no_sim
        )
        assert "Deleted" in result.content[0].text
        assert "gain" not in cir.read_text().lower()

    async def test_delete_one_of_several_params_on_a_line_keeps_siblings(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A .param line can define several parameters; deleting one must not
        # take its siblings down with it (regression — whole-line removal did).
        cir = work_dir / "multi.cir"
        cir.write_text("* m\nR1 in 0 1k\n.param A=1 B=2 C=3\n.END\n")

        # Delete the FIRST parameter on the line.
        await handle_parameter({"path": cir.name, "name": "A", "delete": True}, state_no_sim)
        keys = {
            k.lower()
            for k in (await handle_parameter({"path": cir.name}, state_no_sim)).structuredContent[
                "parameters"
            ]
        }
        assert keys == {"b", "c"}, keys

        # Delete a NON-first parameter (the whole-line approach reported 'not found').
        await handle_parameter({"path": cir.name, "name": "C", "delete": True}, state_no_sim)
        keys2 = {
            k.lower()
            for k in (await handle_parameter({"path": cir.name}, state_no_sim)).structuredContent[
                "parameters"
            ]
        }
        assert keys2 == {"b"}, keys2

    async def test_delete_one_of_several_params_on_asc_directive_keeps_siblings(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        asc = work_dir / "multi.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\nTEXT 0 0 Left 2 !.param A=1 B=2\n")
        await handle_parameter({"path": asc.name, "name": "A", "delete": True}, state_no_sim)
        keys = {
            k.lower()
            for k in (await handle_parameter({"path": asc.name}, state_no_sim)).structuredContent[
                "parameters"
            ]
        }
        assert keys == {"b"}, keys

    async def test_delete_missing_param_raises(self, state_no_sim: SessionState, work_dir: Path):
        cir = work_dir / "nodel.cir"
        cir.write_text("* x\nR1 in 0 1k\n.END\n")
        with pytest.raises(NetlistError, match="not found"):
            await handle_parameter(
                {"path": cir.name, "name": "Nope", "delete": True}, state_no_sim
            )

    async def test_delete_requires_name_and_excludes_value(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        cir = work_dir / "guard.cir"
        cir.write_text("* x\nR1 in 0 1k\n.PARAM Gain=3\n.END\n")
        with pytest.raises(NetlistError, match="'delete' requires 'name'"):
            await handle_parameter({"path": cir.name, "delete": True}, state_no_sim)
        with pytest.raises(NetlistError, match="not both"):
            await handle_parameter(
                {"path": cir.name, "name": "Gain", "value": "5", "delete": True},
                state_no_sim,
            )


@pytest.mark.asyncio
class TestSetComponentValue:
    async def test_set_single(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_set_component_value(
            {"path": sample_netlist.name, "reference": "R1", "value": "4.7k"},
            state_no_sim,
        )
        assert "4.7k" in result.content[0].text

        # Verify persisted
        result2 = await handle_list_components(
            {"path": sample_netlist.name, "reference": "R1"}, state_no_sim
        )
        assert "4.7k" in result2.content[0].text

    async def test_batch_set(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_set_component_value(
            {
                "path": sample_netlist.name,
                "values": {"R1": "10k", "C1": "47n"},
            },
            state_no_sim,
        )
        assert "2 component" in result.content[0].text

        r1 = await handle_list_components(
            {"path": sample_netlist.name, "reference": "R1"}, state_no_sim
        )
        assert "10k" in r1.content[0].text
        c1 = await handle_list_components(
            {"path": sample_netlist.name, "reference": "C1"}, state_no_sim
        )
        assert "47n" in c1.content[0].text

    async def test_invalid_values_type(self, state_no_sim: SessionState, sample_netlist: Path):
        with pytest.raises(ValidationError):
            await handle_set_component_value(
                {"path": sample_netlist.name, "values": "not a dict"},
                state_no_sim,
            )

    async def test_missing_args(self, state_no_sim: SessionState, sample_netlist: Path):
        with pytest.raises(NetlistError, match="Provide either"):
            await handle_set_component_value(
                {"path": sample_netlist.name},
                state_no_sim,
            )

    async def test_batch_with_unknown_ref_is_atomic(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        """A batch ``set_component_value`` with one missing ref used
        to crash AFTER applying earlier writes, leaving the netlist
        half-modified. Validation must happen before any write."""
        before = sample_netlist.read_bytes()  # noqa: ASYNC240
        with pytest.raises(NetlistError, match="not found"):
            await handle_set_component_value(
                {
                    "path": sample_netlist.name,
                    "values": {"R1": "20k", "C1": "47n", "RX": "1k"},
                },
                state_no_sim,
            )
        # Nothing should have been written.
        assert sample_netlist.read_bytes() == before  # noqa: ASYNC240

    async def test_value_with_whitespace_rejected(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        """``set_component_value(R1, "hello world")`` used to write a
        space-separated value into the netlist line, turning ``hello`` into
        a phantom node and ``world`` into a stray token — irrecoverable
        without manual editing."""
        with pytest.raises(NetlistError, match="whitespace"):
            await handle_set_component_value(
                {"path": sample_netlist.name, "reference": "R1", "value": "hello world"},
                state_no_sim,
            )

    async def test_brace_expression_allowed(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        """SPICE expressions in braces include spaces and must NOT be rejected."""
        await handle_set_component_value(
            {
                "path": sample_netlist.name,
                "reference": "R1",
                "value": "{ 1k * 2 }",
            },
            state_no_sim,
        )

    async def test_mosfet_value_with_params_replaces_both(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """Setting a MOSFET value of ``"NMOS1 W=10u L=1u"`` against an
        existing ``M1 ... NMOS1 W=20u L=1u`` element used to leave both
        param sets in place (``... NMOS1 W=10u L=1u W=20u L=1u``) because
        spicelib's ``set_component_value`` only writes the model token. The
        wrapper now splits the trailing ``W=/L=`` tokens and routes them
        through ``set_component_parameters``."""
        cir = work_dir / "m.cir"
        cir.write_text(
            "* MOSFET param replacement test\n"
            ".MODEL NMOS1 NMOS(VTO=0.7 KP=100u)\n"
            ".MODEL NMOS2 NMOS(VTO=0.5 KP=80u)\n"
            "VDD vdd 0 5\n"
            "M1 vdd vg 0 0 NMOS1 W=20u L=1u\n"
            "Vg vg 0 1\n"
            ".END\n"
        )
        await handle_set_component_value(
            {"path": cir.name, "reference": "M1", "value": "NMOS2 W=10u L=2u"},
            state_no_sim,
        )
        text = cir.read_text()
        m1_lines = [ln for ln in text.splitlines() if ln.startswith("M1")]
        assert len(m1_lines) == 1, f"expected one M1 line, got {m1_lines!r}"
        line = m1_lines[0]
        # New params replace the old ones — no duplicate W=/L= tokens left.
        assert line.count("W=") == 1, f"duplicate W= in {line!r}"
        assert line.count("L=") == 1, f"duplicate L= in {line!r}"
        assert "W=10u" in line
        assert "L=2u" in line
        assert "W=20u" not in line


@pytest.mark.asyncio
class TestEditDirective:
    async def test_add_directive(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_edit_directive(
            {"path": sample_netlist.name, "action": "add", "instruction": ".tran 0 10m 0 1u"},
            state_no_sim,
        )
        assert ".tran" in result.content[0].text

    async def test_rejects_non_dot_directive(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        with pytest.raises(NetlistError, match=r"must start with '\.'"):
            await handle_edit_directive(
                {"path": sample_netlist.name, "action": "add", "instruction": "tran 0 10m"},
                state_no_sim,
            )

    async def test_rejects_param_directive(self, state_no_sim: SessionState, sample_netlist: Path):
        # F3: spicelib refuses .param via add_instruction (it surfaced as an
        # opaque "Internal error"); the handler now pre-empts it with a clean
        # message pointing to the 'parameter' tool. Reproduces on .cir too.
        with pytest.raises(NetlistError, match="parameter"):
            await handle_edit_directive(
                {"path": sample_netlist.name, "action": "add", "instruction": ".param foo=1"},
                state_no_sim,
            )
        with pytest.raises(NetlistError, match="parameter"):
            await handle_edit_directive(
                {"path": sample_netlist.name, "action": "add", "instruction": ".PARAM bar=2"},
                state_no_sim,
            )

    async def test_edit_directive_description_mentions_param_refusal(
        self, state_no_sim: SessionState
    ):
        # The registered tool description must steer callers away from adding a
        # '.param' here and point them at the 'parameter' tool, since spicelib's
        # add_instruction refuses .param with an opaque error.
        edit_def = next(td for td in state_no_sim.tool_defs if td.name == "edit_directive")
        desc = edit_def.description or ""
        assert "param" in desc.lower()
        assert "parameter" in desc

    async def test_remove_directive(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_edit_directive(
            {"path": sample_netlist.name, "action": "remove", "instruction": ".ac dec 100 1 1Meg"},
            state_no_sim,
        )
        assert "Removed" in result.content[0].text

    async def test_remove_literal_with_parens(self, state_no_sim: SessionState, work_dir: Path):
        """directives containing ``(``/``)`` (every .meas/.four
        on V(...)/I(...)) used to silently no-op because the legacy
        heuristic routed them through the regex path where unescaped
        parens became capture groups. Verify literal match works AND the
        directive actually disappears from the file."""
        cir = work_dir / "with_parens.cir"
        cir.write_text(
            "* with parens\nV1 in 0 5\n.tran 1m\n"
            ".meas tran v_avg AVG V(in)\n"
            ".four 1k V(in)\n.end\n"
        )
        await handle_edit_directive(
            {"path": cir.name, "action": "remove", "instruction": ".meas tran v_avg AVG V(in)"},
            state_no_sim,
        )
        await handle_edit_directive(
            {"path": cir.name, "action": "remove", "instruction": ".four 1k V(in)"},
            state_no_sim,
        )
        body = cir.read_text()
        assert ".meas tran v_avg" not in body
        assert ".four 1k" not in body

    async def test_remove_no_match_raises(self, state_no_sim: SessionState, work_dir: Path):
        """Silent success when nothing matched was the trap that v4-N1
        exposed — typos or stale lines made the user believe they cleaned
        the netlist when nothing changed. Now it errors."""
        cir = work_dir / "no_match.cir"
        cir.write_text("* test\nV1 a 0 5\n.tran 1m\n.end\n")
        with pytest.raises(NetlistError, match="No directive or comment matched"):
            await handle_edit_directive(
                {"path": cir.name, "action": "remove", "instruction": ".does_not_exist"},
                state_no_sim,
            )

    async def test_remove_regex_explicit(self, state_no_sim: SessionState, work_dir: Path):
        """``regex:`` prefix still works for callers that intend regex."""
        cir = work_dir / "regex.cir"
        cir.write_text("* regex test\nV1 a 0 5\n.tran 1m\n.meas tran v_a MAX V(a)\n.end\n")
        await handle_edit_directive(
            {"path": cir.name, "action": "remove", "instruction": "regex:^\\.meas .*"},
            state_no_sim,
        )
        body = cir.read_text()
        assert ".meas" not in body


@pytest.mark.asyncio
class TestCreateSchematic:
    async def test_seeds_empty_asc(self, state_no_sim: SessionState, work_dir: Path):
        result = await handle_create_schematic({"name": "seed"}, state_no_sim)
        out = work_dir / "seed.asc"
        assert out.exists()
        body = out.read_text()
        assert body.startswith("Version 4")
        assert "SHEET 1 880 680" in body
        assert "seed.asc" in result.content[0].text

    async def test_custom_dimensions(self, state_no_sim: SessionState, work_dir: Path):
        await handle_create_schematic({"name": "small", "width": 320, "height": 240}, state_no_sim)
        body = (work_dir / "small.asc").read_text()
        assert "SHEET 1 320 240" in body

    async def test_rejects_duplicate(self, state_no_sim: SessionState, work_dir: Path):
        await handle_create_schematic({"name": "dup"}, state_no_sim)
        with pytest.raises(NetlistError, match="already exists"):
            await handle_create_schematic({"name": "dup"}, state_no_sim)


@pytest.mark.asyncio
class TestValidateNetlist:
    async def _validate(self, state: SessionState, work_dir: Path, name: str, content: str):
        """Write ``content`` to ``name``, validate it, return the data dict."""
        (work_dir / name).write_text(content)
        result = await handle_validate_netlist({"path": name}, state)
        data = result.structuredContent
        assert data is not None
        return data

    async def test_clean_netlist(self, state_no_sim: SessionState, work_dir: Path):
        cir = work_dir / "clean.cir"
        cir.write_text("* clean\nVin in 0 1\nR1 in 0 1k\n.tran 0 1m\n.end\n")
        result = await handle_validate_netlist({"path": cir.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert data["issue_count"] == 0

    async def test_empty_file_is_error(self, state_no_sim: SessionState, work_dir: Path):
        # LTspice fails an empty deck immediately at line 1; the static
        # gate must not report it as passing.
        data = await self._validate(state_no_sim, work_dir, "empty.cir", "")
        empties = [iss for iss in data["issues"] if "empty" in iss["message"].lower()]
        assert len(empties) == 1, data["issues"]
        assert empties[0]["severity"] == "error"

    async def test_whitespace_only_file_is_error(self, state_no_sim: SessionState, work_dir: Path):
        data = await self._validate(state_no_sim, work_dir, "ws.cir", "  \n\t\n\n")
        assert any(
            iss["severity"] == "error" and "empty" in iss["message"].lower()
            for iss in data["issues"]
        ), data["issues"]

    async def test_asc_without_directives_not_flagged_empty(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # The .asc branch validates the schematic's directive lines only —
        # a schematic with no SPICE directives is not an empty netlist.
        await handle_create_schematic({"name": "blank"}, state_no_sim)
        result = await handle_validate_netlist({"path": "blank.asc"}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert data["issue_count"] == 0, data["issues"]

    async def test_dangling_node_warned(self, state_no_sim: SessionState, work_dir: Path):
        data = await self._validate(
            state_no_sim,
            work_dir,
            "dangling.cir",
            "* dangling\nV1 in 0 1\nR1 in float 1k\n.tran 0 1m\n.end\n",
        )
        dangling = [iss for iss in data["issues"] if "'float'" in iss["message"]]
        assert len(dangling) == 1, data["issues"]
        assert dangling[0]["severity"] == "warning"
        assert "R1" in dangling[0]["message"]

    async def test_bias_topology_floating_gate_warned(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A MOSFET gate reached only through a coupling cap has no DC path
        # to ground — a warning naming the gate and the transistor.
        data = await self._validate(
            state_no_sim,
            work_dir,
            "floatgate.cir",
            "* float gate\nV1 vdd 0 5\nVin in 0 AC 1\nC1 in g 1u\n"
            "M1 d g s 0 NMOS\nRd vdd d 1k\nRs s 0 1k\n.op\n.end\n",
        )
        bias = [iss for iss in data["issues"] if "'g'" in iss["message"]]
        assert len(bias) == 1, data["issues"]
        assert bias[0]["severity"] == "warning"
        assert "M1" in bias[0]["message"]
        assert "no DC path to ground" in bias[0]["message"]

    async def test_bias_topology_not_run_on_asc(self, state_no_sim: SessionState, work_dir: Path):
        # A capacitive island (degree-2 floating node) embedded in .asc
        # directive text would fire the bias pass on a netlist — but the
        # schematic's wires are invisible here, so the pass must not run.
        data = await self._validate(
            state_no_sim,
            work_dir,
            "caps.asc",
            "Version 4\nSHEET 1 880 680\n"
            "TEXT 16 16 Left 2 !.op\n"
            "TEXT 16 48 Left 2 !C1 na mid 1n\n"
            "TEXT 16 80 Left 2 !C2 mid nb 1n\n",
        )
        assert data["issue_count"] == 0, data["issues"]

    async def test_title_line_words_not_phantom_nodes(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Line 1 of a .cir/.net deck is the free-text title. The lexer has
        # no title concept and reads it as an instance card, but its words
        # must not be counted as circuit nodes.
        data = await self._validate(
            state_no_sim,
            work_dir,
            "titled.cir",
            "Voltage divider test circuit\n"
            "V1 in 0 1\nR1 in out 1k\nR2 out 0 1k\n.tran 0 1m\n.end\n",
        )
        assert data["issue_count"] == 0, data["issues"]

    async def test_title_line_short_element_letter_not_arity_flagged(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A title starting with an element letter and too few words ("RC filter"
        # → reads as an R-element with a single node) must not be arity-flagged:
        # line 1 is the free-text title, dropped before the arity pass.
        data = await self._validate(
            state_no_sim,
            work_dir,
            "rc.cir",
            "RC filter\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1n\n.tran 0 1m\n.end\n",
        )
        assert data["issue_count"] == 0, data["issues"]

    async def test_tstep_zero_flagged_only_for_ngspice_target(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # ".tran 0 1m" (auto-timestep) is clean for LTspice but fails on ngspice.
        # The target_simulator arg selects which portability rules apply.
        (work_dir / "tz.cir").write_text(
            "* tstep zero\nVin in 0 1\nR1 in 0 1k\n.tran 0 1m\n.end\n"
        )
        d_lt = (await handle_validate_netlist({"path": "tz.cir"}, state_no_sim)).structuredContent
        assert d_lt is not None
        assert d_lt["issue_count"] == 0, d_lt["issues"]
        d_ng = (
            await handle_validate_netlist(
                {"path": "tz.cir", "target_simulator": "ngspice"}, state_no_sim
            )
        ).structuredContent
        assert d_ng is not None
        assert any("TSTEP" in iss["message"] for iss in d_ng["issues"]), d_ng["issues"]

    async def test_ngspice_target_skips_ltspice_multiple_analysis_gate(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # "More than one analysis" is an LTspice rejection; it must not fire
        # (with LTspice wording) once the user targets ngspice.
        (work_dir / "multi.cir").write_text(
            "* multi\nVin in 0 AC 1\nR1 in 0 1k\n.ac dec 10 1 1Meg\n.tran 1u 1m\n.end\n"
        )
        d_lt = (
            await handle_validate_netlist({"path": "multi.cir"}, state_no_sim)
        ).structuredContent
        assert any("analysis" in iss["message"].lower() for iss in d_lt["issues"]), d_lt["issues"]
        d_ng = (
            await handle_validate_netlist(
                {"path": "multi.cir", "target_simulator": "ngspice"}, state_no_sim
            )
        ).structuredContent
        assert not any(
            "More than one" in iss["message"] or "Multiple distinct" in iss["message"]
            for iss in d_ng["issues"]
        ), d_ng["issues"]

    async def test_ngspice_target_skips_ltspice_meas_kind_mismatch(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # The silent-drop-on-mismatch reasoning is LTspice-specific; under
        # ngspice (which skips all .meas in batch mode at run time) it must not
        # emit the LTspice mismatch error.
        (work_dir / "mm.cir").write_text(
            "* meas mismatch\nVin in 0 1\nR1 in out 1k\nC1 out 0 1n\n"
            ".tran 1u 1m\n.meas ac gain FIND V(out) AT 1k\n.end\n"
        )
        d_lt = (await handle_validate_netlist({"path": "mm.cir"}, state_no_sim)).structuredContent
        assert any(".meas ac" in iss["message"] for iss in d_lt["issues"]), d_lt["issues"]
        d_ng = (
            await handle_validate_netlist(
                {"path": "mm.cir", "target_simulator": "ngspice"}, state_no_sim
            )
        ).structuredContent
        assert not any(".meas ac" in iss["message"] for iss in d_ng["issues"]), d_ng["issues"]

    async def test_ngspice_target_accepts_cl_keyed_primary_value(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # ``C1 a b C=10n`` / ``L1 b 0 L=1u`` are LTspice errors but valid ngspice.
        (work_dir / "cl.cir").write_text(
            "* cl keyed\nV1 a 0 1\nC1 a b C=10n\nL1 b 0 L=1u\n.tran 1u 1m\n.end\n"
        )
        d_lt = (await handle_validate_netlist({"path": "cl.cir"}, state_no_sim)).structuredContent
        assert any("does not accept" in iss["message"] for iss in d_lt["issues"]), d_lt["issues"]
        d_ng = (
            await handle_validate_netlist(
                {"path": "cl.cir", "target_simulator": "ngspice"}, state_no_sim
            )
        ).structuredContent
        assert not any("does not accept" in iss["message"] for iss in d_ng["issues"]), d_ng[
            "issues"
        ]

    async def test_asc_directive_elements_not_dangling_checked(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # An .asc SPICE-directive text block may legally carry element
        # lines; the schematic wires those nodes connect to are invisible
        # to the netlist dangling pass, so it must not run on .asc at all.
        data = await self._validate(
            state_no_sim,
            work_dir,
            "gate.asc",
            "Version 4\nSHEET 1 880 680\n"
            "TEXT 16 16 Left 2 !.tran 1m\n"
            "TEXT 16 80 Left 2 !R99 neta netb 1k\n",
        )
        assert data["issue_count"] == 0, data["issues"]

    async def test_flags_bad_meas(self, state_no_sim: SessionState, work_dir: Path):
        cir = work_dir / "bad.cir"
        cir.write_text(
            "* bad meas\nVin in 0 AC 1\nR1 in 0 1k\n.ac dec 100 1 1Meg\n"
            ".meas ac fc WHEN vdb(out)=-3\n.end\n"
        )
        result = await handle_validate_netlist({"path": cir.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert data["issue_count"] >= 1
        assert any("vdb" in iss["directive"] for iss in data["issues"])

    async def test_accepts_bsource_with_commas(self, state_no_sim: SessionState, work_dir: Path):
        cir = work_dir / "b.cir"
        cir.write_text(
            "* b-source\n"
            "V1 vp 0 1\n"
            "B1 amp 0 V = if(3.5*V(vp)>10, 10, if(3.5*V(vp)<-10, -10, 3.5*V(vp)))\n"
            "R1 amp 0 1k\n"
            ".tran 0 1m\n.end\n"
        )
        result = await handle_validate_netlist({"path": cir.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert data["issue_count"] == 0

    async def test_flags_meas_op_in_tran(self, state_no_sim: SessionState, work_dir: Path):
        """``.meas op`` in a transient run is silently dropped by
        LTspice. The validator should call this out so the user retypes."""
        cir = work_dir / "meas_op_mismatch.cir"
        cir.write_text(
            "* meas op under .tran\n"
            "V1 vdd 0 5\n"
            "R1 vdd a 1k\n"
            "C1 a 0 1n\n"
            ".tran 0 1m\n"
            ".meas op v_op_a FIND V(a)\n"
            ".end\n"
        )
        result = await handle_validate_netlist({"path": cir.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        meas_op_issues = [iss for iss in data["issues"] if ".meas op" in iss["message"]]
        assert meas_op_issues, "validator should flag .meas op without .op analysis"
        assert ".meas tran" in (meas_op_issues[0].get("suggestion") or "")

    async def test_meas_op_with_op_passes(self, state_no_sim: SessionState, work_dir: Path):
        """Inverse of the previous test: .meas op + .op is valid."""
        cir = work_dir / "meas_op_ok.cir"
        cir.write_text("V1 vdd 0 5\nR1 vdd a 1k\n.op\n.meas op v_op_a FIND V(a)\n.end\n")
        result = await handle_validate_netlist({"path": cir.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert not any(".meas op" in iss["message"] for iss in data["issues"])

    async def test_flags_meas_tran_in_ac(self, state_no_sim: SessionState, work_dir: Path):
        """the analysis-vs-meas check used to only catch
        .meas op. Other kinds (.meas tran under .ac, etc.) were silently
        dropped by LTspice. Now they're flagged symmetrically."""
        cir = work_dir / "meas_tran_in_ac.cir"
        cir.write_text(
            "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1n\n"
            ".ac dec 100 1 1Meg\n"
            ".meas tran v_max MAX V(out)\n.end\n"
        )
        result = await handle_validate_netlist({"path": cir.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert any(".meas tran" in iss["message"] for iss in data["issues"])

    async def test_flags_duplicate_analysis(self, state_no_sim: SessionState, work_dir: Path):
        """``.tran 1m`` + ``.tran 2m`` makes LTspice fail with
        "More than one analysis specified." Catch it in the static gate."""
        cir = work_dir / "dup.cir"
        cir.write_text("* dup\nV1 a 0 5\nR1 a 0 1k\n.tran 1m\n.tran 2m\n.end\n")
        result = await handle_validate_netlist({"path": cir.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert any(
            "Duplicate" in iss["message"] or "Multiple distinct" in iss["message"]
            for iss in data["issues"]
        )

    async def test_flags_multiple_distinct_analyses(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """Two different analyses (``.tran`` and ``.ac``) is the same kind
        of failure for LTspice — flag it too."""
        cir = work_dir / "two_kinds.cir"
        cir.write_text("V1 a 0 AC 1\nR1 a 0 1k\n.tran 1m\n.ac dec 10 1 1k\n.end\n")
        result = await handle_validate_netlist({"path": cir.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert any("Multiple distinct" in iss["message"] for iss in data["issues"])

    async def test_op_coexists_with_one_analysis(self, state_no_sim: SessionState, work_dir: Path):
        """``.op`` is a bias-point request, not a competing analysis — LTspice
        runs ``.op`` + one analysis fine (verified live), so the gate must NOT
        flag it. Two real analyses are still flagged (tests above)."""
        op_tran = work_dir / "op_tran.cir"
        op_tran.write_text(
            "* op+tran\nV1 a 0 PULSE(0 1 0 1u 1u 1m 2m)\nR1 a 0 1k\n.op\n.tran 1u 1m\n.end\n"
        )
        d1 = (
            await handle_validate_netlist({"path": op_tran.name}, state_no_sim)
        ).structuredContent
        assert d1 is not None
        assert not any(
            "Multiple distinct" in iss["message"] or "Duplicate analysis" in iss["message"]
            for iss in d1["issues"]
        )
        op_ac = work_dir / "op_ac.cir"
        op_ac.write_text(
            "* op+ac\nV1 a 0 AC 1\nR1 a 0 1k\nC1 a 0 1u\n.op\n.ac dec 10 1 1k\n.end\n"
        )
        d2 = (await handle_validate_netlist({"path": op_ac.name}, state_no_sim)).structuredContent
        assert d2 is not None
        assert not any(
            "Multiple distinct" in iss["message"] or "Duplicate analysis" in iss["message"]
            for iss in d2["issues"]
        )


@pytest.mark.asyncio
class TestDiffCircuit:
    async def test_value_change_surfaces(self, state_no_sim: SessionState, work_dir: Path):
        a = work_dir / "a.cir"
        b = work_dir / "b.cir"
        a.write_text("* a\nR1 in out 1k\nC1 out 0 100n\n.end\n")
        b.write_text("* b\nR1 in out 4.7k\nC1 out 0 100n\n.end\n")
        result = await handle_diff_circuit({"path_a": a.name, "path_b": b.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        changed = data["components_changed"]
        assert any(c["reference"].upper() == "R1" and c["after"] == "4.7k" for c in changed)

    async def test_added_and_removed(self, state_no_sim: SessionState, work_dir: Path):
        a = work_dir / "a.cir"
        b = work_dir / "b.cir"
        a.write_text("* a\nR1 in out 1k\n.end\n")
        b.write_text("* b\nR1 in out 1k\nC1 out 0 100n\n.end\n")
        result = await handle_diff_circuit({"path_a": a.name, "path_b": b.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert "C1" in [r.upper() for r in data["components_added"]]

    async def test_directive_diff(self, state_no_sim: SessionState, work_dir: Path):
        a = work_dir / "a.cir"
        b = work_dir / "b.cir"
        a.write_text("* a\nR1 in out 1k\n.tran 0 1m\n.end\n")
        b.write_text("* b\nR1 in out 1k\n.ac dec 100 1 1Meg\n.end\n")
        result = await handle_diff_circuit({"path_a": a.name, "path_b": b.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert any(".ac" in d for d in data["directives_added"])
        assert any(".tran" in d for d in data["directives_removed"])

    async def test_diff_circuit_micro_sign_not_reported_as_change(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A value LTspice renders with the micro sign (1µ, U+00B5) must compare
        # equal to the same value authored as '1u' — only the rendering differs,
        # not the magnitude. A real magnitude change (1u vs 2u) still surfaces.
        a = work_dir / "a.cir"
        b = work_dir / "b.cir"
        a.write_text("* a\nR1 in out 1k\nC1 out 0 1u\n.end\n")
        b.write_text("* b\nR1 in out 1k\nC1 out 0 1µ\n.end\n")
        result = await handle_diff_circuit({"path_a": a.name, "path_b": b.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert data["components_changed"] == [], data["components_changed"]

        # Guard: a genuine magnitude change is still reported.
        b.write_text("* b\nR1 in out 1k\nC1 out 0 2u\n.end\n")
        result2 = await handle_diff_circuit({"path_a": a.name, "path_b": b.name}, state_no_sim)
        data2 = result2.structuredContent
        assert data2 is not None
        assert any(c["reference"].upper() == "C1" for c in data2["components_changed"]), data2[
            "components_changed"
        ]

    async def test_deck_end_not_a_spurious_directive(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # ``.END`` is the deck terminator, not a meaningful directive: a .cir
        # carries one and an exported .asc netlist does not. It must not surface
        # as a removed directive (regression — it did when diffing .cir vs .asc).
        a = work_dir / "a.cir"
        b = work_dir / "b.cir"
        a.write_text("* a\nR1 in out 1k\n.op\n.end\n")
        b.write_text("* b\nR1 in out 1k\n.op\n")  # no .END terminator
        result = await handle_diff_circuit({"path_a": a.name, "path_b": b.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert not any(d.strip().lower() == ".end" for d in data["directives_removed"])
        assert data["directives_removed"] == []

    async def test_unparseable_file_surfaces_warning_not_wholesale_removal(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A corrupt .asc (AscEditor raises because it has no VERSION line) used
        # to be treated as an empty circuit, so the diff reported every real
        # component as removed with no signal that the file was unreadable. The
        # parse failure must be surfaced as a warning.
        good = work_dir / "good.asc"
        good.write_text(
            "Version 4\nSHEET 1 880 680\n"
            "SYMBOL res 100 100 R0\nSYMATTR InstName R1\nSYMATTR Value 1k\n"
        )
        broken = work_dir / "broken.asc"
        broken.write_text("this is not a schematic file at all\n")
        result = await handle_diff_circuit(
            {"path_a": good.name, "path_b": broken.name}, state_no_sim
        )
        data = result.structuredContent
        assert data is not None
        assert data["warnings"], "a parse failure must be reported, not silently ignored"
        assert any("broken.asc" in w for w in data["warnings"])
        # The interpretation caveat must ride in the structured warnings, not
        # just the text channel — structured-aware clients never see the text.
        assert any(
            "broken.asc" in w and "treats it as empty" in w and "trusting" in w
            for w in data["warnings"]
        )
        assert "WARNING" in result.content[0].text


class TestSourceWaveformValuesAccepted:
    """``set_component_value(V1, "PULSE(...)")`` was rejected as
    whitespace-bearing despite being a legal source spec."""

    def _run(self, body: str, ref: str, value: str) -> str:
        cards = lex(body).cards
        instance = next(c for c in cards if c.kind == "instance" and c.name == ref)
        apply_value_to_instance(instance, value)
        return emit(cards)

    def test_pulse_replaces_value_field(self) -> None:
        out = self._run(
            "V1 in 0 1\n",
            "V1",
            "PULSE(0 1 0 2n 2n 100n 200n)",
        )
        assert out.strip() == "V1 in 0 PULSE(0 1 0 2n 2n 100n 200n)"

    def test_sin_replaces_existing_pulse(self) -> None:
        out = self._run(
            "V1 in 0 PULSE(0 1 0 1n 1n 50n 100n)\n",
            "V1",
            "SIN(0 1 1k) AC 1",
        )
        assert out.strip() == "V1 in 0 SIN(0 1 1k) AC 1"

    def test_ac_magnitude_only(self) -> None:
        out = self._run("V1 in 0 1\n", "V1", "AC 1")
        assert out.strip() == "V1 in 0 AC 1"

    def test_pwl_with_internal_whitespace(self) -> None:
        out = self._run("I1 a 0 0\n", "I1", "PWL(0 0 1m 1 2m 0)")
        assert out.strip() == "I1 a 0 PWL(0 0 1m 1 2m 0)"


class TestBSourcePrefixPreserved:
    """A brace-only value used to drop ``V=``/``I=``."""

    def _run(self, body: str, ref: str, value: str) -> str:
        cards = lex(body).cards
        instance = next(c for c in cards if c.kind == "instance" and c.name == ref)
        apply_value_to_instance(instance, value)
        return emit(cards)

    def test_brace_keeps_v_prefix(self) -> None:
        out = self._run(
            "B1 fb 0 V={V(out)*0.5+1}\n",
            "B1",
            "{V(in)*0.5+1}",
        )
        assert out.strip() == "B1 fb 0 V={V(in)*0.5+1}"

    def test_explicit_kv_overrides_existing_type(self) -> None:
        # Switching from V= to I= drops the old V= rather than leaving
        # a stale slot behind.
        out = self._run(
            "B1 fb 0 V={V(out)*0.5+1}\n",
            "B1",
            "I=1m",
        )
        assert "V=" not in out
        assert "I=1m" in out

    def test_bare_value_with_no_existing_prefix_refuses(self) -> None:
        cards = lex("B1 fb 0 V=0\n").cards
        b1 = next(c for c in cards if c.kind == "instance" and c.name == "B1")
        # Strip V= manually so the body has no prefix to preserve.
        b1.replace_body("B1 fb 0")
        with pytest.raises(NetlistError, match="V=expr"):
            apply_value_to_instance(b1, "10")

    def test_operator_after_call_value_round_trips_without_orphan(self) -> None:
        # Editing ``V=V(in)*2`` used to rewrite only the ``V=V(in)`` span and
        # leave the ``*2`` behind, silently corrupting the card. The whole
        # expression must be replaced.
        out = self._run("B1 out 0 V=V(in)*2\n", "B1", "{V(in)*3}")
        assert out.strip() == "B1 out 0 V={V(in)*3}"

    def test_spaced_operator_expression_refused_not_corrupted(self) -> None:
        # The whitespace-around-operators form can't be re-joined
        # unambiguously, so editing it must refuse rather than leave orphans.
        cards = lex("B1 out 0 V = V(a) + V(b)\n").cards
        b1 = next(c for c in cards if c.kind == "instance" and c.name == "B1")
        with pytest.raises(NetlistError, match="not fully parseable"):
            apply_value_to_instance(b1, "{V(a)+V(b)}")


class TestControlledSourceGainReplacement:
    """``set_component_value(E1, "20")`` used to overwrite the
    controlling-node pair AND the gain. Should replace only the gain."""

    def _run(self, body: str, ref: str, value: str) -> str:
        cards = lex(body).cards
        instance = next(c for c in cards if c.kind == "instance" and c.name == ref)
        apply_value_to_instance(instance, value)
        return emit(cards)

    def test_e_source_gain_only(self) -> None:
        out = self._run("E1 buf 0 in 0 10\n", "E1", "20")
        assert out.strip() == "E1 buf 0 in 0 20"

    def test_g_source_gain_only(self) -> None:
        out = self._run("G1 out 0 in 0 5\n", "G1", "12")
        assert out.strip() == "G1 out 0 in 0 12"

    def test_f_source_gain_only(self) -> None:
        out = self._run("F1 out 0 V_sense 2\n", "F1", "5")
        assert out.strip() == "F1 out 0 V_sense 5"

    def test_f_source_with_control_ref_change(self) -> None:
        out = self._run("F1 out 0 V_sense 2\n", "F1", "V_new 5")
        assert out.strip() == "F1 out 0 V_new 5"


class TestModelNameElementsEditable:
    """A diode and the controlled switches carry a trailing model name, the
    same card shape as M/Q/J. ``set_component_value`` used to raise
    ``Unsupported element prefix`` for them; it must swap the model name."""

    def _run(self, body: str, ref: str, value: str) -> str:
        cards = lex(body).cards
        instance = next(c for c in cards if c.kind == "instance" and c.name == ref)
        apply_value_to_instance(instance, value)
        return emit(cards)

    def test_diode_model_swapped(self) -> None:
        out = self._run("D1 a k 1N4148\n", "D1", "1N5817")
        assert out.strip() == "D1 a k 1N5817"

    def test_voltage_switch_model_swapped(self) -> None:
        out = self._run("S1 n+ n- nc+ nc- SW1\n", "S1", "SW2")
        assert out.strip() == "S1 n+ n- nc+ nc- SW2"

    def test_current_switch_model_swapped(self) -> None:
        out = self._run("W1 n+ n- Vsense ISW1\n", "W1", "ISW2")
        assert out.strip() == "W1 n+ n- Vsense ISW2"

    def test_diode_with_area_factor_model_swapped(self) -> None:
        # The diode carries a trailing area factor. The swap must replace the
        # model and leave the area intact — it used to clobber the area "2" and
        # leave the real model 1N4148 in place.
        out = self._run("D1 a k 1N4148 2\n", "D1", "1N5817")
        assert out.strip() == "D1 a k 1N5817 2"

    def test_voltage_switch_with_state_model_swapped(self) -> None:
        # The trailing ON state must survive; only the model name changes.
        out = self._run("S1 n1 n2 nc1 nc2 MYSW ON\n", "S1", "NEWSW")
        assert out.strip() == "S1 n1 n2 nc1 nc2 NEWSW ON"

    def test_unsupported_prefix_offers_escape_hatch(self) -> None:
        # A still-unsupported prefix should point the user at editing the card
        # directly, not raise a bare "Unsupported element prefix".
        cards = lex("O1 a b c d TLINE\n").cards
        o1 = next(c for c in cards if c.kind == "instance" and c.name == "O1")
        with pytest.raises(NetlistError, match="Edit the card directly"):
            apply_value_to_instance(o1, "TLINE2")


@pytest.mark.asyncio
class TestSetComponentNodes:
    """set_component_value(nodes=...) rewires .cir connectivity — the fix for a
    mis-wired/typo'd net that the value path can't touch."""

    async def test_rewire_simple_node(self, state_no_sim: SessionState, work_dir: Path):
        cir = work_dir / "rewire.cir"
        cir.write_text("* rewire\nR1 in outt 1k\nC1 outt 0 1n\n.END\n")
        await handle_set_component_value(
            {"path": cir.name, "reference": "R1", "nodes": ["in", "out"]},
            state_no_sim,
        )
        text = cir.read_text()
        assert "R1 in out 1k" in text

    async def test_rewire_preserves_multitoken_source_spec(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # The corruption guard: a node rewrite must leave the value tail
        # byte-for-byte, never mangle a multi-token source spec like PULSE(...).
        cir = work_dir / "src.cir"
        cir.write_text("* src\nV1 a 0 PULSE(0 5 0 1n 1n 1m 2m)\nR1 a 0 1k\n.END\n")
        result = await handle_set_component_value(
            {"path": cir.name, "reference": "V1", "nodes": ["in", "0"]},
            state_no_sim,
        )
        text = cir.read_text()
        assert "V1 in 0 PULSE(0 5 0 1n 1n 1m 2m)" in text
        # The before/after message reports the real terminals only — the source
        # function token (PULSE) must not show up as a pseudo-node.
        msg = result.content[0].text
        assert "PULSE" not in msg
        assert "[a 0] -> [in 0]" in msg

    async def test_rewire_rejects_too_few_nodes(self, state_no_sim: SessionState, work_dir: Path):
        cir = work_dir / "few.cir"
        cir.write_text("* few\nR1 in 0 1k\n.END\n")
        with pytest.raises(NetlistError, match="2 node"):
            await handle_set_component_value(
                {"path": cir.name, "reference": "R1", "nodes": ["in"]},
                state_no_sim,
            )

    async def test_nodes_refused_on_variable_arity_device(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A BJT's terminal count isn't fixed (optional substrate node), so node
        # editing must refuse it rather than rewrite a wrong, hardcoded count.
        cir = work_dir / "bjt.cir"
        cir.write_text("* bjt\nQ1 c b e NPNMOD\nVCC c 0 5\n.op\n.END\n")
        with pytest.raises(NetlistError, match="edit the card directly"):
            await handle_set_component_value(
                {"path": cir.name, "reference": "Q1", "nodes": ["x", "y", "z"]},
                state_no_sim,
            )

    async def test_nodes_on_asc_points_at_connect(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        asc = work_dir / "sch.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\nSYMBOL res 100 100 R0\nSYMATTR InstName R1\n")
        with pytest.raises(NetlistError, match="connect"):
            await handle_set_component_value(
                {"path": asc.name, "reference": "R1", "nodes": ["in", "out"]},
                state_no_sim,
            )

    async def test_nodes_not_combined_with_value(self, state_no_sim: SessionState, work_dir: Path):
        cir = work_dir / "both.cir"
        cir.write_text("* both\nR1 in 0 1k\n.END\n")
        with pytest.raises(NetlistError, match="separate call"):
            await handle_set_component_value(
                {"path": cir.name, "reference": "R1", "value": "2k", "nodes": ["a", "b"]},
                state_no_sim,
            )


@pytest.mark.asyncio
class TestSetComponentValueNoOp:
    async def test_noop_value_is_flagged(self, state_no_sim: SessionState, work_dir: Path):
        # A no-op write (old == new) must not read like a real edit.
        cir = work_dir / "noop.cir"
        cir.write_text("* noop\nC1 in 0 159n\n.END\n")
        result = await handle_set_component_value(
            {"path": cir.name, "reference": "C1", "value": "159n"},
            state_no_sim,
        )
        assert "unchanged" in result.content[0].text


@pytest.mark.asyncio
class TestCreateNetlistSubpath:
    async def test_relative_subpath_creates_dirs(self, state_no_sim: SessionState, work_dir: Path):
        # The name field accepts a relative subpath; parent dirs are created.
        await handle_create_netlist(
            {"name": "sub/dir/rc", "content": "* x\nR1 in 0 1k\n.END\n"},
            state_no_sim,
        )
        assert (work_dir / "sub" / "dir" / "rc.cir").exists()
