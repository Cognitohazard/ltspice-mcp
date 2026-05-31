"""Integration tests for circuit management tools."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ltspice_mcp.errors import NetlistError, PathSecurityError
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

    async def test_overwrite_replaces_existing(self, state_no_sim: SessionState, work_dir: Path):
        """``overwrite=True`` skips the FileExistsError path so iterating on
        a design doesn't force read+edit roundtrips. The earlier behaviour
        (always refuse) was friction during early stress-testing."""
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
        """Bug K: a behavioural source whose value has commas inside ``if(...)``
        defeats spicelib's component-line regex. ``list_components`` used to
        return ``Internal error in ltspice_list_components``; we now degrade
        the offending component to ``"<unparseable>"`` and finish the listing."""
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
        # All three components should appear; the B-source's value gets
        # the unparseable placeholder rather than aborting the whole call.
        assert "R1" in text
        assert "B1" in text
        assert "C1" in text
        assert "<unparseable>" in text


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


@pytest.mark.asyncio
class TestParameter:
    async def test_get_params(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_parameter({"path": sample_netlist.name}, state_no_sim)
        text = result.content[0].text
        assert "RVAL" in text or "Rval" in text

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
        """Bug J: a batch ``set_component_value`` with one missing ref used
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
        """Bug L: ``set_component_value(R1, "hello world")`` used to write a
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
    async def test_clean_netlist(self, state_no_sim: SessionState, work_dir: Path):
        cir = work_dir / "clean.cir"
        cir.write_text("* clean\nVin in 0 1\nR1 in 0 1k\n.tran 0 1m\n.end\n")
        result = await handle_validate_netlist({"path": cir.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert data["issue_count"] == 0

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

    async def test_flags_bsource_with_commas(self, state_no_sim: SessionState, work_dir: Path):
        cir = work_dir / "b.cir"
        cir.write_text(
            "* b-source\n"
            "B1 amp 0 V = if(3.5*V(vp)>10, 10, if(3.5*V(vp)<-10, -10, 3.5*V(vp)))\n"
            "R1 amp 0 1k\n"
            ".tran 0 1m\n.end\n"
        )
        result = await handle_validate_netlist({"path": cir.name}, state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert any("Behavioural" in iss["message"] for iss in data["issues"])

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
        flag it (v9-LT). Two real analyses are still flagged (tests above)."""
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
