"""Integration tests for circuit management tools."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ltspice_mcp.errors import NetlistError, PathSecurityError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.circuit import (
    handle_create_netlist,
    handle_edit_directive,
    handle_list_components,
    handle_parameter,
    handle_read_circuit,
    handle_set_component_value,
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

    async def test_overwrite_replaces_existing(
        self, state_no_sim: SessionState, work_dir: Path
    ):
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

    async def test_remove_directive(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_edit_directive(
            {"path": sample_netlist.name, "action": "remove", "instruction": ".ac dec 100 1 1Meg"},
            state_no_sim,
        )
        assert "Removed" in result.content[0].text
