"""Integration tests for .asc schematic editing tools using fixture symbols."""

from pathlib import Path

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.circuit import (
    AddComponentInput,
    CircuitReadInput,
    ComponentInfoInput,
    ConnectInput,
    EditDirectiveInput,
    MoveComponentInput,
    NetLabelInput,
    RemoveComponentInput,
    SetComponentAttributeInput,
    SymbolInfoInput,
    handle_add_component,
    handle_add_net_label,
    handle_component_info,
    handle_connect,
    handle_edit_directive,
    handle_list_components,
    handle_move_component,
    handle_read_circuit,
    handle_remove_component,
    handle_set_component_attribute,
    handle_symbol_info,
)


@pytest.mark.asyncio
class TestReadAscCircuit:
    async def test_reads_components(self, asc_state: SessionState, asc_file: Path):
        result = await handle_read_circuit(CircuitReadInput(path=asc_file.name), asc_state)
        text = result.content[0].text
        assert "C1" in text
        assert "R1" in text
        assert "V1" in text
        # Net labels
        assert "filtered" in text
        assert result.structuredContent["type"] == "asc"

    async def test_list_components_asc(self, asc_state: SessionState, asc_file: Path):
        result = await handle_list_components({"path": asc_file.name}, asc_state)
        text = result.content[0].text
        assert "C1" in text
        assert "R1" in text


@pytest.mark.asyncio
class TestGetSymbolInfo:
    async def test_valid_symbol(self, asc_state: SessionState):
        result = await handle_symbol_info(
            SymbolInfoInput(symbol="res", x=0, y=0, rotation="R0"), asc_state
        )
        text = result.content[0].text
        assert "res" in text
        assert "Pins" in text

    async def test_unknown_symbol(self, asc_state: SessionState):
        with pytest.raises(NetlistError, match="not found"):
            await handle_symbol_info(SymbolInfoInput(symbol="bogus_xyz_zzz"), asc_state)


@pytest.mark.asyncio
class TestGetComponentInfo:
    async def test_existing(self, asc_state: SessionState, asc_file: Path):
        result = await handle_component_info(
            ComponentInfoInput(path=asc_file.name, reference="R1"), asc_state
        )
        text = result.content[0].text
        assert "R1" in text
        assert "1k" in text
        assert "Pins:" in text

    async def test_missing_ref(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="not found"):
            await handle_component_info(
                ComponentInfoInput(path=asc_file.name, reference="ZZZ"), asc_state
            )

    async def test_requires_asc(self, asc_state: SessionState, work_dir: Path):
        cir = work_dir / "x.cir"
        cir.write_text("R1 1 0 1k\n.END\n")
        with pytest.raises(NetlistError, match=r"requires an \.asc"):
            await handle_component_info(
                ComponentInfoInput(path=cir.name, reference="R1"), asc_state
            )


@pytest.mark.asyncio
class TestRemoveComponent:
    async def test_remove(self, asc_state: SessionState, asc_file: Path):
        result = await handle_remove_component(
            RemoveComponentInput(path=asc_file.name, reference="C1"), asc_state
        )
        assert "Removed C1" in result.content[0].text
        # Confirm not present anymore
        read = await handle_read_circuit(CircuitReadInput(path=asc_file.name), asc_state)
        comp_refs = [c["reference"] for c in read.structuredContent["components"]]
        assert "C1" not in comp_refs

    async def test_remove_unknown(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="not found"):
            await handle_remove_component(
                RemoveComponentInput(path=asc_file.name, reference="ZZZ"), asc_state
            )


@pytest.mark.asyncio
class TestMoveComponent:
    async def test_move(self, asc_state: SessionState, asc_file: Path):
        result = await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=300, y=200),
            asc_state,
        )
        assert "Moved R1" in result.content[0].text

    async def test_move_with_rotation(self, asc_state: SessionState, asc_file: Path):
        result = await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=300, y=200, rotation="R180"),
            asc_state,
        )
        assert "R180" in result.content[0].text

    async def test_keep_rotation(self, asc_state: SessionState, asc_file: Path):
        # Omit rotation to exercise the "keep current" branch
        result = await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=400, y=400),
            asc_state,
        )
        assert "R1" in result.content[0].text


@pytest.mark.asyncio
class TestSetComponentAttribute:
    async def test_set(self, asc_state: SessionState, asc_file: Path):
        result = await handle_set_component_attribute(
            SetComponentAttributeInput(
                path=asc_file.name, reference="R1", attribute="SpiceLine", value="tol=1%"
            ),
            asc_state,
        )
        assert "SpiceLine" in result.content[0].text


@pytest.mark.asyncio
class TestAddComponent:
    async def test_add(self, asc_state: SessionState, asc_file: Path):
        result = await handle_add_component(
            AddComponentInput(
                path=asc_file.name,
                reference="R2",
                symbol="res",
                x=400,
                y=300,
                value="2k",
            ),
            asc_state,
        )
        text = result.content[0].text
        assert "Added R2" in text
        assert "2k" in text
        assert result.structuredContent["reference"] == "R2"
        assert "pins" in result.structuredContent

    async def test_add_duplicate(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="already exists"):
            await handle_add_component(
                AddComponentInput(path=asc_file.name, reference="R1", symbol="res", x=0, y=0),
                asc_state,
            )


@pytest.mark.asyncio
class TestAddNetLabel:
    async def test_add_at_xy(self, asc_state: SessionState, asc_file: Path):
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="VCC", x=100, y=100),
            asc_state,
        )
        assert "VCC" in result.content[0].text

    async def test_add_ground(self, asc_state: SessionState, asc_file: Path):
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="0", x=200, y=400),
            asc_state,
        )
        assert "ground" in result.content[0].text

    async def test_missing_xy_and_pin(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Either pin or both"):
            await handle_add_net_label(NetLabelInput(path=asc_file.name, net="X"), asc_state)

    async def test_remove_existing(self, asc_state: SessionState, asc_file: Path):
        # Draft1.asc has "filtered" label at (208, 128)
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="filtered", x=208, y=128, action="remove"),
            asc_state,
        )
        assert "Removed" in result.content[0].text

    async def test_remove_nonexistent(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="No"):
            await handle_add_net_label(
                NetLabelInput(path=asc_file.name, net="zz", x=999, y=999, action="remove"),
                asc_state,
            )

    async def test_duplicate_warning(self, asc_state: SessionState, asc_file: Path):
        # Add a second 'filtered' label at a different position
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="filtered", x=400, y=400),
            asc_state,
        )
        assert "Warning" in result.content[0].text


@pytest.mark.asyncio
class TestEditDirectiveCommentKind:
    async def test_add_comment_via_edit_directive(self, asc_state: SessionState, asc_file: Path):
        """Free-text annotations now go through ``edit_directive`` with
        ``kind='comment'`` instead of the old ``add_text`` tool."""
        result = await handle_edit_directive(
            EditDirectiveInput(
                path=asc_file.name,
                action="add",
                instruction="Test note",
                kind="comment",
                x=100,
                y=200,
            ),
            asc_state,
        )
        assert "Test note" in result.content[0].text

    async def test_comment_rejects_directive_prefix(
        self, asc_state: SessionState, asc_file: Path
    ):
        """Fr5: ``kind='comment'`` with an instruction that starts with
        ``!`` or ``.`` is almost always a mis-typed kind — refuse and
        steer the caller to ``kind='directive'``."""
        from ltspice_mcp.errors import NetlistError

        for instruction in ("!.tran 1m", ".ac dec 100 1 1Meg"):
            with pytest.raises(NetlistError, match="looks like a SPICE directive"):
                await handle_edit_directive(
                    EditDirectiveInput(
                        path=asc_file.name,
                        action="add",
                        instruction=instruction,
                        kind="comment",
                    ),
                    asc_state,
                )

    async def test_remove_spans_directive_and_comment(
        self, asc_state: SessionState, asc_file: Path
    ):
        """``remove`` should hit comments too — previously you could
        ``add_text`` a stray ``;.foo`` line and ``edit_directive remove``
        couldn't touch it."""
        await handle_edit_directive(
            EditDirectiveInput(
                path=asc_file.name,
                action="add",
                instruction="zap me",
                kind="comment",
            ),
            asc_state,
        )
        await handle_edit_directive(
            EditDirectiveInput(
                path=asc_file.name,
                action="remove",
                instruction="regex:zap me",
            ),
            asc_state,
        )
        # Comment should be gone — re-removing yields no error since the
        # underlying spicelib calls are tolerant of misses. ASC files may
        # contain Latin-1 µ characters, so read raw bytes and replace.
        text = asc_file.read_bytes().decode("utf-8", errors="replace")  # noqa: ASYNC240
        assert "zap me" not in text


@pytest.mark.asyncio
class TestConnect:
    async def test_diagonal_rejected(self, asc_state: SessionState, asc_file: Path):
        # First add a unique net label, then try a diagonal route to it
        await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="X", x=100, y=200), asc_state
        )
        with pytest.raises(NetlistError, match="not orthogonal"):
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="net:filtered",
                    to_pin="net:X",
                    waypoints=[],
                ),
                asc_state,
            )

    async def test_multiple_ground_labels_error(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Multiple '0'"):
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="net:filtered",
                    to_pin="net:0",
                ),
                asc_state,
            )

    async def test_invalid_pin_format(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Invalid pin reference"):
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="badformat",
                    to_pin="net:0",
                ),
                asc_state,
            )

    async def test_unknown_component(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="not found"):
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="ZZZ.A",
                    to_pin="net:0",
                ),
                asc_state,
            )

    async def test_missing_net_label(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Net label"):
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="net:nonexistent",
                    to_pin="net:0",
                ),
                asc_state,
            )

    async def test_pin_unknown(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="not found"):
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="R1.ZZ",
                    to_pin="net:0",
                ),
                asc_state,
            )
