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

    async def test_comment_rejects_directive_prefix(self, asc_state: SessionState, asc_file: Path):
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


@pytest.mark.asyncio
class TestEmptyAttributeRejected:
    """P-N1: add_component with empty SYMATTR value used to write a partial
    SYMATTR line and crash mid-write, leaving the .asc permanently
    unreadable. Reject up front."""

    async def test_empty_attribute_raises(self, asc_state: SessionState, asc_file: Path):
        original = asc_file.read_bytes()  # noqa: ASYNC240
        with pytest.raises(NetlistError, match="empty value"):
            await handle_add_component(
                AddComponentInput(
                    path=asc_file.name,
                    reference="M_bad",
                    symbol="res",
                    x=600,
                    y=600,
                    attributes={"SpiceModel": ""},
                ),
                asc_state,
            )
        assert asc_file.read_bytes() == original  # noqa: ASYNC240


@pytest.mark.asyncio
class TestEditingAscRollback:
    """Item 1: uncaught exceptions inside _editing_asc must invalidate the
    cached editor so a later read doesn't see dirty in-memory mutations,
    and the file on disk must remain intact."""

    async def test_uncaught_exception_after_mutation_invalidates_cache(
        self, asc_state: SessionState, asc_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Inject a failure after add_component has already mutated the
        # editor in-memory but before save. _post_op_warnings runs in the
        # handler body after _create_component, so raising from it
        # simulates a real spicelib internal error mid-edit.
        from ltspice_mcp.tools import circuit as circuit_mod

        original = asc_file.read_bytes()  # noqa: ASYNC240
        boom_calls = {"n": 0}

        def boom(*_a, **_kw):
            del _a, _kw
            boom_calls["n"] += 1
            raise RuntimeError("injected post-op failure")

        monkeypatch.setattr(circuit_mod, "_post_op_warnings", boom)

        with pytest.raises(RuntimeError, match="injected"):
            await handle_add_component(
                AddComponentInput(
                    path=asc_file.name,
                    reference="R_uncommitted",
                    symbol="res",
                    x=700,
                    y=700,
                ),
                asc_state,
            )

        # The injection fired (sanity).
        assert boom_calls["n"] == 1
        # File on disk is unchanged — save runs only on the success path.
        assert asc_file.read_bytes() == original  # noqa: ASYNC240
        # Cache eviction means a fresh read doesn't see R_uncommitted.
        monkeypatch.undo()
        result = await handle_list_components({"path": asc_file.name}, asc_state)
        assert "R_uncommitted" not in result.content[0].text


@pytest.mark.asyncio
class TestAtomicAscSave:
    """Item 2: a failure while spicelib is rendering the .asc must not
    leave a partially-written file on disk."""

    async def test_save_failure_preserves_original(
        self, asc_state: SessionState, asc_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from spicelib import AscEditor

        original = asc_file.read_bytes()  # noqa: ASYNC240

        # Inject a save that writes partial bytes to whatever sink it gets,
        # then raises. Two cases to defeat:
        #   1. Pre-fix path: editor.save_netlist(str(path)) opens the file
        #      directly. A partial write would land on disk. To prove the
        #      atomic-rename, route through the StringIO sink only (which
        #      atomic_write_text uses) — so a partial sink write does NOT
        #      reach the target.
        #   2. Post-fix path: editor.save_netlist(buf), then
        #      atomic_write_text(target, buf.getvalue(). On failure, the
        #      sibling temp is cleaned up and target stays intact.
        def failing_save(self_editor, sink):
            del self_editor
            # Write partial content to the sink (StringIO or file handle).
            if hasattr(sink, "write"):
                sink.write("Version 4\nSHEET 1 0 0\n!!CORRUPT!!\n")
            elif isinstance(sink, str):
                # Pre-fix code path: it would have passed a string path,
                # so spicelib opens the file directly. Simulate spicelib
                # writing partial content before crashing.
                Path(sink).write_text("Version 4\nSHEET 1 0 0\n!!CORRUPT!!\n")
            raise OSError("disk full simulation")

        monkeypatch.setattr(AscEditor, "save_netlist", failing_save)

        with pytest.raises(OSError, match="disk full"):
            await handle_add_component(
                AddComponentInput(
                    path=asc_file.name,
                    reference="R_aborted_save",
                    symbol="res",
                    x=600,
                    y=600,
                ),
                asc_state,
            )

        # Atomic-rename guarantee: no partial write reached the target.
        assert asc_file.read_bytes() == original  # noqa: ASYNC240

    async def test_save_failure_evicts_cache(
        self, asc_state: SessionState, asc_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Codex H1: a save that mutates the in-memory editor then crashes
        must still invalidate the cache. Otherwise a follow-up read sees
        the unsaved component."""
        from spicelib import AscEditor

        def failing_save(*args, **_kw):
            raise OSError("disk full simulation")

        monkeypatch.setattr(AscEditor, "save_netlist", failing_save)

        with pytest.raises(OSError, match="disk full"):
            await handle_add_component(
                AddComponentInput(
                    path=asc_file.name,
                    reference="R_uncommitted",
                    symbol="res",
                    x=600,
                    y=600,
                ),
                asc_state,
            )

        # Restore real save so the follow-up read works.
        monkeypatch.undo()

        # The component must NOT be visible — cache was evicted, fresh
        # read from disk shows the pre-failure state.
        result = await handle_list_components({"path": asc_file.name}, asc_state)
        assert "R_uncommitted" not in result.content[0].text
