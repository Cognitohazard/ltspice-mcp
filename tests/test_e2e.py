"""End-to-end tests that start the real MCP server over stdio and exercise it
as a client would.

These tests launch the server as a subprocess via the MCP SDK's stdio_client,
then use ClientSession to send real MCP protocol messages.  Simulator
detection is disabled, so no simulator is needed — circuit editing, status,
resources, and error-path tests all exercise the degraded-mode behaviour.
"""

import json
import os
import re
import sys
import textwrap
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import AnyUrl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOL_TIMEOUT = timedelta(seconds=10)


def _server_params(work_dir: Path) -> StdioServerParameters:
    """Build StdioServerParameters that launch ltspice-mcp in *work_dir*
    with no real simulator."""
    config = work_dir / "ltspice-mcp.toml"
    config.write_text(
        textwrap.dedent("""\
        [simulator]
        default = "ltspice"
        path = ""

        [security]
        allowed_paths = ["."]

        [simulation]
        max_parallel = 1
        timeout = 10.0

        [logging]
        level = "DEBUG"
    """)
    )
    env = {
        **os.environ,
        "LTSPICE_MCP_CONFIG": str(config),
        "LTSPICE_MCP_WORKING_DIR": str(work_dir),
        "LTSPICE_MCP_ALLOWED_PATHS": str(work_dir),
        # Isolate from any global ``recent.json`` index left over from
        # the developer's local sessions; otherwise tests asserting an
        # empty results list pick up unrelated jobs.
        "LTSPICE_MCP_HOME": str(work_dir),
        "XDG_STATE_HOME": str(work_dir),
        # Force "no simulator" code paths regardless of host: this test
        # suite covers the degraded-mode behaviour, and a CI host with
        # ngspice on ``PATH`` would otherwise satisfy auto-detection.
        "LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION": "1",
    }
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "ltspice_mcp"],
        env=env,
        cwd=str(work_dir),
    )


@asynccontextmanager
async def mcp_session(work_dir: Path) -> AsyncIterator[ClientSession]:
    """Open a live MCP client session connected to the server."""
    params = _server_params(work_dir)
    async with (
        stdio_client(params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        init = await session.initialize()
        assert init.serverInfo.name == "ltspice-mcp"
        yield session


def _text(result) -> str:
    """Extract text from the first TextContent in a CallToolResult."""
    return result.content[0].text


def _call(session, name, args=None):
    """Shorthand for call_tool with standard timeout."""
    return session.call_tool(name, args or {}, read_timeout_seconds=TOOL_TIMEOUT)


def _assert_tool_error(result, expected_substring: str):
    """Assert the tool returned an error with isError=True containing expected_substring.

    All tool errors (LTSpiceMCPError, ValueError) propagate to the MCP SDK,
    which wraps them in CallToolResult(isError=True).
    """
    assert result.isError, f"Expected isError=True but got success: {_text(result)[:200]}"
    text = _text(result)
    assert expected_substring.lower() in text.lower(), (
        f"Expected '{expected_substring}' in error text: {text[:200]}"
    )


# Standard netlist content — valid for spicelib (must have * title line)
RC_NETLIST = (
    "* RC Low-Pass Filter\nR1 in out 1k\nC1 out 0 100n\nV1 in 0 AC 1\n.ac dec 100 1 1Meg\n"
)

MINIMAL_NETLIST = "* Test\nR1 a b 1k\nR2 b 0 2.2k\nC1 b 0 10n\n.END\n"


# ---------------------------------------------------------------------------
# 1. Server lifecycle & discovery
# ---------------------------------------------------------------------------


class TestServerLifecycle:
    async def test_initialize_reports_capabilities(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            caps = session.get_server_capabilities()
            assert caps is not None
            assert caps.tools is not None
            assert caps.resources is not None
            assert caps.prompts is not None  # workflow-starter prompts

    async def test_initialize_reports_package_version_and_active_simulator(self, tmp_path):
        # A real handshake must carry the ltspice-mcp package version (not the
        # mcp SDK version), and instructions naming the detected simulators.
        from importlib.metadata import version as pkg_version

        params = _server_params(tmp_path)
        async with (
            stdio_client(params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            init = await session.initialize()
            assert init.serverInfo.version == pkg_version("ltspice-mcp")
            # Detection is disabled in this harness -> the no-simulator line.
            assert init.instructions is not None
            assert "No SPICE simulator detected" in init.instructions

    async def test_server_name_override_via_env(self, tmp_path):
        # The alias packages (circuit-mcp/ngspice-mcp) set LTSPICE_MCP_SERVER_NAME
        # so the handshake identifies as the alias, not the canonical name.
        params = _server_params(tmp_path)
        assert params.env is not None
        params.env["LTSPICE_MCP_SERVER_NAME"] = "circuit-mcp"
        async with (
            stdio_client(params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            init = await session.initialize()
            assert init.serverInfo.name == "circuit-mcp"

    async def test_config_written_lazily_on_first_tool_call(self, tmp_path):
        # The server boots in whatever directory the MCP client launched it
        # from, so it must NOT drop a config file there just for starting up
        # (that litters every unrelated project folder of plugin users). The
        # default config appears only once a tool is actually used.
        env = {
            **os.environ,
            "LTSPICE_MCP_WORKING_DIR": str(tmp_path),
            "LTSPICE_MCP_ALLOWED_PATHS": str(tmp_path),
            "LTSPICE_MCP_HOME": str(tmp_path),
            "XDG_STATE_HOME": str(tmp_path),
            "LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION": "1",
        }
        env.pop("LTSPICE_MCP_CONFIG", None)  # let it resolve to cwd/ltspice-mcp.toml
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ltspice_mcp"],
            env=env,
            cwd=str(tmp_path),
        )
        config_file = tmp_path / "ltspice-mcp.toml"
        async with (
            stdio_client(params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            assert not config_file.exists(), "startup must not write a config file"
            await _call(session, "server_status")
            assert config_file.exists(), "first tool call should write the default config"

    async def test_prompts_list_and_get(self, tmp_path):
        from mcp import types as mcp_types

        async with mcp_session(tmp_path) as session:
            listed = await session.list_prompts()
            names = {p.name for p in listed.prompts}
            assert {"characterize_filter", "run_and_plot", "step_response"} <= names
            got = await session.get_prompt("characterize_filter", {"path": "rc.cir"})
            assert got.messages
            content = got.messages[0].content
            assert isinstance(content, mcp_types.TextContent)
            assert "rc.cir" in content.text

    async def test_list_tools_contains_all_modules(self, tmp_path):
        """Every module's tools appear in the dispatch table."""
        async with mcp_session(tmp_path) as session:
            result = await session.list_tools()
            names = {t.name for t in result.tools}
            expected = {
                "create_netlist",
                "read_circuit",
                "list_components",
                "set_component_value",
                "parameter",
                "edit_directive",
                "run_simulation",
                "check_job",
                "cancel_job",
                "signal_stats",
                "query_value",
                "operating_point",
                "simulation_summary",
                "configure_sweep",
                "run_sweep",
                "configure_montecarlo",
                "run_montecarlo",
                "batch_results",
                "find_model",
                "load_library",
                "unload_library",
                "list_libraries",
                "server_status",
            }
            missing = expected - names
            assert not missing, f"Missing tools: {missing}"

    async def test_plot_waveform_declares_ui_resource_over_protocol(self, tmp_path):
        # The MCP Apps UI link must survive the wire as _meta on the tool
        # declaration (serialized by alias), or an apps host never fetches the
        # renderer. Verify the round-tripped tool carries it.
        async with mcp_session(tmp_path) as session:
            result = await session.list_tools()
            plot = next(t for t in result.tools if t.name == "plot_waveform")
            assert plot.meta == {"ui": {"resourceUri": "ui://ltspice-mcp/plot"}}

    async def test_ping(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await session.send_ping()
            assert result is not None


# ---------------------------------------------------------------------------
# 2. Circuit tools — round-trip verification
# ---------------------------------------------------------------------------


class TestCircuitTools:
    async def test_create_netlist_auto_appends_end(self, tmp_path):
        """create_netlist appends .END if missing from content."""
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "create_netlist",
                {"name": "noend", "content": "* No end\nR1 a b 1k\n"},
            )
            assert not result.isError
            file_content = (tmp_path / "noend.cir").read_text()
            assert file_content.strip().upper().endswith(".END")

    async def test_create_and_read_roundtrip(self, tmp_path):
        """Create a netlist, then read it back — verify components appear."""
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session, "create_netlist", {"name": "rc_filter", "content": RC_NETLIST}
            )
            assert not result.isError
            text = _text(result)
            assert "Components: 3" in text  # R1, C1, V1

            result = await _call(session, "read_circuit", {"path": "rc_filter.cir"})
            assert not result.isError
            text = _text(result)
            assert "R1 in out 1k" in text
            assert "C1 out 0 100n" in text

    async def test_list_components_returns_all_with_values(self, tmp_path):
        (tmp_path / "comps.cir").write_text(MINIMAL_NETLIST)
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "list_components", {"path": "comps.cir"})
            assert not result.isError
            text = _text(result)
            assert "R1" in text and "1k" in text
            assert "R2" in text and "2.2k" in text
            assert "C1" in text and "10n" in text

    async def test_list_components_prefix_excludes_others(self, tmp_path):
        (tmp_path / "prefix.cir").write_text(MINIMAL_NETLIST)
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "list_components", {"path": "prefix.cir", "prefix": "R"})
            assert not result.isError
            text = _text(result)
            assert "R1" in text
            assert "R2" in text
            # C1 must NOT appear
            for line in text.strip().split("\n"):
                assert not line.startswith("C"), f"Unexpected component in filtered output: {line}"

    async def test_single_reference_returns_value(self, tmp_path):
        (tmp_path / "ref.cir").write_text("* Test\nR1 a b 4.7k\n.END\n")
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session, "list_components", {"path": "ref.cir", "reference": "R1"}
            )
            assert not result.isError
            text = _text(result)
            assert text.startswith("R1")
            assert "4.7k" in text

    async def test_set_component_value_roundtrip(self, tmp_path):
        """Set a value, then read it back via list_components."""
        (tmp_path / "setval.cir").write_text("* Test\nR1 a b 1k\n.END\n")
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "set_component_value",
                {"path": "setval.cir", "reference": "R1", "value": "2.2k"},
            )
            assert not result.isError
            assert "1k" in _text(result) and "2.2k" in _text(result)

            # Round-trip
            result = await _call(
                session, "list_components", {"path": "setval.cir", "reference": "R1"}
            )
            assert "2.2k" in _text(result)

    async def test_batch_set_component_values(self, tmp_path):
        (tmp_path / "batch.cir").write_text(MINIMAL_NETLIST)
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "set_component_value",
                {"path": "batch.cir", "values": {"R1": "10k", "R2": "47k"}},
            )
            assert not result.isError
            assert "Updated 2 component(s)" in _text(result)

            r1 = await _call(session, "list_components", {"path": "batch.cir", "reference": "R1"})
            assert "10k" in _text(r1)
            r2 = await _call(session, "list_components", {"path": "batch.cir", "reference": "R2"})
            assert "47k" in _text(r2)

    async def test_parameter_get_then_set_then_verify(self, tmp_path):
        (tmp_path / "params.cir").write_text("* Test\nR1 a b {Rval}\n.param Rval=1k\n.END\n")
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "parameter", {"path": "params.cir"})
            assert not result.isError
            assert ".PARAM" in _text(result).upper()

            result = await _call(
                session,
                "parameter",
                {"path": "params.cir", "name": "Rval", "value": "4.7k"},
            )
            assert not result.isError
            assert "4.7k" in _text(result)

            # Verify
            result = await _call(session, "parameter", {"path": "params.cir"})
            assert "4.7k" in _text(result).lower() or "4700" in _text(result)

    async def test_edit_directive_roundtrip(self, tmp_path):
        (tmp_path / "dir.cir").write_text("* Test\nR1 a b 1k\n.END\n")
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "edit_directive",
                {"path": "dir.cir", "action": "add", "instruction": ".tran 1m"},
            )
            assert not result.isError
            assert ".tran 1m" in (tmp_path / "dir.cir").read_text()

            result = await _call(
                session,
                "edit_directive",
                {"path": "dir.cir", "action": "remove", "instruction": ".tran 1m"},
            )
            assert not result.isError
            assert ".tran 1m" not in (tmp_path / "dir.cir").read_text()

    async def test_create_netlist_rejects_duplicate(self, tmp_path):
        (tmp_path / "dup.cir").write_text("* Existing\nR1 a b 1k\n.END\n")
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session, "create_netlist", {"name": "dup", "content": "* New\nR1 a b 2k"}
            )
            _assert_tool_error(result, "already exists")

    async def test_nonexistent_reference_errors(self, tmp_path):
        (tmp_path / "noref.cir").write_text("* Test\nR1 a b 1k\n.END\n")
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session, "list_components", {"path": "noref.cir", "reference": "C99"}
            )
            _assert_tool_error(result, "not found")


# ---------------------------------------------------------------------------
# 3. Security — path escape
# ---------------------------------------------------------------------------


class TestSecurity:
    async def test_path_traversal_blocked(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "read_circuit", {"path": "../../../etc/passwd"})
            _assert_tool_error(result, "not allowed")

    async def test_absolute_path_outside_sandbox_blocked(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "create_netlist",
                {"name": "/tmp/evil", "content": "* Bad\nR1 a b 1k"},
            )
            _assert_tool_error(result, "outside")

    async def test_read_nonexistent_file_errors(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "read_circuit", {"path": "does_not_exist.cir"})
            _assert_tool_error(result, "not found")


# ---------------------------------------------------------------------------
# 4. Simulation tools (degraded mode — no simulator)
# ---------------------------------------------------------------------------


class TestSimulationDegraded:
    async def test_run_simulation_reports_no_simulator(self, tmp_path):
        (tmp_path / "sim.cir").write_text("* Test\nR1 a b 1k\n.tran 1m\n.END\n")
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "run_simulation", {"netlist": "sim.cir"})
            _assert_tool_error(result, "simulator")

    async def test_check_job_nonexistent_returns_error(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "check_job", {"job_id": "nonexistent-123"})
            _assert_tool_error(result, "Job not found: nonexistent-123")

    async def test_cancel_job_nonexistent_returns_error(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "cancel_job", {"job_id": "nonexistent-456"})
            _assert_tool_error(result, "Job not found: nonexistent-456")

    async def test_check_job_list_empty(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "check_job", {})
            assert not result.isError
            assert "No active jobs" in _text(result)


# ---------------------------------------------------------------------------
# 5. Analysis tools — verify specific error messages
# ---------------------------------------------------------------------------


class TestAnalysisDegraded:
    async def test_signal_stats_missing_file(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "signal_stats",
                {"raw_file": "missing.raw", "signal": "V(out)"},
            )
            _assert_tool_error(result, "not found")

    async def test_get_simulation_summary_missing_file(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "simulation_summary", {"raw_file": "missing.raw"})
            _assert_tool_error(result, "not found")


# ---------------------------------------------------------------------------
# 6. Advanced tools — verify config IDs and two-phase workflow
# ---------------------------------------------------------------------------


class TestAdvancedTools:
    async def test_configure_sweep_returns_config_id(self, tmp_path):
        (tmp_path / "sweep.cir").write_text(
            "* Test\nR1 a b {Rval}\n.param Rval=1k\n.tran 1m\n.END\n"
        )
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "configure_sweep",
                {
                    "netlist": "sweep.cir",
                    "parameters": [
                        {
                            "name": "Rval",
                            "type": "parameter",
                            "start": 100,
                            "stop": 10000,
                            "points": 5,
                        }
                    ],
                },
            )
            assert not result.isError
            text = _text(result)
            assert "Sweep configured" in text
            # Config ID format: sweep_<timestamp>_<hash>
            assert re.search(r"Config ID: sweep_\d+_[0-9a-f]+", text)
            assert "Total simulations: 5" in text
            assert "run_sweep(" in text

    async def test_configure_montecarlo_returns_config_id(self, tmp_path):
        (tmp_path / "mc.cir").write_text("* Test\nR1 a b 1k\nC1 b 0 100n\n.tran 1m\n.END\n")
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "configure_montecarlo",
                {
                    "netlist": "mc.cir",
                    "tolerances": [{"ref": "R1", "tolerance": 0.05}],
                    "num_runs": 10,
                },
            )
            assert not result.isError
            text = _text(result)
            assert "Monte Carlo configured" in text
            assert re.search(r"Config ID: mc_\d+_[0-9a-f]+", text)
            assert "Runs: 10" in text
            assert "5.0%" in text  # tolerance formatted as percentage
            assert "run_montecarlo(" in text

    async def test_run_sweep_without_simulator_errors(self, tmp_path):
        """configure_sweep succeeds, but run_sweep needs a simulator."""
        (tmp_path / "sw2.cir").write_text("* Test\nR1 a b {X}\n.param X=1k\n.tran 1m\n.END\n")
        async with mcp_session(tmp_path) as session:
            cfg = await _call(
                session,
                "configure_sweep",
                {
                    "netlist": "sw2.cir",
                    "parameters": [
                        {"name": "X", "type": "parameter", "start": 1, "stop": 10, "points": 3}
                    ],
                },
            )
            match = re.search(r"Config ID: (sweep_\S+)", _text(cfg))
            assert match, f"No config ID in: {_text(cfg)}"
            config_id = match.group(1)

            result = await _call(session, "run_sweep", {"config_id": config_id})
            _assert_tool_error(result, "simulator")

    async def test_run_sweep_invalid_config_id_errors(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "run_sweep", {"config_id": "sweep_bogus"})
            _assert_tool_error(result, "not found")

    async def test_get_batch_results_invalid_job_returns_error(self, tmp_path):
        """get_batch_results returns error for missing jobs."""
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "batch_results", {"job_id": "batch_bogus"})
            _assert_tool_error(result, "Batch job not found: batch_bogus")


# ---------------------------------------------------------------------------
# 7. Library tools
# ---------------------------------------------------------------------------


class TestLibraryTools:
    async def test_list_libraries_empty_says_no_libraries(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "list_libraries", {})
            assert not result.isError
            assert "No libraries loaded" in _text(result)

    async def test_find_model_no_results(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "find_model", {"name": "LM358"})
            assert not result.isError
            assert "No fuzzy matches" in _text(result)

    async def test_load_nonexistent_library_errors(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "load_library", {"path": "nonexistent.lib"})
            _assert_tool_error(result, "does not exist")

    async def test_unload_not_loaded_library_errors(self, tmp_path):
        lib_file = tmp_path / "empty.lib"
        lib_file.write_text("")
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "unload_library", {"path": "empty.lib"})
            _assert_tool_error(result, "not loaded")


# ---------------------------------------------------------------------------
# 8. Status tool — verify structured output
# ---------------------------------------------------------------------------


class TestStatusTool:
    async def test_get_server_status_content(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "server_status", {})
            assert not result.isError
            text = _text(result)
            assert "=== LTSpice MCP Server Status ===" in text
            assert "Simulators:" in text
            assert "degraded mode" in text
            assert "Default simulator: None" in text
            assert "Configuration:" in text
            assert f"Working directory: {tmp_path}" in text
            assert "Security (Sandbox):" in text
            assert "Runtime State:" in text
            assert "Active jobs: 0" in text


# ---------------------------------------------------------------------------
# 9. Resources — verify data content
# ---------------------------------------------------------------------------


class TestResources:
    async def test_list_resources_returns_static_set(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await session.list_resources()
            resources = {r.name: r for r in result.resources}
            assert len(resources) == 7
            assert set(resources.keys()) == {
                "netlists",
                "results",
                "models",
                "config",
                "recent",
                "plot_widget",
                "guide",
            }
            assert str(resources["config"].uri) == "spice://config"
            assert str(resources["recent"].uri) == "spice://recent"
            assert str(resources["guide"].uri) == "spice://guide"

    async def test_list_resource_templates_returns_three(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await session.list_resource_templates()
            templates = {t.name for t in result.resourceTemplates}
            assert templates == {"netlist_content", "job_signals", "job_measurements"}

    async def test_read_ui_widget_resource_over_protocol(self, tmp_path):
        # The MCP Apps renderer is served under the ui:// scheme — exercise the
        # full SDK read path (AnyUrl parsing of a non-ltspice scheme) end to end.
        async with mcp_session(tmp_path) as session:
            result = await session.read_resource(AnyUrl("ui://ltspice-mcp/plot"))
            entry = result.contents[0]
            assert entry.mimeType == "text/html;profile=mcp-app"
            assert "globalThis.ExtApps" in entry.text  # type: ignore[union-attr]

    async def test_read_config_resource_has_correct_fields(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await session.read_resource(AnyUrl("spice://config"))
            data = json.loads(result.contents[0].text)  # type: ignore[union-attr]
            assert data["working_dir"] == str(tmp_path)
            assert isinstance(data["allowed_paths"], list)
            assert data["detected_simulators"] == []
            assert data["default_simulator"] is None
            assert data["max_parallel_sims"] == 1
            assert data["default_timeout"] == 10.0
            assert data["log_level"] == "DEBUG"

    async def test_read_netlists_lists_cir_files_only(self, tmp_path):
        (tmp_path / "circuit.cir").write_text("* Test\nR1 a b 1k\n.END\n")
        (tmp_path / "notes.txt").write_text("not a netlist")
        async with mcp_session(tmp_path) as session:
            result = await session.read_resource(AnyUrl("spice://netlists/"))
            data = json.loads(result.contents[0].text)  # type: ignore[union-attr]
            names = [n["name"] for n in data["netlists"]]
            assert "circuit.cir" in names
            assert "notes.txt" not in names
            assert "ltspice-mcp.toml" not in names
            assert data["count"] == len(data["netlists"])

    async def test_read_netlist_content_via_resource_template(self, tmp_path):
        netlist_text = "* My Circuit\nR1 a b 1k\nC1 b 0 10n\n.END\n"
        (tmp_path / "mycirc.cir").write_text(netlist_text)
        async with mcp_session(tmp_path) as session:
            result = await session.read_resource(AnyUrl("spice://netlists/mycirc.cir"))
            content = result.contents[0].text  # type: ignore[union-attr]
            assert content == netlist_text

    async def test_read_results_empty(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await session.read_resource(AnyUrl("spice://results/"))
            data = json.loads(result.contents[0].text)  # type: ignore[union-attr]
            assert data == {"count": 0, "jobs": []}

    async def test_read_models_empty(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await session.read_resource(AnyUrl("spice://models/"))
            data = json.loads(result.contents[0].text)  # type: ignore[union-attr]
            assert data["libraries"] == []


# ---------------------------------------------------------------------------
# 11. Error handling — precise error classification
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_unknown_tool_returns_error(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "totally_fake_tool", {})
            assert result.isError
            assert "Unknown tool: totally_fake_tool" in _text(result)

    async def test_missing_required_arg_returns_validation_error(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "read_circuit", {})
            assert result.isError  # SDK-level validation
            assert "path" in _text(result).lower()

    async def test_edit_directive_rejects_non_dot_instruction(self, tmp_path):
        (tmp_path / "nondot.cir").write_text("* Test\nR1 a b 1k\n.END\n")
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "edit_directive",
                {"path": "nondot.cir", "action": "add", "instruction": "not-a-directive"},
            )
            _assert_tool_error(result, "must start with '.'")

    async def test_set_component_missing_both_modes_errors(self, tmp_path):
        (tmp_path / "badset.cir").write_text("* Test\nR1 a b 1k\n.END\n")
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "set_component_value", {"path": "badset.cir"})
            _assert_tool_error(result, "reference")
