"""End-to-end tests that start the real MCP server over stdio and exercise it
as a client would.

These tests launch the server as a subprocess via the MCP SDK's stdio_client,
then use ClientSession to send real MCP protocol messages.  Simulator
detection is disabled, so no simulator is needed — the consolidated tool
surface (schematic authoring, verification, inspection, job control, analysis)
is exercised in its degraded mode, which is where the error quality that an
agent depends on actually shows.

Symbol libraries come from the repo's ``.asy`` fixtures via
``LTSPICE_MCP_SYMBOL_PATHS`` so .asc authoring behaves the same on every host,
with or without LTspice installed.
"""

import json
import os
import sys
import textwrap
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import AnyUrl

from tests.conftest import FIXTURES_DIR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOL_TIMEOUT = timedelta(seconds=20)

SYMBOL_FIXTURES = FIXTURES_DIR / "symbols"

# The seven tools the consolidated profile puts on the wire.
CONSOLIDATED_TOOLS = {
    "run_experiments",
    "jobs",
    "analyze_results",
    "inspect",
    "edit_schematic",
    "verify_circuit",
    "plot_waveform",
}


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
        # .asc authoring needs symbols, not a simulator: point the server at
        # the fixture symbol set so schematic behaviour is host-independent.
        "LTSPICE_MCP_SYMBOL_PATHS": str(SYMBOL_FIXTURES),
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


def _data(result) -> dict:
    """Extract structuredContent, asserting the tool actually emitted one."""
    assert result.structuredContent is not None, f"no structuredContent: {_text(result)[:200]}"
    return result.structuredContent


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

# Two resistors in series, wired at the top, the lower R2 pin grounded — the
# smallest batch that exercises placement, routing and labelling in one call.
DIVIDER_OPS = [
    {"op": "add_component", "reference": "R1", "symbol": "res", "x": 400, "y": 300},
    {"op": "add_component", "reference": "R2", "symbol": "res", "x": 700, "y": 300},
    {
        "op": "wire_pins",
        "from_pin": "R1.1",
        "to_pin": "R2.1",
        "waypoints": [{"x": 400, "y": 200}, {"x": 700, "y": 200}],
    },
    {"op": "add_net_label", "net": "0", "pin": "R2.2"},
]


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
            await _call(session, "inspect", {"queries": [{"kind": "capabilities"}]})
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

    async def test_list_tools_is_exactly_the_consolidated_surface(self, tmp_path):
        """The wire answers the seven consolidated tools and nothing else."""
        async with mcp_session(tmp_path) as session:
            result = await session.list_tools()
            names = {t.name for t in result.tools}
            assert names == CONSOLIDATED_TOOLS

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
# 2. Schematic authoring — edit_schematic / verify_circuit round trip
# ---------------------------------------------------------------------------


class TestSchematicTools:
    async def test_blank_build_commits_and_reports_geometry(self, tmp_path):
        """A base:"blank" op batch writes the sheet and returns the geometry
        the model acts on: stages, sha, wiring metric, touched pins."""
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "edit_schematic",
                {"target": "divider.asc", "base": "blank", "ops": DIVIDER_OPS},
            )
            assert not result.isError, _text(result)
            data = _data(result)
            assert data["outcome"] == "complete"
            assert data["commit_state"] == "committed"
            assert (tmp_path / "divider.asc").exists()
            assert [stage["stage"] for stage in data["stages"]][-1] == "rename"
            assert all(stage["ok"] for stage in data["stages"])

            wiring = data["wiring"]
            assert wiring["pins_total"] == 4
            assert wiring["pins_wired"] == 2
            assert wiring["pins_label_only"] == 1
            assert wiring["label_only_pins"]["items"][0]["net"] == "0"

            touched = {item["ref"] for item in data["views"]["touched"]["items"]}
            assert touched == {"R1", "R2"}

    async def test_edit_requires_the_revision_it_was_written_against(self, tmp_path):
        """An existing sheet needs expected_sha256; a stale one is refused and
        nothing is written."""
        async with mcp_session(tmp_path) as session:
            first = _data(
                await _call(
                    session,
                    "edit_schematic",
                    {"target": "guarded.asc", "base": "blank", "ops": DIVIDER_OPS},
                )
            )
            committed = (tmp_path / "guarded.asc").read_bytes()

            stale = await _call(
                session,
                "edit_schematic",
                {
                    "target": "guarded.asc",
                    "ops": [
                        {
                            "op": "add_component",
                            "reference": "C1",
                            "symbol": "cap",
                            "x": 900,
                            "y": 300,
                        }
                    ],
                    "expected_sha256": "0" * 64,
                },
            )
            assert _data(stale)["error"]["code"] == "revision_conflict"
            assert (tmp_path / "guarded.asc").read_bytes() == committed

            fresh = await _call(
                session,
                "edit_schematic",
                {
                    "target": "guarded.asc",
                    "ops": [
                        {
                            "op": "add_component",
                            "reference": "C1",
                            "symbol": "cap",
                            "x": 900,
                            "y": 300,
                        }
                    ],
                    "expected_sha256": first["sha256"],
                },
            )
            assert _data(fresh)["commit_state"] == "committed"

    async def test_unresolvable_symbol_aborts_the_transaction_structurally(self, tmp_path):
        """A symbol the library cannot supply fails as structured op output —
        named op, named symbol, nothing written — not as a bare exception."""
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "edit_schematic",
                {
                    "target": "nosym.asc",
                    "base": "blank",
                    "ops": [
                        {
                            "op": "add_component",
                            "reference": "U1",
                            "symbol": "definitely_not_a_symbol",
                            "x": 400,
                            "y": 300,
                        }
                    ],
                },
            )
            data = _data(result)
            assert data["outcome"] == "failed"
            assert data["commit_state"] == "not_committed"
            assert data["error"]["code"] == "op_failed"
            assert data["failures"][0]["op"] == "add_component"
            assert "definitely_not_a_symbol" in data["failures"][0]["error"]
            assert not (tmp_path / "nosym.asc").exists()

    async def test_verify_circuit_reports_schematic_findings_and_scene(self, tmp_path):
        """verify_circuit over the wire names the layout facts of the sheet it
        was given, and says which checks it could not run."""
        async with mcp_session(tmp_path) as session:
            await _call(
                session,
                "edit_schematic",
                {"target": "checked.asc", "base": "blank", "ops": DIVIDER_OPS},
            )
            result = await _call(session, "verify_circuit", {"path": "checked.asc"})
            assert not result.isError, _text(result)
            data = _data(result)
            assert data["kind"] == "asc"
            assert "symbols" in data["checks_run"]
            # R1.2 is deliberately left dangling by DIVIDER_OPS.
            assert [f["rule_id"] for f in data["findings"]] == ["floating_pin"]
            assert data["findings"][0]["subject"] == "R1"
            assert data["scene"]["symbols"] == 2
            # No LTspice in this harness -> the exporter-backed check is skipped
            # with a reason rather than silently passing.
            skipped = {item["check"]: item["reason"] for item in data["checks_skipped"]}
            assert "export" in skipped

    async def test_verify_circuit_lints_a_netlist(self, tmp_path):
        (tmp_path / "rc.cir").write_text(RC_NETLIST)
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "verify_circuit", {"path": "rc.cir"})
            assert not result.isError, _text(result)
            data = _data(result)
            assert data["kind"] == "netlist"
            assert data["checks_run"] == ["syntax", "quality"]
            assert data["outcome"] == "complete"


# ---------------------------------------------------------------------------
# 3. Security — path escape
# ---------------------------------------------------------------------------


class TestSecurity:
    async def test_path_traversal_blocked(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "verify_circuit", {"path": "../../../etc/passwd"})
            _assert_tool_error(result, "not allowed")
            finding = _data(result)["findings"][0]
            assert finding["rule_id"] == "path_denied"

    async def test_absolute_path_outside_sandbox_blocked(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "verify_circuit", {"path": "/etc/passwd"})
            _assert_tool_error(result, "outside allowed directories")
            # The guidance must name the roots that ARE allowed, or the caller
            # cannot tell where to put the file instead.
            assert str(tmp_path) in _text(result)

    async def test_analyze_results_source_outside_sandbox_blocked(self, tmp_path):
        """A raw path is a path: the batched read plane enforces the sandbox and
        reports the refusal per source rather than reading the file."""
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "analyze_results",
                {
                    "sources": [{"raw_path": "/etc/passwd", "label": "outside"}],
                    "recipes": [{"key": "s", "metric": "summary"}],
                },
            )
            data = _data(result)
            assert data["coverage"]["runs_analyzed"] == 0
            missing = data["coverage"]["missing_cases"]["items"][0]
            assert missing["label"] == "outside"
            assert missing["code"] == "source_unavailable"
            assert "outside allowed directories" in missing["detail"]
            assert str(tmp_path) in missing["detail"]

    async def test_verify_nonexistent_file_errors(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "verify_circuit", {"path": "does_not_exist.cir"})
            _assert_tool_error(result, "does not exist")


# ---------------------------------------------------------------------------
# 4. Execute plane (degraded mode — no simulator)
# ---------------------------------------------------------------------------


class TestSimulationDegraded:
    async def test_run_experiments_reports_no_simulator_with_recovery(self, tmp_path):
        (tmp_path / "sim.cir").write_text("* Test\nV1 a 0 1\nR1 a 0 1k\n.op\n.end\n")
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "run_experiments",
                {
                    "request_id": "e2e-no-sim",
                    "circuits": [{"path": "sim.cir", "id": "dut"}],
                    "execution": {"wait_s": 1},
                },
            )
            _assert_tool_error(result, "No SPICE simulator detected")
            # Naming the knob is the whole value of the degraded-mode error: an
            # agent that only learns "it failed" installs nothing and retries.
            text = _text(result)
            assert "LTSPICE_MCP_SIMULATOR_EXE" in text
            assert "restart" in text.lower()
            data = _data(result)
            assert data["outcome"] == "failed"
            assert data["error"]["code"] == "submission_failed"
            assert data["error"]["commit_state"] == "not_started"
            assert data["completeness"]["submitted"] == 0

    async def test_jobs_list_empty(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "jobs", {"action": "list"})
            assert not result.isError
            data = _data(result)
            assert data["action"] == "list"
            assert data["outcome"] == "complete"
            assert data["items"] == []
            assert data["total"] == 0

    async def test_jobs_status_nonexistent_returns_job_not_found(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session, "jobs", {"action": "status", "job_id": "nonexistent-123"}
            )
            _assert_tool_error(result, "Job not found: nonexistent-123")
            data = _data(result)
            assert data["action"] == "status"
            assert data["error"]["code"] == "job_not_found"
            assert data["error"]["retryable"] is False
            assert "nonexistent-123" in data["hint"]

    async def test_jobs_cancel_nonexistent_returns_job_not_found(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session, "jobs", {"action": "cancel", "job_id": "nonexistent-456"}
            )
            _assert_tool_error(result, "Job not found: nonexistent-456")
            assert _data(result)["error"]["code"] == "job_not_found"


# ---------------------------------------------------------------------------
# 5. Analysis plane — verify specific error messages
# ---------------------------------------------------------------------------


class TestAnalysisDegraded:
    async def test_missing_raw_file_fails_that_source(self, tmp_path):
        """An unreadable source is a per-call failure item naming the file, not
        a silent empty result."""
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "analyze_results",
                {
                    "sources": [{"raw_path": "missing.raw", "label": "dut"}],
                    "recipes": [{"key": "s", "metric": "summary"}],
                },
            )
            assert not result.isError
            data = _data(result)
            assert data["outcome"] == "failed"
            assert data["coverage"]["runs_analyzed"] == 0
            failure = data["failures"][0]
            assert failure["code"] == "source_unavailable"
            assert "missing.raw" in failure["message"]

    async def test_missing_job_source_is_reported_as_missing_coverage(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "analyze_results",
                {
                    "sources": [{"job_id": "no-such-job", "label": "j"}],
                    "recipes": [{"key": "s", "metric": "summary"}],
                },
            )
            data = _data(result)
            missing = data["coverage"]["missing_cases"]["items"][0]
            assert missing["label"] == "j"
            # The id it could not resolve is the fact; the sentence is not.
            assert "no-such-job" in missing["detail"]
            assert "not found" in missing["detail"]


# ---------------------------------------------------------------------------
# 6. Inspect — capabilities report (the degraded-mode status surface)
# ---------------------------------------------------------------------------


class TestInspectCapabilities:
    async def test_capabilities_reports_degraded_simulator_state(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "inspect", {"queries": [{"kind": "capabilities"}]})
            assert not result.isError, _text(result)
            data = _data(result)
            assert data["ok_count"] == 1
            caps = data["results"][0]["data"]
            # Degraded mode is no longer an empty map: every known simulator
            # appears as unavailable WITH the remediation naming the exact
            # config key that would turn it on — the self-diagnosis surface
            # this state exists for.
            assert caps["simulators"], "degraded state must still list known simulators"
            for name, info in caps["simulators"].items():
                assert info["available"] is False, name
                assert info["remediation"]["config_key"] == "simulator.path"
                assert "restart" in info["remediation"]["action"]
            assert caps["config_path"].endswith("ltspice-mcp.toml")
            assert caps["python"]["executable"]
            assert caps["default_simulator"] is None
            assert caps["tool_profile"] == "consolidated"
            assert caps["allowed_paths"] == [str(tmp_path)]
            assert caps["limits"]["max_parallel_sims"] == 1
            assert caps["limits"]["default_timeout_s"] == 10.0


# ---------------------------------------------------------------------------
# 7. Resources — verify data content
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
# 8. Error handling — precise error classification
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_unknown_tool_returns_error(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "totally_fake_tool", {})
            assert result.isError
            assert "Unknown tool: totally_fake_tool" in _text(result)

    async def test_missing_required_arg_returns_validation_error(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "run_experiments", {"request_id": "e2e-no-circuits"})
            assert result.isError  # SDK-level schema validation
            text = _text(result)
            assert text.startswith("Input validation error:")
            assert "circuits" in text

    async def test_unknown_op_kind_rejected_by_schema(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(
                session,
                "edit_schematic",
                {"target": "bad.asc", "base": "blank", "ops": [{"op": "not_an_op"}]},
            )
            assert result.isError
            assert _text(result).startswith("Input validation error:")
            assert not (tmp_path / "bad.asc").exists()

    async def test_unknown_jobs_action_names_the_legal_set(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "jobs", {"action": "frobnicate"})
            assert result.isError
            text = _text(result)
            assert text.startswith("Input validation error:")
            for action in ("status", "wait", "cancel", "list", "runs"):
                assert action in text

    async def test_jobs_status_without_an_identifier_errors(self, tmp_path):
        async with mcp_session(tmp_path) as session:
            result = await _call(session, "jobs", {"action": "status"})
            _assert_tool_error(result, "requires exactly one of job_id or request_id")
