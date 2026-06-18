"""Real-world scenario: Design and simulate a Sallen-Key low-pass filter.

Run with: uv run python tests/scenario_active_filter.py

This exercises the full tool chain as a real MCP client would:
1. Create a netlist from scratch
2. Read and verify the circuit
3. Set component values to target a specific cutoff frequency
4. Run a simulation (requires LTspice installed)
5. Analyze results: frequency response, measurements
6. Sweep a component to explore design space
7. Monte Carlo: check sensitivity to component tolerances

Steps 1-3 work without a simulator. Steps 4-7 require LTspice.
The script reports which steps succeed and which need a simulator.
"""

import asyncio
import json
import os
import re
import sys
import textwrap
from datetime import timedelta
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

TIMEOUT = timedelta(seconds=30)
SIM_TIMEOUT = timedelta(seconds=120)

# Where to run — use the project workspace dir
WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)


def _params() -> StdioServerParameters:
    env = {**os.environ}
    env["LTSPICE_MCP_WORKING_DIR"] = str(WORKSPACE)
    env["LTSPICE_MCP_ALLOWED_PATHS"] = str(WORKSPACE)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "ltspice_mcp"],
        env=env,
        cwd=str(WORKSPACE),
    )


def text(result) -> str:
    return result.content[0].text


def ok(result) -> bool:
    return not result.isError and not text(result).startswith("ERROR:")


def heading(msg: str):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def step(num: int | str, msg: str):
    print(f"\n--- Step {num}: {msg} ---")


async def run():
    params = _params()
    async with stdio_client(params) as (rs, ws), ClientSession(rs, ws) as session:
        init = await session.initialize()
        heading(f"Connected to {init.serverInfo.name}")

        # ----------------------------------------------------------
        # Step 1: Check server status
        # ----------------------------------------------------------
        step(1, "Check server status")
        r = await session.call_tool("server_status", {}, read_timeout_seconds=TIMEOUT)
        print(text(r))

        has_simulator = "degraded" not in text(r)

        # ----------------------------------------------------------
        # Step 2: Create a Sallen-Key 2nd-order low-pass filter
        # ----------------------------------------------------------
        step(2, "Create Sallen-Key low-pass filter netlist")

        # Target: fc = 1/(2*pi*sqrt(R1*R2*C1*C2)) ≈ 1kHz
        # With R1=R2=15.9k, C1=C2=10nF: fc ≈ 1kHz
        netlist = textwrap.dedent("""\
                * Sallen-Key 2nd-order Low-Pass Filter
                * Target cutoff: 1kHz, Butterworth (Q=0.707)
                R1 in mid 15.9k
                R2 mid inv 15.9k
                C1 mid out 10n
                C2 inv 0 10n
                * Unity-gain buffer (ideal op-amp via VCVS)
                E1 out 0 inv 0 1e6
                * Source
                V1 in 0 AC 1 PULSE(0 1 0 1n 1n 0.5m 1m)
                * Analysis
                .ac dec 200 10 100k
                .tran 5m
                .meas AC fc WHEN mag(V(out)/V(in))=0.707
                .meas AC gain_dc FIND mag(V(out)/V(in)) AT=10
                .meas TRAN vout_max MAX V(out)
            """)

        r = await session.call_tool(
            "create_netlist",
            {"name": "sallen_key_lpf", "content": netlist},
            read_timeout_seconds=TIMEOUT,
        )
        if ok(r):
            print(f"  OK: {text(r)}")
        else:
            print(f"  FAIL: {text(r)}")
            return

        # ----------------------------------------------------------
        # Step 3: Read circuit and list components
        # ----------------------------------------------------------
        step(3, "Read circuit back and list components")

        r = await session.call_tool(
            "read_circuit",
            {"path": "sallen_key_lpf.cir"},
            read_timeout_seconds=TIMEOUT,
        )
        print(f"  Circuit content:\n{textwrap.indent(text(r), '    ')}")

        r = await session.call_tool(
            "list_components",
            {"path": "sallen_key_lpf.cir"},
            read_timeout_seconds=TIMEOUT,
        )
        print(f"\n  Components:\n{textwrap.indent(text(r), '    ')}")

        # ----------------------------------------------------------
        # Step 4: Read parameters
        # ----------------------------------------------------------
        step(4, "Check parameters")
        r = await session.call_tool(
            "parameter",
            {"path": "sallen_key_lpf.cir"},
            read_timeout_seconds=TIMEOUT,
        )
        print(f"  Parameters: {text(r)}")

        # ----------------------------------------------------------
        # Step 5: Modify — change cutoff to ~500Hz by doubling R values
        # ----------------------------------------------------------
        step(5, "Change cutoff to ~500Hz (double R values)")
        r = await session.call_tool(
            "set_component_value",
            {"path": "sallen_key_lpf.cir", "values": {"R1": "31.8k", "R2": "31.8k"}},
            read_timeout_seconds=TIMEOUT,
        )
        print(f"  {text(r)}")

        # Verify
        r = await session.call_tool(
            "list_components",
            {"path": "sallen_key_lpf.cir", "reference": "R1"},
            read_timeout_seconds=TIMEOUT,
        )
        print(f"  Verify R1: {text(r)}")

        # Change back to 1kHz
        r = await session.call_tool(
            "set_component_value",
            {"path": "sallen_key_lpf.cir", "values": {"R1": "15.9k", "R2": "15.9k"}},
            read_timeout_seconds=TIMEOUT,
        )
        print(f"  Restored: {text(r)}")

        # ----------------------------------------------------------
        # Step 6: Run simulation (needs LTspice)
        # ----------------------------------------------------------
        step(6, "Run simulation")
        if not has_simulator:
            print("  SKIPPED: No simulator available")
            print("  (Install LTspice and set simulator.path in ltspice-mcp.toml)")
        else:
            r = await session.call_tool(
                "run_simulation",
                {"netlist": "sallen_key_lpf.cir", "wait": True},
                read_timeout_seconds=SIM_TIMEOUT,
            )
            if ok(r):
                print(f"  Simulation complete:\n{textwrap.indent(text(r), '    ')}")
            else:
                print(f"  FAIL: {text(r)}")
                has_simulator = False

        # ----------------------------------------------------------
        # Step 7: Analyze results (needs completed sim)
        # ----------------------------------------------------------
        step(7, "Analyze results")
        if not has_simulator:
            print("  SKIPPED: No simulation results available")
        else:
            # Measurements from log
            r = await session.call_tool(
                "ltspice_measurements",
                {"log_file": "sallen_key_lpf.log"},
                read_timeout_seconds=TIMEOUT,
            )
            print(f"  Measurements:\n{textwrap.indent(text(r), '    ')}")

            # Simulation summary
            r = await session.call_tool(
                "simulation_summary",
                {"raw_file": "sallen_key_lpf.raw"},
                read_timeout_seconds=TIMEOUT,
            )
            print(f"  Summary:\n{textwrap.indent(text(r), '    ')}")

            # Signal stats at output
            r = await session.call_tool(
                "signal_stats",
                {"raw_file": "sallen_key_lpf.raw", "signal": "V(out)"},
                read_timeout_seconds=TIMEOUT,
            )
            print(f"  V(out) stats:\n{textwrap.indent(text(r), '    ')}")

        # ----------------------------------------------------------
        # Step 8: Configure parameter sweep
        # ----------------------------------------------------------
        step(8, "Configure frequency sweep (vary R1)")
        r = await session.call_tool(
            "configure_sweep",
            {
                "netlist": "sallen_key_lpf.cir",
                "parameters": [
                    {
                        "name": "R1",
                        "type": "component",
                        "start": 5000,
                        "stop": 50000,
                        "points": 10,
                        "scale": "log",
                    }
                ],
            },
            read_timeout_seconds=TIMEOUT,
        )
        if ok(r):
            print(f"  {text(r)}")
            sweep_id = re.search(r"Config ID: (\S+)", text(r))
            if has_simulator and sweep_id:
                step("8b", "Run sweep")
                r = await session.call_tool(
                    "run_sweep",
                    {"config_id": sweep_id.group(1)},
                    read_timeout_seconds=SIM_TIMEOUT,
                )
                print(f"  {text(r)}")
        else:
            print(f"  FAIL: {text(r)}")

        # ----------------------------------------------------------
        # Step 9: Configure Monte Carlo
        # ----------------------------------------------------------
        step(9, "Configure Monte Carlo (5% resistors, 10% caps)")
        r = await session.call_tool(
            "configure_montecarlo",
            {
                "netlist": "sallen_key_lpf.cir",
                "tolerances": [
                    {"ref": "resistors", "tolerance": 0.05},
                    {"ref": "capacitors", "tolerance": 0.10},
                ],
                "num_runs": 50,
            },
            read_timeout_seconds=TIMEOUT,
        )
        if ok(r):
            print(f"  {text(r)}")
            mc_id = re.search(r"Config ID: (\S+)", text(r))
            if has_simulator and mc_id:
                step("9b", "Run Monte Carlo")
                r = await session.call_tool(
                    "run_montecarlo",
                    {"config_id": mc_id.group(1)},
                    read_timeout_seconds=SIM_TIMEOUT,
                )
                print(f"  {text(r)}")
        else:
            print(f"  FAIL: {text(r)}")

        # ----------------------------------------------------------
        # Step 10: Check resources
        # ----------------------------------------------------------
        step(10, "Browse resources")
        from pydantic import AnyUrl

        r = await session.read_resource(AnyUrl("ltspice://config"))
        config = json.loads(r.contents[0].text)  # type: ignore[union-attr]
        print(
            f"  Config: working_dir={config['working_dir']}, "
            f"simulators={config['detected_simulators']}"
        )

        r = await session.read_resource(AnyUrl("ltspice://netlists/"))
        netlists = json.loads(r.contents[0].text)  # type: ignore[union-attr]
        print(f"  Netlists: {[n['name'] for n in netlists['netlists']]}")

        r = await session.read_resource(AnyUrl("ltspice://results/"))
        results = json.loads(r.contents[0].text)  # type: ignore[union-attr]
        print(f"  Jobs: {results['count']}")

        # ----------------------------------------------------------
        # Step 11: Get a prompt
        # ----------------------------------------------------------
        step(11, "Get the characterize-filter prompt")
        r = await session.get_prompt(
            "characterize_filter",
            {"path": "sallen_key_lpf.cir"},
        )
        msg = r.messages[0]
        prompt_text = msg.content if isinstance(msg.content, str) else msg.content.text  # type: ignore[union-attr]
        print(f"  Prompt ({len(prompt_text)} chars):")
        print(f"    {prompt_text[:200]}...")

        # ----------------------------------------------------------
        heading("Scenario complete!")
        if has_simulator:
            print("  All steps executed with real simulation.")
        else:
            print("  Circuit editing steps passed.")
            print("  Simulation steps skipped (no LTspice detected).")
            print("  To run full scenario, set simulator.path in ltspice-mcp.toml")
        print()


if __name__ == "__main__":
    asyncio.run(run())
