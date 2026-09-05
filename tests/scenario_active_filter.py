"""Real-world scenario: Design and simulate a Sallen-Key low-pass filter.

Run with: uv run python tests/scenario_active_filter.py

This exercises the consolidated tool surface as a real MCP client would:
1. Ask the server what it can do (inspect capabilities)
2. Author the decks with plain file writes — the AUTHOR plane offers no
   netlist editing on purpose; an agent writes SPICE text natively
3. Gate them with verify_circuit before spending a simulator on them
4. Run the AC characterization as an experiment with attached bode_filter
   analysis, sweeping R1 across the design space in one call
5. Wait on the receipt if the dwell ran out (jobs)
6. Run the transient step response with attached signal_stats
7. Re-analyze a finished job after the fact (analyze_results)
8. Browse resources and a workflow prompt

Steps 1-3 work without a simulator. Steps 4-7 need one; the script reports
which steps succeed and which were skipped.
"""

import asyncio
import json
import os
import sys
import textwrap
import time
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

TIMEOUT = 30.0
SIM_TIMEOUT = 180.0

# Where to run — use the project workspace dir
WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

# Sallen-Key 2nd-order low-pass, unity gain, fc ~= 1kHz with R=15.9k, C=10n.
# One deck per analysis: a recipe is bound to a run type, so an AC metric and
# a transient metric cannot come from the same run.
FILTER_CORE = textwrap.dedent("""\
    R1 in mid 15.9k
    R2 mid inv 15.9k
    C1 mid out 10n
    C2 inv 0 10n
    * Unity-gain buffer (ideal op-amp via VCVS): out follows the inv node,
    * which is what makes this equal-R/equal-C pair a Sallen-Key section.
    E1 out 0 inv 0 1
""")

AC_DECK = (
    "* Sallen-Key 2nd-order Low-Pass Filter — AC characterization\n"
    + FILTER_CORE
    + "V1 in 0 AC 1\n"
    ".ac dec 200 10 100k\n"
    ".meas AC gain_dc FIND mag(V(out)/V(in)) AT=10\n"
    ".end\n"
)

TRAN_DECK = (
    "* Sallen-Key 2nd-order Low-Pass Filter — step response\n"
    + FILTER_CORE
    + "V1 in 0 PULSE(0 1 0 1n 1n 5m 10m)\n"
    ".tran 10u 5m\n"
    ".end\n"
)


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


def structured(result) -> dict:
    return result.structured_content or {}


def ok(result) -> bool:
    return not result.is_error and not text(result).startswith("ERROR:")


# Steps that produced a tool error despite being expected to succeed; the
# script exits nonzero when any accumulate, so a rotted call can't hide
# behind pretty printing.
FAILURES: list[str] = []


def expect_ok(result, label: str) -> bool:
    if ok(result):
        return True
    FAILURES.append(f"{label}: {text(result)[:200]}")
    print(f"  FAIL ({label}): {text(result)}")
    return False


def heading(msg: str):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def step(num: int | str, msg: str):
    print(f"\n--- Step {num}: {msg} ---")


def show_findings(data: dict):
    findings = data.get("findings") or []
    if not findings:
        print(f"  Clean: {data.get('checks_run')}")
        return
    for finding in findings:
        print(f"  {finding['severity']}: {finding['rule_id']} ({finding.get('subject')})")


def show_receipt(data: dict):
    print(f"  job_id={data.get('job_id')} status={data.get('status')} ({data['outcome']})")
    progress = data.get("progress") or {}
    print(
        f"  progress: {progress.get('terminal')}/{progress.get('expanded')} terminal, "
        f"{progress.get('remaining')} remaining"
    )
    for failure in data.get("failures") or []:
        print(f"  failure: {failure.get('code')} — {failure.get('message', '')[:120]}")


def show_analysis(analysis: dict | None):
    """Print recipe results, from either an analyze_results payload or the
    ``analysis`` stage block a receipt carries (which nests one inside)."""
    if not analysis:
        print("  (no analysis in this payload)")
        return
    if "result" in analysis:
        if analysis.get("error"):
            print(f"  analysis error: {analysis['error']}")
        analysis = analysis.get("result") or {}
    for key, result in (analysis.get("results") or {}).items():
        print(f"  {key} ({result.get('metric')}):")
        for item in (result.get("reduced") or [])[:4]:
            print(f"    {item.get('stat')}={item.get('value')} @ {item.get('assignments')}")
        for item in (result.get("values") or [])[:3]:
            print(f"    {json.dumps(item.get('value'))[:160]}")
        for warning in result.get("warnings") or []:
            print(f"    warning: {warning}")


async def wait_for(session, job_id: str, label: str):
    """Block on a receipt whose dwell ran out, then show what came back."""
    r = await session.call_tool(
        "jobs",
        {"action": "wait", "job_id": job_id, "timeout_s": 180},
        read_timeout_seconds=SIM_TIMEOUT,
    )
    if not expect_ok(r, f"jobs wait ({label})"):
        return
    data = structured(r)
    show_receipt(data)
    print(f"  analysis_status={data.get('analysis_status')}")
    show_analysis(data.get("analysis"))


async def run():
    params = _params()
    stamp = int(time.time())
    async with stdio_client(params) as (rs, ws), ClientSession(rs, ws) as session:
        init = await session.initialize()
        heading(f"Connected to {init.server_info.name}")

        # ----------------------------------------------------------
        # Step 1: Ask the server what it can do
        # ----------------------------------------------------------
        step(1, "Inspect server capabilities")
        r = await session.call_tool(
            "inspect", {"queries": [{"kind": "capabilities"}]}, read_timeout_seconds=TIMEOUT
        )
        expect_ok(r, "inspect capabilities")
        caps = structured(r)["results"][0]["data"]
        print(f"  Simulators: {caps['simulators'] or 'none detected'}")
        print(f"  Default: {caps['default_simulator']}, exporter={caps['exporter_available']}")
        print(f"  Profile: {caps['tool_profile']}, roots={caps['allowed_paths']}")

        has_simulator = caps["default_simulator"] is not None

        # ----------------------------------------------------------
        # Step 2: Author the decks — plain file writes, no tool needed
        # ----------------------------------------------------------
        step(2, "Author the Sallen-Key decks (plain file writes)")
        ac_path = WORKSPACE / "sallen_key_ac.cir"
        tran_path = WORKSPACE / "sallen_key_tran.cir"
        ac_path.write_text(AC_DECK)
        tran_path.write_text(TRAN_DECK)
        print(f"  Wrote {ac_path.name} ({len(AC_DECK)} bytes)")
        print(f"  Wrote {tran_path.name} ({len(TRAN_DECK)} bytes)")

        # ----------------------------------------------------------
        # Step 3: Gate both decks before spending a simulator on them
        # ----------------------------------------------------------
        step(3, "Verify both decks")
        for deck in (ac_path, tran_path):
            r = await session.call_tool(
                "verify_circuit", {"path": deck.name}, read_timeout_seconds=TIMEOUT
            )
            if expect_ok(r, f"verify_circuit {deck.name}"):
                print(f"  {deck.name}:")
                show_findings(structured(r))

        # ----------------------------------------------------------
        # Step 4: AC characterization — one call sweeps R1 and analyzes
        # ----------------------------------------------------------
        step(4, "Run the AC sweep with attached bode_filter analysis")
        ac_job: str | None = None
        if not has_simulator:
            print("  SKIPPED: No simulator available")
            print("  (Install ngspice, or set simulator.path in ltspice-mcp.toml)")
        else:
            r = await session.call_tool(
                "run_experiments",
                {
                    "request_id": f"scenario-ac-{stamp}",
                    "circuits": [{"path": ac_path.name, "id": "sallen_key"}],
                    "variations": [
                        {"kind": "assign", "assign": {"R1": ["7.95k", "15.9k", "31.8k"]}}
                    ],
                    "execution": {"wait_s": 90},
                    "analyze": {
                        "recipes": [
                            {
                                "key": "cutoff",
                                "metric": "bode_filter",
                                "signal": "V(out)",
                                "reduce": ["min", "max"],
                                "field": "cutoff_high_hz",
                            }
                        ],
                        "group_by": ["R1"],
                    },
                },
                read_timeout_seconds=SIM_TIMEOUT,
            )
            if expect_ok(r, "run_experiments (ac)"):
                data = structured(r)
                show_receipt(data)
                ac_job = data.get("job_id")
                show_analysis(data.get("analysis"))

        # ----------------------------------------------------------
        # Step 5: Wait on the receipt if the dwell ran out
        # ----------------------------------------------------------
        step(5, "Wait for the AC job to finish")
        if ac_job is None:
            print("  SKIPPED: no job to wait for")
        else:
            await wait_for(session, ac_job, "ac")

        # ----------------------------------------------------------
        # Step 6: Step response — a transient run with attached stats
        # ----------------------------------------------------------
        step(6, "Run the step response with attached signal_stats")
        tran_job: str | None = None
        if not has_simulator:
            print("  SKIPPED: No simulator available")
        else:
            r = await session.call_tool(
                "run_experiments",
                {
                    "request_id": f"scenario-tran-{stamp}",
                    "circuits": [{"path": tran_path.name, "id": "sallen_key"}],
                    "execution": {"wait_s": 90},
                    "analyze": {
                        "recipes": [{"key": "vout", "metric": "signal_stats", "signal": "V(out)"}]
                    },
                },
                read_timeout_seconds=SIM_TIMEOUT,
            )
            if expect_ok(r, "run_experiments (tran)"):
                data = structured(r)
                show_receipt(data)
                tran_job = data.get("job_id")
                show_analysis(data.get("analysis"))
                if tran_job and data["outcome"] == "in_progress":
                    await wait_for(session, tran_job, "tran")

        # ----------------------------------------------------------
        # Step 7: Re-analyze a finished job after the fact
        # ----------------------------------------------------------
        step(7, "Analyze the finished runs again, asking new questions")
        if ac_job is None:
            print("  SKIPPED: no completed job to analyze")
        else:
            r = await session.call_tool(
                "analyze_results",
                {
                    "sources": [{"job_id": ac_job, "runs": "all", "label": "ac"}],
                    "recipes": [
                        {"key": "measured", "metric": "measurements"},
                        {
                            "key": "gain_1k",
                            "metric": "bode_point",
                            "signal": "V(out)",
                            "at_hz": 1000,
                        },
                    ],
                },
                read_timeout_seconds=SIM_TIMEOUT,
            )
            if expect_ok(r, "analyze_results"):
                data = structured(r)
                coverage = data["coverage"]
                print(
                    f"  coverage: {coverage['runs_analyzed']}/{coverage['runs_requested']} "
                    "run(s) analyzed"
                )
                show_analysis(data)
                for failure in data.get("failures") or []:
                    print(f"  failure: {failure.get('code')} — {failure.get('message', '')[:120]}")

        # ----------------------------------------------------------
        # Step 8: What did the session leave behind?
        # ----------------------------------------------------------
        step(8, "List the circuits this server has touched")
        r = await session.call_tool("jobs", {"action": "list"}, read_timeout_seconds=TIMEOUT)
        if expect_ok(r, "jobs list"):
            for group in structured(r)["items"]:
                print(
                    f"  {Path(group['path']).name}: {group['status_counts'] or 'no jobs'} "
                    f"({group['recent_jobs_total']} recent)"
                )

        # ----------------------------------------------------------
        # Step 9: Check resources
        # ----------------------------------------------------------
        step(9, "Browse resources")

        r = await session.read_resource("spice://config")
        config = json.loads(r.contents[0].text)  # type: ignore[union-attr]
        print(
            f"  Config: working_dir={config['working_dir']}, "
            f"simulators={config['detected_simulators']}"
        )

        r = await session.read_resource("spice://netlists/")
        netlists = json.loads(r.contents[0].text)  # type: ignore[union-attr]
        print(f"  Netlists: {[n['name'] for n in netlists['netlists']]}")

        r = await session.read_resource("spice://results/")
        results = json.loads(r.contents[0].text)  # type: ignore[union-attr]
        print(f"  Jobs: {results['count']}")

        # ----------------------------------------------------------
        # Step 10: Get a prompt
        # ----------------------------------------------------------
        step(10, "Get the characterize-filter prompt")
        r = await session.get_prompt("characterize_filter", {"path": ac_path.name})
        msg = r.messages[0]
        prompt_text = msg.content if isinstance(msg.content, str) else msg.content.text  # type: ignore[union-attr]
        print(f"  Prompt ({len(prompt_text)} chars):")
        print(f"    {prompt_text[:200]}...")

        # ----------------------------------------------------------
        heading("Scenario complete!")
        if FAILURES:
            print(f"  {len(FAILURES)} step(s) FAILED:")
            for f in FAILURES:
                print(f"    - {f}")
        elif has_simulator:
            print(f"  All steps executed with real simulation (jobs {ac_job}, {tran_job}).")
        else:
            print("  Authoring and verification steps passed.")
            print("  Simulation steps skipped (no simulator detected).")
            print("  To run the full scenario, install ngspice or set simulator.path.")
        print()


if __name__ == "__main__":
    asyncio.run(run())
    if FAILURES:
        sys.exit(1)
