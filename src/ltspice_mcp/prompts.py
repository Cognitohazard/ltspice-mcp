"""MCP prompts: user-invoked workflow starters.

A prompt is a launch template a host surfaces as a slash-command / starter. Each
returns a single user message describing the canonical tool pipeline for a common
task, with the circuit path (and optional node/signal) filled in. They are a
human-facing discovery surface, complementary to the tool descriptions and the
server instructions — those remain the agent's primary orientation channel.

Every prompt is written once per tool profile, and both listing and content are
profile-scoped: a starter that walks the caller through tools the connected
client cannot see is a dead end. ``full`` and ``agentic`` share one edition (the
agentic profile drops only the netlist-editing and library wrappers, none of
which these prompts use); ``consolidated`` teaches the same workflows through its
six tools. Same rule as the profile-aware error hints in ``server.py``.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from mcp import types

PromptBuilder = Callable[[Mapping[str, str]], types.GetPromptResult]

# The two prompt editions. A profile reads exactly one of them.
CLASSIC = "classic"
CONSOLIDATED = "consolidated"

# Total over the valid profiles, not a default with one exception: a profile
# added to the config without a line here fails loudly instead of silently
# inheriting an edition that names tools it cannot see. Pinned by test_prompts.
EDITIONS: dict[str, str] = {
    "full": CLASSIC,
    "agentic": CLASSIC,
    "consolidated": CONSOLIDATED,
}


def edition_for(profile: str) -> str:
    """Which prompt edition a tool profile reads."""
    return EDITIONS[profile]


def _text_result(description: str, text: str) -> types.GetPromptResult:
    return types.GetPromptResult(
        description=description,
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=text),
            )
        ],
    )


def _require(arguments: Mapping[str, str], name: str) -> str:
    value = (arguments.get(name) or "").strip()
    if not value:
        raise ValueError(f"Prompt argument '{name}' is required")
    return value


def _characterize_filter(arguments: Mapping[str, str]) -> types.GetPromptResult:
    path = _require(arguments, "path")
    node = (arguments.get("node") or "").strip()
    target = f" at node {node}" if node else ""
    text = (
        f"Characterize the frequency response of the filter in `{path}`.\n"
        "1. Make sure it has an AC sweep covering the band of interest "
        "(e.g. `.ac dec 201 1 1Meg`); add the directive if missing.\n"
        "2. validate_netlist, then run_simulation.\n"
        f"3. Call bode_metrics (mode='filter'){target} for the -3 dB cutoff(s), "
        "passband gain, Q, and roll-off slope.\n"
        "4. plot_waveform to render the Bode plot.\n"
        "Report the filter type, cutoff(s), peak gain, and roll-off (dB/dec)."
    )
    return _text_result("Characterize a filter's AC response", text)


def _characterize_filter_consolidated(arguments: Mapping[str, str]) -> types.GetPromptResult:
    path = _require(arguments, "path")
    signal = f"V({(arguments.get('node') or 'out').strip()})"
    text = (
        f"Characterize the frequency response of the filter in `{path}`.\n"
        "1. Make sure the deck has an AC sweep covering the band of interest "
        "(e.g. `.ac dec 201 1 1Meg`); edit the file directly if the directive "
        "is missing.\n"
        f'2. run_experiments with circuits=[{{"path": "{path}"}}] and an attached '
        'analyze, so one call runs and measures: recipes=[{"key": "response", '
        f'"metric": "bode_filter", "signal": "{signal}"}}, {{"key": "bode", '
        f'"metric": "plot", "signals": ["{signal}"], "log_x": true}}].\n'
        "3. If it returned a receipt instead of results, follow the job with "
        'jobs(action="wait"), then read it with analyze_results using the same '
        "recipes.\n"
        "Report the filter type, cutoff(s), peak gain, and roll-off (dB/dec)."
    )
    return _text_result("Characterize a filter's AC response", text)


def _run_and_plot(arguments: Mapping[str, str]) -> types.GetPromptResult:
    path = _require(arguments, "path")
    signal = (arguments.get("signal") or "").strip()
    target = f" for {signal}" if signal else ""
    text = (
        f"Run a transient simulation of `{path}` and plot the result.\n"
        "1. Ensure a `.tran` directive long enough to show the behavior of interest "
        "(add it if missing).\n"
        "2. validate_netlist, then run_simulation.\n"
        f"3. plot_waveform{target} to visualize; use get_waveform or signal_stats for "
        "numeric detail.\n"
        "Report the key observations (final value, overshoot, settling, anomalies)."
    )
    return _text_result("Run a transient and plot a signal", text)


def _run_and_plot_consolidated(arguments: Mapping[str, str]) -> types.GetPromptResult:
    path = _require(arguments, "path")
    signal = (arguments.get("signal") or "V(out)").strip()
    text = (
        f"Run a transient simulation of `{path}` and plot the result.\n"
        "1. Ensure a `.tran` directive long enough to show the behavior of interest "
        "(add it to the deck if missing).\n"
        f'2. run_experiments with circuits=[{{"path": "{path}"}}] and an attached '
        'analyze: recipes=[{"key": "plot", "metric": "plot", "signals": '
        f'["{signal}"]}}, {{"key": "stats", "metric": "signal_stats", "signal": '
        f'"{signal}"}}].\n'
        "3. If it returned a receipt instead of results, follow the job with "
        'jobs(action="wait"), then analyze_results on it.\n'
        '4. For the numeric table add a {"metric": "waveform"} recipe, or read one '
        'point with {"metric": "value", "expr": ..., "at": ...}.\n'
        "Report the key observations (final value, overshoot, settling, anomalies)."
    )
    return _text_result("Run a transient and plot a signal", text)


def _step_response(arguments: Mapping[str, str]) -> types.GetPromptResult:
    path = _require(arguments, "path")
    node = (arguments.get("node") or "").strip()
    target = f" at node {node}" if node else ""
    text = (
        f"Measure the step response of `{path}`.\n"
        "1. Drive the input with a step (a PULSE/PWL source) and set a `.tran` run long "
        "enough for the output to settle.\n"
        "2. validate_netlist, then run_simulation.\n"
        f"3. Use edge_metrics on the output{target} for rise time and "
        "transient_response(mode='step') for overshoot and settling time; "
        "plot_waveform to visualize.\n"
        "Report rise time, overshoot %, and settling time."
    )
    return _text_result("Measure a step response", text)


def _step_response_consolidated(arguments: Mapping[str, str]) -> types.GetPromptResult:
    path = _require(arguments, "path")
    signal = f"V({(arguments.get('node') or 'out').strip()})"
    text = (
        f"Measure the step response of `{path}`.\n"
        "1. Drive the input with a step (a PULSE/PWL source) and set a `.tran` run long "
        "enough for the output to settle.\n"
        f'2. run_experiments with circuits=[{{"path": "{path}"}}] and an attached '
        'analyze: recipes=[{"key": "edges", "metric": "edges", "signal": '
        f'"{signal}"}}, {{"key": "step", "metric": "transient_response", "signal": '
        f'"{signal}", "mode": "step"}}, {{"key": "plot", "metric": "plot", '
        f'"signals": ["{signal}"]}}].\n'
        "3. If it returned a receipt instead of results, follow the job with "
        'jobs(action="wait"), then analyze_results on it.\n'
        "Report rise time, overshoot %, and settling time."
    )
    return _text_result("Measure a step response", text)


@dataclass(frozen=True)
class _PromptEntry:
    """One prompt's listing entry plus its per-edition builders.

    A prompt appears in a profile's listing only if it has a builder for that
    profile's edition, so a listed prompt always has a body to serve.
    """

    prompt: types.Prompt
    builders: dict[str, PromptBuilder]


_PROMPTS = [
    _PromptEntry(
        prompt=types.Prompt(
            name="characterize_filter",
            description=(
                "Run an AC analysis of an existing filter circuit and report its cutoff, "
                "passband gain, Q, and roll-off, with a Bode plot."
            ),
            arguments=[
                types.PromptArgument(
                    name="path", description="Path to the circuit (.cir/.net/.asc).", required=True
                ),
                types.PromptArgument(
                    name="node", description="Output node to analyze (optional).", required=False
                ),
            ],
        ),
        builders={
            CLASSIC: _characterize_filter,
            CONSOLIDATED: _characterize_filter_consolidated,
        },
    ),
    _PromptEntry(
        prompt=types.Prompt(
            name="run_and_plot",
            description="Run a transient simulation of a circuit and plot a signal.",
            arguments=[
                types.PromptArgument(
                    name="path", description="Path to the circuit (.cir/.net/.asc).", required=True
                ),
                types.PromptArgument(
                    name="signal", description="Node or branch to plot (optional).", required=False
                ),
            ],
        ),
        builders={
            CLASSIC: _run_and_plot,
            CONSOLIDATED: _run_and_plot_consolidated,
        },
    ),
    _PromptEntry(
        prompt=types.Prompt(
            name="step_response",
            description="Drive a step input, measure rise time / overshoot / settling, and plot it.",
            arguments=[
                types.PromptArgument(
                    name="path", description="Path to the circuit (.cir/.net/.asc).", required=True
                ),
                types.PromptArgument(
                    name="node", description="Output node to measure (optional).", required=False
                ),
            ],
        ),
        builders={
            CLASSIC: _step_response,
            CONSOLIDATED: _step_response_consolidated,
        },
    ),
]

_BY_NAME = {entry.prompt.name: entry for entry in _PROMPTS}


def list_prompts(profile: str) -> list[types.Prompt]:
    """Workflow-starter prompts this tool profile has an edition for."""
    edition = edition_for(profile)
    return [entry.prompt for entry in _PROMPTS if edition in entry.builders]


def get_prompt(name: str, arguments: dict[str, str] | None, profile: str) -> types.GetPromptResult:
    """Build a prompt's messages for this tool profile, interpolating arguments."""
    entry = _BY_NAME.get(name)
    builder = entry.builders.get(edition_for(profile)) if entry is not None else None
    if builder is None:
        raise ValueError(f"Unknown prompt: {name}")
    return builder(arguments or {})
