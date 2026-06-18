"""MCP prompts: user-invoked workflow starters.

A prompt is a launch template a host surfaces as a slash-command / starter. Each
returns a single user message describing the canonical tool pipeline for a common
task, with the circuit path (and optional node/signal) filled in. They are a
human-facing discovery surface, complementary to the tool descriptions and the
server instructions — those remain the agent's primary orientation channel.
"""

from collections.abc import Mapping

from mcp import types


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


def _step_response(arguments: Mapping[str, str]) -> types.GetPromptResult:
    path = _require(arguments, "path")
    node = (arguments.get("node") or "").strip()
    target = f" at node {node}" if node else ""
    text = (
        f"Measure the step response of `{path}`.\n"
        "1. Drive the input with a step (a PULSE/PWL source) and set a `.tran` run long "
        "enough for the output to settle.\n"
        "2. validate_netlist, then run_simulation.\n"
        f"3. Use edge_metrics or pulse_response on the output{target} for rise time, "
        "overshoot, and settling time; plot_waveform to visualize.\n"
        "Report rise time, overshoot %, and settling time."
    )
    return _text_result("Measure a step response", text)


# Each prompt paired with its builder, so the listing and the dispatch map share
# one source of truth for the name and cannot drift apart.
_PROMPTS = [
    (
        types.Prompt(
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
        _characterize_filter,
    ),
    (
        types.Prompt(
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
        _run_and_plot,
    ),
    (
        types.Prompt(
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
        _step_response,
    ),
]

_BUILDERS = {prompt.name: builder for prompt, builder in _PROMPTS}


def list_prompts() -> list[types.Prompt]:
    """All workflow-starter prompts."""
    return [prompt for prompt, _ in _PROMPTS]


def get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
    """Build a prompt's messages, interpolating its arguments."""
    builder = _BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown prompt: {name}")
    return builder(arguments or {})
