"""Simulation execution tools. (Phase 3)"""

import asyncio
import logging
from pathlib import Path

from mcp import types
from pydantic import Field

from ltspice_mcp.errors import SimulationError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.lib.sim_runner import SimulationRunner
from ltspice_mcp.lib.simulator import no_simulator_message
from ltspice_mcp.state import (
    NON_TERMINAL_LIVE_STATUSES,
    BatchJob,
    SessionState,
)
from ltspice_mcp.tools._base import (
    ToolInput,
    require_simulator,
    resolve_output_folder,
    text_response,
)

logger = logging.getLogger(__name__)


class CancelJobInput(ToolInput):
    """Inputs for cancel_job."""

    job_id: str = Field(description="Job ID of the running simulation to cancel")


async def _get_or_create_runner(
    state: SessionState,
    netlist_path: Path | None = None,
    simulator_class: type | None = None,
) -> SimulationRunner:
    """Get or create a SimulationRunner via the centralized RunnerManager."""
    sim_cls = simulator_class or state.default_simulator
    if sim_cls is None:
        raise SimulationError(no_simulator_message())
    return state.runners.get_sim_runner(
        loop=asyncio.get_running_loop(),
        simulator_class=sim_cls,
        output_folder=await resolve_output_folder(state, netlist_path, simulator=sim_cls),
        max_parallel=state.config.max_parallel_sims,
    )


async def handle_cancel_job(args: CancelJobInput, state: SessionState) -> types.CallToolResult:
    """Cancel a running simulation job.

    Args:
        args: Tool args with job_id
        state: Current session state

    Returns:
        List containing TextContent with cancellation result
    """
    job_id = args.job_id

    job = await services.resolve_job_async(job_id, state)
    if isinstance(job, ExperimentJob):
        services.reject_experiment_job(job, "cancel_job", state)

    # Check if job is running
    if job.status not in NON_TERMINAL_LIVE_STATUSES:
        # A terminal job has nothing to cancel — this is a job-state error, not
        # a simulator-availability one, so suppress the generic SimulationError
        # hint ("verify simulator availability") and point at check_job instead.
        raise SimulationError(
            f"Job {job_id} is not running (status: {job.status}) — it has already "
            f"finished, so there is nothing to cancel. Use check_job('{job_id}') to "
            "read its result.",
            show_hint=False,
        )

    # Cancel via the runner that owns the job. A batch job's cancel event and
    # live-process map live on the SweepRunner/MonteCarloRunner instance that
    # launched it, so route by ownership rather than assuming one runner per kind.
    require_simulator(state)
    if isinstance(job, BatchJob):
        batch_runner = state.runners.get_batch_runner_for(job)
        if batch_runner is None:
            raise SimulationError(
                f"Job {job_id} is marked running but its {job.job_type} runner is no "
                "longer live (server restarted?), so there is no process to cancel."
            )
        await batch_runner.cancel(job, state)
    else:
        # Resolve the runner via the JOB's netlist and recorded simulator, so
        # the cache key (class, output folder) matches the one the job
        # launched with — a mismatch would resolve to a different runner
        # whose kill scopes by the wrong executable names.
        sim_runner = await _get_or_create_runner(
            state, job.netlist, simulator_class=services.simulator_class_for_job(job, state)
        )
        await sim_runner.cancel(job, state)

    return text_response(f"Job {job_id} cancelled")
