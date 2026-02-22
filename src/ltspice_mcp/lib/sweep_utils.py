"""Sweep range generation and batch job ID utilities.

Provides helper functions for generating parameter sweep value arrays
and unique identifiers for sweep/Monte Carlo batch jobs and configs.
"""

import math
import time
import uuid

import numpy as np


def generate_batch_job_id(job_type: str) -> str:
    """Generate unique batch job ID.

    Format: {job_type}_{timestamp}_{uuid_short}
    Mirrors the generate_job_id() pattern in sim_runner.py.

    Args:
        job_type: Type of batch job (e.g. "sweep", "montecarlo")

    Returns:
        Job ID string (e.g., "sweep_1707916800_a3f7b2c4")
    """
    return f"{job_type}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def generate_config_id(config_type: str) -> str:
    """Generate unique configuration ID for sweep or Monte Carlo configs.

    Format: {config_type}_{timestamp}_{uuid_short}

    Args:
        config_type: Type of config (e.g. "sweep", "mc")

    Returns:
        Config ID string (e.g., "sweep_1707916800_b1e2d3f4")
    """
    return f"{config_type}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def generate_sweep_range(
    start: float,
    stop: float,
    step: float | None,
    points: int | None,
    scale: str,
) -> list[float]:
    """Generate a sweep range as a list of float values.

    Supports linear and logarithmic scales. Either step or points must be
    provided (they are mutually exclusive).

    For linear scale:
        - If points given: uses np.linspace(start, stop, points)
        - If step given: uses np.arange with an epsilon guard to include stop

    For log scale:
        - If points given: uses np.geomspace(start, stop, points)
        - If step given: computes n from the log ratio, then uses np.geomspace

    All returned values are Python float (not numpy float64) for JSON safety.

    Args:
        start: Start value of the range
        stop: Stop value of the range
        step: Step size (mutually exclusive with points)
        points: Number of points (mutually exclusive with step)
        scale: "linear" or "log"

    Returns:
        List of float values covering [start, stop]

    Raises:
        ValueError: If neither or both of step/points are provided,
                    or if scale is not "linear" or "log",
                    or if log scale receives non-positive start/stop values.
    """
    # Enforce mutual exclusivity
    if step is None and points is None:
        raise ValueError("Either step or points must be provided, not neither.")
    if step is not None and points is not None:
        raise ValueError(
            "step and points are mutually exclusive — provide one, not both."
        )

    if scale == "linear":
        if points is not None:
            arr = np.linspace(start, stop, int(points))
        else:
            # Epsilon guard: extend stop slightly so np.arange includes stop
            arr = np.arange(start, stop + step * 1e-10, step)
    elif scale == "log":
        if start <= 0 or stop <= 0:
            raise ValueError(
                "Log scale requires positive start and stop values "
                f"(got start={start}, stop={stop})."
            )
        if points is not None:
            arr = np.geomspace(start, stop, int(points))
        else:
            # Compute n from the log ratio: n = log(stop/start) / log(step) + 1
            # step here is treated as the multiplicative factor per step
            if step <= 0:
                raise ValueError(
                    f"Log scale step must be positive (got step={step})."
                )
            n = int(round(math.log(stop / start) / math.log(step))) + 1
            arr = np.geomspace(start, stop, n)
    else:
        raise ValueError(
            f"Unknown scale '{scale}'. Expected 'linear' or 'log'."
        )

    # Convert all values to Python float for JSON serialization
    return [float(v) for v in arr]
