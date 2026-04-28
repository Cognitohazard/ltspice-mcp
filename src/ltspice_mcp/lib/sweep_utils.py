"""Sweep range generation and batch job ID utilities.

Provides helper functions for generating parameter sweep value arrays
and unique identifiers for sweep/Monte Carlo batch jobs and configs.
"""

import math
import time
import uuid

import numpy as np


def generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix.

    Format: {prefix}_{timestamp}_{uuid_short}

    Args:
        prefix: ID prefix (e.g. "sim", "sweep", "montecarlo", "mc")

    Returns:
        ID string (e.g., "sweep_1707916800_a3f7b2c4")
    """
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def generate_batch_job_id(job_type: str) -> str:
    """Generate unique batch job ID."""
    return generate_id(job_type)


def generate_config_id(config_type: str) -> str:
    """Generate unique configuration ID for sweep or Monte Carlo configs."""
    return generate_id(config_type)


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
    if step is None and points is None:
        raise ValueError("Either step or points must be provided, not neither.")
    if step is not None and points is not None:
        raise ValueError("step and points are mutually exclusive — provide one, not both.")

    if points is not None and points < 1:
        raise ValueError(f"points must be >= 1 (got points={points}).")

    if scale == "linear":
        if points is not None:
            arr = np.linspace(start, stop, int(points))
        else:
            assert step is not None
            if step == 0:
                raise ValueError("Linear scale step must be != 0.")
            # np.arange silently returns an empty array on direction mismatch
            if (stop > start and step < 0) or (stop < start and step > 0):
                raise ValueError(
                    f"Linear sweep step direction does not match range: "
                    f"start={start}, stop={stop}, step={step}. "
                    f"Use step>0 for ascending ranges and step<0 for descending."
                )
            # Epsilon guard: extend stop slightly so np.arange includes the endpoint.
            # Scales with step magnitude to avoid relative precision issues.
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
            assert step is not None
            # step is the multiplicative factor per step (geometric series);
            # step=1 would divide by zero in log(1)
            if step <= 0:
                raise ValueError(f"Log scale step must be positive (got step={step}).")
            if step == 1:
                raise ValueError(
                    "Log scale step must be != 1 (step=1 is a degenerate multiplier)."
                )
            # Direction must agree: ascending range needs step>1 (grow each
            # iteration); descending needs step<1. Otherwise log(stop/start)
            # and log(step) have opposite signs and n comes out negative,
            # which lets numpy raise a cryptic "Number of samples, -N, must
            # be non-negative" error from inside geomspace.
            if (stop > start and step < 1) or (stop < start and step > 1):
                raise ValueError(
                    f"Log sweep step direction does not match range: "
                    f"start={start}, stop={stop}, step={step}. "
                    f"Use step>1 for ascending ranges and 0<step<1 for descending."
                )
            n = round(math.log(stop / start) / math.log(step)) + 1
            arr = np.geomspace(start, stop, n)
    else:
        raise ValueError(f"Unknown scale '{scale}'. Expected 'linear' or 'log'.")

    return [float(v) for v in arr]
