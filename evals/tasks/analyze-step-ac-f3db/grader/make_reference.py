"""The -3 dB point of |V(out)/V(in)| for each step, relative to its first
(lowest-frequency) point, interpolated in log frequency."""

from pathlib import Path

import numpy as np

from ltspice_eval.references import axis, crossing, open_raw, steps, wave


def compute(workspace: Path) -> dict:
    raw = open_raw(workspace / "step_ac.raw")
    cutoffs = []
    for step in steps(raw):
        frequency = axis(raw, step)
        gain = np.abs(wave(raw, "V(out)", step)) / np.abs(wave(raw, "V(in)", step))
        cutoffs.append(crossing(frequency, 20 * np.log10(gain / gain[0]), -3.0, log_x=True))
    return {"f3db_hz": cutoffs}
