"""V(fb)'s gain at 1 kHz, or null when the run has no such trace."""

from pathlib import Path

import numpy as np

from ltspice_eval.references import axis, open_raw, wave


def compute(workspace: Path) -> dict:
    raw = open_raw(workspace / "ac_rc.raw")
    if "V(fb)" not in raw.get_trace_names():
        return {"gain_db": None}
    frequency = axis(raw)
    gain = np.abs(wave(raw, "V(fb)")) / np.abs(wave(raw, "V(in)"))
    return {"gain_db": float(np.interp(3.0, np.log10(frequency), 20 * np.log10(gain)))}
