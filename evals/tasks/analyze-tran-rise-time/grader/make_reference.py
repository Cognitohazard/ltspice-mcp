"""The 10%-90% rise time of V(out), with 0% and 100% its first and last
samples, each crossing linearly interpolated."""

from pathlib import Path

import numpy as np

from ltspice_eval.references import axis, crossing, open_raw, wave


def compute(workspace: Path) -> dict:
    raw = open_raw(workspace / "tran_rc.raw")
    time = axis(raw)
    out = np.real(wave(raw, "V(out)"))
    start, end = float(out[0]), float(out[-1])
    low = crossing(time, out, start + 0.1 * (end - start))
    high = crossing(time, out, start + 0.9 * (end - start))
    assert low is not None and high is not None, "V(out) never crosses 10% and 90%"
    return {"rise_time_s": high - low}
