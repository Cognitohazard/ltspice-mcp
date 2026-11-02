"""The step whose V(out) peaks highest, named by the r value the log's
.step line gives it, and that peak."""

import re
from pathlib import Path

import numpy as np

from ltspice_eval.references import open_raw, steps, wave


def compute(workspace: Path) -> dict:
    raw = open_raw(workspace / "step_tran.raw")
    peaks = [float(np.max(np.real(wave(raw, "V(out)", step)))) for step in steps(raw)]
    log = (workspace / "step_tran.log").read_bytes().decode("latin-1")
    values = [float(value) for value in re.findall(r"^\.step r=(\S+)", log, flags=re.MULTILINE)]
    best = int(np.argmax(peaks))
    return {"r": values[best], "peak_v": peaks[best]}
