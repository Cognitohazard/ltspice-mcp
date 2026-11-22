"""Shared deck fixtures for the per-instance subcircuit-mismatch tests.

Kept as a non-test helper module (leading underscore) so pytest does not try to
collect it. Everything here is shaped like a foundry model deck — a device
wrapped in a ``.subckt`` whose body declares its parameters on a ``.param``
card and drives a BSIM model card — but self-contained, so the suite runs with
no PDK installed.
"""

from __future__ import annotations

# The device every deck here instantiates. LEVEL 54 is ngspice's
# HSPICE-compatible BSIM4 numbering, which is the level that accepts the
# delvto/mulu0 pair the mismatch draw writes.
MINI_FET = (
    ".subckt minifet d g s b\n"
    ".param w = 1 l = 0.15\n"
    "m0 d g s b minifet_model w = {w} l = {l}\n"
    ".model minifet_model nmos level = 54 vth0 = 0.7 u0 = 0.06\n"
    ".ends minifet\n"
)

# The same device with the model card in the BINNED spelling a foundry uses
# (``.model <name>.0``), which instances still name unsuffixed. The extra
# parameter is there so a body with more declared parameters than the mismatch draw
# forwards stays covered.
BINNED_MINI_FET = (
    ".subckt minifet d g s b\n"
    ".param w = 1 l = 0.15 nf = 1\n"
    "m0 d g s b minifet_model w = {w} l = {l}\n"
    ".model minifet_model.0 nmos level = 54 vth0 = 0.7 u0 = 0.06\n"
    ".ends minifet\n"
)
