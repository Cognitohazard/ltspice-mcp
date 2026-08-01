"""Shared deck fixtures for the per-instance subcircuit-mismatch tests.

Kept as a non-test helper module (leading underscore) so pytest does not try to
collect it. Everything here is shaped like a foundry model deck — a device
wrapped in a ``.subckt`` whose body declares its parameters on a ``.param``
card and drives a BSIM model card — but self-contained, so the suite runs with
no PDK installed.
"""

from __future__ import annotations

from ltspice_mcp.lib.spice_lex import lex
from ltspice_mcp.lib.spice_lex_views import InstanceLine

# The device every deck here instantiates. LEVEL 54 is ngspice's
# HSPICE-compatible BSIM4 numbering, which is the level that accepts the
# delvto/mulu0 pair the engine writes.
MINI_FET = (
    ".subckt minifet d g s b\n"
    ".param w = 1 l = 0.15\n"
    "m0 d g s b minifet_model w = {w} l = {l}\n"
    ".model minifet_model nmos level = 54 vth0 = 0.7 u0 = 0.06\n"
    ".ends minifet\n"
)

# The same device with the model card in the BINNED spelling a foundry uses
# (``.model <name>.0``), which instances still name unsuffixed. The extra
# parameter is there so a body with more declared parameters than the engine
# forwards stays covered.
BINNED_MINI_FET = (
    ".subckt minifet d g s b\n"
    ".param w = 1 l = 0.15 nf = 1\n"
    "m0 d g s b minifet_model w = {w} l = {l}\n"
    ".model minifet_model.0 nmos level = 54 vth0 = 0.7 u0 = 0.06\n"
    ".ends minifet\n"
)


def instance_line(text: str, ref: str) -> InstanceLine:
    """The typed view over one instance card of a rendered deck."""
    card = next(
        c
        for c in lex(text).cards
        if c.kind == "instance" and c.name and c.name.casefold() == ref.casefold()
    )
    return InstanceLine.from_card(card)


def instance_params(text: str, ref: str) -> dict[str, str]:
    """One instance's parameters, plus its model/subcircuit name as ``__model__``."""
    view = instance_line(text, ref)
    return {"__model__": view.model or "", **view.params}


def clone_body(text: str, clone_name: str = "minifet__mcpatch") -> str:
    """The emitted deck's copy of one patched subcircuit, as text.

    Read back out of the deck rather than off the plan, because where the copy
    is produced — once when the plan is built, or once per run — is exactly
    what these tests are checking.
    """
    lines = text.splitlines(keepends=True)
    start = next(
        i
        for i, line in enumerate(lines)
        if line.lower().startswith(".subckt") and clone_name.casefold() in line.casefold()
    )
    stop = next(i for i in range(start, len(lines)) if lines[i].lower().startswith(".ends"))
    return "".join(lines[start : stop + 1])
