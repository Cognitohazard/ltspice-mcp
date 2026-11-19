"""What a mismatch draw for a MOS device inside a subcircuit shares between its
readers: the two ngspice instance parameters it writes, the gate that a device
takes them, its error, and the view of a deck's include closure that the
hierarchy editor (``instance_targeting``) reads.

The draw itself is made on the hierarchy the deck's simulator reads
(``variations._nested_mismatch_edits``), so it exists only for a deck with a
semantic profile (LTspice, ngspice). A deck with none (QSPICE, Xyce) is
refused (``variations._refuse_descent_without_profile``): the server does not
model how those simulators read a subcircuit body.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ltspice_mcp.lib.deck_staging import closure_depth
from ltspice_mcp.lib.device_names import path_text
from ltspice_mcp.lib.hierarchy import ResolvedInstance

# The two ngspice instance parameters a draw inside a subcircuit writes.
# ``delvto`` adds to the signed vth0, so a pFET (negative vth0) shifted by
# +30 mV shows a SMALLER threshold magnitude; callers and receipts stay in the
# signed domain.
VTH_PARAM = "delvto"
MOBILITY_PARAM = "mulu0"

# Model levels that carry delvto/mulu0: BSIM3 and BSIM4 under both ngspice's
# native numbering (8, 14) and its HSPICE-compatible numbering (49, 54). The
# 49/54 pair is the measured one; 8/14 name the same two models. A LEVEL 1-3
# device rejects both parameters: measured, ngspice stops with
# ``unknown parameter (delvto)``.
BSIM_LEVELS: frozenset[int] = frozenset({8, 14, 49, 54})


class MismatchPlanError(ValueError):
    """A per-instance mismatch draw cannot be made as asked."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


#: What LTspice does with ``delvto`` and ``mulu0`` written on a device's card,
#: by (parameter, model level), as the recordings ``deck/mos_delvto_bsim3``,
#: ``deck/mos_delvto_bsim4``, ``deck/mos_mulu0_bsim3`` and
#: ``deck/mos_mulu0_bsim4`` show it on both builds: True where it takes the
#: parameter (``delvto`` on BSIM4 shifts the signed threshold, as a model with
#: ``vth0`` shifted by as much), False where it does not (LTspice 26 stops on
#: "Unknown parameter"; XVII says so and runs as if it were not written). A
#: level not listed is not established.
LTSPICE_INSTANCE_PARAMETERS: dict[tuple[str, float], bool] = {
    (VTH_PARAM, 8): False,
    (VTH_PARAM, 54): True,
    (MOBILITY_PARAM, 8): False,
    (MOBILITY_PARAM, 54): False,
}


def require_instance_parameter(instance: ResolvedInstance, simulator: str, parameter: str) -> None:
    """Refuse writing ``parameter`` (``delvto`` or ``mulu0``) on ``instance``'s
    card where ``simulator`` is known not to apply it to the device's model, or
    is not known to: the one answer to whether it does.

    ngspice takes both on a BSIM3 or BSIM4 model and stops on any other
    (measured, ``BSIM_LEVELS``). LTspice is read off its recordings
    (``LTSPICE_INSTANCE_PARAMETERS``). Every model a binned device may take
    must take it, so the gate reads the exact resolved family, never a name.
    """
    path = path_text(instance.instance) if instance.instance else instance.reference
    if not instance.model_family:
        raise MismatchPlanError(
            "unsupported_model",
            f"{path}: its model is not found, so whether the simulator applies "
            f"{parameter} to it is not known",
        )
    if simulator == "ngspice":
        if any(model.level not in BSIM_LEVELS for model in instance.model_family):
            raise MismatchPlanError(
                "unsupported_model",
                f"{path}: ngspice takes {parameter} only on a BSIM3 or BSIM4 model "
                "(level 8, 14, 49 or 54) and stops on 'unknown parameter' on any other; "
                "the exact model family does not prove that",
            )
        return
    if simulator != "ltspice":
        raise MismatchPlanError(
            "instance_parameter_unestablished",
            f"{path}: whether {simulator} applies {parameter} on a device's card is not "
            "established",
        )
    levels = sorted({model.level for model in instance.model_family}, key=lambda v: v or 0)
    # A level written as an expression reads as None, never as level 1: what
    # it evaluates to is not read here, so it is not established.
    known = {
        level: None if level is None else LTSPICE_INSTANCE_PARAMETERS.get((parameter, level))
        for level in levels
    }
    instead = (
        f"A random mismatch rule with instance {list(instance.instance)} gives the device "
        "its own copy of its model with vth_param or k_param changed instead."
    )
    refused = [level for level, taken in known.items() if taken is False]
    if refused:
        raise MismatchPlanError(
            "instance_parameter_not_taken",
            f"{path}: LTspice does not take {parameter} on a level {refused[0]:g} model: "
            "LTspice 26 stops on 'Unknown parameter', and LTspice XVII says so and runs as "
            f"if it were not written, so every case would run the same device (recorded: "
            f"deck/mos_{parameter}_bsim3, deck/mos_{parameter}_bsim4). {instead}",
        )
    unknown = [level for level, taken in known.items() if taken is None]
    if unknown:
        recorded = ", ".join(
            f"{level:g}"
            for (name, level), _ in LTSPICE_INSTANCE_PARAMETERS.items()
            if name == parameter
        )
        level = "an unread" if unknown[0] is None else f"a level {unknown[0]:g}"
        raise MismatchPlanError(
            "instance_parameter_unestablished",
            f"{path}: whether LTspice applies {parameter} on {level} "
            f"model is not established: the recordings deck/mos_{parameter}_bsim3 and "
            f"deck/mos_{parameter}_bsim4 cover levels {recorded} only. A recording of "
            f"deck/mos_{parameter}_<model> on this level, under the mos-instance-mismatch "
            f"behaviour of inputs/cases.toml, would settle it. {instead}",
        )


@dataclass(frozen=True)
class ClosureFile:
    """One file of a deck's include closure, as the hierarchy editor reads it.

    ``index`` is both the file's identity and its position in the closure a
    caller passes: every producer numbers its files in list order.
    """

    index: int
    path: Path
    text: str

    @property
    def depth(self) -> int:
        return closure_depth(self.index)
