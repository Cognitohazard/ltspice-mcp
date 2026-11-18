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


def require_scoped_bsim(instance: ResolvedInstance) -> None:
    """Gate ngspice's delvto/mulu0 route on the exact resolved family, never a name-only lookup."""
    if not instance.model_family or any(
        model.level not in BSIM_LEVELS for model in instance.model_family
    ):
        raise MismatchPlanError(
            "unsupported_model",
            f"{instance.instance}: exact model family does not prove BSIM delvto/mulu0 capability",
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
