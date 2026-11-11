"""Which compatibility flags ngspice reads a ``ngbehavior`` mode string as.

ngspice turns the mode into a set of flags once, at startup, and every
behaviour a mode changes follows the flags, not the string. Every question the
server asks about a mode's reading of a deck goes through ``compat_flags``.
Measured on ngspice-42:

- a flag is set when its letters occur anywhere in the string, case
  sensitively:
  ``hsa`` sets ``hs`` and ``a``, ``Hsa`` only ``a``, ``HSA`` none;
- ``spe`` clears ``ps``, ``lt``, ``ki`` and ``eg``: ``spelt`` reads as ``spe``;
- then ``ps`` clears ``hs``: ``hsps`` reads as ``ps``, ``spehsps`` as
  ``hs spe``;
- ``mc`` turns off every flag but ``s3``: in ``ltmc`` a ``.lib`` section is
  read and in ``hsmc`` an ``m`` is not multiplied, but ``s3mc`` still does
  not read the section. ngspice names no mode in its log then.

``ngspice_mode.flags_for`` asks a running ngspice which flags it gives a mode,
and ``tests/test_ngspice_compat.py`` holds this model to it and to the
behaviours below on the ngspice on PATH.
"""

from __future__ import annotations

FLAGS = ("hs", "ps", "lt", "ki", "a", "ll", "s3", "eg", "spe", "mc", "xs")
"""Every flag ngspice-42 reads from a mode string."""


def mode_as_given(text: str) -> str:
    """A mode as the server takes it from a person, in ``[simulator]
    ngbehavior`` and in a query's ``ngbehavior`` alike: stripped and
    lower-cased, so ``HSA`` names ``hsa`` wherever it is typed. ngspice
    itself matches letters case-sensitively, and every mode the server passes
    it is one this function returned."""
    return text.strip().lower()


def compat_flags(mode: str | None) -> frozenset[str]:
    """The flags ngspice sets for ``mode``; empty for no mode."""
    text = mode or ""
    flags = {flag for flag in FLAGS if flag in text}
    if "mc" in flags:
        return frozenset({"mc"} | (flags & {"s3"}))
    if "spe" in flags:
        flags -= {"ps", "lt", "ki", "eg"}
    if "ps" in flags:
        flags.discard("hs")
    return frozenset(flags)


def drops_lib_sections(mode: str | None) -> bool:
    """Whether ngspice in ``mode`` loads no section a ``.lib <file> <section>``
    line selects, which the lint and hierarchy discovery refuse: with ``lt``
    or ``ps`` set it reads the line as two plain includes
    (``reads_lib_as_includes``); with ``s3`` set it does not read the line
    (spice3f5 has no ``.lib``), so the section's names come back undefined."""
    return reads_lib_as_includes(mode) or "s3" in compat_flags(mode)


def reads_lib_as_includes(mode: str | None) -> bool:
    """Whether ngspice in ``mode`` reads ``.lib <file> <section>`` as two plain
    includes, as it does with ``lt`` or ``ps`` set, so the section's name comes
    back as a missing include file. The failure classifier asks this, not
    ``drops_lib_sections``: an ``s3`` run never fails on a missing include,
    so a missing include there has another cause."""
    return bool(compat_flags(mode) & {"lt", "ps"})


def multiplies_m(mode: str | None) -> bool:
    """Whether ngspice in ``mode`` multiplies a card's own ``m`` by the ``m``
    its subcircuit passes down, rather than replacing it, as it does with
    ``hs`` set (``hierarchy_expr.card_copies``). Measured on ngspice-42:
    ``R1 p 0 1k m=3`` in a body called with ``m=4`` builds 12 copies under
    ``hsa`` and 4 under ``kiltpsa`` or ``hsps``."""
    return "hs" in compat_flags(mode)
