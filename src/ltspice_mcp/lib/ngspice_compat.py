"""Which compatibility flags ngspice reads a ``ngbehavior`` mode string as.

ngspice turns the mode into a set of flags once, at startup, and every
behaviour a mode changes follows the flags, not the string. Every question the
server asks about a mode's reading of a deck goes through ``compat_flags``.
Measured on ngspice-42:

- a flag is set when its letters occur anywhere in the string, case
  sensitively: ``hsa`` sets ``hs`` and ``a``, ``Hsa`` only ``a``, ``HSA``
  none;
- ``spe`` clears ``ps``, ``lt``, ``ki`` and ``eg``: ``spelt`` reads as ``spe``;
- then ``ps`` clears ``hs``: ``hsps`` reads as ``ps``, ``spehsps`` as
  ``hs spe``;
- ``mc`` turns the others off: in ``ltmc`` a ``.lib`` section is read and in
  ``hsmc`` an ``m`` is not multiplied, and ngspice names no mode in its log.

``ngspice_mode.flags_for`` asks a running ngspice which flags it gives a mode,
and ``tests/test_ngspice_compat.py`` holds this model to it and to the
behaviours below on the ngspice on PATH.
"""

from __future__ import annotations

FLAGS = ("hs", "ps", "lt", "ki", "a", "ll", "s3", "eg", "spe", "mc")
"""Every flag ngspice-42 reads from a mode string."""


def compat_flags(mode: str | None) -> frozenset[str]:
    """The flags ngspice sets for ``mode``; empty for no mode."""
    text = mode or ""
    flags = {flag for flag in FLAGS if flag in text}
    if "mc" in flags:
        return frozenset({"mc"})
    if "spe" in flags:
        flags -= {"ps", "lt", "ki", "eg"}
    if "ps" in flags:
        flags.discard("hs")
    return frozenset(flags)


def drops_lib_sections(mode: str | None) -> bool:
    """Whether ngspice in ``mode`` reads ``.lib <file> <section>`` as two plain
    includes and drops the section, as it does with ``lt`` or ``ps`` set: the
    section's name then comes back as a missing include file."""
    return bool(compat_flags(mode) & {"lt", "ps"})


def multiplies_m(mode: str | None) -> bool:
    """Whether ngspice in ``mode`` multiplies a card's own ``m`` by the ``m``
    its subcircuit passes down, rather than replacing it, as it does with
    ``hs`` set (``hierarchy_expr.card_copies``)."""
    return "hs" in compat_flags(mode)
