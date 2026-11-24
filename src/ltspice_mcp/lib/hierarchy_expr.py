"""Bounded, static numeric facts for hierarchy discovery; never executes code."""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.ngspice_compat import signed_power

MAX_EXPRESSION_LENGTH = 2048
MAX_EXPRESSION_NODES = 128
MAX_PARAMETER_DEPTH = 32
# A number, its exponent and the letters after it, then whatever else of the
# word follows the letters (group 1), so that 1k5 is one token, not 1k and a
# stray 5. The letters are possessive, so a failed fullmatch does not backtrack.
_NUMBER = re.compile(r"(?<![\w.])(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[a-zA-Zµμ]*+(\w*)")
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


@dataclass(frozen=True)
class NumericFact:
    expression: str | None
    value: float | None = None
    unit: str | None = None
    status: str = "unresolved"
    reason: str | None = None


def _number(token: re.Match[str]) -> str:
    """A number token as a Python literal, declining one with more after its letters."""
    if token[1]:
        raise ValueError(f"no recorded reading of '{token[0]}' inside an expression")
    return repr(parse_spice_value(token[0]))


def references_sibling(expression: str, siblings: set[str]) -> bool:
    """Conservatively identify references to other assignments on an X call."""
    names = re.findall(r"[A-Za-z_][\w]*", _NUMBER.sub("0", expression))
    return any(name.casefold() in siblings for name in names)


def _require_mode(simulator: str, ngbehavior: str | None) -> None:
    """Refuse to read an ngspice expression with no mode: what a power gives
    there depends on it."""
    if simulator == "ngspice" and ngbehavior is None:
        raise ValueError(
            "an ngspice expression is read in the deck's ngspice mode "
            "(ngbehavior), and none was given"
        )


#: The LTspice builds whose arithmetic is recorded, as ``Naming.read_as``
#: spells them (``SemanticProfile.ltspice_build``).
LTSPICE_26 = "LTspice 26"
LTSPICE_XVII = "LTspice XVII"


def _power(
    base: float,
    exponent: float,
    simulator: str,
    ngbehavior: str | None,
    ltspice_build: str | None = None,
) -> float:
    """``base`` raised to ``exponent`` where the reader's value is the one the
    simulator computes, else a refusal saying why.

    A positive base gives the same value in every ngspice mode and in both
    recorded LTspice builds, and so does a zero base with a positive exponent
    and a negative base with a whole exponent where the reader keeps its sign.

    LTspice (recorded: ``deck/expr_grid_*``, ``deck/pow_neg_base_*``): ``0**0``
    is 1 in both builds. A zero base with a negative exponent gives no power
    (26 takes 1e200 and XVII 1e308), so it is declined. A negative base with
    a fractional exponent is 0 in LTspice 26, and XVII refuses the deck
    ("invalid exponentiation range"); ``ltspice_build`` says which build reads
    the deck, and with none known it is declined.

    ngspice: a zero base with a zero or negative exponent differs by mode
    (``0^0`` is 0 under hsa and 1 otherwise, and ``0^-1`` stops native and
    kiltpsa ngspice where hsa gives 0), and a negative base with a fractional
    exponent gives a different number in each mode, none of them a real
    power; both are declined. A negative base with an integer exponent keeps
    its sign only where ngspice gives it one (``ngspice_compat.signed_power``),
    and is declined elsewhere: ngspice takes the power of its magnitude there.
    ngspice's own arithmetic is not modelled.
    """
    if base > 0 or (base == 0 and exponent > 0):
        return base**exponent
    if simulator == "ltspice":
        return _ltspice_power(base, exponent, ltspice_build)
    if base == 0:
        raise ValueError(
            "a zero base with a zero or negative exponent is not resolved: ngspice "
            "computes it differently in each mode (0^0 is 0 under hsa and 1 otherwise)"
        )
    if simulator == "ngspice" and not signed_power(ngbehavior):
        raise ValueError(
            f"a negative base is not resolved in ngspice mode {ngbehavior!r}, where "
            "ngspice takes the power of its magnitude ((-2)^3 is 8 there)"
        )
    if not float(exponent).is_integer():
        raise ValueError(
            "a negative base with a fractional exponent is not resolved: it has no "
            "real power, and ngspice gives a different number in each mode"
        )
    return base**exponent


def _ltspice_power(base: float, exponent: float, ltspice_build: str | None) -> float:
    """LTspice's power of a zero or negative base (``_power``)."""
    if base == 0 and exponent == 0:
        return 1.0
    if base == 0:
        raise ValueError(
            "a zero base with a negative exponent has no power: LTspice 26 takes "
            "1e200 and LTspice XVII 1e308 for it"
        )
    if float(exponent).is_integer():
        return base**exponent
    if ltspice_build == LTSPICE_26:
        return 0.0
    if ltspice_build == LTSPICE_XVII:
        raise ValueError(
            "LTspice XVII refuses a negative base with a fractional exponent "
            "(invalid exponentiation range)"
        )
    raise ValueError(
        "a negative base with a fractional exponent is not resolved for an LTspice "
        "build that is not known: LTspice 26 takes 0 and LTspice XVII refuses it"
    )


def _parenthesised(source: bytes, node: ast.expr) -> bool:
    """Whether ``node`` is written inside its own parentheses in ``source``
    (an ``ast`` column is a byte offset, and leaves its parentheses out)."""
    return source[: node.col_offset].rstrip()[-1:] == b"("


#: An element's value with no braces that is one number or one name, with a
#: sign or none: any other such value LTspice reads as one parameter name.
_BARE_VALUE = re.compile(
    r"[+-]?\s*(?:[^\W\d]\w*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[^\W\d_]*+\w*)"
)


#: A power operator followed straight by a negated name (``3^-b``), which
#: ngspice-42 refuses in every mode ("wrongly determined negation"); a negated
#: number (``3^-1``) and a parenthesised name (``3^(-b)``) it computes.
_NEGATED_NAME_EXPONENT = re.compile(r"(?:\^|\*\*)\s*-\s*[A-Za-z_]")


def evaluate(
    expression: str,
    lookup: Callable[[str], float],
    *,
    simulator: str,
    ngbehavior: str | None,
    element_value: bool = False,
    ltspice_build: str | None = None,
) -> float:
    """Interpret finite arithmetic, identifiers and SPICE suffixes under fixed bounds.

    ``ngbehavior`` is the ngspice mode the deck is read in (``""`` for
    native), required for ngspice and None for any other simulator: what a
    power gives depends on it (``_power``).

    With ``element_value`` the text is an element's value field, where a bare
    number, alone or alone in braces, is read as LTspice reads a value
    (``1k5`` and ``{1k5}`` are 1500, ``9V1`` is 9, recorded:
    ``deck/number_grid_braces_*``). What LTspice makes of those spellings
    inside a larger expression is not recorded, so there they are declined.
    An element's value with no braces or quotes that is neither a number nor
    one name LTspice reads as one parameter name, and refuses (recorded:
    ``deck/expr_grid_bare``), so the reader declines it too.

    ``ltspice_build`` is the recorded LTspice build that reads the deck
    (``LTSPICE_26``, ``LTSPICE_XVII``), or None when it is not known: where
    the builds compute a power differently, it decides (``_power``).

    LTspice binds a unary sign to the number after it before a power:
    ``-2**2`` is 4 (recorded: ``deck/expr_grid_negated_power_star``,
    ``deck/pow_neg_base_whole``). A sign before a name or a group raised to a
    power is not recorded, so it is declined for LTspice.
    """
    _require_mode(simulator, ngbehavior)
    if simulator == "ltspice" and "^" in expression:
        raise ValueError("LTspice caret semantics are unsupported")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression exceeds {MAX_EXPRESSION_LENGTH} characters")
    text = expression.strip()
    alone = text[1:-1].strip() if text[:1] == "{" and text[-1:] == "}" else text
    if element_value and simulator == "ltspice" and _NUMBER.fullmatch(alone):
        value = parse_spice_value(alone)
        if not math.isfinite(value):
            raise ValueError("expression is not a finite real number")
        return value
    if (
        element_value
        and simulator == "ltspice"
        and text[:1] not in ("{", "'")
        and not _BARE_VALUE.fullmatch(text)
    ):
        raise ValueError(
            f"LTspice reads an element's value with no braces as one parameter "
            f"name, and refuses '{text}' (Unknown parameter); write {{{text}}}"
        )
    while len(text) >= 2 and (text[0], text[-1]) in {("{", "}"), ("'", "'")}:
        text = text[1:-1].strip()
    if re.search(r"[^\w\s.()+*/^µμ-]", text):
        raise ValueError("unsupported expression characters")
    if simulator == "ngspice" and _NEGATED_NAME_EXPONENT.search(text):
        raise ValueError(
            "ngspice refuses a negated name straight after a power operator (3^-b: "
            "'wrongly determined negation'); write 3^(-b)"
        )
    if len(re.findall(r"\^|\*\*", text)) > 1:
        raise ValueError("multiple power operators have unsupported associativity")
    text = _NUMBER.sub(_number, text)
    source = text.replace("^", "**")
    tree = ast.parse(source, mode="eval")
    written = source.encode()
    if sum(1 for _ in ast.walk(tree)) > MAX_EXPRESSION_NODES:
        raise ValueError(f"expression exceeds {MAX_EXPRESSION_NODES} syntax nodes")

    def raised(base: float, exponent: float) -> float:
        if abs(exponent) > 64:
            raise ValueError("power exponent exceeds 64")
        return _power(base, exponent, simulator, ngbehavior, ltspice_build)

    def visit(node: ast.AST) -> float:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            result = float(node.value)
        elif isinstance(node, ast.Name):
            result = lookup(node.id.casefold())
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            power = node.operand
            if (
                simulator == "ltspice"
                and isinstance(power, ast.BinOp)
                and isinstance(power.op, ast.Pow)
                and not _parenthesised(written, power)
            ):
                if not isinstance(power.left, ast.Constant) or _parenthesised(written, power.left):
                    raise ValueError(
                        "LTspice's reading of a sign before a name or a group raised to "
                        "a power is not recorded; write (-x)**y or -(x**y)"
                    )
                result = raised(sign * visit(power.left), visit(power.right))
            else:
                result = visit(node.operand) * sign
        elif isinstance(node, ast.BinOp) and (
            type(node.op) in _BINARY or isinstance(node.op, ast.Pow)
        ):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow):
                result = raised(left, right)
            else:
                result = _BINARY[type(node.op)](left, right)
        else:
            raise ValueError("unsupported expression; only static arithmetic is resolved")
        if not isinstance(result, (float, int)) or not math.isfinite(result):
            raise ValueError("expression is not a finite real number")
        return float(result)

    return visit(tree.body)


class Environment:
    """Lexical environment with caller-evaluated overrides and lazy dependencies.

    ``simulator``, ``ngbehavior`` and ``ltspice_build`` are who reads the
    expressions, as ``evaluate`` takes them: the mode is required for ngspice."""

    def __init__(
        self,
        expressions: Mapping[str, str],
        *,
        simulator: str,
        ngbehavior: str | None,
        ltspice_build: str | None = None,
        parent: Environment | None = None,
        overrides: Mapping[str, NumericFact] | None = None,
        dynamic_reason: str | None = None,
    ) -> None:
        _require_mode(simulator, ngbehavior)
        self.simulator = simulator
        self.ngbehavior = ngbehavior
        self.ltspice_build = ltspice_build
        self.expressions = {key.casefold(): value for key, value in expressions.items()}
        self.parent = parent
        self.overrides = dict(overrides or {})
        self.dynamic_reason = dynamic_reason
        self._stack: set[str] = set()
        self._cache: dict[str, NumericFact] = {}

    def resolve(self, name: str) -> NumericFact:
        key = name.casefold()
        if key in self.overrides:
            return self.overrides[key]
        if key in self._cache:
            return self._cache[key]
        if key not in self.expressions:
            if self.parent is not None:
                return self.parent.resolve(key)
            return NumericFact(name, reason=f"unknown parameter {name}")
        expression = self.expressions[key]
        if key in self._stack or len(self._stack) >= MAX_PARAMETER_DEPTH:
            return NumericFact(expression, reason="cyclic or over-depth parameter dependency")
        self._stack.add(key)
        try:
            fact = self.fact(expression)
            self._cache[key] = fact
            return fact
        finally:
            self._stack.remove(key)

    def fact(
        self, expression: str | None, unit: str | None = None, *, element_value: bool = False
    ) -> NumericFact:
        if expression is None:
            return NumericFact(None, unit=unit, reason="no explicit value")
        if self.dynamic_reason:
            return NumericFact(expression, unit=unit, reason=self.dynamic_reason)

        def lookup(name: str) -> float:
            fact = self.resolve(name)
            if fact.value is None:
                raise ValueError(fact.reason or f"unresolved parameter {name}")
            return fact.value

        try:
            value = evaluate(
                expression,
                lookup,
                simulator=self.simulator,
                ngbehavior=self.ngbehavior,
                element_value=element_value,
                ltspice_build=self.ltspice_build,
            )
            return NumericFact(expression, value, unit, "resolved")
        except (ValueError, SyntaxError, ArithmeticError, RecursionError) as exc:
            return NumericFact(expression, unit=unit, reason=str(exc))

    def facts(self) -> tuple[tuple[str, NumericFact], ...]:
        return tuple(
            (key, self.resolve(key))
            for key in sorted(self.expressions.keys() | self.overrides.keys())
        )


def scaled_geometry(fact: NumericFact, scale: NumericFact) -> NumericFact:
    if fact.value is None:
        return replace(fact, unit="m")
    if scale.value is None or scale.value <= 0:
        return NumericFact(
            fact.expression, unit="m", reason=scale.reason or "scale must be positive"
        )
    value = fact.value * scale.value
    if not math.isfinite(value) or value <= 0:
        return NumericFact(
            fact.expression, unit="m", reason="geometry must be finite and positive"
        )
    return NumericFact(fact.expression, value, "m", "resolved")


#: Element letters whose cards ngspice appends a rewritten subcircuit's ``m``
#: to, and those it leaves as written (``A`` is an XSPICE code model), as
#: ``listing expand`` shows on ngspice-42. Any other letter is not measured.
NGSPICE_M_TAKERS = frozenset("CDFGIJLMQRXZ")
NGSPICE_M_EXEMPT = frozenset("ABEHKSTVW")


def card_copies(
    own: NumericFact | None,
    passed: NumericFact | None,
    *,
    multiply: bool,
    where: str,
    passed_from: str,
) -> NumericFact:
    """How many parallel copies the simulator builds from one card: its effective ``m``.

    LTspice ignores ``m`` on an ``X`` line at any depth (recorded:
    ``raw/subckt_multiplier``, ``deck/subckt_multiplier_nested``), so to it a
    card's copies are its own ``m`` and ``passed`` is always None.

    ngspice rewrites a subcircuit for ``m`` before it expands the deck
    (measured on ngspice-42 against ``listing expand``):

    - it first drops every subcircuit no instance reaches from the top level,
      so an ``X`` line inside one never counts;
    - it then reads the deck once, in order, includes and ``.lib`` sections
      in place. At an ``X`` line that carries ``m`` it rewrites the
      subcircuit called, unless that subcircuit declares ``m`` (on its
      ``.subckt`` line or in a body ``.param``) or was rewritten already: the
      subcircuit gets an ``m`` parameter, default 1, and every card of its
      body in ``NGSPICE_M_TAKERS`` gets ``m={m}`` appended;
    - an ``X`` line so given ``m`` rewrites its own callee too, but only when
      the reading reaches it after the rewrite, that is when it sits later in
      the deck than the line that rewrote its body;
    - every caller of a rewritten subcircuit then passes its ``m``, 1 where it
      writes none; one that calls a subcircuit taking no parameters stops
      ngspice ("Mismatch: 0 formal but 1 actual params").

    ``passed`` is the ``m`` a rewritten body's cards receive, or None where
    the body is not rewritten. With it the appended ``m`` replaces the card's
    own (``multiply`` False) or multiplies it (``multiply`` True, the HSPICE
    modes); without it the card's own ``m`` stands, and no ``m`` is one copy.
    A subcircuit that declares ``m`` takes an ``X`` line's ``m`` as an ordinary
    parameter, which only a card writing ``m={m}`` reads. An ``m`` that is not
    a positive number leaves the answer unresolved, with a reason naming where
    it is written.
    """
    facts = (
        [(where, own)]
        if passed is None
        else [(passed_from, passed), (where, own)]
        if multiply
        else [(passed_from, passed)]
    )
    copies = 1.0
    for place, fact in facts:
        if fact is None:
            continue
        if fact.value is None or not fact.value > 0.0:
            why = fact.reason or "not a positive number"
            return NumericFact(fact.expression, reason=f"m={fact.expression} on {place}: {why}")
        copies *= fact.value
    return NumericFact(None, copies, None, "resolved")
