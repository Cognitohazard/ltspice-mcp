"""Bounded, static numeric facts for hierarchy discovery; never executes code."""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.ngspice_compat import multiplies_m

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
    ast.Pow: operator.pow,
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


def evaluate(
    expression: str,
    lookup: Callable[[str], float],
    *,
    simulator: str,
    element_value: bool = False,
) -> float:
    """Interpret finite arithmetic, identifiers and SPICE suffixes under fixed bounds.

    With ``element_value`` the text is an element's value field, where a bare
    number is read as LTspice reads a value (``1k5`` is 1500, ``9V1`` is 9,
    recorded). What LTspice makes of those spellings inside an expression is
    not recorded, so there they are declined.
    """
    if simulator == "ltspice" and "^" in expression:
        raise ValueError("LTspice caret semantics are unsupported")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression exceeds {MAX_EXPRESSION_LENGTH} characters")
    text = expression.strip()
    if element_value and simulator == "ltspice" and _NUMBER.fullmatch(text):
        value = parse_spice_value(text)
        if not math.isfinite(value):
            raise ValueError("expression is not a finite real number")
        return value
    while len(text) >= 2 and (text[0], text[-1]) in {("{", "}"), ("'", "'")}:
        text = text[1:-1].strip()
    if re.search(r"[^\w\s.()+*/^µμ-]", text):
        raise ValueError("unsupported expression characters")
    if len(re.findall(r"\^|\*\*", text)) > 1:
        raise ValueError("multiple power operators have unsupported associativity")
    text = _NUMBER.sub(_number, text)
    tree = ast.parse(text.replace("^", "**"), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > MAX_EXPRESSION_NODES:
        raise ValueError(f"expression exceeds {MAX_EXPRESSION_NODES} syntax nodes")

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
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ValueError("power exponent exceeds 64")
            result = _BINARY[type(node.op)](left, right)
        else:
            raise ValueError("unsupported expression; only static arithmetic is resolved")
        if not isinstance(result, (float, int)) or not math.isfinite(result):
            raise ValueError("expression is not a finite real number")
        return float(result)

    return visit(tree.body)


class Environment:
    """Lexical environment with caller-evaluated overrides and lazy dependencies."""

    def __init__(
        self,
        expressions: Mapping[str, str],
        *,
        simulator: str,
        parent: Environment | None = None,
        overrides: Mapping[str, NumericFact] | None = None,
        dynamic_reason: str | None = None,
    ) -> None:
        self.simulator = simulator
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
                expression, lookup, simulator=self.simulator, element_value=element_value
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


def ngspice_multiplies_m(ngbehavior: str | None) -> bool:
    """Whether ngspice in this compatibility mode multiplies a card's own ``m``
    by the ``m`` its subcircuit passes down, rather than replacing it.

    Only a mode ngspice reads with ``hs`` set multiplies (``hs``, ``hsa``,
    not ``hsps`` or ``HSA``: ``ngspice_compat``); every other mode, the
    ``kiltpsa`` spicelib runs by default included, replaces it. Measured on
    ngspice-42: ``R1 p 0 1k m=3`` in a body called with ``m=4`` builds 12
    copies under ``hsa`` and 4 under ``kiltpsa``.
    """
    return multiplies_m(ngbehavior)


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
