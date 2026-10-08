"""Bounded, static numeric facts for hierarchy discovery; never executes code."""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from ltspice_mcp.lib.format import parse_spice_value

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
