"""Foundation SPICE netlist lexer.

This module provides the line-level lexer (``lex``) and the body-level
tokenizer (``tokenize_body``) that every netlist-touching helper in the
codebase should build on. See ``.claude/plans/spice_lex.md``.

Layer split:
- **Layer 1** (``lex``): walks lines, classifies cards, merges
  ``+``-continuations, tracks ``.SUBCKT`` scope, preserves raw lines for
  byte-faithful round-trip.
- **Layer 3** (``tokenize_body``): walks a merged card body once and
  emits classified tokens (BARE / QUOTED / BRACED / PARENED / KEY_VALUE
  / COMMENT_TRAIL). Layer 2 typed views (in ``spice_lex_views``) read
  classified tokens directly — no view re-parses character classes.

Layer 2 lives in ``spice_lex_views.py`` to keep this module focused on
parsing.

Round-trip contract: ``emit(lex(text)) == text`` byte-for-byte for any
well-formed netlist. For malformed input (unmatched ``.ENDS``, EOF
mid-subckt, trailing cards after ``.END``), the round-trip still holds
but ``LexResult.warnings`` carries diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

CardKind = Literal[
    "model",
    "param",
    "instance",
    "subckt",
    "ends",
    "meas",
    "directive",
    "end",
    "comment",
    "blank",
]


class SpiceLexErrorCategory(Enum):
    """Classification of tokenizer / view faults.

    Callers can pattern-match on category instead of parsing the
    message text. Each category implies a recovery strategy:

    - ``UNTERMINATED_QUOTE``, ``UNBALANCED_BRACE``, ``UNBALANCED_PAREN``:
      input has unmatched delimiters; no recovery, fix the source.
    - ``UNEXPECTED_CHAR``: stray closer at top level; fix the source.
    - ``MALFORMED_CARD``: a card body doesn't match the expected
      shape for its kind (raised by typed views, not by the tokenizer).
    - ``INVALID_RANGE``: caller passed bad offsets to a primitive like
      ``replace_span``.
    """

    UNTERMINATED_QUOTE = "unterminated_quote"
    UNBALANCED_BRACE = "unbalanced_brace"
    UNBALANCED_PAREN = "unbalanced_paren"
    UNEXPECTED_CHAR = "unexpected_char"
    MALFORMED_CARD = "malformed_card"
    INVALID_RANGE = "invalid_range"


class SpiceLexError(ValueError):
    """Structured tokenizer / view fault.

    Carries a machine-readable ``category``, the offending ``position``
    in the source body (when applicable), the source ``body``, and an
    optional ``suggestion`` for the user. Stringifies to a human
    message that bundles all of the above.
    """

    def __init__(
        self,
        category: SpiceLexErrorCategory,
        message: str,
        *,
        position: int = -1,
        body: str = "",
        suggestion: str = "",
    ) -> None:
        self.category = category
        self.position = position
        self.body = body
        self.suggestion = suggestion
        self.message = message
        super().__init__(self._format())

    def _format(self) -> str:
        parts: list[str] = [self.message] if self.message else []
        if self.position >= 0:
            parts.append(f"at position {self.position}")
        if self.body:
            display = self.body if len(self.body) <= 80 else self.body[:77] + "..."
            parts.append(f"in body {display!r}")
        if self.suggestion:
            parts.append(f"— suggestion: {self.suggestion}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Layer 3: body tokenizer
# ---------------------------------------------------------------------------


class TokenKind(Enum):
    BARE = "bare"
    QUOTED = "quoted"
    BRACED = "braced"
    PARENED = "parened"
    KEY_VALUE = "key_value"
    EQUALS = "equals"
    COMMENT_TRAIL = "comment_trail"


@dataclass(frozen=True)
class Token:
    """One classified token from a card body.

    ``text`` always carries the original substring exactly as it appeared
    in the source — no normalization. For ``KEY_VALUE`` tokens, ``key``
    and ``value`` decompose ``text`` around the first ``=``; ``value`` may
    itself be a balanced ``{...}`` / ``(...)`` / ``"..."`` group.

    ``body_offset`` is the start position of the token in its source
    body and ``body_length`` is the actual byte span. For atomic
    tokens (BARE/QUOTED/BRACED/PARENED) ``body_length == len(text)``.
    For ``KEY_VALUE`` ``text`` is a canonical ``key=value`` rendering
    (no surrounding whitespace) but ``body_length`` covers the
    original span including any whitespace around ``=`` — setters use
    the body span for format-preserving replacement.
    """

    kind: TokenKind
    text: str
    key: str | None = None
    value: str | None = None
    body_offset: int = -1
    body_length: int = -1

    @property
    def body_end(self) -> int:
        """End offset in body (exclusive). ``body_offset + body_length``."""
        return self.body_offset + self.body_length


# Characters that terminate a BARE token outside any nesting context.
_BARE_TERMINATORS: frozenset[str] = frozenset(" \t\r\n={}(),\"';$")


def tokenize_body(body: str) -> list[Token]:
    """Tokenize a merged card body into classified tokens.

    Single-pass hand-rolled state machine. State is implicit in the
    recursive structure: balanced ``{...}`` and ``(...)`` are consumed
    as one atom, quoted strings are consumed to their closing quote.
    ``KEY_VALUE`` is recognized in a post-pass that merges
    ``BARE/QUOTED  EQUALS  ATOM`` triples.

    Raises ``SpiceLexError`` on unbalanced delimiters. Empty / whitespace
    bodies return an empty list.
    """
    atoms = list(_iter_atoms(body))
    return list(_merge_key_values(atoms))


# Internal atom alias for the equals sentinel. Passed through to callers
# as ``TokenKind.EQUALS`` when not part of a ``KEY_VALUE`` triple — that
# matters for ``.MEAS WHEN expr=value`` clauses where ``=`` is a
# comparison, not an assignment.
_EQUALS = TokenKind.EQUALS.value


@dataclass(frozen=True)
class _Atom:
    kind: str  # TokenKind value or _EQUALS
    text: str
    offset: int  # start position in the body string


def _iter_atoms(body: str) -> Iterator[_Atom]:
    """Walk ``body`` once, emitting atomic tokens (incl. EQUALS sentinel)."""
    i = 0
    n = len(body)
    while i < n:
        c = body[i]

        # Whitespace and comma are token boundaries with no atom emitted.
        # Comma at top level is rare in SPICE bodies but harmless to skip.
        if c.isspace() or c == ",":
            i += 1
            continue

        # Comment trail consumes the rest of the body. Layer 1 strips
        # trailing comments before they reach us, but tokenize_body is
        # also called directly on raw bodies (e.g. ad-hoc validation).
        if c in (";", "$"):
            yield _Atom(TokenKind.COMMENT_TRAIL.value, body[i:], i)
            return

        if c == '"':
            end = body.find('"', i + 1)
            if end < 0:
                raise SpiceLexError(
                    SpiceLexErrorCategory.UNTERMINATED_QUOTE,
                    "unterminated quoted string",
                    position=i,
                    body=body,
                    suggestion='add a closing " after the opening quote',
                )
            yield _Atom(TokenKind.QUOTED.value, body[i : end + 1], i)
            i = end + 1
            continue

        # Single-quoted expressions (ngspice numparam: rth='(expr)').
        # Treat like double-quoted strings — consume to the closing quote.
        if c == "'":
            end = body.find("'", i + 1)
            if end < 0:
                raise SpiceLexError(
                    SpiceLexErrorCategory.UNTERMINATED_QUOTE,
                    "unterminated single-quoted string",
                    position=i,
                    body=body,
                    suggestion="add a closing ' after the opening quote",
                )
            yield _Atom(TokenKind.QUOTED.value, body[i : end + 1], i)
            i = end + 1
            continue

        if c == "{":
            end = _scan_balanced(body, i, "{", "}")
            yield _Atom(TokenKind.BRACED.value, body[i : end + 1], i)
            i = end + 1
            continue

        if c == "(":
            end = _scan_balanced(body, i, "(", ")")
            yield _Atom(TokenKind.PARENED.value, body[i : end + 1], i)
            i = end + 1
            continue

        if c == "=":
            yield _Atom(_EQUALS, "=", i)
            i += 1
            continue

        # Stray closer at top level — unbalanced.
        if c in (")", "}"):
            raise SpiceLexError(
                SpiceLexErrorCategory.UNEXPECTED_CHAR,
                f"unexpected {c!r} (no matching opener)",
                position=i,
                body=body,
                suggestion=f"check for a missing opening "
                f"{'(' if c == ')' else '{'} earlier in the body",
            )

        # BARE token: read until a terminator.
        start = i
        while i < n and body[i] not in _BARE_TERMINATORS:
            i += 1
        yield _Atom(TokenKind.BARE.value, body[start:i], start)


def _scan_balanced(body: str, start: int, opener: str, closer: str) -> int:
    """Return the index of the matching ``closer`` for ``body[start] == opener``.

    ``{`` and ``(`` nest independently inside their own kind; quoted
    strings inside them are skipped as opaque. Raises ``SpiceLexError``
    on unbalanced input.
    """
    assert body[start] == opener
    depth = 1
    i = start + 1
    n = len(body)
    while i < n:
        c = body[i]
        if c == '"':
            end = body.find('"', i + 1)
            if end < 0:
                raise SpiceLexError(
                    SpiceLexErrorCategory.UNTERMINATED_QUOTE,
                    f"unterminated quoted string inside {opener}...{closer}",
                    position=start,
                    body=body,
                )
            i = end + 1
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    cat = (
        SpiceLexErrorCategory.UNBALANCED_BRACE
        if opener == "{"
        else SpiceLexErrorCategory.UNBALANCED_PAREN
    )
    raise SpiceLexError(
        cat,
        f"unterminated {opener}...{closer}",
        position=start,
        body=body,
        suggestion=f"check for a missing {closer} matching the {opener}",
    )


def _merge_key_values(atoms: Sequence[_Atom]) -> Iterator[Token]:
    """Merge ``BARE/QUOTED  EQUALS  ATOM`` triples into ``KEY_VALUE`` tokens.

    Atoms outside such a triple pass through as their original kind —
    including standalone ``EQUALS``, which appears when ``=`` is used
    as a comparison operator (``.MEAS WHEN mag(V(out))=0.7``) rather
    than a key-value assignment.
    """
    n = len(atoms)
    i = 0
    while i < n:
        a = atoms[i]
        # KEY=VALUE merge: BARE/QUOTED followed by EQUALS followed by
        # any value-bearing atom.
        if (
            a.kind in (TokenKind.BARE.value, TokenKind.QUOTED.value)
            and i + 2 < n
            and atoms[i + 1].kind == _EQUALS
            and atoms[i + 2].kind
            in {
                TokenKind.BARE.value,
                TokenKind.QUOTED.value,
                TokenKind.BRACED.value,
                TokenKind.PARENED.value,
            }
        ):
            v = atoms[i + 2]
            # The body span runs from the start of the key to the end
            # of the value, so any whitespace around ``=`` (``KP = 100u``)
            # is included. ``text`` is the canonical no-whitespace form.
            body_length = (v.offset + len(v.text)) - a.offset
            yield Token(
                kind=TokenKind.KEY_VALUE,
                text=f"{a.text}={v.text}",
                key=a.text,
                value=v.text,
                body_offset=a.offset,
                body_length=body_length,
            )
            i += 3
            continue
        # Plain atom passes through (EQUALS included — see docstring).
        yield Token(
            kind=TokenKind(a.kind),
            text=a.text,
            body_offset=a.offset,
            body_length=len(a.text),
        )
        i += 1


# ---------------------------------------------------------------------------
# Layer 1: line-level lexer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodySegment:
    """One contiguous slice of ``body`` mapped back to a raw line.

    Built by ``lex`` per card. For a single-line card there is exactly
    one segment covering the entire body. For a continuation card,
    one segment per raw line — ``raw_col_start`` accounts for the
    ``+`` continuation marker (which is replaced with a space in
    ``body`` so tokenizer offsets are 1:1 with raw column offsets).
    """

    body_start: int
    body_end: int  # exclusive
    raw_line_idx: int
    raw_col_start: int


@dataclass
class SpiceCard:
    """One logical SPICE card.

    ``raw_lines`` preserves the original source lines verbatim — emit
    copies them when the card is clean. ``body`` is the merged,
    comment-stripped string for parsers; for ``kind="comment"`` and
    ``kind="blank"``, ``body`` is the empty string.

    ``name`` is a kind-dependent fast-lookup field, set at lex time so
    iteration helpers (``find_model``, ``iter_by_kind``) don't have to
    re-parse every card. The exact meaning depends on ``kind``:

    - ``"model"`` — model name (``.MODEL <name> ...``)
    - ``"param"`` — first param name (``.PARAM <name>=...``)
    - ``"instance"`` — instance ref (``Rxxx``, ``M1``, ...)
    - ``"subckt"`` — subcircuit name (``.SUBCKT <name> ...``)
    - ``"ends"`` — matching subckt name on ``.ENDS [name]``, else ``None``
    - ``"meas"`` — measurement label
    - everything else — ``None``

    For type-safe access prefer the typed properties below
    (``model_name``, ``instance_ref``, etc.) which return ``None`` when
    the card kind doesn't match.

    ``body_layout`` maps body offsets back to ``raw_lines`` positions
    so setters can do format-preserving in-place replacement via
    ``replace_span``. For multi-line continuation cards, ``body_layout``
    has one ``BodySegment`` per raw line.

    ``scope`` is the ``.SUBCKT`` nesting tuple — ``()`` is top level,
    ``("OUTER",)`` is inside ``.SUBCKT OUTER``, ``("OUTER", "INNER")``
    inside a nested ``.SUBCKT INNER`` within ``OUTER``.

    ``trailing`` is set on cards that appear after a top-level ``.END``;
    LTspice ignores them but we preserve them for round-trip.

    ``dirty`` is flipped by typed-view setters (in ``spice_lex_views``)
    when fields change. ``emit`` re-renders dirty cards; clean cards
    copy ``raw_lines`` verbatim.
    """

    kind: CardKind
    raw_lines: list[str]
    body: str
    line_start: int
    name: str | None = None
    scope: tuple[str, ...] = ()
    trailing: bool = False
    dirty: bool = False
    body_layout: list[BodySegment] = field(default_factory=list)

    # ---- typed name accessors -------------------------------------------------
    @property
    def model_name(self) -> str | None:
        """Model name if this is a ``.MODEL`` card, else ``None``."""
        return self.name if self.kind == "model" else None

    @property
    def param_name(self) -> str | None:
        """Param name if this is a ``.PARAM`` card, else ``None``."""
        return self.name if self.kind == "param" else None

    @property
    def instance_ref(self) -> str | None:
        """Instance reference (``R1``, ``M1``, ...) for instance cards, else ``None``."""
        return self.name if self.kind == "instance" else None

    @property
    def subckt_name(self) -> str | None:
        """Subcircuit name for ``.SUBCKT`` openers, else ``None``."""
        return self.name if self.kind == "subckt" else None

    @property
    def meas_label(self) -> str | None:
        """Measurement label for ``.MEAS`` cards, else ``None``."""
        return self.name if self.kind == "meas" else None

    def replace_span(self, body_start: int, body_end: int, new_text: str) -> None:
        """Replace ``body[body_start:body_end]`` with ``new_text`` atomically.

        Format-preserving when the span sits within a single
        ``BodySegment`` — only the affected slice of one raw line is
        rewritten; other raw lines stay byte-identical.

        Raises ``ValueError`` for spans crossing segment boundaries
        (caller falls back to ``replace_body`` for the canonical
        rerender). Empty spans are allowed (``body_end == body_start``,
        meaning insertion at that position).
        """
        if body_start < 0 or body_end > len(self.body) or body_start > body_end:
            raise SpiceLexError(
                SpiceLexErrorCategory.INVALID_RANGE,
                f"replace_span: invalid range [{body_start}, {body_end}) "
                f"for body of length {len(self.body)}",
            )
        seg = _segment_for_offset(self.body_layout, body_start)
        if seg is None:
            raise SpiceLexError(
                SpiceLexErrorCategory.INVALID_RANGE,
                f"replace_span: offset {body_start} has no covering segment",
            )
        # End offset must fall within the same segment (inclusive of segment end).
        if body_end > seg.body_end:
            raise SpiceLexError(
                SpiceLexErrorCategory.INVALID_RANGE,
                f"replace_span: range [{body_start}, {body_end}) crosses segment boundary "
                f"at {seg.body_end}",
                suggestion="use replace_body for multi-segment edits",
            )
        # Map body offsets to raw column offsets.
        col_start = seg.raw_col_start + (body_start - seg.body_start)
        col_end = seg.raw_col_start + (body_end - seg.body_start)
        line = self.raw_lines[seg.raw_line_idx]
        self.raw_lines[seg.raw_line_idx] = line[:col_start] + new_text + line[col_end:]
        # Update body and shift downstream segment offsets by the length delta.
        delta = len(new_text) - (body_end - body_start)
        self.body = self.body[:body_start] + new_text + self.body[body_end:]
        self.body_layout = _shift_segments_after(self.body_layout, seg, body_end, delta)
        self.dirty = True

    def replace_body(self, new_body: str) -> None:
        """Canonical-form rerender. Collapses to a single line.

        Used when a setter's change can't be expressed as a span
        replacement — typically when appending or inserting new tokens.
        Preserves the trailing line ending of the original last raw
        line. After this call, ``body_layout`` holds a single segment.
        """
        eol = _trailing_newline_str(self.raw_lines)
        self.body = new_body
        self.raw_lines = [new_body + eol]
        self.body_layout = [
            BodySegment(
                body_start=0,
                body_end=len(new_body),
                raw_line_idx=0,
                raw_col_start=0,
            )
        ]
        self.dirty = True


def _segment_for_offset(segments: list[BodySegment], offset: int) -> BodySegment | None:
    """Return the segment containing ``offset``, or ``None`` if out of range."""
    for seg in segments:
        if seg.body_start <= offset < seg.body_end or (
            offset == seg.body_end and offset == seg.body_start
        ):
            return seg
    # Allow ``offset == body_end`` of last segment for insertion at end.
    if segments and offset == segments[-1].body_end:
        return segments[-1]
    return None


def _shift_segments_after(
    segments: list[BodySegment],
    target: BodySegment,
    body_end: int,
    delta: int,
) -> list[BodySegment]:
    """Return new segment list reflecting an edit of ``[body_start, body_end)``.

    The target segment grows/shrinks by ``delta``; segments after it
    shift by ``delta``.
    """
    new_segments: list[BodySegment] = []
    for seg in segments:
        if seg is target:
            new_segments.append(
                BodySegment(
                    body_start=seg.body_start,
                    body_end=seg.body_end + delta,
                    raw_line_idx=seg.raw_line_idx,
                    raw_col_start=seg.raw_col_start,
                )
            )
        elif seg.body_start >= body_end:
            new_segments.append(
                BodySegment(
                    body_start=seg.body_start + delta,
                    body_end=seg.body_end + delta,
                    raw_line_idx=seg.raw_line_idx,
                    raw_col_start=seg.raw_col_start,
                )
            )
        else:
            new_segments.append(seg)
    return new_segments


def _trailing_newline_str(raw_lines: list[str]) -> str:
    """Return the line ending of the last raw line ('', '\\n', '\\r\\n')."""
    if not raw_lines:
        return ""
    last = raw_lines[-1]
    if last.endswith("\r\n"):
        return "\r\n"
    if last.endswith("\n"):
        return "\n"
    return ""


@dataclass
class LexResult:
    """Output of ``lex``: cards plus any warnings from malformed input."""

    cards: list[SpiceCard]
    warnings: list[str] = field(default_factory=list)


# Directive recognition. Order matters: ``.ends`` must be checked before
# ``.end`` since the former is a prefix of the latter.
def _classify_directive(head: str) -> CardKind:
    """Classify a card by its first whitespace-delimited token (lowercased).

    ``head`` is ``.something`` already known to start with ``.``.
    """
    h = head.lower()
    if h == ".model":
        return "model"
    if h == ".param":
        return "param"
    if h == ".subckt":
        return "subckt"
    if h == ".ends":
        return "ends"
    if h == ".end":
        return "end"
    if h in (".meas", ".measure"):
        return "meas"
    return "directive"


def _is_continuation(line: str) -> bool:
    """Continuation lines start with ``+`` (after optional whitespace)."""
    s = line.lstrip()
    return s.startswith("+")


def _build_body_with_layout(
    raw_group: list[str],
) -> tuple[str, list[BodySegment]]:
    """Construct the merged body and the body→raw layout map.

    Each raw line contributes exactly one ``BodySegment``. Whitespace
    is preserved verbatim per line so a body offset within a segment
    maps 1:1 to a raw-line column. The ``+`` continuation marker is
    replaced with a space in the body so the tokenizer doesn't see it;
    that space lives at the same column where the ``+`` sat in the
    raw line, so any token positioned at or after the ``+`` in the
    raw line maps correctly to body offsets.
    """
    parts: list[str] = []
    segments: list[BodySegment] = []
    body_offset = 0
    for j, raw_line in enumerate(raw_group):
        # Strip line ending for body purposes. Raw line itself stays untouched.
        if raw_line.endswith("\r\n"):
            line_text = raw_line[:-2]
        elif raw_line.endswith("\n"):
            line_text = raw_line[:-1]
        else:
            line_text = raw_line
        if j == 0:
            piece = line_text
        else:
            # Continuation: replace the leading ``+`` with a space at
            # the same column. Find the ``+`` position (after any
            # leading whitespace).
            stripped = line_text.lstrip()
            assert stripped.startswith("+")
            plus_col = len(line_text) - len(stripped)
            piece = line_text[:plus_col] + " " + line_text[plus_col + 1 :]
        parts.append(piece)
        segments.append(
            BodySegment(
                body_start=body_offset,
                body_end=body_offset + len(piece),
                raw_line_idx=j,
                raw_col_start=0,
            )
        )
        body_offset += len(piece)
    return "".join(parts), segments


def _truncate_segments(segments: list[BodySegment], new_body_len: int) -> list[BodySegment]:
    """Drop or shorten segments past ``new_body_len`` (used by inline-comment strip)."""
    out: list[BodySegment] = []
    for seg in segments:
        if seg.body_start >= new_body_len:
            break
        if seg.body_end > new_body_len:
            out.append(
                BodySegment(
                    body_start=seg.body_start,
                    body_end=new_body_len,
                    raw_line_idx=seg.raw_line_idx,
                    raw_col_start=seg.raw_col_start,
                )
            )
        else:
            out.append(seg)
    return out


def _strip_inline_comment(merged: str) -> str:
    """Strip trailing ``;`` / ``$`` inline comment from a merged body.

    Respects quoted strings and balanced ``{}`` / ``()`` so a comment
    marker inside an expression isn't mistaken for the start of a
    comment. Returns ``merged`` unchanged if no comment is present.
    """
    in_quote = False
    brace = 0
    paren = 0
    for i, c in enumerate(merged):
        if in_quote:
            if c == '"':
                in_quote = False
            continue
        if c == '"':
            in_quote = True
            continue
        if c == "{":
            brace += 1
            continue
        if c == "}":
            brace = max(0, brace - 1)
            continue
        if c == "(":
            paren += 1
            continue
        if c == ")":
            paren = max(0, paren - 1)
            continue
        if c in (";", "$") and brace == 0 and paren == 0:
            return merged[:i].rstrip()
    return merged


def _classify_line(line: str) -> CardKind:
    """Classify a single non-continuation line by its leading character.

    Continuation lines must be merged before classification.
    """
    s = line.lstrip()
    if not s:
        return "blank"
    c = s[0]
    if c == "*":
        return "comment"
    if c == ".":
        # Take the first whitespace-delimited token, lowercased.
        head = s.split(None, 1)[0]
        return _classify_directive(head)
    # Element instance: any letter prefix is an element. Digits or
    # punctuation (other than ``.`` and ``*`` already handled) are
    # malformed but we treat them as raw instance for round-trip.
    return "instance"


def _extract_subckt_name(body: str) -> str | None:
    """Pull the subcircuit name from a ``.SUBCKT NAME ...`` body."""
    parts = body.split(None, 2)
    if len(parts) < 2:
        return None
    return parts[1]


def _extract_ends_name(body: str) -> str | None:
    """Pull the optional name from a ``.ENDS [NAME]`` body."""
    parts = body.split(None, 2)
    if len(parts) < 2:
        return None
    return parts[1]


def _extract_model_name(body: str) -> str | None:
    """Pull the model name from a ``.MODEL NAME TYPE(...)`` body."""
    parts = body.split(None, 3)
    if len(parts) < 2:
        return None
    return parts[1]


def _extract_param_name(body: str) -> str | None:
    """Pull the param name from a ``.PARAM NAME=VALUE`` body.

    Handles whitespace around the ``=`` sign. Returns ``None`` for
    multi-param ``.PARAM`` lines (rare; handled at typed-view layer).
    """
    rest = body.split(None, 1)
    if len(rest) < 2:
        return None
    tail = rest[1]
    eq = tail.find("=")
    if eq < 0:
        return tail.split(None, 1)[0]
    return tail[:eq].strip().split(None, 1)[0] if tail[:eq].strip() else None


def _extract_instance_ref(body: str) -> str | None:
    """Pull the reference name from an instance line (first token)."""
    parts = body.split(None, 1)
    return parts[0] if parts else None


def _extract_meas_name(body: str) -> str | None:
    """Pull the measurement label from ``.MEAS [analysis] NAME ...``.

    LTspice accepts ``.MEAS NAME ...`` (analysis kind omitted, defaults
    to the active analysis) or ``.MEAS TRAN NAME ...``. We return the
    label in either case.
    """
    parts = body.split(None, 4)
    if len(parts) < 2:
        return None
    second = parts[1].lower()
    if second in ("tran", "ac", "dc", "op", "noise") and len(parts) >= 3:
        return parts[2]
    return parts[1]


_NAME_EXTRACTORS = {
    "model": _extract_model_name,
    "param": _extract_param_name,
    "instance": _extract_instance_ref,
    "subckt": _extract_subckt_name,
    "ends": _extract_ends_name,
    "meas": _extract_meas_name,
}


def lex(netlist_text: str) -> LexResult:
    """Tokenize ``netlist_text`` into a flat list of ``SpiceCard``.

    Round-trip invariant: ``emit(lex(text).cards) == text`` for
    well-formed netlists. Malformed input still round-trips
    byte-faithful; warnings are surfaced via ``LexResult.warnings``.
    """
    # ``splitlines(keepends=True)`` preserves CRLF/LF for round-trip.
    lines = netlist_text.splitlines(keepends=True)
    cards: list[SpiceCard] = []
    warnings: list[str] = []

    scope: list[str] = []
    seen_top_end = False

    i = 0
    n = len(lines)
    while i < n:
        raw_line = lines[i]
        line_start = i + 1  # 1-based

        # Continuation lines without a preceding card become a "raw"
        # comment-style card so they round-trip without dropping.
        if _is_continuation(raw_line):
            warnings.append(
                f"line {line_start}: continuation '+' with no preceding card; "
                "preserving as comment"
            )
            cards.append(
                SpiceCard(
                    kind="comment",
                    raw_lines=[raw_line],
                    body="",
                    line_start=line_start,
                    scope=tuple(scope),
                    trailing=seen_top_end,
                )
            )
            i += 1
            continue

        kind = _classify_line(raw_line)

        # Comment / blank cards never have continuations; emit and move on.
        if kind in ("comment", "blank"):
            cards.append(
                SpiceCard(
                    kind=kind,
                    raw_lines=[raw_line],
                    body="",
                    line_start=line_start,
                    scope=tuple(scope),
                    trailing=seen_top_end,
                )
            )
            i += 1
            continue

        # Collect this line plus any following continuation lines.
        raw_group = [raw_line]
        i += 1
        while i < n and _is_continuation(lines[i]):
            raw_group.append(lines[i])
            i += 1

        # Build the merged body and a body→raw layout map. Whitespace
        # is preserved verbatim per line so token offsets in body are
        # 1:1 with raw column offsets. ``+`` continuation markers are
        # replaced with a single space so tokenizer sees no leftover
        # syntax (the ``+`` itself never overlaps any token because
        # tokens never start at whitespace).
        body_with_comment, segments = _build_body_with_layout(raw_group)
        body = _strip_inline_comment(body_with_comment)
        # If comment stripping shortened the body, truncate the last
        # affected segment (keep the layout consistent with body).
        if len(body) < len(body_with_comment):
            segments = _truncate_segments(segments, len(body))

        name_extractor = _NAME_EXTRACTORS.get(kind)
        name = name_extractor(body) if name_extractor else None

        # Scope on the card is the *enclosing* (parent) scope for both
        # opener and closer — symmetric. Inner cards carry the extended
        # scope. This makes ``iter_body(scope=("FOO",))`` yield FOO's
        # body without its boundaries.
        if kind == "ends":
            if not scope:
                warnings.append(
                    f"line {line_start}: .ENDS with no matching .SUBCKT; treating as top-level"
                )
                card_scope: tuple[str, ...] = ()
            else:
                if name and name != scope[-1]:
                    warnings.append(
                        f"line {line_start}: .ENDS {name!r} does not match "
                        f"opener {scope[-1]!r}; closing the open scope anyway"
                    )
                scope.pop()
                card_scope = tuple(scope)
        else:
            card_scope = tuple(scope)

        if kind == "end" and not scope:
            seen_top_end = True

        card = SpiceCard(
            kind=kind,
            raw_lines=raw_group,
            body=body,
            line_start=line_start,
            name=name,
            scope=card_scope,
            trailing=seen_top_end and kind != "end",
            body_layout=segments,
        )
        cards.append(card)

        if kind == "subckt":
            if name:
                scope.append(name)
            else:
                warnings.append(f"line {line_start}: .SUBCKT with no name")
                scope.append("<anon>")

    if scope:
        warnings.append(f"EOF with unclosed .SUBCKT scope(s): {scope}")

    return LexResult(cards=cards, warnings=warnings)


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------


def emit(cards: Sequence[SpiceCard]) -> str:
    """Re-render ``cards`` to text.

    Clean cards (``dirty=False``) copy ``raw_lines`` verbatim — bytes,
    whitespace, line endings preserved. Dirty cards re-render from
    their typed fields; the typed-view layer is responsible for
    producing a sensible ``raw_lines`` replacement when it flips
    ``dirty=True``.

    Round-trip contract: ``emit(lex(text).cards) == text`` when no
    setters have run.
    """
    out: list[str] = []
    for card in cards:
        out.extend(card.raw_lines)
    return "".join(out)


# ---------------------------------------------------------------------------
# Interop helpers for spicelib and file-based workflows
# ---------------------------------------------------------------------------


def cards_from_path(path: object) -> LexResult:
    """Read a netlist file and lex it.

    ``path`` is anything ``pathlib.Path`` accepts. Encoding detection
    (BOM + UTF-16-without-BOM heuristic) is shared with ``library_parser``
    via ``lib.encoding`` so LTspice's stock UTF-16 libraries round-trip.
    """
    from pathlib import Path

    from ltspice_mcp.lib.encoding import read_spice_text

    return lex(read_spice_text(Path(path)))  # type: ignore[arg-type]


def write_cards(cards: Sequence[SpiceCard], path: object) -> None:
    """Render ``cards`` and write to ``path``. Encoding: UTF-8."""
    from pathlib import Path

    p = Path(path)  # type: ignore[arg-type]
    p.write_text(emit(cards), encoding="utf-8")


def cards_from_editor(editor: object) -> LexResult:
    """Extract a netlist from a spicelib editor and lex it.

    Works with ``SpiceEditor`` and ``AscEditor`` instances. Falls back
    to ``str(editor)`` when no ``get_netlist_text``/``netlist`` access
    is available — spicelib's internals vary by version.
    """
    # Try the documented attribute paths in order of preference.
    text: str | None = None
    if hasattr(editor, "get_netlist_text"):
        text = editor.get_netlist_text()  # type: ignore[attr-defined]
    elif hasattr(editor, "netlist"):
        netlist = editor.netlist  # type: ignore[attr-defined]
        if isinstance(netlist, list):
            text = "".join(netlist)
        elif isinstance(netlist, str):
            text = netlist
    if text is None:
        # Last resort: round-trip via temp file.
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False, encoding="utf-8") as f:
            tmp = Path(f.name)
        try:
            editor.write_netlist(tmp)  # type: ignore[attr-defined]
            return cards_from_path(tmp)
        finally:
            tmp.unlink(missing_ok=True)
    return lex(text)


def apply_to_editor(cards: Sequence[SpiceCard], editor: object) -> None:
    """Write emitted ``cards`` back through a spicelib editor.

    Resolution order:

    1. ``editor.reset_netlist(path)`` — spicelib's preferred reload API.
       Text is written to a temp file and the editor reloads it.
    2. ``editor.netlist = lines`` — direct assignment to spicelib's
       internal line list. ``netlist`` must be a settable attribute.
    3. Otherwise raise ``TypeError``. ``editor.write_netlist`` is *not*
       a fallback — it writes the editor's current contents to the
       given path, which would silently no-op for our needs.
    """
    import tempfile
    from pathlib import Path

    text = emit(cards)

    if hasattr(editor, "reset_netlist"):
        with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False, encoding="utf-8") as f:
            f.write(text)
            tmp = Path(f.name)
        try:
            editor.reset_netlist(tmp)  # type: ignore[attr-defined]
        finally:
            tmp.unlink(missing_ok=True)
        return

    if hasattr(editor, "netlist"):
        try:
            editor.netlist = text.splitlines(keepends=True)  # type: ignore[attr-defined]
        except AttributeError as e:
            raise TypeError(
                f"apply_to_editor: {type(editor).__name__}.netlist is not "
                f"settable; cannot push card list back into the editor"
            ) from e
        return

    raise TypeError(
        f"apply_to_editor: {type(editor).__name__} has neither "
        f"reset_netlist() nor a settable netlist attribute; "
        f"cannot push card list back into the editor"
    )


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------


def find_matching_ends(cards: Sequence[SpiceCard], opener_idx: int) -> int | None:
    """Return the index of the ``.ENDS`` matching the ``.SUBCKT`` at ``opener_idx``.

    Walks forward at the opener's parent scope, tracking nested
    ``.SUBCKT`` / ``.ENDS`` pairs. Returns ``None`` if no matching
    closer is found before EOF.
    """
    opener = cards[opener_idx]
    parent_scope = opener.scope
    depth = 0
    for i in range(opener_idx + 1, len(cards)):
        c = cards[i]
        if c.scope != parent_scope:
            continue
        if c.kind == "subckt":
            depth += 1
        elif c.kind == "ends":
            if depth == 0:
                return i
            depth -= 1
    return None


def iter_by_kind(
    cards: Sequence[SpiceCard],
    kind: CardKind,
    scope: tuple[str, ...] | None = None,
) -> Iterator[SpiceCard]:
    """Iterate cards of ``kind``, optionally filtered to an exact scope.

    ``scope=None`` returns cards in any scope. ``scope=()`` is top level.
    """
    for card in cards:
        if card.kind != kind:
            continue
        if scope is not None and card.scope != scope:
            continue
        yield card


def iter_body(
    cards: Sequence[SpiceCard],
    scope: tuple[str, ...],
    *,
    recursive: bool = True,
) -> Iterator[SpiceCard]:
    """Iterate cards inside a given ``.SUBCKT`` scope.

    With ``recursive=True`` (default), yields every card whose scope is
    ``scope`` or a strict extension (i.e. nested subckts' bodies are
    included). With ``recursive=False``, yields only direct children
    (cards with exactly ``scope``). Boundary cards (opener and closer
    of ``scope`` itself) live at the parent scope and are never yielded.
    """
    for card in cards:
        if recursive:
            if len(card.scope) >= len(scope) and card.scope[: len(scope)] == scope:
                yield card
        else:
            if card.scope == scope:
                yield card
