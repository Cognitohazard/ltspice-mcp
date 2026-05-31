# `lib/spice_lex` — SPICE netlist parser

The shared tokenizer + typed-card library underneath every netlist-touching
helper in the codebase. Replaces hand-rolled regex passes with a single
parser that handles SPICE corner cases (quoted tokens, balanced
brace expressions, scoped `.SUBCKT` blocks, inline `;`/`$` comments) once.

This document describes the architecture as built. For the migration
history that produced it, see git tags around 2026-05-03.

## Why a parser

Every site that read or rewrote a SPICE netlist used to roll its own
regex. Each one handled a fraction of SPICE syntax with different
corner-case behaviour:

| Caller | What it edits | Failure modes |
|-|-|-|
| `lib/montecarlo.py` | `.MODEL` / instance / `.PARAM` rewriting (MC) | balanced expressions, quoted tokens, scope, comments |
| `lib/library_parser.py` | `.MODEL` / `.SUBCKT` indexing for `find_model` | continuation merge, comments, nested subcircuits |
| `tools/circuit.py:_apply_component_value` | Splits `"NMOS1 W=10u L=1u"` into model + params | quoted strings, multi-token values, `=` inside braces |
| `lib/spice_validator.py` | Layer-A `.MEAS` expression checks | substring matching of function calls, no AST |

Every caller failed differently on the same input class. The fix wasn't
"write better regex"; it was "have one parser that handles SPICE
properly, and let everyone use it." Same architectural shift compilers
made when they replaced hand-rolled per-pass regexes with a shared lexer
+ AST. Every adversarial fixture filed against one consumer now hardens
the parser for all consumers.

## Terminology

**Card** — SPICE's name for one logical statement in a netlist. From the
punched-card era. A card is one element instance (`R1 a b 1k`), one
directive (`.MODEL …`, `.TRAN …`, `.SUBCKT …`, `.ENDS`, `.END`,
`.PARAM …`), one comment line (`* …`), or one blank line. Continuation
lines (`+ …`) extend the previous card — physically separate lines, one
logical card. `SpiceCard.raw_lines: list[str]` holds those physical
lines so a multi-line `.MODEL` round-trips byte-for-byte.

## Scope

In:

- `.MODEL <name> <type>(...)` — process variation, library indexing.
- `.PARAM <name>=<value>` — `.PARAM` perturbation, expression chasing.
- Element instance lines — value editing, model rewriting, W/L
  extraction. Covered prefixes:
  - `Mxxx` / `Qxxx` / `Jxxx` — MOSFET / BJT / JFET, model-after-nodes shape.
  - `Rxxx` / `Cxxx` / `Lxxx` — passives, value-after-nodes shape.
  - `Vxxx` / `Ixxx` — independent sources.
  - `Bxxx` / `Exxx` / `Gxxx` — behavioural / VCVS / VCCS, value carried
    as a `V=` or `I=` `KEY_VALUE` token (typically a `{...}` expression).
  - `Fxxx` / `Hxxx` — current-controlled sources, control by source name.
  - `Xxxx` — subcircuit calls, variable port count.
  - `Kxxx` — mutual inductance, references only (no nodes).
- `.SUBCKT <name> ... .ENDS` — scope tracking for the above.
- `.MEAS` — body tokenized into classified tokens via Layer 3 so
  validation operates on tokens, not substrings.
- `.INCLUDE`, `.LIB`, `.OPTIONS`, `.AC`/`.TRAN`/`.DC` — read as opaque
  directives (we don't dissect their bodies).
- Comments, blank lines — preserved verbatim for byte-faithful round-trip.

Out:

- Expression evaluation. `{2*kp_n}` is a string token; we don't compute it.
- Cross-simulator dialect handling (HSPICE-only syntax, Spectre
  `simulator lang=spice` blocks). LTspice/ngspice grammar only.
- Whole-program transformation passes. Targeted edits, not netlist-wide
  rewriting.
- Semantic validation (does this `.MODEL` actually match a simulator?).
  That's the simulator's job.

Coexistence with spicelib's `REPLACE_REGEXS`: spicelib's `SpiceEditor`
still owns on-disk `.cir` writes for the simulation pipeline. spice_lex
shadows the *lookup* and *edit* paths — every internal caller goes
through `lex` + typed views and emits via spicelib only when the file
needs to land for the simulator. spicelib's element-line regex stays in
its own lane.

## Architecture

Three layers.

### Layer 1: line-level lexer

```python
def lex(netlist_text: str) -> LexResult: ...
```

Walks lines once, classifies by first non-whitespace character, merges
`+`-continuation chains, tracks `.SUBCKT`/`.ENDS` scope. Each `SpiceCard`
keeps:

```python
class SpiceCard:
    kind: Literal["model", "param", "instance", "subckt", "ends",
                  "directive", "comment", "blank", "raw"]
    raw_lines: list[str]      # original source lines (preserved for emit)
    body: str                 # merged & comment-stripped (for parser)
    line_start: int           # 1-based line in the source
    name: str | None          # extracted by per-kind parser layer
    scope: tuple[str, ...]    # subckt nesting; () == top level
    dirty: bool = False       # set by typed-view setters
```

Atomic mutation primitives on `SpiceCard`:

- `replace_span(body_start, body_end, new_text)` — surgical replacement
  of a body slice. Preserves surrounding whitespace and continuation
  structure.
- `replace_body(new_body)` — full body replacement when the whole card
  needs re-rendering.

Both flip the card's `dirty` flag.

### Layer 2: typed views

Ephemeral wrappers re-derived via `from_card()`; they never own state.
Setters mutate the underlying `SpiceCard` directly.

```python
class ModelCard:           # .MODEL <name> <type>(<params>)
    name: str
    type: str              # NMOS, NPN, R, ...
    level: int | None      # parsed from LEVEL=
    params: dict[str, str] # raw value tokens (caller parses)
    def set_param(self, key, value) -> None: ...
    def remove_param(self, key) -> None: ...
    def set_name(self, new_name) -> None: ...

class InstanceLine:        # element instance
    ref: str
    nodes: list[str]
    model: str | None      # for M/Q/J/X
    value: str | None      # for R/C/L
    params: dict[str, str] # W=, L=, m=, etc.
    def set_model(self, name) -> None: ...
    def set_value(self, value) -> None: ...
    def set_param(self, key, value) -> None: ...
    def remove_param(self, key) -> None: ...

class ParamCard:           # .PARAM <name>=<value>
    name: str
    value: str
    def set_value(self, value) -> None: ...

class SubcktCard:          # .SUBCKT opener
    name: str
    ports: list[str]
    param_defaults: dict[str, str]   # PARAMS: W=10u L=1u
    def set_name_local(self, new_name) -> None: ...   # opener only
    def set_ports(self, new_ports) -> None: ...
    def set_param_default(self, key, value) -> None: ...
    def remove_param_default(self, key) -> None: ...

class MeasCard:            # .MEAS — read/validated, rarely rewritten
    function_calls: list[FunctionCall]
    signal_refs: list[str]
    analysis: MeasAnalysis | None    # tran/ac/dc/op
    def set_label(self, new_name) -> None: ...
    def set_analysis(self, kind) -> None: ...
```

`MeasCard` exists so `spice_validator` can ask "does this body contain a
`vdb()` call" or "does it reference a signal not present in the .raw"
without substring matching. Layer 3's classified tokens give it that:
walk `body_tokens`, find `BARE` immediately followed by `PARENED`, that's
a function call.

`ELEMENT_SPECS` registry covers M/Q/J/X/F/H/R/C/L/V/I/B/E/G/K with
context-aware classification (E/G/F/H positional-gain vs `VALUE=` form).

### Layer 3: body tokenizer

After continuation merge, the card body is a single string. A
single-pass hand-rolled state machine emits **classified tokens** — not
raw strings — so Layer 2 views never re-parse character classes.

```python
class TokenKind(Enum):
    BARE          # identifier or numeric literal (R1, NMOS_lvt, 10u, 0.7)
    QUOTED        # "NMOS_lvt" — single token, quotes preserved in raw
    BRACED        # {2*max(kp_n, kp_min)} — single token, balanced
    PARENED       # (VTO=0.7 KP=100u) — single token, balanced
    KEY_VALUE     # W=10u, KP={2*kp_n}, MODEL="NMOS_lvt" — split on first =
    EQUALS        # standalone = (whitespace-around-equals form)
    COMMENT_TRAIL # ;... or $... to end of body, preserved verbatim

@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str          # original substring, no normalization
    key: str | None    # for KEY_VALUE; original case preserved
    value: str | None  # for KEY_VALUE; may itself be BRACED/PARENED/QUOTED
    body_offset: int   # for replace_span surgery
    body_length: int
```

**State machine.** Single forward walk over the body. State is
`(brace_depth, paren_depth, in_quote)`. Transitions:

| Char | Outside | In quote | brace_depth ≥ 1 | paren_depth ≥ 1, brace_depth = 0 |
|-|-|-|-|-|
| `"` | enter quote | exit quote | literal | literal |
| `{` | enter brace, depth=1 | literal | depth++ | enter brace |
| `}` | error (unbalanced) | literal | depth-- | error |
| `(` | enter paren, depth=1 | literal | literal | depth++ |
| `)` | error (unbalanced) | literal | literal | depth-- |
| `=` | token boundary, mark KEY_VALUE | literal | literal | literal |
| `,` | token boundary | literal | literal | token boundary |
| whitespace | token boundary | literal | literal | token boundary |
| `;` `$` | start COMMENT_TRAIL, consume to end | literal | literal | literal |
| other | accumulate | literal | literal | literal |

Rules in plain English:

- `(` inside `{...}` does not open a paren context — braces dominate.
- `;` and `$` start a comment only outside quotes and braces.
- `*` line comments are handled at Layer 1, never reach Layer 3.
- Unbalanced delimiters at body end raise `SpiceLexError` with the
  position; recovery is not attempted at this layer.
- `KEY_VALUE` is recognized when a `=` is seen at depth 0 outside a
  quote; the preceding `BARE` token becomes `key`, the following token
  (which may itself be `BRACED`/`PARENED`/`QUOTED`/`BARE`) becomes
  `value`. Whitespace around `=` is allowed (becomes `EQUALS` + adjacent
  tokens that subsequent passes pair up).

Case: `text`, `key`, `value` all preserve original case. Identifier
matching (model names, refs, param keys) folds to lower at lookup time.

### The model-position rule

The instance-line nodes-vs-model-token problem reduces to one rule once
tokens are classified:

> **Model/value position is the last `BARE`-or-`QUOTED` token before
> any `KEY_VALUE` token** (or the last such token if the line has no
> `KEY_VALUE` tokens). Tokens before that are nodes.

This handles every adversarial case the old heuristic missed:

- `M1 d g s b "NMOS_lvt" W=10u` — the quoted token is correctly
  identified as the model name.
- `X1 a b c MYSUB W=10u` — `MYSUB` is the last `BARE` before the
  `KEY_VALUE`, regardless of how many ports preceded it.
- `M1 d g s NMOS1` — no `KEY_VALUE`, so the last `BARE` is the model.
- `R1 n1 n2 1k TC=0.001` — `1k` is BARE (numeric literal), correctly
  picked as the value.

Per-prefix arity collapses to a node-count *minimum* (sanity check),
not the model-position lookup:

```python
MIN_NODES = {
    "M": 3, "Q": 3, "J": 3,        # MOSFET / BJT / JFET
    "R": 2, "C": 2, "L": 2,        # passives
    "V": 2, "I": 2,                # sources
    "B": 2, "E": 2, "G": 2,        # behavioural / VCVS / VCCS
    "F": 2, "H": 2,                # CCCS / CCVS (control source ref is BARE)
    "X": 1,                        # subckt call: at least one port
    "K": 0,                        # mutual inductance: refs only, no nodes
}
```

`Bxxx` / `Exxx` / `Gxxx` carry their value as a `KEY_VALUE` token like
`V={expression}` or `I={...}`; the expression body is one `BRACED`
token by construction. There is no "model" position — `InstanceLine`
exposes the relevant `KEY_VALUE` (`V=` or `I=`) as `value`.

`Fxxx` / `Hxxx` reference a controlling source by name (a `BARE` token),
then take a gain — same shape as `M`-style "model after nodes."

## Ownership and emit invariants

Canonical store: `lex()` returns a single flat `list[SpiceCard]` in
source order. Typed views are ephemeral wrappers; they never own
children. Two consequences:

- A `.SUBCKT` block contributes one `kind="subckt"` card (opener), one
  `kind="ends"` card (closer), and the inner cards as flat-list entries
  with `scope=(name,)`. Mutating an inner card flips that card's dirty
  flag only — the opener and closer stay byte-identical unless they
  themselves are mutated. `SubcktCard.dirty` propagation is local: it
  marks the opener card, never its scope's children.
- `emit(cards)` walks the flat list once and, per card, copies
  `raw_lines` if clean or re-renders from typed fields if dirty. Order
  and surrounding whitespace are preserved.

Cross-card invariants the parser is responsible for, even though they
span two cards:

- `.SUBCKT NAME ... .ENDS [NAME]`: the optional trailing name on `.ENDS`
  must track the opener. Renames go through `rename_subckt` which marks
  both the opener and the matching ends card dirty in one transactional
  operation. Plain field assignment on the typed view is not enough.
- Nested `.SUBCKT`: scope is a tuple, so `.SUBCKT INNER` inside
  `.SUBCKT OUTER` gets `scope=("OUTER", "INNER")`. Each inner subckt is
  its own pair of subckt + ends cards in the flat list; `iter_body`
  filters by scope prefix and recurses by extending the prefix.

Recovery rules for malformed input — the round-trip-byte-faithful
contract holds even when input is broken:

- `.ENDS` without a matching opener: the closer card is emitted with
  `scope=()` and a warning surfaced via `lex()` metadata. Subsequent
  cards resume at the outermost still-open scope, or `()` if none.
- EOF inside an open `.SUBCKT`: cards return as-is with the unclosed
  scope, plus a warning. `emit()` still round-trips byte-faithful.
- Cards after a top-level `.END`: keep their position and scope but
  carry `trailing=True`. LTspice ignores them; we preserve them rather
  than dropping.

Mutation safety: typed views are short-lived. Holding two views over the
same card and mutating both is undefined — re-derive after each
mutation. The flat list is the only synchronization point. Sequential
setters on the same view are safe: token caches re-shift on each
mutation via `_shift_cached_param_tokens` / `_shift_single_token`.

## Public API surface

```python
# lib/spice_lex.py
def lex(netlist_text: str) -> LexResult: ...
def emit(cards: Sequence[SpiceCard]) -> str: ...
def tokenize_body(body: str) -> list[Token]: ...
def find_matching_ends(cards: Sequence[SpiceCard], opener_idx: int) -> int | None: ...
def iter_by_kind(cards, kind, scope=None) -> Iterator[SpiceCard]: ...
def iter_body(cards, scope, *, recursive=True) -> Iterator[SpiceCard]: ...
def cards_from_path(path) -> LexResult: ...
def write_cards(cards, path) -> None: ...
def cards_from_editor(editor) -> LexResult: ...
def apply_to_editor(cards, editor) -> None: ...

class SpiceCard:                 # canonical store
    def replace_span(self, body_start, body_end, new_text) -> None: ...
    def replace_body(self, new_body) -> None: ...
    @property
    def model_name(self) -> str | None: ...
    @property
    def param_name(self) -> str | None: ...
    @property
    def instance_ref(self) -> str | None: ...
    @property
    def subckt_name(self) -> str | None: ...
    @property
    def meas_label(self) -> str | None: ...

class Token: ...                 # body_offset / body_length / kind / text
class TokenKind(Enum): ...       # BARE / QUOTED / BRACED / PARENED / KEY_VALUE / EQUALS / COMMENT_TRAIL
class BodySegment: ...           # body→raw_lines mapping for replace_span
class SpiceLexError(ValueError): ...           # category, position, body, suggestion
class SpiceLexErrorCategory(Enum): ...

# lib/spice_lex_views.py
class ModelCard:
    @classmethod
    def from_card(cls, card: SpiceCard) -> ModelCard: ...
    def set_param(self, key, value) -> None: ...
    def remove_param(self, key) -> None: ...
    def set_name(self, new_name) -> None: ...

class InstanceLine:
    @classmethod
    def from_card(cls, card: SpiceCard) -> InstanceLine: ...
    def set_model(self, name) -> None: ...
    def set_value(self, value) -> None: ...
    def set_param(self, key, value) -> None: ...
    def remove_param(self, key) -> None: ...

class ParamCard:
    @classmethod
    def from_card(cls, card: SpiceCard) -> ParamCard: ...
    def set_value(self, value) -> None: ...

class SubcktCard:
    @classmethod
    def from_card(cls, card: SpiceCard) -> SubcktCard: ...
    def set_name_local(self, new_name) -> None: ...   # opener only
    def set_ports(self, new_ports) -> None: ...
    def set_param_default(self, key, value) -> None: ...
    def remove_param_default(self, key) -> None: ...

class MeasCard:
    function_calls: list[FunctionCall]
    signal_refs: list[str]
    analysis: MeasAnalysis | None
    @classmethod
    def from_card(cls, card: SpiceCard) -> MeasCard: ...
    def set_label(self, new_name) -> None: ...
    def set_analysis(self, kind) -> None: ...

class ElementSpec: ...           # ELEMENT_SPECS registry: M/Q/J/X/F/H/R/C/L/V/I/B/E/G/K

def iter_models(cards, scope=None) -> Iterator[ModelCard]: ...
def iter_instances(cards, prefix=None) -> Iterator[InstanceLine]: ...
def find_model(cards, name, scope=()) -> ModelCard | None: ...
def instances_by_ref(cards) -> dict[str, SpiceCard]: ...
def body_has_stray_kv_remnant(body: str) -> bool: ...

# lib/spice_lex_ops.py — cross-card transformations
def inject_card_before_end(cards, text, *, scope=()) -> SpiceCard: ...
def rename_subckt(cards, old_name, new_name) -> int: ...
def rename_model(cards, old_name, new_name, *, scope=()) -> int: ...

# lib/encoding.py — shared text decoding
def decode_spice_bytes(raw: bytes) -> str: ...
def detect_utf16_endianness(probe: bytes) -> str | None: ...
def read_spice_text(path: Path) -> str: ...
```

## Future work

Not implemented; bring online when a real consumer needs them. No stubs
in tree — importable `NotImplementedError` placeholders are a lying API
surface.

- **`rename_component(cards, old_ref, new_ref)`** — atomic rename of an
  instance ref across the instance card *and* every `.PARAM` expression
  / `.MEAS` body / behavioural-source body that references it. Sub-token
  surgery: walk `BRACED` token contents and patch the substring while
  preserving balanced delimiters.
- **`inline_subckt(cards, instance_ref)`** — flatten an `Xxxx`
  invocation into its parent scope. Substitute the call's actual nodes
  for the subckt's formal ports throughout the body, generate fresh refs
  to avoid collisions, splice the body cards into the parent scope, drop
  the `Xxxx` card.
- **`extract_subckt(cards, body_indices, new_subckt_name)`** — inverse
  of inline. Wrap a span of cards in a new `.SUBCKT` opener / `.ENDS`
  closer, replace external node references with formal ports, leave a
  single `Xxxx` invocation in place of the original span.
- **`netlist_diff(a, b)`** — structural diff that ignores formatting.
  Compare card-by-card by `(kind, name, body)` after canonical
  normalization; report added / removed / modified cards. Useful as a
  regression-pinning primitive for tests.
- **`ChangeSet`** — atomic-commit primitive. Stage cross-card mutations
  as a list of `(card, replacement_raw_lines)` pairs; commit in one pass
  after validation; roll back via the captured original `raw_lines` if
  any commit step raises. Would replace `rename_subckt`'s
  "validate-then-commit" best-effort with a real transactional contract.
- **Static linters** — every `Mxxx` references a `.MODEL` in scope;
  every `.PARAM` is consumed somewhere; no duplicate refs; no dangling
  `.SUBCKT` calls. Each is a single-pass walk over cards using the typed
  views.

Design notes for when the deferred items land:

- **Cross-card transactions**: `ChangeSet` is the natural container.
  Snapshot raw_lines on entry, mutate via `replace_body` / `replace_span`,
  commit on `__exit__` if no exception otherwise restore. The current
  `rename_subckt` should migrate to `ChangeSet` once it exists.
- **Sub-token surgery** for `rename_component`: the body-tokenizer's
  `BRACED` / `PARENED` tokens are opaque strings. To rewrite `{R1*2}` →
  `{Rnew*2}`, we need a sub-tokenizer for expression bodies. Keep it
  shallow — just identifier-replacement, not full expression parsing.
- **Scope manipulation**: `inline_subckt` and `extract_subckt` rewrite
  `card.scope` tuples for spliced cards. The flat-list-as-canonical-store
  decision keeps this mechanical: detach a slice, rewrite scopes, splice
  into target position.
- **Atomicity guarantees**: today's `rename_subckt` is "validate then
  commit, no rollback". Phase 6 with `ChangeSet` should default to true
  atomicity. Document the contract change in the migration commit.
