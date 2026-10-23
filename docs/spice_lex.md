# `lib/spice_lex` — SPICE netlist parser

`lib/spice_lex` provides the shared tokenizer and the typed card API that
every netlist helper in the codebase uses. It replaces several regex
passes with one parser that handles the SPICE corner cases (quoted
tokens, balanced brace expressions, scoped `.SUBCKT` blocks, inline
`;`/`$` comments) in one place.

This document describes the architecture as built. For the migration
that produced it, see the git history around 2026-05-03.

## Why a parser

Every site that read or rewrote a SPICE netlist used to have its own
regex. Each one handled a fraction of SPICE syntax with different
corner-case behaviour:

| Caller | What it edits | Failure modes |
|-|-|-|
| `lib/montecarlo.py` | `.MODEL` / instance / `.PARAM` rewriting (Monte Carlo) | balanced expressions, quoted tokens, scope, comments |
| `lib/library_parser.py` | `.MODEL` / `.SUBCKT` indexing for `find_model` | continuation merge, comments, nested subcircuits |
| `lib/schematic_ops.py:_apply_component_value` | Splits `"NMOS1 W=10u L=1u"` into model + params | quoted strings, multi-token values, `=` inside braces |
| `lib/spice_validator.py` | `.MEAS` expression checks (blocklist of known-bad patterns) | substring matching of function calls, no AST |

Every caller failed differently on the same kind of input, so the fix was
one parser that handles SPICE properly and is used by every caller,
rather than better regexes at each site. A test fixture added for one
consumer now covers every consumer.

## Terminology

**Card** — SPICE's name for one logical statement in a netlist. The name
comes from punched cards. A card is one element instance (`R1 a b 1k`),
one directive (`.MODEL …`, `.TRAN …`, `.SUBCKT …`, `.ENDS`, `.END`,
`.PARAM …`), one comment line (`* …`), or one blank line. Continuation
lines (`+ …`) extend the previous card, so several physical lines can
form one logical card. `SpiceCard.raw_lines: list[str]` holds those
physical lines so a multi-line `.MODEL` round-trips byte-for-byte.

## Scope

In:

- `.MODEL <name> <type>(...)` — process variation, library indexing.
- `.PARAM <name>=<value>` — `.PARAM` perturbation, following references
  through expressions.
- Element instance lines — value editing, model rewriting, W/L
  extraction. Covered prefixes:
  - `Mxxx` / `Qxxx` / `Jxxx` — MOSFET / BJT / JFET, model-after-nodes shape.
  - `Rxxx` / `Cxxx` / `Lxxx` — passives, value-after-nodes shape.
  - `Vxxx` / `Ixxx` — independent sources.
  - `Bxxx` / `Exxx` / `Gxxx` — behavioural / VCVS / VCCS, value given as
    a `V=` or `I=` `KEY_VALUE` token (typically a `{...}` expression).
  - `Fxxx` / `Hxxx` — current-controlled sources, control by source name.
  - `Xxxx` — subcircuit calls, variable port count.
  - `Kxxx` — mutual inductance, references only (no nodes).
- `.SUBCKT <name> ... .ENDS` — scope tracking for the above.
- `.MEAS` — body tokenized into classified tokens via Layer 3 so
  validation works on tokens instead of substrings.
- `.INCLUDE`, `.LIB`, `.OPTIONS`, `.AC`/`.TRAN`/`.DC` — read as opaque
  directives (their bodies are not parsed).
- Comments, blank lines — preserved verbatim for byte-faithful round-trip.

Out:

- Expression evaluation. `{2*kp_n}` is a string token; we don't compute it.
- Dialects of other simulators (HSPICE-only syntax, Spectre
  `simulator lang=spice` blocks). Only LTspice and ngspice grammar is
  handled.
- Transformation passes over the whole netlist. The parser supports
  targeted edits only.
- Semantic validation, such as whether a `.MODEL` matches a simulator.
  That is left to the simulator.

spice_lex coexists with spicelib's `REPLACE_REGEXS`. spicelib's
`SpiceEditor` still writes the `.cir` files on disk for the simulation
pipeline, and spice_lex handles the lookup and edit paths. Every internal
caller goes through `lex` and the typed views, and writes through
spicelib only when the file must be written for the simulator.
spicelib's element-line regex is not used on those paths.

## Architecture

The parser has three layers.

### Layer 1: line-level lexer

```python
def lex(netlist_text: str) -> LexResult: ...
```

`lex` walks the lines once, classifies each line by its first
non-whitespace character, merges `+` continuation chains and tracks
`.SUBCKT`/`.ENDS` scope. Each `SpiceCard` has these fields:

```python
class SpiceCard:
    kind: Literal["model", "param", "instance", "subckt", "ends",
                  "meas", "directive", "end", "comment", "blank",
                  "control"]
    raw_lines: list[str]      # original source lines (preserved for emit)
    body: str                 # merged & comment-stripped (for parser)
    line_start: int           # 1-based line in the source
    name: str | None          # extracted by per-kind parser layer
    scope: tuple[str, ...]    # subckt nesting; () == top level
    dirty: bool = False       # set by typed-view setters
```

Atomic mutation primitives on `SpiceCard`:

- `replace_span(body_start, body_end, new_text)` — replaces one slice of
  the body and preserves the surrounding whitespace and continuation
  structure.
- `replace_body(new_body)` — replaces the whole body, for when the whole
  card must be re-rendered.

Both set the card's `dirty` flag.

### Layer 2: typed views

Typed views are short-lived wrappers re-derived via `from_card()`, and
they never own state. Their setters change the underlying `SpiceCard`
directly.

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
    def set_nodes(self, new_nodes) -> None: ...   # connectivity edit (fixed-arity only)
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

`MeasCard` lets `spice_validator` check whether a body contains a
`vdb()` call, or references a signal not present in the `.raw` file,
without substring matching. It walks Layer 3's classified tokens in
`body_tokens`, where a `BARE` token immediately followed by a `PARENED`
token is a function call.

The `ELEMENT_SPECS` registry covers M/Q/J/X/F/H/R/C/L/V/I/B/E/G/K and
classifies by context (E/G/F/H with a positional gain or in the
`VALUE=` form).

### Layer 3: body tokenizer

After continuation lines are merged, the card body is a single string.
A hand-written, single-pass state machine turns it into **classified
tokens** rather than raw strings, so Layer 2 views never re-parse
character classes.

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

**State machine.** The tokenizer makes a single forward walk over the
body. Its state is `(brace_depth, paren_depth, in_quote)`, with these
transitions:

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

The transitions give these rules:

- `(` inside `{...}` does not open a paren context, because braces take
  precedence.
- `;` and `$` start a comment only outside quotes and braces.
- `*` line comments are handled at Layer 1 and never reach Layer 3.
- Unbalanced delimiters at body end raise `SpiceLexError` with the
  position; recovery is not attempted at this layer.
- `KEY_VALUE` is recognized when a `=` is seen at depth 0 outside a
  quote; the preceding `BARE` token becomes `key`, the following token
  (which may itself be `BRACED`/`PARENED`/`QUOTED`/`BARE`) becomes
  `value`. Whitespace around `=` is allowed; it produces an `EQUALS`
  token and adjacent tokens that later passes pair up.

`text`, `key` and `value` all preserve the original case. Identifier
matching (model names, refs, param keys) folds to lower case at lookup
time.

### The model-position rule

Once tokens are classified, one rule separates the nodes of an instance
line from its model token.

> **Model/value position is the last `BARE`-or-`QUOTED` token before
> any `KEY_VALUE` token** (or the last such token if the line has no
> `KEY_VALUE` tokens). Tokens before that are nodes.

The rule handles every hard case that the old heuristic missed:

- `M1 d g s b "NMOS_lvt" W=10u` — the quoted token is correctly
  identified as the model name.
- `X1 a b c MYSUB W=10u` — `MYSUB` is the last `BARE` before the
  `KEY_VALUE`, regardless of how many ports preceded it.
- `M1 d g s NMOS1` — no `KEY_VALUE`, so the last `BARE` is the model.
- `R1 n1 n2 1k TC=0.001` — `1k` is BARE (numeric literal), correctly
  picked as the value.

The arity of each prefix is only a minimum node count, used as a sanity
check. It is not used to find the model position:

```python
MIN_NODES = {
    "M": 3, "Q": 3, "J": 3,        # MOSFET / BJT / JFET
    "R": 2, "C": 2, "L": 2,        # passives
    "V": 2, "I": 2,                # sources
    "B": 2,                        # behavioural
    "E": 4, "G": 4,                # VCVS / VCCS (2 output + 2 control nodes)
    "F": 2, "H": 2,                # CCCS / CCVS (control source ref is BARE)
    "X": 1,                        # subckt call: at least one port
    "K": 0,                        # mutual inductance: refs only, no nodes
}
```

`Bxxx` / `Exxx` / `Gxxx` give their value as a `KEY_VALUE` token like
`V={expression}` or `I={...}`; the tokenizer always emits the expression
body as one `BRACED` token. These lines have no "model" position, so
`InstanceLine` exposes the relevant `KEY_VALUE` (`V=` or `I=`) as
`value`.

`Fxxx` / `Hxxx` reference a controlling source by name (a `BARE` token)
and then take a gain, the same model-after-nodes shape as `M` lines.

## Ownership and emit invariants

`lex()` returns a single flat `list[SpiceCard]` in source order, and
that list is the canonical store. Typed views are short-lived wrappers
and never own child cards. This has two consequences:

- A `.SUBCKT` block contributes one `kind="subckt"` card (opener), one
  `kind="ends"` card (closer), and the inner cards as flat-list entries
  with `scope=(name,)`. Mutating an inner card sets only that card's
  dirty flag. The opener and closer stay byte-identical unless they
  themselves are mutated. `SubcktCard.dirty` does not propagate; it
  marks the opener card and never the cards in its scope.
- `emit(cards)` walks the flat list once and, for each card, copies
  `raw_lines` if clean or re-renders from typed fields if dirty. Order
  and surrounding whitespace are preserved.

The parser also maintains invariants that span two cards:

- `.SUBCKT NAME ... .ENDS [NAME]`: the optional trailing name on `.ENDS`
  must match the opener. Renames go through `rename_subckt`, which marks
  both the opener and the matching ends card dirty in one transactional
  operation. Plain field assignment on the typed view is not enough.
- Nested `.SUBCKT`: scope is a tuple, so `.SUBCKT INNER` inside
  `.SUBCKT OUTER` gets `scope=("OUTER", "INNER")`. Each inner subckt is
  its own pair of subckt + ends cards in the flat list; `iter_body`
  filters by scope prefix and recurses by extending the prefix.

The byte-faithful round trip holds even for malformed input, under these
recovery rules:

- `.ENDS` without a matching opener: the closer card is emitted with
  `scope=()`, and a warning is reported in `lex()` metadata. Subsequent
  cards resume at the outermost still-open scope, or `()` if none.
- EOF inside an open `.SUBCKT`: the cards are returned unchanged with
  the unclosed scope, plus a warning. `emit()` still round-trips
  byte-faithful.
- Cards after a top-level `.END`: they keep their position and scope but
  have `trailing=True`. LTspice ignores them, but the parser keeps them
  rather than dropping them.
- `.control` without a matching `.endc`: the region runs to EOF, plus a
  warning. Every remaining line stays an opaque `kind="control"` card.

A `.control` ... `.endc` region (ngspice) contains simulator script
rather than netlist, so the parser treats it as opaque. Each of its
lines becomes one `kind="control"` card with an empty `body`, one card
per source line with no `+` continuation merging, so device, arity and
model rules find nothing to inspect. Without this, every control command
would lex as the element sharing its first letter (`let` → L, `dc` → D,
`meas` → M, `foreach` → F, `alter` → A, `set` → S) and a valid ngspice
deck would produce many spurious arity errors. Both delimiters are part
of the region; a stray `.endc` with no opener stays an ordinary
directive.

Holding two typed views over the same card and mutating both is
undefined, so re-derive the view after each mutation. The flat list is
the only synchronization point. Sequential setters on
the same view are safe, because `_shift_cached_param_tokens` /
`_shift_single_token` shift the token caches on each mutation.

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

class SpiceCard:                 # canonical store
    def replace_span(self, body_start, body_end, new_text) -> None: ...
    def replace_body(self, new_body) -> None: ...
    def line_at(self, body_offset) -> int: ...   # 1-based source line of body[body_offset]
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
    def set_nodes(self, new_nodes) -> None: ...
    def get_param(self, key) -> str | None: ...
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
def value_suffix_sites(cards) -> list[ValueSuffixSite]: ...  # non-ASCII at a scale-suffix position
def fold_micro_suffix_cards(cards) -> tuple[ValueSuffixSite, ...]: ...  # µ/μ suffix -> u, in place
def strip_instance_section_signs(cards) -> dict[str, str]: ...  # R§Load -> RLoad wherever named

# lib/encoding.py — shared text decoding
def decode_spice_bytes(raw: bytes) -> str: ...
def decode_spice_bytes_with_encoding(raw: bytes) -> tuple[str, str]: ...  # (text, codec)
def rewrite_codec(raw: bytes) -> str: ...  # codec a rewritten copy keeps (UTF-16/32 -> utf-8)
def encode_spice_text(text: str, codec: str) -> bytes: ...  # UTF-8 where the codec cannot spell it
def detect_utf16_endianness(probe: bytes) -> str | None: ...
def read_spice_text(path: Path) -> str: ...
def read_spice_text_with_encoding(path: Path) -> tuple[str, str]: ...
```

## Future work

None of these is implemented. Add each one when code in the project
needs it. Do not leave stubs in the tree, because an importable
`NotImplementedError` placeholder misrepresents what the API supports.

- **`rename_component(cards, old_ref, new_ref)`** — atomic rename of an
  instance ref across the instance card *and* every `.PARAM` expression
  / `.MEAS` body / behavioural-source body that references it. It needs
  edits inside tokens, walking the contents of `BRACED` tokens and
  patching the substring while keeping the delimiters balanced.
- **`inline_subckt(cards, instance_ref)`** — flatten an `Xxxx`
  invocation into its parent scope. Substitute the call's actual nodes
  for the subckt's formal ports throughout the body, generate fresh refs
  to avoid collisions, splice the body cards into the parent scope, and
  remove the `Xxxx` card.
- **`extract_subckt(cards, body_indices, new_subckt_name)`** — inverse
  of inline. Wrap a span of cards in a new `.SUBCKT` opener / `.ENDS`
  closer, replace external node references with formal ports, and leave
  a single `Xxxx` invocation in place of the original span.
- **`netlist_diff(a, b)`** — structural diff that ignores formatting.
  Compare card-by-card by `(kind, name, body)` after canonical
  normalization; report added / removed / modified cards. Useful for
  regression tests that compare netlist structure.
- **`ChangeSet`** — atomic-commit primitive. Stage cross-card mutations
  as a list of `(card, replacement_raw_lines)` pairs; commit in one pass
  after validation; roll back via the captured original `raw_lines` if
  any commit step raises. It would replace the best-effort
  "validate-then-commit" of `rename_subckt` with a transactional
  contract.
- **Static linters** — every `Mxxx` references a `.MODEL` in scope;
  every `.PARAM` is consumed somewhere; no duplicate refs; no dangling
  `.SUBCKT` calls. Each is a single-pass walk over cards using the typed
  views.

Design notes for implementing the deferred items:

- **Cross-card transactions**: `ChangeSet` is the intended container.
  Snapshot raw_lines on entry, mutate via `replace_body` / `replace_span`,
  and on `__exit__` commit if no exception was raised, otherwise restore.
  The current `rename_subckt` should migrate to `ChangeSet` once it
  exists.
- **Edits inside tokens** for `rename_component`: the body tokenizer's
  `BRACED` / `PARENED` tokens are opaque strings. Rewriting `{R1*2}` →
  `{Rnew*2}` needs a sub-tokenizer for expression bodies. It should only
  replace identifiers and should not parse full expressions.
- **Scope manipulation**: `inline_subckt` and `extract_subckt` rewrite
  `card.scope` tuples for spliced cards. Because the flat list is the
  canonical store, this is mechanical. Detach a slice, rewrite its
  scopes, and splice it into the target position.
- **Atomicity guarantees**: the current `rename_subckt` validates and
  then commits, with no rollback. Once `ChangeSet` exists, it should
  default to full atomicity. Document the contract change in the commit
  that makes it.
