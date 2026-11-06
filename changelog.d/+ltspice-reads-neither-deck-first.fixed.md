LTspice reads neither a deck's first line, its title, nor any card after a
top-level `.end` (recorded: `deck/title_card`, `deck/title_directive`,
`deck/after_end`), but several readers did. Every reader of a deck now
skips both, in the deck and in an included file; what LTspice does after an
included file's `.end` is not recorded, so every reader follows the reading
the hierarchy already took.

- The equivalence compare counted a title that reads as a part as a second
  component, and the structural diff counted a title `.param` as a
  directive.
- `inspect(kind="net")` and `inspect(kind="components")` listed a part
  after `.end`.
- The linter refused a deck for an analysis after `.end`, and a title
  `.tran` counted as a requested analysis.
- `verify_circuit`'s directive check walked physical lines, so it checked a
  title that reads as a `.meas` and missed a `.meas` whose function sits on
  a `+` continuation line, which the linter refuses. It now reads whole
  cards.
- `run_experiments` accepted assignments and random rules naming a part,
  `.param` or `.model` found only in the title or after `.end`, wrote the
  change where LTspice never reads it, and reported it in the receipt.
  Mismatch rules perturbed a title that reads as a transistor or a
  subcircuit call, and staging copied, or refused to run without, an
  include named in the title or after `.end`.
- The `.options logopinfo` added to an `.op` run could land after the
  deck's `.end`, or be left out because of a line LTspice ignores; it is now
  added before the `.end` that ends the deck, also when the `.op` is in an
  included file or the deck is UTF-16, and a comment that mentions
  `logopinfo` no longer keeps it out.
- The ngspice `.control` write injection read the title too, so a deck
  titled "Write-up ..." got no `write` and ended with no raw. Measurement
  kinds, the noise input unit and the ngspice `.lib` section check read
  those lines too.
- The run observations also find a source whose value sits on a `+` line
  past a comment, and no longer take a `;` comment on a `.four` card for a
  requested signal.
