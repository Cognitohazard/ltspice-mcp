A card with a quote, brace or parenthesis left open, or a closer with
nothing open, such as `R1 in out {1k`, cannot be split into its parts, and
every check left it out without a word, so `verify_circuit` and the linter
read the deck as one without it: a node wired through it reported as wired
to one element, and a measurement of it as naming nothing. Such a card is
now reported at its line (`unreadable_card` in `verify_circuit`, the
`unreadable-card` lint rule, a warning in both). On a part, `.subckt`
header or `.global` like it, the nodes and names, inside an expression
too, count as connected and defined, and an `X` of a subcircuit holding one
as tied to ground at every port, so no other rule reports on a guess about
it; a `.save` or `.meas` like it connects nothing.
