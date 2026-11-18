The server now reads a `.lib` card as the simulator and ngspice mode that
run the deck do, in every check: the lint before a run, the analyses a run
expects, `inspect(kind="hierarchy")` and `verify_circuit`'s comparison.

- ngspice reads only the selected section of `.lib <file> <section>`, not the
  cards outside every section. A model or `.save` that only those cards hold
  no longer counts as present, and the comparison reads the definition in the
  selected section.
- A `.lib <file>` with no section, or an `.include`, of a file that declares
  sections, in ngspice's default `kiltpsa` mode, is now refused
  (`lib-sections-dropped-ngspice`): ngspice reads none of the file's cards,
  and then, depending on the build, either stops (ngspice-45) or goes on
  with a wrong answer (ngspice-42).
- A `.lib <file>` with no section in native or `hsa` mode, and an `.include`
  of a file that declares sections there, on which ngspice stops, are now
  refused before the run (`lib-without-section-ngspice`).
- A `.lib <file> <section>` in a file the deck includes is now refused in
  `kiltpsa` too (`lib-section-ngspice`), as it was in the deck itself.
- LTspice reads the rest of a `.lib` line as one path, so an unquoted path
  holding a space (`.lib my models/amp.lib`) is no longer refused, and
  `.lib file section` is refused with what LTspice looks for.

A power in a parameter expression (`^` or `**`) is now resolved only where its
value is the one ngspice computes in the deck's mode: a positive base, and a
negative base with an integer exponent in a mode with `hs` or `lt` set
(`hsa`, `kiltpsa`). Native ngspice takes the power of a negative base's
magnitude, so `(-2)^3` is 8 there, where `inspect` reported -8; a zero base
with a zero or negative exponent differs by mode; and ngspice refuses a
negated name straight after the operator (`3^-b`). Those are now reported
unresolved with the reason, and a mismatch draw that depends on one is
refused.
