The server now reads a `.lib` card as the simulator and ngspice mode that
run the deck do, in every check: the lint before a run, the analyses a run
expects, `inspect(kind="hierarchy")` and `verify_circuit`'s comparison.

- ngspice reads only the selected section of `.lib <file> <section>`, not the
  cards outside every section. A model or `.save` that only those cards hold
  no longer counts as present, and the comparison reads the definition in the
  selected section.
- A `.lib <file>` with no section on a file that declares sections, in
  ngspice's default `kiltpsa` mode, is now refused (`lib-sections-dropped-ngspice`):
  ngspice drops every card of the file and ran on with a wrong answer.
- A `.lib <file>` with no section in native or `hsa` mode, on which ngspice
  stops, is now refused before the run (`lib-without-section-ngspice`).
- LTspice reads the rest of a `.lib` line as one path, so an unquoted path
  holding a space (`.lib my models/amp.lib`) is no longer refused, and
  `.lib file section` is refused with what LTspice looks for.

A power in a parameter expression (`^` or `**`) is now resolved only where its
value is the one ngspice computes in the deck's mode: a positive base, and a
negative base with an integer exponent in `kiltpsa` and `hsa`. Native ngspice
takes the power of a negative base's magnitude, so `(-2)^3` is 8 there, where
`inspect` reported -8; and a zero base with a zero or negative exponent differs
by mode. Those are now reported unresolved with the reason, and a mismatch draw
that depends on one is refused.
