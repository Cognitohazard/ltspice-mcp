The `equivalence` compare of `verify_circuit` and of `edit_schematic`'s
reference check also compares the directives, so `equivalent` is true only
when the wiring, the parts and values, and the directives all match.
Before, it read no directive but the includes it followed for subcircuit
definitions, so a changed `.param` that every part still read as `{rv}`
came back `equivalent: true`. The directives are read as `structural_diff`
reads them, so spelling is not a difference, LTspice's export boilerplate
is left out, and the two modes list a directive change the same way. Beside
a false verdict, `directives_equivalent` says whether the directives
matched, as `structurally_equivalent` does for the wiring, and
`directives_added` and `directives_removed` list the difference. A
subcircuit definition is compared as wiring wherever it is defined, so its
`.subckt` line, `PARAMS:` defaults and body directives are not compared,
and moving a definition into an included file shows as the `.include` line
added. The `mode` and tool descriptions now say what each mode compares,
and that `structural_diff` does not compare nodes.
