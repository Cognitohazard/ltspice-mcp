`inspect(kind="components")` of a netlist, and the Python API's `inspect`,
showed a card inside a `.subckt` body in place of a top-level card of the
same reference, with the body card's nodes and value under the top-level
name. The top-level cards are now listed first and unchanged; a card of a
subcircuit body follows them with `subcircuit`, the names of the bodies it
sits in, outermost first, and a reference is one card only within its body.
A subcircuit defined more than once in one scope has its bodies' cards left
out of the listing, with a warning that names it.
