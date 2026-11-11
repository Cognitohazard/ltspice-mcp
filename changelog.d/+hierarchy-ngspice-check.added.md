`inspect(kind="hierarchy")` checks the values discovery resolved against
ngspice's own reading of the deck when ngspice is installed. ngspice reads
the deck in the query's mode through a driver that prints each value: a
resistor's value, a MOSFET's W and L, and the copies discovery counts for
either card where `m` is written or a subcircuit is rewritten for it, which
measures discovery's model of ngspice's `m` rewrite. The reply's
`ngspice_check` says whether they agree, lists any value that differs with
both readings, counts what was not compared and why, and says why the check
did not run when it did not. Nodes, models and sources are not compared.
