`inspect(kind="hierarchy")` checks the values discovery resolved against
ngspice's own reading of the deck when ngspice is installed. ngspice reads
the deck in the query's mode through a driver that prints each value: a
resistor's value, a MOSFET's W and L, and an `m` written on either card.
The reply's `ngspice_check` says whether they agree, lists any value that
differs with both readings, counts what was not compared and why, and says
why the check did not run when it did not. An `m` inside a subcircuit once
any call sets `m`, and a 0 ohm resistor, are left out, because ngspice
changes them by design. Nodes, models and sources are not compared.
