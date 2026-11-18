A netlist comparison now refuses a subcircuit that calls itself in every case.
Before, after a port-arity mismatch elsewhere in the deck, it treated the
recursive instance as a black box and left that part of the circuit
uncompared with nothing to say so. `verify_circuit`'s and `edit_schematic`'s
equivalence comparison return the failure with its fix instead of failing the
call, and `diff_vs_prior` says the connections were not compared.
