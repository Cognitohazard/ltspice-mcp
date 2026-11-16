A deck that defines one subcircuit twice was read as if the first definition
were in force for every simulator. Each reader now reads the definition its
simulator uses: ngspice the first (measured in every compatibility mode, at
the top level, inside a subcircuit and across an include), and LTspice XVII
the last of two in one deck. LTspice 26 refuses such a deck, so
`run_experiments` now refuses it before the run (`subckt-defined-twice`) and
`verify_circuit`'s `syntax` check reports it (`subckt_defined_twice`), naming
the fix. Where what the simulator does is not recorded (another LTspice build,
one whose build is not known yet, QSPICE, Xyce, or for LTspice two definitions
inside one subcircuit or across an include) no definition is read, and the
reply says so: a `subckt-defined-twice-unrecorded` warning, a `reason` on the
comparison's `duplicate_subckts` entry, and a refusal from nested targeting.
The netlist checks (`dangling_node`, `floating_net`, `undefined_reference`)
now read the definition in force instead of both bodies together.
