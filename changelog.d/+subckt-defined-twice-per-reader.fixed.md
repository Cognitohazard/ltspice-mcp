A deck that defines one subcircuit twice was read as if the first definition
were in force for every simulator. Each reader now reads the definition its
simulator uses: ngspice the first it reads (measured in every compatibility
mode, at the top level, inside a subcircuit, within and across an include, and
with three), and LTspice XVII the last of two in the deck itself. LTspice 26
refuses such a deck, so `run_experiments` now refuses it before the run
(`subckt-defined-twice`) and `verify_circuit`'s `syntax` check reports it
(`subckt_defined_twice`), naming the fix. Where what the simulator does is not
recorded (another LTspice build, one whose build is not known yet, QSPICE,
Xyce, or for LTspice more than two definitions, two inside one subcircuit or
one in an included file) no definition is read, and the reply says so: a
`subckt-defined-twice-unrecorded` warning, a `reason` on the comparison's
`duplicate_subckts` entry, and a refusal from nested targeting. The netlist
checks (`dangling_node`, `floating_net`, `undefined_reference`) now read the
definition in force instead of both bodies together. A mismatch rule refuses a
device subcircuit defined in more than one file or in more than one scope,
since it cannot tell which one the simulator reads.
