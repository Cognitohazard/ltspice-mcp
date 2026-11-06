The warning `set_component_value` gave of a subcircuit call's Value (an
opamp's complexity label such as `Level.2`, or any Value on a part whose
model `SpiceModel` selects) is the sheet finding `subcircuit_value_slot`:
`edit_schematic` says it once of the finished sheet however many ops set the
value, and `verify_circuit` reports it under `quality`. It reads the part's
element letter from its own or its symbol's `Prefix`, so a transistor or
diode whose Value after its `SpiceModel` is an area factor is no longer
warned of.
