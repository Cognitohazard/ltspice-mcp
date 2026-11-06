`floating_pin` did not report the pins of a part joined only by a wire
running straight between them. LTspice drops that wire and leaves both
pins unconnected (recorded: `connectivity/same_instance_bare`), so
`verify_circuit` and `edit_schematic` now report each pin as floating,
beside the `dropped_wire` finding for the wire. The rule now reads the
sheet's nets as LTspice netlists them.
