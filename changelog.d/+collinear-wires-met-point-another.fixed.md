Where two collinear wires met at a point that another wire passes through,
the sheet checks merged the two into one wire and did not join the other
wire there. LTspice joins it, because each wire end on the point lies on
that wire (recorded: `connectivity/wire_ends_meeting_on_wire_interior`).
So a wire between two pins of one part, drawn with such a point, was
reported as a wire LTspice leaves out (`dropped_wire`) by `verify_circuit`
and `edit_schematic`, and `inspect(kind="net")` warned the same. LTspice
keeps that wire (recorded: `connectivity/same_instance_waypoint_on_wire`).
