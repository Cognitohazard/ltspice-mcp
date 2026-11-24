A pin of an A-device (a digital gate or other mixed-signal primitive) with
nothing on it is now read as LTspice exports it: as node 0, exactly as a pin
on ground. Before, the server read such a pin as alone on a net of its own,
reported it as a floating pin, and compared a sheet's netlist against an
export as if the pin were not on ground. `verify_circuit` and
`edit_schematic` now report one `a_device_open_pins` observation per part
naming its open pins; the pin legend and `inspect(kind="net")` give each open
pin the node LTspice writes for it (`open_node`), or say why it is not known
(`open_node_unknown`); and a trace of ground names the A-device pins LTspice
also writes as node 0. A floating pin of an element letter no recording
covers now says so.
