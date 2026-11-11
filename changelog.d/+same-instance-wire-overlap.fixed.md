A wire between two pins of one part that a second wire on the same line
overlaps, from a pin or between the pins, is now read as LTspice 26 and
LTspice XVII export it: dropped, with the two pins left apart. The server cut
the run where the second wire ended and called the pins joined, so
`verify_circuit` and `edit_schematic` missed the `dropped_wire` finding there,
and `inspect` traced the two pins as one net. `wire_pins` now judges whether
LTspice drops a route by the sheet with the route drawn on it, not by the
route alone. It refuses a route that merges with a wire already on the sheet
into a dropped run, and no longer calls a route dropped when a wire already on
the sheet keeps it; such a route that would join another net is refused for
that instead. A route laid along a dropped run and ending inside it, or one
that extends a wire on the sheet into such a run, is no longer refused for
joining the nets of the run's two pins, which LTspice leaves apart: it is
refused as a same-instance wire.
