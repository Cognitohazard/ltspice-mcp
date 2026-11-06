`inspect(kind="net")` put two pins of one part on one net where LTspice
leaves them apart. One case is a wire running straight between the two
pins, which LTspice drops (recorded: `connectivity/same_instance_wire`). The
other is the part's pins stacked on one point with nothing else there
(`connectivity/pins_of_one_part_on_one_point`). A trace now follows the nets
as LTspice netlists them. A trace from inside such a wire, or from a point
holding stacked pins, is refused and names the pins to trace instead.
