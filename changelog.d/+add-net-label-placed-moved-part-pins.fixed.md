`add_net_label`, a placed or moved part's pins and `wire_pins` counted a
net's names through ground. A wire whose only label is a ground flag took
the names of every other grounded net, so joining it to `VSS` while `VDD`
was grounded elsewhere was refused as a short of the two, though moving a
`VSS` label there was allowed. A net in these checks is now known by the
names its own labels carry, as in `move`: joining a ground-only wire to a
named net grounds that net, which a net may be whatever else is. The
route check's refusal for a label picked up mid-route now lists only the
names the route's wiring joins. A route that also runs straight across
two pins of one part is refused for that.
