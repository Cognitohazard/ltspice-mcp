`inspect(kind="net")`, the pin legend, the edit refusals and the route
rules read a sheet's nets by its wiring alone, where LTspice drops a wire
running straight between two pins of one part (recorded:
`connectivity/same_instance_wire`). On such a sheet `inspect` said both
pins were on one shorted net, the legend named both `a/b`, and a label
`b` on `R1.1` was accepted although it shorts R1. Every reader now reads
the nets LTspice netlists. With it, `wire_pins` no longer refuses joining
two names a wire already joins; a route treats a pin, a wire's end and a
label it passes alike, as a junction on either end's net and a refusal on
any other; a pin, label or route end placed between the ends of a dropped
wire keeps that wire, and a short it makes is refused; and `inspect`
refuses a start on a dropped wire as ambiguous.
