A Pelgrom `mismatch` rule ignored a device's multiplier `m`. The simulator
builds m parallel copies from one card and they share its one draw, so
their area is `m·W·L`, but the draw took the area of a single copy: a device
written as `W=1u m=12` got a threshold spread √12 (about 3.5×) too large,
with no warning. The area now counts the copies the simulator builds from
the device's card, on a top-level `M` card, a device one subcircuit level
down and a nested `instance` target alike: on LTspice the device's own `m`,
including an inner `m={mult}` set from the X line; on ngspice its `m` after
ngspice passes an X line's `m` down to a subcircuit that declares no `m`,
replacing each card's own `m` or, in the HSPICE modes, multiplying it, in
the order ngspice reads the deck and leaving out unused subcircuits, as
measured on ngspice-42. A deck ngspice stops on for that rewrite is refused
with `multiplier_rewrite_stops`, and a case whose edits or private
subcircuit copies would change what a card no edit writes builds with
`multiplier_rewrite_changed`. A deck with no semantic profile (QSPICE,
Xyce) counts the product of the X line's `m` and the device's own, as
before; neither simulator's rule is measured. An `m`
that does not resolve to a positive number refuses the request with
`mismatch_geometry_unresolved` instead of being drawn as one copy. A refusal
while drawing for a device one subcircuit level down now reaches the caller
as a coded error rather than an uncoded one.
