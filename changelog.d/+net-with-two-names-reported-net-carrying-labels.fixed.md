`net_with_two_names` reported one net carrying two labels where a wire
running straight between two labelled pins of one part joined them on the
sheet. LTspice drops that wire and keeps the labels on separate nodes
(recorded: `connectivity/same_instance_wire`, exported as `R1 a b`), so the
finding is no longer given there. It and `joined_at_crossing` now read the
sheet's nets as LTspice netlists them.
