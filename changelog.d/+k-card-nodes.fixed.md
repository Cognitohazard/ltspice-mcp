A mutual-inductance card (`K1 L1 L2 0.9`) was read as if the inductors it
couples were its nodes, so `inspect` listed it with nodes named `L1` and `L2`
and a net query counted it on nets of those names, and a comparison of two
netlists reported no change when only its coupling value changed. Both
LTspice 26 and XVII were recorded wiring no node to such a card; it now has
no nodes, and its value reads as the inductors it couples followed by the
coupling value (`L1 L2 0.9`), so a changed coupling shows in a comparison.
