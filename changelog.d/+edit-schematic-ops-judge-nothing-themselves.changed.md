`edit_schematic` ops judge nothing themselves, so `warnings` holds the
sheet's findings alone. `wire_pins` no longer warns that a route passing
over a pin or touching wiring already on its net is joined there; the point
is in its `results` entry under `junctions`, as it already was.
