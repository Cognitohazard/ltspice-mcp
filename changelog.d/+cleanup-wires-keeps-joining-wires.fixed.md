`remove_component` with `cleanup_wires: true` deleted every wire with an end
on the removed part's pins, so two wires that met at a pin and still met each
other once the part was gone were both deleted, splitting their net with no
warning. It now deletes only the wires the removal leaves dangling: a wire at
one of the pins goes when taking it away parts nothing it joins, so wires that
meet at the pin, or reach other pins along them, stay.
