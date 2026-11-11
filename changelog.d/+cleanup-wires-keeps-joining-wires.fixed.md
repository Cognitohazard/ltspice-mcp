`remove_component` with `cleanup_wires: true` deleted every wire with an end
on the removed part's pins, so two wires that met at a pin, and still met each
other once the part was gone, were both deleted and their net split. It now
deletes only the wires the removal leaves dangling: a wire that still joins
two or more pins or labels by wire stays, whether through another wire at the
pin, a label or ground flag on the pin, or pins along it.
