A sheet finding, `pin_inside_wire`, reported by `edit_schematic` and by
`verify_circuit`'s `layout` check: a pin on a wire's interior, where no wire
ends, which LTspice joins to that wire's net though the drawing shows no
join. A part placed or moved so that a pin landed inside a wire joined its net
with nothing said, unless the pin was where wires cross.
