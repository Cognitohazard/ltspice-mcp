`edit_schematic`'s `replace_component` op gives a part a new symbol where it
stands, in one call: a gate for a buffer, `nmos4` for `nmos`. The part
keeps its reference, place, turn, attributes and value (unless `value`
gives one). Its pins pair with the new symbol's by name, or as `pins` pairs
them, and a pin that moved is wired back to its net by a route the
`wire_pins` planner accepts. The swap is refused, and the sheet left as it
was, if a pin would end up joined to anything but what its old pin was, or
any other net would change. The result names the wires drawn, the old pins
whose wiring was left joined to nothing of the part (`left_at`) and the new
pins left unconnected (`unpaired_pins`). Removing and adding the part, then
wiring each pin again, took a dozen ops. The Python API exports
`ReplaceComponentOp`.
