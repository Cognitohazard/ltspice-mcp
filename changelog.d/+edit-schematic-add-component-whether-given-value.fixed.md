`edit_schematic`'s `add_component`, whether given `value` or a `Value` in
`attributes`, and `set_component_attribute` with attribute `Value` refuse a
Value LTspice would read as another node, such as a resistor's `1 k` or
`1 k tc=0.001`, as `set_component_value` did. Before, they wrote it and
the sheet netlisted with the wrong nodes. The check uses the element class
the part's own `Prefix` gives, including one set in the same op. Setting
or clearing a part's `Prefix` with `set_component_attribute` is refused
when the part's `Value` would read as another node in the new class, as
`add_component` refuses the same pair; before, a source holding `AC 1`
could be given Prefix `R`.
