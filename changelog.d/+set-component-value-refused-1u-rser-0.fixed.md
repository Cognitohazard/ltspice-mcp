`set_component_value` refused `1u Rser={0.1 * k}` with advice to put the
expression in braces, although it is braced and the write would have set
it. The check now splits the value as the write does.
