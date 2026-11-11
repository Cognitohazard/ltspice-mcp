Setting the value of a voltage or current source keeps the source's own
parameters: a voltage source's `Rser=` and `Cpar=`, the AC magnitude
written `AC=2` or `AC 2`, and a current source's `load`. Before, an `assign`
variation and a value edit of an instance inside a subcircuit wrote the card
as its nodes and the new value alone, so `V1 in 0 PULSE(...) AC 1` assigned
another pulse lost its AC magnitude and `V1 a 0 5 Rser=1k` set to 4 lost its
series resistance. On a sheet, `set_component_value` lost a parameter kept in
the part's Value the same way, set only one copy of a parameter the part gave
twice, and moved a PWL's `TIME_SCALE_FACTOR=` out of the value into
SpiceLine. A value edit now also accepts a parameter of the source's own on
an `assign`, which it refused. A source that plays a .wav file (`wavefile=`)
has no value, and setting one is refused, where before the file was silently
replaced by the value.
