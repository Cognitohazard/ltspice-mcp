`edit_schematic` and the other sheet readers opened the sheet beside a
block symbol that gives a `Prefix` as the block's contents. LTspice reads
such a block as a call to the subcircuit its `Value` names and reads no
sheet for it, so neither does the server now; a block symbol with no
`Prefix`, as LTspice draws one, is read from its own sheet as before.
