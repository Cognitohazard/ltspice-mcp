`edit_schematic` found each part's symbol by spicelib's own search, unlike
`inspect`, `verify_circuit` and the drawing. It refused a sheet naming
`lib\part` with `part.asy` beside it, which LTspice XVII netlists, and
opened a sheet whose bare `part` sits in a subfolder, which neither build
netlists. It now finds symbols as the other tools do.
