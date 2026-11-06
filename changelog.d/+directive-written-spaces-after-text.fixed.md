A directive written with two spaces after `TEXT` was shown by `inspect`
and `verify_circuit` and deleted from the file by the next `edit_schematic`
call, with no warning, because spicelib's reader skips such a line.
`edit_schematic` now refuses to edit a sheet with a text, wire or flag line
its editor did not read, naming the first such text line.
