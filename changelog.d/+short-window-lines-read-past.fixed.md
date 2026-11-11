A `WINDOW` line short of its justification or size is read past, as both
LTspice builds read past one. On a sheet, the drawing places the part's text
at its symbol's window, and `inspect` and the drawing no longer report the
line as one they could not read. In a symbol, the attribute is not drawn,
where it was drawn left-aligned at the normal size. `edit_schematic` still
cannot open such a sheet, because the schematic editor it uses stops on the
line, and now refuses it naming the line and how to fix it, where it failed
with an unpacking error.
