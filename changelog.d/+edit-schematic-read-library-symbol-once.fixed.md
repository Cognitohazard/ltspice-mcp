`edit_schematic` read each library symbol once per session. A symbol
redrawn in a library while the server ran kept its old pins in every later
edit, and a symbol not found stayed not found after it was added, until the
server restarted. Every tool now reads a symbol again after its file
changes, and looks again beside the sheet and in the configured folders for
one it did not find. LTspice's own library, which does not change while the
server runs, is searched once per name.
