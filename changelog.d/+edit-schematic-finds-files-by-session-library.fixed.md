`edit_schematic` opens and edits a sheet with a subcircuit part that names a
model library (`ModelFile`), wherever that library is. Before, it refused
the sheet when the library could not be found, though `inspect` and
`verify_circuit` read it, and when the library was beside the sheet or in
the folder the server was started in, every edit failed. A hierarchical
block's own sheet is looked for beside the block's symbol, then beside the
sheet and in the symbol libraries, and never in the server's working
directory, so where the server was started no longer decides whether a
sheet can be edited. `inspect(kind="symbols")` lists exactly the names a
lookup finds, including when a configured symbol folder is the sheet's own
folder written another way.
