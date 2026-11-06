**Breaking:** `verify_circuit`'s `export_to` defaults to `"managed"`: the
export check exports a copy in the server's own folder and writes nothing
next to the schematic. It used to overwrite `<name>.net` beside the sheet
unless asked not to, the one default left that wrote into the caller's
folder after `run_experiments` stopped. `export_to="sidecar"` still writes
that file. Both modes report what changed since the schematic's previous
export (`diff_vs_prior`): a sidecar export against the file it replaces, a
managed one against the previous managed export, which the server keeps.
