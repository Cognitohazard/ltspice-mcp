With `[security] allowed_paths` whose first entry is not the working
directory, every URI `spice://netlists/` listed read back as not found,
and the Python API's note on a refused path named the wrong base for
relative paths. Both now take relative paths from one base.
