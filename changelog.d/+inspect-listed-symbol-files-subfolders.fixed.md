`inspect(kind="symbols")` listed symbol files in subfolders of the sheet's
folder, which neither LTspice build finds by a bare name, so `add_component`
refused a name the listing had just offered. It now lists the sheet's own
folder and each library in full, where a bare name is found.
