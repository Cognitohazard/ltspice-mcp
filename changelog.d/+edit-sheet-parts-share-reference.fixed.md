An edit of a sheet on which two parts share a reference deleted all but the
last of those parts from the file, whatever the edit was. `edit_schematic`
now refuses to edit such a sheet and names each shared reference and where
its parts are. Neither LTspice build netlists such a sheet (recorded:
`connectivity/shared_reference`).
