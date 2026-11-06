The equivalence compare merged two definitions of one subcircuit into the
first, where the first should win alone; read a `.lib tt` section line
inside a library as an include of a file named `tt`; and flattened every
section of a library whatever section the deck selected, so
`.lib corners.lib ff` gave every corner's parts and reported `tt` and `ff`
as missing includes. It now reads each definition's own body and only the
selected section, and a run's requirements follow includes the same way.
