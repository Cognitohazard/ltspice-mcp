The recovery check of a run's frozen inputs refused a seeded ngspice run
whose second line is its one analysis, as having two, and a recoverable run
whose second line opens a subcircuit. It read the deck's first card twice,
once in place of the title line. It now reads line 1 as the title.
