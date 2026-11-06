An AC results file from a deck with `.options numdgt` above 6 was refused
("unsupported double flag for this layout"). Under that option both
LTspice builds mark an AC plot `double`, but store it as any AC plot, so
it now reads as the same run without the option.
