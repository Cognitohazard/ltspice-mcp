Every stepped `.fra` run on LTspice 26 lost all its measurements: LTspice 26
prints the margins in its measurement table with their units (`90.6295°`,
and `infdB` for a gain margin whose phase never crosses). A number followed
by `°` or `dB`, and `inf`, now reads as that number, and an infinite margin
is null with the section's count of non-finite values.
