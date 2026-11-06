A stepped run's summary copied the log header's temperature into every
step's row. When the run steps its temperature, the header holds the first
step's value on LTspice 26 and the last step's on LTspice XVII, so a row
for the 27 °C step said -40 or 125. Each row now gives its own step's
temperature, and leaves `temp_c` out when that step's is not known.
