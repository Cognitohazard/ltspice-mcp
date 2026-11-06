A step selector's value written in SPICE notation (`2k`, `0.68k`) matched no
step, and the selector fell back to the first step without a word. It is
now read as every other value of a request: a number picks the nearest
step, and a value or axis the run does not have is refused with the ones it
has. A stepped temperature was named `temp` on one run and `temperature` on
a run LTspice stores a point a step (a stepped `.op`, `.tf` or
one-frequency `.ac`), so a selector or `group_by` that worked on one failed
on the other; it is now named as the deck's `.step` names it, and
`temperature` also selects it.
