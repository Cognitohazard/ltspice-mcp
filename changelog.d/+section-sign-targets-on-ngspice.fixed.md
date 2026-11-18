A variation on a schematic run on ngspice now finds a part LTspice exported
with its `§` marker (`R§Load`) by every one of its names: `Load` (the name on
the sheet and in LTspice's results), `R§Load` and `RLoad`. Before, only `RLoad`
worked there, because ngspice's copy of the deck takes the marker out. That copy
now records each name it changed in a comment after its first line.
