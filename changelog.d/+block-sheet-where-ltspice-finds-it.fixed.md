A hierarchical block's own sheet is found where LTspice finds it: beside the
block's symbol, else beside the sheet being exported, and nowhere else.
Before, when neither held it, the server searched the symbol libraries and
their folders, and beside a block's sheet that placed another block, for a
sheet of the block's name, which LTspice does not, and read the block from a
sheet LTspice would not netlist. A block's sheet now also finds a symbol it
places beside the sheet being exported, as LTspice does, where the schematic
editor refused the whole sheet for want of it. A run of a sheet whose block
symbol is in a library and whose block sheet sits beside it now takes that
sheet along to the export, which failed without it.
