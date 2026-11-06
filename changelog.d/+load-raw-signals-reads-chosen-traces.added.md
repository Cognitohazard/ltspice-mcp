The Python API's `load_raw` takes `signals`: a read that keeps only the
traces those names read (one trace, or a node pair's two nodes, as `trace`
reads them), the axis and any step parameters. The 512 MiB limit on a read's
numbers applies to what is kept, so a raw too large to read whole, such as a
long transient saving every node, can be read for the few signals a
measurement needs. A raw stored a point at a time, as LTspice stores a run,
is still decoded whole while it is read, which a read holds to 1 GiB of
numbers; one converted to LTspice's Fast Access layout is decoded only for
the traces kept. A name the plot has no trace for, or one that holds each
step's stepped value, is refused with the traces it has, and the result's
`observations` include `signals_chosen`. `compare` with such a result lists
the traces one result was not read for as `unread_there` or `unread_here`,
not as missing from it.
