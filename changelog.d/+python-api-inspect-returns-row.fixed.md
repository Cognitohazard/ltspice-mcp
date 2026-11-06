The Python API's `inspect` returns every row of a paged query. It follows
each query's cursor, and joined the pages only where an item paged two
lists, so for every kind that pages one (`components`, `symbols`, `model`,
`simulator_docs`, a netlist `net`, the `open_in_ltspice` listing) it
returned the last page alone, under a `total` that counted every row: 50
of a netlist's 150 components. Every paged `inspect` item now names the
list its cursor pages in `page.collections`, one collection or two, which
is what the API joins by; `hierarchy` and `results`, which had added that
block themselves, now get it the same way.
