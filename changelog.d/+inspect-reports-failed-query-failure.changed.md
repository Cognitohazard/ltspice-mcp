`inspect` reports a failed query by its failure's own error code, as every
other tool does, where every one read `error`: `netlist_invalid` for a
netlist or sheet it cannot read, `symbol_unresolved`, `library_error`,
`other_raw_not_written`, and so on. The code `error` is no longer emitted.
