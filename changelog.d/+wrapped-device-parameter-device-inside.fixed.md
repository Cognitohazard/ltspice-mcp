A wrapped device parameter of a device inside a subcircuit, such as
`v(@xa:xin:mtail[gm])`, was not matched to LTspice's warning that it is
unknown, so its zero trace read as a value. Every reader of
`@device[param]` names now takes them apart one way.
