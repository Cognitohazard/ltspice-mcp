`verify_circuit` cut each rule's findings at 25, and `analyze_results` its
failure rows at 100, with no way to read the rest, and `verify_circuit`'s
hint counted the cut list. Each cut now gives the full count and names the
Python API call that returns every entry, and where it runs. A failure in
the attached analysis now says where its `api` snippet runs, as an
`analyze_results` failure does.
