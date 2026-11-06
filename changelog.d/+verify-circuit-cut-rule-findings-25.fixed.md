`analyze_results` cut its failure rows at 100, with no way to read the
rest. The cut now gives the full count and names the Python API call that
returns every row, and where it runs. A failure in
the attached analysis now says where its `api` snippet runs, as an
`analyze_results` failure does.
