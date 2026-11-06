`verify_circuit` draws a `render.scale` above 4 at 4, with a warning,
instead of refusing it. An `analyze_results` `thd` recipe's `harmonics`
above 50 is summed to the 50th harmonic, with a warning in the recipe's
`warnings`, instead of being refused; `ltspice_mcp.api.analyze_thd` does
the same for `n_harmonics`, warning in its result, instead of raising
`ValueError`. Neither schema declares a maximum for these fields any more.
