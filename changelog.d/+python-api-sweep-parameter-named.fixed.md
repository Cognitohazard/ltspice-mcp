In the Python API, a sweep over a parameter named `budget` came back with a
false warning that `budget` was ignored, and one named `bias_cursor` was
refused, though the MCP tool took both. Only the controls an operation's
own arguments declare are now treated as wire-only.
