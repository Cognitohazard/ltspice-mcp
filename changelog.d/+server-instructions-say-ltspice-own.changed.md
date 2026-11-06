The server's instructions say that LTspice's own MCP server
(`set_design_content`, `start_simulation`) edits and runs only the copy
open in its window, and to edit sheets with `edit_schematic` and run them
with `run_experiments`. A client that connects both servers met that rule
only in the guide, while LTspice's server's own instructions send sheet
edits to `set_design_content`.
