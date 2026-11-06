The `run_code` tool's title is now "Run Python Code", and its description, the
guide and the Python API's messages no longer call the server's code an
"engine". An `Api` opened while another is active now raises
`ApiSessionError("An API session is already active in this process")`, and a
failed operation reports "The operation returned a call-level error".
