A relative path now means the same folder in a tool call, in a `run_code`
snippet, in the Python API and in a detached job: the working directory, or
the first allowed directory when `[security] allowed_paths` leaves the working
directory out. A relative `allowed_paths` entry, such as `"."`, is taken from
the working directory. Before, a server started in one folder with
`LTSPICE_MCP_WORKING_DIR` naming another read a relative path, and placed a
relative sandbox, in the folder it was started in, so the server and
`run_code` opened different files, and the server refused the results it had
stored by their path. A relative `LTSPICE_MCP_WORKING_DIR` is taken from the
folder the server was started in; `run_code` took it from the working
directory again and worked one folder too deep. A sandbox that lists the
working directory after another folder now takes relative paths from the
working directory, not from that first folder.
