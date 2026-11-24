On macOS, and under Wine on Linux, a simulator is now started without the
`NoDefaultCurrentDirectoryInExePath` variable that some MCP clients set in the
environment of the servers they start. With it, LTspice 26 on macOS was
reported to finish a run with no log and no raw file. Native Windows and
WSL keep the variable. The server's own environment is not changed.
