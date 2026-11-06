The guide named the config key that sets the tool listing as
`[tools] tool_listing`, which the server does not read, and an unknown key is
dropped without a warning, so following the guide changed nothing. It now
names `[tools] listing`, the key the server reads.
