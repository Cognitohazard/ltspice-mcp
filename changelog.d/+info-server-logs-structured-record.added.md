At `INFO`, the server logs one structured record per tool call on the
`ltspice_mcp.events` logger: the tool, its duration, how it ended (a
result, an error and its code, a cancellation by the client, or a protocol
error), and the size of the reply as sent. Slow calls, calls a client gave
up on, and replies too large for a client show in the server's log.
Arguments are not logged.
