**Breaking:** a shutdown, the server's or the Python API's `close()`, stops
the cases it runs as failed with `server_restarted`, as a crash leaves them,
so a default resume retries them; it left them cancelled, which only
`retry_cancelled` retried. `server_shutdown` is no longer written, and a
default resume also retries a case recorded with it.
