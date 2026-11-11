`run_code`'s worker has a memory limit, `[tools] run_code_memory_mb`
(default half the machine's physical memory). While a snippet runs, the
server reads the worker's memory ten times a second and stops the worker
once it holds more than the limit. The call then answers an error of type
`MemoryLimit` that names the setting, and the next call starts a fresh
worker. A worker already over the limit when a snippet arrives, from memory
an earlier snippet left it holding or a job it runs between calls, is
replaced before the snippet runs, and a limit below what a fresh worker
needs is refused with the size to set. The limit counts the worker alone,
not a simulator it starts: resident memory, or on Windows the memory it has
committed. Before, nothing but the machine's own memory bounded a snippet.
