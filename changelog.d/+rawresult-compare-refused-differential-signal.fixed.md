`RawResult.compare(signals=...)` refused a differential signal that
`trace()` reads, such as `V(a,b)`, as not a signal of both results. It now
resolves each name as `trace()` does.
