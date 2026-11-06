`verify_circuit`'s comparison called two sources equivalent when their
value fields differed after the first token, such as `V1 a 0 5V AC 1` and
`V1 a 0 5V AC 2`, because the shared value reader read the whole field by
its leading number. A value field is now compared token by token, each
numerically within the tolerance, so `5V AC 1` still matches `5.0V AC 1`.
`parse_spice_value`, in the server and in the Python API, refuses a string
of several tokens; a suffix is written against its number (`1ms`).
