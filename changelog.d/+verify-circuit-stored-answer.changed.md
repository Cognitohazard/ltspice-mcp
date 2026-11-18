`verify_circuit` sends every finding when the reply fits the configured reply
target, instead of 25 per rule. A reply too long to send whole stores the
complete answer and says how to read the rest: the stored file's path, the
call with `findings_cursor`, or the Python call. Reading the rest no longer
runs the check again.
