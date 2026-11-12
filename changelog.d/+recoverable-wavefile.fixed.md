A recoverable run of a deck with a source that plays a .wav file
(`V1 a 0 wavefile=level.wav`, quoted or not) was accepted, though recovery
saves no copy of the file, so a resumed case could run against a changed one.
Such a deck is now refused before submission with `recovery_external_reader`,
naming each card, as a PWL that reads a file already was; resuming a job
recorded with one is refused the same way.
