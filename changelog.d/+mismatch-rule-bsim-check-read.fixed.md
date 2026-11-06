The `mismatch` rule's BSIM check read a model's `LEVEL` from the card's raw
lines, so a full-line comment such as `* level=1 was the old model` between
its `+` lines decided the level and a BSIM device was refused; `level=54.0`
read as no level. The level is now read from the card's own `LEVEL=`.
