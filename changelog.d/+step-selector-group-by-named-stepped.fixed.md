A step selector or `group_by` that named a stepped parameter as the deck
spells it, such as `Rv`, was refused, or grouped every run under null:
both LTspice builds print the name lower-cased in the log, `.step rv=1000`
(recorded: `log/meas_when_threshold_tran_step`). A step axis now matches
in any case, as SPICE compares names.
