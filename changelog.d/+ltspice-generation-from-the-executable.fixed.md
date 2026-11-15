An LTspice XVII under a name other than `XVIIx64.exe` or `XVIIx86.exe` (an
`LTspice.exe`, `spice17.exe`, a renamed copy) is now known as XVII before any
run on it, from the file version in its own version resource, which the
server reads from the file on Windows, under Wine and on WSL alike. Before,
it was taken for LTspice 26 until a run reported its build: the server added
`.options logopinfo` to every `.op` deck run on it, XVII ended each such run
with `unrecognized option`, and since that run reported no build every `.op`
run kept failing; its names were spelled, and its log read, as LTspice 26
writes them, and an XVII named `LTspice.exe` staged against LTspice 26's
library. When nothing about the executable shows which LTspice it is (a
wrapper script, a copy without its version resource), the server no longer
assumes LTspice 26: an `.op` run goes without the option and says so in
`observations` (`logopinfo_withheld`), the names note that the build is not
known, and both builds' libraries are kept, on WSL too.
