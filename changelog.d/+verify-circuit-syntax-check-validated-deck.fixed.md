`verify_circuit`'s syntax check validated a deck against LTspice's rules
whenever the session's default simulator was anything but ngspice, so a
QSPICE or Xyce session was told its deck broke rules LTspice has and those
simulators do not. It now checks the deck with the dialect-neutral rules
only for a family with no declared validator rules, and says so in an
observation; with no simulator detected it says the deck was checked as
LTspice reads it, instead of choosing LTspice silently.
