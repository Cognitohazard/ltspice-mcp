With LTspice XVII and a later LTspice both installed, a server running XVII
read the later build's symbols when it edited and checked sheets. On
Windows and under Wine it searched them before XVII's own, and on WSL it
used them instead. On WSL, with XVII the only build installed, it found no
symbols at all. It now reads the symbols of the build it runs, as it
already did for model libraries.
