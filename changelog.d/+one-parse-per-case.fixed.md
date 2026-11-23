An `analyze_results` call with many recipes over more than 32 cases now reads
each case's files once. Each recipe used to read every case again, so a call
with twelve `measurements` recipes over 40 logs parsed some 600 times instead
of 40. The session keeps up to 256 parsed cases within the same memory budget,
and a call with a `measurements` recipe reads a results file LTspice may have
saved compressed together with its log in one read.
