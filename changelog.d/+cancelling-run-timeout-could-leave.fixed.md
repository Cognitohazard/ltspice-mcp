Cancelling a run, or its timeout, could leave its simulator running. psutil
keeps the process objects of one process scan for the next, and when the
simulator had taken the process id of a process that had exited since, the
kill found it by its command line and then skipped it as an id reused. The
kill and the check whether a case's simulator still runs now read every
process afresh.
