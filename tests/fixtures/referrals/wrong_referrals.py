"""Referrals the tools refuse, for the referral checker's own tests.

Never imported: tests/test_referrals.py reads it as text and must report
every referral below.
"""

from ltspice_mcp.lib.continuation import mcp_call

FLAT_INSPECT = "List the builds with inspect(kind='capabilities')."
JOBS_WITHOUT_ID = "Stop it with jobs(action='cancel')."
NET_WITHOUT_POINT = 'Call inspect(queries=[{"kind": "net", "path": "<your sheet>"}]).'
API_WITHOUT_JOB = "Read it with api.wait()."
BUILT_FLAT_INSPECT = mcp_call("inspect", {"kind": "capabilities"})
EITHER_WITHOUT_JOB = "Read it with jobs(action='status'|'runs')."
UNQUOTED_PLACEHOLDER = "Save it with edit_schematic(path=<this sheet>, open_window='save')."
PROSE_ARGUMENTS = 'Check it with jobs (action:"status").'
