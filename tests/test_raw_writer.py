"""Which simulator wrote a raw, as every reader of a raw answers it.

The full decode and the progress read of a running job both read the
``Command`` header field through ``raw_header.command_writer``, so a header
one of them names a writer by, the other names the same writer by.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib import raw_parser
from ltspice_mcp.lib.raw_header import RawDialectError, command_writer, preflight_raw
from ltspice_mcp.state import SessionState
from tests.conftest import SyncApi
from tests.test_raw_header import LIMITS, synthetic_binary, write_raw
from tests.test_simulator_families import QSPICE_COMMAND, _qspice_ac_qraw

#: Command fields, and the writer each names.
COMMANDS = {
    "Linear Technology Corporation LTspice": "ltspice",
    "Linear Technology Corporation LTspice XVII": "ltspice",
    "LTspice 24.1.9 for Windows": "ltspice",
    "ngspice-42": "ngspice",
    "ngspice-46 : Circuit level simulation program": "ngspice",
    "QSPICE": "qspice",
    "QSPICE64": "qspice",
    "QSPICE 64-bit": "qspice",
    "Xyce": "xyce",
    "Xyce Release 7.8.0-opensource": "xyce",
    "": None,
    "a simulator": None,
}


@pytest.mark.parametrize(("command", "writer"), list(COMMANDS.items()))
@pytest.mark.parametrize("width", [1, 2])
def test_the_progress_reader_names_the_writer_the_full_reader_does(
    command: str, writer: str | None, width: int
):
    assert command_writer(command) == writer
    fields = {"command": command} if command else {}
    progress = raw_parser._raw_writer(fields, width, None)
    if writer is not None:
        assert progress == writer
    else:
        # Named by nothing, the progress read falls back as it always has: a
        # UTF-16 raw is LTspice's, an ASCII one ngspice's.
        assert progress == ("ltspice" if width == 2 else "ngspice")


@pytest.mark.parametrize("command", ["LTspice and ngspice", "ngspice, Xyce"])
def test_a_command_naming_two_writers_names_none(command: str, tmp_path: Path):
    with pytest.raises(RawDialectError, match="names multiple writers"):
        command_writer(command)
    with pytest.raises(RawDialectError, match="names multiple writers"):
        preflight_raw(write_raw(tmp_path, synthetic_binary(command=command)), limits=LIMITS)
    assert raw_parser._raw_writer({"command": command}, 1, "xyce") == "xyce"


@pytest.mark.parametrize("command", ["ngspice-42", "ngspice46", "ngspice"])
def test_the_full_reader_takes_the_writer_from_command(command: str, tmp_path: Path):
    (plot,) = preflight_raw(
        write_raw(tmp_path, synthetic_binary(command=command)), limits=LIMITS
    ).plots
    assert (plot.dialect, plot.dialect_evidence) == ("ngspice", ("header:Command",))


def test_a_qraw_whose_command_names_its_build_reads_by_path(
    state_no_sim: SessionState, work_dir: Path
):
    """``Command: QSPICE64`` names QSPICE, read by path with no dialect given."""
    path = work_dir / "rc.qraw"
    path.write_bytes(_qspice_ac_qraw())
    (plot,) = preflight_raw(path, limits=LIMITS).plots
    assert (plot.dialect, plot.dialect_evidence) == ("qspice", ("header:Command",))
    assert raw_parser.raw_writer_command(path) == QSPICE_COMMAND

    raw = SyncApi(state_no_sim).load_raw(path)

    assert raw.dialect == "qspice"
    assert list(raw.axis()) == [10.0, 1_000.0, 100_000.0]
    assert raw.trace("V(out)")[1] == pytest.approx(0.5 - 0.5j)
