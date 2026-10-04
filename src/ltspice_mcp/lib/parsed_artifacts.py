"""One immutable captured result shared by RAW and log consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ltspice_mcp.lib.decoded_log import DecodedLog
    from ltspice_mcp.lib.decoded_raw import DecodedRaw


@dataclass(frozen=True)
class ParsedArtifacts:
    snapshot_id: str
    raw: DecodedRaw | None
    logs: DecodedLog
