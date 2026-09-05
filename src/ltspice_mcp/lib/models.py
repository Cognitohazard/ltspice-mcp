"""The strict Pydantic base every declared model in this project shares.

It lives in ``lib`` rather than beside the tool inputs because ``lib`` modules
declare models too (the schematic op union, for one) and the tool layer sits
above ``lib``: putting the base up there would make a core module import the
layer that imports it. ``tools/_base`` re-exports it, so
``from ltspice_mcp.tools._base import StrictModel`` keeps working.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Shared Pydantic config for all strict models (tool inputs and nested schemas)."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )
