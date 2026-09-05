"""The strict Pydantic base every declared model in this project shares.

It lives in ``lib`` rather than beside the tool inputs because ``lib`` modules
declare models too (the schematic op union, for one) and the tool layer sits
above ``lib``: putting the base up there would make a core module import the
layer that imports it. ``tools/_base`` re-exports it, so
``from ltspice_mcp.tools._base import StrictModel`` keeps working.

The one schema annotation a model and the schema publisher must agree on,
``KEEP_DESCRIPTION``, lives here for the same reason.
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


# A schema node carrying this key keeps its ``description`` when the compact
# tool listing strips prose. It exists for a branch whose advertised schema is
# only its discriminant: there the description is the branch's entire content,
# so removing it would publish a branch that says nothing at all. The key lives
# here, beside the model base, because the model layer sets it and the tool
# layer's schema publisher reads it; ``strip_argument_descriptions`` consumes
# it, so it never reaches a compact listing.
KEEP_DESCRIPTION = "x-keep-description"
