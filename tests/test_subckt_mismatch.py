"""The level a mismatch draw inside a subcircuit gates on.

The draw is made on the hierarchy the deck's simulator reads
(``variations._nested_mismatch_edits``); on ngspice it writes ``delvto`` and
``mulu0`` on the device's card, which only a BSIM3 or BSIM4 model takes
(``subckt_mismatch.require_instance_parameter``). Fixtures are shaped like a foundry
model deck (``tests._mismatch_fixtures``) in the BINNED model spelling a real
PDK ships, but self-contained, so the suite runs with no PDK installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.spice_lex import lex
from ltspice_mcp.lib.spice_lex_views import ModelCard
from ltspice_mcp.lib.subckt_mismatch import MismatchPlanError, require_instance_parameter
from tests._mismatch_fixtures import BINNED_MINI_FET as MINI_FET

DECK = f"""\
* mini pdk-shaped deck
{MINI_FET}
XN0 dn0 g 0 0 minifet W=1 L=0.15
XN1 dn1 g 0 0 minifet W=1 L=0.15
V1 dn0 0 0.9
Vg g 0 0.9
.op
.end
"""


#: Ways a ``.model`` card gives its LEVEL, with the level each declares. A
#: comment is no part of the card, though the card is continued past it.
_LEVEL_SPELLINGS = [
    pytest.param("level = 54 vth0 = 0.4\n", 54.0, id="on-the-first-line"),
    pytest.param(
        "\n* level=1 was the old model\n+ level = 54 vth0 = 0.4\n",
        54.0,
        id="after-a-comment-line",
    ),
    pytest.param("level = 54 vth0 = 0.4 ; not level=1\n", 54.0, id="beside-a-comment"),
    pytest.param("(level=54.0 vth0=0.4)\n", 54.0, id="as-a-decimal"),
    pytest.param("level = {54} vth0 = 0.4\n", None, id="braced"),
]


@pytest.mark.parametrize(("spelling", "level"), _LEVEL_SPELLINGS)
def test_every_reader_of_a_model_level_reads_one_level(
    tmp_path: Path, spelling: str, level: float | None
):
    """The model card's view, the hierarchy and the mismatch draw's BSIM
    gate each read the level the card declares, and the same one."""
    deck = DECK.replace(
        ".model minifet_model.0 nmos level = 54 vth0 = 0.7 u0 = 0.06\n",
        ".model minifet_model nmos " + spelling,
    )
    (card,) = [card for card in lex(deck).cards if card.kind == "model"]
    assert ModelCard.from_card(card).level == level

    path = tmp_path / "deck.cir"
    path.write_text(deck, encoding="utf-8")
    hierarchy = load_hierarchy(str(path), [tmp_path], SemanticProfile("ngspice", "hsa"))
    rows = [row for row in hierarchy.instances if row.element == "M"]
    assert {tuple(m.level for m in row.model_family) for row in rows} == {(level,)}

    for row in rows:
        if level is None:
            with pytest.raises(MismatchPlanError) as exc:
                require_instance_parameter(row, "ngspice", "delvto")
            assert exc.value.code == "unsupported_model"
        else:
            require_instance_parameter(row, "ngspice", "delvto")
