"""Which text is a model name: read in a deck, and written into one.

Two questions, two answers, both in ``spice_lex``. ``is_model_name`` reads a
name a user's deck declares, with the characters the lexer reads a ``.model``
card's name with; a Monte Carlo ``rule: "model"`` target names such a card and
asks it. ``written_model_name_refusal`` answers for a name the server writes:
a ``REF@model`` swap and the ``instances`` form ask it through
``instance_targeting.validate_payload``. A written name is held to what the
recordings show LTspice reading in a model name (letters, digits and
``_ . : -``, a leading digit included, as in ``2N2222``), and a text
``format.is_scaled_number`` reads whole as a number (a number with at most a
scale suffix) is
refused, since a simulator can take it for an area or a value. Each refusal
comes while the call's arguments are validated, so nothing runs. The value
path (``assign {"J1": [...]}``) writes a component value, which may be a model
name followed by an area, so it asks neither.
"""

from __future__ import annotations

import itertools
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ltspice_mcp.api import Api, ApiValidationError
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.instance_targeting import InstanceEdit, apply_edit, validate_payload
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.lib.spice_lex import (
    SpiceLexError,
    TokenKind,
    is_model_name,
    lex,
    tokenize_body,
    written_model_name_refusal,
)
from ltspice_mcp.lib.spice_lex_views import InstanceLine
from ltspice_mcp.lib.variations import AssignVariation, ModelRule
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.experiments import RunExperimentsInput
from tests import _ltspice_recorded as rec
from tests.conftest import FIXTURES_DIR, dispatch, patch_stub_bootstrap, terminal_experiment

#: Part numbers whose model names start with a digit.
PART_NUMBERS = ("2N5486", "1N4148", "2N2222")

#: Texts ``format.is_scaled_number`` reads whole as a number. On a ``.model``
#: card the lexer reads each as the model's name, but the server writes none.
NUMBER_LIKE = ("2222", "1k", "1e3", "2N")

#: Names holding a control character.
CONTROL = ("NM\x00", "NM\x07", "NM\x7f")

#: Names holding a character no recording shows in a model name, or starting
#: with one that no recording shows first.
UNRECORDED = ("N$1", "a!b", "a+b", "a/b", "a#b", "a[1]", "a|b", "\u00e41", ".nm", "-nm", ":nm")

#: Names the rule before leading digits were allowed accepted: a letter or
#: underscore first, then each character it allowed after it.
EARLIER_NAMES = ("NM", "_nm", "nch_1v8", "BSIM4.0", "lib:nch", "nch-lvt", "Q2N3904")

#: Texts that are a model name nowhere: what the read refusal is for.
NOT_NAMES = ("", "2N 5486", 'NM"x', "NM'x", "NM,x", "NM=x", "NM;x", "{NM}", "NM(x)", "$")

#: What every refusal says to do.
FIX = ".model card declares it"

#: The reason each kind of written name is refused for.
AS_NUMBER = "reads whole as a number"
AS_CONTROL = "a control character"
AS_UNRECORDED = "no recording shows LTspice reading"

SWAP_FORMS = ("ref_at_model", "instances")


def _forms(name: Any) -> dict[str, dict[str, Any]]:
    """One variation per form that names a model, swapping J1 (or drawing on
    the model) to ``name``."""
    return {
        "ref_at_model": {"kind": "assign", "assign": {"J1@model": [name]}},
        "instances": {
            "kind": "assign",
            "instances": [{"instance": ["J1"], "attribute": "model", "values": [name]}],
        },
        "monte_carlo_model_rule": {
            "kind": "random",
            "runs": 1,
            "seed": 1,
            "rules": [{"rule": "model", "target": name, "param": "VTO", "tolerance": 0.1}],
        },
    }


def _validate(form: str, name: Any) -> None:
    RunExperimentsInput.model_validate(
        {"circuits": [{"path": "dut.cir"}], "variations": [_forms(name)[form]]}
    )


# --- the two definitions -------------------------------------------------


@pytest.mark.parametrize("name", PART_NUMBERS + NUMBER_LIKE + CONTROL + UNRECORDED + EARLIER_NAMES)
def test_the_lexer_still_reads_each_name_on_a_model_card_of_an_existing_deck(name: str):
    (card,) = [c for c in lex(f".model {name} NJF(Beta=1m)\n").cards if c.kind == "model"]
    assert card.name == name
    assert is_model_name(name)


@pytest.mark.parametrize("name", NUMBER_LIKE + CONTROL + UNRECORDED)
def test_a_name_the_lexer_reads_is_not_always_one_the_server_writes(name: str):
    # The two definitions differ on purpose: reading must take whatever a
    # user's deck declares, writing only what a recording shows LTspice read.
    assert is_model_name(name)
    assert written_model_name_refusal(name) is not None


def _earlier_rule_accepts(text: str) -> bool:
    """The rule ``validate_payload("model", ...)`` held a written name to before
    leading digits were allowed."""
    if not text.strip() or any(c in text for c in "\r\n;$"):
        return False
    try:
        tokens = tokenize_body(text)
    except SpiceLexError:
        return False
    return (
        len(tokens) == 1
        and tokens[0].kind == TokenKind.BARE
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.$:-]*", text) is not None
    )


def test_a_written_name_the_earlier_rule_refused_is_refused_unless_it_only_starts_with_a_digit():
    alphabet = "aZ_0.:-$+ \"'\x00e"
    for size in range(1, 4):
        for chars in itertools.product(alphabet, repeat=size):
            text = "".join(chars)
            accepted = written_model_name_refusal(text) is None
            if _earlier_rule_accepts(text):
                assert accepted, text
            elif accepted:
                assert text[0].isdigit() and _earlier_rule_accepts("a" + text[1:]), text


@pytest.mark.parametrize("text", NOT_NAMES)
def test_a_text_that_is_no_model_name_is_refused_by_both_definitions(text: str):
    assert not is_model_name(text)
    with pytest.raises(ValueError, match=FIX):
        validate_payload("model", None, text)


@pytest.mark.parametrize("build", ["ltspice26", "ltspice17"])
def test_the_recorded_export_names_2n2222_a_model_by_both_definitions(build: str):
    path = FIXTURES_DIR / "ltspice_recorded" / build / "export" / "attributes.net"
    cards = {
        card.name: card for card in lex(path.read_text(encoding="utf-8", errors="replace")).cards
    }
    for ref in ("Q1", "Q2"):
        model = InstanceLine.from_card(cards[ref]).model
        assert model == "2N2222"
        assert is_model_name(model)
        assert validate_payload("model", None, model) == "2N2222"


# --- each caller, at input ---------------------------------------------------


@pytest.mark.parametrize("name", PART_NUMBERS + EARLIER_NAMES)
def test_every_form_accepts_a_model_name_at_input(name: str):
    forms = _forms(name)
    AssignVariation.model_validate(forms["ref_at_model"])
    AssignVariation.model_validate(forms["instances"])
    ModelRule.model_validate(forms["monte_carlo_model_rule"]["rules"][0])


@pytest.mark.parametrize(
    ("names", "reason"),
    [(NUMBER_LIKE, AS_NUMBER), (CONTROL, AS_CONTROL), (UNRECORDED, AS_UNRECORDED)],
    ids=["number_like_by_the_lexers_number_reader", "control", "unrecorded"],
)
@pytest.mark.parametrize("form", SWAP_FORMS)
def test_every_swap_form_refuses_a_name_it_must_not_write_at_input(
    form: str, names: tuple[str, ...], reason: str
):
    for name in names:
        with pytest.raises(ValidationError, match=reason):
            _validate(form, name)


@pytest.mark.parametrize("name", NUMBER_LIKE + UNRECORDED)
def test_a_monte_carlo_model_rule_reads_a_name_the_deck_declares_however_written(name: str):
    # The rule writes no name: it draws on a card the deck already declares.
    ModelRule.model_validate(_forms(name)["monte_carlo_model_rule"]["rules"][0])


def test_an_integer_swap_value_reads_as_a_number_and_is_refused():
    with pytest.raises(ValidationError, match=AS_NUMBER):
        AssignVariation.model_validate({"kind": "assign", "assign": {"J1@model": [2222]}})


@pytest.mark.parametrize("text", NOT_NAMES)
@pytest.mark.parametrize("form", [*SWAP_FORMS, "monte_carlo_model_rule"])
def test_every_form_refuses_a_text_that_is_no_model_name_at_input(form: str, text: str):
    with pytest.raises(ValidationError, match=FIX):
        _validate(form, text)


@pytest.mark.parametrize("name", PART_NUMBERS + EARLIER_NAMES)
def test_the_write_claim_rewrites_the_instance_to_the_swapped_name(name: str):
    # The route a REF@model swap takes after expansion: apply_edit on a copy
    # of the card, then the written card is read back as an instance.
    (card,) = [c for c in lex("J1 d g s 2N5485\n").cards if c.kind == "instance"]
    apply_edit(card, InstanceEdit(("J1",), "model", name))
    assert InstanceLine.from_card(card).model == name


# --- end to end, through the tool and the Python API, on ngspice -------------

_needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

#: V(out) of the source follower below: the drain current
#: Beta*(Vgs-Vto)^2 = V(out)/1k with Vgs = -V(out), solved for each model.
VOUT_2N5485 = (6.04 - 11.08**0.5) / 1.8  # Beta=0.9m, Vto=-2.8: 1.5063
VOUT_SWAPPED = (9 - 17**0.5) / 8  # Beta=4m, Vto=-1: 0.6096


def _follower(work_dir: Path, swapped: str) -> str:
    """A JFET follower on model 2N5485, which declares ``swapped`` beside it;
    the two models differ in Beta and Vto, so V(out) says which one ran. The
    file name holds no model name, because a model name may hold a character
    (``:``) that no Windows file name can."""
    name = "follower.cir"
    (work_dir / name).write_text(
        "* JFET source follower\n"
        "V1 vdd 0 5\n"
        "V2 in 0 0\n"
        "J1 vdd in out 2N5485\n"
        "R1 out 0 1k\n"
        ".model 2N5485 NJF(Beta=0.9m Vto=-2.8)\n"
        f".model {swapped} NJF(Beta=4m Vto=-1)\n"
        ".dc V2 0 0 1\n"
        ".end\n"
    )
    return name


def _payload(netlist: str, variation: dict[str, Any]) -> dict[str, Any]:
    return {
        "circuits": [{"path": netlist, "id": "dut"}],
        "variations": [variation],
        "execution": {"wait_s": 90, "simulator": "ngspice"},
        "analyze": {
            "recipes": [{"key": "vout", "metric": "value", "expr": "v(out)", "at": "0"}],
            "include": {"per_run": {"limit": 5}},
        },
    }


def _raw_files(work_dir: Path) -> list[Path]:
    """Every result file a run under ``work_dir`` wrote."""
    return list(work_dir.rglob("*.raw"))


def _vout(receipt: dict[str, Any]) -> float:
    assert receipt["status"] == "completed", receipt
    (row,) = receipt["analysis"]["result"]["results"]["vout"]["per_run"]["items"]
    return row["value"]["value"]


@pytest.fixture
def ngspice_state(work_dir: Path) -> SessionState:
    config = ServerConfig(simulator="ngspice", working_dir=work_dir, allowed_paths=[work_dir])
    available = detect_simulators(config)
    if "ngspice" not in available:
        pytest.fail("ngspice is on PATH but detect_simulators did not report it usable")
    return SessionState.create(config, available)


_SWAP_FORMS = {
    "ref_at_model": lambda name: {"kind": "assign", "assign": {"J1@model": [name]}},
    "instances": lambda name: {
        "kind": "assign",
        "instances": [{"instance": ["J1"], "attribute": "model", "values": [name]}],
    },
    "value_path": lambda name: {"kind": "assign", "assign": {"J1": [name]}},
}


def _drawn_deck(work_dir: Path, name: str) -> str:
    """The follower on model ``name`` alone, for a Monte Carlo draw on it."""
    netlist = "drawn.cir"
    (work_dir / netlist).write_text(
        "* JFET source follower\nV1 vdd 0 5\nV2 in 0 0\n"
        f"J1 vdd in out {name}\nR1 out 0 1k\n"
        f".model {name} NJF(Beta=4m Vto=-1)\n.dc V2 0 0 1\n.end\n"
    )
    return netlist


def _assert_drawn(receipt: dict[str, Any], name: str) -> None:
    (row,) = receipt["analysis"]["result"]["results"]["vout"]["per_run"]["items"]
    assert any(key.endswith(f"{name}.VTO") for key in row["assignments"]), row
    assert _vout(receipt) != pytest.approx(VOUT_SWAPPED, rel=1e-6)


@pytest.fixture
def api(ngspice_state: SessionState, monkeypatch: pytest.MonkeyPatch):
    """A real ``Api`` on its own loop thread, over the ngspice session."""
    patch_stub_bootstrap(monkeypatch, ngspice_state)
    with Api() as instance:
        yield instance


@_needs_ngspice
@pytest.mark.asyncio
@pytest.mark.parametrize("form", list(_SWAP_FORMS))
@pytest.mark.parametrize("name", PART_NUMBERS + EARLIER_NAMES)
async def test_a_swap_to_a_part_number_or_a_name_the_earlier_rule_accepts_runs_it_through_the_tool(
    ngspice_state: SessionState, work_dir: Path, name: str, form: str
):
    payload = _payload(_follower(work_dir, name), _SWAP_FORMS[form](name))
    receipt = await terminal_experiment(ngspice_state, payload)
    assert _vout(receipt) == pytest.approx(VOUT_SWAPPED, rel=1e-3)
    assert abs(VOUT_SWAPPED - VOUT_2N5485) > 0.5


@_needs_ngspice
@pytest.mark.parametrize("form", list(_SWAP_FORMS))
@pytest.mark.parametrize("name", PART_NUMBERS + EARLIER_NAMES)
def test_a_swap_to_a_part_number_or_a_name_the_earlier_rule_accepts_runs_it_through_the_api(
    api: Api, work_dir: Path, name: str, form: str
):
    payload = _payload(_follower(work_dir, name), _SWAP_FORMS[form](name))
    assert _vout(api.run_experiments(**payload)) == pytest.approx(VOUT_SWAPPED, rel=1e-3)


@_needs_ngspice
@pytest.mark.asyncio
@pytest.mark.parametrize("name", PART_NUMBERS)
async def test_a_monte_carlo_model_rule_draws_on_a_part_number_through_the_tool(
    ngspice_state: SessionState, work_dir: Path, name: str
):
    payload = _payload(_drawn_deck(work_dir, name), _forms(name)["monte_carlo_model_rule"])
    _assert_drawn(await terminal_experiment(ngspice_state, payload), name)


@_needs_ngspice
@pytest.mark.parametrize("name", PART_NUMBERS)
def test_a_monte_carlo_model_rule_draws_on_a_part_number_through_the_api(
    api: Api, work_dir: Path, name: str
):
    payload = _payload(_drawn_deck(work_dir, name), _forms(name)["monte_carlo_model_rule"])
    _assert_drawn(api.run_experiments(**payload), name)


_REFUSED = ["", "2N 5486", 'NM"x']
_REFUSING_FORMS = ["ref_at_model", "instances", "monte_carlo_model_rule"]


@_needs_ngspice
@pytest.mark.asyncio
@pytest.mark.parametrize("text", _REFUSED)
@pytest.mark.parametrize("form", _REFUSING_FORMS)
async def test_a_text_that_is_no_model_name_is_refused_by_the_tool_before_a_run(
    ngspice_state: SessionState, work_dir: Path, form: str, text: str
):
    payload = _payload(_follower(work_dir, "2N5486"), _forms(text)[form])
    result = await dispatch(ngspice_state, "run_experiments", payload)
    assert result.is_error
    assert FIX in str(result.structured_content), result.structured_content
    assert "job_id" not in (result.structured_content or {})
    assert not _raw_files(work_dir)


@_needs_ngspice
@pytest.mark.parametrize("text", _REFUSED)
@pytest.mark.parametrize("form", _REFUSING_FORMS)
def test_a_text_that_is_no_model_name_is_refused_by_the_api_before_a_run(
    api: Api, work_dir: Path, form: str, text: str
):
    payload = _payload(_follower(work_dir, "2N5486"), _forms(text)[form])
    with pytest.raises(ApiValidationError, match=FIX):
        api.run_experiments(**payload)
    assert not _raw_files(work_dir)


#: One written name of each refused kind, with the reason it is refused for.
_UNWRITTEN = [
    ("2222", AS_NUMBER),
    ("1k", AS_NUMBER),
    ("1e3", AS_NUMBER),
    ("NM\x01", AS_CONTROL),
    ("N$1", AS_UNRECORDED),
]


@_needs_ngspice
@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "reason"), _UNWRITTEN)
@pytest.mark.parametrize("form", SWAP_FORMS)
async def test_a_swap_to_a_name_it_must_not_write_is_refused_by_the_tool_before_a_run(
    ngspice_state: SessionState, work_dir: Path, form: str, text: str, reason: str
):
    payload = _payload(_follower(work_dir, "2N5486"), _forms(text)[form])
    result = await dispatch(ngspice_state, "run_experiments", payload)
    assert result.is_error
    assert reason in str(result.structured_content), result.structured_content
    assert "job_id" not in (result.structured_content or {})
    assert not _raw_files(work_dir)


@_needs_ngspice
@pytest.mark.parametrize(("text", "reason"), _UNWRITTEN)
@pytest.mark.parametrize("form", SWAP_FORMS)
def test_a_swap_to_a_name_it_must_not_write_is_refused_by_the_api_before_a_run(
    api: Api, work_dir: Path, form: str, text: str, reason: str
):
    payload = _payload(_follower(work_dir, "2N5486"), _forms(text)[form])
    with pytest.raises(ApiValidationError, match=re.escape(reason)):
        api.run_experiments(**payload)
    assert not _raw_files(work_dir)


@_needs_ngspice
@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["2N 5486", 'NM"x'])
async def test_the_value_path_accepts_a_component_value_at_input_and_fails_the_case_it_cannot_write(
    ngspice_state: SessionState, work_dir: Path, text: str
):
    # ``assign {"J1": [...]}`` writes a component value, which may be a model
    # name and an area ("2N5486 2"), so it does not ask whether its text is a
    # model name: the call is accepted and the case fails, with nothing run.
    payload = _payload(_follower(work_dir, "2N5486"), {"kind": "assign", "assign": {"J1": [text]}})
    receipt = await terminal_experiment(ngspice_state, payload)
    (run,) = receipt["runs"]["items"]
    assert run["status"] == "failed"
    assert receipt["completeness"]["produced"] == 0
    assert not _raw_files(work_dir)


def test_the_value_path_refuses_an_empty_value_at_input():
    with pytest.raises(ValidationError, match="cannot be empty"):
        AssignVariation.model_validate({"kind": "assign", "assign": {"J1": [""]}})


# --- what LTspice reads -------------------------------------------------------


@pytest.mark.parametrize("build", rec.BUILDS)
def test_ltspice_runs_the_model_a_leading_digit_name_swaps_to(build: str, tmp_path: Path):
    """Both builds run the model a swap to 2N5486 names, so the server
    writes that name and reads it on a card."""
    assert written_model_name_refusal("2N5486") is None
    assert is_model_name("2N5486")
    raw = rec.decode(build, "deck/model_name_leading_digit", tmp_path).raw
    assert float(raw.get_wave("V(out)", 0)[0]) == pytest.approx(VOUT_SWAPPED, rel=1e-3)
