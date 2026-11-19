"""A job names the simulator it ran with in the words a caller can pass back.

The question: which simulator did this job run with, as the selector
``execution.simulator`` takes (``ltspice``, or ``ltspice:xvii`` for an entry of
``[simulator.executables]``). A job records the selector its run resolved to,
the default's when the request named none, and every reply that names the job
gives it once, as ``simulator``: the receipt, ``jobs`` status, wait, resume and
runs, each row of ``jobs(action="list")``, and the same replies through the
Python API. Passed back in the same session, it runs the same executable.

A record written before the selector was recorded replies with what it
establishes: the family from its class name, the name of a configured build
whose executable is the one it recorded, else the family and a note that the
build's name was not recorded. A session with no simulator under the selector
says so beside it.

The builds are the stand-in programs of ``tests/test_named_executables.py``.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest
from spicelib.simulators.ltspice_simulator import LTspice

from ltspice_mcp.api import Api
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import store
from ltspice_mcp.lib.ltspice_wsl import LTspiceWSL
from ltspice_mcp.lib.simulator import SIMULATORS
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import resolve_run_simulator
from ltspice_mcp.tools.jobs import JobsInput, handle_jobs
from tests import test_named_executables as named_executables
from tests._source_scan import PACKAGE, modules, scopes
from tests.conftest import (
    SyncApi,
    capabilities_report,
    patch_stub_bootstrap,
    terminal_experiment,
)
from tests.test_named_executables import _payload, _record, _state

# The builds, the deck and the recording classes of the named-executable tests.
builds = named_executables.builds
deck = named_executables.deck
recording_families = named_executables.recording_families

#: What a run's request named, and the selector its job names.
REQUESTS: list[tuple[str | None, str]] = [
    (None, "ltspice"),
    ("ltspice", "ltspice"),
    ("ltspice:xvii", "ltspice:xvii"),
]


def _request(deck: Path, request_id: str, simulator: str | None) -> dict[str, Any]:
    payload = _payload(deck, request_id, simulator or "ltspice")
    if simulator is None:
        del payload["execution"]["simulator"]
    return payload


async def _jobs(state: SessionState, **arguments: Any) -> dict[str, Any]:
    result = await handle_jobs(JobsInput.model_validate(arguments), state)
    data = result.structured_content
    assert data is not None, result.content
    return data


def _list_rows(reply: dict[str, Any], job_id: str) -> list[dict[str, Any]]:
    return [
        row for group in reply["items"] for row in group["recent_jobs"] if row["job_id"] == job_id
    ]


async def _tool_replies(state: SessionState, job_id: str, deck: Path) -> dict[str, dict[str, Any]]:
    """Every ``jobs`` reply that names ``job_id``, through the tool."""
    status = await _jobs(state, action="status", job_id=job_id)
    wait = await _jobs(state, action="wait", job_id=job_id, timeout_s=30)
    runs = await _jobs(state, action="runs", job_id=job_id)
    (row,) = _list_rows(await _jobs(state, action="list", circuit=str(deck)), job_id)
    return {"status": status, "wait": wait, "runs": runs, "list": row}


def _api_replies(state: SessionState, job_id: str, deck: Path) -> dict[str, dict[str, Any]]:
    """The same replies through the Python API."""
    api = SyncApi(state)
    (row,) = _list_rows(api.jobs(action="list", circuit=str(deck)), job_id)
    return {
        "status": api.jobs(action="status", job_id=job_id),
        "wait": api.wait(job_id, timeout_s=30),
        "runs": api.jobs(action="runs", job_id=job_id),
        "list": row,
    }


def _names_once(reply: dict[str, Any], selector: str) -> None:
    """``reply`` names the job's simulator at its own level, and nothing under
    it repeats a simulator."""
    assert reply["simulator"] == selector, reply
    for source in reply.get("source", []):
        assert "simulator" not in source, source
    runs = reply.get("runs", {}).get("items", []) if "runs" in reply else []
    for row in [*runs, *reply.get("items", [])]:
        assert "simulator" not in row, row


# ---------------------------------------------------------------------------
# One answer: every selector a session offers comes back as itself
# ---------------------------------------------------------------------------


class TestOneAnswer:
    def test_every_selector_a_session_offers_resolves_and_comes_back_as_itself(
        self, config: ServerConfig, builds: dict[str, Path], recording_families: None
    ):
        from ltspice_mcp.lib.simulator import simulator_selector

        state = _state(config, builds)
        offered = state.offered_simulators
        assert set(offered) == {"ltspice", "ltspice:xvii", "ltspice:lt24"}

        for selector in offered:
            resolved = resolve_run_simulator(selector, state)
            assert simulator_selector(resolved, offered) == selector
        # No selector runs the default, which is named by its own.
        assert simulator_selector(resolve_run_simulator(None, state), offered) == "ltspice"

    @pytest.mark.parametrize("base", [LTspice, LTspiceWSL, SIMULATORS["ltspice"]])
    def test_the_selector_does_not_depend_on_the_platform_class(
        self, base: type, builds: dict[str, Path]
    ):
        """Windows, WSL and Wine each hold a class of their own for LTspice;
        the same configuration names it the same way on each."""
        from ltspice_mcp.lib.simulator import bind_named_executable, simulator_selector

        bound = bind_named_executable(base, "ltspice:xvii", builds["xvii"])
        assert bound.__name__ == base.__name__
        assert simulator_selector(bound) == "ltspice:xvii"
        assert simulator_selector(base) == "ltspice"
        assert simulator_selector(base, {"ltspice": base}) == "ltspice"

    async def test_capabilities_names_the_default_by_its_selector(
        self, config: ServerConfig, builds: dict[str, Path], recording_families: None
    ):
        state = _state(config, builds)
        caps = await capabilities_report(state)
        # The class here is RecordingLTspice: the name a run passes is not it.
        assert caps["default_simulator"] == "ltspice"
        assert caps["default_simulator"] in caps["simulators"]


# ---------------------------------------------------------------------------
# Every route names it, and the value passed back runs the same executable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEveryRoute:
    @pytest.mark.parametrize(("requested", "selector"), REQUESTS)
    async def test_the_tool_replies_name_the_selector_and_it_runs_the_same_executable(
        self,
        config: ServerConfig,
        deck: Path,
        builds: dict[str, Path],
        recording_families: None,
        requested: str | None,
        selector: str,
    ):
        state = _state(config, builds)
        receipt = await terminal_experiment(state, _request(deck, "first", requested))
        assert receipt["status"] == "completed", receipt
        _names_once(receipt, selector)
        record = await _record(state, receipt["job_id"])
        assert record["simulator_selector"] == selector

        for route, reply in (await _tool_replies(state, receipt["job_id"], deck)).items():
            assert "simulator_notes" not in reply, (route, reply)
            _names_once(reply, selector)

        again = await terminal_experiment(
            state, _payload(deck, "passed-back", receipt["simulator"])
        )
        assert again["status"] == "completed", again
        assert again["simulator"] == selector
        rerun = await _record(state, again["job_id"])
        assert rerun["simulator_executable"] == record["simulator_executable"]

    async def test_the_reply_does_not_depend_on_the_class_name_recorded(
        self,
        config: ServerConfig,
        deck: Path,
        builds: dict[str, Path],
        recording_families: None,
    ):
        """A record holds the platform's class name (``LTspice``,
        ``LTspiceWSL``); the reply holds the selector, whichever it is."""
        state = _state(config, builds)
        receipt = await terminal_experiment(state, _payload(deck, "platform", "ltspice:xvii"))
        path = store.Store(state.working_dir).job_record(receipt["job_id"])
        for class_name in ("LTspice", "LTspiceWSL"):
            data = json.loads(path.read_text())
            data["simulator"] = class_name
            path.write_text(json.dumps(data))
            fresh = _state(config, builds)
            status = await _jobs(fresh, action="status", job_id=receipt["job_id"])
            assert status["simulator"] == "ltspice:xvii", (class_name, status)


# ---------------------------------------------------------------------------
# The same replies through the Python API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("requested", "selector"), REQUESTS)
def test_the_python_api_replies_name_the_selector_and_it_runs_the_same_executable(
    config: ServerConfig,
    deck: Path,
    builds: dict[str, Path],
    recording_families: None,
    monkeypatch: pytest.MonkeyPatch,
    requested: str | None,
    selector: str,
):
    """``Api`` on its own loop, as a script holds it: the receipt and every
    ``jobs`` reply name the selector, and passing it back runs the same build."""
    state = _state(config, builds)
    patch_stub_bootstrap(monkeypatch, state)
    api = Api()
    try:
        receipt = api.run_experiments(**_request(deck, "by-api", requested))
        assert receipt["status"] == "completed", receipt
        _names_once(receipt, selector)
        (row,) = _list_rows(api.jobs(action="list", circuit=str(deck)), receipt["job_id"])
        replies = {
            "status": api.jobs(action="status", job_id=receipt["job_id"]),
            "wait": api.wait(receipt["job_id"], timeout_s=30),
            "runs": api.jobs(action="runs", job_id=receipt["job_id"]),
            "list": row,
        }
        for route, reply in replies.items():
            assert "simulator_notes" not in reply, (route, reply.get("simulator_notes"))
            _names_once(reply, selector)

        again = api.run_experiments(**_payload(deck, "by-api-again", receipt["simulator"]))
        assert again["simulator"] == selector
        first, rerun = (
            json.loads(store.Store(state.working_dir).job_record(job).read_text())
            for job in (receipt["job_id"], again["job_id"])
        )
        assert rerun["simulator_executable"] == first["simulator_executable"]
    finally:
        api.close()


# ---------------------------------------------------------------------------
# A record written before the selector was recorded
# ---------------------------------------------------------------------------


def _forget_selector(state: SessionState, job_id: str, **changes: Any) -> None:
    """Rewrite ``job_id``'s record as an earlier version wrote it: with no
    selector, and with ``changes`` to its executable identity."""
    path = store.Store(state.working_dir).job_record(job_id)
    data = json.loads(path.read_text())
    data.pop("simulator_selector", None)
    data["simulator_executable"].update(changes)
    path.write_text(json.dumps(data))


@pytest.mark.asyncio
class TestEarlierRecords:
    @pytest.mark.parametrize("simulator", ["ltspice", "ltspice:xvii"])
    async def test_a_record_whose_executable_is_a_configured_build_is_named_by_it(
        self,
        config: ServerConfig,
        deck: Path,
        builds: dict[str, Path],
        recording_families: None,
        simulator: str,
    ):
        state = _state(config, builds)
        receipt = await terminal_experiment(state, _payload(deck, "earlier", simulator))
        await state.job_registry.drain_pending()
        _forget_selector(state, receipt["job_id"])

        fresh = _state(config, builds)
        tool = await _tool_replies(fresh, receipt["job_id"], deck)
        api = await asyncio.to_thread(
            _api_replies, _state(config, builds), receipt["job_id"], deck
        )
        for route, reply in [*tool.items(), *api.items()]:
            assert "simulator_notes" not in reply, (route, reply.get("simulator_notes"))
            _names_once(reply, simulator)

    async def test_a_record_whose_executable_is_none_of_the_sessions_gives_the_family(
        self,
        config: ServerConfig,
        deck: Path,
        builds: dict[str, Path],
        recording_families: None,
    ):
        state = _state(config, builds)
        receipt = await terminal_experiment(state, _payload(deck, "elsewhere", "ltspice:xvii"))
        await state.job_registry.drain_pending()
        gone = str(builds["xvii"].parent / "moved" / "XVIIx64.exe")
        _forget_selector(state, receipt["job_id"], path=gone)

        fresh = _state(config, builds)
        tool = await _tool_replies(fresh, receipt["job_id"], deck)
        api = await asyncio.to_thread(
            _api_replies, _state(config, builds), receipt["job_id"], deck
        )
        for route, reply in [*tool.items(), *api.items()]:
            _names_once(reply, "ltspice")
            (note,) = reply["simulator_notes"]
            assert note["code"] == "simulator_build_not_recorded", (route, note)
            assert note["executable"] == gone
            assert "not recorded" in note["detail"]


# ---------------------------------------------------------------------------
# A selector the session answering does not offer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSelectorNotInSession:
    async def test_a_selector_the_session_does_not_offer_is_said(
        self,
        config: ServerConfig,
        deck: Path,
        builds: dict[str, Path],
        recording_families: None,
    ):
        state = _state(config, builds)
        receipt = await terminal_experiment(state, _payload(deck, "dropped", "ltspice:xvii"))
        await state.job_registry.drain_pending()

        # A later session whose configuration no longer names that build.
        tool = await _tool_replies(
            _state(config, builds, {"lt24": builds["lt24"]}), receipt["job_id"], deck
        )
        api = await asyncio.to_thread(
            _api_replies,
            _state(config, builds, {"lt24": builds["lt24"]}),
            receipt["job_id"],
            deck,
        )
        for route, reply in [*tool.items(), *api.items()]:
            _names_once(reply, "ltspice:xvii")
            (note,) = reply["simulator_notes"]
            assert note["code"] == "simulator_not_in_session", (route, note)
            assert "ltspice:xvii" in note["detail"]


# ---------------------------------------------------------------------------
# The guard: no reply builder names a simulator any other way
# ---------------------------------------------------------------------------

#: The modules whose code builds replies.
_REPLY_BUILDERS = ("tools/", "api/", "resources.py")

#: An expression naming a simulator class, whose ``__name__`` is its class name.
_SIMULATOR_CLASS = re.compile(r"(^|\.)(simulator|sim_cls|cls|default_simulator)$")

#: A job record, whose ``simulator`` is a class name.
_JOB_RECORD = re.compile(r"(^|\.)(job|experiment|parent|child|record|source)$")

#: Each use the scan finds that puts no class name into a reply, with why.
_ALLOWED: dict[tuple[str, str, str], str] = {
    ("tools/experiments.py", "handle_run_experiments", "simulator.__name__"): (
        "the request fingerprint and the job record's class name, which resume reads"
    ),
    ("tools/experiments.py", "_prepare_circuit", "simulator.__name__"): (
        "the source record's class name, kept on the record and left out of replies"
    ),
    ("tools/inspect_tools.py", "_build_facts", "cls.__name__"): (
        "chooses the dialect a family's raw is read in"
    ),
    ("tools/inspect_tools.py", "_do_capabilities", "cls.__name__"): (
        "chooses the dialect a family's raw is read in"
    ),
    ("tools/receipts.py", "snapshot_receipt", "job.simulator"): (
        "decides whether results are placed beside a sheet; names nothing"
    ),
}


def _uses() -> set[tuple[str, str, str]]:
    found: set[tuple[str, str, str]] = set()
    for module, tree in modules().items():
        if not module.startswith(_REPLY_BUILDERS):
            continue
        for scope, nodes in scopes(tree):
            for node in nodes:
                if not isinstance(node, ast.Attribute):
                    continue
                owner = ast.unparse(node.value)
                if (node.attr == "__name__" and _SIMULATOR_CLASS.search(owner)) or (
                    node.attr == "simulator" and _JOB_RECORD.search(owner)
                ):
                    found.add((module, scope, f"{owner}.{node.attr}"))
    return found


def test_no_reply_builder_names_a_simulator_by_its_class():
    """A reply names a job's simulator through ``job_simulator`` and a class
    through ``simulator_selector``; a class name or a record's ``simulator``
    read anywhere else in a reply builder is a second answer."""
    assert PACKAGE.is_dir()
    found = _uses()
    assert not found - _ALLOWED.keys(), sorted(found - _ALLOWED.keys())
    assert not _ALLOWED.keys() - found, f"no longer used: {sorted(_ALLOWED.keys() - found)}"


# ---------------------------------------------------------------------------
# Records written by an earlier version still load and reply
# ---------------------------------------------------------------------------


def _drop_key(value: Any, key: str) -> int:
    """Remove ``key`` from every object in ``value``; how many it removed."""
    if isinstance(value, dict):
        removed = int(value.pop(key, None) is not None or False)
        return removed + sum(_drop_key(item, key) for item in value.values())
    if isinstance(value, list):
        return sum(_drop_key(item, key) for item in value)
    return 0


def test_a_recovery_journal_written_before_the_selector_still_loads(state_no_sim: SessionState):
    """A journal's prepared candidate must re-serialize to exactly its own
    keys. One written before the selector was recorded has no such key, and
    still loads, so its resume and replay keep their answers."""
    from ltspice_mcp.lib.recovery_journal import load_journal
    from ltspice_mcp.lib.store import Store
    from tests.test_recovery_launch_pool import _recoverable_job

    job, _request = _recoverable_job(
        state_no_sim, job_id="exp_earlier_journal", request_id="earlier-journal", cases=1
    )
    store_ = Store(state_no_sim.working_dir)
    path = store_.recovery_journal(job.request_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    # As the earlier version wrote it: whatever this one wrote, no selector.
    _drop_key(data, "simulator_selector")
    path.write_text(json.dumps(data), encoding="utf-8")

    journal = load_journal(store_, job.request_id)
    assert journal is not None
    assert journal.head_job_id == job.job_id


@pytest.mark.asyncio
@pytest.mark.parametrize("class_name", ["LTspice", "LTspiceWSL"])
async def test_an_earlier_record_is_named_the_same_whatever_its_platform_class(
    config: ServerConfig,
    deck: Path,
    builds: dict[str, Path],
    recording_families: None,
    class_name: str,
):
    """The family of a record written before the selector comes from its class
    name, which differs by platform (Windows' own class keeps the name
    ``LTspice``; WSL's is ``LTspiceWSL``); the reply does not."""
    from ltspice_mcp.lib.ltspice_windows import LTspice as OffDesktop

    assert OffDesktop.__name__ == "LTspice" and LTspiceWSL.__name__ == "LTspiceWSL"
    state = _state(config, builds)
    receipt = await terminal_experiment(state, _payload(deck, "earlier-class", "ltspice:xvii"))
    await state.job_registry.drain_pending()
    _forget_selector(state, receipt["job_id"])
    path = store.Store(state.working_dir).job_record(receipt["job_id"])
    data = json.loads(path.read_text())
    data["simulator"] = class_name
    path.write_text(json.dumps(data))

    status = await _jobs(_state(config, builds), action="status", job_id=receipt["job_id"])
    assert status["simulator"] == "ltspice:xvii", status
    assert "simulator_notes" not in status, status["simulator_notes"]


@pytest.mark.asyncio
async def test_an_earlier_record_matches_its_build_as_the_platform_compares_paths(
    config: ServerConfig,
    deck: Path,
    builds: dict[str, Path],
    recording_families: None,
    monkeypatch: pytest.MonkeyPatch,
):
    """Windows compares paths without regard to case: a record that wrote
    the build's path in other case still names that build there."""
    import os

    import ltspice_mcp.lib.job_simulator as job_simulator_mod

    state = _state(config, builds)
    receipt = await terminal_experiment(state, _payload(deck, "earlier-case", "ltspice:xvii"))
    await state.job_registry.drain_pending()
    _forget_selector(state, receipt["job_id"], path=str(builds["xvii"]).upper())
    monkeypatch.setattr(job_simulator_mod.os.path, "normcase", lambda p: os.fspath(p).lower())

    status = await _jobs(_state(config, builds), action="status", job_id=receipt["job_id"])
    assert status["simulator"] == "ltspice:xvii", status
