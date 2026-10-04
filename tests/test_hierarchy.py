"""Public hierarchy discovery and bounded source resolution."""

from pathlib import Path

import pytest

from tests.test_inspect_tools import _run


@pytest.fixture
def hierarchy_deck(work_dir: Path) -> Path:
    (work_dir / "models.lib").write_text(
        ".lib tt\n.model n NMOS(level=1 vto=.5 kp=100u)\n.endl tt\n"
        ".lib ff\n.model n NMOS(level=1 vto=.4)\n.endl ff\n",
        encoding="utf-8",
    )
    (work_dir / "leaf.inc").write_text(
        ".param r=3k extra=7\nR1 p internal {r}\nR2 internal 0 {twice}\n"
        "M0 p gate 0 0 n w={w} l={l}\n",
        encoding="utf-8",
    )
    deck = work_dir / "hierarchy.cir"
    deck.write_text(
        "* repeated amplifier blocks\n.lib models.lib tt\n.global gate\n"
        ".param r=1k\n"
        ".subckt leaf p r=2k twice={2*r} w=10u l=2u\n.include leaf.inc\n.ends leaf\n"
        ".subckt wrapper p r=4k\nXleaf p leaf r={r}\n.ends wrapper\n"
        "XA a wrapper r=5k\nXB b wrapper\n.end\nRignored a 0 1\n",
        encoding="utf-8",
    )
    return deck


async def test_repeated_hierarchy_values_ports_and_active_model(state_no_sim, hierarchy_deck):
    (result,) = await _run(
        state_no_sim,
        [
            {
                "kind": "hierarchy",
                "path": str(hierarchy_deck),
                "simulator": "ngspice",
                "ngbehavior": "hsa",
            }
        ],
    )
    assert result["ok"], result
    rows = {tuple(row["instance"]): row for row in result["data"]["instances"]}
    assert len(rows) == 10
    left = rows[("XA", "Xleaf", "R1")]
    right = rows[("XB", "Xleaf", "R1")]
    assert left["value"]["value"] == 5000
    assert right["value"]["value"] == 4000
    assert rows[("XA", "Xleaf", "R2")]["value"]["value"] == 10000
    assert left["nodes"][0]["scope"] == []
    assert left["nodes"][0]["name"] == "a"
    assert left["nodes"][1]["scope"] == ["XA", "Xleaf"]
    assert right["nodes"][1]["scope"] == ["XB", "Xleaf"]
    mos = rows[("XA", "Xleaf", "M0")]
    assert mos["geometry"]["w"]["value"] == pytest.approx(10e-6)
    assert mos["geometry"]["l"]["value"] == pytest.approx(2e-6)
    assert mos["model"]["source"]["section"] == "tt"
    assert mos["source"]["definition"] == "leaf"
    assert mos["source"]["path"].endswith("leaf.inc")
    assert mos["nodes"][1]["scope"] == []
    assert mos["address"]["device"] == "m.xa.xleaf.m0"


async def test_spice_suffixed_deck_is_read(state_no_sim, hierarchy_deck):
    """``.spice`` is what xschem and the sky130 testbenches write."""
    deck = hierarchy_deck.with_suffix(".spice")
    deck.write_bytes(hierarchy_deck.read_bytes())
    result = await _inspect(state_no_sim, deck)
    assert result["ok"], result
    assert len(result["data"]["instances"]) == 10


async def _inspect(state, path, **query):
    (result,) = await _run(
        state, [{"kind": "hierarchy", "path": str(path), "simulator": "ltspice", **query}]
    )
    return result


@pytest.mark.parametrize(("simulator", "expected"), [("ltspice", 2000), ("ngspice", 3000)])
async def test_backend_default_body_precedence(state_no_sim, hierarchy_deck, simulator, expected):
    text = hierarchy_deck.read_text(encoding="utf-8").replace(
        "Xleaf p leaf r={r}", "Xleaf p leaf PARAMS:"
    )
    hierarchy_deck.write_text(text, encoding="utf-8")
    query = {"simulator": simulator, "instance": ["xa", "xLEAF"], "prefix": "r"}
    if simulator == "ngspice":
        query["ngbehavior"] = "hsa"
    result = await _inspect(state_no_sim, hierarchy_deck, **query)
    assert result["ok"], result
    rows = result["data"]["instances"]
    assert [r["instance"] for r in rows] == [["XA", "Xleaf", "R1"], ["XA", "Xleaf", "R2"]]
    assert [r["value"]["value"] for r in rows] == [expected, 2 * expected]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (".include absent.inc", "missing include"),
        ("X1 a absent", "unresolved child"),
        (".subckt a p\nX1 p a\n.ends a\nX0 n a", "recursive"),
        (".subckt a p\nR1 p 0 1k\n.ends b", "mismatched"),
        (".subckt a p\nR1 p 0 1k", "unclosed"),
        (".ends", "unmatched"),
        ("R1 a 0 1k\nr1 b 0 2k", "duplicate active reference"),
        (".param a=1 A=2\nR1 a 0 {a}", "duplicate assignment"),
        (".subckt a p r=1 r=2\n.ends a", "duplicate assignment"),
        (".subckt a p\n.subckt local q\n.ends local\n.ends a", "local/nested"),
        (".if 1\nR1 a 0 1\n.endif", "conditional"),
        (".control\nalter R1 2\n.endc", "opaque control"),
        (".subckt a p q\n.ends a\nX0 n a", "port arity"),
        ("R1 a 0 1k x=2 bad", "unsupported tokens"),
    ],
)
async def test_structural_refusals_are_isolated(state_no_sim, work_dir, body, message):
    path = work_dir / "invalid.cir"
    path.write_text("* invalid\n" + body + "\n.end\n", encoding="utf-8")
    bad, good = await _run(
        state_no_sim,
        [
            {"kind": "hierarchy", "path": str(path), "simulator": "ltspice"},
            {"kind": "components", "path": str(path)},
        ],
    )
    assert not bad["ok"], bad
    assert message in bad["error"]["message"]
    assert good["ok"], good


async def test_non_node_references_and_ambiguous_mos_are_unknown(state_no_sim, work_dir):
    path = work_dir / "forms.cir"
    path.write_text(
        "* forms\nK1 L1 L2 .9\nF1 a b Vctrl 2\nM1 d g s b n 2\n.end\n", encoding="utf-8"
    )
    result = await _inspect(state_no_sim, path)
    assert result["ok"], result
    for row in result["data"]["instances"]:
        assert row["nodes"] == []
        assert row["connectivity_reason"]
        assert row["model"]["source"] is None
        assert row["address"]["device"] is None
        assert row["raw"]


@pytest.mark.parametrize(
    "expression", ["missing", "sqrt(4)", "1/0", "2^1000", "1e999", "a", "(1+2"]
)
async def test_unknown_numeric_facts_keep_structure(state_no_sim, work_dir, expression):
    path = work_dir / "expressions.cir"
    path.write_text(
        f"* expressions\n.param a={{b}} b={{a}}\nR1 p 0 {{{expression}}}\n.end\n", encoding="utf-8"
    )
    result = await _inspect(state_no_sim, path)
    assert result["ok"], result
    row = result["data"]["instances"][0]
    assert len(row["nodes"]) == 2
    assert row["value"]["status"] == "unresolved"
    assert row["value"]["value"] is None
    assert row["value"]["reason"]


async def test_step_invalidates_geometry_and_values(state_no_sim, hierarchy_deck):
    text = hierarchy_deck.read_text(encoding="utf-8").replace(
        ".end\n", ".step param r list 1k 2k\n.end\n"
    )
    hierarchy_deck.write_text(text, encoding="utf-8")
    result = await _inspect(state_no_sim, hierarchy_deck)
    assert result["ok"], result
    for row in result["data"]["instances"]:
        assert row["value"]["value"] is None
        for fact in row["geometry"].values():
            assert fact["value"] is None
            assert "context-dependent" in fact["reason"]
    assert result["data"]["profile"]["evaluation_context"] == "initial_deck"


async def test_profile_section_interpretation_and_offline_default(
    state_no_sim, hierarchy_deck, monkeypatch
):
    from ltspice_mcp.tools import inspect_tools

    monkeypatch.setattr(inspect_tools, "current_ngbehavior", lambda: "kiltpsa")
    result = await _inspect(state_no_sim, hierarchy_deck, simulator="ngspice")
    assert not result["ok"]
    assert "lt/ps" in result["error"]["message"]
    result = await _inspect(state_no_sim, hierarchy_deck, ngbehavior="hsa")
    assert not result["ok"]
    assert result["error"]["code"] == "invalid_query"


async def test_content_cursor_rejects_same_size_dependency_rewrite(
    state_no_sim, hierarchy_deck, monkeypatch
):
    import os

    from ltspice_mcp.tools import inspect_tools

    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 1)
    first = await _inspect(state_no_sim, hierarchy_deck)
    assert first["ok"], first
    cursor = first["next_cursor"]
    leaf = hierarchy_deck.parent / "leaf.inc"
    stamp = leaf.stat()
    leaf.write_bytes(leaf.read_bytes().replace(b"R1", b"R9"))
    os.utime(leaf, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    stale = await _inspect(state_no_sim, hierarchy_deck, cursor=cursor)
    assert not stale["ok"]
    assert stale["error"]["code"] == "invalid_cursor"


async def test_capture_hashes_parsed_bytes_and_pure_resolve_ignores_disk(
    state_no_sim, hierarchy_deck, monkeypatch
):
    from ltspice_mcp.lib import hierarchy

    captured = hierarchy.load_hierarchy(
        str(hierarchy_deck), [hierarchy_deck.parent], hierarchy.SemanticProfile("ltspice")
    )
    leaf = hierarchy_deck.parent / "leaf.inc"
    old = leaf.read_bytes()
    leaf.write_bytes(old.replace(b"R1", b"R9"))

    # Filesystem access after capture must not affect the pure resolver.
    def denied(*args, **kwargs):
        raise AssertionError("pure resolver touched filesystem")

    monkeypatch.setattr(Path, "exists", denied)
    resolved = hierarchy.resolve_hierarchy(
        hierarchy_deck, {f.path: f for f in captured.inputs}, captured.profile
    )
    assert resolved == captured
    assert any(row.reference == "R1" for row in resolved.instances)
    assert not any(row.reference == "R9" for row in resolved.instances)


async def test_section_filename_collision_is_refused(state_no_sim, hierarchy_deck):
    (hierarchy_deck.parent / "tt").write_text("* colliding filename\n", encoding="utf-8")
    result = await _inspect(state_no_sim, hierarchy_deck)
    assert not result["ok"]
    assert "malformed .endl" in result["error"]["message"]


@pytest.mark.parametrize(
    ("limit", "value", "message"),
    [("MAX_BYTES", 20, "bytes"), ("MAX_CARDS", 2, "cards"), ("MAX_INSTANCES", 2, "instances")],
)
async def test_resource_bounds_refuse_complete_looking_prefix(
    state_no_sim, hierarchy_deck, monkeypatch, limit, value, message
):
    from ltspice_mcp.lib import hierarchy

    monkeypatch.setattr(hierarchy, limit, value)
    result = await _inspect(state_no_sim, hierarchy_deck)
    assert not result["ok"]
    assert message in result["error"]["message"]
    assert "data" not in result


async def test_include_cycle_and_escape(state_no_sim, work_dir, tmp_path_factory):
    from tests.conftest import symlink_or_skip

    root = work_dir / "cycle.cir"
    root.write_text("* cycle\n.include cycle.cir\n.end\n", encoding="utf-8")
    result = await _inspect(state_no_sim, root)
    assert not result["ok"] and "cycle" in result["error"]["message"]
    outside = tmp_path_factory.mktemp("outside-hierarchy") / "outside.inc"
    outside.write_text("R1 a 0 1k\n", encoding="utf-8")
    link = work_dir / "link.inc"
    symlink_or_skip(link, outside)
    root.write_text("* escape\n.include link.inc\n.end\n", encoding="utf-8")
    result = await _inspect(state_no_sim, root)
    assert not result["ok"] and "outside" in result["error"]["message"]


def test_python_api_collects_hierarchy_pages(state_no_sim, hierarchy_deck, monkeypatch):
    from ltspice_mcp.tools import inspect_tools
    from tests.conftest import SyncApi

    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 2)
    result = SyncApi(state_no_sim).inspect(
        queries=[{"kind": "hierarchy", "path": str(hierarchy_deck), "simulator": "ltspice"}]
    )
    item = result["results"][0]
    assert item["ok"], item
    assert len(item["data"]["instances"]) == item["data"]["total"] == 10
    assert item["data"]["returned"] == 10


@pytest.mark.parametrize(
    ("backend", "ngbehavior"),
    [("ngspice", "hsa"), ("ngspice", "kiltpsa"), ("ngspice", ""), ("ltspice", None)],
)
async def test_discovery_to_real_device_operating_point(
    work_dir, backend, ngbehavior, monkeypatch
):
    import os
    import shutil

    from spicelib.simulators.ngspice_simulator import NGspiceSimulator

    from ltspice_mcp.config import ServerConfig
    from ltspice_mcp.lib.simulator import detect_simulators
    from ltspice_mcp.state import SessionState
    from tests.conftest import terminal_experiment
    from tests.test_ngspice_e2e import _analyze

    if backend == "ngspice" and shutil.which("ngspice") is None:
        pytest.skip("ngspice is not on PATH")
    if backend == "ltspice" and not (
        os.name == "nt" and os.environ.get("LTSPICE_MCP_RUN_LTSPICE_INTEGRATION") == "1"
    ):
        pytest.skip("native Windows LTspice integration is opt-in")
    config = ServerConfig(
        working_dir=work_dir, allowed_paths=[work_dir], simulator=backend, ngbehavior=ngbehavior
    )
    available = detect_simulators(config)
    assert backend in available, available
    monkeypatch.setattr(NGspiceSimulator, "_compatibility_mode", ngbehavior or "")
    state = SessionState.create(config, available)
    path = work_dir / "electrical.cir"
    text = (
        "* resistor and MOS hierarchy\n.model n NMOS(level=1 vto=.5 kp=100u lambda=0 cgso=1e-10)\n"
        ".param r=1k\n.subckt sibling p r=2k s=1k\nR1 p 0 {s}\n.ends\n"
        "XS ps sibling r=3k s={r*2}\nVS ps 0 1\n"
        "Rpower pp 0 {2^3^2}\nVP pp 0 1\nRmil pm 0 1mil\nVM pm 0 1\n"
        ".subckt leaf p d g r=2k twice={2*r} w=10u l=2u\n.param r=3k\n"
        "R1 p 0 {r}\nR2 p 0 {twice}\nM0 d g 0 0 n w={w} l={l}\n.ends leaf\n"
        ".subckt wrapper p d g r=4k w=30u\nXleaf p d g leaf r={r} w={w}\n.ends wrapper\n"
        "XA pa da ga wrapper r=5k w=20u\nXB pb db gb wrapper\nXC pc dc gc leaf\n"
        "VC pc 0 1\nVDC dc 0 2\nVGC gc 0 1.5\n"
        "VA pa 0 1\nVB pb 0 1\nVDA da 0 2\nVDB db 0 2\nVGA ga 0 1.5\nVGB gb 0 1.5\n.op\n"
    )
    if backend == "ngspice" and ngbehavior != "kiltpsa":
        text += (
            ".subckt grounded p\nR1 p gnd 1k\nR2 p 0 2k\n.ends\n"
            "XG1 pg1 grounded\nXG2 pg2 grounded\nVG1 pg1 0 1\nVG2 pg2 0 1\n"
        )
    path.write_text(text + ".end\n", encoding="utf-8")
    query = {"simulator": backend}
    if backend == "ngspice":
        query["ngbehavior"] = ngbehavior
    discovered = await _inspect(state, path, **query)
    assert discovered["ok"], discovered
    rows = {tuple(row["instance"]): row for row in discovered["data"]["instances"]}
    assert discovered["data"]["profile"]["ngbehavior"] == ngbehavior
    cases = [
        (("XA", "Xleaf"), 0.001, 5000),
        (("XB", "Xleaf"), 0.0015, 4000),
        (("XC",), 0.0005, 2000 if backend == "ltspice" else 3000),
    ]
    selected = [rows[(*scope, ref)] for scope, _, _ in cases for ref in ("M0", "R1", "R2")]
    extra_resistors = [
        (("XS", "R1"), 2000 if backend == "ltspice" else 6000),
        (("Rpower",), 1 if backend == "ltspice" else 64),
        (("Rmil",), 25.4e-6),
    ]
    if backend == "ngspice" and ngbehavior != "kiltpsa":
        extra_resistors += [
            ((parent, ref), resistance)
            for parent in ("XG1", "XG2")
            for ref, resistance in (("R1", 1000), ("R2", 2000))
        ]
    selected.extend(rows[scope] for scope, _ in extra_resistors)
    saves = sorted({row["address"]["save"] for row in selected})
    path.write_text(text + "\n".join(saves) + "\n.end\n", encoding="utf-8")
    try:
        if backend == "ngspice":
            assert NGspiceSimulator._compatibility_mode == ngbehavior
        receipt = await terminal_experiment(
            state,
            {
                "request_id": f"hierarchy-electrical-{backend}",
                "circuits": [{"path": str(path)}],
                "execution": {"simulator": backend, "wait_s": 90},
            },
        )
        assert receipt["status"] == "completed", receipt
        if backend == "ngspice":
            from ltspice_mcp.lib.services import resolve_job

            job = resolve_job(receipt["job_id"], state)
            log_file = job.cases[0].log_file
            assert log_file is not None
            log = log_file.read_text(encoding="utf-8")
            modes = {"hsa": "hs a", "kiltpsa": "ps lt ki a"}
            if ngbehavior:
                assert f"Compatibility modes selected: {modes[ngbehavior]}" in log, log
            else:
                assert "No compatibility mode selected" in log, log
        for scope, expected_gm, expected_r in cases:
            mos = rows[(*scope, "M0")]
            resistor = rows[(*scope, "R1")]
            dependent = rows[(*scope, "R2")]
            result = await _analyze(
                state,
                receipt["job_id"],
                [
                    {
                        "key": "mos",
                        "metric": "operating_point",
                        "device": mos["address"]["device"],
                    },
                    {
                        "key": "dependent",
                        "metric": "operating_point",
                        "device": dependent["address"]["device"],
                    },
                    {
                        "key": "resistor",
                        "metric": "operating_point",
                        "device": resistor["address"]["device"],
                    },
                ],
            )
            op = result["results"]["mos"]["values"][0]["value"]
            gm = {k: v for k, v in op["device_op_points"].items() if k.lower().endswith("[gm]")}
            assert len(gm) == 1, op
            assert next(iter(gm.values())) == pytest.approx(expected_gm, rel=1e-5)
            if backend == "ltspice":
                overlap = {
                    key: value
                    for key, value in op["device_op_points"].items()
                    if key.lower().endswith("[cgsov]")
                }
                assert len(overlap) == 1, op
                assert next(iter(overlap.values())) == pytest.approx(
                    mos["geometry"]["w"]["value"] * 1e-10, rel=1e-5, abs=1e-22
                )
            assert all(scope[0].lower() in key.lower() for key in gm)
            assert mos["geometry"]["w"]["value"] == pytest.approx(expected_gm / 0.0001 * 2e-6)
            op = result["results"]["resistor"]["values"][0]["value"]
            currents = {**op["currents"], **op["device_op_points"]}
            assert len(currents) == 1, op
            assert abs(next(iter(currents.values()))) == pytest.approx(1 / expected_r, rel=1e-5)
            assert resistor["value"]["value"] == expected_r
            op = result["results"]["dependent"]["values"][0]["value"]
            currents = {**op["currents"], **op["device_op_points"]}
            assert len(currents) == 1, op
            assert abs(next(iter(currents.values()))) == pytest.approx(
                1 / (2 * expected_r), rel=1e-5
            )
            assert dependent["value"]["value"] == 2 * expected_r
        for scope, resistance in extra_resistors:
            row = rows[scope]
            result = await _analyze(
                state,
                receipt["job_id"],
                [{"key": "r", "metric": "operating_point", "device": row["address"]["device"]}],
            )
            op = result["results"]["r"]["values"][0]["value"]
            currents = {**op["currents"], **op["device_op_points"]}
            assert len(currents) == 1, op
            assert abs(next(iter(currents.values()))) == pytest.approx(1 / resistance, rel=1e-5)
            if scope == ("Rpower",) or (scope == ("XS", "R1") and backend == "ngspice"):
                assert row["value"]["value"] is None
                assert row["value"]["reason"]
            else:
                assert row["value"]["value"] == pytest.approx(resistance)
    finally:
        await state.shutdown()


async def test_changed_after_read_cursor_binds_old_bytes(state_no_sim, work_dir, monkeypatch):
    import hashlib

    from ltspice_mcp.lib import hierarchy
    from ltspice_mcp.tools import inspect_tools

    path = work_dir / "interleaving.cir"
    old = b"* interleaving\nR1 a 0 1k\nR2 a 0 2k\n.end\n"
    path.write_bytes(old)
    decode = hierarchy.decode_spice_bytes
    rewritten = False

    def rewrite_after_read(content):
        nonlocal rewritten
        if not rewritten:
            rewritten = True
            path.write_bytes(old.replace(b"R1", b"R9"))
        return decode(content)

    monkeypatch.setattr(hierarchy, "decode_spice_bytes", rewrite_after_read)
    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 1)
    first = await _inspect(state_no_sim, path)
    assert first["ok"], first
    assert first["data"]["instances"][0]["reference"] == "R1"
    assert first["data"]["inputs"][0]["sha256"] == hashlib.sha256(old).hexdigest()
    second = await _inspect(state_no_sim, path, cursor=first["next_cursor"])
    assert not second["ok"]
    assert second["error"]["code"] == "invalid_cursor"


async def test_oversized_dependency_is_read_with_remaining_bound(
    state_no_sim, work_dir, monkeypatch
):
    from ltspice_mcp.lib import hierarchy

    path = work_dir / "bounded.cir"
    root = b"* bounded\n.include large.inc\n.end\n"
    path.write_bytes(root)
    dependency = work_dir / "large.inc"
    dependency.write_bytes(b"*" * 10000)
    monkeypatch.setattr(hierarchy, "MAX_BYTES", 100)
    original_open = Path.open
    reads = []

    class CheckedRead:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def read(self, size=-1):
            reads.append(size)
            assert 0 < size <= 101
            return self.stream.read(size)

        def __exit__(self, *args):
            self.stream.close()

    def bounded_open(self, *args, **kwargs):
        stream = original_open(self, *args, **kwargs)
        return CheckedRead(stream) if self in {path, dependency} else stream

    monkeypatch.setattr(Path, "open", bounded_open)
    result = await _inspect(state_no_sim, path)
    assert not result["ok"]
    assert "bytes" in result["error"]["message"]
    assert reads == [101, 101 - len(root)]


async def test_inactive_include_is_not_followed_and_missing_section_fails(state_no_sim, work_dir):
    path = work_dir / "sections.cir"
    library = work_dir / "corners.lib"
    library.write_text(
        ".lib tt\nR1 a 0 1k\n.endl tt\n.lib ff\n.include missing\n.endl ff\n", encoding="utf-8"
    )
    path.write_text("* sections\n.lib corners.lib tt\n.end\n", encoding="utf-8")
    result = await _inspect(state_no_sim, path)
    assert result["ok"], result
    assert result["data"]["instances"][0]["value"]["value"] == 1000
    path.write_text("* sections\n.lib corners.lib absent\n.end\n", encoding="utf-8")
    result = await _inspect(state_no_sim, path)
    assert not result["ok"] and "missing library section" in result["error"]["message"]


async def test_legal_punctuation_keeps_exact_identity_without_backend_guess(
    state_no_sim, work_dir
):
    path = work_dir / "identity.cir"
    path.write_text(
        "* identity\n.subckt leaf p\nR1 p 0 1k\n.ends\nX.a n leaf\nX_a n leaf\n.end\n",
        encoding="utf-8",
    )
    result = await _inspect(state_no_sim, path, instance=["x.a"])
    assert result["ok"], result
    rows = result["data"]["instances"]
    assert [r["instance"] for r in rows] == [["X.a"], ["X.a", "R1"]]
    assert rows[1]["address"]["device"] is None
    assert rows[1]["address"]["reason"]


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "cp1252"])
async def test_captured_windows_encodings(state_no_sim, work_dir, encoding):
    path = work_dir / "encoded.cir"
    path.write_bytes("* café\r\nR1 a 0 1k\r\n.end\r\n".encode(encoding))
    result = await _inspect(state_no_sim, path)
    assert result["ok"], result
    assert result["data"]["instances"][0]["value"]["value"] == 1000


async def test_selected_section_excludes_outside_cards(state_no_sim, work_dir):
    path = work_dir / "selected.cir"
    (work_dir / "selected.lib").write_text(
        ".include absent.inc\nRoutside a 0 1k\n.lib tt\nRinside a 0 2k\n.endl tt\n",
        encoding="utf-8",
    )
    path.write_text("* selected\n.lib selected.lib tt\n.end\n", encoding="utf-8")
    result = await _inspect(state_no_sim, path, simulator="ngspice", ngbehavior="hsa")
    assert result["ok"], result
    assert [r["reference"] for r in result["data"]["instances"]] == ["Rinside"]


@pytest.mark.parametrize(
    "body",
    [
        ".subckt leaf p r=1 bad\n.ends leaf\nX1 n leaf",
        ".subckt leaf p\n.ends leaf extra\nX1 n leaf",
        ".model n NMOS(vto=1 VTO=2)\nM1 d g 0 0 n",
        "+ orphan\nR1 a 0 1k",
    ],
)
async def test_lossy_boundary_and_assignment_shapes_are_refused(state_no_sim, work_dir, body):
    path = work_dir / "malformed.cir"
    path.write_text("* malformed\n" + body + "\n.end\n", encoding="utf-8")
    result = await _inspect(state_no_sim, path)
    assert not result["ok"], result


async def test_internal_node_case_and_ground_port_alias(state_no_sim, work_dir):
    path = work_dir / "nodecase.cir"
    path.write_text(
        "* nodes\n.subckt leaf p\nR1 p Internal 1k\nR2 INTERNAL 0 2k\n.ends\nX1 0 leaf\n.end\n",
        encoding="utf-8",
    )
    result = await _inspect(state_no_sim, path)
    assert result["ok"], result
    rows = result["data"]["instances"]
    assert rows[1]["nodes"][0]["name"] == "0"
    assert rows[1]["nodes"][1] == rows[2]["nodes"][0]


async def test_numeric_arithmetic_scale_and_binned_model(state_no_sim, work_dir):
    path = work_dir / "numbers.cir"
    path.write_text(
        "* numbers\n.param a=2 b='(a+3)*2'\n.option scale={missing}\n"
        ".model n.1 NMOS(level=1)\nM1 d g 0 0 n w={b} l=1\n"
        "R1 a 0 {-(2+3)/5+2^3}\n.end\n",
        encoding="utf-8",
    )
    result = await _inspect(state_no_sim, path, simulator="ngspice", ngbehavior="hsa")
    assert result["ok"], result
    mos, resistor = result["data"]["instances"]
    assert resistor["value"]["value"] == 7
    assert mos["parameters"]["w"]["value"] == 10
    assert mos["geometry"]["w"]["value"] is None
    assert "missing" in mos["geometry"]["w"]["reason"]
    assert "binned" in mos["model"]["reason"]


async def test_cursor_binds_profile_and_filters(state_no_sim, hierarchy_deck, monkeypatch):
    from ltspice_mcp.tools import inspect_tools

    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 1)
    first = await _inspect(state_no_sim, hierarchy_deck)
    for changed in (
        {"prefix": "R"},
        {"instance": ["XA"]},
        {"simulator": "ngspice", "ngbehavior": "hsa"},
    ):
        result = await _inspect(
            state_no_sim, hierarchy_deck, cursor=first["next_cursor"], **changed
        )
        assert not result["ok"] and result["error"]["code"] == "invalid_cursor"


async def test_ambiguous_existing_selector_is_unavailable(state_no_sim, work_dir):
    path = work_dir / "suffix.cir"
    path.write_text(
        "* suffix\n.subckt leaf p\nR1 p 0 1k\n.ends\n"
        ".subckt wrapper p\nXA p leaf\n.ends\nXA a leaf\nXZ b wrapper\n.end\n",
        encoding="utf-8",
    )
    result = await _inspect(state_no_sim, path, instance=["XA", "R1"])
    assert result["ok"], result
    row = result["data"]["instances"][0]
    assert row["address"]["device"] is None
    assert "also matches" in row["address"]["reason"]


async def test_include_inside_repeated_declarations_has_lexical_occurrence(state_no_sim, work_dir):
    path = work_dir / "occurrence.cir"
    (work_dir / "body.inc").write_text("R1 p 0 {r}\n", encoding="utf-8")
    path.write_text(
        "* occurrence\n.subckt one p r=1k\n.include body.inc\n.ends\n"
        ".subckt two p r=2k\n.include body.inc\n.ends\nX1 a one\nX2 b two\n.end\n",
        encoding="utf-8",
    )
    result = await _inspect(state_no_sim, path, prefix="R")
    assert result["ok"], result
    left, right = result["data"]["instances"]
    assert left["source"]["path"] == right["source"]["path"]
    assert left["source"]["line"] == right["source"]["line"] == 1
    assert left["source"]["definition"] == "one"
    assert right["source"]["definition"] == "two"
    assert left["value"]["value"] == 1000
    assert right["value"]["value"] == 2000


async def test_inspection_io_runs_off_loop(state_no_sim, hierarchy_deck, monkeypatch):
    import threading

    from ltspice_mcp.lib import hierarchy

    loop_thread = threading.get_ident()
    read = Path.open
    threads = []

    def checked_open(self, *args, **kwargs):
        if self.parent == hierarchy_deck.parent:
            threads.append(threading.get_ident())
        return read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    result = await _inspect(state_no_sim, hierarchy_deck)
    assert result["ok"], result
    assert threads and all(t != loop_thread for t in threads)
    assert hierarchy.MAX_BYTES > 0


async def test_unsupported_profile_has_no_authoritative_facts(state_no_sim, hierarchy_deck):
    result = await _inspect(state_no_sim, hierarchy_deck, simulator="ngspice", ngbehavior="all")
    assert not result["ok"]
    assert "unsupported hierarchy ngbehavior" in result["error"]["message"]
    assert "data" not in result


@pytest.mark.parametrize("mode", ["", "hsa", "kiltpsa"])
async def test_sibling_override_is_unresolved_but_self_reference_survives(
    state_no_sim, work_dir, mode
):
    path = work_dir / "siblings.cir"
    path.write_text(
        "* siblings\n.param r=1k\n.subckt leaf p r=2k s=1k\n"
        "R1 p 0 {s}\nR2 p 0 {r}\n.ends\n"
        "X1 a leaf r=3k s={r*2}\nX2 b leaf r={r*2}\n.end\n",
        encoding="utf-8",
    )
    result = await _inspect(state_no_sim, path, simulator="ngspice", ngbehavior=mode)
    assert result["ok"], result
    rows = {tuple(row["instance"]): row for row in result["data"]["instances"]}
    fact = rows[("X1", "R1")]["value"]
    assert fact["value"] is None
    assert "sibling" in fact["reason"]
    assert rows[("X2", "R2")]["value"]["value"] == 2000


@pytest.mark.parametrize("directive", [".option", ".options"])
async def test_ltspice_explicit_scale_is_refused(state_no_sim, work_dir, directive):
    path = work_dir / "scale.cir"
    path.write_text(
        f"* scale\n{directive} scale=1u\nM1 d g 0 0 n w=20 l=2\n.end\n", encoding="utf-8"
    )
    result = await _inspect(state_no_sim, path)
    assert not result["ok"]
    assert "LTspice" in result["error"]["message"]


async def test_ltspice_mos_save_directive(state_no_sim, work_dir):
    path = work_dir / "mos.cir"
    path.write_text("* mos\nM1 d g 0 0 n w=20u l=2u\n.end\n", encoding="utf-8")
    result = await _inspect(state_no_sim, path)
    assert result["data"]["instances"][0]["address"]["save"] == ".options logopinfo"


@pytest.mark.parametrize("expression", ["2^3^2", "2**3**2"])
async def test_power_chain_is_explicitly_unresolved(state_no_sim, work_dir, expression):
    path = work_dir / "power.cir"
    path.write_text(f"* power\nR1 a 0 {{{expression}}}\n.end\n", encoding="utf-8")
    result = await _inspect(state_no_sim, path)
    fact = result["data"]["instances"][0]["value"]
    assert fact["value"] is None
    assert "power" in fact["reason"] or "caret" in fact["reason"]


async def test_mil_values_and_geometry_are_not_milli(state_no_sim, work_dir):
    path = work_dir / "mil.cir"
    path.write_text("* mil\nR1 a 0 1mil\nM1 d g 0 0 n w=2mil l=1MIL\n.end\n", encoding="utf-8")
    result = await _inspect(state_no_sim, path)
    mos, resistor = result["data"]["instances"]
    assert resistor["value"]["value"] == pytest.approx(25.4e-6), resistor
    assert mos["geometry"]["w"]["value"] == pytest.approx(50.8e-6)
    assert mos["geometry"]["l"]["value"] == pytest.approx(25.4e-6)


@pytest.mark.parametrize(
    "title", ["Rtitle a 0 2000", ".include missing.inc", ".end", ".subckt bogus a"]
)
async def test_root_title_is_not_executed_and_include_first_card_is_retained(
    state_no_sim, work_dir, title
):
    path = work_dir / "title.cir"
    (work_dir / "first.inc").write_text("R1 a 0 1000\n", encoding="utf-8")
    path.write_text(f"{title}\n.include first.inc\n.end\n", encoding="utf-8")
    result = await _inspect(state_no_sim, path, simulator="ngspice", ngbehavior="")
    assert result["ok"], result
    rows = result["data"]["instances"]
    assert [r["reference"] for r in rows] == ["R1"]
    assert rows[0]["source"]["line"] == 1


async def test_hierarchy_serializes_only_selected_page(state_no_sim, work_dir, monkeypatch):
    from ltspice_mcp.lib.hierarchy import ResolvedInstance
    from ltspice_mcp.tools import inspect_tools

    path = work_dir / "page.cir"
    path.write_text(
        "* page\n" + "\n".join(f"R{i} a 0 1k" for i in range(100)) + "\n.end\n", encoding="utf-8"
    )
    serialized = []
    original = ResolvedInstance.row

    def counted(row):
        serialized.append(row.reference)
        return original(row)

    monkeypatch.setattr(ResolvedInstance, "row", counted)
    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 1)
    first = await _inspect(state_no_sim, path)
    assert first["ok"], first
    assert serialized == ["R0"]
    assert first["data"]["total"] == 100
    second = await _inspect(state_no_sim, path, cursor=first["next_cursor"])
    assert second["ok"], second
    assert serialized == ["R0", "R1"]
    assert second["data"]["instances"][0]["reference"] == "R1"


async def test_ltspice_caret_is_not_reported_as_power(state_no_sim, work_dir):
    path = work_dir / "caret.cir"
    path.write_text("* caret\nR1 a 0 {2^3}\n.end\n", encoding="utf-8")
    result = await _inspect(state_no_sim, path)
    fact = result["data"]["instances"][0]["value"]
    assert fact["value"] is None
    assert "LTspice" in fact["reason"]


@pytest.mark.parametrize("mode", ["", "hsa", "kiltpsa"])
async def test_ngspice_ground_alias_precedes_scope_and_ports(state_no_sim, work_dir, mode):
    path = work_dir / "ground.cir"
    path.write_text(
        "* ground\n.subckt leaf p gnd\nR1 p gnd 1k\nR2 p 0 2k\n.ends\n"
        "X1 a other leaf\nX2 b another leaf\n.end\n",
        encoding="utf-8",
    )
    result = await _inspect(state_no_sim, path, simulator="ngspice", ngbehavior=mode, prefix="R")
    if mode == "kiltpsa":
        assert not result["ok"], result
        assert "version-dependent ground alias" in result["error"]["message"]
        return
    assert result["ok"], result
    grounds = [row["nodes"][1] for row in result["data"]["instances"]]
    assert len(grounds) == 4
    assert all(
        node == {"scope": [], "name": "0", "voltage_trace": "V(0)", "reason": None}
        for node in grounds
    )


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("MXO", {("XA", "MXO1"), ("MXO2",)}),
        ("mxo", {("XA", "MXO1"), ("MXO2",)}),
        ("M", {("XA", "MXO1"), ("XA", "M1"), ("MXO2",)}),
    ],
)
async def test_prefix_is_a_case_insensitive_prefix_of_the_instance_reference(
    state_no_sim, work_dir, prefix, expected
):
    path = work_dir / "prefixes.cir"
    path.write_text(
        "* prefixes\n.model nch nmos(level=1)\n"
        ".subckt cell d g\nMXO1 d g 0 0 nch\nM1 d g 0 0 nch\n.ends cell\n"
        "XA a b cell\nMXO2 a b 0 0 nch\n.end\n",
        encoding="utf-8",
    )
    result = await _inspect(state_no_sim, path, prefix=prefix)
    assert result["ok"], result
    assert {tuple(row["instance"]) for row in result["data"]["instances"]} == expected


async def test_prefix_refuses_a_glob(state_no_sim, hierarchy_deck):
    result = await _inspect(state_no_sim, hierarchy_deck, prefix="M*")
    assert not result["ok"] and result["error"]["code"] == "invalid_prefix"
