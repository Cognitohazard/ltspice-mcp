"""The Python API's argument catalogue: coverage, shape, and delivery."""

from __future__ import annotations

import importlib
import inspect as inspect_mod
import subprocess
import sys

import pytest
from pydantic import ValidationError

from ltspice_mcp.api import _reference
from ltspice_mcp.api._methods import ApiMethodsMixin
from ltspice_mcp.errors import compact_validation_error
from ltspice_mcp.lib.model_fields import field_name
from ltspice_mcp.lib.recipes import DISCRIMINANTS
from ltspice_mcp.tools.inspect_tools import SUPPORTED_KINDS
from ltspice_mcp.tools.schematic_edit import EditSchematicInput

OPS = ("run_experiments", "jobs", "analyze_results", "inspect", "edit_schematic", "verify_circuit")


def _spelled_fields(model) -> list[str]:
    """The names a caller writes — an alias makes some differ from the attribute."""
    return [field_name(model, name, field) for name, field in model.model_fields.items()]


def _op_kinds() -> set[str]:
    """Every op discriminant edit_schematic accepts, read off the union."""
    from typing import get_args

    from ltspice_mcp.tools import schematic_edit

    union = get_args(get_args(schematic_edit.ConsolidatedOp)[0])
    return {value for model in union for value in get_args(model.model_fields["op"].annotation)}


class TestOwnerLifecycleDocumentation:
    """``wait=False`` dies with its process, and the catalogue is where a
    caller looks BEFORE submitting — the receipt's ``process_owned_job``
    observation arrives only after. Measured live: agents that met the
    rule undocumented burned minutes reverse-engineering it from a bare
    executor-shutdown traceback, three sessions in a row.
    """

    def test_run_experiments_reference_names_the_owner_lifecycle(self):
        # Whitespace-normalised: the note is wrapped, so a line break may
        # fall anywhere inside the asserted phrase.
        text = " ".join(_reference.reference("run_experiments").split())
        assert "wait=False" in text
        assert "cancelled when" in text and "process" in text

    def test_method_doc_carries_the_same_note(self):
        # The ``help(api.run_experiments)`` surface must not say less than
        # the catalogue — both render from the same operation entry.
        assert "wait=False" in _reference.method_doc("run_experiments")


class TestIndex:
    def test_index_names_all_six_operations(self):
        text = _reference.reference()
        for name in OPS:
            assert name in text

    def test_index_says_how_to_drill_in(self):
        assert "api.reference(" in _reference.reference()

    def test_op_names_is_the_six(self):
        assert set(_reference.op_names()) == set(OPS)


class TestPerOperationTree:
    @pytest.mark.parametrize("name", OPS)
    def test_every_operation_renders(self, name: str):
        text = _reference.reference(name)
        assert text.startswith(name)
        assert "arguments" in text
        assert "example" in text

    @pytest.mark.parametrize("name", OPS)
    def test_tree_carries_no_json_schema_artifacts(self, name: str):
        text = _reference.reference(name)
        for artifact in ("$ref", "anyOf", "allOf", "$defs", "propertyName"):
            assert artifact not in text, f"{name} reference leaked {artifact}"

    @pytest.mark.parametrize("name", OPS)
    def test_top_level_fields_are_all_listed(self, name: str):
        operation = _reference._find(name)
        text = _reference.reference(name)
        for spelled in _spelled_fields(operation.model):
            assert spelled in text, f"{name}.{spelled} missing from its reference"

    def test_recipe_union_is_enumerated_member_by_member(self):
        text = _reference.reference("analyze_results")
        for metric in DISCRIMINANTS:
            assert repr(metric) in text, f"recipe metric {metric} missing"
        # ...and with the per-member fields, not just the names.
        assert "level_deg" in text
        assert "at_hz" in text

    def test_inspect_query_kinds_are_enumerated(self):
        text = _reference.reference("inspect")
        for kind in SUPPORTED_KINDS:
            assert repr(kind) in text

    def test_edit_op_kinds_are_enumerated_with_their_fields(self):
        text = _reference.reference("edit_schematic")
        for kind in _op_kinds():
            assert repr(kind) in text, f"op {kind} missing"
        assert "waypoints" in text
        assert "expected_sha256" in text

    def test_jobs_actions_are_enumerated_with_their_own_fields(self):
        """jobs' arguments ARE the union, so the branches carry nearly every
        field: a catalogue that rendered only the shared half would leave a
        caller with no way to learn how to address, dwell on, or cancel a job."""
        from ltspice_mcp.tools.jobs import JOBS_ACTIONS

        text = _reference.reference("jobs")
        for action in JOBS_ACTIONS:
            assert repr(action) in text, f"jobs action {action} missing"
        for field in (
            "job_id",
            "request_id",
            "timeout_s",
            "wait_for",
            "control_token",
            "circuit",
            "limit",
            "cursor",
        ):
            assert field in text, f"jobs.{field} missing from its reference"

    def test_nested_models_are_flattened_onto_dotted_paths(self):
        text = _reference.reference("verify_circuit")
        assert "render.mode" in text
        assert "render.scale" in text

    def test_enum_members_are_written_out_inline(self):
        text = _reference.reference("verify_circuit")
        assert "'with_checks'" in text
        assert "'only'" in text

    def test_defaults_and_requiredness_are_stated(self):
        text = _reference.reference("verify_circuit")
        assert "path" in text
        assert "[required]" in text
        assert "[default" in text

    def test_unknown_operation_names_the_six(self):
        with pytest.raises(ValueError, match="unknown operation") as excinfo:
            _reference.reference("run_simulation")
        for name in OPS:
            assert name in str(excinfo.value)

    def test_a_non_string_op_is_a_type_error(self):
        with pytest.raises(TypeError):
            _reference.reference(3)  # type: ignore[arg-type]


class TestMethodDocstrings:
    """Docstrings install lazily with the catalogue (the boot never pays for
    them), so the contract under test is: AFTER any public catalogue read,
    help() on a method shows the full argument tree."""

    @pytest.fixture(autouse=True)
    def _catalogue_read(self):
        # The public trigger — a bare cold process shows only signatures
        # until reference() (or any operation call) pays the tool-model
        # import that rendering needs.
        ApiMethodsMixin.reference()

    @pytest.mark.parametrize("name", OPS)
    def test_method_doc_is_non_empty_and_names_its_fields(self, name: str):
        method = getattr(ApiMethodsMixin, name)
        doc = inspect_mod.getdoc(method)
        assert doc
        operation = _reference._find(name)
        for spelled in _spelled_fields(operation.model):
            assert spelled in doc, f"{name}.{spelled} missing from its docstring"

    def test_docstring_and_reference_come_from_one_renderer(self):
        doc = inspect_mod.getdoc(ApiMethodsMixin.verify_circuit)
        assert doc is not None
        # The tree body is shared; only the header differs.
        assert "render.mode" in doc
        assert "render.mode" in _reference.reference("verify_circuit")

    def test_reference_is_reachable_without_an_engine_session(self):
        # A static method: reading the catalogue must not take this process's
        # single session lease, nor require one to have been opened.
        assert "run_experiments" in ApiMethodsMixin.reference()


class TestMethodSignatures:
    """``inspect.signature`` and ``help`` on an op show its arguments: an agent's
    first move on an unfamiliar API is exactly that, and ``**arguments`` told it
    nothing. The signature is generated from the model the call validates
    against, so it cannot drift from what the call accepts."""

    @pytest.mark.parametrize("name", _reference.op_names())
    def test_signature_names_every_top_level_field(self, name: str):
        from ltspice_mcp.lib.model_fields import field_name

        method = getattr(ApiMethodsMixin, name)
        parameters = inspect_mod.signature(method).parameters
        operation = _reference._find(name)
        import keyword

        unnameable = False
        for raw, field in operation.model.model_fields.items():
            caller_name = field_name(operation.model, raw, field)
            if keyword.iskeyword(caller_name):  # 'continue': passed by dict unpacking
                unnameable = True
                continue
            assert caller_name in parameters, (name, raw)
        # The catch-all stays only where a field cannot be a named parameter.
        assert any(p.kind is p.VAR_KEYWORD for p in parameters.values()) is unnameable, name
        assert "raw_page" in parameters

    def test_a_cold_process_gets_the_signature_on_first_read(self, tmp_path):
        """The first thing a caller tries on an unfamiliar API is
        inspect.signature or help(): both install the catalogue on that read,
        so a cold process answers with the arguments, not ``**arguments``."""
        probe = (
            "import inspect\n"
            "from ltspice_mcp.api import Api\n"
            "sig = inspect.signature(Api.run_experiments)\n"
            "print('SIG', 'circuits' in sig.parameters, 'wait' in sig.parameters)\n"
            "print('DOC', 'recipes' in (inspect.getdoc(Api.analyze_results) or ''))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )
        assert "SIG True True" in proc.stdout, proc.stdout + proc.stderr
        assert "DOC True" in proc.stdout, proc.stdout + proc.stderr

    def test_run_experiments_keeps_its_own_keywords_and_takes_the_call(self):
        parameters = inspect_mod.signature(ApiMethodsMixin.run_experiments).parameters
        assert {"wait", "detach", "raw_page", "circuits", "request_id"} <= set(parameters)
        assert parameters["circuits"].default is inspect_mod.Parameter.empty
        assert parameters["wait"].default is True


class TestCatalogueDelivery:
    """`python -m ltspice_mcp.api reference` is the same renderer behind a print,
    and reading it must stay a documentation read: no engine, no heavy imports."""

    @staticmethod
    def _probe(body: str, tmp_path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", body],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )

    def test_module_entry_point_prints_the_catalogue(self, tmp_path):
        """`python -m ltspice_mcp.api reference [OP]` prints the catalogue and
        exits 0. It is documentation, not a server start: starting the engine
        writes an ltspice-mcp.toml into the working directory, so the read must
        leave the directory exactly as it found it."""
        proc = self._probe(
            "from ltspice_mcp.api.__main__ import main\n"
            "print('RC', main(['reference', 'verify_circuit']))\n",
            tmp_path,
        )
        assert "render.mode" in proc.stdout
        assert "RC 0" in proc.stdout
        assert list(tmp_path.iterdir()) == []

    def test_module_entry_point_index_prints_the_six(self, tmp_path):
        proc = self._probe(
            "from ltspice_mcp.api.__main__ import main\nprint('RC', main(['reference']))\n",
            tmp_path,
        )
        assert "RC 0" in proc.stdout
        for name in OPS:
            assert name in proc.stdout

    def test_module_entry_point_refuses_an_unknown_operation(self, tmp_path):
        """Exit 2, and the refusal names the real operations rather than
        leaving the caller to guess."""
        proc = self._probe(
            "from ltspice_mcp.api.__main__ import main\n"
            "print('RC', main(['reference', 'run_simulation']))\n",
            tmp_path,
        )
        assert "RC 2" in proc.stdout
        for name in OPS:
            assert name in proc.stderr

    def test_no_console_script_survives_the_removed_command_line(self):
        """The `spice-mcp` command line was removed at 0.6.0; the package
        publishes exactly one console script, and `ltspice_mcp.cli` is gone.
        A reintroduced module or entry point fails here."""
        from importlib.metadata import distribution

        scripts = {
            entry.name
            for entry in distribution("ltspice-mcp").entry_points
            if entry.group == "console_scripts"
        }
        assert scripts == {"ltspice-mcp"}
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("ltspice_mcp.cli")

    def test_import_and_engine_boot_load_no_heavy_modules(self, tmp_path):
        """Neither ``import ltspice_mcp.api`` nor ``Api()`` itself may load
        the forbidden modules: the lazy package __init__ (PEP 562), the
        lazily built tool surface on SessionState, and the deferred tool
        imports in _methods together keep the engine boot at config +
        detection + registry preload. A cold subprocess is the only honest
        measurement — an in-process check would see whatever the suite
        already imported. (A catalogue CALL still derives from the live tool
        models and pays the mcp import then; that is the call's cost, not
        the boot's.)"""
        # Modules an Api() must not pay for: scipy arrives only when a
        # peak-detecting analysis actually runs; mcp only when a handler
        # builds a wire result.
        forbidden = ("scipy", "mcp")
        probe = (
            "import sys\n"
            f"FORBIDDEN = {forbidden!r}\n"
            "def heavy():\n"
            "    return sorted({m.split('.')[0] for m in sys.modules} & set(FORBIDDEN))\n"
            "import ltspice_mcp.api as api\n"
            "print('AFTER-IMPORT', heavy())\n"
            "inst = api.Api(working_dir='.')\n"
            "print('AFTER-BOOT', heavy())\n"
            "inst.close()\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )
        assert "AFTER-IMPORT []" in proc.stdout, proc.stdout + proc.stderr
        assert "AFTER-BOOT []" in proc.stdout, proc.stdout + proc.stderr

    def test_docstrings_install_on_first_deferred_tool_resolution(self, tmp_path):
        """The catalogue read is one docstring trigger; the OTHER is resolving
        a deferred tool module (what any operation call does first). Only a
        cold subprocess can see it — in-process the suite installed the docs
        long ago, so deleting the proxy-side trigger would stay green here."""
        probe = (
            "import inspect\n"
            "from ltspice_mcp.api import _methods\n"
            "_methods.experiments.RunExperimentsInput\n"
            "print('INSTALLED', _methods._method_docs_installed)\n"
            "doc = inspect.getdoc(_methods.ApiMethodsMixin.run_experiments)\n"
            "print('WARM-FULL', bool(doc) and 'wait=False' in doc)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )
        assert "INSTALLED True" in proc.stdout, proc.stdout + proc.stderr
        assert "WARM-FULL True" in proc.stdout, proc.stdout + proc.stderr

    def test_analysis_modules_keep_scipy_out_of_their_import(self, tmp_path):
        """The find_peaks imports are call-site-local so importing the three
        analysis modules stays cheap. The boot pin alone cannot notice a
        reintroduced top-level scipy import — the boot never imports these
        modules — so this pins the deferral at its actual fan-in."""
        probe = (
            "import sys\n"
            "import ltspice_mcp.lib.ac_analysis\n"
            "import ltspice_mcp.lib.ac_structure\n"
            "import ltspice_mcp.lib.signal_analysis\n"
            "print('SCIPY-LOADED', 'scipy' in {m.split('.')[0] for m in sys.modules})\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )
        assert "SCIPY-LOADED False" in proc.stdout, proc.stdout + proc.stderr

    def test_lazy_surface_covers_the_pinned_all_exactly(self):
        """Every pinned __all__ name resolves through the lazy table and the
        table advertises nothing beyond the pin — a name added to one side
        without the other fails here instead of at a user's import."""
        import ltspice_mcp.api as api

        assert set(api.__all__) == set(api._SOURCES)
        for name in api.__all__:
            assert getattr(api, name) is not None


class TestOpsUnionErrorEnumerates:
    """A mistyped op used to produce an error per branch, truncated at
    "… and 23 more"; the caller learned neither the kinds nor their own gap."""

    def test_unknown_op_error_names_every_kind(self):
        with pytest.raises(ValidationError) as excinfo:
            EditSchematicInput.model_validate({"target": "s.asc", "ops": [{"op": "bogus"}]})
        detail = compact_validation_error(excinfo.value)
        assert "more" not in detail, detail
        for kind in _op_kinds():
            assert kind in detail, f"op {kind} missing from the union error"

    def test_a_known_op_reports_against_that_kind_alone(self):
        with pytest.raises(ValidationError) as excinfo:
            EditSchematicInput.model_validate(
                {"target": "s.asc", "ops": [{"op": "add_component", "reference": "R1"}]}
            )
        detail = compact_validation_error(excinfo.value)
        assert "symbol" in detail
        assert "from_pin" not in detail, "a sibling branch's fields leaked into the error"


class TestCatalogueMatchesTheRegistry:
    """_operations() names the six ops by hand next to the registry that
    already maps them; this pin is what turns a drift into a failure."""

    def test_operation_names_equal_the_consolidated_profile(self):
        from ltspice_mcp.tools import get_tools
        from tests.conftest import CONSOLIDATED_TOOLS

        tool_defs, _dispatch = get_tools()
        # The API exposes exactly the envelope six. plot_waveform is MCP-only
        # by design: it renders an interactive client-side widget (an iframe
        # resource), which has no meaning in-process — the Python API's
        # plotting path is load_raw + the caller's own tooling.
        assert set(_reference.op_names()) == set(CONSOLIDATED_TOOLS)
        assert {tool.name for tool in tool_defs} - set(_reference.op_names()) == {"plot_waveform"}
