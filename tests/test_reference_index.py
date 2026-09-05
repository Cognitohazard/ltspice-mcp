"""The vocabulary index behind ``inspect(kind="reference")``.

Three things have to hold. The index must cover everything the models declare
— every branch, and every tool's own arguments — because anything missing from
it is a capability a caller cannot find. Every entry must carry a readable one
line, because a name alone answers nothing. And search has to reach a recipe
both by its own name and by the plain words a person types instead — that last
one is what the whole lookup exists for.
"""

from __future__ import annotations

import pytest

from ltspice_mcp.lib.model_fields import literal_values
from ltspice_mcp.lib.recipes import DISCRIMINANTS
from ltspice_mcp.tools import reference_index
from ltspice_mcp.tools.reference_index import build_index, search_branches, table_of_contents


def _entries():
    return build_index()


def _named(tool: str, name: str) -> reference_index.BranchEntry:
    return next(e for e in _entries() if e.tool == tool and e.name == name)


def _names(tool: str) -> set[str]:
    return {entry.name for entry in _entries() if entry.tool == tool}


class TestIndexCompleteness:
    """Every discriminator literal the models declare has an entry.

    Derived from the live unions, not from a list kept alongside them: a recipe
    added to ``Recipe`` or an op added to ``SchematicOp`` fails here until it is
    in the index.
    """

    def test_every_recipe_metric_is_indexed(self):
        assert set(DISCRIMINANTS) <= _names("analyze_results")

    def test_every_schematic_op_is_indexed(self):
        from ltspice_mcp.lib.schematic_ops import SchematicOp

        declared = {
            values[0]
            for model in reference_index._members(SchematicOp)
            if (values := literal_values(model, "op"))
        }
        assert declared <= _names("edit_schematic")

    def test_every_inspect_kind_is_indexed(self):
        from ltspice_mcp.tools.inspect_tools import SUPPORTED_KINDS

        assert set(SUPPORTED_KINDS) <= _names("inspect")

    def test_every_jobs_action_is_indexed(self):
        from ltspice_mcp.tools.jobs import JOBS_ACTIONS

        assert set(JOBS_ACTIONS) <= _names("jobs")

    def test_every_variation_kind_and_random_rule_is_indexed(self):
        from ltspice_mcp.lib.variations import RandomRule, Variation

        declared = {
            values[0]
            for union, tag in ((Variation, "kind"), (RandomRule, "rule"))
            for model in reference_index._members(union)
            if (values := literal_values(model, tag))
        }
        assert declared <= _names("run_experiments")

    def test_every_verify_check_is_indexed(self):
        """The checks are a Literal on the ``checks`` argument, so they are read
        off the input model rather than off a union of branch models."""
        from typing import get_args

        from ltspice_mcp.tools.verify import VerifyCircuitInput

        annotation = VerifyCircuitInput.model_fields["checks"].annotation
        declared = {
            value
            for member in get_args(annotation)
            for item in get_args(member)
            for value in get_args(item)
            if isinstance(value, str)
        }
        assert declared, "the checks argument no longer declares its literals"
        assert declared <= _names("verify_circuit")

    def test_every_tool_indexes_its_own_arguments(self):
        """A tool's own arguments are vocabulary too.

        The branch families describe what goes *inside* a discriminated item,
        which left 'all_steps', 'budget' and 'expected_sha256' described
        nowhere a compact session can reach: the listing strips their prose and
        no branch carries them.
        """
        from ltspice_mcp.tools import get_tools

        _, dispatch = get_tools()
        expected = {name for name, tool in dispatch.items() if tool.input_model is not None}
        assert expected, "no registered tool declares an input model"
        indexed = {entry.name for entry in _entries() if entry.family == "argument" and entry.tool}
        assert indexed == expected
        for entry in _entries():
            if entry.family == "argument":
                assert entry.tool == entry.name, "an argument table is named for its tool"
                assert entry.fields, f"{entry.name} indexed no arguments"
                assert len(entry.fields) < reference_index._MAX_FIELDS, (
                    f"{entry.name}'s argument table hit the field cap, so the "
                    "tail of it is silently missing"
                )

    def test_the_index_has_no_branch_the_models_do_not_declare(self):
        """The reverse direction: nothing is indexed that cannot be called."""
        from ltspice_mcp.lib.schematic_ops import SchematicOp
        from ltspice_mcp.lib.variations import RandomRule, Variation
        from ltspice_mcp.tools import get_tools
        from ltspice_mcp.tools.inspect_tools import SUPPORTED_KINDS
        from ltspice_mcp.tools.jobs import JOBS_ACTIONS
        from ltspice_mcp.tools.verify import CHECK_ORDER

        _, dispatch = get_tools()
        declared = (
            {(name, name) for name, tool in dispatch.items() if tool.input_model is not None}
            | {("analyze_results", metric) for metric in DISCRIMINANTS}
            | {("inspect", kind) for kind in SUPPORTED_KINDS}
            | {("jobs", action) for action in JOBS_ACTIONS}
            | {("verify_circuit", check) for check in CHECK_ORDER}
            | {
                ("edit_schematic", values[0])
                for model in reference_index._members(SchematicOp)
                if (values := literal_values(model, "op"))
            }
            | {
                ("run_experiments", values[0])
                for union, tag in ((Variation, "kind"), (RandomRule, "rule"))
                for model in reference_index._members(union)
                if (values := literal_values(model, tag))
            }
        )
        assert {(entry.tool, entry.name) for entry in _entries()} == declared


class TestEntryShape:
    @pytest.mark.parametrize("entry", _entries(), ids=lambda e: f"{e.tool}.{e.name}")
    def test_every_branch_carries_a_summary_and_a_call(self, entry):
        assert entry.summary.strip(), (
            f"{entry.tool}.{entry.name} has no one-line summary — give the model a "
            "docstring, or add one to reference_index._SUMMARIES"
        )
        assert entry.name in entry.call

    def test_fields_carry_types_defaults_and_units(self):
        stability = _named("analyze_results", "stability")
        by_name = {field.name: field for field in stability.fields}
        assert "signal" in by_name and by_name["signal"].required
        assert by_name["signal"].type == "string"
        # Inherited reduction fields are part of what the branch takes.
        assert {"key", "reduce", "field", "spec.min", "spec.max"} <= set(by_name)
        assert by_name["reduce"].default == "empty"
        assert by_name["sources"].default == "null"
        # .step selection is one choice for the whole call, so it is on
        # analyze_results itself and not restated on every recipe branch.
        assert {"step", "all_steps"}.isdisjoint(by_name)
        # The discriminator is not an argument the caller chooses twice.
        assert "metric" not in by_name

    def test_a_field_entry_carries_the_whole_description(self):
        """The lookup is the only channel this prose has on the compact
        listing, so a first-sentence cut leaves the rest reaching nobody.

        'applies_to' is the case that showed it: the sentence saying it is not
        a device filter is the second one, and losing it is how a caller sends
        device references to a circuit-id argument.
        """
        from ltspice_mcp.lib.model_fields import describe_field
        from ltspice_mcp.lib.variations import AssignVariation

        assign = _named("run_experiments", "assign")
        indexed = {field.name: field.description for field in assign.fields}
        for name, field in AssignVariation.model_fields.items():
            declared = " ".join(describe_field(field).split())
            if declared and name in indexed:
                assert indexed[name] == declared, name
        assert "not a device filter" in indexed["applies_to"].lower()

    def test_bounds_are_read_off_the_model(self):
        thd = _named("analyze_results", "thd")
        harmonics = next(field for field in thd.fields if field.name == "harmonics")
        assert ">= 1" in harmonics.type and "<= 50" in harmonics.type

    def test_a_check_with_no_model_still_names_its_argument(self):
        compare = _named("verify_circuit", "compare")
        names = {field.name for field in compare.fields}
        assert "compare.reference" in names
        assert not _named("verify_circuit", "syntax").fields

    def test_serialization_omits_fields_for_the_contents_listing(self):
        entry = _named("jobs", "wait").as_dict(with_fields=False)
        assert set(entry) == {"tool", "family", "name", "summary"}


class TestTableOfContents:
    def test_groups_every_branch_by_tool_and_family(self):
        toc = table_of_contents()
        assert [group["tool"] for group in toc][:2] == ["run_experiments", "run_experiments"]
        listed = sum(len(group["branches"]) for group in toc)
        assert listed == len(_entries())
        for group in toc:
            for branch in group["branches"]:
                assert set(branch) == {"name", "summary"}
                assert branch["summary"]


#: What a person types when they want each recipe, written as the question
#: rather than as the discriminant. Search has to reach the recipe from here —
#: this is the whole reason the lookup exists, and a ranking change that broke
#: it would otherwise pass every other test in the suite.
RECIPE_SEARCH_PHRASES: dict[str, str] = {
    "summary": "what is in this run",
    "measurements": "read the .meas results",
    "value": "evaluate an expression",
    "signal_stats": "output ripple rms",
    "edges": "rise time",
    "timing": "propagation delay",
    "periodic": "duty cycle",
    "transient_response": "overshoot and settling time",
    "thd": "distortion",
    "bode_filter": "cutoff frequency",
    "bode_point": "gain at a frequency",
    "bode_crossing": "0 db crossing",
    "bode_slope": "roll off db per decade",
    "stability": "phase margin",
    "ac_structure": "poles and zeros",
    "resonance": "q factor",
    "return_loss": "vswr",
    "noise_integral": "integrated noise",
    "operating_point": "bias point gm",
    "waveform": "export the samples to csv",
    "plot": "chart the waveform",
}


class TestSearch:
    @pytest.mark.parametrize("metric", DISCRIMINANTS)
    def test_a_recipe_is_found_by_its_own_name(self, metric: str):
        hits, total = search_branches(metric, limit=5)
        assert total >= 1
        assert hits[0].name == metric and hits[0].tool == "analyze_results", (
            f"searching {metric!r} ranked {hits[0].tool}.{hits[0].name} first"
        )

    def test_the_phrase_table_covers_every_recipe(self):
        assert set(RECIPE_SEARCH_PHRASES) == set(DISCRIMINANTS)

    @pytest.mark.parametrize(("metric", "phrase"), sorted(RECIPE_SEARCH_PHRASES.items()))
    def test_a_recipe_is_found_by_the_words_a_person_types(self, metric: str, phrase: str):
        hits, _ = search_branches(phrase, limit=5)
        found = [hit for hit in hits if hit.tool == "analyze_results" and hit.name == metric]
        assert found, f"{phrase!r} did not reach the {metric} recipe; it returned " + ", ".join(
            f"{hit.tool}.{hit.name}" for hit in hits
        )

    def test_ranking_prefers_the_name_over_a_field_or_prose_hit(self):
        hits, _ = search_branches("waveform", limit=5)
        assert hits[0].name == "waveform"

    def test_search_is_deterministic(self):
        assert [entry.name for entry in search_branches("gain", limit=8)[0]] == [
            entry.name for entry in search_branches("gain", limit=8)[0]
        ]

    def test_limit_bounds_the_hits_but_not_the_count(self):
        hits, total = search_branches("signal", limit=2)
        assert len(hits) == 2
        assert total > 2

    def test_a_query_that_matches_nothing_returns_nothing(self):
        hits, total = search_branches("xyzzy quuxbar", limit=5)
        assert hits == [] and total == 0

    def test_a_tool_level_argument_is_reachable_by_its_own_name(self):
        """The branch families cover what goes inside a discriminated item.

        A tool's own arguments sat on neither channel a compact session has:
        the listing strips their prose, and no branch declares them. Searching
        for one now lands on its tool's argument table.
        """
        for argument, tool in (
            ("all_steps", "analyze_results"),
            ("expected_sha256", "edit_schematic"),
            ("allow_live_includes", "run_experiments"),
            ("export_to", "verify_circuit"),
        ):
            hits, _ = search_branches(argument, limit=5)
            found = next(
                (hit for hit in hits if hit.tool == tool and hit.family == "argument"), None
            )
            assert found is not None, f"{argument!r} returned " + ", ".join(
                f"{hit.tool}.{hit.name}" for hit in hits
            )
            assert argument in {field.name for field in found.fields}

    def test_other_tools_are_reachable_too(self):
        for phrase, expected in (
            ("connect two pins with a wire", ("edit_schematic", "wire_pins")),
            ("monte carlo", ("run_experiments", "random")),
            ("cancel a run", ("jobs", "cancel")),
            ("trace a net", ("inspect", "net")),
            ("pelgrom mismatch", ("run_experiments", "mismatch")),
        ):
            hits, _ = search_branches(phrase, limit=5)
            assert expected in [(hit.tool, hit.name) for hit in hits], (
                f"{phrase!r} returned " + ", ".join(f"{h.tool}.{h.name}" for h in hits)
            )
