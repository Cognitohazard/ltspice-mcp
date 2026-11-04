"""A misused result source is refused with one exception type on every method.

The same mistake, a case selector with a file or a source named twice or not
at all, raises ``ApiValidationError`` (a ``ValueError``) whether the method is
a result primitive or an operation, so one ``except`` clause catches it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.api import ApiValidationError
from ltspice_mcp.state import SessionState
from tests.conftest import SyncApi, stage_recorded_fixture

_Misuse = Callable[[SyncApi, Path], Any]


def _primitive_cases() -> list[tuple[str, _Misuse]]:
    cases: list[tuple[str, _Misuse]] = []
    for method in ("load_raw", "measurements", "device_op_points"):
        cases += [
            (f"{method}: neither", lambda api, _f, m=method: getattr(api, m)()),
            (
                f"{method}: both",
                lambda api, f, m=method: getattr(api, m)(path=f, job_id="exp_x"),
            ),
            (f"{method}: empty path", lambda api, _f, m=method: getattr(api, m)(path=" ")),
            (
                f"{method}: case_id with a path",
                lambda api, f, m=method: getattr(api, m)(path=f, case_id="c0"),
            ),
            (
                f"{method}: run_index with a path",
                lambda api, f, m=method: getattr(api, m)(path=f, run_index=1),
            ),
        ]
    cases.append(
        ("load_raw: other_raw with a path", lambda api, f: api.load_raw(f, other_raw="op"))
    )
    cases.append(
        (
            "analyze_results: case_ids with a path",
            lambda api, f: api.analyze_results(
                sources=[{"path": str(f), "runs": {"case_ids": ["c0"]}}],
                recipes=[{"key": "k", "metric": "summary"}],
            ),
        )
    )
    return cases


@pytest.mark.parametrize(("name", "misuse"), _primitive_cases(), ids=lambda v: v)
def test_a_misused_source_raises_api_validation_error(
    name: str, misuse: _Misuse, state_no_sim: SessionState, work_dir: Path
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    with pytest.raises(ApiValidationError) as caught:
        misuse(SyncApi(state_no_sim), raw)
    assert isinstance(caught.value, ValueError)
    assert not isinstance(caught.value, TypeError)
