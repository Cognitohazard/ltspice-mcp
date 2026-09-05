"""The shared page shape and the plain offset cursor.

Every list surface returns the same five keys, and a reader that loops on
``next_cursor`` has to be able to read it unconditionally — the cases below pin
that, the guards a caller-supplied offset and limit need, and the refusal of a
cursor that is not one of ours (a silently clamped offset would restart a
listing the caller believed it was continuing).
"""

from ltspice_mcp.lib.pagination import (
    decode_offset,
    encode_offset,
    page,
    page_of,
    retotal_page,
    unpaged,
)

PAGE_KEYS = {"items", "total", "returned", "truncated", "next_cursor"}


def test_a_page_always_carries_every_key():
    full = page([1, 2, 3], limit=10)
    partial = page([1, 2, 3], limit=2)
    assert set(full) == set(partial) == PAGE_KEYS
    assert full["next_cursor"] is None
    assert partial["next_cursor"] == "o:2"


def test_page_reports_the_whole_list_and_the_slice_it_returned():
    data = page(list(range(10)), offset=3, limit=4)
    assert data["items"] == [3, 4, 5, 6]
    assert (data["total"], data["returned"], data["truncated"]) == (10, 4, True)
    assert data["next_cursor"] == "o:7"


def test_a_final_page_is_not_truncated_and_has_no_cursor():
    data = page(list(range(5)), offset=3, limit=4)
    assert data["items"] == [3, 4]
    assert (data["total"], data["returned"], data["truncated"]) == (5, 2, False)
    assert data["next_cursor"] is None


def test_limit_zero_is_floored_and_advances():
    # A zero limit would report truncated with a cursor pointing back at the
    # same offset — a pagination loop that never advances.
    first = page(list(range(10)), limit=0)
    assert first["items"] == [0]
    assert first["truncated"] is True
    assert first["next_cursor"] == "o:1"


def test_negative_limit_is_floored():
    assert page(list(range(10)), offset=2, limit=-5)["items"] == [2]


def test_an_offset_past_the_end_yields_an_empty_last_page():
    last = page(list(range(3)), offset=99, limit=10)
    assert last["items"] == [] and last["truncated"] is False
    assert last["total"] == 3 and last["next_cursor"] is None


def test_a_surface_that_mints_its_own_cursor_gets_a_null_one():
    # The analysis pages carry a work position alongside the row offset, so
    # they fill next_cursor in themselves rather than take the o: form.
    data = page_of([1, 2], offset=0, total=9, cursor=None)
    assert data["truncated"] is True
    assert data["next_cursor"] is None


def test_page_of_reports_a_window_a_caller_already_cut():
    data = page_of(["c", "d"], offset=2, total=10)
    assert (data["total"], data["returned"], data["truncated"]) == (10, 2, True)
    assert data["next_cursor"] == "o:4"


def test_unpaged_carries_everything_and_continues_nowhere():
    data = unpaged(list(range(4)))
    assert data["items"] == [0, 1, 2, 3]
    assert (data["total"], data["returned"], data["truncated"]) == (4, 4, False)
    assert data["next_cursor"] is None


def test_offset_cursor_round_trips():
    assert decode_offset(encode_offset(0)) == 0
    assert decode_offset(encode_offset(4210)) == 4210


def test_an_absent_cursor_is_the_first_page():
    assert decode_offset(None) == 0


def test_a_cursor_that_is_not_ours_is_refused_rather_than_clamped():
    # None, not 0: a caller resuming with a corrupted or foreign cursor must be
    # told, never silently handed page one of the same listing again.
    for bad in ("", "o:", "o:-3", "o:12x", "x:12", "12", "o:1:2"):
        assert decode_offset(bad) is None


def test_retotal_reconciles_rows_a_caller_replaced_without_moving_the_total():
    data = page(list(range(10)), limit=4)
    # Projection replaces the rows with rendered ones; the count of what was
    # asked about does not change with how the rows are dressed.
    next_offset = retotal_page(data, ["a", "b"], 0)
    assert data["items"] == ["a", "b"]
    assert (data["total"], data["returned"], data["truncated"]) == (10, 2, True)
    assert data["next_cursor"] is None
    assert next_offset == 2
