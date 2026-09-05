"""The shared page shape and the plain offset cursor.

Every list surface returns the same five keys, and a reader that loops on
``next_cursor`` has to be able to read it unconditionally — the cases below pin
that, plus the refusal of a cursor that is not one of ours (a silently clamped
offset would restart a listing the caller believed it was continuing).
"""

from ltspice_mcp.lib.pagination import decode_offset, encode_offset, page, retotal_page

PAGE_KEYS = {"items", "total", "returned", "truncated", "next_cursor"}


def test_a_page_always_carries_every_key():
    full, _ = page([1, 2, 3], 0, 10)
    partial, _ = page([1, 2, 3], 0, 2)
    assert set(full) == set(partial) == PAGE_KEYS
    assert full["next_cursor"] is None and partial["next_cursor"] is None


def test_page_reports_the_whole_list_and_the_slice_it_returned():
    data, next_offset = page(list(range(10)), 3, 4)
    assert data["items"] == [3, 4, 5, 6]
    assert (data["total"], data["returned"], data["truncated"]) == (10, 4, True)
    assert next_offset == 7


def test_a_final_page_is_not_truncated_and_has_no_cursor():
    data, next_offset = page(list(range(5)), 3, 4)
    assert data["items"] == [3, 4]
    assert (data["total"], data["returned"], data["truncated"]) == (5, 2, False)
    assert data["next_cursor"] is None
    assert next_offset == 5


def test_the_cursor_builder_runs_only_when_there_is_a_next_page():
    truncated, _ = page(list(range(10)), 0, 4, cursor=encode_offset)
    assert truncated["next_cursor"] == "o:4"
    complete, _ = page(list(range(3)), 0, 4, cursor=encode_offset)
    assert complete["next_cursor"] is None


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
    data, _ = page(list(range(10)), 0, 4)
    # Projection replaces the rows with rendered ones; the count of what was
    # asked about does not change with how the rows are dressed.
    next_offset = retotal_page(data, ["a", "b"], 0)
    assert data["items"] == ["a", "b"]
    assert (data["total"], data["returned"], data["truncated"]) == (10, 2, True)
    assert next_offset == 2
