"""Opaque, tamper-evident cursor codec shared by the pagination surfaces.

A cursor is a URL-safe base64 blob wrapping ``{"body": <caller dict>, "check":
canonical_hash(body)}``. ``body`` carries whatever the caller needs to resume
(a result-set id + position, or a view kind + offset); ``check`` is a canonical
SHA-256 over it, so a mutated payload fails to decode. The codec is payload-
agnostic — each caller defines and validates its own body fields — and
stdlib-only, so both ``result_store`` and ``pin_legend`` import it acyclically.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


class CursorError(ValueError):
    """A cursor did not decode or failed its integrity check."""


def encode_cursor(body: dict[str, Any]) -> str:
    """Wrap ``body`` in a checksummed, URL-safe base64 cursor."""
    envelope = {"body": body, "check": canonical_hash(body)}
    return base64.urlsafe_b64encode(canonical_json(envelope)).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> dict[str, Any]:
    """Recover and integrity-check a cursor body; raise ``CursorError`` on any fault."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        body = data["body"]
        if data["check"] != canonical_hash(body):
            raise CursorError("checksum mismatch")
        if not isinstance(body, dict):
            raise CursorError("cursor body is not an object")
        return body
    except CursorError:
        raise
    except (
        binascii.Error,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise CursorError(str(exc)) from exc
