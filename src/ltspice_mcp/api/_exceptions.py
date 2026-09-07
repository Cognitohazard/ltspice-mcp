"""Exceptions raised by the in-process Python API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ApiError(Exception):
    """Base class for Python API failures."""


class ApiCallError(ApiError):
    """A completed API invocation returned a call-level error envelope."""

    def __init__(
        self,
        message: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.payload = dict(payload or {})
        error = self.payload.get("error")
        details = error if isinstance(error, Mapping) else self.payload
        self.code = details.get("code")
        self.commit_state = details.get("commit_state")
        self.job_id = details.get("job_id", self.payload.get("job_id"))
        self.control_token = details.get(
            "control_token",
            self.payload.get("control_token"),
        )


class ApiSessionError(ApiError):
    """The process cannot safely use or create the requested engine session."""


class ApiClosedError(ApiSessionError):
    """The API session is closing or has already closed."""


class ApiInterrupted(KeyboardInterrupt):
    """A caller interrupt occurred after an operation produced a durable handle."""

    def __init__(
        self,
        message: str = "API call interrupted",
        *,
        receipt: Mapping[str, Any] | None = None,
        job_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.receipt = dict(receipt) if receipt is not None else None
        receipt_job_id = (self.receipt or {}).get("job_id")
        self.job_id = job_id or (receipt_job_id if isinstance(receipt_job_id, str) else None)


class ApiInternalError(ApiError):
    """The API bridge encountered an invalid internal result or state."""


class ApiValidationError(ValueError):
    """Python arguments failed the consolidated operation's validation."""
