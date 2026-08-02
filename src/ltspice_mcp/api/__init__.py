"""Public in-process Python API for the six consolidated engine operations."""

from ltspice_mcp.api._exceptions import (
    ApiCallError,
    ApiClosedError,
    ApiError,
    ApiInternalError,
    ApiInterrupted,
    ApiSessionError,
    ApiValidationError,
)
from ltspice_mcp.api._session import Api

__all__ = [
    "Api",
    "ApiCallError",
    "ApiClosedError",
    "ApiError",
    "ApiInternalError",
    "ApiInterrupted",
    "ApiSessionError",
    "ApiValidationError",
]
