"""Error type and JSON error format of the web app's REST API.

Every non-2xx JSON response has the shape::

    {"error": {"code": "validation_error",
               "message": "2 problems with the submission",
               "details": [{"field": "params.fmax", "message": "must be > 0"}]}}

``details`` is non-empty for ``validation_error`` and ``[]`` otherwise (one
exception: a ``not_found`` for an unknown upload id names the offending field).

This module imports only the standard library, so it is safe to import
without the ``webapp`` extra installed.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# HTTP status -> default error code (PLAN section 5).
STATUS_CODES: Dict[int, str] = {
    400: "bad_request",
    403: "forbidden",
    404: "not_found",
    405: "bad_request",
    409: "conflict",
    413: "too_large",
    422: "validation_error",
    500: "internal_error",
}


def code_for_status(status: int) -> str:
    """Return the error code used for an HTTP status (``bad_request`` for unknown 4xx)."""
    if status in STATUS_CODES:
        return STATUS_CODES[status]
    return "internal_error" if status >= 500 else "bad_request"


def error_body(code: str, message: str, details: Optional[List[dict]] = None) -> dict:
    """Build the JSON error body.

    Parameters
    ----------
    code : str
        Machine-readable error code (``bad_request``, ``not_found``, ...).
    message : str
        Human-readable summary.
    details : list of dict, optional
        ``{"field": str, "message": str}`` entries.

    Returns
    -------
    dict
        ``{"error": {"code", "message", "details"}}``.
    """
    return {"error": {"code": code, "message": str(message),
                      "details": list(details or [])}}


class ApiError(Exception):
    """An error the API turns into a JSON error response.

    Parameters
    ----------
    status : int
        HTTP status code.
    message : str
        Human-readable summary.
    code : str, optional
        Error code; derived from ``status`` when omitted.
    details : list of dict, optional
        Field-level problems, each ``{"field": str, "message": str}``.
    """

    def __init__(self, status: int, message: str, code: Optional[str] = None,
                 details: Optional[List[dict]] = None):
        super().__init__(message)
        self.status = int(status)
        self.message = str(message)
        self.code = code or code_for_status(self.status)
        self.details = list(details or [])

    def body(self) -> dict:
        """The JSON body for this error."""
        return error_body(self.code, self.message, self.details)


def validation_error(details: List[dict], message: Optional[str] = None) -> ApiError:
    """A 422 ``validation_error`` with a "N problem(s)" message by default."""
    n = len(details)
    if message is None:
        message = f"{n} problem{'s' if n != 1 else ''} with the submission"
    return ApiError(422, message, "validation_error", details)
