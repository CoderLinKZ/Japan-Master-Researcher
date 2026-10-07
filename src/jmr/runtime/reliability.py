"""Bounded retry classification shared by model, API, and database adapters."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

RETRYABLE_HTTP_STATUSES = frozenset({429, 502, 503, 504})
NON_RETRYABLE_HTTP_STATUSES = frozenset({400, 401, 402, 403, 404, 407, 451})
RETRYABLE_SQLSTATES = frozenset(
    {
        "40001",  # serialization_failure
        "40P01",  # deadlock_detected
        "53300",  # too_many_connections
        "57P01",  # admin_shutdown
        "57P02",  # crash_shutdown
        "57P03",  # cannot_connect_now
    }
)


class OperationKind(StrEnum):
    MODEL = "model"
    API = "api"
    DATABASE = "database"


@dataclass(frozen=True, slots=True)
class BoundedRetryPolicy:
    """Retry only transient failures and cap server-requested waits."""

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds cannot be negative")
        if not 0 <= self.max_delay_seconds <= 30:
            raise ValueError("max_delay_seconds must be between 0 and 30")

    def should_retry(self, error: Exception, attempt: int) -> bool:
        return attempt < self.max_attempts and is_transient_error(error)

    def delay_seconds(self, error: Exception, attempt: int) -> float:
        requested = retry_after_seconds(error)
        if requested is None:
            requested = self.base_delay_seconds * (2 ** max(0, attempt - 1))
        return min(max(0.0, requested), self.max_delay_seconds)


def call_with_retry[T](
    operation: Callable[[], T],
    *,
    policy: BoundedRetryPolicy | Any | None = None,
    idempotent: bool,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, float, str], None] | None = None,
) -> T:
    """Run an operation with bounded retries when replay is explicitly safe."""

    selected = policy or BoundedRetryPolicy()
    max_attempts = int(getattr(selected, "max_attempts", 3))
    max_attempts = min(max(1, max_attempts), 3)
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            should_retry = bool(selected.should_retry(exc, attempt))
            if not idempotent or attempt >= max_attempts or not should_retry:
                raise
            delay = min(
                max(0.0, float(selected.delay_seconds(exc, attempt))),
                30.0,
            )
            if on_retry is not None:
                on_retry(attempt, delay, type(exc).__name__)
            sleep(delay)
    raise AssertionError("retry loop exited unexpectedly")


def is_transient_error(error: Exception) -> bool:
    """Classify retryability without matching secret-bearing error messages."""

    if isinstance(error, (PermissionError, ValueError, TypeError)):
        return False
    status = http_status(error)
    if status in NON_RETRYABLE_HTTP_STATUSES:
        return False
    if status is not None:
        return status in RETRYABLE_HTTP_STATUSES
    sqlstate = getattr(error, "sqlstate", None)
    if isinstance(sqlstate, str):
        return sqlstate in RETRYABLE_SQLSTATES or sqlstate.startswith("08")
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    error_name = type(error).__name__.casefold()
    blocked_markers = (
        "auth",
        "permission",
        "schema",
        "ssrf",
        "robots",
        "paywall",
        "validation",
    )
    if any(marker in error_name for marker in blocked_markers):
        return False
    return "timeout" in error_name or "connection" in error_name


def http_status(error: Exception) -> int | None:
    """Extract an HTTP status from common SDK exception shapes."""

    candidates: list[Any] = [
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
        _nested_value(getattr(error, "body", None), "status"),
        _nested_value(getattr(error, "body", None), "status_code"),
    ]
    for argument in getattr(error, "args", ()):
        candidates.extend(
            [
                _nested_value(argument, "status"),
                _nested_value(argument, "status_code"),
            ]
        )
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def retry_after_seconds(error: Exception) -> float | None:
    """Read numeric Retry-After hints; HTTP-date parsing stays adapter-owned."""

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    candidates: list[Any] = []
    if isinstance(headers, Mapping):
        candidates.extend([headers.get("retry-after"), headers.get("Retry-After")])
    candidates.append(_nested_value(getattr(error, "body", None), "retry_after"))
    for argument in getattr(error, "args", ()):
        candidates.append(_nested_value(argument, "retry_after"))
    for value in candidates:
        try:
            delay = float(value)
        except (TypeError, ValueError):
            continue
        if delay >= 0:
            return delay
    return None


def _nested_value(value: Any, key: str, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for item in value.values():
            found = _nested_value(item, key, depth + 1)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _nested_value(item, key, depth + 1)
            if found is not None:
                return found
    return None
