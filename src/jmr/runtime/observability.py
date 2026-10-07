"""Structured, redacted observability primitives for the production graph.

The graph emits one small envelope instead of logging arbitrary state, model
messages, tool arguments, or exception text.  Event sinks may turn the same
envelope into JSON logs, metrics, traces, or alerts without becoming part of
the checkpointed state.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any, TextIO
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from jmr.runtime.context import JMRRuntimeContext

REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "body",
        "content",
        "cookie",
        "database_url",
        "dsn",
        "edited_text",
        "feedback",
        "messages",
        "professor",
        "applicant",
        "email",
        "phone",
        "password",
        "prompt",
        "secret",
        "text",
        "token",
        "tool_arguments",
    }
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb)://[^\s]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)
_EVENT_STATUSES = frozenset(
    {
        "STARTED",
        "RUNNING",
        "RETRYING",
        "SUCCEEDED",
        "PARTIAL",
        "INTERRUPTED",
        "SKIPPED",
        "BLOCKED",
        "FAILED",
        "COMPLETED",
    }
)


def new_run_id() -> str:
    """Return an opaque run identifier suitable for log correlation."""

    return f"run_{uuid4().hex}"


def redact_observability_value(value: Any, *, key: str = "") -> Any:
    """Return a JSON-safe value with secrets and business bodies removed."""

    normalized_key = key.casefold()
    if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
        return REDACTED
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_PATTERNS:
            redacted = pattern.sub(REDACTED, redacted)
        # Details are diagnostic labels, never a place for arbitrary bodies.
        if len(redacted.encode("utf-8")) > 512:
            return f"{redacted[:128]}...[TRUNCATED]"
        return redacted
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_observability_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [redact_observability_value(item) for item in list(value)[:20]]
    return type(value).__name__


def structured_event(
    *,
    context: JMRRuntimeContext,
    state: Mapping[str, Any],
    node: str,
    status: str,
    config: Mapping[str, Any] | None = None,
    tool: str | None = None,
    duration_ms: float = 0.0,
    retry_count: int = 0,
    failure_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable event envelope required by operations tooling."""

    normalized_status = str(status).strip().upper()
    if normalized_status not in _EVENT_STATUSES:
        # MCP response statuses use SUCCESS/NO_RESULT.  Keep the source value
        # in details while mapping it to the finite operational vocabulary.
        operational_status = {
            "SUCCESS": "SUCCEEDED",
            "NO_RESULT": "PARTIAL",
            "NEEDS_USER_CONFIRMATION": "INTERRUPTED",
            "CONFLICT": "BLOCKED",
        }.get(normalized_status, "FAILED")
        details = dict(details or {}, source_status=normalized_status)
        normalized_status = operational_status
    configurable = dict((config or {}).get("configurable", {}))
    metadata = dict(context.metadata)
    run_id = str(metadata.get("run_id") or "run_unassigned")
    trace_id = str(metadata.get("trace_id") or run_id)
    event = {
        "event_version": 1,
        "timestamp": context.clock.now().isoformat(),
        "user_id": state.get("user_id"),
        "case_id": state.get("case_id"),
        "thread_id": configurable.get("thread_id") or state.get("case_id"),
        "run_id": run_id,
        "trace_id": trace_id,
        "node": node,
        "tool": tool,
        "status": normalized_status,
        "duration_ms": round(max(0.0, float(duration_ms)), 3),
        "retry_count": max(0, int(retry_count)),
        "failure_id": failure_id,
        "details": redact_observability_value(dict(details or {})),
    }
    return event


def emit_runtime_event(
    *,
    context: JMRRuntimeContext,
    state: Mapping[str, Any],
    node: str,
    status: str,
    config: Mapping[str, Any] | None = None,
    tool: str | None = None,
    duration_ms: float = 0.0,
    retry_count: int = 0,
    failure_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Emit telemetry without allowing a sink outage to fail business work."""

    if context.event_sink is None:
        return
    event = structured_event(
        context=context,
        state=state,
        node=node,
        status=status,
        config=config,
        tool=tool,
        duration_ms=duration_ms,
        retry_count=retry_count,
        failure_id=failure_id,
        details=details,
    )
    try:
        context.event_sink.emit(event)
    except Exception:
        # Observability is deliberately fail-open. A sink outage must not
        # change the business result.
        return


def failure_reason_code(error: BaseException) -> str:
    """Classify failures without copying model output or exception text to logs."""

    names = {type(error).__name__}
    cause = getattr(error, "__cause__", None)
    if cause is not None:
        names.add(type(cause).__name__)
    if "APITimeoutError" in names or "TimeoutError" in names:
        return "MODEL_OR_NETWORK_TIMEOUT"
    if "APIConnectionError" in names:
        return "MODEL_CONNECTION_FAILED"
    if "JSONSchemaValidationError" in names and cause is not None:
        if "too many items" in str(cause):
            return "MODEL_OUTPUT_TOO_MANY_ITEMS"
    if "ModelOutputValidationError" in names or "JSONSchemaValidationError" in names:
        return "MODEL_OUTPUT_SCHEMA_INVALID"
    if any("MCP" in name for name in names):
        return "MCP_CALL_FAILED"
    return "STAGE_EXECUTION_FAILED"


def instrument_node(name: str, node: Callable[..., Mapping[str, Any]]) -> Callable:
    """Wrap a LangGraph node while retaining config/runtime injection.

    All wrappers expose the same three-argument LangGraph signature, then call
    the underlying node according to its declared parameters.  This keeps
    interrupt replay ordering unchanged: no durable write is introduced before
    the interrupt and the original ``GraphInterrupt`` is immediately re-raised.
    """

    parameter_names = tuple(inspect.signature(node).parameters)

    def instrumented(
        state: Mapping[str, Any],
        config: RunnableConfig,
        runtime: Runtime[JMRRuntimeContext],
    ) -> Mapping[str, Any]:
        context = runtime.context
        if context is None:
            raise RuntimeError("JMRRuntimeContext is required")
        started = time.monotonic()
        emit_runtime_event(
            context=context,
            state=state,
            node=name,
            status="STARTED",
            config=config,
            details={"workflow_stage": state.get("workflow_stage")},
        )
        try:
            if "config" in parameter_names and "runtime" in parameter_names:
                result = node(state, config, runtime)
            elif "runtime" in parameter_names:
                result = node(state, runtime)
            else:
                result = node(state)
        except BaseException as exc:
            duration = (time.monotonic() - started) * 1000
            if type(exc).__name__ in {"GraphInterrupt", "NodeInterrupt"}:
                emit_runtime_event(
                    context=context,
                    state=state,
                    node=name,
                    status="INTERRUPTED",
                    config=config,
                    duration_ms=duration,
                    details={"workflow_stage": state.get("workflow_stage")},
                )
            else:
                failure_id = f"failure_{uuid4().hex}"
                emit_runtime_event(
                    context=context,
                    state=state,
                    node=name,
                    status="FAILED",
                    config=config,
                    duration_ms=duration,
                    failure_id=failure_id,
                    details={
                        "workflow_stage": state.get("workflow_stage"),
                        "exception_type": type(exc).__name__,
                        "reason_code": failure_reason_code(exc),
                    },
                )
            raise
        emit_runtime_event(
            context=context,
            state=state,
            node=name,
            status="SUCCEEDED",
            config=config,
            duration_ms=(time.monotonic() - started) * 1000,
            details={"workflow_stage": state.get("workflow_stage")},
        )
        return result

    instrumented.__name__ = f"observed_{name}"
    instrumented.__qualname__ = instrumented.__name__
    instrumented.__doc__ = node.__doc__
    return instrumented


@dataclass(slots=True)
class InMemoryEventSink:
    """Thread-safe event collector for tests and local diagnostics."""

    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def emit(self, event: Mapping[str, Any]) -> None:
        safe = redact_observability_value(dict(event))
        if not isinstance(safe, dict):  # pragma: no cover - defensive invariant
            raise TypeError("event must be an object")
        with self._lock:
            self.events.append(safe)


@dataclass(slots=True)
class JsonLineEventSink:
    """Write one redacted JSON object per line to an injected stream."""

    stream: TextIO
    _lock: RLock = field(default_factory=RLock, repr=False)

    def emit(self, event: Mapping[str, Any]) -> None:
        safe = redact_observability_value(dict(event))
        with self._lock:
            self.stream.write(
                json.dumps(safe, ensure_ascii=False, sort_keys=True) + "\n"
            )
            self.stream.flush()


@dataclass(slots=True)
class ConsoleEventSink:
    """Print bounded stage and MCP progress to the backend console."""

    stream: TextIO = field(default_factory=lambda: sys.stderr)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def emit(self, event: Mapping[str, Any]) -> None:
        safe = redact_observability_value(dict(event))
        details = safe.get("details") or {}
        if not isinstance(details, Mapping):
            details = {}
        fields = [
            "[JMR]",
            f"case={safe.get('case_id') or '-'}",
            f"stage={details.get('workflow_stage') or '-'}",
            f"node={safe.get('node') or '-'}",
        ]
        if safe.get("tool"):
            fields.append(f"mcp={safe['tool']}")
        fields.append(f"status={safe.get('status') or '-'}")
        if safe.get("status") not in {"STARTED", "RUNNING"}:
            fields.append(f"duration_ms={safe.get('duration_ms', 0)}")
        if details.get("reason_code"):
            fields.append(f"reason={details['reason_code']}")
        if safe.get("failure_id"):
            fields.append(f"failure_id={safe['failure_id']}")
        with self._lock:
            self.stream.write(" ".join(fields) + "\n")
            self.stream.flush()


@dataclass(slots=True)
class MetricsEventSink:
    """Low-cardinality in-process metrics projection of structured events."""

    event_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    duration_ms: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    retry_counts: Counter[str] = field(default_factory=Counter)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def emit(self, event: Mapping[str, Any]) -> None:
        node = str(event.get("node") or "unknown")
        status = str(event.get("status") or "unknown")
        with self._lock:
            self.event_counts[(node, status)] += 1
            if status not in {"STARTED", "RUNNING"}:
                self.duration_ms[node].append(float(event.get("duration_ms") or 0))
            self.retry_counts[node] += int(event.get("retry_count") or 0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "event_counts": {
                    f"{node}:{status}": count
                    for (node, status), count in sorted(self.event_counts.items())
                },
                "duration_ms": {
                    node: {
                        "count": len(values),
                        "max": max(values, default=0.0),
                        "sum": round(sum(values), 3),
                    }
                    for node, values in sorted(self.duration_ms.items())
                },
                "retry_counts": dict(sorted(self.retry_counts.items())),
            }


@dataclass(slots=True)
class CompositeEventSink:
    """Fan out an event to logging, metrics, tracing, and alert sinks."""

    sinks: Sequence[Any]

    def emit(self, event: Mapping[str, Any]) -> None:
        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception:
                continue


@dataclass(slots=True)
class AlertingEventSink:
    """Forward actionable failures to a callback without including bodies."""

    callback: Callable[[Mapping[str, Any]], None]
    statuses: frozenset[str] = frozenset({"FAILED", "BLOCKED"})

    def emit(self, event: Mapping[str, Any]) -> None:
        if str(event.get("status")) not in self.statuses:
            return
        self.callback(
            {
                key: event.get(key)
                for key in (
                    "timestamp",
                    "case_id",
                    "thread_id",
                    "run_id",
                    "trace_id",
                    "node",
                    "tool",
                    "status",
                    "failure_id",
                )
            }
        )


def pseudonymous_identifier(value: str, *, salt: str) -> str:
    """Create a stable identifier for sinks that must not retain raw IDs."""

    if not value or not salt:
        raise ValueError("value and salt must be non-empty")
    digest = hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()
    return f"id_{digest[:16]}"


def utc_now_iso() -> str:
    """Small helper for operations code that does not own a RuntimeContext."""

    return datetime.now(UTC).isoformat()
