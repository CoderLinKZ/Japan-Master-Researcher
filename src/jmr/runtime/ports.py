"""Minimal runtime ports required by the first target workflow slice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4


class StructuredModel(Protocol):
    """Model capable of returning one schema-constrained object."""

    def invoke_structured(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system_prompt: str,
        output_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    """One provider-neutral tool request emitted by a node model."""

    call_id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("call_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("arguments must be an object")
        object.__setattr__(self, "call_id", self.call_id.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolModelTurn:
    """One ReAct turn containing tool requests or a final structured result.

    Provider adapters translate native tool-use responses into this value.
    Exactly one branch must be present so a graph node never has to interpret
    an ambiguous "call tools and finish" response.
    """

    tool_calls: tuple[ModelToolCall, ...] = ()
    final_result: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        calls = tuple(self.tool_calls)
        if not all(isinstance(call, ModelToolCall) for call in calls):
            raise TypeError("tool_calls must contain ModelToolCall values")
        if self.final_result is not None and not isinstance(self.final_result, Mapping):
            raise TypeError("final_result must be an object")
        if bool(calls) == (self.final_result is not None):
            raise ValueError(
                "a tool model turn must contain either tool_calls or final_result"
            )
        object.__setattr__(self, "tool_calls", calls)
        if self.final_result is not None:
            object.__setattr__(self, "final_result", dict(self.final_result))


class ToolCallingModel(Protocol):
    """Model capable of one provider-neutral ReAct/tool-calling turn."""

    def invoke_tool_step(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system_prompt: str,
        tools: Sequence[Mapping[str, Any]],
        output_schema: Mapping[str, Any],
    ) -> ToolModelTurn: ...


class ModelRegistry(Protocol):
    """Resolve the configured structured or tool model for a graph node."""

    def for_node(self, node_name: str) -> StructuredModel | ToolCallingModel: ...


class CaseRepository(Protocol):
    """Persistence boundary needed by case bootstrap and input validation."""

    def create_case(
        self,
        *,
        user_id: str,
        case_id: str,
        schema_version: int,
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def get_case(
        self,
        *,
        user_id: str,
        case_id: str,
    ) -> Mapping[str, Any] | None: ...

    def update_target(
        self,
        *,
        user_id: str,
        case_id: str,
        target: Mapping[str, Any],
        expected_version: int,
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def record_stage_transition(
        self,
        *,
        user_id: str,
        case_id: str,
        transition: Mapping[str, Any],
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def save_research_plan(
        self,
        *,
        user_id: str,
        case_id: str,
        plan: Mapping[str, Any],
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def get_research_plan(
        self, *, user_id: str, case_id: str, research_plan_id: str
    ) -> Mapping[str, Any]: ...

    def save_retrieval_run(
        self,
        *,
        user_id: str,
        case_id: str,
        worker_kind: str,
        status: str,
        result_summary: Mapping[str, Any] | None = None,
        source_object: Mapping[str, Any] | None = None,
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def get_retrieval_runs(
        self,
        *,
        user_id: str,
        case_id: str,
        retrieval_run_ids: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]: ...

    def save_verified_evidence(
        self,
        *,
        user_id: str,
        case_id: str,
        evidence_ids: Sequence[str],
        summary: Mapping[str, Any],
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def get_verified_evidence_bundle(
        self,
        *,
        user_id: str,
        case_id: str,
        bundle_id: str,
    ) -> Mapping[str, Any]: ...

    def save_case_memory_selection(
        self,
        *,
        user_id: str,
        case_id: str,
        memory_ids: Sequence[str],
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def save_direction_batch(
        self,
        *,
        user_id: str,
        case_id: str,
        directions: Sequence[Mapping[str, Any]],
        evidence_bundle_id: str | None,
        revision_round: int,
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def get_direction_batch(
        self,
        *,
        user_id: str,
        case_id: str,
        direction_batch_id: str,
    ) -> Mapping[str, Any]: ...

    def save_direction_selection(
        self,
        *,
        user_id: str,
        case_id: str,
        direction_batch_id: str,
        selected_direction_ids: Sequence[str],
        custom_direction: Mapping[str, Any] | None,
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def save_outreach_iteration(
        self,
        *,
        user_id: str,
        case_id: str,
        content: str,
        citation_ids: Sequence[str],
        parent_iteration_id: str | None,
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def get_outreach_iteration(
        self,
        *,
        user_id: str,
        case_id: str,
        iteration_id: str,
    ) -> Mapping[str, Any]: ...

    def list_review_results(
        self,
        *,
        user_id: str,
        case_id: str,
        iteration_id: str,
    ) -> Sequence[Mapping[str, Any]]: ...

    def save_review_result(
        self,
        *,
        user_id: str,
        case_id: str,
        iteration_id: str,
        reviewer_kind: str,
        status: str,
        result: Mapping[str, Any],
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def complete_case(
        self,
        *,
        user_id: str,
        case_id: str,
        final_iteration_id: str,
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...


class MCPGateway(Protocol):
    """Expose only the exact MCP capabilities allowed for one node."""

    def tools_for_node(
        self,
        node_name: str,
    ) -> Sequence[Mapping[str, Any]]: ...

    def call_tool(
        self,
        *,
        node_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        trusted_context: Mapping[str, str],
    ) -> Mapping[str, Any]: ...


class MemoryStore(Protocol):
    """Cross-case store port scoped by a user-owned namespace."""

    def list(
        self,
        *,
        namespace: tuple[str, ...],
        filters: Mapping[str, Any],
        limit: int,
    ) -> Sequence[Mapping[str, Any]]: ...

    def get(
        self,
        *,
        namespace: tuple[str, ...],
        key: str,
    ) -> Mapping[str, Any] | None: ...


class ObjectStore(Protocol):
    """Large-object storage kept outside PostgreSQL business tables."""

    def put(
        self,
        *,
        object_key: str,
        content: bytes,
        content_type: str,
    ) -> str: ...

    def get(self, *, object_uri: str) -> bytes: ...


class RetryPolicy(Protocol):
    """Deterministic retry decision injected into host-side operations."""

    def should_retry(self, error: Exception, attempt: int) -> bool: ...

    def delay_seconds(self, error: Exception, attempt: int) -> float: ...


class SecurityPolicy(Protocol):
    """Authorize a trusted principal before case or object access."""

    def authorize_case(
        self,
        *,
        principal_user_id: str,
        user_id: str,
        case_id: str,
    ) -> None: ...


class EventSink(Protocol):
    """Structured observability boundary for logs, metrics, and progress."""

    def emit(self, event: Mapping[str, Any]) -> None: ...


class Clock(Protocol):
    """Injectable wall clock used for durable payloads and audit records."""

    def now(self) -> datetime: ...


class IdGenerator(Protocol):
    """Injectable stable identifier generator."""

    def new_id(self, prefix: str) -> str: ...


class SystemClock:
    """UTC system clock used outside deterministic tests."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class UUIDGenerator:
    """Generate opaque identifiers without encoding business data."""

    def new_id(self, prefix: str) -> str:
        normalized_prefix = prefix.strip()
        if not normalized_prefix:
            raise ValueError("prefix must be a non-empty string")
        return f"{normalized_prefix}_{uuid4().hex}"
