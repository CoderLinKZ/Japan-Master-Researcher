"""Checkpoint-safe state and reducers for the single production graph."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date
from enum import Enum
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langgraph.graph.message import add_messages

from jmr.domain import InterruptKind, MCPStatus, RunStatus, WorkflowStage

GRAPH_STATE_SCHEMA_VERSION = 3
MAX_GRAPH_MESSAGES = 32
MAX_GRAPH_MESSAGE_BYTES = 16 * 1024
MAX_GRAPH_MESSAGES_BYTES = 64 * 1024


class RetrievalRef(TypedDict):
    """Small reference produced by one parallel retrieval worker."""

    worker_kind: str
    retrieval_run_id: str
    mcp_status: MCPStatus


class InterruptDescriptor(TypedDict):
    """Checkpoint-safe description of a pending human decision."""

    kind: InterruptKind
    payload_ref: str
    resume_schema_version: int
    issued_at: str


def keep_immutable_identifier(current: str | None, update: str | None) -> str:
    """Keep an identity field immutable after its first non-empty value."""

    if current is None or current == "":
        return _non_empty_string(update, "identifier")
    normalized_current = _non_empty_string(current, "identifier")
    if update is None:
        return normalized_current
    normalized_update = _non_empty_string(update, "identifier")
    if normalized_current != normalized_update:
        raise ValueError("immutable identifier cannot be changed")
    return normalized_current


def merge_unique_strings(
    current: Sequence[str] | None,
    update: Sequence[str] | None,
) -> list[str]:
    """Append warning/error strings while preserving order and uniqueness."""

    left = _string_list(current or (), "current")
    right = _string_list(update or (), "update")
    return list(dict.fromkeys([*left, *right]))


def merge_retrieval_refs(
    current: Sequence[Mapping[str, Any]] | None,
    update: Sequence[Mapping[str, Any]] | None,
) -> list[RetrievalRef]:
    """Merge parallel worker references idempotently by durable run ID."""

    merged: list[RetrievalRef] = []
    positions: dict[tuple[str, str], int] = {}
    for index, raw_ref in enumerate([*(current or ()), *(update or ())]):
        ref = _normalize_retrieval_ref(raw_ref, f"retrieval_refs[{index}]")
        key = (ref["worker_kind"], ref["retrieval_run_id"])
        previous = positions.get(key)
        if previous is None:
            positions[key] = len(merged)
            merged.append(ref)
        else:
            merged[previous] = ref
    return merged


def add_bounded_messages(
    current: Sequence[Any] | None,
    update: Sequence[Any] | Any | None,
) -> list[AnyMessage]:
    """Apply LangGraph message semantics, then retain a bounded recent window.

    The reducer itself owns the limit so repeated invocations cannot grow a
    checkpoint indefinitely after the initial state has been validated.
    """

    merged = add_messages(list(current or ()), update or [])
    selected: list[AnyMessage] = []
    total_bytes = 0
    for message in reversed(merged):
        value = convert_to_openai_messages(message, include_id=True)
        plain = _to_json_value(value, path="messages")
        encoded_size = len(
            json.dumps(
                plain,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        )
        if encoded_size > MAX_GRAPH_MESSAGE_BYTES:
            raise ValueError("one graph message exceeds the size limit")
        if (
            len(selected) >= MAX_GRAPH_MESSAGES
            or total_bytes + encoded_size > MAX_GRAPH_MESSAGES_BYTES
        ):
            break
        selected.append(message)
        total_bytes += encoded_size
    selected.reverse()
    return selected


class JMRGraphState(TypedDict, total=False):
    """Durable graph contract containing only execution state and IDs."""

    schema_version: int
    user_id: Annotated[str, keep_immutable_identifier]
    case_id: Annotated[str, keep_immutable_identifier]
    workflow_stage: WorkflowStage
    run_status: RunStatus
    messages: Annotated[list[AnyMessage], add_bounded_messages]
    case_context_id: str | None
    research_plan_id: str | None
    retrieval_refs: Annotated[list[RetrievalRef], merge_retrieval_refs]
    verified_evidence_bundle_id: str | None
    selected_memory_ids: list[str]
    direction_batch_id: str | None
    selected_direction_ids: list[str]
    current_outreach_iteration_id: str | None
    generation_revision_round: int
    review_round: int
    pending_interrupt: InterruptDescriptor | None
    warnings: Annotated[list[str], merge_unique_strings]
    errors: Annotated[list[str], merge_unique_strings]


_STATE_FIELDS = frozenset(JMRGraphState.__annotations__)


def create_jmr_graph_state(
    user_id: str,
    case_id: str,
    *,
    messages: Sequence[Mapping[str, Any] | Any] = (),
    workflow_stage: WorkflowStage | str = WorkflowStage.VALIDATING_APPLICATION_INPUTS,
    run_status: RunStatus | str = RunStatus.READY,
) -> JMRGraphState:
    """Create the durable portion of a new production state."""

    return normalize_jmr_graph_state(
        {
            "schema_version": GRAPH_STATE_SCHEMA_VERSION,
            "user_id": user_id,
            "case_id": case_id,
            "workflow_stage": workflow_stage,
            "run_status": run_status,
            "messages": list(messages),
            "case_context_id": None,
            "research_plan_id": None,
            "retrieval_refs": [],
            "verified_evidence_bundle_id": None,
            "selected_memory_ids": [],
            "direction_batch_id": None,
            "selected_direction_ids": [],
            "current_outreach_iteration_id": None,
            "generation_revision_round": 0,
            "review_round": 0,
            "pending_interrupt": None,
            "warnings": [],
            "errors": [],
        }
    )


def normalize_jmr_graph_state(state: Mapping[str, Any]) -> JMRGraphState:
    """Validate durable state and reject bodies or runtime resources."""

    if not isinstance(state, Mapping):
        raise TypeError("graph state must be a mapping")
    unknown_fields = set(state) - _STATE_FIELDS
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"graph state contains unknown field(s): {names}")
    missing_fields = _STATE_FIELDS - set(state)
    if missing_fields:
        names = ", ".join(sorted(missing_fields))
        raise ValueError(f"graph state is missing field(s): {names}")
    _schema_version(state.get("schema_version"))
    try:
        workflow_stage = WorkflowStage(state["workflow_stage"])
    except (TypeError, ValueError) as exc:
        raise ValueError("workflow_stage has an unknown value") from exc
    try:
        run_status = RunStatus(state["run_status"])
    except (TypeError, ValueError) as exc:
        raise ValueError("run_status has an unknown value") from exc

    return {
        "schema_version": GRAPH_STATE_SCHEMA_VERSION,
        "user_id": _identifier(state.get("user_id"), "user_id"),
        "case_id": _identifier(state.get("case_id"), "case_id"),
        "workflow_stage": workflow_stage.value,
        "run_status": run_status.value,
        "messages": normalize_messages(state["messages"]),
        "case_context_id": _optional_identifier(
            state.get("case_context_id"), "case_context_id"
        ),
        "research_plan_id": _optional_identifier(
            state.get("research_plan_id"), "research_plan_id"
        ),
        "retrieval_refs": merge_retrieval_refs((), state["retrieval_refs"]),
        "verified_evidence_bundle_id": _optional_identifier(
            state.get("verified_evidence_bundle_id"),
            "verified_evidence_bundle_id",
        ),
        "selected_memory_ids": _identifier_list(
            state["selected_memory_ids"], "selected_memory_ids"
        ),
        "direction_batch_id": _optional_identifier(
            state.get("direction_batch_id"), "direction_batch_id"
        ),
        "selected_direction_ids": _identifier_list(
            state["selected_direction_ids"], "selected_direction_ids"
        ),
        "current_outreach_iteration_id": _optional_identifier(
            state.get("current_outreach_iteration_id"),
            "current_outreach_iteration_id",
        ),
        "generation_revision_round": _bounded_round(
            state["generation_revision_round"], "generation_revision_round"
        ),
        "review_round": _bounded_round(state["review_round"], "review_round"),
        "pending_interrupt": _normalize_interrupt_descriptor(
            state.get("pending_interrupt")
        ),
        "warnings": merge_unique_strings((), state["warnings"]),
        "errors": merge_unique_strings((), state["errors"]),
    }


def serialize_jmr_graph_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return detached JSON/msgpack-safe primitives."""

    normalized = normalize_jmr_graph_state(state)
    serializable = dict(normalized)
    serializable["messages"] = messages_as_dicts(normalized["messages"])
    plain = _to_json_value(serializable, path="graph_state")
    return json.loads(json.dumps(plain, ensure_ascii=False))


def deserialize_jmr_graph_state(payload: Mapping[str, Any]) -> JMRGraphState:
    """Restore schema-v3 state; older migration checkpoints are unsupported."""

    return normalize_jmr_graph_state(payload)


def normalize_messages(messages: Any) -> list[AnyMessage]:
    """Normalize and bound the graph's short conversational context."""

    dictionaries = _message_dicts(messages)
    return add_bounded_messages([], dictionaries)


def messages_as_dicts(messages: Any) -> list[dict[str, Any]]:
    """Convert supported message objects into detached dictionaries."""

    return _message_dicts(messages)


def _message_dicts(messages: Any) -> list[dict[str, Any]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("messages must be a sequence")
    if len(messages) > MAX_GRAPH_MESSAGES:
        raise ValueError(
            f"messages cannot contain more than {MAX_GRAPH_MESSAGES} items"
        )
    normalized: list[dict[str, Any]] = []
    total_bytes = 0
    for index, raw in enumerate(messages):
        message = (
            convert_to_openai_messages(raw, include_id=True)
            if isinstance(raw, BaseMessage)
            else raw
        )
        value = _to_json_value(message, path=f"messages[{index}]")
        if not isinstance(value, dict):
            raise TypeError(f"messages[{index}] must be an object")
        role = value.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{index}].role must be a non-empty string")
        size = len(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        )
        if size > MAX_GRAPH_MESSAGE_BYTES:
            raise ValueError(f"messages[{index}] exceeds the size limit")
        total_bytes += size
        if total_bytes > MAX_GRAPH_MESSAGES_BYTES:
            raise ValueError("messages exceed the total size limit")
        normalized.append(value)
    return normalized


def _to_json_value(value: Any, *, path: str) -> Any:
    if isinstance(value, Enum):
        return _to_json_value(value.value, path=path)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, date):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_json_value(
                getattr(value, item.name), path=f"{path}.{item.name}"
            )
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            result[key] = _to_json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _to_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_json_value(model_dump(), path=path)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _to_json_value(to_dict(), path=path)
    raise TypeError(f"{path} contains non-serializable value {type(value).__name__}")


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, field_name: str) -> str:
    normalized = _non_empty_string(value, field_name)
    if len(normalized) > 255:
        raise ValueError(f"{field_name} cannot exceed 255 characters")
    return normalized


def _optional_identifier(value: Any, field_name: str) -> str | None:
    return None if value is None else _identifier(value, field_name)


def _identifier_list(value: Any, field_name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of identifiers")
    normalized = [
        _identifier(item, f"{field_name}[{index}]") for index, item in enumerate(value)
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} cannot contain duplicate identifiers")
    return normalized


def _bounded_round(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0 or value > 2:
        raise ValueError(f"{field_name} must be between 0 and 2")
    return value


def _schema_version(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("graph state schema_version must be an integer")
    if value != GRAPH_STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported graph state schema_version: {value!r}")


def _normalize_retrieval_ref(value: Any, field_name: str) -> RetrievalRef:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    expected = {"worker_kind", "retrieval_run_id", "mcp_status"}
    if set(value) != expected:
        raise ValueError(f"{field_name} must contain exactly {sorted(expected)}")
    try:
        status = MCPStatus(value["mcp_status"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}.mcp_status has an unknown value") from exc
    return {
        "worker_kind": _non_empty_string(
            value["worker_kind"], f"{field_name}.worker_kind"
        ),
        "retrieval_run_id": _identifier(
            value["retrieval_run_id"], f"{field_name}.retrieval_run_id"
        ),
        "mcp_status": status.value,
    }


def _normalize_interrupt_descriptor(value: Any) -> InterruptDescriptor | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("pending_interrupt must be an object or null")
    expected = {"kind", "payload_ref", "resume_schema_version", "issued_at"}
    if set(value) != expected:
        raise ValueError(f"pending_interrupt must contain exactly {sorted(expected)}")
    try:
        kind = InterruptKind(value["kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("pending_interrupt.kind has an unknown value") from exc
    version = value["resume_schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("resume_schema_version must be a positive integer")
    return {
        "kind": kind.value,
        "payload_ref": _identifier(value["payload_ref"], "payload_ref"),
        "resume_schema_version": version,
        "issued_at": _non_empty_string(value["issued_at"], "issued_at"),
    }


def _string_list(value: Any, field_name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    return [
        _non_empty_string(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    ]


__all__ = [
    "GRAPH_STATE_SCHEMA_VERSION",
    "InterruptDescriptor",
    "JMRGraphState",
    "RetrievalRef",
    "add_bounded_messages",
    "create_jmr_graph_state",
    "deserialize_jmr_graph_state",
    "keep_immutable_identifier",
    "merge_retrieval_refs",
    "merge_unique_strings",
    "messages_as_dicts",
    "normalize_jmr_graph_state",
    "normalize_messages",
    "serialize_jmr_graph_state",
]
