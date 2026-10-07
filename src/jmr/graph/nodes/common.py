"""Shared host-side helpers for the formal P4-P6 graph nodes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from langgraph.runtime import Runtime

from jmr.runtime import JMRRuntimeContext
from jmr.runtime.observability import emit_runtime_event


class ModelOutputValidationError(ValueError):
    """A model answered, but its structured output violated the node contract."""

    def __init__(self, node_name: str) -> None:
        self.node_name = node_name
        super().__init__(f"{node_name} failed structured output validation")


def runtime_context(
    runtime: Runtime[JMRRuntimeContext] | JMRRuntimeContext,
) -> JMRRuntimeContext:
    if isinstance(runtime, JMRRuntimeContext):
        return runtime
    if runtime.context is None:
        raise RuntimeError("JMRRuntimeContext is required")
    return runtime.context


def repository(runtime: Runtime[JMRRuntimeContext] | JMRRuntimeContext) -> Any:
    context = runtime_context(runtime)
    if context.case_repository is None:
        raise RuntimeError("formal workflow requires case_repository")
    return context.case_repository


def required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def invoke_model(
    context: JMRRuntimeContext,
    node_name: str,
    *,
    messages: Sequence[Mapping[str, Any]],
    system_prompt: str,
    output_schema: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Invoke an optional structured model with the node's bounded retry cap."""

    if context.model_registry is None:
        return None
    model = context.model_registry.for_node(node_name)
    last_error: Exception | None = None
    for _ in range(2):
        try:
            result = model.invoke_structured(
                messages=compact_messages_for_model(messages),
                system_prompt=system_prompt,
                output_schema=output_schema,
            )
            if not isinstance(result, Mapping):
                raise TypeError("structured model output must be an object")
            return result
        except (TypeError, ValueError) as exc:
            last_error = exc
    raise ModelOutputValidationError(node_name) from last_error


def compact_messages_for_model(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_messages: int = 16,
    max_bytes: int = 48 * 1024,
) -> list[Mapping[str, Any]]:
    """Keep the newest bounded context and mark omitted history explicitly."""

    if max_messages < 2 or max_bytes < 1024:
        raise ValueError("model context limits are too small")
    normalized = [dict(message) for message in messages]
    selected: list[Mapping[str, Any]] = []
    used_bytes = 0
    for message in reversed(normalized):
        encoded = json.dumps(
            message,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode()
        if selected and (
            len(selected) >= max_messages - 1
            or used_bytes + len(encoded) > max_bytes - 256
        ):
            break
        if len(encoded) > max_bytes - 256:
            raise ValueError("one model message exceeds the bounded context limit")
        selected.append(message)
        used_bytes += len(encoded)
    selected.reverse()
    omitted = len(normalized) - len(selected)
    if omitted:
        selected.insert(
            0,
            {
                "role": "system",
                "content": (
                    f"[Context boundary: {omitted} earlier message(s) omitted; "
                    "durable business facts must be reloaded by ID.]"
                ),
            },
        )
    return selected


def compact_evidence_for_model(
    bundle: Mapping[str, Any],
    *,
    priority_ids: Sequence[str] = (),
    max_records: int = 18,
) -> dict[str, Any]:
    """Project durable evidence into a bounded, citable model context.

    The complete evidence bundle stays in persistence. Model prompts only need
    the identifying facts and a short source-grounded summary for each record.
    """

    allowed = set(bundle.get("evidence_ids", []))
    summary = bundle.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    raw_bundle = summary.get("bundle", {})
    if not isinstance(raw_bundle, Mapping):
        raw_bundle = {}
    raw_records = raw_bundle.get("records", summary.get("records", []))
    if not isinstance(raw_records, list):
        raw_records = []
    priority = {value: index for index, value in enumerate(priority_ids)}
    records: list[Mapping[str, Any]] = []
    for item in raw_records:
        if not isinstance(item, Mapping):
            continue
        evidence = item.get("evidence", item)
        if isinstance(evidence, Mapping) and evidence.get("evidence_id") in allowed:
            records.append(evidence)
    records.sort(
        key=lambda item: (
            0 if item["evidence_id"] in priority else 1,
            priority.get(item["evidence_id"], 0),
            0 if item.get("verification_status") == "VERIFIED" else 1,
        )
    )
    projected = []
    for item in records[:max_records]:
        projected.append(
            {
                "evidence_id": item["evidence_id"],
                "title": _brief_text(item.get("title"), 220),
                "source_name": _brief_text(item.get("source_name"), 80),
                "source_url": _brief_text(item.get("source_url"), 350),
                "publication_date": item.get("publication_date"),
                "venue": _brief_text(item.get("venue"), 120),
                "verification_status": item.get("verification_status"),
                "abstract_or_summary": _brief_text(
                    item.get("abstract_or_summary"), 240
                ),
                "source_citation": _brief_text(item.get("source_citation"), 160),
            }
        )
    selected_ids = [item["evidence_id"] for item in projected]
    if not selected_ids:
        selected_ids = list(bundle.get("evidence_ids", []))[:max_records]
    return {
        "records": projected,
        "allowed_evidence_ids": selected_ids,
        "total_evidence_count": len(allowed),
        "verified_count": summary.get("verified_count", 0),
        "partial_count": summary.get("partial_count", 0),
    }


def _brief_text(value: Any, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip()[:max_chars] or None


def emit_event(
    context: JMRRuntimeContext,
    state: Mapping[str, Any],
    *,
    node: str,
    status: str,
    tool: str | None = None,
    duration_ms: float = 0.0,
    retry_count: int = 0,
    details: Mapping[str, Any] | None = None,
) -> None:
    emit_runtime_event(
        context=context,
        state=state,
        node=node,
        status=status,
        tool=tool,
        duration_ms=duration_ms,
        retry_count=retry_count,
        details={"workflow_stage": state.get("workflow_stage"), **(details or {})},
    )


def write_json_artifact(
    context: JMRRuntimeContext,
    state: Mapping[str, Any],
    *,
    artifact_kind: str,
    value: Any,
) -> str:
    """Persist phase-local business bodies outside the graph checkpoint."""

    if context.object_store is None:
        raise RuntimeError("complete workflow requires object_store")
    object_id = context.id_generator.new_id(artifact_kind)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode()
    user_id = required_text(state.get("user_id"), "user_id")
    case_id = required_text(state.get("case_id"), "case_id")
    user_segment = quote(user_id, safe="")
    case_segment = quote(case_id, safe="")
    return context.object_store.put(
        object_key=(
            f"users/{user_segment}/cases/{case_segment}/workflow/{object_id}.json"
        ),
        content=payload,
        content_type="application/json",
    )


def read_json_artifact(context: JMRRuntimeContext, object_uri: Any) -> Any:
    if context.object_store is None:
        raise RuntimeError("complete workflow requires object_store")
    uri = required_text(object_uri, "object_uri")
    try:
        return json.loads(context.object_store.get(object_uri=uri).decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("workflow artifact is not valid JSON") from exc


def stage_transition(
    context: JMRRuntimeContext,
    state: Mapping[str, Any],
    *,
    from_stage: str,
    to_stage: str,
    node_name: str,
    run_status: str = "RUNNING",
) -> Mapping[str, Any]:
    repo = repository(context)
    return repo.record_stage_transition(
        user_id=required_text(state.get("user_id"), "user_id"),
        case_id=required_text(state.get("case_id"), "case_id"),
        transition={
            "from_stage": from_stage,
            "to_stage": to_stage,
            "run_status": run_status,
            "node_name": node_name,
        },
        idempotency_key=f"stage:{state.get('case_id')}:{to_stage}:{node_name}",
    )
