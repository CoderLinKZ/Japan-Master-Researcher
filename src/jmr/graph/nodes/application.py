"""P3 application-input nodes for the target JMR graph.

This slice is intentionally deterministic at its boundaries.  A structured
model may extract fields, but defaults, validation, the interrupt payload and
all writes are owned by the host nodes below.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from jmr.domain import RunStatus, WorkflowStage
from jmr.graph.identity import validate_invocation_identity
from jmr.graph.state import (
    GRAPH_STATE_SCHEMA_VERSION,
    InterruptDescriptor,
    messages_as_dicts,
)
from jmr.persistence.errors import PersistenceError
from jmr.runtime.context import JMRRuntimeContext

from .common import compact_messages_for_model

APPLICATION_INPUT_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "extracted": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "university": {"type": ["string", "null"]},
                "graduate_school": {"type": ["string", "null"]},
                "laboratory": {"type": ["string", "null"]},
                "professor_name": {"type": ["string", "null"]},
                "professor_name_variants": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "official_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "date_from": {"type": ["string", "null"]},
                "date_to": {"type": ["string", "null"]},
            },
            "required": [
                "university",
                "graduate_school",
                "laboratory",
                "professor_name",
                "professor_name_variants",
                "official_urls",
                "date_from",
                "date_to",
            ],
        },
        "explicit_corrections": {
            "type": "array",
            "items": {"type": "string"},
        },
        "ambiguous_fragments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["text", "reason"],
            },
        },
    },
    "required": ["extracted", "explicit_corrections", "ambiguous_fragments"],
}

_FIELD_ALIASES = {
    "university": "university_name",
    "university_name": "university_name",
    "大学": "university_name",
    "graduate_school": "graduate_school_name",
    "graduate_school_name": "graduate_school_name",
    "研究科": "graduate_school_name",
    "学院": "graduate_school_name",
    "laboratory": "laboratory_name",
    "laboratory_name": "laboratory_name",
    "研究室": "laboratory_name",
    "professor": "professor_name",
    "professor_name": "professor_name",
    "教授": "professor_name",
    "date_from": "date_from",
    "date_to": "date_to",
    "start_date": "date_from",
    "end_date": "date_to",
    "检索开始日期": "date_from",
    "检索结束日期": "date_to",
}
_DATE_PATTERN = re.compile(r"(?<!\d)(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?")
_URL_PATTERN = re.compile(r"https?://[^\s,，)）]+")


def initialize_case(
    state: Mapping[str, Any],
    config: RunnableConfig,
    runtime: Runtime[JMRRuntimeContext],
) -> dict[str, Any]:
    """Validate identity and idempotently create the business Case."""

    identity = validate_invocation_identity(state, config)
    if state.get("schema_version") != GRAPH_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported or missing graph state schema_version")
    context = _runtime_context(runtime)
    if context.security_policy is not None:
        if not isinstance(context.principal_user_id, str):
            raise PermissionError("an authenticated principal is required")
        context.security_policy.authorize_case(
            principal_user_id=context.principal_user_id,
            user_id=identity.user_id,
            case_id=identity.case_id,
        )
    repository = _require_repository(context)
    if state.get("workflow_stage") != WorkflowStage.VALIDATING_APPLICATION_INPUTS.value:
        return {"application_stage_done": True}
    if state.get("case_context_id"):
        return {"run_status": RunStatus.RUNNING.value}
    result = repository.create_case(
        user_id=identity.user_id,
        case_id=identity.case_id,
        schema_version=GRAPH_STATE_SCHEMA_VERSION,
        idempotency_key=f"initialize:{identity.case_id}",
    )
    return {
        "case_context_id": result["case_context_id"],
        "run_status": RunStatus.RUNNING.value,
        "workflow_stage": WorkflowStage.VALIDATING_APPLICATION_INPUTS.value,
        "application_case_version": result.get("version", 1),
    }


def load_case_context(
    state: Mapping[str, Any],
    runtime: Runtime[JMRRuntimeContext],
) -> dict[str, Any]:
    """Load only the small target projection needed by input validation."""

    context = _runtime_context(runtime)
    repository = _require_repository(context)
    result = repository.get_case(
        user_id=_required(state.get("user_id"), "user_id"),
        case_id=_required(state.get("case_id"), "case_id"),
    )
    if result is None:
        raise ValueError("case_context_id points to a missing case")
    target = result.get("target") or {}
    if not isinstance(target, Mapping):
        raise ValueError("persisted case target must be an object")
    return {
        "case_context_id": state.get("case_context_id")
        or result.get("case_context_id"),
        "application_fields": {},
        "persisted_application_fields": _normalize_fields(target),
        "application_case_version": result.get("version", 1),
        "run_status": result.get("run_status", RunStatus.RUNNING.value),
    }


def extract_application_inputs(
    state: Mapping[str, Any],
    runtime: Runtime[JMRRuntimeContext],
) -> dict[str, Any]:
    """Extract explicit target fields using a structured model or safe fallback."""

    context = _runtime_context(runtime)
    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in messages_as_dicts(state.get("messages", ()))
    ]
    model_registry = context.model_registry
    if model_registry is not None:
        model = model_registry.for_node("extract_application_inputs")
        raw: Mapping[str, Any] = {}
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                candidate = model.invoke_structured(
                    messages=compact_messages_for_model(messages),
                    system_prompt=(
                        "从用户消息中只提取明确给出的日本大学、研究科、研究室、"
                        "教授和检索日期。未明确给出的字段必须为 null。"
                        "专业、学科或研究方向（如‘情報理工’）不是研究室；"
                        "只有用户明确写出研究室或 Lab 名称时才填写 laboratory。"
                    ),
                    output_schema=APPLICATION_INPUT_EXTRACTION_SCHEMA,
                )
                if not isinstance(candidate, Mapping):
                    raise TypeError("structured extraction must return an object")
                raw = candidate
                break
            except (TypeError, ValueError) as exc:
                last_error = exc
        else:
            raise ValueError(
                "structured extraction failed schema validation"
            ) from last_error
        extracted = raw.get("extracted", raw) if isinstance(raw, Mapping) else {}
    else:
        extracted = _extract_without_model(messages)
    fields = _normalize_fields(extracted)
    return {
        "application_fields": fields,
        "application_extraction": {
            "explicit_corrections": list(
                raw.get("explicit_corrections", [])
                if model_registry is not None and isinstance(raw, Mapping)
                else []
            ),
            "ambiguous_fragments": list(
                raw.get("ambiguous_fragments", [])
                if model_registry is not None and isinstance(raw, Mapping)
                else []
            ),
        },
    }


def validate_application_inputs(
    state: Mapping[str, Any],
    runtime: Runtime[JMRRuntimeContext] | JMRRuntimeContext | None = None,
) -> dict[str, Any]:
    """Apply deterministic required-field, date and default validation."""

    fields = _normalize_fields(state.get("application_fields", {}))
    existing = _normalize_fields(state.get("persisted_application_fields", {}))
    merged = dict(existing)
    merged.update(
        {key: value for key, value in fields.items() if value not in (None, "", [])}
    )
    warnings: list[str] = []
    runtime_context = _runtime_context(runtime) if runtime is not None else None
    if not merged.get("date_from") and not merged.get("date_to"):
        reference_date = date.today()
        if runtime_context is not None:
            reference_date = runtime_context.clock.now().date()
        end = reference_date
        start = end - timedelta(days=365)
        merged["date_from"] = start.isoformat()
        merged["date_to"] = end.isoformat()
        merged["date_source"] = "default"
        warnings.append("检索时间范围采用系统默认的过去十二个月")
    missing: list[str] = []
    invalid: list[str] = []
    required = (
        ("university_name", "大学名称"),
        ("graduate_school_name", "研究科或学院名称"),
        ("professor_name", "教授姓名"),
    )
    for field_name, label in required:
        if not _is_non_empty(merged.get(field_name)):
            missing.append(field_name)
            warnings.append(f"缺少{label}")
    for field_name in ("date_from", "date_to"):
        value = merged.get(field_name)
        if value is None:
            missing.append(field_name)
            continue
        try:
            date.fromisoformat(str(value))
        except ValueError:
            invalid.append(field_name)
    if not missing and not invalid:
        if date.fromisoformat(merged["date_from"]) > date.fromisoformat(
            merged["date_to"]
        ):
            invalid.append("date_range")
            warnings.append("检索开始日期不能晚于结束日期")
    result = {
        "missing_fields": list(dict.fromkeys(missing)),
        "invalid_fields": list(dict.fromkeys(invalid)),
        "current_values": merged,
        "ready": not missing and not invalid,
    }
    descriptor: InterruptDescriptor | None = None
    if not result["ready"]:
        descriptor = {
            "kind": "APPLICATION_INPUT_REQUIRED",
            "payload_ref": f"application-inputs:{state.get('case_id')}",
            "resume_schema_version": 1,
            "issued_at": _now(runtime_context),
        }
    update: dict[str, Any] = {
        "application_validation": result,
        "warnings": warnings,
        "pending_interrupt": descriptor,
    }
    if result["ready"]:
        update["run_status"] = RunStatus.RUNNING.value
    else:
        update["run_status"] = RunStatus.WAITING_FOR_USER.value
    return update


def await_application_inputs(state: Mapping[str, Any]) -> dict[str, Any]:
    """Pause once with a compact payload; the resume value is never persisted."""

    validation = state.get("application_validation") or {}
    payload = {
        "kind": "APPLICATION_INPUT_REQUIRED",
        "user_id": state.get("user_id"),
        "case_id": state.get("case_id"),
        "workflow_stage": WorkflowStage.VALIDATING_APPLICATION_INPUTS.value,
        "prompt": "请补充或更正申请目标信息后继续。",
        "required_fields": list(validation.get("missing_fields", [])),
        "invalid_fields": list(validation.get("invalid_fields", [])),
        "current_values": _safe_public_fields(validation.get("current_values", {})),
        "resume_schema_version": 1,
    }
    resume_payload = interrupt(payload)
    return {
        "application_resume_payload": resume_payload,
        "pending_interrupt": None,
        "run_status": RunStatus.RUNNING.value,
    }


def apply_application_inputs(
    state: Mapping[str, Any],
    runtime: Runtime[JMRRuntimeContext],
) -> dict[str, Any]:
    """Validate a resume payload, persist the target, and hand off to retrieval."""

    validation = state.get("application_validation") or {}
    resume = state.get("application_resume_payload")
    if validation.get("ready"):
        candidate = _normalize_fields(validation.get("current_values", {}))
        idempotency_key = f"application:{state.get('case_id')}:ready"
    else:
        parsed = _parse_resume_payload(resume)
        if parsed["error"]:
            return _invalid_resume(state, parsed["error"])
        candidate = dict(validation.get("current_values", {}))
        candidate.update(parsed["fields"])
        if "date_from" in parsed["fields"] or "date_to" in parsed["fields"]:
            candidate["date_source"] = "user"
        next_validation = validate_application_inputs(
            {
                "case_id": state.get("case_id"),
                "application_fields": candidate,
                "persisted_application_fields": {},
            },
        )
        if not next_validation["application_validation"]["ready"]:
            return {
                **next_validation,
                "application_resume_payload": None,
                "run_status": RunStatus.WAITING_FOR_USER.value,
                "errors": ["申请目标信息仍不完整或包含非法字段"],
            }
        candidate = next_validation["application_validation"]["current_values"]
        idempotency_key = parsed["idempotency_key"]

    context = _runtime_context(runtime)
    repository = _require_repository(context)
    user_id = _required(state.get("user_id"), "user_id")
    case_id = _required(state.get("case_id"), "case_id")
    try:
        updated = repository.update_target(
            user_id=user_id,
            case_id=case_id,
            target=candidate,
            expected_version=int(state.get("application_case_version", 1)),
            idempotency_key=idempotency_key,
        )
        repository.record_stage_transition(
            user_id=user_id,
            case_id=case_id,
            transition={
                "from_stage": WorkflowStage.VALIDATING_APPLICATION_INPUTS.value,
                "to_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
                "run_status": RunStatus.READY.value,
                "node_name": "apply_application_inputs",
            },
            idempotency_key=f"stage:{case_id}:retrieval",
        )
    except PersistenceError as exc:
        return {
            "run_status": RunStatus.BLOCKED.value,
            "errors": [str(exc)],
        }
    return {
        "workflow_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
        "run_status": RunStatus.READY.value,
        "application_validation": {
            "missing_fields": [],
            "invalid_fields": [],
            "current_values": candidate,
            "ready": True,
        },
        "case_context_id": state.get("case_context_id"),
        "application_case_version": updated.get("version"),
        "application_resume_payload": None,
        "pending_interrupt": None,
        "errors": [],
    }


def _runtime_context(
    runtime: Runtime[JMRRuntimeContext] | JMRRuntimeContext,
) -> JMRRuntimeContext:
    if isinstance(runtime, JMRRuntimeContext):
        return runtime
    context = runtime.context
    if context is None:
        raise RuntimeError("JMRRuntimeContext is required")
    return context


def _require_repository(context: JMRRuntimeContext) -> Any:
    if context.case_repository is None:
        raise RuntimeError("P3 application nodes require case_repository")
    return context.case_repository


def _required(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _is_non_empty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _safe_public_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "university_name",
        "graduate_school_name",
        "laboratory_name",
        "professor_name",
        "professor_name_variants",
        "official_urls",
        "date_from",
        "date_to",
        "date_source",
    }
    return {key: value for key, value in fields.items() if key in allowed}


def _normalize_fields(raw: Mapping[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = _FIELD_ALIASES.get(str(key), str(key))
        if canonical not in {
            "university_name",
            "graduate_school_name",
            "laboratory_name",
            "professor_name",
            "professor_name_variants",
            "official_urls",
            "date_from",
            "date_to",
            "date_source",
        }:
            continue
        if isinstance(value, str):
            value = value.strip() or None
        elif canonical in {"professor_name_variants", "official_urls"}:
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                value = []
            value = [str(item).strip() for item in value if str(item).strip()]
        result[canonical] = value
    return result


def _extract_without_model(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    text_parts: list[str] = []
    structured: dict[str, Any] = {}
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, Mapping):
            structured.update(content)
        elif isinstance(content, Sequence):
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
    text = "\n".join(text_parts)
    for label, canonical in _FIELD_ALIASES.items():
        if len(label) < 2 or label in {
            "university",
            "graduate_school",
            "laboratory",
            "professor",
        }:
            continue
        match = re.search(rf"{re.escape(label)}\s*[:：=]\s*([^,，;；\n]+)", text, re.I)
        if match:
            structured[canonical] = match.group(1).strip()
    dates = [
        date(int(year), int(month), int(day)).isoformat()
        for year, month, day in _DATE_PATTERN.findall(text)
        if _valid_date_parts(year, month, day)
    ]
    if len(dates) >= 2:
        structured.setdefault("date_from", dates[0])
        structured.setdefault("date_to", dates[1])
    elif len(dates) == 1:
        structured.setdefault("date_from", dates[0])
    structured["official_urls"] = _URL_PATTERN.findall(text)
    return structured


def _valid_date_parts(year: str, month: str, day: str) -> bool:
    try:
        date(int(year), int(month), int(day))
    except ValueError:
        return False
    return True


def _parse_resume_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"fields": {}, "idempotency_key": "", "error": "resume 必须是对象"}
    allowed = set(_FIELD_ALIASES) | {
        "idempotency_key",
        "fields",
        "resume_schema_version",
    }
    unknown = set(payload) - allowed
    if unknown:
        return {"fields": {}, "idempotency_key": "", "error": "resume 包含未知字段"}
    if payload.get("resume_schema_version", 1) != 1:
        return {
            "fields": {},
            "idempotency_key": "",
            "error": "resume schema 版本不受支持",
        }
    key = payload.get("idempotency_key")
    if not isinstance(key, str) or not key.strip():
        return {"fields": {}, "idempotency_key": "", "error": "缺少 idempotency_key"}
    raw_fields = payload.get("fields")
    if raw_fields is None:
        raw_fields = {
            name: value
            for name, value in payload.items()
            if name not in {"idempotency_key", "resume_schema_version"}
        }
    if not isinstance(raw_fields, Mapping):
        return {
            "fields": {},
            "idempotency_key": key.strip(),
            "error": "fields 必须是对象",
        }
    return {
        "fields": _normalize_fields(raw_fields),
        "idempotency_key": key.strip(),
        "error": None,
    }


def _invalid_resume(state: Mapping[str, Any], error: str) -> dict[str, Any]:
    validation = dict(state.get("application_validation") or {})
    validation["ready"] = False
    return {
        "application_validation": validation,
        "application_resume_payload": None,
        "pending_interrupt": {
            "kind": "APPLICATION_INPUT_REQUIRED",
            "payload_ref": f"application-inputs:{state.get('case_id')}",
            "resume_schema_version": 1,
            "issued_at": _now(None),
        },
        "run_status": RunStatus.WAITING_FOR_USER.value,
        "errors": [error],
    }


def _now(context: JMRRuntimeContext | None) -> str:
    if context is not None:
        return context.clock.now().isoformat()
    return datetime.now(UTC).isoformat()
