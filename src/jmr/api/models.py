"""Bounded HTTP request and response contracts."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

MAX_API_JSON_BYTES = 64 * 1024
MAX_USER_MESSAGE_CHARS = 12_000
IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$"


def _require_true(value: Any) -> Literal[True]:
    if value is not True:
        raise ValueError("confirmed_by_user must be true")
    return True


ConfirmedByUser = Annotated[Literal[True], BeforeValidator(_require_true)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateCaseRequest(StrictModel):
    message: str = Field(min_length=1, max_length=MAX_USER_MESSAGE_CHARS)
    case_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        pattern=IDENTIFIER_PATTERN,
    )

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message cannot be blank")
        return normalized


class SendMessageRequest(StrictModel):
    message: str = Field(min_length=1, max_length=MAX_USER_MESSAGE_CHARS)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message cannot be blank")
        return normalized


class ResumeCaseRequest(StrictModel):
    payload: dict[str, Any]
    interrupt_token: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )

    @field_validator("payload")
    @classmethod
    def bound_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_bounded_json(value, "resume payload")
        return value


class CaseView(StrictModel):
    user_id: str
    case_id: str
    workflow_stage: str
    run_status: str
    pending_interrupt: dict[str, Any] | None = None
    interrupt_token: str | None = None
    retry_available: bool = False
    failure: dict[str, str] | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    result_available: bool = False
    selected_paper_evidence_ids: list[str] = Field(default_factory=list)
    selected_papers: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)


class CaseListView(StrictModel):
    items: list[dict[str, Any]] = Field(default_factory=list)


class ConversationView(StrictModel):
    case_id: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)


class CaseProgressView(StrictModel):
    case_id: str
    workflow_stage: str
    run_status: str
    events: list[dict[str, Any]] = Field(default_factory=list)


class WorkspaceView(StrictModel):
    case_id: str
    user_id: str
    target: dict[str, Any] | None = None
    plans: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_runs: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    directions: list[dict[str, Any]] = Field(default_factory=list)
    selections: list[dict[str, Any]] = Field(default_factory=list)
    drafts: list[dict[str, Any]] = Field(default_factory=list)
    reviews: list[dict[str, Any]] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    selected_memory_ids: list[str] = Field(default_factory=list)


class FinalResultView(StrictModel):
    user_id: str
    case_id: str
    iteration_id: str
    content: str
    citation_ids: list[str] = Field(default_factory=list)


class MemoryListView(StrictModel):
    status: str
    items: list[dict[str, Any]] = Field(default_factory=list)


class MemoryView(StrictModel):
    status: str
    memory: dict[str, Any]


class CreateMemoryRequest(StrictModel):
    kind: str = Field(min_length=1, max_length=100)
    content: dict[str, Any]
    tags: list[str] = Field(default_factory=list, max_length=20)
    confirmed_by_user: ConfirmedByUser
    idempotency_key: str = Field(min_length=1, max_length=255)

    @field_validator("content")
    @classmethod
    def bound_content(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_bounded_json(value, "memory content")
        return value

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        return _tags(value)


class UpdateMemoryRequest(StrictModel):
    content: dict[str, Any]
    tags: list[str] | None = Field(default=None, max_length=20)
    expected_version: int = Field(strict=True, ge=1)
    confirmed_by_user: ConfirmedByUser
    idempotency_key: str = Field(min_length=1, max_length=255)

    @field_validator("content")
    @classmethod
    def bound_content(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_bounded_json(value, "memory content")
        return value

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str] | None) -> list[str]:
        if value is None:
            raise ValueError("tags must be an array when provided")
        return _tags(value)


class DeleteMemoryRequest(StrictModel):
    expected_version: int = Field(strict=True, ge=1)
    confirmed_by_user: ConfirmedByUser
    idempotency_key: str = Field(min_length=1, max_length=255)


class ConfirmCaseDeletionRequest(StrictModel):
    confirmed_by_user: ConfirmedByUser
    plan_token: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")


class CaseDeletionView(StrictModel):
    case_id: str
    user_id: str
    status: str
    dry_run: bool
    object_count: int
    object_uris: list[str] = Field(default_factory=list)
    checkpoint_deleted: bool
    business_deleted: bool
    audit_retained: bool
    plan_token: str | None = None


class ErrorDetail(StrictModel):
    code: str
    message: str
    failure_id: str | None = None
    fields: list[str] | None = None


class ErrorView(StrictModel):
    error: ErrorDetail


def _require_bounded_json(value: Any, label: str) -> None:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc
    if len(encoded) > MAX_API_JSON_BYTES:
        raise ValueError(f"{label} exceeds {MAX_API_JSON_BYTES} bytes")


def _tags(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("memory tags must be non-empty strings")
        tag = value.strip()
        if len(tag) > 100:
            raise ValueError("memory tags cannot exceed 100 characters")
        if tag not in normalized:
            normalized.append(tag)
    return normalized
