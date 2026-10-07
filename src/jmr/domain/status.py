"""Canonical status enumerations used across graph, MCP, and persistence."""

from __future__ import annotations

from enum import StrEnum


class WorkflowStage(StrEnum):
    """The eight business stages controlled by the compiled StateGraph."""

    VALIDATING_APPLICATION_INPUTS = "VALIDATING_APPLICATION_INPUTS"
    RETRIEVING_RESEARCH_EVIDENCE = "RETRIEVING_RESEARCH_EVIDENCE"
    MERGING_AND_VERIFYING_EVIDENCE = "MERGING_AND_VERIFYING_EVIDENCE"
    GENERATING_RESEARCH_DIRECTIONS = "GENERATING_RESEARCH_DIRECTIONS"
    AWAITING_DIRECTION_SELECTION = "AWAITING_DIRECTION_SELECTION"
    DRAFTING_OUTREACH_RESEARCH_PLAN = "DRAFTING_OUTREACH_RESEARCH_PLAN"
    REVIEWING_OUTREACH_RESEARCH_PLAN = "REVIEWING_OUTREACH_RESEARCH_PLAN"
    COMPLETED = "COMPLETED"


class RunStatus(StrEnum):
    """Lifecycle status for one case execution."""

    READY = "READY"
    RUNNING = "RUNNING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    RETRYING = "RETRYING"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class NodeExecutionStatus(StrEnum):
    """Result status for one node invocation."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    INTERRUPTED = "INTERRUPTED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class MCPStatus(StrEnum):
    """Normalized status returned by every MCP capability."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    NO_RESULT = "NO_RESULT"
    NEEDS_USER_CONFIRMATION = "NEEDS_USER_CONFIRMATION"
    BLOCKED = "BLOCKED"
    CONFLICT = "CONFLICT"
    FAILED = "FAILED"


class EvidenceVerificationStatus(StrEnum):
    """Whether a persisted evidence record is safe to cite."""

    UNVERIFIED = "UNVERIFIED"
    PARTIAL = "PARTIAL"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"


class ReviewStatus(StrEnum):
    """Evaluator result for a direction batch or outreach iteration."""

    PASS = "PASS"
    REVISE = "REVISE"
    BLOCKED = "BLOCKED"


class MemoryStatus(StrEnum):
    """Lifecycle status for a versioned long-term memory."""

    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    DELETED = "DELETED"


class InterruptKind(StrEnum):
    """All user decisions that are allowed to pause the graph."""

    APPLICATION_INPUT_REQUIRED = "APPLICATION_INPUT_REQUIRED"
    TARGET_IDENTITY_CONFIRMATION_REQUIRED = "TARGET_IDENTITY_CONFIRMATION_REQUIRED"
    EVIDENCE_SOURCE_REQUIRED = "EVIDENCE_SOURCE_REQUIRED"
    APPLICANT_MEMORY_CONFIRMATION_REQUIRED = "APPLICANT_MEMORY_CONFIRMATION_REQUIRED"
    DIRECTION_SELECTION_REQUIRED = "DIRECTION_SELECTION_REQUIRED"
    REVISION_INPUT_REQUIRED = "REVISION_INPUT_REQUIRED"


class MCPErrorCode(StrEnum):
    """Stable machine-readable error codes for the common MCP envelope."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    SOURCE_BLOCKED = "SOURCE_BLOCKED"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
