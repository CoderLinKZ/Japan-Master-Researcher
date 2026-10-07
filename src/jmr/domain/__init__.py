"""Shared domain contracts for the JMR workflow."""

from .evidence import (
    DiscoveredSource,
    DiscoveredSourceType,
    EvidenceType,
    ResearchEvidence,
    RetrievalResult,
    RetrievalStatus,
    SourceDiscoveryMethod,
    VerifiedEvidenceBundle,
    VerifiedEvidenceRecord,
)
from .status import (
    EvidenceVerificationStatus,
    InterruptKind,
    MCPErrorCode,
    MCPStatus,
    MemoryStatus,
    NodeExecutionStatus,
    ReviewStatus,
    RunStatus,
    WorkflowStage,
)

__all__ = [
    "DiscoveredSource",
    "DiscoveredSourceType",
    "EvidenceType",
    "EvidenceVerificationStatus",
    "InterruptKind",
    "MCPErrorCode",
    "MCPStatus",
    "MemoryStatus",
    "NodeExecutionStatus",
    "ResearchEvidence",
    "RetrievalResult",
    "RetrievalStatus",
    "ReviewStatus",
    "RunStatus",
    "SourceDiscoveryMethod",
    "VerifiedEvidenceBundle",
    "VerifiedEvidenceRecord",
    "WorkflowStage",
]
