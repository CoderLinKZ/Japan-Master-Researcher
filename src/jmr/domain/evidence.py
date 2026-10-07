# File: evidence.py
# Author: L1nzhk0
# Purpose: 正式 JMR 领域中的检索证据、核验记录与批次结构。

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from .status import EvidenceVerificationStatus, MCPStatus


# 返回用于记录检索时间的当前 UTC 时间
def _utc_now() -> datetime:
    return datetime.now(UTC)


class EvidenceType(StrEnum):
    """表示研究证据是论文条目还是 KAKEN 课题。"""

    PUBLICATION = "publication"
    KAKEN_PROJECT = "kaken_project"


# Providers and MCP servers use the same canonical status enum as the graph.
# The alias keeps domain terminology readable without creating a second status
# authority with incompatible lowercase values.
RetrievalStatus = MCPStatus


class DiscoveredSourceType(StrEnum):
    """表示一个候选官网来源在检索流程中的用途。"""

    LABORATORY_WEBSITE = "laboratory_website"
    PROFESSOR_PROFILE = "professor_profile"
    PUBLICATION_LIST = "publication_list"
    OTHER = "other"


class SourceDiscoveryMethod(StrEnum):
    """表示候选官网来源是由用户提供还是由系统发现。"""

    USER_PROVIDED = "user_provided"
    WEB_SEARCH = "web_search"
    SAME_DOMAIN_LINK = "same_domain_link"
    DATA_SOURCE = "data_source"


@dataclass(slots=True)
class DiscoveredSource:
    """保存官网发现阶段得到的结构化来源及其核验状态。"""

    url: str
    source_type: DiscoveredSourceType
    discovery_method: SourceDiscoveryMethod
    verification_status: EvidenceVerificationStatus
    title: str | None = None
    matched_signals: list[str] = field(default_factory=list)

    # 校验候选官网来源并复制可变字段
    def __post_init__(self) -> None:
        self.url = _validate_required_url(self.url, "url")
        if not isinstance(self.source_type, DiscoveredSourceType):
            raise TypeError("source_type must be a DiscoveredSourceType")
        if not isinstance(self.discovery_method, SourceDiscoveryMethod):
            raise TypeError("discovery_method must be a SourceDiscoveryMethod")
        if not isinstance(
            self.verification_status,
            EvidenceVerificationStatus,
        ):
            raise TypeError("verification_status must be an EvidenceVerificationStatus")
        self.title = _optional_non_empty_string(self.title, "title")
        self.matched_signals = _copy_string_list(
            self.matched_signals,
            "matched_signals",
        )

    # 从经过 JSON 解码的对象构造候选官网来源
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DiscoveredSource:
        _reject_unknown_fields(
            payload,
            {
                "url",
                "source_type",
                "discovery_method",
                "verification_status",
                "title",
                "matched_signals",
            },
            "DiscoveredSource",
        )
        try:
            source_type = DiscoveredSourceType(payload.get("source_type"))
        except (TypeError, ValueError) as exc:
            raise ValueError("source_type is invalid") from exc
        try:
            discovery_method = SourceDiscoveryMethod(payload.get("discovery_method"))
        except (TypeError, ValueError) as exc:
            raise ValueError("discovery_method is invalid") from exc
        try:
            verification_status = EvidenceVerificationStatus(
                payload.get("verification_status")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("verification_status is invalid") from exc
        return cls(
            url=payload.get("url"),
            source_type=source_type,
            discovery_method=discovery_method,
            verification_status=verification_status,
            title=payload.get("title"),
            matched_signals=payload.get("matched_signals", []),
        )

    # 将候选官网来源转换为可序列化的 JSON 对象
    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "source_type": self.source_type.value,
            "discovery_method": self.discovery_method.value,
            "verification_status": self.verification_status.value,
            "title": self.title,
            "matched_signals": list(self.matched_signals),
        }


@dataclass(slots=True)
class ResearchEvidence:
    """保存一篇论文或一个 KAKEN 课题对应的规范化证据记录。"""

    evidence_id: str
    evidence_type: EvidenceType
    title: str
    source_name: str
    retrieved_at: datetime = field(default_factory=_utc_now)
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    publication_date: str | None = None
    venue: str | None = None
    source_url: str | None = None
    identifiers: dict[str, str] = field(default_factory=dict)
    abstract_or_summary: str | None = None
    source_citation: str | None = None
    alternate_title: str | None = None
    project_start_year: int | None = None
    project_end_year: int | None = None
    project_status: str | None = None
    participant_roles: list[str] = field(default_factory=list)
    matched_professor: str | None = None
    matched_professor_position: int | None = None
    matched_professor_affiliations: list[str] = field(default_factory=list)
    verification_status: EvidenceVerificationStatus = (
        EvidenceVerificationStatus.UNVERIFIED
    )

    # 校验并复制单条证据的可变字段
    def __post_init__(self) -> None:
        self.evidence_id = _require_non_empty_string(
            self.evidence_id,
            "evidence_id",
        )
        if not isinstance(self.evidence_type, EvidenceType):
            raise TypeError("evidence_type must be an EvidenceType")
        self.title = _require_non_empty_string(self.title, "title")
        self.source_name = _require_non_empty_string(
            self.source_name,
            "source_name",
        )
        self.retrieved_at = _require_aware_datetime(
            self.retrieved_at,
            "retrieved_at",
        )
        self.authors = _copy_string_list(self.authors, "authors")
        self.affiliations = _copy_string_list(
            self.affiliations,
            "affiliations",
        )
        self.publication_date = _validate_partial_date(self.publication_date)
        self.venue = _optional_non_empty_string(self.venue, "venue")
        self.source_url = _validate_optional_url(self.source_url)
        self.identifiers = _copy_string_mapping(
            self.identifiers,
            "identifiers",
        )
        self.abstract_or_summary = _optional_non_empty_string(
            self.abstract_or_summary,
            "abstract_or_summary",
        )
        self.source_citation = _optional_non_empty_string(
            self.source_citation,
            "source_citation",
        )
        self.alternate_title = _optional_non_empty_string(
            self.alternate_title,
            "alternate_title",
        )
        self.project_start_year = _optional_year(
            self.project_start_year,
            "project_start_year",
        )
        self.project_end_year = _optional_year(
            self.project_end_year,
            "project_end_year",
        )
        if (
            self.project_start_year is not None
            and self.project_end_year is not None
            and self.project_start_year > self.project_end_year
        ):
            raise ValueError("project_start_year cannot be after project_end_year")
        self.project_status = _optional_non_empty_string(
            self.project_status,
            "project_status",
        )
        self.participant_roles = _copy_string_list(
            self.participant_roles,
            "participant_roles",
        )
        if self.participant_roles and len(self.participant_roles) != len(self.authors):
            raise ValueError("participant_roles must be empty or align with authors")
        self.matched_professor = _optional_non_empty_string(
            self.matched_professor,
            "matched_professor",
        )
        self.matched_professor_position = _optional_positive_integer(
            self.matched_professor_position,
            "matched_professor_position",
        )
        if (
            self.matched_professor_position is not None
            and self.matched_professor_position > len(self.authors)
        ):
            raise ValueError("matched_professor_position cannot exceed authors length")
        self.matched_professor_affiliations = _copy_string_list(
            self.matched_professor_affiliations,
            "matched_professor_affiliations",
        )
        if not isinstance(
            self.verification_status,
            EvidenceVerificationStatus,
        ):
            raise TypeError("verification_status must be an EvidenceVerificationStatus")

    # 从经过 JSON 解码的对象构造一条规范化证据
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResearchEvidence:
        _reject_unknown_fields(
            payload,
            {
                "evidence_id",
                "evidence_type",
                "title",
                "source_name",
                "retrieved_at",
                "authors",
                "affiliations",
                "publication_date",
                "venue",
                "source_url",
                "identifiers",
                "abstract_or_summary",
                "source_citation",
                "alternate_title",
                "project_start_year",
                "project_end_year",
                "project_status",
                "participant_roles",
                "matched_professor",
                "matched_professor_position",
                "matched_professor_affiliations",
                "verification_status",
            },
            "ResearchEvidence",
        )
        try:
            evidence_type = EvidenceType(payload.get("evidence_type"))
        except (TypeError, ValueError) as exc:
            raise ValueError("evidence_type is invalid") from exc
        try:
            verification_status = EvidenceVerificationStatus(
                payload.get(
                    "verification_status",
                    EvidenceVerificationStatus.UNVERIFIED.value,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("verification_status is invalid") from exc

        return cls(
            evidence_id=payload.get("evidence_id"),
            evidence_type=evidence_type,
            title=payload.get("title"),
            source_name=payload.get("source_name"),
            retrieved_at=_parse_iso_datetime(payload.get("retrieved_at")),
            authors=payload.get("authors", []),
            affiliations=payload.get("affiliations", []),
            publication_date=payload.get("publication_date"),
            venue=payload.get("venue"),
            source_url=payload.get("source_url"),
            identifiers=payload.get("identifiers", {}),
            abstract_or_summary=payload.get("abstract_or_summary"),
            source_citation=payload.get("source_citation"),
            alternate_title=payload.get("alternate_title"),
            project_start_year=payload.get("project_start_year"),
            project_end_year=payload.get("project_end_year"),
            project_status=payload.get("project_status"),
            participant_roles=payload.get("participant_roles", []),
            matched_professor=payload.get("matched_professor"),
            matched_professor_position=payload.get("matched_professor_position"),
            matched_professor_affiliations=payload.get(
                "matched_professor_affiliations",
                [],
            ),
            verification_status=verification_status,
        )

    # 将单条证据转换为可写入 Artifact 的 JSON 对象
    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "title": self.title,
            "source_name": self.source_name,
            "retrieved_at": self.retrieved_at.isoformat(),
            "authors": list(self.authors),
            "affiliations": list(self.affiliations),
            "publication_date": self.publication_date,
            "venue": self.venue,
            "source_url": self.source_url,
            "identifiers": dict(self.identifiers),
            "abstract_or_summary": self.abstract_or_summary,
            "source_citation": self.source_citation,
            "alternate_title": self.alternate_title,
            "project_start_year": self.project_start_year,
            "project_end_year": self.project_end_year,
            "project_status": self.project_status,
            "participant_roles": list(self.participant_roles),
            "matched_professor": self.matched_professor,
            "matched_professor_position": self.matched_professor_position,
            "matched_professor_affiliations": list(self.matched_professor_affiliations),
            "verification_status": self.verification_status.value,
        }


@dataclass(slots=True)
class RetrievalResult:
    """保存一次同类型证据检索的状态、查询条件和多条记录。"""

    request_id: str
    evidence_type: EvidenceType
    status: RetrievalStatus
    query: str
    source_name: str
    retrieved_at: datetime = field(default_factory=_utc_now)
    records: list[ResearchEvidence] = field(default_factory=list)
    discovered_sources: list[DiscoveredSource] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    artifact_ref: str | None = None

    # 校验检索批次及其包含的所有证据记录
    def __post_init__(self) -> None:
        self.request_id = _require_non_empty_string(
            self.request_id,
            "request_id",
        )
        if not isinstance(self.evidence_type, EvidenceType):
            raise TypeError("evidence_type must be an EvidenceType")
        if not isinstance(self.status, RetrievalStatus):
            raise TypeError("status must be a RetrievalStatus")
        self.query = _require_non_empty_string(self.query, "query")
        self.source_name = _require_non_empty_string(
            self.source_name,
            "source_name",
        )
        self.retrieved_at = _require_aware_datetime(
            self.retrieved_at,
            "retrieved_at",
        )
        if not isinstance(self.records, list):
            raise TypeError("records must be a list")
        self.records = list(self.records)
        evidence_ids: set[str] = set()
        for record in self.records:
            if not isinstance(record, ResearchEvidence):
                raise TypeError("records must contain ResearchEvidence items")
            if record.evidence_type != self.evidence_type:
                raise ValueError(
                    "all records must match the RetrievalResult evidence_type"
                )
            if record.evidence_id in evidence_ids:
                raise ValueError("records cannot contain duplicate evidence_id values")
            evidence_ids.add(record.evidence_id)
        if not isinstance(self.discovered_sources, list):
            raise TypeError("discovered_sources must be a list")
        self.discovered_sources = list(self.discovered_sources)
        source_urls: set[str] = set()
        for source in self.discovered_sources:
            if not isinstance(source, DiscoveredSource):
                raise TypeError(
                    "discovered_sources must contain DiscoveredSource items"
                )
            normalized_url = source.url.casefold()
            if normalized_url in source_urls:
                raise ValueError("discovered_sources cannot contain duplicate URLs")
            source_urls.add(normalized_url)
        self.warnings = _copy_string_list(self.warnings, "warnings")
        self.errors = _copy_string_list(self.errors, "errors")
        self.artifact_ref = _optional_non_empty_string(
            self.artifact_ref,
            "artifact_ref",
        )

        if self.status == RetrievalStatus.SUCCESS and not self.records:
            raise ValueError("a successful retrieval must contain at least one record")
        if self.status == RetrievalStatus.NO_RESULT and self.records:
            raise ValueError("a no_result retrieval cannot contain records")

    # 返回当前检索批次包含的证据数量
    @property
    def record_count(self) -> int:
        return len(self.records)

    # 返回当前检索批次包含的全部证据 ID
    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(record.evidence_id for record in self.records)

    # 从经过 JSON 解码的对象构造一次检索结果
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RetrievalResult:
        _reject_unknown_fields(
            payload,
            {
                "request_id",
                "evidence_type",
                "status",
                "query",
                "source_name",
                "retrieved_at",
                "records",
                "discovered_sources",
                "warnings",
                "errors",
                "artifact_ref",
            },
            "RetrievalResult",
        )
        try:
            evidence_type = EvidenceType(payload.get("evidence_type"))
        except (TypeError, ValueError) as exc:
            raise ValueError("evidence_type is invalid") from exc
        try:
            status = RetrievalStatus(payload.get("status"))
        except (TypeError, ValueError) as exc:
            raise ValueError("status is invalid") from exc

        raw_records = payload.get("records", [])
        if not isinstance(raw_records, list):
            raise TypeError("records must be a list")
        records: list[ResearchEvidence] = []
        for record in raw_records:
            if not isinstance(record, Mapping):
                raise TypeError("records must contain objects")
            records.append(ResearchEvidence.from_dict(record))
        raw_sources = payload.get("discovered_sources", [])
        if not isinstance(raw_sources, list):
            raise TypeError("discovered_sources must be a list")
        discovered_sources: list[DiscoveredSource] = []
        for source in raw_sources:
            if not isinstance(source, Mapping):
                raise TypeError("discovered_sources must contain objects")
            discovered_sources.append(DiscoveredSource.from_dict(source))
        return cls(
            request_id=payload.get("request_id"),
            evidence_type=evidence_type,
            status=status,
            query=payload.get("query"),
            source_name=payload.get("source_name"),
            retrieved_at=_parse_iso_datetime(payload.get("retrieved_at")),
            records=records,
            discovered_sources=discovered_sources,
            warnings=payload.get("warnings", []),
            errors=payload.get("errors", []),
            artifact_ref=payload.get("artifact_ref"),
        )

    # 将一次检索结果转换为可写入 Artifact 的 JSON 对象
    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "evidence_type": self.evidence_type.value,
            "status": self.status.value,
            "query": self.query,
            "source_name": self.source_name,
            "retrieved_at": self.retrieved_at.isoformat(),
            "records": [record.to_dict() for record in self.records],
            "discovered_sources": [
                source.to_dict() for source in self.discovered_sources
            ],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "artifact_ref": self.artifact_ref,
        }


@dataclass(slots=True)
class VerifiedEvidenceRecord:
    """保存一条整合后证据及其全部来源、合并依据和冲突说明。"""

    evidence: ResearchEvidence
    source_evidence_ids: list[str]
    source_names: list[str]
    merge_reasons: list[str] = field(default_factory=list)
    verification_reasons: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    # 校验整合后证据记录及其可追溯来源
    def __post_init__(self) -> None:
        if not isinstance(self.evidence, ResearchEvidence):
            raise TypeError("evidence must be a ResearchEvidence")
        if self.evidence.verification_status not in {
            EvidenceVerificationStatus.VERIFIED,
            EvidenceVerificationStatus.PARTIAL,
            EvidenceVerificationStatus.REJECTED,
        }:
            raise ValueError(
                "verified evidence status must be verified, partial, or rejected"
            )
        self.source_evidence_ids = _copy_unique_string_list(
            self.source_evidence_ids,
            "source_evidence_ids",
        )
        if not self.source_evidence_ids:
            raise ValueError("source_evidence_ids cannot be empty")
        self.source_names = _copy_unique_string_list(
            self.source_names,
            "source_names",
        )
        if not self.source_names:
            raise ValueError("source_names cannot be empty")
        self.merge_reasons = _copy_unique_string_list(
            self.merge_reasons,
            "merge_reasons",
        )
        self.verification_reasons = _copy_unique_string_list(
            self.verification_reasons,
            "verification_reasons",
        )
        self.conflicts = _copy_unique_string_list(
            self.conflicts,
            "conflicts",
        )

    # 从经过 JSON 解码的对象构造一条整合后证据
    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> VerifiedEvidenceRecord:
        _reject_unknown_fields(
            payload,
            {
                "evidence",
                "source_evidence_ids",
                "source_names",
                "merge_reasons",
                "verification_reasons",
                "conflicts",
            },
            "VerifiedEvidenceRecord",
        )
        raw_evidence = payload.get("evidence")
        if not isinstance(raw_evidence, Mapping):
            raise TypeError("evidence must be an object")
        return cls(
            evidence=ResearchEvidence.from_dict(raw_evidence),
            source_evidence_ids=payload.get("source_evidence_ids", []),
            source_names=payload.get("source_names", []),
            merge_reasons=payload.get("merge_reasons", []),
            verification_reasons=payload.get(
                "verification_reasons",
                [],
            ),
            conflicts=payload.get("conflicts", []),
        )

    # 将整合后证据转换为可写入 Artifact 的 JSON 对象
    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence.to_dict(),
            "source_evidence_ids": list(self.source_evidence_ids),
            "source_names": list(self.source_names),
            "merge_reasons": list(self.merge_reasons),
            "verification_reasons": list(self.verification_reasons),
            "conflicts": list(self.conflicts),
        }


@dataclass(slots=True)
class VerifiedEvidenceBundle:
    """保存一次证据整合与核验形成的独立可追溯 Artifact。"""

    bundle_id: str
    source_artifact_refs: list[str]
    records: list[VerifiedEvidenceRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    artifact_ref: str | None = None

    # 校验证据核验批次及其内部规范化记录
    def __post_init__(self) -> None:
        self.bundle_id = _require_non_empty_string(
            self.bundle_id,
            "bundle_id",
        )
        self.source_artifact_refs = _copy_unique_string_list(
            self.source_artifact_refs,
            "source_artifact_refs",
        )
        if not self.source_artifact_refs:
            raise ValueError("source_artifact_refs cannot be empty")
        if not isinstance(self.records, list):
            raise TypeError("records must be a list")
        self.records = list(self.records)
        evidence_ids: set[str] = set()
        for record in self.records:
            if not isinstance(record, VerifiedEvidenceRecord):
                raise TypeError("records must contain VerifiedEvidenceRecord items")
            evidence_id = record.evidence.evidence_id
            if evidence_id in evidence_ids:
                raise ValueError(
                    "records cannot contain duplicate canonical evidence IDs"
                )
            evidence_ids.add(evidence_id)
        self.warnings = _copy_unique_string_list(
            self.warnings,
            "warnings",
        )
        self.created_at = _require_aware_datetime(
            self.created_at,
            "created_at",
        )
        self.artifact_ref = _optional_non_empty_string(
            self.artifact_ref,
            "artifact_ref",
        )

    # 返回达到完全核验状态的证据数量
    @property
    def verified_count(self) -> int:
        return self._count_status(EvidenceVerificationStatus.VERIFIED)

    # 返回允许带限制引用的部分核验证据数量
    @property
    def partial_count(self) -> int:
        return self._count_status(EvidenceVerificationStatus.PARTIAL)

    # 返回已经被冲突或身份规则拒绝的证据数量
    @property
    def rejected_count(self) -> int:
        return self._count_status(EvidenceVerificationStatus.REJECTED)

    # 返回可以被后续方向生成阶段引用的证据数量
    @property
    def citable_count(self) -> int:
        return self.verified_count + self.partial_count

    # 返回跨来源合并形成的重复证据组数量
    @property
    def duplicate_group_count(self) -> int:
        return sum(len(record.source_evidence_ids) > 1 for record in self.records)

    # 返回整合过程中保留的字段冲突总数
    @property
    def conflict_count(self) -> int:
        return sum(len(record.conflicts) for record in self.records)

    # 统计指定核验状态的规范化证据数量
    def _count_status(self, status: EvidenceVerificationStatus) -> int:
        return sum(
            record.evidence.verification_status == status for record in self.records
        )

    # 从经过 JSON 解码的对象构造完整核验 Artifact
    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> VerifiedEvidenceBundle:
        _reject_unknown_fields(
            payload,
            {
                "bundle_id",
                "source_artifact_refs",
                "records",
                "warnings",
                "created_at",
                "artifact_ref",
            },
            "VerifiedEvidenceBundle",
        )
        raw_records = payload.get("records", [])
        if not isinstance(raw_records, list):
            raise TypeError("records must be a list")
        records: list[VerifiedEvidenceRecord] = []
        for record in raw_records:
            if not isinstance(record, Mapping):
                raise TypeError("records must contain objects")
            records.append(VerifiedEvidenceRecord.from_dict(record))
        return cls(
            bundle_id=payload.get("bundle_id"),
            source_artifact_refs=payload.get("source_artifact_refs", []),
            records=records,
            warnings=payload.get("warnings", []),
            created_at=_parse_iso_datetime(payload.get("created_at")),
            artifact_ref=payload.get("artifact_ref"),
        )

    # 将完整核验 Artifact 转换为严格的 JSON 对象
    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "source_artifact_refs": list(self.source_artifact_refs),
            "records": [record.to_dict() for record in self.records],
            "warnings": list(self.warnings),
            "created_at": self.created_at.isoformat(),
            "artifact_ref": self.artifact_ref,
        }


# 校验必填字段为非空字符串并返回清理后的值
def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


# 校验可选字段为非空字符串并返回清理后的值
def _optional_non_empty_string(
    value: Any,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name)


# 校验可选字段为正整数
def _optional_positive_integer(
    value: Any,
    field_name: str,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer or None")
    return value


# 校验可选年份处于 Python date 支持的范围
def _optional_year(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer or None")
    if not 1 <= value <= 9999:
        raise ValueError(f"{field_name} must be between 1 and 9999")
    return value


# 复制并校验字符串列表
def _copy_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return [_require_non_empty_string(item, f"{field_name} item") for item in value]


# 复制字符串列表并按首次出现顺序移除重复值
def _copy_unique_string_list(value: Any, field_name: str) -> list[str]:
    return list(dict.fromkeys(_copy_string_list(value, field_name)))


# 复制并校验字符串键值映射
def _copy_string_mapping(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return {
        _require_non_empty_string(key, f"{field_name} key"): _require_non_empty_string(
            item, f"{field_name} value"
        )
        for key, item in value.items()
    }


# 校验论文日期使用 YYYY、YYYY-MM 或 YYYY-MM-DD 格式
def _validate_partial_date(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _require_non_empty_string(value, "publication_date")
    formats = {4: "%Y", 7: "%Y-%m", 10: "%Y-%m-%d"}
    date_format = formats.get(len(normalized))
    if date_format is None:
        raise ValueError("publication_date must use YYYY, YYYY-MM, or YYYY-MM-DD")
    try:
        datetime.strptime(normalized, date_format)
    except ValueError as exc:
        raise ValueError(
            "publication_date must use YYYY, YYYY-MM, or YYYY-MM-DD"
        ) from exc
    return normalized


# 校验可选来源链接为 HTTP 或 HTTPS URL
def _validate_optional_url(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _require_non_empty_string(value, "source_url")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source_url must be a valid HTTP or HTTPS URL")
    return normalized


# 校验必填来源链接为 HTTP 或 HTTPS URL
def _validate_required_url(value: Any, field_name: str) -> str:
    normalized = _require_non_empty_string(value, field_name)
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be a valid HTTP or HTTPS URL")
    return normalized


# 校验时间值包含明确时区
def _require_aware_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return value


# 解析包含明确时区的 ISO-8601 时间字符串
def _parse_iso_datetime(value: Any) -> datetime:
    normalized = _require_non_empty_string(value, "retrieved_at")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("retrieved_at must use ISO-8601 format") from exc
    return _require_aware_datetime(parsed, "retrieved_at")


# 拒绝内部数据结构未声明的字段
def _reject_unknown_fields(
    payload: Mapping[str, Any],
    allowed_fields: set[str],
    model_name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{model_name} payload must be an object")
    unknown_fields = sorted(set(payload) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            f"{model_name} contains unknown fields: " + ", ".join(unknown_fields)
        )
