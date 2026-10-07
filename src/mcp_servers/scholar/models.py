# File: models.py
# Author: L1nzhk0
# Purpose: 本文件用于定义学术索引、官网发现和官网论文检索之间传递的内部结构。

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from jmr.domain import (
    DiscoveredSource,
    EvidenceVerificationStatus,
    RetrievalStatus,
)


@dataclass(slots=True)
class RawPublicationCandidate:
    """保存 Adapter 返回且尚未转换为正式研究证据的论文候选。"""

    source_record_id: str
    title: str
    source_name: str
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    publication_date: str | None = None
    venue: str | None = None
    source_url: str | None = None
    identifiers: dict[str, str] = field(default_factory=dict)
    abstract_or_summary: str | None = None
    source_citation: str | None = None
    matched_professor: str | None = None
    matched_professor_position: int | None = None
    matched_professor_affiliations: list[str] = field(default_factory=list)
    verification_status: EvidenceVerificationStatus = (
        EvidenceVerificationStatus.UNVERIFIED
    )

    # 校验原始论文候选并复制 Adapter 返回的可变字段
    def __post_init__(self) -> None:
        self.source_record_id = _require_non_empty_string(
            self.source_record_id,
            "source_record_id",
        )
        self.title = _require_non_empty_string(self.title, "title")
        self.source_name = _require_non_empty_string(
            self.source_name,
            "source_name",
        )
        self.authors = _copy_string_list(self.authors, "authors")
        self.affiliations = _copy_string_list(
            self.affiliations,
            "affiliations",
        )
        self.publication_date = _optional_non_empty_string(
            self.publication_date,
            "publication_date",
        )
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


@dataclass(slots=True)
class PublicationAdapterResult:
    """保存单个论文 Adapter 的候选、状态与诊断信息。"""

    status: RetrievalStatus
    candidates: list[RawPublicationCandidate] = field(default_factory=list)
    source_updates: list[DiscoveredSource] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # 校验单个论文 Adapter 的返回结果
    def __post_init__(self) -> None:
        if not isinstance(self.status, RetrievalStatus):
            raise TypeError("status must be a RetrievalStatus")
        if not isinstance(self.candidates, list):
            raise TypeError("candidates must be a list")
        if not all(
            isinstance(candidate, RawPublicationCandidate)
            for candidate in self.candidates
        ):
            raise TypeError("candidates must contain RawPublicationCandidate items")
        self.candidates = list(self.candidates)
        if not isinstance(self.source_updates, list):
            raise TypeError("source_updates must be a list")
        if not all(
            isinstance(source, DiscoveredSource) for source in self.source_updates
        ):
            raise TypeError("source_updates must contain DiscoveredSource items")
        self.source_updates = list(self.source_updates)
        self.warnings = _copy_string_list(self.warnings, "warnings")
        self.errors = _copy_string_list(self.errors, "errors")


@dataclass(slots=True)
class SourceDiscoveryResult:
    """保存官网发现 Adapter 返回的候选来源和歧义状态。"""

    status: RetrievalStatus
    sources: list[DiscoveredSource] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # 校验官网发现 Adapter 的返回结果
    def __post_init__(self) -> None:
        if not isinstance(self.status, RetrievalStatus):
            raise TypeError("status must be a RetrievalStatus")
        if not isinstance(self.sources, list):
            raise TypeError("sources must be a list")
        if not all(isinstance(source, DiscoveredSource) for source in self.sources):
            raise TypeError("sources must contain DiscoveredSource items")
        self.sources = list(self.sources)
        self.warnings = _copy_string_list(self.warnings, "warnings")
        self.errors = _copy_string_list(self.errors, "errors")


# 校验必填字符串并返回清理后的值
def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


# 校验可选字符串并返回清理后的值
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


# 复制并校验字符串列表
def _copy_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return [_require_non_empty_string(item, f"{field_name} item") for item in value]


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


# 校验可选来源链接为 HTTP 或 HTTPS URL
def _validate_optional_url(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _require_non_empty_string(value, "source_url")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source_url must be a valid HTTP or HTTPS URL")
    return normalized
