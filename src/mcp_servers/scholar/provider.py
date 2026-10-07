# File: provider.py
# Author: L1nzhk0
# Purpose: 本文件用于定义 Scholar MCP 可替换的论文检索查询和数据源接口。

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol
from urllib.parse import urlparse

from jmr.domain import DiscoveredSource, ResearchEvidence, RetrievalStatus


@dataclass(slots=True)
class PublicationSearchQuery:
    """保存一次论文检索需要的目标教授、机构和时间范围。"""

    university_name: str
    graduate_school_name: str
    professor_name: str
    search_start_date: date
    search_end_date: date
    professor_name_variants: list[str] = field(default_factory=list)
    laboratory_name: str | None = None
    official_urls: list[str] = field(default_factory=list)
    official_sources: list[DiscoveredSource] = field(default_factory=list)
    max_results: int = 50

    # 校验论文检索查询并复制姓名变体
    def __post_init__(self) -> None:
        self.university_name = _require_non_empty_string(
            self.university_name,
            "university_name",
        )
        self.graduate_school_name = _require_non_empty_string(
            self.graduate_school_name,
            "graduate_school_name",
        )
        self.professor_name = _require_non_empty_string(
            self.professor_name,
            "professor_name",
        )
        if not isinstance(self.search_start_date, date):
            raise TypeError("search_start_date must be a date")
        if not isinstance(self.search_end_date, date):
            raise TypeError("search_end_date must be a date")
        if self.search_start_date > self.search_end_date:
            raise ValueError("search_start_date cannot be after search_end_date")
        if not isinstance(self.professor_name_variants, list):
            raise TypeError("professor_name_variants must be a list")
        self.professor_name_variants = [
            _require_non_empty_string(value, "professor_name_variants item")
            for value in self.professor_name_variants
        ]
        self.laboratory_name = _optional_non_empty_string(
            self.laboratory_name,
            "laboratory_name",
        )
        if not isinstance(self.official_urls, list):
            raise TypeError("official_urls must be a list")
        self.official_urls = [
            _validate_http_url(value, "official_urls item")
            for value in self.official_urls
        ]
        if not isinstance(self.official_sources, list):
            raise TypeError("official_sources must be a list")
        if not all(
            isinstance(source, DiscoveredSource) for source in self.official_sources
        ):
            raise TypeError("official_sources must contain DiscoveredSource items")
        self.official_sources = list(self.official_sources)
        if isinstance(self.max_results, bool) or not isinstance(
            self.max_results,
            int,
        ):
            raise TypeError("max_results must be an integer")
        if not 1 <= self.max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")

    # 从 MCP Tool 参数构造论文检索查询
    @classmethod
    def from_arguments(
        cls,
        arguments: dict[str, Any],
    ) -> PublicationSearchQuery:
        allowed_fields = {
            "university_name",
            "graduate_school_name",
            "professor_name",
            "professor_name_variants",
            "search_start_date",
            "search_end_date",
            "laboratory_name",
            "official_urls",
            "official_sources",
            "max_results",
        }
        unknown_fields = sorted(set(arguments) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                "search_publications received unknown fields: "
                + ", ".join(unknown_fields)
            )
        raw_official_sources = arguments.get("official_sources", [])
        if not isinstance(raw_official_sources, list):
            raise TypeError("official_sources must be a list")
        official_sources: list[DiscoveredSource] = []
        for source in raw_official_sources:
            if not isinstance(source, dict):
                raise TypeError("official_sources must contain objects")
            official_sources.append(DiscoveredSource.from_dict(source))
        return cls(
            university_name=arguments.get("university_name"),
            graduate_school_name=arguments.get("graduate_school_name"),
            professor_name=arguments.get("professor_name"),
            professor_name_variants=arguments.get(
                "professor_name_variants",
                [],
            ),
            search_start_date=_parse_iso_date(
                arguments.get("search_start_date"),
                "search_start_date",
            ),
            search_end_date=_parse_iso_date(
                arguments.get("search_end_date"),
                "search_end_date",
            ),
            laboratory_name=arguments.get("laboratory_name"),
            official_urls=arguments.get("official_urls", []),
            official_sources=official_sources,
            max_results=arguments.get("max_results", 50),
        )

    # 返回可写入 RetrievalResult 的查询说明
    def to_description(self) -> str:
        name_variants = ", ".join(self.professor_name_variants) or "(none)"
        return (
            f"professor={self.professor_name}; "
            f"variants={name_variants}; "
            f"university={self.university_name}; "
            f"graduate_school={self.graduate_school_name}; "
            f"laboratory={self.laboratory_name or '(none)'}; "
            "official_urls="
            f"{len(self.official_urls) + len(self.official_sources)}; "
            f"max_results={self.max_results}; "
            f"date={self.search_start_date.isoformat()}"
            f"..{self.search_end_date.isoformat()}"
        )


@dataclass(slots=True)
class PublicationSearchProviderResult:
    """保存论文数据源返回的状态、记录和诊断信息。"""

    status: RetrievalStatus
    records: list[ResearchEvidence] = field(default_factory=list)
    discovered_sources: list[DiscoveredSource] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # 校验 Provider 返回值的基础类型并复制可变字段
    def __post_init__(self) -> None:
        if not isinstance(self.status, RetrievalStatus):
            raise TypeError("status must be a RetrievalStatus")
        if not isinstance(self.records, list):
            raise TypeError("records must be a list")
        if not all(isinstance(record, ResearchEvidence) for record in self.records):
            raise TypeError("records must contain ResearchEvidence items")
        self.records = list(self.records)
        if not isinstance(self.discovered_sources, list):
            raise TypeError("discovered_sources must be a list")
        if not all(
            isinstance(source, DiscoveredSource) for source in self.discovered_sources
        ):
            raise TypeError("discovered_sources must contain DiscoveredSource items")
        self.discovered_sources = list(self.discovered_sources)
        self.warnings = _copy_string_list(self.warnings, "warnings")
        self.errors = _copy_string_list(self.errors, "errors")


class PublicationSearchProvider(Protocol):
    """定义 Scholar MCP 调用实际论文数据源所需的最小接口。"""

    # 返回用于标记检索结果来源的数据源名称
    @property
    def source_name(self) -> str: ...

    # 使用规范化查询检索论文候选记录
    def search(
        self,
        query: PublicationSearchQuery,
    ) -> PublicationSearchProviderResult: ...


# 校验必填查询字段为非空字符串
def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


# 解析 MCP 参数中的 ISO-8601 日期
def _parse_iso_date(value: Any, field_name: str) -> date:
    normalized = _require_non_empty_string(value, field_name)
    try:
        return date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format") from exc


# 复制并校验诊断信息字符串列表
def _copy_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return [_require_non_empty_string(item, f"{field_name} item") for item in value]


# 校验可选字符串并返回清理后的值
def _optional_non_empty_string(
    value: Any,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name)


# 校验查询中的官网链接为 HTTP 或 HTTPS URL
def _validate_http_url(value: Any, field_name: str) -> str:
    normalized = _require_non_empty_string(value, field_name)
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be a valid HTTP or HTTPS URL")
    return normalized
