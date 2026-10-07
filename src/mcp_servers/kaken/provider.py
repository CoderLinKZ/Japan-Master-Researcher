# File: provider.py
# Author: L1nzhk0
# Purpose: 本文件用于定义 KAKEN MCP 的规范化课题查询、结果和可替换 Provider 接口。

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from jmr.domain import ResearchEvidence, RetrievalStatus


@dataclass(slots=True)
class KakenSearchQuery:
    """保存一次 KAKEN 课题检索所需的教授、机构和时间范围。"""

    university_name: str
    graduate_school_name: str
    professor_name: str
    search_start_date: date
    search_end_date: date
    professor_name_variants: list[str] = field(default_factory=list)
    max_results: int = 50

    # 校验 KAKEN 查询的全部结构化字段
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
        if isinstance(self.max_results, bool) or not isinstance(self.max_results, int):
            raise TypeError("max_results must be an integer")
        if not 1 <= self.max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")

    # 从 MCP Tool 参数构造 KAKEN 课题查询
    @classmethod
    def from_arguments(cls, arguments: dict[str, Any]) -> KakenSearchQuery:
        allowed_fields = {
            "university_name",
            "graduate_school_name",
            "professor_name",
            "professor_name_variants",
            "search_start_date",
            "search_end_date",
            "max_results",
        }
        unknown_fields = sorted(set(arguments) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                "search_projects received unknown fields: " + ", ".join(unknown_fields)
            )
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
            max_results=arguments.get("max_results", 50),
        )

    # 返回不包含密钥的检索条件说明
    def to_description(self) -> str:
        variants = ", ".join(self.professor_name_variants) or "(none)"
        return (
            f"professor={self.professor_name}; variants={variants}; "
            f"university={self.university_name}; "
            f"graduate_school={self.graduate_school_name}; "
            f"max_results={self.max_results}; "
            f"date={self.search_start_date.isoformat()}"
            f"..{self.search_end_date.isoformat()}"
        )


@dataclass(slots=True)
class KakenSearchProviderResult:
    """保存 KAKEN 数据源返回的课题、状态和诊断信息。"""

    status: RetrievalStatus
    records: list[ResearchEvidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # 校验 Provider 结果并复制可变字段
    def __post_init__(self) -> None:
        if not isinstance(self.status, RetrievalStatus):
            raise TypeError("status must be a RetrievalStatus")
        if not isinstance(self.records, list) or not all(
            isinstance(record, ResearchEvidence) for record in self.records
        ):
            raise TypeError("records must contain ResearchEvidence items")
        self.records = list(self.records)
        self.warnings = _copy_string_list(self.warnings, "warnings")
        self.errors = _copy_string_list(self.errors, "errors")


class KakenSearchProvider(Protocol):
    """定义 KAKEN MCP 调用课题数据源所需的最小接口。"""

    # 返回用于标记检索结果来源的数据源名称
    @property
    def source_name(self) -> str: ...

    # 使用规范化查询检索 KAKEN 课题
    def search(self, query: KakenSearchQuery) -> KakenSearchProviderResult: ...


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


# 复制并校验诊断字符串列表
def _copy_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return [_require_non_empty_string(item, f"{field_name} item") for item in value]
