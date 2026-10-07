# File: server.py
# Author: L1nzhk0
# Purpose: 本文件用于向 MCP Gateway 暴露规范化的 KAKEN 课题检索工具。

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from jmr.domain import EvidenceType, RetrievalResult

from .nii import NIIKakenSearchProvider
from .provider import KakenSearchProvider, KakenSearchQuery

SEARCH_PROJECTS_TOOL_NAME = "search_projects"


class KakenMCPServer:
    """通过可替换 Provider 执行 KAKEN 课题检索并返回统一结果。"""

    # 初始化 KAKEN MCP Server
    def __init__(self, provider: KakenSearchProvider) -> None:
        self._provider = provider

    # 返回 KAKEN MCP Server 当前提供的工具定义
    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": SEARCH_PROJECTS_TOOL_NAME,
                "description": (
                    "Search official KAKEN research projects for the target "
                    "professor, institution and date range. Returns normalized "
                    "project numbers, titles, periods, member roles, summaries, "
                    "official URLs and verification status."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "university_name": {"type": "string", "minLength": 1},
                        "graduate_school_name": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "professor_name": {"type": "string", "minLength": 1},
                        "professor_name_variants": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "default": [],
                        },
                        "search_start_date": {
                            "type": "string",
                            "format": "date",
                        },
                        "search_end_date": {
                            "type": "string",
                            "format": "date",
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 50,
                        },
                    },
                    "required": [
                        "university_name",
                        "graduate_school_name",
                        "professor_name",
                        "search_start_date",
                        "search_end_date",
                    ],
                    "additionalProperties": False,
                },
            }
        ]

    # 调用 KAKEN MCP Server 中的指定工具
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name != SEARCH_PROJECTS_TOOL_NAME:
            raise ValueError(f"Unknown KAKEN tool '{name}'")
        if not isinstance(arguments, Mapping):
            raise TypeError("KAKEN tool arguments must be an object")
        return self._search_projects(dict(arguments))

    # 执行课题检索并构造统一 RetrievalResult
    def _search_projects(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = KakenSearchQuery.from_arguments(arguments)
        provider_result = self._provider.search(query)
        result = RetrievalResult(
            request_id=f"kaken_{uuid4().hex}",
            evidence_type=EvidenceType.KAKEN_PROJECT,
            status=provider_result.status,
            query=query.to_description(),
            source_name=self._provider.source_name,
            retrieved_at=datetime.now(UTC),
            records=provider_result.records,
            warnings=provider_result.warnings,
            errors=provider_result.errors,
        )
        return result.to_dict()


# 创建使用指定 Provider 或 KAKEN 官方 API Provider 的 Server
def create_kaken_server(
    provider: KakenSearchProvider | None = None,
) -> KakenMCPServer:
    selected_provider = (
        provider
        if provider is not None
        else NIIKakenSearchProvider(os.getenv("KAKEN_APP_ID"))
    )
    return KakenMCPServer(selected_provider)
