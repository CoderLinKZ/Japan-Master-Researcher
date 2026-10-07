# File: server.py
# Author: L1nzhk0
# Purpose: 本文件用于向 MCP Gateway 暴露规范化的论文检索工具。

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from jmr.domain import EvidenceType, RetrievalResult, RetrievalStatus

from .composite import create_default_publication_search_provider
from .provider import PublicationSearchProvider, PublicationSearchQuery

SEARCH_PUBLICATIONS_TOOL_NAME = "search_publications"
DISCOVER_OFFICIAL_SOURCES_TOOL_NAME = "discover_official_sources"


class ScholarMCPServer:
    """通过可替换 Provider 执行论文检索并返回统一检索结果。"""

    # 初始化 Scholar MCP Server
    def __init__(self, provider: PublicationSearchProvider) -> None:
        self._provider = provider

    # 返回 Scholar MCP Server 当前提供的工具定义
    def list_tools(self) -> list[dict[str, Any]]:
        search_tool = {
            "name": SEARCH_PUBLICATIONS_TOOL_NAME,
            "description": (
                "Search publication records for a target professor and "
                "institution within an explicit date range. Returns a "
                "normalized RetrievalResult ordered with verified official-"
                "site records first. Author order, professor position, "
                "source citation, venue and truncation warnings must be "
                "preserved; do not invent missing records or venue names."
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
                    "laboratory_name": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "official_urls": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "format": "uri",
                            "minLength": 1,
                        },
                        "default": [],
                    },
                    "official_sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "format": "uri"},
                                "source_type": {"type": "string"},
                                "discovery_method": {"type": "string"},
                                "verification_status": {"type": "string"},
                                "title": {"type": ["string", "null"]},
                                "matched_signals": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "url",
                                "source_type",
                                "discovery_method",
                                "verification_status",
                            ],
                            "additionalProperties": False,
                        },
                        "default": [],
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 50,
                    },
                    "search_start_date": {
                        "type": "string",
                        "format": "date",
                    },
                    "search_end_date": {
                        "type": "string",
                        "format": "date",
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
        discovery_tool = {
            "name": DISCOVER_OFFICIAL_SOURCES_TOOL_NAME,
            "description": (
                "Discover official laboratory or professor source candidates "
                "for a confirmed target. Returns candidates with provenance "
                "and an explicit status; ambiguous candidates require user "
                "confirmation before they are used."
            ),
            "inputSchema": search_tool["inputSchema"],
        }
        return [search_tool, discovery_tool]

    # 调用 Scholar MCP Server 中的指定工具
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if not isinstance(arguments, Mapping):
            raise TypeError("Scholar tool arguments must be an object")
        if name == SEARCH_PUBLICATIONS_TOOL_NAME:
            return self._search_publications(dict(arguments))
        if name == DISCOVER_OFFICIAL_SOURCES_TOOL_NAME:
            return self._discover_official_sources(dict(arguments))
        raise ValueError(f"Unknown Scholar tool '{name}'")

    def _discover_official_sources(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Return only official-source candidates and their provenance."""

        query = PublicationSearchQuery.from_arguments(arguments)
        discover = getattr(self._provider, "discover_official_sources", None)
        if callable(discover):
            provider_result = discover(query)
            sources = provider_result.sources
        else:
            provider_result = self._provider.search(query)
            sources = provider_result.discovered_sources
        status = provider_result.status
        status_value = (
            RetrievalStatus.NO_RESULT.value
            if not sources and status is RetrievalStatus.SUCCESS
            else status.value
        )
        return {
            "request_id": f"scholar-discovery_{uuid4().hex}",
            "status": status_value,
            "query": query.to_description(),
            "source_name": self._provider.source_name,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "sources": [source.to_dict() for source in sources],
            "warnings": list(provider_result.warnings),
            "errors": list(provider_result.errors),
        }

    # 执行论文检索并构造统一 RetrievalResult
    def _search_publications(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        query = PublicationSearchQuery.from_arguments(arguments)
        provider_result = self._provider.search(query)
        result = RetrievalResult(
            request_id=f"scholar_{uuid4().hex}",
            evidence_type=EvidenceType.PUBLICATION,
            status=provider_result.status,
            query=query.to_description(),
            source_name=self._provider.source_name,
            retrieved_at=datetime.now(UTC),
            records=provider_result.records,
            discovered_sources=provider_result.discovered_sources,
            warnings=provider_result.warnings,
            errors=provider_result.errors,
        )
        return result.to_dict()


# 创建使用指定 Provider 或 OpenAlex 默认组合 Provider 的 Scholar Server
def create_scholar_server(
    provider: PublicationSearchProvider | None = None,
) -> ScholarMCPServer:
    selected_provider = (
        provider
        if provider is not None
        else create_default_publication_search_provider(
            openalex_api_key=os.getenv("OPENALEX_API_KEY"),
        )
    )
    return ScholarMCPServer(selected_provider)
