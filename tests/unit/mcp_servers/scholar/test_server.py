# File: test_server.py
# Author: L1nzhk0
# Purpose: 本文件用于验证 Scholar MCP 工具定义、查询校验和规范化检索结果。

from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.domain import (  # noqa: E402
    EvidenceType,
    ResearchEvidence,
    RetrievalResult,
    RetrievalStatus,
    SourceDiscoveryMethod,
)
from mcp_servers.scholar import (  # noqa: E402
    DISCOVER_OFFICIAL_SOURCES_TOOL_NAME,
    SEARCH_PUBLICATIONS_TOOL_NAME,
    PublicationSearchProviderResult,
    PublicationSearchQuery,
    create_scholar_server,
)


class FakePublicationSearchProvider:
    """返回两篇固定论文并记录收到的规范化查询。"""

    # 初始化 Fake Provider 的查询记录
    def __init__(self) -> None:
        self.queries: list[PublicationSearchQuery] = []

    # 返回测试数据源名称
    @property
    def source_name(self) -> str:
        return "fake-scholar-index"

    # 返回两条论文证据
    def search(
        self,
        query: PublicationSearchQuery,
    ) -> PublicationSearchProviderResult:
        self.queries.append(query)
        retrieved_at = datetime(2026, 8, 30, tzinfo=UTC)
        records = [
            ResearchEvidence(
                evidence_id="pub_fake_001",
                evidence_type=EvidenceType.PUBLICATION,
                title="First Paper",
                source_name=self.source_name,
                retrieved_at=retrieved_at,
                authors=["Example Professor", "First Author"],
                publication_date="2026-03",
                source_url="https://example.test/papers/1",
            ),
            ResearchEvidence(
                evidence_id="pub_fake_002",
                evidence_type=EvidenceType.PUBLICATION,
                title="Second Paper",
                source_name=self.source_name,
                retrieved_at=retrieved_at,
                authors=["Example Professor", "Second Author"],
                publication_date="2025",
                source_url="https://example.test/papers/2",
            ),
        ]
        return PublicationSearchProviderResult(
            status=RetrievalStatus.SUCCESS,
            records=records,
        )


class ScholarMCPServerTests(unittest.TestCase):
    # 返回一组合法的论文检索 Tool 参数
    def search_arguments(self) -> dict[str, object]:
        return {
            "university_name": "Example University",
            "graduate_school_name": "Engineering",
            "professor_name": "Example Professor",
            "professor_name_variants": ["E. Professor"],
            "laboratory_name": "Example Laboratory",
            "official_urls": ["https://example.test/lab"],
            "max_results": 12,
            "search_start_date": "2025-08-30",
            "search_end_date": "2026-08-30",
        }

    # 验证 Server 只暴露严格参数结构的论文检索工具
    def test_lists_search_publications_tool(self) -> None:
        server = create_scholar_server(FakePublicationSearchProvider())

        tools = server.list_tools()

        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["name"], SEARCH_PUBLICATIONS_TOOL_NAME)
        self.assertEqual(tools[1]["name"], DISCOVER_OFFICIAL_SOURCES_TOOL_NAME)
        self.assertFalse(tools[0]["inputSchema"]["additionalProperties"])
        self.assertEqual(
            tools[0]["inputSchema"]["properties"]["max_results"]["default"],
            50,
        )

    # 验证未显式指定数量时使用适合两年清单的默认五十条上限
    def test_search_uses_default_max_results(self) -> None:
        provider = FakePublicationSearchProvider()
        server = create_scholar_server(provider)
        arguments = self.search_arguments()
        arguments.pop("max_results")

        server.call_tool(SEARCH_PUBLICATIONS_TOOL_NAME, arguments)

        self.assertEqual(provider.queries[0].max_results, 50)

    # 验证一次 Tool 调用能够返回包含多篇论文的 RetrievalResult
    def test_search_returns_normalized_retrieval_result(self) -> None:
        provider = FakePublicationSearchProvider()
        server = create_scholar_server(provider)

        payload = server.call_tool(
            SEARCH_PUBLICATIONS_TOOL_NAME,
            self.search_arguments(),
        )
        result = RetrievalResult.from_dict(payload)

        self.assertEqual(result.status, RetrievalStatus.SUCCESS)
        self.assertEqual(result.evidence_type, EvidenceType.PUBLICATION)
        self.assertEqual(result.record_count, 2)
        self.assertEqual(result.source_name, "fake-scholar-index")
        self.assertEqual(
            provider.queries[0].professor_name_variants,
            ["E. Professor"],
        )
        self.assertEqual(
            provider.queries[0].official_urls,
            ["https://example.test/lab"],
        )
        self.assertEqual(provider.queries[0].max_results, 12)

    # 验证宿主注入的结构化官网候选可以保留模型搜索来源
    def test_accepts_structured_official_source_provenance(self) -> None:
        provider = FakePublicationSearchProvider()
        server = create_scholar_server(provider)
        arguments = self.search_arguments()
        arguments["official_urls"] = []
        arguments["official_sources"] = [
            {
                "url": "https://lab.example.edu/publications/",
                "source_type": "publication_list",
                "discovery_method": "web_search",
                "verification_status": "UNVERIFIED",
                "title": "Official Publications",
                "matched_signals": [
                    "Example Professor",
                    "Example University",
                ],
            }
        ]

        server.call_tool(SEARCH_PUBLICATIONS_TOOL_NAME, arguments)

        source = provider.queries[0].official_sources[0]
        self.assertEqual(
            source.discovery_method,
            SourceDiscoveryMethod.WEB_SEARCH,
        )
        self.assertEqual(source.source_type.value, "publication_list")

    # 验证非法日期会在调用 Provider 前被拒绝
    def test_rejects_invalid_search_date(self) -> None:
        provider = FakePublicationSearchProvider()
        server = create_scholar_server(provider)
        arguments = self.search_arguments()
        arguments["search_start_date"] = "2026/01/01"

        with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
            server.call_tool(SEARCH_PUBLICATIONS_TOOL_NAME, arguments)

        self.assertEqual(provider.queries, [])

    # 验证官网协议和最大结果数会在调用 Provider 前完成校验
    def test_rejects_invalid_optional_search_fields(self) -> None:
        provider = FakePublicationSearchProvider()
        server = create_scholar_server(provider)
        invalid_url_arguments = self.search_arguments()
        invalid_url_arguments["official_urls"] = ["file:///tmp/lab.html"]

        with self.assertRaisesRegex(ValueError, "HTTP or HTTPS"):
            server.call_tool(
                SEARCH_PUBLICATIONS_TOOL_NAME,
                invalid_url_arguments,
            )

        invalid_limit_arguments = self.search_arguments()
        invalid_limit_arguments["max_results"] = 101
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            server.call_tool(
                SEARCH_PUBLICATIONS_TOOL_NAME,
                invalid_limit_arguments,
            )
        self.assertEqual(provider.queries, [])

    # 验证生产默认 Server 已改用 OpenAlex 组合 Provider 且测试不发起网络请求
    def test_default_server_uses_openalex_composite_provider(self) -> None:
        server = create_scholar_server()

        self.assertEqual(
            server._provider.source_name,
            "composite:OpenAlex+official-site",
        )


if __name__ == "__main__":
    unittest.main()
