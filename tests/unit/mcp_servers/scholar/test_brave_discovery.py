# File: test_brave_discovery.py
# Author: L1nzhk0
# Purpose: 本文件用于验证 Brave 官网发现的身份信号筛选和歧义处理。

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.domain import (  # noqa: E402
    DiscoveredSourceType,
    RetrievalStatus,
    SourceDiscoveryMethod,
)
from mcp_servers.scholar import (  # noqa: E402
    BraveOfficialSiteDiscoveryAdapter,
    BraveSearchResult,
    PublicationSearchQuery,
)


class FakeBraveSearchClient:
    """返回固定 Brave 搜索结果并记录查询。"""

    # 初始化固定搜索结果
    def __init__(self, results: list[BraveSearchResult]) -> None:
        self._results = results
        self.queries: list[str] = []

    # 记录查询并返回固定搜索结果
    def search(self, query: str, count: int = 10) -> list[BraveSearchResult]:
        self.queries.append(query)
        return list(self._results)


# 构造官网发现测试使用的目标查询
def search_query() -> PublicationSearchQuery:
    return PublicationSearchQuery.from_arguments(
        {
            "university_name": "Example University",
            "graduate_school_name": "Graduate School of Engineering",
            "professor_name": "Example Professor",
            "laboratory_name": "Example Laboratory",
            "search_start_date": "2025-01-01",
            "search_end_date": "2026-08-30",
        }
    )


class BraveOfficialSiteDiscoveryAdapterTests(unittest.TestCase):
    # 验证唯一高校域名候选会被转换为结构化官网来源
    def test_discovers_unique_official_candidate(self) -> None:
        client = FakeBraveSearchClient(
            [
                BraveSearchResult(
                    title="Example Laboratory Publications",
                    url="https://example.ac.jp/lab/publications",
                    description=(
                        "Example Professor, Example University, Example Laboratory"
                    ),
                ),
                BraveSearchResult(
                    title="Example Professor on ResearchGate",
                    url="https://researchgate.net/example",
                    description="Example University",
                ),
            ]
        )
        adapter = BraveOfficialSiteDiscoveryAdapter(client)

        result = adapter.discover(search_query())

        self.assertEqual(result.status, RetrievalStatus.SUCCESS)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(
            result.sources[0].source_type,
            DiscoveredSourceType.PUBLICATION_LIST,
        )
        self.assertEqual(
            result.sources[0].discovery_method,
            SourceDiscoveryMethod.WEB_SEARCH,
        )

    # 验证多个得分接近的官网候选会请求用户确认
    def test_returns_confirmation_for_close_candidates(self) -> None:
        client = FakeBraveSearchClient(
            [
                BraveSearchResult(
                    title="Example Laboratory",
                    url="https://lab-one.example.ac.jp/",
                    description="Example Professor Example University",
                ),
                BraveSearchResult(
                    title="Example Laboratory",
                    url="https://lab-two.example.ac.jp/",
                    description="Example Professor Example University",
                ),
            ]
        )
        adapter = BraveOfficialSiteDiscoveryAdapter(client)

        result = adapter.discover(search_query())

        self.assertEqual(
            result.status,
            RetrievalStatus.NEEDS_USER_CONFIRMATION,
        )
        self.assertEqual(len(result.sources), 2)

    # 验证缺少教授身份信号的普通网页不会被当成官网
    def test_rejects_result_without_professor_signal(self) -> None:
        client = FakeBraveSearchClient(
            [
                BraveSearchResult(
                    title="Example University",
                    url="https://example.ac.jp/",
                    description="Graduate School of Engineering",
                )
            ]
        )
        adapter = BraveOfficialSiteDiscoveryAdapter(client)

        result = adapter.discover(search_query())

        self.assertEqual(result.status, RetrievalStatus.NO_RESULT)
        self.assertEqual(result.sources, [])


if __name__ == "__main__":
    unittest.main()
