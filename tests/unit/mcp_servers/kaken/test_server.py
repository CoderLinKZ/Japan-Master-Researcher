# File: test_server.py
# Author: L1nzhk0
# Purpose: 本文件用于验证 KAKEN MCP 工具定义和 RetrievalResult 组装。

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[5] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.domain import EvidenceType, RetrievalResult, RetrievalStatus  # noqa: E402
from mcp_servers.kaken import (  # noqa: E402
    SEARCH_PROJECTS_TOOL_NAME,
    KakenSearchProviderResult,
    KakenSearchQuery,
    create_kaken_server,
)


class FakeKakenProvider:
    """为 Server 测试返回固定无结果状态。"""

    # 返回测试数据源名称
    @property
    def source_name(self) -> str:
        return "fake-kaken"

    # 记录查询类型并返回无结果
    def search(self, query: KakenSearchQuery) -> KakenSearchProviderResult:
        if not isinstance(query, KakenSearchQuery):
            raise TypeError("unexpected query")
        return KakenSearchProviderResult(status=RetrievalStatus.NO_RESULT)


class KakenMCPServerTests(unittest.TestCase):
    # 验证 Server 暴露 search_projects 工具
    def test_lists_search_projects_tool(self) -> None:
        server = create_kaken_server(FakeKakenProvider())

        tools = server.list_tools()

        self.assertEqual(tools[0]["name"], SEARCH_PROJECTS_TOOL_NAME)
        self.assertIn(
            "professor_name",
            tools[0]["inputSchema"]["required"],
        )

    # 验证 Server 把 Provider 结果包装为 KAKEN RetrievalResult
    def test_returns_typed_retrieval_result(self) -> None:
        server = create_kaken_server(FakeKakenProvider())

        payload = server.call_tool(
            SEARCH_PROJECTS_TOOL_NAME,
            {
                "university_name": "Example University",
                "graduate_school_name": "Engineering",
                "professor_name": "Example Professor",
                "search_start_date": "2025-01-01",
                "search_end_date": "2026-08-30",
            },
        )
        result = RetrievalResult.from_dict(payload)

        self.assertEqual(result.evidence_type, EvidenceType.KAKEN_PROJECT)
        self.assertEqual(result.status, RetrievalStatus.NO_RESULT)
        self.assertTrue(result.request_id.startswith("kaken_"))


if __name__ == "__main__":
    unittest.main()
