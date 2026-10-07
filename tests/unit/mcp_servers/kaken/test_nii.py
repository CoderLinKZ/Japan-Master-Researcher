# File: test_nii.py
# Author: L1nzhk0
# Purpose: 本文件用于验证 KAKEN 官方 XML 课题解析、教授匹配、机构核验和配置降级。

from __future__ import annotations

import sys
import unittest
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[5] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.domain import (  # noqa: E402
    EvidenceType,
    EvidenceVerificationStatus,
    RetrievalStatus,
)
from mcp_servers.kaken import (  # noqa: E402
    KakenSearchQuery,
    NIIKakenSearchProvider,
)

KAKEN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<grantAwards xmlns="https://kaken.nii.ac.jp/xml/">
  <totalResults>1</totalResults>
  <grantAward id="KAKENHI-PROJECT-24K00001" awardNumber="24K00001">
    <urlList>
      <url>https://kaken.nii.ac.jp/grant/KAKENHI-PROJECT-24K00001/</url>
    </urlList>
    <title xml:lang="ja">拡張現実における協調作業の研究</title>
    <title xml:lang="en">Collaborative Work in Augmented Reality</title>
    <category>基盤研究(B)</category>
    <periodOfAward>
      <fiscalYear>2024</fiscalYear>
      <fiscalYear>2027</fiscalYear>
    </periodOfAward>
    <projectStatus>granted</projectStatus>
    <member role="principal_investigator">
      <personalName><fullName>清川 清</fullName></personalName>
      <affiliation>
        <institution>奈良先端科学技術大学院大学</institution>
        <department>先端科学技術研究科</department>
      </affiliation>
    </member>
    <member role="co_investigator_buntan">
      <personalName><fullName>共同 研究者</fullName></personalName>
      <affiliation><institution>他大学</institution></affiliation>
    </member>
    <paragraphList>
      <paragraph>没入型環境における協調作業を研究する。</paragraph>
    </paragraphList>
  </grantAward>
</grantAwards>
"""


class FakeXMLHTTPClient:
    """返回固定 KAKEN XML 并记录官方 API 参数。"""

    # 初始化请求记录
    def __init__(self, xml_text: str = KAKEN_XML) -> None:
        self._xml_text = xml_text
        self.calls: list[tuple[str, dict[str, str | int]]] = []

    # 记录请求并解析固定 XML
    def get_xml(
        self,
        url: str,
        params: Mapping[str, str | int],
    ) -> ET.Element:
        self.calls.append((url, dict(params)))
        return ET.fromstring(self._xml_text)


class NIIKakenSearchProviderTests(unittest.TestCase):
    # 构造覆盖固定课题期间的 KAKEN 查询
    def query(self) -> KakenSearchQuery:
        from datetime import date

        return KakenSearchQuery(
            university_name="奈良先端科学技術大学院大学",
            graduate_school_name="先端科学技術研究科",
            professor_name="清川清",
            search_start_date=date(2025, 1, 1),
            search_end_date=date(2026, 8, 30),
            max_results=20,
        )

    # 验证官方 XML 会转换为包含 KAKEN 专属字段的统一证据
    def test_normalizes_official_project_xml(self) -> None:
        http_client = FakeXMLHTTPClient()
        provider = NIIKakenSearchProvider("test-app-id", http_client)

        result = provider.search(self.query())

        self.assertEqual(result.status, RetrievalStatus.SUCCESS)
        self.assertEqual(len(result.records), 1)
        record = result.records[0]
        self.assertEqual(record.evidence_type, EvidenceType.KAKEN_PROJECT)
        self.assertEqual(record.identifiers["kaken_project_id"], "24K00001")
        self.assertEqual(record.project_start_year, 2024)
        self.assertEqual(record.project_end_year, 2027)
        self.assertEqual(record.project_status, "granted")
        self.assertEqual(record.matched_professor, "清川 清")
        self.assertEqual(record.matched_professor_position, 1)
        self.assertEqual(
            record.participant_roles,
            ["principal_investigator", "co_investigator_buntan"],
        )
        self.assertEqual(
            record.verification_status,
            EvidenceVerificationStatus.VERIFIED,
        )
        params = http_client.calls[0][1]
        self.assertEqual(params["qg"], "清川清")
        self.assertEqual(params["s1"], 2025)
        self.assertEqual(params["s2"], 2026)
        self.assertEqual(params["format"], "xml")

    # 验证未配置官方 Application ID 时返回可保存的阻塞结果
    def test_missing_app_id_returns_blocked_result(self) -> None:
        provider = NIIKakenSearchProvider(None, FakeXMLHTTPClient())

        result = provider.search(self.query())

        self.assertEqual(result.status, RetrievalStatus.BLOCKED)
        self.assertEqual(result.records, [])
        self.assertIn("KAKEN_APP_ID", result.errors[0])


if __name__ == "__main__":
    unittest.main()
