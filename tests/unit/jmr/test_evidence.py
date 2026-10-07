# File: test_evidence.py
# Author: L1nzhk0
# Purpose: 本文件用于验证研究证据、检索批次和 Artifact 引用的数据边界。

from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.domain import (  # noqa: E402
    DiscoveredSource,
    DiscoveredSourceType,
    EvidenceType,
    EvidenceVerificationStatus,
    ResearchEvidence,
    RetrievalResult,
    RetrievalStatus,
    SourceDiscoveryMethod,
)


# 构造用于检索批次测试的论文证据
def publication_record(
    evidence_id: str,
    title: str,
) -> ResearchEvidence:
    return ResearchEvidence(
        evidence_id=evidence_id,
        evidence_type=EvidenceType.PUBLICATION,
        title=title,
        source_name="scholar-test",
        retrieved_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        authors=["Example Professor", "Example Author"],
        affiliations=["Example University"],
        publication_date="2026-08",
        venue="Example Journal",
        source_url=f"https://example.test/{evidence_id}",
        identifiers={"doi": f"10.0000/{evidence_id}"},
        source_citation=(
            f"Example Author, Example Professor: {title}, Example Journal, 2026."
        ),
        matched_professor="Example Professor",
        matched_professor_position=1,
        matched_professor_affiliations=["Example University"],
        verification_status=EvidenceVerificationStatus.PARTIAL,
    )


class ResearchEvidenceTests(unittest.TestCase):
    # 验证一个检索批次能够保存多条相同类型的论文证据
    def test_retrieval_result_contains_multiple_publications(self) -> None:
        first = publication_record("pub_001", "Paper One")
        second = publication_record("pub_002", "Paper Two")

        result = RetrievalResult(
            request_id="request_001",
            evidence_type=EvidenceType.PUBLICATION,
            status=RetrievalStatus.SUCCESS,
            query="Example Professor Example University",
            source_name="scholar-test",
            retrieved_at=datetime(
                2026,
                8,
                29,
                12,
                1,
                tzinfo=UTC,
            ),
            records=[first, second],
        )

        self.assertEqual(result.record_count, 2)
        self.assertEqual(result.evidence_ids, ("pub_001", "pub_002"))

    # 验证检索结果能够完成 JSON 对象的严格往返转换
    def test_retrieval_result_round_trips_through_dict(self) -> None:
        original = RetrievalResult(
            request_id="request_002",
            evidence_type=EvidenceType.PUBLICATION,
            status=RetrievalStatus.PARTIAL,
            query="Example Professor",
            source_name="scholar-test",
            records=[publication_record("pub_003", "Paper Three")],
            discovered_sources=[
                DiscoveredSource(
                    url="https://example.test/lab",
                    source_type=DiscoveredSourceType.LABORATORY_WEBSITE,
                    discovery_method=SourceDiscoveryMethod.WEB_SEARCH,
                    verification_status=EvidenceVerificationStatus.PARTIAL,
                    matched_signals=["大学名称", "教授姓名"],
                )
            ],
            warnings=["部分来源尚未核验"],
        )

        restored = RetrievalResult.from_dict(original.to_dict())

        self.assertEqual(restored.request_id, original.request_id)
        self.assertEqual(restored.evidence_ids, original.evidence_ids)
        self.assertEqual(restored.records[0].authors, original.records[0].authors)
        self.assertEqual(restored.records[0].venue, "Example Journal")
        self.assertEqual(restored.records[0].matched_professor_position, 1)
        self.assertEqual(
            restored.records[0].matched_professor_affiliations,
            ["Example University"],
        )
        self.assertIn(
            "Paper Three",
            restored.records[0].source_citation,
        )
        self.assertIsNot(restored.records, original.records)
        self.assertEqual(
            restored.discovered_sources[0].url,
            "https://example.test/lab",
        )

    # 验证内部模型会拒绝未声明的任意字段
    def test_research_evidence_rejects_unknown_fields(self) -> None:
        payload = publication_record("pub_004", "Paper Four").to_dict()
        payload["invented_field"] = "unexpected"

        with self.assertRaisesRegex(ValueError, "unknown fields"):
            ResearchEvidence.from_dict(payload)

    # 验证无结果状态与成功状态对记录数量有不同约束
    def test_retrieval_status_validates_record_presence(self) -> None:
        no_result = RetrievalResult(
            request_id="request_003",
            evidence_type=EvidenceType.KAKEN_PROJECT,
            status=RetrievalStatus.NO_RESULT,
            query="Example Professor",
            source_name="kaken-test",
        )

        self.assertEqual(no_result.records, [])
        with self.assertRaisesRegex(ValueError, "at least one record"):
            RetrievalResult(
                request_id="request_004",
                evidence_type=EvidenceType.PUBLICATION,
                status=RetrievalStatus.SUCCESS,
                query="Example Professor",
                source_name="scholar-test",
            )

    # 验证一次检索批次不能混入其他证据类型的记录
    def test_retrieval_result_rejects_mixed_evidence_types(self) -> None:
        kaken_record = ResearchEvidence(
            evidence_id="kaken_001",
            evidence_type=EvidenceType.KAKEN_PROJECT,
            title="KAKEN Project",
            source_name="kaken-test",
        )

        with self.assertRaisesRegex(ValueError, "must match"):
            RetrievalResult(
                request_id="request_005",
                evidence_type=EvidenceType.PUBLICATION,
                status=RetrievalStatus.PARTIAL,
                query="Example Professor",
                source_name="combined-test",
                records=[kaken_record],
            )

    # 验证一次检索批次不能包含重复的内部证据 ID
    def test_retrieval_result_rejects_duplicate_evidence_ids(self) -> None:
        first = publication_record("pub_duplicate", "Paper One")
        second = publication_record("pub_duplicate", "Paper Two")

        with self.assertRaisesRegex(ValueError, "duplicate evidence_id"):
            RetrievalResult(
                request_id="request_006",
                evidence_type=EvidenceType.PUBLICATION,
                status=RetrievalStatus.SUCCESS,
                query="Example Professor",
                source_name="scholar-test",
                records=[first, second],
            )


if __name__ == "__main__":
    unittest.main()
