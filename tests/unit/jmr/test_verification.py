"""Tests for deterministic evidence merge and verification."""

from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.domain import (  # noqa: E402
    EvidenceType,
    EvidenceVerificationStatus,
    ResearchEvidence,
    RetrievalResult,
    RetrievalStatus,
)
from jmr.domain.verification import (  # noqa: E402
    merge_and_verify_retrieval_results,
)


def publication_record(
    evidence_id: str,
    source_name: str,
    *,
    title: str = "Shared Research Paper",
    publication_date: str = "2026-05",
    doi: str | None = "10.1000/shared",
    status: EvidenceVerificationStatus = EvidenceVerificationStatus.PARTIAL,
) -> ResearchEvidence:
    return ResearchEvidence(
        evidence_id=evidence_id,
        evidence_type=EvidenceType.PUBLICATION,
        title=title,
        source_name=source_name,
        retrieved_at=datetime(2026, 8, 30, tzinfo=UTC),
        authors=["Example Professor", "Example Author"],
        affiliations=["Example University"],
        publication_date=publication_date,
        source_url=f"https://example.test/{evidence_id}",
        identifiers={"doi": doi} if doi else {},
        matched_professor="Example Professor",
        matched_professor_position=1,
        matched_professor_affiliations=["Example University"],
        verification_status=status,
    )


def retrieval_result(records: list[ResearchEvidence]) -> RetrievalResult:
    return RetrievalResult(
        request_id="publication_request",
        evidence_type=EvidenceType.PUBLICATION,
        status=RetrievalStatus.SUCCESS if records else RetrievalStatus.NO_RESULT,
        query="Example Professor Example University",
        source_name="test-publication",
        records=records,
    )


class EvidenceVerificationTests(unittest.TestCase):
    def test_merges_same_doi_and_preserves_provenance(self) -> None:
        official = publication_record(
            "pub_official",
            "official-site:example.test",
            doi="https://doi.org/10.1000/SHARED",
            status=EvidenceVerificationStatus.VERIFIED,
        )
        openalex = publication_record("pub_openalex", "OpenAlex")

        bundle = merge_and_verify_retrieval_results(
            [retrieval_result([official, openalex])],
            ["retrieval-run-1"],
        )

        self.assertEqual(len(bundle.records), 1)
        self.assertEqual(bundle.duplicate_group_count, 1)
        record = bundle.records[0]
        self.assertEqual(
            record.evidence.verification_status,
            EvidenceVerificationStatus.VERIFIED,
        )
        self.assertEqual(
            record.source_evidence_ids,
            ["pub_official", "pub_openalex"],
        )

    def test_conflicting_strong_identifier_is_partial(self) -> None:
        first = publication_record(
            "pub_first",
            "official-site:example.test",
            status=EvidenceVerificationStatus.VERIFIED,
        )
        second = publication_record(
            "pub_second",
            "OpenAlex",
            title="Conflicting Paper Title",
            publication_date="2025",
            status=EvidenceVerificationStatus.VERIFIED,
        )

        bundle = merge_and_verify_retrieval_results(
            [retrieval_result([first, second])],
            ["retrieval-run-1"],
        )

        self.assertEqual(len(bundle.records), 1)
        self.assertEqual(
            bundle.records[0].evidence.verification_status,
            EvidenceVerificationStatus.PARTIAL,
        )
        self.assertTrue(bundle.records[0].conflicts)

    def test_different_dois_are_not_merged(self) -> None:
        first = publication_record("pub_first", "Official", doi="10.1000/first")
        second = publication_record("pub_second", "OpenAlex", doi="10.1000/second")

        bundle = merge_and_verify_retrieval_results(
            [retrieval_result([first, second])],
            ["retrieval-run-1"],
        )

        self.assertEqual(len(bundle.records), 2)
        self.assertEqual(bundle.duplicate_group_count, 0)


if __name__ == "__main__":
    unittest.main()
