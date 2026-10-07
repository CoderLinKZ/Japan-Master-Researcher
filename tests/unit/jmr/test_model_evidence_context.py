"""Large evidence bundles must not exceed a single model-message budget."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.graph.nodes.common import (  # noqa: E402
    compact_evidence_for_model,
    compact_messages_for_model,
)


class ModelEvidenceContextTests(unittest.TestCase):
    def test_large_bundle_retains_citable_priority_records_within_limit(self) -> None:
        records = [
            {
                "evidence": {
                    "evidence_id": f"paper-{index}",
                    "title": f"Paper {index}",
                    "source_name": "OpenAlex",
                    "source_url": f"https://example.org/{index}",
                    "verification_status": "VERIFIED" if index < 25 else "PARTIAL",
                    "abstract_or_summary": "学术摘要" * 2500,
                    "source_citation": "引用正文" * 1500,
                }
            }
            for index in range(50)
        ]
        bundle = {
            "evidence_ids": [f"paper-{index}" for index in range(50)],
            "summary": {
                "bundle": {"records": records},
                "verified_count": 25,
                "partial_count": 25,
            },
        }

        projection = compact_evidence_for_model(
            bundle, priority_ids=["paper-42", "paper-43"]
        )
        self.assertEqual(
            projection["allowed_evidence_ids"][:2], ["paper-42", "paper-43"]
        )
        self.assertEqual(len(projection["records"]), 18)
        self.assertEqual(projection["total_evidence_count"], 50)
        compact_messages_for_model(
            [{"role": "user", "content": json.dumps(projection, ensure_ascii=False)}]
        )


if __name__ == "__main__":
    unittest.main()
