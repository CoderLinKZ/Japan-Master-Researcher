"""Rolling one-year publication choice boundary."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from jmr.graph.nodes.directions import _paper_options


class RecentPaperChoiceTests(unittest.TestCase):
    def test_only_papers_within_rolling_year_are_offered(self) -> None:
        dates = ["2025-09-26", "2025-09-27", "2026-06-01", "2026-09-28", "2025", "2026"]
        records = [
            {
                "evidence": {
                    "evidence_id": f"paper-{index}",
                    "evidence_type": "publication",
                    "title": f"Paper {index}",
                    "publication_date": published,
                }
            }
            for index, published in enumerate(dates)
        ]
        bundle = {
            "evidence_ids": [f"paper-{index}" for index in range(len(records))],
            "summary": {"bundle": {"records": records}},
        }

        options = _paper_options(bundle, reference_date=date(2026, 9, 27))

        self.assertEqual(
            {item["evidence_id"] for item in options},
            {"paper-1", "paper-2", "paper-5"},
        )


if __name__ == "__main__":
    unittest.main()
