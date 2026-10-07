"""Keep simplified user spellings compatible with Japanese institution text."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(SRC_ROOT))

from mcp_servers.scholar import (  # noqa: E402
    PublicationSearchQuery,
    RawPublicationCandidate,
)
from mcp_servers.scholar.composite import (  # noqa: E402
    _candidate_matches_institution,
)
from mcp_servers.scholar.japanese_text import (  # noqa: E402
    normalize_search_text,
    to_japanese_spelling,
)
from mcp_servers.scholar.official_site import _contains_text  # noqa: E402
from mcp_servers.scholar.openalex import _contains_text_match  # noqa: E402


class JapaneseTextTests(unittest.TestCase):
    def test_school_variant_matches_across_scholar_adapters(self) -> None:
        query = PublicationSearchQuery.from_arguments(
            {
                "university_name": "早稻田大学",
                "graduate_school_name": "基干研究科",
                "professor_name": "酒井哲也",
                "search_start_date": "2025-01-01",
                "search_end_date": "2026-09-26",
            }
        )
        candidate = RawPublicationCandidate(
            source_record_id="paper-1",
            title="研究论文",
            source_name="official-site",
            affiliations=["早稲田大学 基幹理工学研究科"],
        )

        self.assertEqual(to_japanese_spelling("早稻田大学"), "早稲田大学")
        self.assertEqual(
            normalize_search_text("早稻田大学"),
            normalize_search_text("早稲田大学"),
        )
        self.assertTrue(_candidate_matches_institution(candidate, query))
        self.assertTrue(_contains_text("早稲田大学 酒井研究室", "早稻田大学"))
        self.assertTrue(_contains_text_match(["早稲田大学"], "早稻田大学"))


if __name__ == "__main__":
    unittest.main()
