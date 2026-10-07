"""Verify SerpApi transport, result parsing, and official-site selection."""

from __future__ import annotations

import json
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.domain import RetrievalStatus  # noqa: E402
from mcp_servers.scholar import (  # noqa: E402
    PublicationSearchQuery,
    SerpApiOfficialSiteDiscoveryAdapter,
    SerpApiSearchError,
    SerpApiWebSearchClient,
)


class _Response(BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _query() -> PublicationSearchQuery:
    return PublicationSearchQuery.from_arguments(
        {
            "university_name": "早稲田大学",
            "graduate_school_name": "基幹理工学研究科",
            "professor_name": "酒井哲也",
            "search_start_date": "2025-01-01",
            "search_end_date": "2026-08-30",
        }
    )


class SerpApiDiscoveryTests(unittest.TestCase):
    def test_google_search_uses_japanese_locale_and_retains_links(self) -> None:
        payload = {
            "organic_results": [
                {
                    "title": "酒井哲也 研究室",
                    "link": "https://www.waseda.jp/lab/sakai/",
                    "snippet": "早稲田大学 基幹理工学研究科",
                },
                {
                    "title": "No URL",
                    "snippet": "ignore",
                },
            ]
        }
        with patch(
            "mcp_servers.scholar.serpapi_discovery.urlopen",
            return_value=_Response(json.dumps(payload).encode()),
        ) as urlopen_mock:
            result = SerpApiOfficialSiteDiscoveryAdapter(
                SerpApiWebSearchClient("secret-key")
            ).discover(_query())

        params = parse_qs(urlsplit(urlopen_mock.call_args.args[0].full_url).query)
        self.assertEqual(params["engine"], ["google"])
        self.assertEqual(params["gl"], ["jp"])
        self.assertEqual(params["hl"], ["ja"])
        self.assertEqual(params["api_key"], ["secret-key"])
        self.assertEqual(result.status, RetrievalStatus.NEEDS_USER_CONFIRMATION)
        self.assertEqual(result.sources[0].url, "https://www.waseda.jp/lab/sakai/")

    def test_api_error_never_echoes_secret(self) -> None:
        with patch(
            "mcp_servers.scholar.serpapi_discovery.urlopen",
            side_effect=HTTPError(
                "https://serpapi.com/search.json?api_key=secret-key",
                401,
                "unauthorized",
                None,
                None,
            ),
        ):
            result = SerpApiOfficialSiteDiscoveryAdapter(
                SerpApiWebSearchClient("secret-key")
            ).discover(_query())

        self.assertEqual(result.status, RetrievalStatus.FAILED)
        self.assertIn("HTTP 401", result.errors[0])
        self.assertNotIn("secret-key", str(result.errors))

    def test_lab_surname_match_is_offered_only_for_user_confirmation(self) -> None:
        payload = {
            "organic_results": [
                {
                    "title": "リンク集 - 後藤研究室 - 早稲田大学",
                    "link": "https://www.waseda.ac.jp/links/",
                    "snippet": "酒井哲也 早稲田大学 酒井研究室",
                },
                {
                    "title": "SIGIR 2025 - 酒井研究室",
                    "link": "https://sakailab.com/sigir2025/",
                    "snippet": "早稲田大学 基幹理工学研究科 酒井研究室",
                },
            ]
        }
        with patch(
            "mcp_servers.scholar.serpapi_discovery.urlopen",
            return_value=_Response(json.dumps(payload).encode()),
        ):
            result = SerpApiOfficialSiteDiscoveryAdapter(
                SerpApiWebSearchClient("secret-key")
            ).discover(_query())

        self.assertEqual(result.status, RetrievalStatus.NEEDS_USER_CONFIRMATION)
        self.assertEqual(result.sources[0].url, "https://sakailab.com/")
        self.assertIn("教授姓氏", result.sources[0].matched_signals)

    def test_simplified_university_spelling_finds_japanese_site(self) -> None:
        payload = {
            "organic_results": [
                {
                    "title": "酒井研究室",
                    "link": "https://sakailab.com/sigir2025/",
                    "snippet": "早稲田大学 基幹理工学研究科",
                }
            ]
        }
        query = PublicationSearchQuery.from_arguments(
            {
                "university_name": "早稻田大学",
                "graduate_school_name": "基干研究科",
                "professor_name": "酒井哲也",
                "laboratory_name": "情报理工",
                "search_start_date": "2025-01-01",
                "search_end_date": "2026-08-30",
            }
        )
        with patch(
            "mcp_servers.scholar.serpapi_discovery.urlopen",
            return_value=_Response(json.dumps(payload).encode()),
        ) as urlopen_mock:
            result = SerpApiOfficialSiteDiscoveryAdapter(
                SerpApiWebSearchClient("secret-key")
            ).discover(query)

        params = parse_qs(urlsplit(urlopen_mock.call_args.args[0].full_url).query)
        self.assertIn("早稲田大学", params["q"][0])
        self.assertEqual(result.status, RetrievalStatus.NEEDS_USER_CONFIRMATION)
        self.assertEqual(result.sources[0].url, "https://sakailab.com/")

    def test_rejects_invalid_response_and_serpapi_error(self) -> None:
        for payload in ({"organic_results": {}}, {"error": "secret-key"}):
            with self.subTest(payload=payload):
                with patch(
                    "mcp_servers.scholar.serpapi_discovery.urlopen",
                    return_value=_Response(json.dumps(payload).encode()),
                ):
                    with self.assertRaises(SerpApiSearchError) as raised:
                        SerpApiWebSearchClient("secret-key").search("query")
                self.assertNotIn("secret-key", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
