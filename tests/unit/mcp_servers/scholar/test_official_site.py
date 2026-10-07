# File: test_official_site.py
# Author: L1nzhk0
# Purpose: 本文件用于验证研究室官网同域抓取、身份核验和论文候选提取。

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.domain import (  # noqa: E402
    DiscoveredSource,
    DiscoveredSourceType,
    EvidenceVerificationStatus,
    RetrievalStatus,
    SourceDiscoveryMethod,
)
from mcp_servers.scholar import (  # noqa: E402
    CrawlingOfficialSitePublicationAdapter,
    HTTPTextResponse,
    PublicationSearchQuery,
    SafeHTTPBlockedError,
    SafeHTTPError,
    parse_official_page,
)


class FakeSiteHTTPClient:
    """按 URL 返回固定官网响应或错误。"""

    # 初始化官网响应映射
    def __init__(
        self,
        responses: dict[str, HTTPTextResponse | SafeHTTPError],
    ) -> None:
        self._responses = responses
        self.calls: list[str] = []

    # 记录 URL 并返回固定响应或抛出错误
    def fetch(self, url: str) -> HTTPTextResponse:
        self.calls.append(url)
        response = self._responses.get(url)
        if response is None:
            raise SafeHTTPError(f"missing fake response: {url}")
        if isinstance(response, SafeHTTPError):
            raise response
        return response


# 构造官网论文检索测试使用的目标查询
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


# 构造用户提供的研究室官网来源
def official_source() -> DiscoveredSource:
    return DiscoveredSource(
        url="https://lab.example/",
        source_type=DiscoveredSourceType.LABORATORY_WEBSITE,
        discovery_method=SourceDiscoveryMethod.USER_PROVIDED,
        verification_status=EvidenceVerificationStatus.UNVERIFIED,
        matched_signals=["用户提供"],
    )


# 构造 Fake Client 使用的 HTML 响应
def html_response(url: str, html: str) -> HTTPTextResponse:
    return HTTPTextResponse(
        url=url,
        status=200,
        content_type="text/html",
        text=html,
    )


class OfficialSiteParserTests(unittest.TestCase):
    # 验证 HTML Parser 能够解析绝对链接、citation meta 和 JSON-LD
    def test_parses_links_metadata_and_json_ld(self) -> None:
        page = parse_official_page(
            """
            <html><head>
              <title>Example Paper</title>
              <meta name="citation_title" content="Example Paper">
              <script type="application/ld+json">
                {"@type":"ScholarlyArticle","headline":"Example Paper"}
              </script>
            </head><body>
              <a href="/publications">Publications</a>
            </body></html>
            """,
            "https://lab.example/",
        )

        self.assertEqual(page.title, "Example Paper")
        self.assertEqual(
            page.links[0].url,
            "https://lab.example/publications",
        )
        self.assertEqual(
            page.metadata["citation_title"],
            ("Example Paper",),
        )
        self.assertEqual(page.json_ld[0]["@type"], "ScholarlyArticle")


class CrawlingOfficialSitePublicationAdapterTests(unittest.TestCase):
    # 验证 Adapter 会遵守 robots 并从同域论文列表提取目标教授论文
    def test_crawls_publication_page_and_extracts_candidate(self) -> None:
        client = FakeSiteHTTPClient(
            {
                "https://lab.example/robots.txt": html_response(
                    "https://lab.example/robots.txt",
                    "User-agent: *\nAllow: /",
                ),
                "https://lab.example/": html_response(
                    "https://lab.example/",
                    """
                <html><head><title>Example Laboratory</title></head><body>
                  <h1>Example University Example Laboratory</h1>
                  <p>Director: Example Professor</p>
                  <a href="/publications">Publications</a>
                </body></html>
                """,
                ),
                "https://lab.example/publications": html_response(
                    "https://lab.example/publications",
                    """
                <html><head><title>Publications</title></head><body>
                  <h2>Publications</h2>
                  <ul><li>
                    Example Professor and Coauthor, 2026.
                    <a href="https://doi.org/10.1000/example">
                      A Reliable Publication Title
                    </a>
                  </li></ul>
                </body></html>
                """,
                ),
            }
        )
        adapter = CrawlingOfficialSitePublicationAdapter(client)

        result = adapter.search(search_query(), [official_source()])

        self.assertEqual(result.status, RetrievalStatus.SUCCESS)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            result.candidates[0].title,
            "A Reliable Publication Title",
        )
        self.assertEqual(
            result.candidates[0].identifiers["doi"],
            "10.1000/example",
        )
        self.assertEqual(
            result.source_updates[0].verification_status,
            EvidenceVerificationStatus.VERIFIED,
        )

    def test_homepage_failure_tries_same_site_publications_page(self) -> None:
        client = FakeSiteHTTPClient(
            {
                "https://lab.example/robots.txt": html_response(
                    "https://lab.example/robots.txt", "User-agent: *\nAllow: /"
                ),
                "https://lab.example/": SafeHTTPError("homepage unavailable"),
                "https://lab.example/publications/": html_response(
                    "https://lab.example/publications/",
                    """
                    <html><body><h1>Example University Example Laboratory</h1>
                    <h2>Publications</h2><li>
                    Example Professor and Coauthor, 2026.
                    <a href="https://doi.org/10.1000/fallback">Fallback Paper</a>
                    </li></body></html>
                    """,
                ),
            }
        )

        result = CrawlingOfficialSitePublicationAdapter(client).search(
            search_query(), [official_source()]
        )

        self.assertEqual(len(result.candidates), 1)
        self.assertIn("https://lab.example/publications/", client.calls)
        self.assertIn("已尝试同站 /publications/", result.warnings[0])

    # 验证 citation meta 可以直接生成完整论文候选
    def test_extracts_citation_metadata(self) -> None:
        client = FakeSiteHTTPClient(
            {
                "https://lab.example/robots.txt": html_response(
                    "https://lab.example/robots.txt",
                    "User-agent: *\nAllow: /",
                ),
                "https://lab.example/": html_response(
                    "https://lab.example/",
                    """
                <html><head>
                  <title>Metadata Paper</title>
                  <meta name="citation_title" content="Metadata Paper">
                  <meta name="citation_author" content="Example Professor">
                  <meta name="citation_publication_date" content="2026/05/01">
                  <meta name="citation_doi" content="10.1000/meta">
                  <meta name="citation_journal_title" content="Journal A">
                </head><body>
                  Example University Example Laboratory Example Professor
                </body></html>
                """,
                ),
            }
        )
        adapter = CrawlingOfficialSitePublicationAdapter(client)

        result = adapter.search(search_query(), [official_source()])

        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].publication_date, "2026-05-01")
        self.assertEqual(result.candidates[0].venue, "Journal A")

    # 验证没有标题链接的常见作者冒号引文仍能保守提取论文
    def test_extracts_plain_text_citation(self) -> None:
        client = FakeSiteHTTPClient(
            {
                "https://lab.example/robots.txt": html_response(
                    "https://lab.example/robots.txt",
                    "User-agent: *\nAllow: /",
                ),
                "https://lab.example/": html_response(
                    "https://lab.example/",
                    """
                <html><head><title>Publications</title></head><body>
                  Example University Example Laboratory
                  <h2>Publications</h2>
                  <p>
                    Coauthor and Example Professor:
                    A Plain Text Research Paper Title,
                    ACM CIKM 2026, 2026.
                  </p>
                </body></html>
                """,
                ),
            }
        )
        adapter = CrawlingOfficialSitePublicationAdapter(client)

        result = adapter.search(search_query(), [official_source()])

        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            result.candidates[0].title,
            "A Plain Text Research Paper Title",
        )
        self.assertEqual(result.candidates[0].venue, "ACM CIKM 2026")
        self.assertEqual(
            result.candidates[0].authors,
            ["Coauthor", "Example Professor"],
        )
        self.assertEqual(result.candidates[0].matched_professor_position, 2)
        self.assertIn(
            "A Plain Text Research Paper Title",
            result.candidates[0].source_citation,
        )

    # 验证 DEIM 引文保留届次、发表编号和官网列出的作者顺序
    def test_preserves_deim_venue_code_and_author_order(self) -> None:
        client = FakeSiteHTTPClient(
            {
                "https://lab.example/robots.txt": html_response(
                    "https://lab.example/robots.txt",
                    "User-agent: *\nAllow: /",
                ),
                "https://lab.example/": html_response(
                    "https://lab.example/",
                    """
                <html><head><title>Publications</title></head><body>
                  Example University Example Laboratory
                  <h2>Publications</h2>
                  <p>
                    First Student, Example Professor:
                    A Structured Retrieval Research Paper,
                    DEIM 2026, 2026.
                    <a href="https://pub.confit.atlas.jp/ja/event/deim2026/presentation/5A-03">pdf</a>
                  </p>
                </body></html>
                """,
                ),
            }
        )
        adapter = CrawlingOfficialSitePublicationAdapter(client)

        result = adapter.search(search_query(), [official_source()])

        self.assertEqual(result.candidates[0].venue, "DEIM 2026, 5A-03")
        self.assertEqual(
            result.candidates[0].authors,
            ["First Student", "Example Professor"],
        )
        self.assertEqual(result.candidates[0].matched_professor_position, 2)

    # 验证同一论文列表页面上的无链接引文不会因共享页面 URL 被错误去重
    def test_keeps_multiple_plain_text_citations_from_same_page(self) -> None:
        client = FakeSiteHTTPClient(
            {
                "https://lab.example/robots.txt": html_response(
                    "https://lab.example/robots.txt",
                    "User-agent: *\nAllow: /",
                ),
                "https://lab.example/": html_response(
                    "https://lab.example/",
                    """
                <html><head><title>Publications</title></head><body>
                  Example University Example Laboratory
                  <h2>Publications</h2>
                  <p>
                    Example Professor: First Reliable Research Paper,
                    ACM CIKM 2026, 2026.
                  </p>
                  <p>
                    Example Professor: Second Reliable Research Paper,
                    ACM WSDM 2026, 2026.
                  </p>
                </body></html>
                """,
                ),
            }
        )
        adapter = CrawlingOfficialSitePublicationAdapter(client)

        result = adapter.search(search_query(), [official_source()])

        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(
            {candidate.title for candidate in result.candidates},
            {
                "First Reliable Research Paper",
                "Second Reliable Research Paper",
            },
        )

    # 验证 robots.txt 禁止的页面不会被访问或提取
    def test_respects_robots_txt_disallow(self) -> None:
        client = FakeSiteHTTPClient(
            {
                "https://lab.example/robots.txt": html_response(
                    "https://lab.example/robots.txt",
                    "User-agent: *\nDisallow: /",
                ),
            }
        )
        adapter = CrawlingOfficialSitePublicationAdapter(client)

        result = adapter.search(search_query(), [official_source()])

        self.assertEqual(result.status, RetrievalStatus.BLOCKED)
        self.assertEqual(result.candidates, [])
        self.assertEqual(
            client.calls,
            ["https://lab.example/robots.txt"],
        )

    # 验证安全 Client 阻止官网时返回 blocked 而不是伪装成无结果
    def test_returns_blocked_when_safe_http_rejects_source(self) -> None:
        client = FakeSiteHTTPClient(
            {
                "https://lab.example/robots.txt": SafeHTTPBlockedError(
                    "官网主机解析到非公网 IP"
                ),
                "https://lab.example/": SafeHTTPBlockedError("官网主机解析到非公网 IP"),
            }
        )
        adapter = CrawlingOfficialSitePublicationAdapter(client)

        result = adapter.search(search_query(), [official_source()])

        self.assertEqual(result.status, RetrievalStatus.BLOCKED)
        self.assertEqual(result.candidates, [])


if __name__ == "__main__":
    unittest.main()
