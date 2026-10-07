# File: test_composite.py
# Author: L1nzhk0
# Purpose: 本文件用于验证组合论文 Provider 的官网降级、去重和状态汇总逻辑。

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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
    CompositePublicationSearchProvider,
    PublicationAdapterResult,
    PublicationSearchQuery,
    RawPublicationCandidate,
    SourceDiscoveryResult,
    create_default_publication_search_provider,
)


class FakeAcademicAdapter:
    """返回预设学术索引结果。"""

    # 初始化预设学术索引结果
    def __init__(self, result: PublicationAdapterResult) -> None:
        self._result = result

    # 返回测试学术索引名称
    @property
    def source_name(self) -> str:
        return "academic-test"

    # 返回预设学术索引结果
    def search(self, query: PublicationSearchQuery) -> PublicationAdapterResult:
        return self._result


class FakeDiscoveryAdapter:
    """返回预设官网发现结果并记录调用次数。"""

    # 初始化预设官网发现结果
    def __init__(self, result: SourceDiscoveryResult) -> None:
        self._result = result
        self.call_count = 0

    # 返回预设官网发现结果
    def discover(self, query: PublicationSearchQuery) -> SourceDiscoveryResult:
        self.call_count += 1
        return self._result


class FakeOfficialSiteAdapter:
    """返回预设官网论文结果并记录收到的来源。"""

    # 初始化预设官网论文结果
    def __init__(self, result: PublicationAdapterResult) -> None:
        self._result = result
        self.sources: list[DiscoveredSource] = []

    # 记录官网来源并返回预设论文结果
    def search(
        self,
        query: PublicationSearchQuery,
        sources: list[DiscoveredSource],
    ) -> PublicationAdapterResult:
        self.sources = list(sources)
        return self._result


# 构造组合 Provider 测试使用的规范化查询
def search_query(
    official_urls: list[str] | None = None,
    max_results: int = 20,
) -> PublicationSearchQuery:
    return PublicationSearchQuery.from_arguments(
        {
            "university_name": "Example University",
            "graduate_school_name": "Engineering",
            "professor_name": "Example Professor",
            "search_start_date": "2025-01-01",
            "search_end_date": "2026-08-30",
            "laboratory_name": "Example Laboratory",
            "official_urls": official_urls or [],
            "max_results": max_results,
        }
    )


# 构造组合 Provider 测试使用的原始论文候选
def raw_candidate(
    source_name: str = "academic-test",
    record_id: str = "record-1",
    doi: str = "10.1000/shared",
    publication_date: str = "2026-04-01",
) -> RawPublicationCandidate:
    return RawPublicationCandidate(
        source_record_id=record_id,
        title=f"Paper {record_id}",
        source_name=source_name,
        authors=["Example Professor"],
        affiliations=["Example University"],
        publication_date=publication_date,
        source_url=f"https://example.test/{source_name}/{record_id}",
        identifiers={"doi": doi},
        matched_professor="Example Professor",
        verification_status=EvidenceVerificationStatus.VERIFIED,
    )


# 构造一个通过联网检索发现的研究室官网来源
def discovered_source() -> DiscoveredSource:
    return DiscoveredSource(
        url="https://lab.example.test/",
        source_type=DiscoveredSourceType.LABORATORY_WEBSITE,
        discovery_method=SourceDiscoveryMethod.WEB_SEARCH,
        verification_status=EvidenceVerificationStatus.VERIFIED,
        matched_signals=["大学名称", "教授姓名"],
    )


class CompositePublicationSearchProviderTests(unittest.TestCase):
    def test_default_factory_prefers_serpapi_when_key_is_present(self) -> None:
        with (
            patch(
                "mcp_servers.scholar.composite.SerpApiWebSearchClient"
            ) as serp_client,
            patch("mcp_servers.scholar.composite.BraveWebSearchClient") as brave_client,
        ):
            provider = create_default_publication_search_provider(
                serpapi_api_key="test-serpapi-key",
                brave_api_key="test-brave-key",
            )

        serp_client.assert_called_once_with("test-serpapi-key")
        brave_client.assert_not_called()
        self.assertEqual(
            provider._official_site_discovery_adapter.__class__.__name__,
            "SerpApiOfficialSiteDiscoveryAdapter",
        )

    def test_default_factory_configures_brave_when_key_is_present(self) -> None:
        with patch("mcp_servers.scholar.composite.BraveWebSearchClient") as client_type:
            provider = create_default_publication_search_provider(
                brave_api_key="test-brave-key"
            )

        client_type.assert_called_once_with("test-brave-key")
        self.assertEqual(
            provider._official_site_discovery_adapter.__class__.__name__,
            "BraveOfficialSiteDiscoveryAdapter",
        )

    # 验证官网缺失且发现能力未配置时仍保留学术索引结果并标记降级
    def test_missing_official_site_degrades_without_failing(self) -> None:
        provider = CompositePublicationSearchProvider(
            FakeAcademicAdapter(
                PublicationAdapterResult(
                    status=RetrievalStatus.SUCCESS,
                    candidates=[raw_candidate()],
                )
            )
        )

        result = provider.search(search_query())

        self.assertEqual(result.status, RetrievalStatus.PARTIAL)
        self.assertEqual(len(result.records), 1)
        self.assertIn("SERPAPI_API_KEY", result.warnings[0])

    # 验证用户提供官网时跳过发现步骤并把来源交给官网 Adapter
    def test_user_provided_url_skips_discovery(self) -> None:
        discovery = FakeDiscoveryAdapter(
            SourceDiscoveryResult(
                status=RetrievalStatus.FAILED,
                errors=["should not run"],
            )
        )
        official = FakeOfficialSiteAdapter(
            PublicationAdapterResult(
                status=RetrievalStatus.NO_RESULT,
            )
        )
        provider = CompositePublicationSearchProvider(
            FakeAcademicAdapter(
                PublicationAdapterResult(
                    status=RetrievalStatus.SUCCESS,
                    candidates=[raw_candidate()],
                )
            ),
            discovery,
            official,
        )

        result = provider.search(search_query(["https://lab.example.test/"]))

        self.assertEqual(discovery.call_count, 0)
        self.assertEqual(len(official.sources), 1)
        self.assertEqual(
            result.discovered_sources[0].discovery_method,
            SourceDiscoveryMethod.USER_PROVIDED,
        )

    # 验证 OpenAlex 作者歧义不会阻止读取用户已经提供的研究室官网
    def test_academic_ambiguity_does_not_block_user_official_site(self) -> None:
        official = FakeOfficialSiteAdapter(
            PublicationAdapterResult(
                status=RetrievalStatus.SUCCESS,
                candidates=[raw_candidate("official-site", "official")],
            )
        )
        provider = CompositePublicationSearchProvider(
            FakeAcademicAdapter(
                PublicationAdapterResult(
                    status=RetrievalStatus.NEEDS_USER_CONFIRMATION,
                    warnings=["OpenAlex 作者档案存在歧义"],
                )
            ),
            FakeDiscoveryAdapter(
                SourceDiscoveryResult(
                    status=RetrievalStatus.FAILED,
                    errors=["用户提供官网时不应执行发现"],
                )
            ),
            official,
        )

        result = provider.search(
            search_query(["https://lab.example.test/publications/"])
        )

        self.assertEqual(len(official.sources), 1)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.status, RetrievalStatus.PARTIAL)
        self.assertIn("OpenAlex 作者档案存在歧义", result.warnings)

    def test_verified_official_source_with_partial_papers_resolves_ambiguity(
        self,
    ) -> None:
        candidate = raw_candidate("official-site:lab.example.test", "official")
        candidate.affiliations = []
        candidate.verification_status = EvidenceVerificationStatus.PARTIAL
        official = FakeOfficialSiteAdapter(
            PublicationAdapterResult(
                status=RetrievalStatus.PARTIAL,
                candidates=[candidate],
                source_updates=[discovered_source()],
            )
        )
        provider = CompositePublicationSearchProvider(
            FakeAcademicAdapter(
                PublicationAdapterResult(status=RetrievalStatus.NEEDS_USER_CONFIRMATION)
            ),
            official_site_publication_adapter=official,
        )

        result = provider.search(search_query(["https://lab.example.test/"]))

        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.status, RetrievalStatus.PARTIAL)

    # 验证官网读取阶段能够更新本次返回的来源核验状态
    def test_applies_official_site_source_updates(self) -> None:
        verified_source = discovered_source()
        official = FakeOfficialSiteAdapter(
            PublicationAdapterResult(
                status=RetrievalStatus.NO_RESULT,
                source_updates=[verified_source],
            )
        )
        provider = CompositePublicationSearchProvider(
            FakeAcademicAdapter(
                PublicationAdapterResult(
                    status=RetrievalStatus.NO_RESULT,
                )
            ),
            FakeDiscoveryAdapter(
                SourceDiscoveryResult(
                    status=RetrievalStatus.SUCCESS,
                    sources=[
                        DiscoveredSource(
                            url=verified_source.url,
                            source_type=verified_source.source_type,
                            discovery_method=verified_source.discovery_method,
                            verification_status=(EvidenceVerificationStatus.PARTIAL),
                        )
                    ],
                )
            ),
            official,
        )

        result = provider.search(search_query())

        self.assertEqual(
            result.discovered_sources[0].verification_status,
            EvidenceVerificationStatus.VERIFIED,
        )

    # 验证官网候选存在歧义时停止官网抓取并请求用户确认
    def test_ambiguous_discovery_requires_confirmation(self) -> None:
        discovery = FakeDiscoveryAdapter(
            SourceDiscoveryResult(
                status=RetrievalStatus.NEEDS_USER_CONFIRMATION,
                sources=[discovered_source()],
                warnings=["存在多个官网候选"],
            )
        )
        official = FakeOfficialSiteAdapter(
            PublicationAdapterResult(
                status=RetrievalStatus.SUCCESS,
                candidates=[raw_candidate("official-site")],
            )
        )
        provider = CompositePublicationSearchProvider(
            FakeAcademicAdapter(
                PublicationAdapterResult(
                    status=RetrievalStatus.NO_RESULT,
                )
            ),
            discovery,
            official,
        )

        result = provider.search(search_query())

        self.assertEqual(
            result.status,
            RetrievalStatus.NEEDS_USER_CONFIRMATION,
        )
        self.assertEqual(official.sources, [])

    # 验证相同来源精确重复会合并而跨来源相同 DOI 会保留
    def test_deduplicates_only_within_the_same_source(self) -> None:
        academic = PublicationAdapterResult(
            status=RetrievalStatus.SUCCESS,
            candidates=[
                raw_candidate(record_id="first"),
                raw_candidate(record_id="duplicate"),
            ],
        )
        official = FakeOfficialSiteAdapter(
            PublicationAdapterResult(
                status=RetrievalStatus.SUCCESS,
                candidates=[raw_candidate("official-site", "official")],
            )
        )
        provider = CompositePublicationSearchProvider(
            FakeAcademicAdapter(academic),
            FakeDiscoveryAdapter(
                SourceDiscoveryResult(
                    status=RetrievalStatus.SUCCESS,
                    sources=[discovered_source()],
                )
            ),
            official,
        )

        result = provider.search(search_query())

        self.assertEqual(result.status, RetrievalStatus.SUCCESS)
        self.assertEqual(len(result.records), 2)
        self.assertEqual(
            {record.source_name for record in result.records},
            {"academic-test", "official-site"},
        )

    # 验证日期无法落入检索范围的候选不会成为正式证据
    def test_filters_candidates_outside_date_range(self) -> None:
        provider = CompositePublicationSearchProvider(
            FakeAcademicAdapter(
                PublicationAdapterResult(
                    status=RetrievalStatus.SUCCESS,
                    candidates=[raw_candidate(publication_date="2020")],
                )
            ),
            FakeDiscoveryAdapter(
                SourceDiscoveryResult(
                    status=RetrievalStatus.NO_RESULT,
                )
            ),
        )

        result = provider.search(search_query())

        self.assertEqual(result.status, RetrievalStatus.NO_RESULT)
        self.assertEqual(result.records, [])
        self.assertIn("超出范围", result.warnings[0])

    # 验证最终数量受限时优先保留经过官网核验的记录并明确提示截断
    def test_prioritizes_official_records_before_final_limit(self) -> None:
        academic = PublicationAdapterResult(
            status=RetrievalStatus.SUCCESS,
            candidates=[
                raw_candidate(record_id="academic-1", doi="10.1000/a1"),
                raw_candidate(record_id="academic-2", doi="10.1000/a2"),
            ],
        )
        official = FakeOfficialSiteAdapter(
            PublicationAdapterResult(
                status=RetrievalStatus.SUCCESS,
                candidates=[
                    raw_candidate(
                        "official-site:lab.example.test",
                        "official-1",
                        "10.1000/o1",
                    ),
                    raw_candidate(
                        "official-site:lab.example.test",
                        "official-2",
                        "10.1000/o2",
                    ),
                ],
            )
        )
        provider = CompositePublicationSearchProvider(
            FakeAcademicAdapter(academic),
            FakeDiscoveryAdapter(
                SourceDiscoveryResult(
                    status=RetrievalStatus.SUCCESS,
                    sources=[discovered_source()],
                )
            ),
            official,
        )

        result = provider.search(search_query(max_results=2))

        self.assertEqual(len(result.records), 2)
        self.assertTrue(
            all(
                record.source_name.startswith("official-site")
                for record in result.records
            )
        )
        self.assertIn("不代表完整论文清单", result.warnings[-1])

    # 验证官网可用时剔除未获官网标题佐证的部分核验索引记录
    def test_drops_uncorroborated_partial_index_records(self) -> None:
        uncertain = raw_candidate(
            source_name="OpenAlex",
            record_id="uncertain",
            doi="10.1000/uncertain",
        )
        uncertain.affiliations = []
        uncertain.verification_status = EvidenceVerificationStatus.PARTIAL
        official = FakeOfficialSiteAdapter(
            PublicationAdapterResult(
                status=RetrievalStatus.SUCCESS,
                candidates=[
                    raw_candidate(
                        "official-site:lab.example.test",
                        "official",
                        "10.1000/official",
                    )
                ],
            )
        )
        provider = CompositePublicationSearchProvider(
            FakeAcademicAdapter(
                PublicationAdapterResult(
                    status=RetrievalStatus.PARTIAL,
                    candidates=[uncertain],
                )
            ),
            FakeDiscoveryAdapter(
                SourceDiscoveryResult(
                    status=RetrievalStatus.SUCCESS,
                    sources=[discovered_source()],
                )
            ),
            official,
        )

        result = provider.search(search_query())

        self.assertEqual(len(result.records), 1)
        self.assertTrue(result.records[0].source_name.startswith("official-site"))
        self.assertIn("未获官网标题佐证", result.warnings[-1])

    def test_drops_partial_index_without_official_site(self) -> None:
        uncertain = raw_candidate(source_name="OpenAlex", record_id="same-name")
        uncertain.affiliations = []
        uncertain.verification_status = EvidenceVerificationStatus.PARTIAL
        provider = CompositePublicationSearchProvider(
            FakeAcademicAdapter(
                PublicationAdapterResult(
                    status=RetrievalStatus.PARTIAL,
                    candidates=[uncertain],
                )
            )
        )

        result = provider.search(search_query())

        self.assertEqual(result.records, [])
        self.assertIn("未获官网标题佐证", result.warnings[-1])


if __name__ == "__main__":
    unittest.main()
