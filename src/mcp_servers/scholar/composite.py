# File: composite.py
# Author: L1nzhk0
# Purpose: 本文件用于编排学术索引、官网发现和官网论文来源并生成统一研究证据。

from __future__ import annotations

import hashlib
import os
from datetime import UTC, date, datetime

from jmr.domain import (
    DiscoveredSource,
    DiscoveredSourceType,
    EvidenceType,
    EvidenceVerificationStatus,
    ResearchEvidence,
    RetrievalStatus,
    SourceDiscoveryMethod,
)

from .adapters import (
    AcademicPublicationAdapter,
    OfficialSiteDiscoveryAdapter,
    OfficialSitePublicationAdapter,
    UnavailableOfficialSiteDiscoveryAdapter,
    UnavailableOfficialSitePublicationAdapter,
)
from .brave_discovery import BraveOfficialSiteDiscoveryAdapter, BraveWebSearchClient
from .japanese_text import normalize_search_text
from .models import (
    PublicationAdapterResult,
    RawPublicationCandidate,
    SourceDiscoveryResult,
)
from .official_site import CrawlingOfficialSitePublicationAdapter
from .openalex import OpenAlexPublicationAdapter
from .provider import (
    PublicationSearchProviderResult,
    PublicationSearchQuery,
)
from .serpapi_discovery import (
    SerpApiOfficialSiteDiscoveryAdapter,
    SerpApiWebSearchClient,
)


class CompositePublicationSearchProvider:
    """组合多个受控 Adapter, 并统一执行过滤、去重和状态汇总。"""

    # 初始化组合 Provider 及其三个检索边界
    def __init__(
        self,
        academic_adapter: AcademicPublicationAdapter,
        official_site_discovery_adapter: (OfficialSiteDiscoveryAdapter | None) = None,
        official_site_publication_adapter: (
            OfficialSitePublicationAdapter | None
        ) = None,
    ) -> None:
        self._academic_adapter = academic_adapter
        self._official_site_discovery_adapter = (
            official_site_discovery_adapter or UnavailableOfficialSiteDiscoveryAdapter()
        )
        self._official_site_publication_adapter = (
            official_site_publication_adapter
            or UnavailableOfficialSitePublicationAdapter()
        )

    # 返回组合 Provider 的稳定来源名称
    @property
    def source_name(self) -> str:
        return f"composite:{self._academic_adapter.source_name}+official-site"

    # 按学术索引优先、官网可选的策略完成一次论文检索
    def search(
        self,
        query: PublicationSearchQuery,
    ) -> PublicationSearchProviderResult:
        adapter_results: list[PublicationAdapterResult] = []
        statuses: list[RetrievalStatus] = []
        warnings: list[str] = []
        errors: list[str] = []
        discovery_status: RetrievalStatus | None = None

        academic_result = self._academic_adapter.search(query)
        adapter_results.append(academic_result)
        statuses.append(academic_result.status)
        warnings.extend(academic_result.warnings)
        errors.extend(academic_result.errors)

        discovered_sources: list[DiscoveredSource]
        if query.official_sources:
            discovered_sources = _deduplicate_sources(query.official_sources)
        elif query.official_urls:
            discovered_sources = _build_user_provided_sources(query)
        else:
            discovery_result = self._official_site_discovery_adapter.discover(query)
            discovered_sources = _deduplicate_sources(discovery_result.sources)
            statuses.append(discovery_result.status)
            discovery_status = discovery_result.status
            warnings.extend(discovery_result.warnings)
            errors.extend(discovery_result.errors)

        if discovered_sources and (
            bool(query.official_sources or query.official_urls)
            or discovery_status != RetrievalStatus.NEEDS_USER_CONFIRMATION
        ):
            official_result = self._official_site_publication_adapter.search(
                query,
                discovered_sources,
            )
            adapter_results.append(official_result)
            statuses.append(official_result.status)
            warnings.extend(official_result.warnings)
            errors.extend(official_result.errors)
            discovered_sources = _merge_source_updates(
                discovered_sources,
                official_result.source_updates,
            )

        raw_candidates = [
            candidate for result in adapter_results for candidate in result.candidates
        ]
        records, normalization_warnings = _normalize_candidates(
            raw_candidates,
            query,
        )
        warnings.extend(normalization_warnings)
        records, corroboration_warnings = _filter_uncorroborated_partial_index_records(
            records
        )
        warnings.extend(corroboration_warnings)
        records = sorted(records, key=_record_priority_key)
        available_record_count = len(records)
        records = records[: query.max_results]
        if available_record_count > query.max_results:
            warnings.append(
                "多个来源共形成 "
                f"{available_record_count} 条规范化论文记录，当前优先返回 "
                f"{query.max_results} 条官网及高核验证据；"
                "该结果不代表完整论文清单"
            )
        has_verified_official_record = any(
            record.source_name.casefold().startswith("official-site")
            and record.verification_status == EvidenceVerificationStatus.VERIFIED
            for record in records
        )
        has_verified_official_source_with_records = any(
            source.verification_status == EvidenceVerificationStatus.VERIFIED
            for source in discovered_sources
        ) and any(
            record.source_name.casefold().startswith("official-site")
            and record.verification_status
            in {
                EvidenceVerificationStatus.VERIFIED,
                EvidenceVerificationStatus.PARTIAL,
            }
            for record in records
        )
        confirmation_required = (
            discovery_status == RetrievalStatus.NEEDS_USER_CONFIRMATION
            or (
                academic_result.status == RetrievalStatus.NEEDS_USER_CONFIRMATION
                and not (
                    has_verified_official_record
                    or has_verified_official_source_with_records
                )
            )
        )
        status = _combine_statuses(
            statuses,
            records,
            confirmation_required=confirmation_required,
        )
        return PublicationSearchProviderResult(
            status=status,
            records=records,
            discovered_sources=discovered_sources,
            warnings=_deduplicate_strings(warnings),
            errors=_deduplicate_strings(errors),
        )

    def discover_official_sources(
        self,
        query: PublicationSearchQuery,
    ) -> SourceDiscoveryResult:
        """Expose official-source discovery without running publication search."""

        if query.official_sources:
            return SourceDiscoveryResult(
                status=RetrievalStatus.SUCCESS,
                sources=_deduplicate_sources(query.official_sources),
            )
        if query.official_urls:
            return SourceDiscoveryResult(
                status=RetrievalStatus.SUCCESS,
                sources=_build_user_provided_sources(query),
            )
        return self._official_site_discovery_adapter.discover(query)


# 创建以 OpenAlex 为主数据源并允许官网能力后续注入的默认组合 Provider
def create_default_publication_search_provider(
    openalex_api_key: str | None = None,
    brave_api_key: str | None = None,
    serpapi_api_key: str | None = None,
) -> CompositePublicationSearchProvider:
    serpapi_key = serpapi_api_key or (
        None if brave_api_key else os.getenv("SERPAPI_API_KEY")
    )
    brave_key = brave_api_key or os.getenv("BRAVE_API_KEY")
    if serpapi_key and serpapi_key.strip():
        discovery_adapter: OfficialSiteDiscoveryAdapter | None = (
            SerpApiOfficialSiteDiscoveryAdapter(SerpApiWebSearchClient(serpapi_key))
        )
    elif brave_key and brave_key.strip():
        discovery_adapter = BraveOfficialSiteDiscoveryAdapter(
            BraveWebSearchClient(brave_key)
        )
    else:
        discovery_adapter = None
    return CompositePublicationSearchProvider(
        academic_adapter=OpenAlexPublicationAdapter(
            api_key=openalex_api_key,
        ),
        official_site_discovery_adapter=discovery_adapter,
        official_site_publication_adapter=(CrawlingOfficialSitePublicationAdapter()),
    )


# 将用户提供的官网链接转换为待后续核验的结构化来源
def _build_user_provided_sources(
    query: PublicationSearchQuery,
) -> list[DiscoveredSource]:
    source_type = (
        DiscoveredSourceType.LABORATORY_WEBSITE
        if query.laboratory_name is not None
        else DiscoveredSourceType.OTHER
    )
    return _deduplicate_sources(
        [
            DiscoveredSource(
                url=url,
                source_type=source_type,
                discovery_method=SourceDiscoveryMethod.USER_PROVIDED,
                verification_status=EvidenceVerificationStatus.UNVERIFIED,
                matched_signals=["用户提供"],
            )
            for url in query.official_urls
        ]
    )


# 将原始论文候选过滤、按单一来源精确去重并转换为正式证据
def _normalize_candidates(
    candidates: list[RawPublicationCandidate],
    query: PublicationSearchQuery,
) -> tuple[list[ResearchEvidence], list[str]]:
    records: list[ResearchEvidence] = []
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        duplicate_key = _candidate_duplicate_key(candidate)
        if duplicate_key in seen:
            continue
        seen.add(duplicate_key)
        if not _publication_date_in_range(candidate.publication_date, query):
            warnings.append(
                f"已跳过日期缺失、格式错误或超出范围的候选：{candidate.title}"
            )
            continue
        matched_professor = _match_professor(candidate, query)
        if matched_professor is None:
            warnings.append(f"已跳过无法匹配目标教授姓名的候选：{candidate.title}")
            continue
        if candidate.verification_status == EvidenceVerificationStatus.REJECTED:
            continue
        institution_match = _candidate_matches_institution(candidate, query)
        verification_status = candidate.verification_status
        if institution_match:
            verification_status = EvidenceVerificationStatus.VERIFIED
        elif verification_status != EvidenceVerificationStatus.VERIFIED:
            verification_status = EvidenceVerificationStatus.PARTIAL
        records.append(
            ResearchEvidence(
                evidence_id=_build_evidence_id(candidate),
                evidence_type=EvidenceType.PUBLICATION,
                title=candidate.title,
                source_name=candidate.source_name,
                retrieved_at=datetime.now(UTC),
                authors=list(candidate.authors),
                affiliations=list(candidate.affiliations),
                publication_date=candidate.publication_date,
                venue=candidate.venue,
                source_url=candidate.source_url,
                identifiers=dict(candidate.identifiers),
                abstract_or_summary=candidate.abstract_or_summary,
                source_citation=candidate.source_citation,
                matched_professor=matched_professor,
                matched_professor_position=(candidate.matched_professor_position),
                matched_professor_affiliations=list(
                    candidate.matched_professor_affiliations
                ),
                verification_status=verification_status,
            )
        )
    return records, warnings


# 汇总所有 Adapter 的状态并区分成功、降级、无结果和需确认
def _combine_statuses(
    statuses: list[RetrievalStatus],
    records: list[ResearchEvidence],
    *,
    confirmation_required: bool = False,
) -> RetrievalStatus:
    if confirmation_required:
        return RetrievalStatus.NEEDS_USER_CONFIRMATION
    problem_statuses = {
        RetrievalStatus.PARTIAL,
        RetrievalStatus.BLOCKED,
        RetrievalStatus.FAILED,
        RetrievalStatus.NEEDS_USER_CONFIRMATION,
    }
    if records:
        if any(status in problem_statuses for status in statuses):
            return RetrievalStatus.PARTIAL
        if any(
            record.verification_status != EvidenceVerificationStatus.VERIFIED
            for record in records
        ):
            return RetrievalStatus.PARTIAL
        return RetrievalStatus.SUCCESS
    if statuses and all(status == RetrievalStatus.FAILED for status in statuses):
        return RetrievalStatus.FAILED
    if statuses and all(status == RetrievalStatus.BLOCKED for status in statuses):
        return RetrievalStatus.BLOCKED
    if any(status in problem_statuses for status in statuses):
        if any(status == RetrievalStatus.NO_RESULT for status in statuses):
            return RetrievalStatus.PARTIAL
        if RetrievalStatus.FAILED in statuses:
            return RetrievalStatus.FAILED
        return RetrievalStatus.BLOCKED
    return RetrievalStatus.NO_RESULT


# 构造只在同一来源内部生效的 DOI 或标题日期去重键
def _candidate_duplicate_key(
    candidate: RawPublicationCandidate,
) -> tuple[str, str]:
    source_key = candidate.source_name.casefold()
    doi = candidate.identifiers.get("doi")
    record_key = (
        f"doi:{doi.casefold()}"
        if doi is not None
        else (
            f"title:{_normalize_text(candidate.title)}:"
            f"{candidate.publication_date or 'unknown-date'}"
        )
    )
    return source_key, record_key


# 根据来源和来源记录 ID 构造稳定且跨来源不冲突的证据 ID
def _build_evidence_id(candidate: RawPublicationCandidate) -> str:
    identity = (
        f"{candidate.source_name.casefold()}\0{candidate.source_record_id.casefold()}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"pub_{digest}"


# 匹配 Adapter 已确认的教授或论文作者列表中的目标姓名
def _match_professor(
    candidate: RawPublicationCandidate,
    query: PublicationSearchQuery,
) -> str | None:
    if candidate.matched_professor is not None:
        return candidate.matched_professor
    target_names = {
        _normalize_text(query.professor_name),
        *(_normalize_text(value) for value in query.professor_name_variants),
    }
    for author in candidate.authors:
        if _normalize_text(author) in target_names:
            return author
    return None


# 判断论文候选的机构信息是否匹配目标大学或研究科
def _candidate_matches_institution(
    candidate: RawPublicationCandidate,
    query: PublicationSearchQuery,
) -> bool:
    targets = [query.university_name, query.graduate_school_name]
    affiliations = (
        candidate.matched_professor_affiliations
        if candidate.matched_professor_position is not None
        else candidate.affiliations
    )
    for affiliation in affiliations:
        normalized_affiliation = _normalize_text(affiliation)
        for target in targets:
            normalized_target = _normalize_text(target)
            if (
                normalized_target in normalized_affiliation
                or normalized_affiliation in normalized_target
            ):
                return True
    return False


# 按官网权威性、核验状态、日期和标题稳定排序论文记录
def _record_priority_key(
    record: ResearchEvidence,
) -> tuple[int, int, int, str]:
    source_priority = (
        0 if record.source_name.casefold().startswith("official-site") else 1
    )
    verification_priority = (
        0 if record.verification_status == EvidenceVerificationStatus.VERIFIED else 1
    )
    try:
        _, end = _partial_date_interval(record.publication_date or "")
        date_priority = -end.toordinal()
    except ValueError:
        date_priority = 0
    return (
        source_priority,
        verification_priority,
        date_priority,
        _normalize_text(record.title),
    )


# 部分核验的索引记录只有获得可靠官网标题佐证才可进入证据集合
def _filter_uncorroborated_partial_index_records(
    records: list[ResearchEvidence],
) -> tuple[list[ResearchEvidence], list[str]]:
    official_titles = {
        _normalize_text(record.title)
        for record in records
        if record.source_name.casefold().startswith("official-site")
        and record.verification_status == EvidenceVerificationStatus.VERIFIED
    }
    retained: list[ResearchEvidence] = []
    dropped_count = 0
    for record in records:
        is_partial_index_record = (
            record.source_name.casefold() == "openalex"
            and record.verification_status != EvidenceVerificationStatus.VERIFIED
        )
        if (
            is_partial_index_record
            and _normalize_text(record.title) not in official_titles
        ):
            dropped_count += 1
            continue
        retained.append(record)
    warnings = (
        [
            "已剔除 "
            f"{dropped_count} 条目标作者机构缺失且未获官网标题佐证的 "
            "OpenAlex 部分核验记录"
        ]
        if dropped_count
        else []
    )
    return retained, warnings


# 判断部分精度的论文日期是否与查询日期范围相交
def _publication_date_in_range(
    publication_date: str | None,
    query: PublicationSearchQuery,
) -> bool:
    if publication_date is None:
        return False
    try:
        start, end = _partial_date_interval(publication_date)
    except ValueError:
        return False
    return start <= query.search_end_date and end >= query.search_start_date


# 将 YYYY、YYYY-MM 或 YYYY-MM-DD 转换为闭区间
def _partial_date_interval(value: str) -> tuple[date, date]:
    if len(value) == 4:
        year = int(value)
        return date(year, 1, 1), date(year, 12, 31)
    if len(value) == 7:
        parsed = date.fromisoformat(f"{value}-01")
        if parsed.month == 12:
            next_month = date(parsed.year + 1, 1, 1)
        else:
            next_month = date(parsed.year, parsed.month + 1, 1)
        return parsed, date.fromordinal(next_month.toordinal() - 1)
    if len(value) == 10:
        parsed = date.fromisoformat(value)
        return parsed, parsed
    raise ValueError("unsupported partial date")


# 统一姓名和机构文本以执行保守的精确比较
def _normalize_text(value: str) -> str:
    return normalize_search_text(value)


# 按 URL 去除候选官网来源中的重复项
def _deduplicate_sources(
    sources: list[DiscoveredSource],
) -> list[DiscoveredSource]:
    result: list[DiscoveredSource] = []
    seen: set[str] = set()
    for source in sources:
        key = source.url.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result


# 用官网读取阶段的核验结果覆盖同 URL 的早期来源候选
def _merge_source_updates(
    sources: list[DiscoveredSource],
    updates: list[DiscoveredSource],
) -> list[DiscoveredSource]:
    if not updates:
        return list(sources)
    updates_by_url = {source.url.rstrip("/").casefold(): source for source in updates}
    result = [
        updates_by_url.get(source.url.rstrip("/").casefold(), source)
        for source in sources
    ]
    known = {source.url.rstrip("/").casefold() for source in result}
    for source in updates:
        key = source.url.rstrip("/").casefold()
        if key not in known:
            known.add(key)
            result.append(source)
    return result


# 按大小写不敏感规则去除诊断字符串中的重复项
def _deduplicate_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
