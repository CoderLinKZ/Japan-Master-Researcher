# File: official_site.py
# Author: L1nzhk0
# Purpose: 本文件用于受限抓取研究室官网并从元数据、JSON-LD 和论文列表区块提取论文候选。

from __future__ import annotations

import hashlib
import json
import re
import urllib.robotparser
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

from jmr.domain import (
    DiscoveredSource,
    DiscoveredSourceType,
    EvidenceVerificationStatus,
    RetrievalStatus,
)

from .japanese_text import normalize_search_text
from .models import PublicationAdapterResult, RawPublicationCandidate
from .provider import PublicationSearchQuery
from .safe_http import SafeHTTPBlockedError, SafeHTTPError, SafeHTTPTextClient

DEFAULT_MAX_SITE_PAGES = 8
DEFAULT_MAX_SITE_DEPTH = 2
PUBLICATION_KEYWORDS = (
    "publication",
    "publications",
    "papers",
    "researchoutput",
    "achievement",
    "achievements",
    "業績",
    "論文",
    "発表",
    "研究成果",
)
ARTICLE_JSON_LD_TYPES = {
    "article",
    "scholarlyarticle",
    "techarticle",
    "report",
    "thesis",
}
BLOCK_TAGS = {"article", "li", "p", "tr"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


@dataclass(frozen=True, slots=True)
class PageLink:
    """保存官网页面中的链接文本和绝对 URL。"""

    text: str
    url: str


@dataclass(frozen=True, slots=True)
class ContentBlock:
    """保存可能对应单条论文的 HTML 内容区块。"""

    text: str
    links: tuple[PageLink, ...]
    section_heading: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedOfficialPage:
    """保存完成 HTML 解析后的官网页面信息。"""

    url: str
    title: str | None
    text: str
    links: tuple[PageLink, ...]
    blocks: tuple[ContentBlock, ...]
    metadata: Mapping[str, tuple[str, ...]]
    json_ld: tuple[Any, ...]


@dataclass(slots=True)
class _MutableBlock:
    """在 HTML 流式解析期间暂存区块文本和链接。"""

    tag: str
    section_heading: str | None
    text_parts: list[str] = field(default_factory=list)
    links: list[PageLink] = field(default_factory=list)


@dataclass(slots=True)
class _CrawlOutcome:
    """保存单个官网根来源的抓取结果和诊断状态。"""

    candidates: list[RawPublicationCandidate] = field(default_factory=list)
    source_update: DiscoveredSource | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    fetched_pages: int = 0
    blocked_requests: int = 0


class _OfficialHTMLParser(HTMLParser):
    """从官网 HTML 中提取文本、链接、内容区块和结构化元数据。"""

    # 初始化官网 HTML 流式解析状态
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.blocks: list[tuple[str, list[tuple[str, str]], str | None]] = []
        self.metadata: dict[str, list[str]] = {}
        self.json_ld_texts: list[str] = []
        self.title_parts: list[str] = []
        self._block_stack: list[_MutableBlock] = []
        self._current_heading: str | None = None
        self._heading_tag: str | None = None
        self._heading_parts: list[str] = []
        self._anchor_href: str | None = None
        self._anchor_parts: list[str] = []
        self._title_depth = 0
        self._ignored_depth = 0
        self._json_ld_depth = 0
        self._json_ld_parts: list[str] = []

    # 处理 HTML 开始标签并建立区块、链接和元数据状态
    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized_tag = tag.casefold()
        attributes = {
            key.casefold(): value for key, value in attrs if value is not None
        }
        if normalized_tag == "meta":
            name = (
                attributes.get("name") or attributes.get("property") or ""
            ).casefold()
            content = attributes.get("content", "").strip()
            if name and content:
                self.metadata.setdefault(name, []).append(content)
            return
        if normalized_tag == "script":
            script_type = attributes.get("type", "").casefold()
            if script_type == "application/ld+json":
                self._json_ld_depth += 1
                if self._json_ld_depth == 1:
                    self._json_ld_parts = []
            else:
                self._ignored_depth += 1
            return
        if normalized_tag in {"style", "noscript"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth or self._json_ld_depth:
            return
        if normalized_tag == "title":
            self._title_depth += 1
        if normalized_tag in HEADING_TAGS:
            self._heading_tag = normalized_tag
            self._heading_parts = []
        if normalized_tag == "a":
            self._anchor_href = attributes.get("href")
            self._anchor_parts = []
        if normalized_tag in BLOCK_TAGS:
            self._block_stack.append(
                _MutableBlock(
                    tag=normalized_tag,
                    section_heading=self._current_heading,
                )
            )

    # 处理 HTML 文本并写入当前活动解析容器
    def handle_data(self, data: str) -> None:
        if self._json_ld_depth:
            self._json_ld_parts.append(data)
            return
        if self._ignored_depth:
            return
        normalized = _clean_text(data)
        if not normalized:
            return
        self.text_parts.append(normalized)
        if self._title_depth:
            self.title_parts.append(normalized)
        if self._heading_tag is not None:
            self._heading_parts.append(normalized)
        if self._anchor_href is not None:
            self._anchor_parts.append(normalized)
        for block in self._block_stack:
            block.text_parts.append(normalized)

    # 处理 HTML 结束标签并提交完成的解析对象
    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag == "script":
            if self._json_ld_depth:
                self._json_ld_depth -= 1
                if self._json_ld_depth == 0:
                    raw_json = " ".join(self._json_ld_parts).strip()
                    if raw_json:
                        self.json_ld_texts.append(raw_json)
                    self._json_ld_parts = []
            elif self._ignored_depth:
                self._ignored_depth -= 1
            return
        if normalized_tag in {"style", "noscript"}:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return
        if self._ignored_depth or self._json_ld_depth:
            return
        if normalized_tag == "title" and self._title_depth:
            self._title_depth -= 1
        if normalized_tag == self._heading_tag:
            heading = _clean_text(" ".join(self._heading_parts))
            if heading:
                self._current_heading = heading
            self._heading_tag = None
            self._heading_parts = []
        if normalized_tag == "a" and self._anchor_href is not None:
            text = _clean_text(" ".join(self._anchor_parts))
            link = (text, self._anchor_href)
            self.links.append(link)
            for block in self._block_stack:
                block.links.append(PageLink(text=text, url=self._anchor_href))
            self._anchor_href = None
            self._anchor_parts = []
        if (
            normalized_tag in BLOCK_TAGS
            and self._block_stack
            and self._block_stack[-1].tag == normalized_tag
        ):
            block = self._block_stack.pop()
            text = _clean_text(" ".join(block.text_parts))
            if text:
                self.blocks.append(
                    (
                        text,
                        [(link.text, link.url) for link in block.links],
                        block.section_heading,
                    )
                )


class CrawlingOfficialSitePublicationAdapter:
    """受限抓取官网并从可靠结构中提取目标教授的论文候选。"""

    # 初始化官网抓取 Client、最大页面数和最大深度
    def __init__(
        self,
        http_client: SafeHTTPTextClient | None = None,
        max_pages: int = DEFAULT_MAX_SITE_PAGES,
        max_depth: int = DEFAULT_MAX_SITE_DEPTH,
    ) -> None:
        if not 1 <= max_pages <= 50:
            raise ValueError("max_pages must be between 1 and 50")
        if not 0 <= max_depth <= 5:
            raise ValueError("max_depth must be between 0 and 5")
        self._http_client = http_client or SafeHTTPTextClient()
        self._max_pages = max_pages
        self._max_depth = max_depth

    # 抓取所有已选择官网并汇总论文候选和来源核验状态
    def search(
        self,
        query: PublicationSearchQuery,
        sources: list[DiscoveredSource],
    ) -> PublicationAdapterResult:
        outcomes = [self._crawl_source(query, source) for source in sources]
        matching_candidates = _deduplicate_candidates(
            [
                candidate
                for outcome in outcomes
                for candidate in outcome.candidates
                if _candidate_in_query_range(candidate, query)
            ]
        )
        candidates = matching_candidates[: query.max_results]
        source_updates = [
            outcome.source_update
            for outcome in outcomes
            if outcome.source_update is not None
        ]
        warnings = _deduplicate_strings(
            [warning for outcome in outcomes for warning in outcome.warnings]
        )
        if len(matching_candidates) > query.max_results:
            warnings.append(
                "官网在检索范围内提取到 "
                f"{len(matching_candidates)} 条论文，当前仅返回前 "
                f"{query.max_results} 条；该结果不代表完整论文清单"
            )
        errors = _deduplicate_strings(
            [error for outcome in outcomes for error in outcome.errors]
        )
        fetched_pages = sum(outcome.fetched_pages for outcome in outcomes)
        blocked_requests = sum(outcome.blocked_requests for outcome in outcomes)
        if candidates:
            status = (
                RetrievalStatus.PARTIAL
                if errors or blocked_requests
                else RetrievalStatus.SUCCESS
            )
        elif fetched_pages:
            status = RetrievalStatus.NO_RESULT
            warnings.append(
                "官网访问成功，但未提取到同时满足教授姓名、标题和日期要求的论文"
            )
        elif blocked_requests and not errors:
            status = RetrievalStatus.BLOCKED
        else:
            status = RetrievalStatus.FAILED
        return PublicationAdapterResult(
            status=status,
            candidates=candidates,
            source_updates=source_updates,
            warnings=_deduplicate_strings(warnings),
            errors=errors,
        )

    # 在单一官网内按页面数、深度和同站点规则执行广度优先抓取
    def _crawl_source(
        self,
        query: PublicationSearchQuery,
        source: DiscoveredSource,
    ) -> _CrawlOutcome:
        outcome = _CrawlOutcome()
        queue: deque[tuple[str, int]] = deque([(source.url, 0)])
        visited: set[str] = set()
        robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        allowed_hosts = {(urlparse(source.url).hostname or "").casefold()}
        combined_signals = list(source.matched_signals)
        best_verification = source.verification_status
        source_title = source.title
        saw_publication_page = (
            source.source_type == DiscoveredSourceType.PUBLICATION_LIST
        )
        while queue and outcome.fetched_pages < self._max_pages:
            page_url, depth = queue.popleft()
            canonical_url = _canonicalize_url(page_url)
            if canonical_url in visited:
                continue
            visited.add(canonical_url)
            if not self._robots_allowed(
                page_url,
                robots_cache,
                outcome.warnings,
            ):
                outcome.blocked_requests += 1
                outcome.warnings.append(f"robots.txt 不允许抓取官网页面：{page_url}")
                continue
            try:
                response = self._http_client.fetch(page_url)
            except SafeHTTPBlockedError as exc:
                outcome.blocked_requests += 1
                outcome.warnings.append(str(exc))
                continue
            except SafeHTTPError as exc:
                outcome.errors.append(str(exc))
                if depth == 0 and urlparse(page_url).path in {"", "/"}:
                    queue.append((urljoin(page_url, "/publications/"), 1))
                    outcome.warnings.append(
                        "官网首页不可读取，已尝试同站 /publications/ 论文列表"
                    )
                continue
            outcome.fetched_pages += 1
            effective_host = (urlparse(response.url).hostname or "").casefold()
            if effective_host:
                allowed_hosts.add(effective_host)
            page = parse_official_page(response.text, response.url)
            robots_directives = " ".join(page.metadata.get("robots", ())).casefold()
            source_title = source_title or page.title
            page_signals = _identity_signals(page.text, query)
            combined_signals = _deduplicate_strings(
                [
                    *combined_signals,
                    *page_signals,
                ]
            )
            page_verification = _verification_from_signals(page_signals)
            best_verification = _stronger_verification(
                best_verification,
                page_verification,
            )
            page_is_publication = _has_publication_context(
                f"{page.url} {page.title or ''}"
            )
            saw_publication_page = saw_publication_page or page_is_publication
            if "noindex" not in robots_directives and (
                page_signals
                or best_verification != (EvidenceVerificationStatus.UNVERIFIED)
            ):
                outcome.candidates.extend(
                    _extract_page_candidates(page, query, page_signals)
                )
            if depth >= self._max_depth or "nofollow" in robots_directives:
                continue
            for link in page.links:
                link_host = (urlparse(link.url).hostname or "").casefold()
                if not any(_same_site_host(link_host, host) for host in allowed_hosts):
                    continue
                if not _should_follow_site_link(link):
                    continue
                queue.append((link.url, depth + 1))
        source_type = (
            DiscoveredSourceType.PUBLICATION_LIST
            if saw_publication_page
            else source.source_type
        )
        outcome.source_update = DiscoveredSource(
            url=source.url,
            source_type=source_type,
            discovery_method=source.discovery_method,
            verification_status=best_verification,
            title=source_title,
            matched_signals=combined_signals,
        )
        return outcome

    # 获取并缓存同一站点的 robots.txt 规则后判断页面是否允许抓取
    def _robots_allowed(
        self,
        page_url: str,
        cache: dict[str, urllib.robotparser.RobotFileParser | None],
        warnings: list[str],
    ) -> bool:
        parsed = urlparse(page_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in cache:
            robots_url = f"{origin}/robots.txt"
            try:
                response = self._http_client.fetch(robots_url)
            except SafeHTTPError:
                cache[origin] = None
                warnings.append(f"未能读取 robots.txt，按无显式规则处理：{origin}")
            else:
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(robots_url)
                parser.parse(response.text.splitlines())
                cache[origin] = parser
        policy = cache[origin]
        if policy is None:
            return True
        return policy.can_fetch("Japan-Master-Researcher", page_url)


# 将 HTML 文本解析为链接、区块、元数据和 JSON-LD
def parse_official_page(html: str, page_url: str) -> ParsedOfficialPage:
    parser = _OfficialHTMLParser()
    parser.feed(html)
    parser.close()
    links = tuple(_resolve_links(parser.links, page_url))
    blocks: list[ContentBlock] = []
    for text, raw_links, heading in parser.blocks:
        blocks.append(
            ContentBlock(
                text=text,
                links=tuple(_resolve_links(raw_links, page_url)),
                section_heading=heading,
            )
        )
    json_ld: list[Any] = []
    for raw_json in parser.json_ld_texts:
        try:
            json_ld.append(json.loads(raw_json))
        except json.JSONDecodeError:
            continue
    metadata = {key: tuple(values) for key, values in parser.metadata.items()}
    title = _clean_text(" ".join(parser.title_parts)) or None
    return ParsedOfficialPage(
        url=page_url,
        title=title,
        text=_clean_text(" ".join(parser.text_parts)),
        links=links,
        blocks=tuple(blocks),
        metadata=metadata,
        json_ld=tuple(json_ld),
    )


# 从单个官网页面的三类结构中提取论文候选
def _extract_page_candidates(
    page: ParsedOfficialPage,
    query: PublicationSearchQuery,
    page_signals: list[str],
) -> list[RawPublicationCandidate]:
    candidates: list[RawPublicationCandidate] = []
    metadata_candidate = _candidate_from_citation_metadata(
        page,
        query,
        page_signals,
    )
    if metadata_candidate is not None:
        candidates.append(metadata_candidate)
    for item in page.json_ld:
        for node in _walk_json_ld(item):
            candidate = _candidate_from_json_ld(
                node,
                page,
                query,
                page_signals,
            )
            if candidate is not None:
                candidates.append(candidate)
    for index, block in enumerate(page.blocks):
        candidate = _candidate_from_block(
            block,
            index,
            page,
            query,
            page_signals,
        )
        if candidate is not None:
            candidates.append(candidate)
    return _deduplicate_candidates(candidates)


# 从 Highwire citation meta 标签构造论文候选
def _candidate_from_citation_metadata(
    page: ParsedOfficialPage,
    query: PublicationSearchQuery,
    page_signals: list[str],
) -> RawPublicationCandidate | None:
    title = _first_metadata(page, "citation_title")
    publication_date = _first_metadata(
        page, "citation_publication_date"
    ) or _first_metadata(page, "citation_date")
    authors = list(page.metadata.get("citation_author", ()))
    matched_professor = _find_matching_professor(authors, query)
    if title is None or publication_date is None or matched_professor is None:
        return None
    doi = _normalize_doi(_first_metadata(page, "citation_doi"))
    source_url = (
        _first_metadata(page, "citation_public_url")
        or _first_metadata(page, "citation_pdf_url")
        or page.url
    )
    return _build_site_candidate(
        title=title,
        authors=authors,
        publication_date=_normalize_publication_date(publication_date),
        venue=(
            _first_metadata(page, "citation_journal_title")
            or _first_metadata(page, "citation_conference_title")
        ),
        source_url=_safe_candidate_url(source_url, page.url),
        doi=doi,
        matched_professor=matched_professor,
        page=page,
        query=query,
        page_signals=page_signals,
        record_seed=f"meta:{title}:{publication_date}",
    )


# 从 schema.org JSON-LD Article 节点构造论文候选
def _candidate_from_json_ld(
    node: Mapping[str, Any],
    page: ParsedOfficialPage,
    query: PublicationSearchQuery,
    page_signals: list[str],
) -> RawPublicationCandidate | None:
    raw_types = node.get("@type", [])
    types = [raw_types] if isinstance(raw_types, str) else raw_types
    if not isinstance(types, list) or not any(
        isinstance(value, str) and value.casefold() in ARTICLE_JSON_LD_TYPES
        for value in types
    ):
        return None
    title = node.get("headline") or node.get("name")
    publication_date = node.get("datePublished")
    if not isinstance(title, str) or not isinstance(publication_date, str):
        return None
    authors = _json_ld_author_names(node.get("author"))
    matched_professor = _find_matching_professor(authors, query)
    if matched_professor is None:
        return None
    doi = _json_ld_doi(node)
    source_url = _json_ld_url(node) or page.url
    venue = _json_ld_venue(node)
    abstract = node.get("abstract") or node.get("description")
    if not isinstance(abstract, str):
        abstract = None
    return _build_site_candidate(
        title=title,
        authors=authors,
        publication_date=_normalize_publication_date(publication_date),
        venue=venue,
        source_url=_safe_candidate_url(source_url, page.url),
        doi=doi,
        matched_professor=matched_professor,
        page=page,
        query=query,
        page_signals=page_signals,
        record_seed=f"jsonld:{title}:{publication_date}",
        abstract_or_summary=abstract,
    )


# 从论文列表中的带标题链接区块构造保守候选
def _candidate_from_block(
    block: ContentBlock,
    index: int,
    page: ParsedOfficialPage,
    query: PublicationSearchQuery,
    page_signals: list[str],
) -> RawPublicationCandidate | None:
    context = f"{page.url} {page.title or ''} {block.section_heading or ''}"
    if not _has_publication_context(context):
        return None
    matched_professor = _find_matching_professor([block.text], query)
    if matched_professor is None:
        return None
    publication_date = _extract_year(block.text)
    if publication_date is None:
        return None
    citation_title = _extract_title_from_citation(
        block.text,
        matched_professor,
    )
    title_link = _select_title_link(block.links)
    title = citation_title or (title_link.text if title_link is not None else None)
    if title is None:
        return None
    authors = _extract_authors_from_citation(
        block.text,
        matched_professor,
    )
    doi = _extract_doi(f"{block.text} {' '.join(link.url for link in block.links)}")
    source_link = title_link or _select_source_link(block.links)
    source_url = source_link.url if source_link is not None else page.url
    venue = _extract_venue_from_citation(block.text, title)
    venue = _enrich_venue_from_source_url(venue, source_url)
    return _build_site_candidate(
        title=title,
        authors=authors or [matched_professor],
        publication_date=publication_date,
        venue=venue,
        source_url=source_url,
        doi=doi,
        matched_professor=matched_professor,
        page=page,
        query=query,
        page_signals=page_signals,
        record_seed=f"block:{index}:{block.text}",
        source_citation=block.text,
    )


# 构造带稳定来源记录 ID 的官网论文候选
def _build_site_candidate(
    *,
    title: str,
    authors: list[str],
    publication_date: str | None,
    venue: str | None,
    source_url: str,
    doi: str | None,
    matched_professor: str,
    page: ParsedOfficialPage,
    query: PublicationSearchQuery,
    page_signals: list[str],
    record_seed: str,
    abstract_or_summary: str | None = None,
    source_citation: str | None = None,
) -> RawPublicationCandidate | None:
    cleaned_title = _clean_text(title)
    if not cleaned_title or publication_date is None:
        return None
    identifiers: dict[str, str] = {}
    if doi is not None:
        identifiers["doi"] = doi
    digest = hashlib.sha256(f"{page.url}\0{record_seed}".encode()).hexdigest()[:24]
    hostname = urlparse(page.url).hostname or "unknown-host"
    verification_status = _verification_from_signals(page_signals)
    if verification_status == EvidenceVerificationStatus.UNVERIFIED:
        verification_status = EvidenceVerificationStatus.PARTIAL
    affiliations = [query.university_name] if "大学名称" in page_signals else []
    return RawPublicationCandidate(
        source_record_id=f"site_{digest}",
        title=cleaned_title,
        source_name=f"official-site:{hostname}",
        authors=_deduplicate_strings(authors),
        affiliations=affiliations,
        publication_date=publication_date,
        venue=_clean_text(venue) if venue else None,
        source_url=source_url,
        identifiers=identifiers,
        abstract_or_summary=(
            _clean_text(abstract_or_summary) if abstract_or_summary else None
        ),
        source_citation=(_clean_text(source_citation) if source_citation else None),
        matched_professor=matched_professor,
        matched_professor_position=_professor_position(
            authors,
            matched_professor,
        ),
        matched_professor_affiliations=list(affiliations),
        verification_status=verification_status,
    )


# 递归遍历 JSON-LD 中的对象节点
def _walk_json_ld(value: Any) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        result.append(value)
        graph = value.get("@graph")
        if graph is not None:
            result.extend(_walk_json_ld(graph))
    elif isinstance(value, list):
        for item in value:
            result.extend(_walk_json_ld(item))
    return result


# 提取 JSON-LD 作者字段中的姓名
def _json_ld_author_names(value: Any) -> list[str]:
    raw_authors = value if isinstance(value, list) else [value]
    authors: list[str] = []
    for author in raw_authors:
        if isinstance(author, str) and author.strip():
            authors.append(author.strip())
        elif isinstance(author, Mapping):
            name = author.get("name")
            if isinstance(name, str) and name.strip():
                authors.append(name.strip())
    return _deduplicate_strings(authors)


# 从 JSON-LD 节点中提取 DOI
def _json_ld_doi(node: Mapping[str, Any]) -> str | None:
    values = [node.get("identifier"), node.get("sameAs"), node.get("url")]
    for value in values:
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if isinstance(candidate, str):
                doi = _extract_doi(candidate)
                if doi is not None:
                    return doi
            elif isinstance(candidate, Mapping):
                raw_value = candidate.get("value") or candidate.get("@id")
                if isinstance(raw_value, str):
                    doi = _extract_doi(raw_value)
                    if doi is not None:
                        return doi
    return None


# 从 JSON-LD 节点中提取论文链接
def _json_ld_url(node: Mapping[str, Any]) -> str | None:
    for field_name in ("url", "mainEntityOfPage"):
        value = node.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            identifier = value.get("@id")
            if isinstance(identifier, str) and identifier.strip():
                return identifier.strip()
    return None


# 从 JSON-LD 节点中提取期刊或会议名称
def _json_ld_venue(node: Mapping[str, Any]) -> str | None:
    container = node.get("isPartOf")
    if isinstance(container, Mapping):
        name = container.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    publisher = node.get("publisher")
    if isinstance(publisher, Mapping):
        name = publisher.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


# 返回指定 citation meta 字段的第一个非空值
def _first_metadata(page: ParsedOfficialPage, name: str) -> str | None:
    values = page.metadata.get(name, ())
    return values[0].strip() if values and values[0].strip() else None


# 将原始链接解析为去重后的 HTTP(S) 绝对链接
def _resolve_links(
    links: list[tuple[str, str]],
    page_url: str,
) -> list[PageLink]:
    result: list[PageLink] = []
    seen: set[str] = set()
    for text, href in links:
        absolute_url = urljoin(page_url, href)
        parsed = urlparse(absolute_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        canonical = _canonicalize_url(absolute_url)
        if canonical in seen:
            continue
        seen.add(canonical)
        result.append(PageLink(text=_clean_text(text), url=absolute_url))
    return result


# 将元数据中的论文链接限制为 HTTP(S), 否则回退到当前官网页面
def _safe_candidate_url(value: str, page_url: str) -> str:
    absolute_url = urljoin(page_url, value)
    parsed = urlparse(absolute_url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return absolute_url
    return page_url


# 使用页面文本计算教授、大学、研究科和研究室身份信号
def _identity_signals(
    page_text: str,
    query: PublicationSearchQuery,
) -> list[str]:
    signals: list[str] = []
    if _find_matching_professor([page_text], query) is not None:
        signals.append("教授姓名")
    if _contains_text(page_text, query.university_name):
        signals.append("大学名称")
    if _contains_text(page_text, query.graduate_school_name):
        signals.append("研究科名称")
    if query.laboratory_name is not None and _contains_text(
        page_text, query.laboratory_name
    ):
        signals.append("研究室名称")
    return signals


# 根据身份信号确定官网来源核验等级
def _verification_from_signals(
    signals: list[str],
) -> EvidenceVerificationStatus:
    signal_set = set(signals)
    if "教授姓名" in signal_set and (
        {"大学名称", "研究科名称", "研究室名称"} & signal_set
    ):
        return EvidenceVerificationStatus.VERIFIED
    if signal_set:
        return EvidenceVerificationStatus.PARTIAL
    return EvidenceVerificationStatus.UNVERIFIED


# 返回两个核验状态中可信度更高的状态
def _stronger_verification(
    first: EvidenceVerificationStatus,
    second: EvidenceVerificationStatus,
) -> EvidenceVerificationStatus:
    ranks = {
        EvidenceVerificationStatus.REJECTED: -1,
        EvidenceVerificationStatus.UNVERIFIED: 0,
        EvidenceVerificationStatus.PARTIAL: 1,
        EvidenceVerificationStatus.VERIFIED: 2,
    }
    return first if ranks[first] >= ranks[second] else second


# 在作者或区块文本中查找目标教授的明确姓名变体
def _find_matching_professor(
    values: list[str],
    query: PublicationSearchQuery,
) -> str | None:
    targets = [query.professor_name, *query.professor_name_variants]
    for target in targets:
        if any(_contains_text(value, target) for value in values):
            return target
    return None


# 从区块链接中选择最可能是论文完整标题的链接
def _select_title_link(links: tuple[PageLink, ...]) -> PageLink | None:
    candidates: list[PageLink] = []
    for link in links:
        text = _clean_text(link.text)
        normalized = _normalize_text(text)
        if not 12 <= len(text) <= 500:
            continue
        if normalized in {"pdf", "doi", "detail", "details", "abstract"}:
            continue
        if text.casefold().startswith(("http://", "https://")):
            continue
        candidates.append(PageLink(text=text, url=link.url))
    return max(candidates, key=lambda item: len(item.text), default=None)


# 从无标题链接的常见“作者: 标题, 载体, 年份”格式中提取标题
def _extract_title_from_citation(
    citation: str,
    matched_professor: str,
) -> str | None:
    professor_position = _normalize_text(citation).find(
        _normalize_text(matched_professor)
    )
    if professor_position < 0:
        return None
    separator_index = citation.find(":")
    if separator_index < 0:
        separator_index = citation.find("：")
    if separator_index < 0:
        return None
    remainder = citation[separator_index + 1 :].strip()
    venue_boundary = re.search(
        r",\s+(?=(?:ACM|IEEE|ACL|SIGIR|CIKM|WSDM|WWW|The Web "
        r"Conference|arxiv|DEIM|ICMIP|CVPR|ICPR|AAAI|IJCAI|EMNLP|"
        r"NAACL|KDD|RecSys|ECIR|Proceedings|Journal|Transactions|"
        r"Information Processing|情報処理|電子情報通信))",
        remainder,
        flags=re.IGNORECASE,
    )
    if venue_boundary is None:
        venue_boundary = re.search(
            r",\s+(?=[A-Z][A-Z0-9-]{1,15}(?:\s|$))",
            remainder,
        )
    if venue_boundary is None:
        return None
    title = _clean_text(remainder[: venue_boundary.start()])
    return title if 12 <= len(title) <= 500 else None


# 从官网原始引文冒号前缀中提取保留顺序的作者姓名
def _extract_authors_from_citation(
    citation: str,
    matched_professor: str,
) -> list[str]:
    separator_positions = [
        position
        for separator in (":", "：")
        if (position := citation.find(separator)) >= 0
    ]
    if not separator_positions:
        return []
    prefix = citation[: min(separator_positions)].strip()
    raw_authors = re.split(
        r"\s*(?:,|，|、|\band\b|&)\s*",
        prefix,
        flags=re.IGNORECASE,
    )
    authors = _deduplicate_strings(
        [_clean_text(author) for author in raw_authors if _clean_text(author)]
    )
    if _professor_position(authors, matched_professor) is None:
        return []
    return authors


# 返回目标教授在官网作者顺序中的一基位置
def _professor_position(
    authors: list[str],
    matched_professor: str,
) -> int | None:
    target = _normalize_text(matched_professor)
    for index, author in enumerate(authors, start=1):
        if _normalize_text(author) == target:
            return index
    return None


# 从论文区块链接中选择 DOI、PDF 或其他可访问来源链接
def _select_source_link(links: tuple[PageLink, ...]) -> PageLink | None:
    if not links:
        return None
    for link in links:
        if "doi.org/" in link.url.casefold():
            return link
    for link in links:
        if "pdf" in _normalize_text(link.text) or link.url.endswith(".pdf"):
            return link
    return links[0]


# 从规范化引文中提取标题之后、年份之前的发表载体文本
def _extract_venue_from_citation(
    citation: str,
    title: str,
) -> str | None:
    title_index = citation.find(title)
    if title_index < 0:
        return None
    remainder = citation[title_index + len(title) :].lstrip(" ,，")
    year_matches = list(
        re.finditer(
            r"(?<!\d)(?:19|20)\d{2}(?!\d)",
            remainder,
        )
    )
    if not year_matches:
        return None
    publication_year = year_matches[-1] if len(year_matches) > 1 else year_matches[0]
    venue = remainder[: publication_year.start()].strip(" ,，.;")
    venue = re.sub(
        r"(?:,\s*)?(?:to appear|accepted)\s*$",
        "",
        venue,
        flags=re.IGNORECASE,
    ).strip(" ,，.;")
    return venue[:300] or None


# 从 Confit 发表链接补充 DEIM 报告编号
def _enrich_venue_from_source_url(
    venue: str | None,
    source_url: str,
) -> str | None:
    if venue is None or not _normalize_text(venue).startswith("deim"):
        return venue
    parsed = urlparse(source_url)
    if parsed.hostname != "pub.confit.atlas.jp":
        return venue
    match = re.search(r"/presentation/([^/?#]+)", parsed.path)
    if match is None:
        return venue
    presentation_code = unquote(match.group(1)).strip()
    if not presentation_code or presentation_code.casefold() in venue.casefold():
        return venue
    return f"{venue}, {presentation_code}"


# 判断页面、标题或链接是否位于论文成果语境
def _has_publication_context(value: str) -> bool:
    normalized = _normalize_text(value)
    return any(
        _normalize_text(keyword) in normalized for keyword in PUBLICATION_KEYWORDS
    )


# 判断同站点链接是否值得在有限页面预算内继续抓取
def _should_follow_site_link(link: PageLink) -> bool:
    if _has_publication_context(f"{link.text} {link.url}"):
        return True
    normalized = _normalize_text(f"{link.text} {link.url}")
    return any(
        keyword in normalized
        for keyword in (
            "english",
            "英語",
            "日本語",
        )
    )


# 从文本中提取首个四位论文年份
def _extract_year(value: str) -> str | None:
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", value)
    return match.group(1) if match else None


# 统一官网元数据中的日期为支持的部分日期格式
def _normalize_publication_date(value: str) -> str | None:
    normalized = value.strip()
    match = re.search(
        r"(?<!\d)((?:19|20)\d{2})(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?",
        normalized,
    )
    if match is None:
        return None
    year, month, day = match.groups()
    if month is None:
        return year
    if day is None:
        return f"{year}-{int(month):02d}"
    return f"{year}-{int(month):02d}-{int(day):02d}"


# 判断官网候选的部分日期是否落在本次查询年份范围内
def _candidate_in_query_range(
    candidate: RawPublicationCandidate,
    query: PublicationSearchQuery,
) -> bool:
    publication_date = candidate.publication_date
    if publication_date is None or len(publication_date) < 4:
        return False
    try:
        year = int(publication_date[:4])
    except ValueError:
        return False
    return query.search_start_date.year <= year <= query.search_end_date.year


# 从文本或链接中提取 DOI
def _extract_doi(value: str) -> str | None:
    match = re.search(
        r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
        value,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(0).rstrip(".,;:)]}").casefold()


# 将 DOI URL 或 DOI 字符串转换为统一标识符
def _normalize_doi(value: str | None) -> str | None:
    return _extract_doi(value) if value is not None else None


# 判断两个主机是否相同或只相差 www 前缀
def _same_site_host(first: str, second: str) -> bool:
    return first.removeprefix("www.") == second.removeprefix("www.")


# 统一 URL 以执行单次抓取中的去重
def _canonicalize_url(value: str) -> str:
    parsed = urlparse(value)
    return parsed._replace(fragment="").geturl().rstrip("/").casefold()


# 按 DOI 或规范化标题去除官网 Adapter 内部的精确重复项
def _deduplicate_candidates(
    candidates: list[RawPublicationCandidate],
) -> list[RawPublicationCandidate]:
    result: list[RawPublicationCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        doi = candidate.identifiers.get("doi")
        identity = (
            f"doi:{doi.casefold()}"
            if doi is not None
            else f"title:{_normalize_text(candidate.title)}"
        )
        key = candidate.source_name.casefold(), identity
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


# 按大小写不敏感规则去除字符串列表中的重复项
def _deduplicate_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _clean_text(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


# 按 Unicode 规范化规则判断文本包含目标字符串
def _contains_text(value: str, target: str) -> bool:
    normalized_target = _normalize_text(target)
    return bool(normalized_target) and normalized_target in _normalize_text(value)


# 统一文本以比较姓名、机构和页面关键词
def _normalize_text(value: str) -> str:
    return normalize_search_text(value)


# 合并 HTML 文本中的连续空白
def _clean_text(value: str) -> str:
    return " ".join(value.split())
