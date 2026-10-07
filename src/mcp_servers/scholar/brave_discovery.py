# File: brave_discovery.py
# Author: L1nzhk0
# Purpose: 本文件用于通过 Brave Search API 发现并保守筛选目标研究室或教授的官网候选。

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from jmr.domain import (
    DiscoveredSource,
    DiscoveredSourceType,
    EvidenceVerificationStatus,
    RetrievalStatus,
    SourceDiscoveryMethod,
)

from .japanese_text import normalize_search_text
from .models import SourceDiscoveryResult
from .provider import PublicationSearchQuery

BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_TIMEOUT_SECONDS = 15.0
BRAVE_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
NON_OFFICIAL_HOST_SUFFIXES = {
    "academia.edu",
    "arxiv.org",
    "dblp.org",
    "facebook.com",
    "github.com",
    "google.com",
    "linkedin.com",
    "openalex.org",
    "researchgate.net",
    "researchmap.jp",
    "semanticscholar.org",
    "wikipedia.org",
    "x.com",
}


class OfficialSiteSearchError(RuntimeError):
    """表示官网搜索服务请求或响应解析失败。"""


class BraveSearchError(OfficialSiteSearchError):
    """表示 Brave Search API 请求或响应解析失败。"""


@dataclass(frozen=True, slots=True)
class BraveSearchResult:
    """保存 Brave Web Search 返回的一条网页候选。"""

    title: str
    url: str
    description: str


class BraveSearchClient(Protocol):
    """定义官网发现 Adapter 所需的最小 Web Search 接口。"""

    # 执行网页搜索并返回结构化搜索结果
    def search(self, query: str, count: int = 10) -> list[BraveSearchResult]: ...


class BraveWebSearchClient:
    """使用 Brave 官方 Web Search API 返回网页标题、链接和摘要。"""

    # 初始化 Brave Search API Key 和请求限制
    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = BRAVE_TIMEOUT_SECONDS,
        max_response_bytes: int = BRAVE_MAX_RESPONSE_BYTES,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Brave Search API key must be non-empty")
        self._api_key = api_key.strip()
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes

    # 调用 Brave Web Search API 并解析 web.results
    def search(self, query: str, count: int = 10) -> list[BraveSearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Brave Search query must be non-empty")
        if not 1 <= count <= 20:
            raise ValueError("Brave Search count must be between 1 and 20")
        request_url = f"{BRAVE_WEB_SEARCH_URL}?{
            urlencode(
                {
                    'q': normalized_query,
                    'count': count,
                    'country': 'jp',
                    'search_lang': 'ja',
                    'result_filter': 'web',
                    'text_decorations': 'false',
                }
            )
        }"
        request = Request(
            request_url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self._api_key,
                "User-Agent": "Japan-Master-Researcher/0.1",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = response.read(self._max_response_bytes + 1)
        except HTTPError as exc:
            raise BraveSearchError(f"Brave Search returned HTTP {exc.code}") from exc
        except (TimeoutError, URLError) as exc:
            raise BraveSearchError(
                "Brave Search request could not be completed"
            ) from exc
        if len(payload) > self._max_response_bytes:
            raise BraveSearchError("Brave Search response exceeded the size limit")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BraveSearchError("Brave Search returned invalid JSON") from exc
        return _parse_search_results(decoded)


class BraveOfficialSiteDiscoveryAdapter:
    """通过网页搜索结果中的身份信号发现官网并处理候选歧义。"""

    # 初始化用于官网发现的 Brave Search Client
    def __init__(self, search_client: BraveSearchClient) -> None:
        self._search_client = search_client

    # 搜索官网并只自动选择唯一且明显领先的高可信候选
    def discover(self, query: PublicationSearchQuery) -> SourceDiscoveryResult:
        try:
            search_results = self._search_client.search(
                _build_discovery_query(query),
                count=10,
            )
        except OfficialSiteSearchError as exc:
            return SourceDiscoveryResult(
                status=RetrievalStatus.FAILED,
                errors=[str(exc)],
            )
        ranked = _rank_official_candidates(search_results, query)
        if not ranked:
            return SourceDiscoveryResult(
                status=RetrievalStatus.NO_RESULT,
                warnings=["Web Search 未发现满足身份信号要求的研究室官网"],
            )
        top_score = ranked[0][0]
        contenders = [item for item in ranked if item[0] >= top_score - 1]
        if len(contenders) > 1:
            return SourceDiscoveryResult(
                status=RetrievalStatus.NEEDS_USER_CONFIRMATION,
                sources=[item[1] for item in contenders[:3]],
                warnings=["发现多个接近的研究室官网候选，需要用户确认后再抓取"],
            )
        return SourceDiscoveryResult(
            status=RetrievalStatus.SUCCESS,
            sources=[ranked[0][1]],
        )


# 解析 Brave Search API 响应中的网页结果列表
def _parse_search_results(payload: Any) -> list[BraveSearchResult]:
    if not isinstance(payload, Mapping):
        raise BraveSearchError("Brave Search response root must be an object")
    web = payload.get("web", {})
    if web is None:
        return []
    if not isinstance(web, Mapping):
        raise BraveSearchError("Brave Search web field must be an object")
    raw_results = web.get("results", [])
    if not isinstance(raw_results, list):
        raise BraveSearchError("Brave Search results field must be a list")
    results: list[BraveSearchResult] = []
    for item in raw_results:
        if not isinstance(item, Mapping):
            continue
        title = item.get("title")
        url = item.get("url")
        description = item.get("description", "")
        if (
            isinstance(title, str)
            and title.strip()
            and isinstance(url, str)
            and _is_http_url(url)
            and isinstance(description, str)
        ):
            results.append(
                BraveSearchResult(
                    title=_strip_html(title),
                    url=url,
                    description=_strip_html(description),
                )
            )
    return results


# 构造包含教授、大学、研究科和可选研究室的官网发现查询
def _build_discovery_query(query: PublicationSearchQuery) -> str:
    terms = [
        f'"{query.professor_name}"',
        f'"{query.university_name}"',
    ]
    if query.laboratory_name is not None:
        terms.append(f'"{query.laboratory_name}"')
    else:
        terms.append(f'"{query.graduate_school_name}"')
    terms.append("研究室 laboratory lab")
    return " ".join(terms)


# 按身份信号和高校域名特征排序官网候选
def _rank_official_candidates(
    results: list[BraveSearchResult],
    query: PublicationSearchQuery,
) -> list[tuple[int, DiscoveredSource]]:
    ranked: list[tuple[int, DiscoveredSource]] = []
    seen_hosts: set[str] = set()
    for result in results:
        parsed = urlparse(result.url)
        hostname = (parsed.hostname or "").casefold()
        if not hostname or _is_non_official_host(hostname):
            continue
        if parsed.path.casefold().endswith((".pdf", ".doc", ".docx")):
            continue
        searchable = " ".join(
            [
                result.title,
                result.description,
                hostname,
                parsed.path,
            ]
        )
        signals: list[str] = []
        professor_names = [
            query.professor_name,
            *query.professor_name_variants,
        ]
        if _contains_any_text(searchable, professor_names):
            signals.append("教授姓名")
        if _contains_text(searchable, query.university_name):
            signals.append("大学名称")
        if _contains_text(searchable, query.graduate_school_name):
            signals.append("研究科名称")
        if query.laboratory_name is not None and _contains_text(
            searchable, query.laboratory_name
        ):
            signals.append("研究室名称")
        if "教授姓名" not in signals or not ({"大学名称", "研究室名称"} & set(signals)):
            continue
        score = 3 * len(signals)
        if hostname.endswith((".ac.jp", ".edu", ".edu.jp")):
            score += 2
        if _looks_like_publication_page(result.url, result.title):
            score += 1
        if hostname in seen_hosts:
            score -= 1
        seen_hosts.add(hostname)
        ranked.append(
            (
                score,
                DiscoveredSource(
                    url=result.url,
                    source_type=_classify_source(result.url, result.title, query),
                    discovery_method=SourceDiscoveryMethod.WEB_SEARCH,
                    verification_status=EvidenceVerificationStatus.PARTIAL,
                    title=result.title,
                    matched_signals=signals,
                ),
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1].url))
    return ranked


# 根据页面名称和 URL 判断候选来源类型
def _classify_source(
    url: str,
    title: str,
    query: PublicationSearchQuery,
) -> DiscoveredSourceType:
    if _looks_like_publication_page(url, title):
        return DiscoveredSourceType.PUBLICATION_LIST
    if query.laboratory_name is not None and _contains_text(
        f"{url} {title}", query.laboratory_name
    ):
        return DiscoveredSourceType.LABORATORY_WEBSITE
    return DiscoveredSourceType.PROFESSOR_PROFILE


# 判断 URL 或标题是否具有论文列表页面特征
def _looks_like_publication_page(url: str, title: str) -> bool:
    normalized = _normalize_text(f"{url} {title}")
    return any(
        keyword in normalized
        for keyword in (
            "publication",
            "publications",
            "papers",
            "researchoutput",
            "業績",
            "論文",
            "研究成果",
        )
    )


# 判断主机是否属于已知的第三方学术或社交平台
def _is_non_official_host(hostname: str) -> bool:
    return any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in NON_OFFICIAL_HOST_SUFFIXES
    )


# 判断文本是否包含任一候选目标字符串
def _contains_any_text(value: str, targets: list[str]) -> bool:
    return any(_contains_text(value, target) for target in targets)


# 按 Unicode 规范化后的保守规则判断文本包含关系
def _contains_text(value: str, target: str) -> bool:
    normalized_target = _normalize_text(target)
    return bool(normalized_target) and normalized_target in _normalize_text(value)


# 统一文本以比较日文、英文、大小写和常见标点差异
def _normalize_text(value: str) -> str:
    return normalize_search_text(value)


# 移除 Brave 搜索摘要中的简单 HTML 标签
def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value).strip()


# 判断字符串是否为 HTTP 或 HTTPS URL
def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
        )
    except ValueError:
        return False
