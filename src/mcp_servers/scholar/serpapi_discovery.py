"""Use SerpApi Google organic results for official laboratory-site discovery."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from jmr.domain import (
    DiscoveredSource,
    DiscoveredSourceType,
    EvidenceVerificationStatus,
    RetrievalStatus,
    SourceDiscoveryMethod,
)

from .brave_discovery import (
    BraveOfficialSiteDiscoveryAdapter,
    BraveSearchResult,
    OfficialSiteSearchError,
    _contains_text,
    _is_http_url,
    _is_non_official_host,
    _rank_official_candidates,
    _strip_html,
)
from .japanese_text import to_japanese_spelling
from .models import SourceDiscoveryResult
from .provider import PublicationSearchQuery

SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
SERPAPI_TIMEOUT_SECONDS = 45.0
SERPAPI_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class SerpApiSearchError(OfficialSiteSearchError):
    """Represent a SerpApi failure without revealing the API key."""


class SerpApiWebSearchClient:
    """Call the same SerpApi Google engine as search_tool.py, retaining result URLs."""

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = SERPAPI_TIMEOUT_SECONDS,
        max_response_bytes: int = SERPAPI_MAX_RESPONSE_BYTES,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("SerpApi API key must be non-empty")
        if timeout_seconds <= 0 or max_response_bytes <= 0:
            raise ValueError("SerpApi request limits must be positive")
        self._api_key = api_key.strip()
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes

    def search(self, query: str, count: int = 10) -> list[BraveSearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("SerpApi search query must be non-empty")
        if not 1 <= count <= 20:
            raise ValueError("SerpApi search count must be between 1 and 20")
        request = Request(
            f"{SERPAPI_SEARCH_URL}?"
            + urlencode(
                {
                    "engine": "google",
                    "q": normalized_query,
                    "api_key": self._api_key,
                    "gl": "jp",
                    "hl": "ja",
                    "num": count,
                    "output": "json",
                }
            ),
            headers={
                "Accept": "application/json",
                "User-Agent": "Japan-Master-Researcher/0.1",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = response.read(self._max_response_bytes + 1)
        except HTTPError as exc:
            raise SerpApiSearchError(f"SerpApi returned HTTP {exc.code}") from None
        except (TimeoutError, URLError):
            raise SerpApiSearchError("SerpApi request could not be completed") from None
        if len(payload) > self._max_response_bytes:
            raise SerpApiSearchError("SerpApi response exceeded the size limit")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SerpApiSearchError("SerpApi returned invalid JSON") from None
        return _parse_organic_results(decoded)


class SerpApiOfficialSiteDiscoveryAdapter(BraveOfficialSiteDiscoveryAdapter):
    """Search broadly; require confirmation for weaker lab-name identity clues."""

    def discover(self, query: PublicationSearchQuery) -> SourceDiscoveryResult:
        try:
            results = self._search_client.search(
                f"{query.professor_name} {to_japanese_spelling(query.university_name)}",
                count=10,
            )
        except OfficialSiteSearchError as exc:
            return SourceDiscoveryResult(
                status=RetrievalStatus.FAILED,
                errors=[str(exc)],
            )
        ranked = _rank_official_candidates(
            [result for result in results if _looks_like_target_site(result, query)],
            query,
        )
        if ranked:
            top_score = ranked[0][0]
            contenders = [item for item in ranked if item[0] >= top_score - 1]
            selected_host = urlsplit(ranked[0][1].url).hostname or ""
            academic_host = selected_host.endswith((".ac.jp", ".edu", ".edu.jp"))
            if len(contenders) > 1 or not academic_host:
                return SourceDiscoveryResult(
                    status=RetrievalStatus.NEEDS_USER_CONFIRMATION,
                    sources=[item[1] for item in contenders[:3]],
                    warnings=[
                        "发现多个接近的候选或非高校域名网站，需要用户确认后再抓取"
                    ],
                )
            return SourceDiscoveryResult(
                status=RetrievalStatus.SUCCESS,
                sources=[ranked[0][1]],
            )
        partial = _partial_lab_candidates(results, query)
        if partial:
            return SourceDiscoveryResult(
                status=RetrievalStatus.NEEDS_USER_CONFIRMATION,
                sources=partial[:3],
                warnings=[
                    "搜索结果仅匹配教授姓氏及学校，尚未核实教授全名；"
                    "请确认研究室官网后再抓取"
                ],
            )
        return SourceDiscoveryResult(
            status=RetrievalStatus.NO_RESULT,
            warnings=["SerpApi 未发现满足身份信号要求的研究室官网"],
        )


def _looks_like_target_site(
    result: BraveSearchResult, query: PublicationSearchQuery
) -> bool:
    """Reject generic link lists, news items, and individual paper pages."""

    title = result.title
    if (
        query.laboratory_name
        and _contains_text(title, query.laboratory_name)
        and any(
            _contains_text(title, marker) for marker in ("研究室", "laboratory", " lab")
        )
    ):
        return True
    if any(
        _contains_text(title, name)
        for name in (query.professor_name, *query.professor_name_variants)
    ):
        return True
    return any(
        _contains_text(title, f"{surname}研究室")
        or _contains_text(title, f"{surname} lab")
        or _contains_text(title, f"{surname} laboratory")
        for surname in _name_prefixes(query.professor_name)
    )


def _name_prefixes(professor_name: str) -> list[str]:
    name = professor_name.replace(" ", "").replace("　", "")
    if len(name) < 3:
        return []
    return [name[:2]] if len(name) >= 4 else [name[:2], name[:1]]


def _partial_lab_candidates(
    results: list[BraveSearchResult], query: PublicationSearchQuery
) -> list[DiscoveredSource]:
    """Offer surname-lab matches as human-reviewed candidates, never auto-accept."""

    surnames = _name_prefixes(query.professor_name)
    if not surnames:
        return []
    candidates: list[DiscoveredSource] = []
    seen_hosts: set[str] = set()
    for result in results:
        parsed = urlsplit(result.url)
        hostname = (parsed.hostname or "").casefold()
        if not hostname or hostname in seen_hosts or _is_non_official_host(hostname):
            continue
        searchable = f"{result.title} {result.description}"
        if not _contains_text(searchable, query.university_name):
            continue
        if not any(
            _contains_text(result.title, f"{surname}研究室")
            or _contains_text(result.title, f"{surname} lab")
            or _contains_text(result.title, f"{surname} laboratory")
            for surname in surnames
        ):
            continue
        if hostname.endswith((".ac.jp", ".edu", ".edu.jp")):
            candidate_url = result.url
        else:
            candidate_url = urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))
        candidates.append(
            DiscoveredSource(
                url=candidate_url,
                source_type=DiscoveredSourceType.LABORATORY_WEBSITE,
                discovery_method=SourceDiscoveryMethod.WEB_SEARCH,
                verification_status=EvidenceVerificationStatus.PARTIAL,
                title=result.title,
                matched_signals=["教授姓氏", "大学名称", "研究室标题"],
            )
        )
        seen_hosts.add(hostname)
    return candidates


def _parse_organic_results(payload: Any) -> list[BraveSearchResult]:
    if not isinstance(payload, Mapping):
        raise SerpApiSearchError("SerpApi response root must be an object")
    if payload.get("error"):
        raise SerpApiSearchError("SerpApi search failed; check key and quota")
    raw_results = payload.get("organic_results", [])
    if not isinstance(raw_results, list):
        raise SerpApiSearchError("SerpApi organic_results must be a list")
    results: list[BraveSearchResult] = []
    for item in raw_results:
        if not isinstance(item, Mapping):
            continue
        title = item.get("title")
        url = item.get("link")
        snippet = item.get("snippet", "")
        if (
            isinstance(title, str)
            and title.strip()
            and isinstance(url, str)
            and _is_http_url(url)
        ):
            results.append(
                BraveSearchResult(
                    title=_strip_html(title),
                    url=url,
                    description=(
                        _strip_html(snippet) if isinstance(snippet, str) else ""
                    ),
                )
            )
    return results
