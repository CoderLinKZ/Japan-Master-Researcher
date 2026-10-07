# File: openalex.py
# Author: L1nzhk0
# Purpose: 本文件用于通过 OpenAlex 官方 API 完成作者消歧和论文候选检索。

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from jmr.domain import EvidenceVerificationStatus, RetrievalStatus

from .japanese_text import normalize_search_text, to_japanese_spelling
from .models import PublicationAdapterResult, RawPublicationCandidate
from .provider import PublicationSearchQuery

OPENALEX_API_BASE_URL = "https://api.openalex.org"
OPENALEX_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
OPENALEX_TIMEOUT_SECONDS = 15.0


class OpenAlexRequestError(RuntimeError):
    """表示 OpenAlex 请求失败或返回了无法解析的响应。"""


class JSONHTTPClient(Protocol):
    """定义 OpenAlex Adapter 所需的最小 JSON GET 接口。"""

    # 请求 JSON 对象并返回解析后的映射
    def get_json(
        self,
        url: str,
        params: Mapping[str, str | int],
    ) -> Mapping[str, Any]: ...


class UrllibJSONHTTPClient:
    """使用 Python 标准库执行受大小和超时限制的 OpenAlex 请求。"""

    # 初始化 OpenAlex HTTP Client 的超时和响应大小限制
    def __init__(
        self,
        timeout_seconds: float = OPENALEX_TIMEOUT_SECONDS,
        max_response_bytes: int = OPENALEX_MAX_RESPONSE_BYTES,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes

    # 向固定的 OpenAlex HTTPS 主机请求并解析 JSON 对象
    def get_json(
        self,
        url: str,
        params: Mapping[str, str | int],
    ) -> Mapping[str, Any]:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "api.openalex.org":
            raise OpenAlexRequestError(
                "OpenAlex client only allows https://api.openalex.org"
            )
        request_url = f"{url}?{urlencode(params)}"
        request = Request(
            request_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Japan-Master-Researcher/0.1",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = response.read(self._max_response_bytes + 1)
        except HTTPError as exc:
            if exc.code == 429:
                raise OpenAlexRequestError(
                    "OpenAlex returned HTTP 429; configure "
                    "OPENALEX_API_KEY or retry after the request budget resets"
                ) from exc
            raise OpenAlexRequestError(f"OpenAlex returned HTTP {exc.code}") from exc
        except (TimeoutError, URLError) as exc:
            raise OpenAlexRequestError(
                "OpenAlex request could not be completed"
            ) from exc
        if len(payload) > self._max_response_bytes:
            raise OpenAlexRequestError(
                "OpenAlex response exceeded the configured size limit"
            )
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenAlexRequestError("OpenAlex returned invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise OpenAlexRequestError("OpenAlex response root must be a JSON object")
        return decoded


@dataclass(frozen=True, slots=True)
class _ResolvedAuthor:
    """保存完成消歧后的 OpenAlex 作者身份。"""

    author_ids: tuple[str, ...]
    display_name: str
    verification_status: EvidenceVerificationStatus
    orcid: str | None = None


@dataclass(frozen=True, slots=True)
class _AuthorResolution:
    """保存作者消歧步骤的状态、身份与诊断信息。"""

    status: RetrievalStatus
    author: _ResolvedAuthor | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _AuthorMatch:
    """保存一个经过姓名与机构打分的 OpenAlex 作者档案。"""

    score: int
    university_match: bool
    author_id: str
    display_name: str
    orcid: str | None
    works_count: int


class OpenAlexPublicationAdapter:
    """通过 OpenAlex 作者搜索和 Works 接口返回结构化论文候选。"""

    # 初始化 OpenAlex Adapter 及可选 API Key
    def __init__(
        self,
        http_client: JSONHTTPClient | None = None,
        api_key: str | None = None,
    ) -> None:
        self._http_client = http_client or UrllibJSONHTTPClient()
        self._api_key = api_key.strip() if api_key and api_key.strip() else None

    # 返回该 Adapter 的稳定来源名称
    @property
    def source_name(self) -> str:
        return "OpenAlex"

    # 完成作者消歧并检索指定日期范围内的论文候选
    def search(
        self,
        query: PublicationSearchQuery,
    ) -> PublicationAdapterResult:
        try:
            authors_payload = self._request(
                "/authors",
                {
                    "search": query.professor_name,
                    "per_page": 50,
                    "select": (
                        "id,display_name,display_name_alternatives,"
                        "affiliations,last_known_institutions,orcid,"
                        "works_count"
                    ),
                },
            )
            resolution = self._resolve_author(authors_payload, query)
            university_id: str | None = None
            if (
                resolution.author is None
                or resolution.author.verification_status
                != EvidenceVerificationStatus.VERIFIED
            ):
                university_id = self._resolve_university_id(
                    to_japanese_spelling(query.university_name)
                )
                author_results = authors_payload.get("results", [])
                if not isinstance(author_results, list):
                    raise OpenAlexRequestError(
                        "OpenAlex authors response has invalid results"
                    )
                all_authors = list(author_results)
                for name in _deduplicate_strings(query.professor_name_variants)[:3]:
                    if _normalize_text(name) == _normalize_text(query.professor_name):
                        continue
                    variant_payload = self._request(
                        "/authors",
                        {
                            "search": name,
                            "per_page": 25,
                            "select": (
                                "id,display_name,display_name_alternatives,"
                                "affiliations,last_known_institutions,orcid,"
                                "works_count"
                            ),
                        },
                    )
                    variant_results = variant_payload.get("results", [])
                    if not isinstance(variant_results, list):
                        raise OpenAlexRequestError(
                            "OpenAlex authors response has invalid results"
                        )
                    all_authors.extend(variant_results)
                unique_authors = {
                    item.get("id"): item
                    for item in all_authors
                    if isinstance(item, Mapping) and isinstance(item.get("id"), str)
                }
                resolution = self._resolve_author(
                    {"results": list(unique_authors.values())},
                    query,
                    university_id=university_id,
                )
            if resolution.author is None:
                return PublicationAdapterResult(
                    status=resolution.status,
                    warnings=list(resolution.warnings),
                )
            if (
                resolution.author.verification_status
                != EvidenceVerificationStatus.VERIFIED
            ):
                return PublicationAdapterResult(
                    status=RetrievalStatus.NEEDS_USER_CONFIRMATION,
                    warnings=["OpenAlex 作者身份未能用目标大学核验，已停止读取其论文"],
                )
            if university_id is not None and resolution.author.orcid is not None:
                linked_payload = self._request(
                    "/authors",
                    {
                        "filter": f"orcid:{resolution.author.orcid}",
                        "per_page": 25,
                        "select": "id,orcid",
                    },
                )
                linked_results = linked_payload.get("results", [])
                if not isinstance(linked_results, list):
                    raise OpenAlexRequestError(
                        "OpenAlex ORCID response has invalid results"
                    )
                linked_ids = [
                    _extract_openalex_id(item.get("id"), "A")
                    for item in linked_results
                    if isinstance(item, Mapping)
                    and isinstance(item.get("orcid"), str)
                    and item["orcid"].strip().casefold() == resolution.author.orcid
                ]
                resolution = replace(
                    resolution,
                    author=replace(
                        resolution.author,
                        author_ids=tuple(
                            dict.fromkeys(
                                [
                                    *resolution.author.author_ids,
                                    *(item for item in linked_ids if item is not None),
                                ]
                            )
                        ),
                    ),
                )
            works_payload = self._request(
                "/works",
                {
                    "filter": (
                        "authorships.author.id:"
                        f"{'|'.join(resolution.author.author_ids)},"
                        "from_publication_date:"
                        f"{query.search_start_date.isoformat()},"
                        "to_publication_date:"
                        f"{query.search_end_date.isoformat()}"
                    ),
                    "sort": "publication_date:desc",
                    "per_page": query.max_results,
                    "select": (
                        "id,display_name,title,publication_date,doi,"
                        "authorships,primary_location,"
                        "abstract_inverted_index"
                    ),
                },
            )
            candidates, affiliation_conflicts = self._build_candidates(
                works_payload,
                resolution.author,
                query,
                university_id=university_id,
            )
        except OpenAlexRequestError as exc:
            return PublicationAdapterResult(
                status=RetrievalStatus.FAILED,
                errors=[str(exc)],
            )
        warnings = list(resolution.warnings)
        reported_work_count = _extract_result_count(works_payload)
        returned_work_count = _extract_returned_result_count(works_payload)
        if (
            reported_work_count is not None
            and reported_work_count > returned_work_count
        ):
            warnings.append(
                "OpenAlex 在检索范围内报告 "
                f"{reported_work_count} 条候选，当前响应仅包含前 "
                f"{returned_work_count} 条；该来源结果不代表完整论文清单"
            )
        if affiliation_conflicts:
            warnings.append(
                "已剔除 "
                f"{affiliation_conflicts} 条目标作者在该论文中的机构与"
                "目标大学明确冲突的 OpenAlex 同名污染记录"
            )
        if not candidates:
            return PublicationAdapterResult(
                status=RetrievalStatus.NO_RESULT,
                warnings=warnings,
            )
        status = (
            RetrievalStatus.SUCCESS
            if resolution.author.verification_status
            == EvidenceVerificationStatus.VERIFIED
            and all(
                candidate.verification_status == EvidenceVerificationStatus.VERIFIED
                for candidate in candidates
            )
            else RetrievalStatus.PARTIAL
        )
        return PublicationAdapterResult(
            status=status,
            candidates=candidates,
            warnings=warnings,
        )

    # 向 OpenAlex 固定 API 端点发送带可选 API Key 的请求
    def _request(
        self,
        path: str,
        params: dict[str, str | int],
    ) -> Mapping[str, Any]:
        request_params = dict(params)
        if self._api_key is not None:
            request_params["api_key"] = self._api_key
        return self._http_client.get_json(
            f"{OPENALEX_API_BASE_URL}{path}",
            request_params,
        )

    # 使用教授姓名变体和机构信息消除 OpenAlex 作者同名歧义
    def _resolve_author(
        self,
        payload: Mapping[str, Any],
        query: PublicationSearchQuery,
        *,
        university_id: str | None = None,
    ) -> _AuthorResolution:
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise OpenAlexRequestError("OpenAlex authors response has invalid results")
        target_names = [
            query.professor_name,
            *query.professor_name_variants,
        ]
        matches: list[_AuthorMatch] = []
        for item in raw_results:
            if not isinstance(item, Mapping):
                continue
            display_name = item.get("display_name")
            if not isinstance(display_name, str) or not display_name.strip():
                continue
            author_names = [display_name]
            alternatives = item.get("display_name_alternatives", [])
            if isinstance(alternatives, list):
                author_names.extend(
                    value
                    for value in alternatives
                    if isinstance(value, str) and value.strip()
                )
            if not _contains_matching_name(author_names, target_names):
                continue
            institution_names = _extract_author_institutions(item)
            university_match = _contains_text_match(
                institution_names,
                query.university_name,
            ) or _author_has_institution_id(item, university_id)
            graduate_school_match = _contains_text_match(
                institution_names,
                query.graduate_school_name,
            )
            score = 4 * int(university_match) + 2 * int(graduate_school_match)
            author_id = _extract_openalex_id(item.get("id"), "A")
            if author_id is None:
                continue
            raw_orcid = item.get("orcid")
            orcid = (
                raw_orcid.strip().casefold()
                if isinstance(raw_orcid, str) and raw_orcid.strip()
                else None
            )
            raw_works_count = item.get("works_count", 0)
            works_count = (
                raw_works_count
                if isinstance(raw_works_count, int)
                and not isinstance(raw_works_count, bool)
                and raw_works_count >= 0
                else 0
            )
            matches.append(
                _AuthorMatch(
                    score=score,
                    university_match=university_match,
                    author_id=author_id,
                    display_name=display_name,
                    orcid=orcid,
                    works_count=works_count,
                )
            )
        if not matches:
            return _AuthorResolution(status=RetrievalStatus.NO_RESULT)
        matches.sort(key=lambda value: value.score, reverse=True)
        best_score = matches[0].score
        best_matches = [item for item in matches if item.score == best_score]
        if len(best_matches) > 1:
            merged_resolution = _merge_fragmented_author_matches(best_matches)
            if merged_resolution is not None:
                return merged_resolution
            candidate_names = ", ".join(
                f"{item.display_name} ({item.author_id})" for item in best_matches[:5]
            )
            return _AuthorResolution(
                status=RetrievalStatus.NEEDS_USER_CONFIRMATION,
                warnings=(
                    "OpenAlex 中存在多个同等匹配的作者身份，需要用户确认："
                    f"{candidate_names}",
                ),
            )
        selected = best_matches[0]
        verification_status = (
            EvidenceVerificationStatus.VERIFIED
            if selected.university_match
            else EvidenceVerificationStatus.PARTIAL
        )
        warnings: tuple[str, ...] = ()
        if not selected.university_match:
            warnings = ("作者姓名在 OpenAlex 中唯一匹配，但未能用目标大学完成机构核验",)
        return _AuthorResolution(
            status=(
                RetrievalStatus.SUCCESS
                if selected.university_match
                else RetrievalStatus.PARTIAL
            ),
            author=_ResolvedAuthor(
                author_ids=(selected.author_id,),
                display_name=selected.display_name,
                verification_status=verification_status,
                orcid=selected.orcid,
            ),
            warnings=warnings,
        )

    # 将 OpenAlex Works 响应转换为内部原始论文候选
    def _build_candidates(
        self,
        payload: Mapping[str, Any],
        author: _ResolvedAuthor,
        query: PublicationSearchQuery,
        *,
        university_id: str | None = None,
    ) -> tuple[list[RawPublicationCandidate], int]:
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise OpenAlexRequestError("OpenAlex works response has invalid results")
        candidates: list[RawPublicationCandidate] = []
        affiliation_conflicts = 0
        for item in raw_results:
            if not isinstance(item, Mapping):
                continue
            candidate, has_affiliation_conflict = _openalex_work_to_candidate(
                item, author, query, university_id=university_id
            )
            affiliation_conflicts += int(has_affiliation_conflict)
            if candidate is not None:
                candidates.append(candidate)
        return candidates, affiliation_conflicts

    def _resolve_university_id(self, name: str) -> str | None:
        payload = self._request(
            "/institutions",
            {
                "search": name,
                "per_page": 10,
                "select": "id,display_name,display_name_alternatives",
            },
        )
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise OpenAlexRequestError(
                "OpenAlex institutions response has invalid results"
            )
        matches: set[str] = set()
        for item in results:
            if not isinstance(item, Mapping):
                continue
            names = [item.get("display_name")]
            alternatives = item.get("display_name_alternatives")
            if isinstance(alternatives, list):
                names.extend(alternatives)
            if any(
                isinstance(candidate, str)
                and _normalize_text(candidate) == _normalize_text(name)
                for candidate in names
            ):
                institution_id = _extract_openalex_id(item.get("id"), "I")
                if institution_id is not None:
                    matches.add(institution_id)
        return next(iter(matches)) if len(matches) == 1 else None


# 使用唯一共享 ORCID 合并 OpenAlex 拆分出的同一作者档案
def _merge_fragmented_author_matches(
    matches: list[_AuthorMatch],
) -> _AuthorResolution | None:
    distinct_orcids = {match.orcid for match in matches if match.orcid is not None}
    if len(distinct_orcids) != 1:
        return None
    selected_orcid = next(iter(distinct_orcids))
    linked_matches = [match for match in matches if match.orcid == selected_orcid]
    if not linked_matches:
        return None
    linked_matches.sort(
        key=lambda match: (match.works_count, match.author_id),
        reverse=True,
    )
    primary = linked_matches[0]
    university_match = any(match.university_match for match in linked_matches)
    verification_status = (
        EvidenceVerificationStatus.VERIFIED
        if university_match
        else EvidenceVerificationStatus.PARTIAL
    )
    linked_ids = tuple(match.author_id for match in linked_matches)
    ignored_count = len(matches) - len(linked_matches)
    warning = (
        "OpenAlex 返回多个同分作者档案；已根据唯一共享 ORCID 合并"
        f"碎片档案：{', '.join(linked_ids)}"
    )
    if ignored_count:
        warning += f"，并忽略 {ignored_count} 个无法用 ORCID 关联的候选"
    return _AuthorResolution(
        status=(
            RetrievalStatus.SUCCESS if university_match else RetrievalStatus.PARTIAL
        ),
        author=_ResolvedAuthor(
            author_ids=linked_ids,
            display_name=primary.display_name,
            verification_status=verification_status,
            orcid=selected_orcid,
        ),
        warnings=(warning,),
    )


# 将一条 OpenAlex Work 转换为原始论文候选
def _openalex_work_to_candidate(
    item: Mapping[str, Any],
    author: _ResolvedAuthor,
    query: PublicationSearchQuery,
    *,
    university_id: str | None = None,
) -> tuple[RawPublicationCandidate | None, bool]:
    record_id = _extract_openalex_id(item.get("id"), "W")
    title = item.get("display_name") or item.get("title")
    if record_id is None or not isinstance(title, str) or not title.strip():
        return None, False
    (
        authors,
        affiliations,
        matched_affiliations,
        matched_position,
    ) = _extract_work_authorships(item, set(author.author_ids))
    if matched_position is None:
        return None, False
    institution_match = _contains_text_match(
        matched_affiliations,
        query.university_name,
    ) or _work_author_has_institution_id(item, set(author.author_ids), university_id)
    if matched_affiliations and not institution_match:
        return None, True
    doi = _normalize_doi(item.get("doi"))
    identifiers = {"openalex": record_id}
    if doi is not None:
        identifiers["doi"] = doi
    source_url, venue = _extract_primary_location(item, doi)
    publication_date = item.get("publication_date")
    if not isinstance(publication_date, str):
        publication_date = None
    matched_name = authors[matched_position - 1]
    return RawPublicationCandidate(
        source_record_id=record_id,
        title=title,
        source_name="OpenAlex",
        authors=authors,
        affiliations=affiliations,
        publication_date=publication_date,
        venue=venue,
        source_url=source_url,
        identifiers=identifiers,
        abstract_or_summary=_reconstruct_abstract(item.get("abstract_inverted_index")),
        matched_professor=matched_name,
        matched_professor_position=matched_position,
        matched_professor_affiliations=matched_affiliations,
        verification_status=(
            EvidenceVerificationStatus.VERIFIED
            if institution_match
            else EvidenceVerificationStatus.PARTIAL
        ),
    ), False


# 提取 Work 中的作者姓名和机构名称
def _extract_work_authorships(
    item: Mapping[str, Any],
    target_author_ids: set[str],
) -> tuple[list[str], list[str], list[str], int | None]:
    authors: list[str] = []
    affiliations: list[str] = []
    matched_affiliations: list[str] = []
    matched_position: int | None = None
    authorships = item.get("authorships", [])
    if not isinstance(authorships, list):
        return authors, affiliations, matched_affiliations, matched_position
    for authorship in authorships:
        if not isinstance(authorship, Mapping):
            continue
        author = authorship.get("author")
        is_target_author = False
        if isinstance(author, Mapping):
            name = author.get("display_name")
            if isinstance(name, str) and name.strip():
                authors.append(name.strip())
            author_id = _extract_openalex_id(author.get("id"), "A")
            is_target_author = author_id in target_author_ids
            if is_target_author and isinstance(name, str) and name.strip():
                matched_position = len(authors)
        institutions = authorship.get("institutions", [])
        if not isinstance(institutions, list):
            continue
        for institution in institutions:
            if not isinstance(institution, Mapping):
                continue
            name = institution.get("display_name")
            if isinstance(name, str) and name.strip():
                affiliations.append(name.strip())
                if is_target_author:
                    matched_affiliations.append(name.strip())
    return (
        authors,
        _deduplicate_strings(affiliations),
        _deduplicate_strings(matched_affiliations),
        matched_position,
    )


def _author_has_institution_id(
    author: Mapping[str, Any], institution_id: str | None
) -> bool:
    if institution_id is None:
        return False
    current = author.get("last_known_institutions", [])
    if isinstance(current, list) and any(
        isinstance(item, Mapping)
        and _extract_openalex_id(item.get("id"), "I") == institution_id
        for item in current
    ):
        return True
    affiliations = author.get("affiliations", [])
    return isinstance(affiliations, list) and any(
        isinstance(item, Mapping)
        and isinstance(item.get("institution"), Mapping)
        and _extract_openalex_id(item["institution"].get("id"), "I") == institution_id
        for item in affiliations
    )


def _work_author_has_institution_id(
    work: Mapping[str, Any], author_ids: set[str], institution_id: str | None
) -> bool:
    if institution_id is None:
        return False
    authorships = work.get("authorships", [])
    if not isinstance(authorships, list):
        return False
    for authorship in authorships:
        if not isinstance(authorship, Mapping):
            continue
        author = authorship.get("author")
        if (
            not isinstance(author, Mapping)
            or _extract_openalex_id(author.get("id"), "A") not in author_ids
        ):
            continue
        institutions = authorship.get("institutions", [])
        if isinstance(institutions, list) and any(
            isinstance(item, Mapping)
            and _extract_openalex_id(item.get("id"), "I") == institution_id
            for item in institutions
        ):
            return True
    return False


# 提取作者对象中的当前和历史机构名称
def _extract_author_institutions(item: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    current = item.get("last_known_institutions", [])
    if isinstance(current, list):
        names.extend(_extract_institution_names(current))
    affiliations = item.get("affiliations", [])
    if isinstance(affiliations, list):
        for affiliation in affiliations:
            if not isinstance(affiliation, Mapping):
                continue
            institution = affiliation.get("institution")
            if isinstance(institution, Mapping):
                name = institution.get("display_name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
    return _deduplicate_strings(names)


# 提取一组 OpenAlex 机构对象中的名称
def _extract_institution_names(items: list[Any]) -> list[str]:
    names: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = item.get("display_name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


# 提取 Work 的落地页链接和期刊或会议名称
def _extract_primary_location(
    item: Mapping[str, Any],
    doi: str | None,
) -> tuple[str, str | None]:
    source_url = f"https://doi.org/{doi}" if doi is not None else None
    venue: str | None = None
    primary_location = item.get("primary_location")
    if isinstance(primary_location, Mapping):
        landing_page_url = primary_location.get("landing_page_url")
        if isinstance(landing_page_url, str) and _is_http_url(landing_page_url):
            source_url = landing_page_url
        source = primary_location.get("source")
        if isinstance(source, Mapping):
            display_name = source.get("display_name")
            if isinstance(display_name, str) and display_name.strip():
                venue = display_name.strip()
    if source_url is None:
        raw_id = item.get("id")
        if isinstance(raw_id, str) and _is_http_url(raw_id):
            source_url = raw_id
    if source_url is None:
        raise OpenAlexRequestError("OpenAlex work is missing a usable source URL")
    return source_url, venue


# 从 OpenAlex 倒排索引重建摘要文本
def _reconstruct_abstract(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    positioned_words: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and position >= 0:
                positioned_words.append((position, word))
    if not positioned_words:
        return None
    positioned_words.sort(key=lambda item: item[0])
    abstract = " ".join(word for _, word in positioned_words).strip()
    return abstract[:10000] or None


# 从 OpenAlex 分页元数据读取命中总数
def _extract_result_count(payload: Mapping[str, Any]) -> int | None:
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        return None
    count = meta.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return count


# 统计 OpenAlex 当前响应实际包含的结果数量
def _extract_returned_result_count(payload: Mapping[str, Any]) -> int:
    results = payload.get("results")
    return len(results) if isinstance(results, list) else 0


# 提取并校验 OpenAlex 实体 ID
def _extract_openalex_id(value: Any, prefix: str) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.rstrip("/").rsplit("/", 1)[-1]
    if re.fullmatch(rf"{re.escape(prefix)}\d+", candidate):
        return candidate
    return None


# 将 DOI URL 或 DOI 字符串转换为统一标识符
def _normalize_doi(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.casefold().startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized.strip() or None


# 判断候选作者姓名集合是否与任一目标姓名严格匹配
def _contains_matching_name(
    candidate_names: list[str],
    target_names: list[str],
) -> bool:
    normalized_targets = {_normalize_text(name) for name in target_names}
    return any(_normalize_text(name) in normalized_targets for name in candidate_names)


# 判断机构名称集合是否包含目标机构名称
def _contains_text_match(values: list[str], target: str) -> bool:
    normalized_target = _normalize_text(target)
    if not normalized_target:
        return False
    for value in values:
        normalized_value = _normalize_text(value)
        if (
            normalized_target in normalized_value
            or normalized_value in normalized_target
        ):
            return True
    return False


# 统一姓名和机构文本以执行保守的精确比较
def _normalize_text(value: str) -> str:
    return normalize_search_text(value)


# 按大小写不敏感规则去除字符串列表中的重复项
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


# 判断字符串是否是可用的 HTTP 或 HTTPS URL
def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
