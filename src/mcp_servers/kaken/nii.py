# File: nii.py
# Author: L1nzhk0
# Purpose: 通过国立情报学研究所 KAKEN OpenSearch XML API
# 检索并规范化科研课题。

from __future__ import annotations

import hashlib
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from jmr.domain import (
    EvidenceType,
    EvidenceVerificationStatus,
    ResearchEvidence,
    RetrievalStatus,
)

from .provider import (
    KakenSearchProviderResult,
    KakenSearchQuery,
)

KAKEN_API_URL = "https://kaken.nii.ac.jp/opensearch/"
KAKEN_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
KAKEN_TIMEOUT_SECONDS = 20.0
_YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


class KakenRequestError(RuntimeError):
    """表示 KAKEN API 请求失败或返回了无效 XML。"""


class XMLHTTPClient(Protocol):
    """定义 KAKEN Provider 所需的最小 XML GET 接口。"""

    # 请求并解析 XML 根元素
    def get_xml(
        self,
        url: str,
        params: Mapping[str, str | int],
    ) -> ET.Element: ...


class UrllibXMLHTTPClient:
    """使用 Python 标准库访问固定的 KAKEN HTTPS API。"""

    # 初始化 KAKEN HTTP Client 的超时和响应大小限制
    def __init__(
        self,
        timeout_seconds: float = KAKEN_TIMEOUT_SECONDS,
        max_response_bytes: int = KAKEN_MAX_RESPONSE_BYTES,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes

    # 请求固定 KAKEN 主机并解析受大小限制的 XML
    def get_xml(
        self,
        url: str,
        params: Mapping[str, str | int],
    ) -> ET.Element:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "kaken.nii.ac.jp":
            raise KakenRequestError("KAKEN client only allows https://kaken.nii.ac.jp")
        request = Request(
            f"{url}?{urlencode(params)}",
            headers={
                "Accept": "application/xml,text/xml",
                "User-Agent": "Japan-Master-Researcher/0.1",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = response.read(self._max_response_bytes + 1)
        except HTTPError as exc:
            raise KakenRequestError(f"KAKEN API returned HTTP {exc.code}") from exc
        except (TimeoutError, URLError) as exc:
            raise KakenRequestError("KAKEN API request could not be completed") from exc
        if len(payload) > self._max_response_bytes:
            raise KakenRequestError(
                "KAKEN API response exceeded the configured size limit"
            )
        try:
            return ET.fromstring(payload)
        except ET.ParseError as exc:
            raise KakenRequestError("KAKEN API returned invalid XML") from exc


class NIIKakenSearchProvider:
    """通过 KAKEN 官方 OpenSearch API 检索并保守核验教授课题。"""

    # 初始化 KAKEN Provider 的 Application ID 和 HTTP Client
    def __init__(
        self,
        app_id: str | None,
        http_client: XMLHTTPClient | None = None,
    ) -> None:
        self._app_id = app_id.strip() if app_id and app_id.strip() else None
        self._http_client = http_client or UrllibXMLHTTPClient()

    # 返回 KAKEN 官方数据源名称
    @property
    def source_name(self) -> str:
        return "KAKEN (NII)"

    # 按教授姓名变体检索课题并执行时间、身份和机构核验
    def search(self, query: KakenSearchQuery) -> KakenSearchProviderResult:
        if self._app_id is None:
            return KakenSearchProviderResult(
                status=RetrievalStatus.BLOCKED,
                errors=[
                    "KAKEN_APP_ID 未配置；KAKEN 官方 OpenSearch API "
                    "要求 CiNii API Application ID"
                ],
            )

        records_by_id: dict[str, ResearchEvidence] = {}
        warnings: list[str] = []
        errors: list[str] = []
        for professor_name in _deduplicate_names(
            [
                query.professor_name,
                *query.professor_name_variants,
            ]
        ):
            try:
                root = self._http_client.get_xml(
                    KAKEN_API_URL,
                    self._request_parameters(query, professor_name),
                )
            except KakenRequestError as exc:
                errors.append(f"{professor_name}: {exc}")
                continue
            total_results = _total_result_count(root)
            grant_elements = list(_iter_local(root, {"grantAward"}))
            if total_results is not None and total_results > len(grant_elements):
                warnings.append(
                    f"KAKEN 对姓名 {professor_name} 报告 {total_results} 条结果，"
                    f"本次 API 页面只返回 {len(grant_elements)} 条；结果可能被截断"
                )
            for grant in grant_elements:
                record, record_warnings = _normalize_grant(grant, query)
                warnings.extend(record_warnings)
                if record is not None:
                    records_by_id.setdefault(record.evidence_id, record)

        records = sorted(
            records_by_id.values(),
            key=_record_sort_key,
        )
        available_count = len(records)
        records = records[: query.max_results]
        if available_count > query.max_results:
            warnings.append(
                f"KAKEN 命中 {available_count} 个课题，当前只返回 "
                f"{query.max_results} 个；该结果不代表完整课题清单"
            )
        status = _result_status(records, errors)
        return KakenSearchProviderResult(
            status=status,
            records=records,
            warnings=_deduplicate_strings(warnings),
            errors=_deduplicate_strings(errors),
        )

    # 构造不泄露 Application ID 的官方 API 查询参数
    def _request_parameters(
        self,
        query: KakenSearchQuery,
        professor_name: str,
    ) -> dict[str, str | int]:
        return {
            "appid": self._app_id or "",
            "format": "xml",
            "lang": "ja",
            "qg": professor_name,
            "s1": query.search_start_date.year,
            "s2": query.search_end_date.year,
            "rw": _supported_page_size(query.max_results),
        }


# 将单个 KAKEN grantAward 元素转换为统一研究证据
def _normalize_grant(
    grant: ET.Element,
    query: KakenSearchQuery,
) -> tuple[ResearchEvidence | None, list[str]]:
    warnings: list[str] = []
    award_number = _award_number(grant)
    japanese_title, english_title = _project_titles(grant)
    title = japanese_title or english_title
    if title is None:
        return None, [f"已跳过缺少课题标题的 KAKEN 记录：{award_number}"]
    start_year, end_year = _project_period(grant)
    if not _period_overlaps_query(start_year, end_year, query):
        return None, []

    members = _project_members(grant)
    matched_index = _matched_member_index(members, query)
    if matched_index is None:
        return None, [f"已跳过无法在课题成员中匹配目标教授的 KAKEN 记录：{title}"]
    matched_name, _, matched_affiliations = members[matched_index]
    institution_match = _affiliations_match_target(
        matched_affiliations,
        query,
    )
    verification_status = (
        EvidenceVerificationStatus.VERIFIED
        if institution_match
        else EvidenceVerificationStatus.PARTIAL
    )
    if not institution_match:
        warnings.append(
            f"KAKEN 课题 {award_number} 已匹配教授姓名，但目标机构未能从该"
            "成员的课题机构信息中确认"
        )

    authors = [member[0] for member in members]
    roles = [member[1] for member in members]
    affiliations = _deduplicate_strings(
        [
            affiliation
            for _, _, member_affiliations in members
            for affiliation in member_affiliations
        ]
    )
    source_url = _project_url(grant, award_number)
    category = _first_text(grant, {"category", "researchCategory"})
    project_status = _project_status(grant)
    summary = _project_summary(grant)
    citation_period = _period_label(start_year, end_year)
    source_citation = f"{title} (KAKENHI {award_number}, {citation_period})"
    return ResearchEvidence(
        evidence_id=_evidence_id(award_number, title),
        evidence_type=EvidenceType.KAKEN_PROJECT,
        title=title,
        alternate_title=(
            english_title if english_title and english_title != title else None
        ),
        source_name="KAKEN (NII)",
        authors=authors,
        affiliations=affiliations,
        publication_date=str(start_year) if start_year is not None else None,
        venue=category,
        source_url=source_url,
        identifiers={"kaken_project_id": award_number},
        abstract_or_summary=summary,
        source_citation=source_citation,
        project_start_year=start_year,
        project_end_year=end_year,
        project_status=project_status,
        participant_roles=roles,
        matched_professor=matched_name,
        matched_professor_position=matched_index + 1,
        matched_professor_affiliations=matched_affiliations,
        verification_status=verification_status,
    ), warnings


# 从课题成员列表中匹配目标教授及其姓名变体
def _matched_member_index(
    members: list[tuple[str, str, list[str]]],
    query: KakenSearchQuery,
) -> int | None:
    target_names = {
        _normalize_text(query.professor_name),
        *(_normalize_text(value) for value in query.professor_name_variants),
    }
    for index, (name, _, _) in enumerate(members):
        if _normalize_text(name) in target_names:
            return index
    return None


# 提取课题成员姓名、角色和该成员的机构列表
def _project_members(
    grant: ET.Element,
) -> list[tuple[str, str, list[str]]]:
    members: list[tuple[str, str, list[str]]] = []
    for member in _iter_local(grant, {"member"}):
        name = _first_text_by_priority(
            member,
            ("fullName", "name", "personalName"),
        )
        if name is None:
            continue
        role = member.attrib.get("role") or _first_text(member, {"role"}) or "member"
        affiliations = _deduplicate_strings(
            [
                value
                for element in _iter_local(
                    member,
                    {"institution", "department", "affiliation"},
                )
                if (value := _element_text(element))
            ]
        )
        members.append((name, role, affiliations))
    return members


# 提取 KAKEN 课题的日文与英文题名
def _project_titles(grant: ET.Element) -> tuple[str | None, str | None]:
    localized: list[tuple[str, str]] = []
    for element in _iter_local(grant, {"title"}):
        value = _element_text(element)
        if value is None:
            continue
        language = (
            element.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
            or element.attrib.get("lang")
            or ""
        ).casefold()
        localized.append((language, value))
    japanese = next(
        (value for language, value in localized if language.startswith("ja")),
        None,
    )
    english = next(
        (value for language, value in localized if language.startswith("en")),
        None,
    )
    fallback = localized[0][1] if localized else None
    return japanese or fallback, english


# 提取课题编号并在缺失时返回稳定占位标识
def _award_number(grant: ET.Element) -> str:
    value = (
        grant.attrib.get("awardNumber")
        or _first_text(grant, {"awardNumber", "projectNumber"})
        or grant.attrib.get("id")
        or "unknown"
    )
    return value.removeprefix("KAKENHI-PROJECT-").strip()


# 提取课题起止年份
def _project_period(grant: ET.Element) -> tuple[int | None, int | None]:
    period = next(iter(_iter_local(grant, {"periodOfAward"})), None)
    if period is None:
        return None, None
    fragments = [*period.attrib.values()]
    text = _element_text(period)
    if text is not None:
        fragments.append(text)
    years = [
        int(value)
        for fragment in fragments
        for value in _YEAR_PATTERN.findall(fragment)
    ]
    if not years:
        return None, None
    return min(years), max(years)


# 判断课题年度与用户检索时间范围是否相交
def _period_overlaps_query(
    start_year: int | None,
    end_year: int | None,
    query: KakenSearchQuery,
) -> bool:
    if start_year is None and end_year is None:
        return True
    effective_start = start_year if start_year is not None else end_year
    effective_end = end_year if end_year is not None else start_year
    return (
        effective_start <= query.search_end_date.year
        and effective_end >= query.search_start_date.year
    )


# 判断成员机构是否匹配目标大学或研究科
def _affiliations_match_target(
    affiliations: list[str],
    query: KakenSearchQuery,
) -> bool:
    targets = [query.university_name, query.graduate_school_name]
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


# 提取课题官方 URL 或按编号构造官方详情页 URL
def _project_url(grant: ET.Element, award_number: str) -> str:
    for element in _iter_local(grant, {"url"}):
        value = _element_text(element)
        if value and value.startswith("https://kaken.nii.ac.jp/"):
            return value
    return f"https://kaken.nii.ac.jp/grant/KAKENHI-PROJECT-{award_number}/"


# 合并课题概要相关段落并限制单条证据文本大小
def _project_summary(grant: ET.Element) -> str | None:
    values = _deduplicate_strings(
        [
            value
            for element in _iter_local(
                grant,
                {"abstract", "paragraph", "plainText"},
            )
            if (value := _element_text(element))
        ]
    )
    if not values:
        return None
    return "\n".join(values)[:20_000]


# 提取课题状态文本或官方状态码属性
def _project_status(grant: ET.Element) -> str | None:
    for element in _iter_local(grant, {"projectStatus", "status"}):
        value = (
            _element_text(element)
            or element.attrib.get("statusCode")
            or element.attrib.get("code")
        )
        if value:
            return value.strip()
    return None


# 返回指定本地名称集合对应的全部 XML 后代元素
def _iter_local(
    root: ET.Element,
    local_names: set[str],
):
    for element in root.iter():
        if _local_name(element.tag) in local_names:
            yield element


# 从 XML 元素后代中读取第一个非空文本
def _first_text(root: ET.Element, local_names: set[str]) -> str | None:
    for element in _iter_local(root, local_names):
        value = _element_text(element)
        if value is not None:
            return value
    return None


# 按指定标签优先级读取第一个非空 XML 文本
def _first_text_by_priority(
    root: ET.Element,
    local_names: tuple[str, ...],
) -> str | None:
    for local_name in local_names:
        value = _first_text(root, {local_name})
        if value is not None:
            return value
    return None


# 合并 XML 元素的全部文本节点
def _element_text(element: ET.Element) -> str | None:
    value = " ".join("".join(element.itertext()).split())
    return value or None


# 去除 XML Namespace 并返回本地标签名
def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


# 从 OpenSearch 响应读取总结果数量
def _total_result_count(root: ET.Element) -> int | None:
    value = _first_text(root, {"totalResults"})
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


# 把请求数量提升到 KAKEN API 支持的分页大小
def _supported_page_size(max_results: int) -> int:
    if max_results <= 20:
        return 20
    if max_results <= 50:
        return 50
    return 100


# 根据记录数量、核验状态和请求错误汇总检索状态
def _result_status(
    records: list[ResearchEvidence],
    errors: list[str],
) -> RetrievalStatus:
    if records:
        if errors or any(
            record.verification_status != EvidenceVerificationStatus.VERIFIED
            for record in records
        ):
            return RetrievalStatus.PARTIAL
        return RetrievalStatus.SUCCESS
    return RetrievalStatus.FAILED if errors else RetrievalStatus.NO_RESULT


# 构造 KAKEN 课题的稳定内部证据 ID
def _evidence_id(award_number: str, title: str) -> str:
    normalized_number = "".join(
        character for character in award_number if character.isalnum()
    )
    if normalized_number and award_number != "unknown":
        return f"kaken_{normalized_number}"
    digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:20]
    return f"kaken_{digest}"


# 生成课题期间的可读标签
def _period_label(
    start_year: int | None,
    end_year: int | None,
) -> str:
    if start_year is None and end_year is None:
        return "period unknown"
    if start_year == end_year or end_year is None:
        return str(start_year)
    if start_year is None:
        return str(end_year)
    return f"{start_year}-{end_year}"


# 按课题开始年份倒序和标题稳定排序
def _record_sort_key(record: ResearchEvidence) -> tuple[int, str]:
    return (
        -(record.project_start_year or 0),
        _normalize_text(record.title),
    )


# 统一姓名和机构文本以执行保守比较
def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


# 按规范化姓名去重并限制请求次数
def _deduplicate_names(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _normalize_text(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
        if len(result) >= 5:
            break
    return result


# 按大小写不敏感规则去除诊断字符串或机构中的重复项
def _deduplicate_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
