# File: verification.py
# Author: L1nzhk0
# Purpose: 正式 JMR 领域中的确定性证据合并与核验算法。

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable
from typing import Any

from .evidence import (
    EvidenceType,
    EvidenceVerificationStatus,
    ResearchEvidence,
    RetrievalResult,
    RetrievalStatus,
    VerifiedEvidenceBundle,
    VerifiedEvidenceRecord,
)


# 合并论文和 KAKEN 检索批次并返回独立核验结果
def merge_and_verify_retrieval_results(
    retrieval_results: list[RetrievalResult],
    source_artifact_refs: list[str],
) -> VerifiedEvidenceBundle:
    if not isinstance(retrieval_results, list) or not all(
        isinstance(result, RetrievalResult) for result in retrieval_results
    ):
        raise TypeError("retrieval_results must contain RetrievalResult items")
    if len(retrieval_results) != len(source_artifact_refs):
        raise ValueError("retrieval_results and source_artifact_refs must align")

    records = [record for result in retrieval_results for record in result.records]
    groups = _group_equivalent_records(records)
    verified_records = [_merge_record_group(group) for group in groups]
    verified_records.sort(key=_verified_record_sort_key)
    warnings = _collect_bundle_warnings(
        retrieval_results,
        verified_records,
    )
    return VerifiedEvidenceBundle(
        bundle_id=_build_bundle_id(source_artifact_refs),
        source_artifact_refs=list(source_artifact_refs),
        records=verified_records,
        warnings=warnings,
    )


# 按强标识符或严格标题信息把同一证据划分为合并组
def _group_equivalent_records(
    records: list[ResearchEvidence],
) -> list[list[ResearchEvidence]]:
    groups: list[list[ResearchEvidence]] = []
    for record in sorted(records, key=_raw_record_sort_key):
        matching_group = next(
            (
                group
                for group in groups
                if all(_records_are_equivalent(record, existing) for existing in group)
            ),
            None,
        )
        if matching_group is None:
            groups.append([record])
        else:
            matching_group.append(record)
    return groups


# 判断两条相同类型证据是否具有足够强的一致标识
def _records_are_equivalent(
    left: ResearchEvidence,
    right: ResearchEvidence,
) -> bool:
    if left.evidence_type != right.evidence_type:
        return False
    identifier_name = (
        "doi" if left.evidence_type == EvidenceType.PUBLICATION else "kaken_project_id"
    )
    left_identifier = _normalized_identifier(
        left.identifiers.get(identifier_name),
        identifier_name,
    )
    right_identifier = _normalized_identifier(
        right.identifiers.get(identifier_name),
        identifier_name,
    )
    if left_identifier and right_identifier:
        return left_identifier == right_identifier
    if _normalize_text(left.title) != _normalize_text(right.title):
        return False
    if not _years_are_compatible(left, right):
        return False
    return _professors_are_compatible(left, right)


# 将一个重复证据组整理成单条带来源追踪的规范化记录
def _merge_record_group(
    group: list[ResearchEvidence],
) -> VerifiedEvidenceRecord:
    ordered = sorted(group, key=_canonical_record_priority)
    conflicts = _collect_record_conflicts(ordered)
    status, verification_reasons = _resolve_verification_status(
        ordered,
        conflicts,
    )
    merged_payload = _merge_record_payloads(ordered)
    source_evidence_ids = sorted({record.evidence_id for record in ordered})
    merged_payload["evidence_id"] = _build_canonical_evidence_id(
        ordered[0].evidence_type,
        source_evidence_ids,
    )
    merged_payload["verification_status"] = status.value
    canonical = ResearchEvidence.from_dict(merged_payload)
    return VerifiedEvidenceRecord(
        evidence=canonical,
        source_evidence_ids=source_evidence_ids,
        source_names=sorted({record.source_name for record in ordered}),
        merge_reasons=[_merge_reason(ordered)],
        verification_reasons=verification_reasons,
        conflicts=conflicts,
    )


# 选择高质量记录为主记录并只用其他来源补齐缺失字段
def _merge_record_payloads(
    records: list[ResearchEvidence],
) -> dict[str, Any]:
    payload = records[0].to_dict()
    optional_scalar_fields = (
        "publication_date",
        "venue",
        "source_url",
        "abstract_or_summary",
        "source_citation",
        "alternate_title",
        "project_start_year",
        "project_end_year",
        "project_status",
        "matched_professor",
        "matched_professor_position",
    )
    list_fields = (
        "authors",
        "affiliations",
        "participant_roles",
        "matched_professor_affiliations",
    )
    for record in records[1:]:
        candidate = record.to_dict()
        for field_name in optional_scalar_fields:
            if payload[field_name] is None and candidate[field_name] is not None:
                payload[field_name] = candidate[field_name]
        for field_name in list_fields:
            if not payload[field_name] and candidate[field_name]:
                payload[field_name] = candidate[field_name]
        for key, value in candidate["identifiers"].items():
            payload["identifiers"].setdefault(key, value)
    if payload["participant_roles"] and len(payload["participant_roles"]) != len(
        payload["authors"]
    ):
        payload["participant_roles"] = []
    position = payload["matched_professor_position"]
    if position is not None and position > len(payload["authors"]):
        payload["matched_professor_position"] = None
    return payload


# 根据原始核验状态和字段冲突计算最终核验等级
def _resolve_verification_status(
    records: list[ResearchEvidence],
    conflicts: list[str],
) -> tuple[EvidenceVerificationStatus, list[str]]:
    usable_records = [
        record
        for record in records
        if record.verification_status != EvidenceVerificationStatus.REJECTED
    ]
    if not usable_records:
        return (
            EvidenceVerificationStatus.REJECTED,
            ["所有来源均已把该记录判定为拒绝"],
        )
    has_verified_source = any(
        record.verification_status == EvidenceVerificationStatus.VERIFIED
        for record in usable_records
    )
    source_families = {_source_family(record.source_name) for record in usable_records}
    reasons: list[str] = []
    if has_verified_source:
        reasons.append("至少一个原始来源已完成姓名与机构核验")
    if len(source_families) > 1:
        reasons.append("多个独立来源对同一记录形成交叉印证")
    if conflicts:
        reasons.append("来源间仍有字段冲突，因此降级为部分核验")
        return EvidenceVerificationStatus.PARTIAL, reasons
    if has_verified_source:
        return EvidenceVerificationStatus.VERIFIED, reasons
    reasons.append("只有部分核验来源，后续引用必须保留限制说明")
    return EvidenceVerificationStatus.PARTIAL, reasons


# 收集同一证据组内会影响可靠性的关键字段冲突
def _collect_record_conflicts(
    records: list[ResearchEvidence],
) -> list[str]:
    conflicts: list[str] = []
    normalized_titles = {_normalize_text(record.title) for record in records}
    if len(normalized_titles) > 1:
        conflicts.append("同一强标识符对应的题名不一致")
    years = {year for record in records if (year := _evidence_year(record)) is not None}
    if len(years) > 1:
        conflicts.append("同一强标识符对应的年份不一致")
    professors = {
        _normalize_text(record.matched_professor)
        for record in records
        if record.matched_professor
    }
    if len(professors) > 1:
        conflicts.append("不同来源匹配到的教授姓名不一致")

    identifier_values: dict[str, set[str]] = {}
    for record in records:
        for name, value in record.identifiers.items():
            identifier_values.setdefault(name.casefold(), set()).add(
                _normalized_identifier(value, name)
            )
    for name, values in sorted(identifier_values.items()):
        if len(values) > 1:
            conflicts.append(f"标识符 {name} 在不同来源中不一致")
    return conflicts


# 汇总检索降级信息和核验结果中的非阻断警告
def _collect_bundle_warnings(
    retrieval_results: list[RetrievalResult],
    records: list[VerifiedEvidenceRecord],
) -> list[str]:
    warnings: list[str] = []
    for result in retrieval_results:
        if result.status in {
            RetrievalStatus.PARTIAL,
            RetrievalStatus.NO_RESULT,
            RetrievalStatus.BLOCKED,
            RetrievalStatus.FAILED,
        }:
            warnings.append(
                f"{result.evidence_type.value} 检索以 {result.status.value} 状态完成"
            )
        warnings.extend(result.warnings)
        warnings.extend(f"{result.source_name}: {error}" for error in result.errors)
    if not records:
        warnings.append("论文与 KAKEN 检索结果中没有可供核验的证据")
    if any(
        record.evidence.verification_status == EvidenceVerificationStatus.PARTIAL
        for record in records
    ):
        warnings.append("部分证据只能带限制说明引用")
    if any(record.conflicts for record in records):
        warnings.append("部分合并记录存在来源字段冲突")
    return list(dict.fromkeys(warnings))


# 返回一个证据组采用的确定性合并依据
def _merge_reason(records: list[ResearchEvidence]) -> str:
    if len(records) == 1:
        return "单一记录，无需跨来源去重"
    identifier_name = (
        "doi"
        if records[0].evidence_type == EvidenceType.PUBLICATION
        else "kaken_project_id"
    )
    identifiers = {
        _normalized_identifier(
            record.identifiers.get(identifier_name),
            identifier_name,
        )
        for record in records
        if record.identifiers.get(identifier_name)
    }
    if len(identifiers) == 1:
        return f"{identifier_name} 相同"
    return "规范化题名、年份和目标教授一致"


# 判断两条记录的年份信息是否相容
def _years_are_compatible(
    left: ResearchEvidence,
    right: ResearchEvidence,
) -> bool:
    left_year = _evidence_year(left)
    right_year = _evidence_year(right)
    return left_year is None or right_year is None or left_year == right_year


# 判断两条记录匹配到的教授姓名是否相容
def _professors_are_compatible(
    left: ResearchEvidence,
    right: ResearchEvidence,
) -> bool:
    if not left.matched_professor or not right.matched_professor:
        return True
    return _normalize_text(left.matched_professor) == _normalize_text(
        right.matched_professor
    )


# 返回论文发表年份或 KAKEN 课题起始年份
def _evidence_year(record: ResearchEvidence) -> int | None:
    if record.evidence_type == EvidenceType.KAKEN_PROJECT:
        return record.project_start_year
    if record.publication_date is None:
        return None
    return int(record.publication_date[:4])


# 清理 DOI 等标识符前缀并生成可比较文本
def _normalized_identifier(value: str | None, name: str) -> str:
    if value is None:
        return ""
    normalized = value.strip().casefold()
    if name.casefold() == "doi":
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break
    return normalized


# 将题名和姓名规范化为适合精确比较的字符序列
def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


# 将来源名称归并为官网、OpenAlex、KAKEN 或其他来源族
def _source_family(source_name: str) -> str:
    normalized = source_name.casefold()
    if "official-site" in normalized:
        return "official-site"
    if "openalex" in normalized:
        return "openalex"
    if "kaken" in normalized:
        return "kaken"
    return normalized


# 按原始核验等级、来源权威性和信息完整度选择主记录
def _canonical_record_priority(
    record: ResearchEvidence,
) -> tuple[int, int, int, str]:
    verification_priority = {
        EvidenceVerificationStatus.VERIFIED: 0,
        EvidenceVerificationStatus.PARTIAL: 1,
        EvidenceVerificationStatus.UNVERIFIED: 2,
        EvidenceVerificationStatus.REJECTED: 3,
    }[record.verification_status]
    source_priority = {
        "official-site": 0,
        "kaken": 0,
        "openalex": 1,
    }.get(_source_family(record.source_name), 2)
    completeness = sum(
        value not in (None, "", [], {}) for value in record.to_dict().values()
    )
    return (
        verification_priority,
        source_priority,
        -completeness,
        record.evidence_id,
    )


# 为分组前的原始证据提供稳定排序键
def _raw_record_sort_key(record: ResearchEvidence) -> tuple[str, str, str]:
    return (
        record.evidence_type.value,
        _normalize_text(record.title),
        record.evidence_id,
    )


# 为整合后证据提供类型、年份和题名稳定排序键
def _verified_record_sort_key(
    record: VerifiedEvidenceRecord,
) -> tuple[str, int, str]:
    year = _evidence_year(record.evidence) or 0
    return (
        record.evidence.evidence_type.value,
        -year,
        _normalize_text(record.evidence.title),
    )


# 根据来源证据 ID 生成稳定的规范化证据 ID
def _build_canonical_evidence_id(
    evidence_type: EvidenceType,
    source_evidence_ids: Iterable[str],
) -> str:
    identity = "\0".join(sorted(source_evidence_ids))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    prefix = "pub" if evidence_type == EvidenceType.PUBLICATION else "kaken"
    return f"verified_{prefix}_{digest}"


# 根据两条当前检索 Artifact 引用生成可重复执行的核验批次 ID
def _build_bundle_id(source_artifact_refs: Iterable[str]) -> str:
    identity = "\0".join(source_artifact_refs)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"verification_{digest}"
