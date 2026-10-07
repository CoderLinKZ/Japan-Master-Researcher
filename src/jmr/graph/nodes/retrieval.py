"""P4 research planning, parallel retrieval, interrupts, and verification."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from typing import Any
from urllib.parse import urlparse

from langgraph.runtime import Runtime
from langgraph.types import Send, interrupt

from jmr.domain import (
    EvidenceVerificationStatus,
    InterruptKind,
    MCPStatus,
    RetrievalResult,
    RunStatus,
    WorkflowStage,
)
from jmr.domain.verification import merge_and_verify_retrieval_results
from jmr.runtime import JMRRuntimeContext

from .common import (
    invoke_model,
    repository,
    required_text,
    runtime_context,
)
from .retrieval_react import run_bounded_react_worker


def plan_research(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    """Persist a bounded deterministic plan derived from the confirmed target."""

    context = runtime_context(runtime)
    repo = repository(context)
    user_id, case_id = _identity(state)
    case = repo.get_case(user_id=user_id, case_id=case_id)
    if case is None or not isinstance(case.get("target"), Mapping):
        raise ValueError("research planning requires a persisted target")
    target = dict(case["target"])
    requested_workers = state.get("retrieval_workers_needed")
    if not isinstance(requested_workers, list) or not requested_workers:
        requested_workers = ["publication", "kaken"]
    workers = [
        worker
        for worker in dict.fromkeys(requested_workers)
        if worker in {"publication", "kaken"}
    ]
    model_plan = invoke_model(
        context,
        "plan_research",
        messages=[
            {
                "role": "user",
                "content": json.dumps(target, ensure_ascii=False, default=str),
            }
        ],
        system_prompt=(
            "为论文与 KAKEN 检索生成简短查询词和停止条件。"
            "每个数组最多给出四条。"
            "不得修改目标教授、机构或日期范围。"
        ),
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "publication_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "kaken_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "stop_conditions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "publication_queries",
                "kaken_queries",
                "stop_conditions",
            ],
        },
    )
    plan = {
        "target": {
            key: target.get(key)
            for key in (
                "university_name",
                "graduate_school_name",
                "laboratory_name",
                "professor_name",
                "professor_name_variants",
                "official_urls",
                "date_from",
                "date_to",
            )
        },
        "workers": workers,
        "max_results_per_worker": 50,
        "max_model_loops_per_worker": 6,
        "stop_after_terminal_result": True,
        "model_plan": {
            field: [
                item.strip()[:160] for item in (model_plan or {}).get(field, [])[:4]
            ]
            for field in (
                "publication_queries",
                "kaken_queries",
                "stop_conditions",
            )
        },
    }
    saved = repo.save_research_plan(
        user_id=user_id,
        case_id=case_id,
        plan=plan,
        idempotency_key=f"plan:{case_id}:{len(state.get('retrieval_refs', []))}",
    )
    return {
        "research_plan_id": saved["research_plan_id"],
        "retrieval_workers_needed": workers,
        "run_status": RunStatus.RUNNING.value,
    }


def dispatch_retrieval_workers(state: Mapping[str, Any]) -> dict[str, Any]:
    """Provide a named graph node before the Send-based fan-out."""

    workers = state.get("retrieval_workers_needed") or ["publication", "kaken"]
    if not workers:
        raise ValueError("at least one retrieval worker is required")
    return {"retrieval_workers_needed": list(workers)}


def route_retrieval_workers(state: Mapping[str, Any]) -> list[Send]:
    """Fan out only the workers that still need to be executed."""

    base = {
        "user_id": state.get("user_id"),
        "case_id": state.get("case_id"),
        "workflow_stage": state.get("workflow_stage"),
        "research_plan_id": state.get("research_plan_id"),
        "retrieval_refs": state.get("retrieval_refs", []),
        "warnings": [],
        "errors": [],
    }
    routes = []
    for worker in state.get("retrieval_workers_needed", []):
        node = (
            "publication_research_agent"
            if worker == "publication"
            else "kaken_research_agent"
        )
        routes.append(Send(node, dict(base, retrieval_worker_kind=worker)))
    return routes


def publication_research_agent(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    return _run_worker(state, runtime, worker_kind="publication")


def kaken_research_agent(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    return _run_worker(state, runtime, worker_kind="kaken")


def _run_worker(
    state: Mapping[str, Any],
    runtime: Runtime[JMRRuntimeContext],
    *,
    worker_kind: str,
) -> dict[str, Any]:
    context = runtime_context(runtime)
    return run_bounded_react_worker(state, context, worker_kind=worker_kind)


def join_retrieval_results(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    """Read committed worker records and choose exactly one top-level route."""

    repo = repository(runtime)
    user_id, case_id = _identity(state)
    refs = _latest_worker_refs(state.get("retrieval_refs", []))
    if not refs:
        raise ValueError("retrieval join requires committed worker references")
    runs = repo.get_retrieval_runs(
        user_id=user_id,
        case_id=case_id,
        retrieval_run_ids=[ref["retrieval_run_id"] for ref in refs],
    )
    statuses = {_mcp_status(run.get("status")) for run in runs}
    record_count = sum(
        len((run.get("result_summary") or {}).get("records", [])) for run in runs
    )
    reported_no_site = False
    if MCPStatus.NEEDS_USER_CONFIRMATION in statuses:
        case = repo.get_case(user_id=user_id, case_id=case_id)
        target = (case or {}).get("target") or {}
        reported_no_site = (
            isinstance(target, Mapping)
            and target.get("official_site_status") == "USER_REPORTED_NONE"
        )
    if MCPStatus.NEEDS_USER_CONFIRMATION in statuses and not reported_no_site:
        decision = "identity"
        kind = InterruptKind.TARGET_IDENTITY_CONFIRMATION_REQUIRED.value
    elif record_count == 0:
        decision = "evidence"
        kind = InterruptKind.EVIDENCE_SOURCE_REQUIRED.value
    else:
        decision = "verify"
        kind = None
        repo.record_stage_transition(
            user_id=user_id,
            case_id=case_id,
            transition={
                "from_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
                "to_stage": WorkflowStage.MERGING_AND_VERIFYING_EVIDENCE.value,
                "run_status": RunStatus.RUNNING.value,
                "node_name": "join_retrieval_results",
            },
            idempotency_key=f"stage:{case_id}:verification",
        )
    update: dict[str, Any] = {
        "retrieval_join_decision": decision,
        "retrieval_run_ids": [run["retrieval_run_id"] for run in runs],
        "workflow_stage": (
            WorkflowStage.MERGING_AND_VERIFYING_EVIDENCE.value
            if decision == "verify"
            else WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value
        ),
    }
    if kind:
        update["pending_interrupt"] = _descriptor(runtime_context(runtime), state, kind)
        update["run_status"] = RunStatus.WAITING_FOR_USER.value
    return update


def await_target_identity_confirmation(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    candidates = _identity_candidates(state, repository(runtime))
    manual_url_required = not candidates
    resume = interrupt(
        {
            "kind": InterruptKind.TARGET_IDENTITY_CONFIRMATION_REQUIRED.value,
            "user_id": state.get("user_id"),
            "case_id": state.get("case_id"),
            "workflow_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
            "prompt": (
                "尚未找到可信的研究室官网。你可以重新检索、填写已知网址，"
                "或明确说明没有研究室官网并继续。"
                if manual_url_required
                else "请选择正确的官网候选；若都不对，可重新检索或说明没有官网。"
            ),
            "candidate_ids": [item["url"] for item in candidates],
            "options": candidates,
            "allow_manual_url": True,
            "allowed_actions": [
                "select_candidate",
                "retry_discovery",
                "no_official_site",
            ],
            "resume_schema_version": 1,
        }
    )
    return {"retrieval_resume_payload": resume, "pending_interrupt": None}


def apply_target_identity_confirmation(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    resume = state.get("retrieval_resume_payload")
    if not isinstance(resume, Mapping):
        return _invalid_retrieval_resume(state, "身份确认载荷必须为对象")
    action = resume.get("action", "select_candidate")
    if action == "retry_discovery":
        if (
            not isinstance(resume.get("idempotency_key"), str)
            or not resume["idempotency_key"].strip()
        ):
            return _invalid_retrieval_resume(state, "缺少 idempotency_key")
        return {
            "retrieval_workers_needed": ["publication"],
            "retrieval_resume_payload": None,
            "pending_interrupt": None,
            "run_status": RunStatus.RUNNING.value,
        }
    if action == "no_official_site":
        return _apply_no_official_site(state, runtime, resume)
    if action != "select_candidate":
        return _invalid_retrieval_resume(state, "请选择有效的官网确认操作")
    selected = resume.get("selected_candidate_id")
    try:
        parsed = urlparse(selected) if isinstance(selected, str) else None
    except ValueError:
        parsed = None
    if (
        parsed is None
        or parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return _invalid_retrieval_resume(state, "请选择有效的官网候选")
    repo = repository(runtime)
    candidate_urls = {item["url"] for item in _identity_candidates(state, repo)}
    manually_confirmed = (
        resume.get("manual_url") is True and resume.get("confirmed_by_user") is True
    )
    if candidate_urls and selected not in candidate_urls and not manually_confirmed:
        return _invalid_retrieval_resume(state, "请选择当前展示的官网候选")
    if not candidate_urls and resume.get("confirmed_by_user") is not True:
        return _invalid_retrieval_resume(state, "请确认你填写的官网 URL")
    user_id, case_id = _identity(state)
    case = repo.get_case(user_id=user_id, case_id=case_id)
    target = dict((case or {}).get("target") or {})
    target["official_urls"] = list(
        dict.fromkeys([*target.get("official_urls", []), selected])
    )
    target["official_site_status"] = "USER_CONFIRMED_URL"
    repo.update_target(
        user_id=user_id,
        case_id=case_id,
        target=target,
        expected_version=case["version"],
        idempotency_key=required_text(resume.get("idempotency_key"), "idempotency_key"),
    )
    return {
        "retrieval_workers_needed": ["publication"],
        "retrieval_resume_payload": None,
        "pending_interrupt": None,
        "run_status": RunStatus.RUNNING.value,
    }


def _apply_no_official_site(
    state: Mapping[str, Any],
    runtime: Runtime[JMRRuntimeContext],
    resume: Mapping[str, Any],
) -> dict[str, Any]:
    """Record explicit user feedback and continue with existing evidence only."""

    if resume.get("confirmed_by_user") is not True:
        return _invalid_retrieval_resume(state, "请明确确认没有研究室官网")
    if resume.get("selected_candidate_id"):
        return _invalid_retrieval_resume(state, "不能同时选择官网和声明没有官网")
    idempotency_key = resume.get("idempotency_key")
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        return _invalid_retrieval_resume(state, "缺少 idempotency_key")
    repo = repository(runtime)
    user_id, case_id = _identity(state)
    case = repo.get_case(user_id=user_id, case_id=case_id)
    if case is None:
        raise ValueError("identity confirmation requires a persisted Case")
    target = dict(case.get("target") or {})
    target["official_urls"] = []
    target["official_site_status"] = "USER_REPORTED_NONE"
    repo.update_target(
        user_id=user_id,
        case_id=case_id,
        target=target,
        expected_version=case["version"],
        idempotency_key=idempotency_key.strip(),
    )
    refs = _latest_worker_refs(state.get("retrieval_refs", []))
    repo.record_stage_transition(
        user_id=user_id,
        case_id=case_id,
        transition={
            "from_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
            "to_stage": WorkflowStage.MERGING_AND_VERIFYING_EVIDENCE.value,
            "run_status": RunStatus.RUNNING.value,
            "node_name": "apply_target_identity_confirmation",
        },
        idempotency_key=(
            f"stage:{case_id}:no-site:"
            + ":".join(ref["retrieval_run_id"] for ref in refs)
        ),
    )
    return {
        "retrieval_join_decision": "verify",
        "workflow_stage": WorkflowStage.MERGING_AND_VERIFYING_EVIDENCE.value,
        "retrieval_resume_payload": None,
        "pending_interrupt": None,
        "run_status": RunStatus.RUNNING.value,
        "warnings": ["用户报告没有研究室官网，已跳过官网确认并继续核验现有证据"],
    }


def await_evidence_source_input(state: Mapping[str, Any]) -> dict[str, Any]:
    resume = interrupt(
        {
            "kind": InterruptKind.EVIDENCE_SOURCE_REQUIRED.value,
            "user_id": state.get("user_id"),
            "case_id": state.get("case_id"),
            "workflow_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
            "prompt": "当前没有足够的可引用证据，请补充来源或调整检索范围。",
            "allowed_actions": ["provide_sources", "adjust_scope", "cancel"],
            "resume_schema_version": 1,
        }
    )
    return {"retrieval_resume_payload": resume, "pending_interrupt": None}


def apply_evidence_source_input(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    resume = state.get("retrieval_resume_payload")
    if not isinstance(resume, Mapping):
        return _invalid_retrieval_resume(state, "证据补充载荷必须为对象")
    action = resume.get("action")
    idempotency_key = resume.get("idempotency_key")
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        return _invalid_retrieval_resume(state, "缺少 idempotency_key")
    if action == "cancel":
        repo = repository(runtime)
        user_id, case_id = _identity(state)
        repo.record_stage_transition(
            user_id=user_id,
            case_id=case_id,
            transition={
                "from_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
                "to_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
                "run_status": RunStatus.CANCELLED.value,
                "node_name": "apply_evidence_source_input",
            },
            idempotency_key=idempotency_key.strip(),
        )
        return {
            "run_status": RunStatus.CANCELLED.value,
            "retrieval_resume_payload": None,
            "retrieval_cancelled": True,
        }
    if action not in {"provide_sources", "adjust_scope"}:
        return _invalid_retrieval_resume(state, "请选择有效的证据补充操作")
    repo = repository(runtime)
    user_id, case_id = _identity(state)
    case = repo.get_case(user_id=user_id, case_id=case_id)
    target = dict((case or {}).get("target") or {})
    if action == "provide_sources":
        sources = resume.get("source_urls")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(
                isinstance(item, str) and item.startswith(("http://", "https://"))
                for item in sources
            )
        ):
            return _invalid_retrieval_resume(state, "请提供至少一个有效来源 URL")
        target["official_urls"] = list(
            dict.fromkeys([*target.get("official_urls", []), *sources])
        )
        workers = ["publication"]
    else:
        date_from = resume.get("date_from")
        date_to = resume.get("date_to")
        if not isinstance(date_from, str) or not isinstance(date_to, str):
            return _invalid_retrieval_resume(state, "调整范围需要起止日期")
        try:
            start = date.fromisoformat(date_from)
            end = date.fromisoformat(date_to)
        except ValueError:
            return _invalid_retrieval_resume(state, "日期必须为 YYYY-MM-DD")
        if start.isoformat() != date_from or end.isoformat() != date_to or start > end:
            return _invalid_retrieval_resume(
                state, "日期格式无效或起始日期晚于结束日期"
            )
        target.update(date_from=date_from, date_to=date_to, date_source="user")
        workers = ["publication", "kaken"]
    repo.update_target(
        user_id=user_id,
        case_id=case_id,
        target=target,
        expected_version=case["version"],
        idempotency_key=idempotency_key.strip(),
    )
    return {
        "retrieval_workers_needed": workers,
        "retrieval_resume_payload": None,
        "run_status": RunStatus.RUNNING.value,
    }


def _identity_candidates(state: Mapping[str, Any], repo: Any) -> list[dict[str, Any]]:
    user_id, case_id = _identity(state)
    refs = _latest_worker_refs(state.get("retrieval_refs", []))
    runs = repo.get_retrieval_runs(
        user_id=user_id,
        case_id=case_id,
        retrieval_run_ids=[item["retrieval_run_id"] for item in refs],
    )
    unique: dict[str, dict[str, Any]] = {}
    for run in runs:
        summary = run.get("result_summary") or {}
        raw_candidates = summary.get("discovered_sources", summary.get("sources", []))
        if not isinstance(raw_candidates, list):
            continue
        for item in raw_candidates:
            if isinstance(item, Mapping) and isinstance(item.get("url"), str):
                unique[item["url"]] = dict(item)
    return list(unique.values())


def merge_and_verify_evidence(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    """Apply existing deterministic merge rules and persist a durable bundle."""

    context = runtime_context(runtime)
    repo = repository(context)
    user_id, case_id = _identity(state)
    refs = _latest_worker_refs(state.get("retrieval_refs", []))
    runs = repo.get_retrieval_runs(
        user_id=user_id,
        case_id=case_id,
        retrieval_run_ids=[item["retrieval_run_id"] for item in refs],
    )
    results = []
    source_refs = []
    for run in runs:
        summary = run.get("result_summary") or {}
        try:
            results.append(RetrievalResult.from_dict(summary))
            source_refs.append(run["retrieval_run_id"])
        except (TypeError, ValueError):
            continue
    if results:
        bundle = merge_and_verify_retrieval_results(results, source_refs)
        bundle_body = bundle.to_dict()
        evidence_ids = [
            item.evidence.evidence_id
            for item in bundle.records
            if item.evidence.verification_status
            in {
                EvidenceVerificationStatus.VERIFIED,
                EvidenceVerificationStatus.PARTIAL,
            }
        ]
        summary = {
            "bundle": bundle_body,
            "verified_count": bundle.verified_count,
            "partial_count": bundle.partial_count,
            "rejected_count": bundle.rejected_count,
            "conflict_count": bundle.conflict_count,
        }
    else:
        evidence_ids = _simple_evidence_ids(runs)
        summary = {"verified_count": len(evidence_ids), "records": []}
    if not evidence_ids:
        repo.record_stage_transition(
            user_id=user_id,
            case_id=case_id,
            transition={
                "from_stage": WorkflowStage.MERGING_AND_VERIFYING_EVIDENCE.value,
                "to_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
                "run_status": RunStatus.WAITING_FOR_USER.value,
                "node_name": "merge_and_verify_evidence",
            },
            idempotency_key=(
                f"stage:{case_id}:evidence-gap:"
                + ":".join(item["retrieval_run_id"] for item in refs)
            ),
        )
        return {
            "workflow_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
            "retrieval_join_decision": "evidence",
            "pending_interrupt": _descriptor(
                context, state, InterruptKind.EVIDENCE_SOURCE_REQUIRED.value
            ),
            "run_status": RunStatus.WAITING_FOR_USER.value,
        }
    saved = repo.save_verified_evidence(
        user_id=user_id,
        case_id=case_id,
        evidence_ids=evidence_ids,
        summary=summary,
        idempotency_key=f"verify:{case_id}:{':'.join(source_refs)}",
    )
    repo.record_stage_transition(
        user_id=user_id,
        case_id=case_id,
        transition={
            "from_stage": WorkflowStage.MERGING_AND_VERIFYING_EVIDENCE.value,
            "to_stage": WorkflowStage.GENERATING_RESEARCH_DIRECTIONS.value,
            "run_status": RunStatus.RUNNING.value,
            "node_name": "merge_and_verify_evidence",
        },
        idempotency_key=f"stage:{case_id}:directions",
    )
    return {
        "verified_evidence_bundle_id": saved["verified_evidence_bundle_id"],
        "workflow_stage": WorkflowStage.GENERATING_RESEARCH_DIRECTIONS.value,
        "run_status": RunStatus.RUNNING.value,
        "pending_interrupt": None,
    }


def _identity(state: Mapping[str, Any]) -> tuple[str, str]:
    return (
        required_text(state.get("user_id"), "user_id"),
        required_text(state.get("case_id"), "case_id"),
    )


def _mcp_status(value: Any) -> MCPStatus:
    normalized = str(value or "FAILED").upper()
    try:
        return MCPStatus(normalized)
    except ValueError:
        return MCPStatus.FAILED


def _latest_worker_refs(value: Any) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping) and item.get("worker_kind") in {
                "publication",
                "kaken",
            }:
                latest[str(item["worker_kind"])] = dict(item)
    return [latest[key] for key in ("publication", "kaken") if key in latest]


def _descriptor(
    context: JMRRuntimeContext, state: Mapping[str, Any], kind: str
) -> dict[str, Any]:
    return {
        "kind": kind,
        "payload_ref": f"{kind.lower()}:{state.get('case_id')}",
        "resume_schema_version": 1,
        "issued_at": context.clock.now().isoformat(),
    }


def _invalid_retrieval_resume(state: Mapping[str, Any], message: str) -> dict[str, Any]:
    kind = (
        InterruptKind.TARGET_IDENTITY_CONFIRMATION_REQUIRED.value
        if state.get("retrieval_join_decision") == "identity"
        else InterruptKind.EVIDENCE_SOURCE_REQUIRED.value
    )
    return {
        "retrieval_resume_payload": None,
        "run_status": RunStatus.WAITING_FOR_USER.value,
        "pending_interrupt": {
            "kind": kind,
            "payload_ref": f"{kind.lower()}:{state.get('case_id')}",
            "resume_schema_version": 1,
            "issued_at": "retry",
        },
        "errors": [message],
    }


def _simple_evidence_ids(runs: list[Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    for run in runs:
        for record in (run.get("result_summary") or {}).get("records", []):
            if not isinstance(record, Mapping):
                continue
            evidence_id = record.get("evidence_id") or record.get("id")
            if isinstance(evidence_id, str) and evidence_id:
                ids.append(evidence_id)
    return list(dict.fromkeys(ids))
