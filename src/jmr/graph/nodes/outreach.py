"""P6 outreach planning, drafting, review, bounded revision, and completion."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from jmr.domain import InterruptKind, ReviewStatus, RunStatus, WorkflowStage
from jmr.runtime import JMRRuntimeContext

from .common import (
    compact_evidence_for_model,
    invoke_model,
    read_json_artifact,
    repository,
    runtime_context,
    write_json_artifact,
)

_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "content": {"type": "string", "minLength": 1},
        "citation_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["content", "citation_ids"],
}

_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["PASS", "REVISE", "BLOCKED"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "feedback": {"type": "string"},
    },
    "required": ["status", "issues", "feedback"],
}


def plan_outreach_paragraph(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    repo = repository(runtime)
    batch = repo.get_direction_batch(
        user_id=state["user_id"],
        case_id=state["case_id"],
        direction_batch_id=state["direction_batch_id"],
    )
    selected = set(state.get("selected_direction_ids", []))
    chosen = [item for item in batch["directions"] if item["direction_id"] in selected]
    if state.get("selected_custom_direction_ref"):
        chosen.append(
            dict(
                read_json_artifact(
                    runtime_context(runtime),
                    state["selected_custom_direction_ref"],
                )
            )
        )
    if not chosen:
        raise ValueError("outreach planning requires a persisted direction selection")
    evidence_bundle = repo.get_verified_evidence_bundle(
        user_id=state["user_id"],
        case_id=state["case_id"],
        bundle_id=state["verified_evidence_bundle_id"],
    )
    cited_ids = [
        evidence_id
        for item in chosen
        for evidence_id in item.get("evidence_ids", [])
        if isinstance(evidence_id, str)
    ]
    model_evidence = compact_evidence_for_model(evidence_bundle, priority_ids=cited_ids)
    revision_input = None
    if state.get("revision_input_ref"):
        revision_input = read_json_artifact(
            runtime_context(runtime), state["revision_input_ref"]
        )
    elif state.get("current_outreach_iteration_id"):
        revision_input = repo.list_review_results(
            user_id=state["user_id"],
            case_id=state["case_id"],
            iteration_id=state["current_outreach_iteration_id"],
        )
    proposed = invoke_model(
        runtime_context(runtime),
        "plan_outreach_paragraph",
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "selected_directions": chosen,
                        "verified_evidence": model_evidence,
                        "revision_input": revision_input,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            }
        ],
        system_prompt=(
            "为套磁研究段落规划四个具体部分：现实问题、教授研究、"
            "申请者理解、可执行研究提案。只使用给定证据，不补造经历。"
        ),
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                field: {"type": "string", "minLength": 1}
                for field in (
                    "real_world_problem",
                    "professor_research",
                    "applicant_understanding",
                    "research_proposal",
                )
            },
            "required": [
                "real_world_problem",
                "professor_research",
                "applicant_understanding",
                "research_proposal",
            ],
        },
    )
    plan = {
        "real_world_problem": "说明该研究方向所回应的现实或学术问题",
        "professor_research": "准确连接目标教授已公开的研究成果",
        "applicant_understanding": "说明申请者对问题与证据边界的理解",
        "research_proposal": "提出硕士阶段可完成的问题、方法与预期贡献",
        "selected_directions": chosen,
        "verified_evidence": model_evidence,
        "revision_input": revision_input,
    }
    for field in (
        "real_world_problem",
        "professor_research",
        "applicant_understanding",
        "research_proposal",
    ):
        value = proposed.get(field) if isinstance(proposed, Mapping) else None
        if isinstance(value, str) and value.strip():
            plan[field] = value.strip()
    return {
        "outreach_plan_ref": write_json_artifact(
            runtime_context(runtime),
            state,
            artifact_kind="outreach-plan",
            value=plan,
        )
    }


def draft_outreach_paragraph(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    context = runtime_context(runtime)
    plan = read_json_artifact(context, state.get("outreach_plan_ref"))
    bundle = repository(context).get_verified_evidence_bundle(
        user_id=state["user_id"],
        case_id=state["case_id"],
        bundle_id=state["verified_evidence_bundle_id"],
    )
    allowed = list(bundle.get("evidence_ids", []))
    target_language = context.policies.get("target_language", "zh-CN")
    chinese = target_language in {None, "zh", "zh-CN", "Chinese"}
    result = invoke_model(
        context,
        "draft_outreach_paragraph",
        messages=[
            {
                "role": "user",
                "content": json.dumps(plan, ensure_ascii=False, default=str),
            }
        ],
        system_prompt=(
            "按照现实问题、教授研究、个人理解、研究提案的顺序，"
            f"生成{('中文' if chinese else str(target_language))}"
            "的 1–2 段套磁研究计划；中文稿 200–400 字。"
            "事实只能引用给定 evidence_id。"
        ),
        output_schema=_DRAFT_SCHEMA,
    )
    if result is None:
        if not chinese:
            raise ValueError("non-Chinese drafting requires a configured model")
        content = _fallback_draft(plan.get("revision_input"))
        citations = allowed[:3]
    else:
        content = result.get("content")
        citations = result.get("citation_ids")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("draft content must be non-empty")
    if not isinstance(citations, list) or not citations:
        raise ValueError("draft must cite at least one evidence record")
    if not all(isinstance(item, str) and item in allowed for item in citations):
        raise ValueError("draft contains an unverified citation ID")
    draft_ref = write_json_artifact(
        context,
        state,
        artifact_kind="outreach-draft",
        value={
            "content": content.strip(),
            "citation_ids": list(dict.fromkeys(citations)),
        },
    )
    return {"outreach_draft_ref": draft_ref}


def persist_outreach_iteration(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    repo = repository(runtime)
    parent = state.get("current_outreach_iteration_id")
    draft = read_json_artifact(
        runtime_context(runtime), state.get("outreach_draft_ref")
    )
    result = repo.save_outreach_iteration(
        user_id=state["user_id"],
        case_id=state["case_id"],
        content=draft["content"],
        citation_ids=draft["citation_ids"],
        parent_iteration_id=parent,
        idempotency_key=(
            f"outreach:{state['case_id']}:{state.get('review_round', 0)}:"
            f"{parent or 'initial'}"
        ),
    )
    repo.record_stage_transition(
        user_id=state["user_id"],
        case_id=state["case_id"],
        transition={
            "from_stage": WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN.value,
            "to_stage": WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN.value,
            "run_status": RunStatus.RUNNING.value,
            "node_name": "persist_outreach_iteration",
        },
        idempotency_key=(f"stage:{state['case_id']}:review:{result['iteration_id']}"),
    )
    return {
        "current_outreach_iteration_id": result["iteration_id"],
        "workflow_stage": WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN.value,
        "run_status": RunStatus.RUNNING.value,
        "content_review_status": None,
        "language_review_status": None,
    }


def review_outreach_content(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    context = runtime_context(runtime)
    repo = repository(context)
    iteration = repo.get_outreach_iteration(
        user_id=state["user_id"],
        case_id=state["case_id"],
        iteration_id=state["current_outreach_iteration_id"],
    )
    bundle = repo.get_verified_evidence_bundle(
        user_id=state["user_id"],
        case_id=state["case_id"],
        bundle_id=state["verified_evidence_bundle_id"],
    )
    citation_ids = iteration.get("citation_ids", [])
    allowed_ids = set(bundle.get("evidence_ids", []))
    if not citation_ids or not set(citation_ids).issubset(allowed_ids):
        result = {
            "status": ReviewStatus.BLOCKED.value,
            "issues": ["引用缺失或不属于当前核验证据批次"],
            "feedback": "请重新选择当前批次中的证据引用。",
        }
    else:
        result = None
    model_evidence = compact_evidence_for_model(
        bundle, priority_ids=citation_ids, max_records=max(len(citation_ids), 1)
    )
    evidence_records = [
        item
        for item in model_evidence["records"]
        if item["evidence_id"] in citation_ids
    ]
    if result is None:
        result = invoke_model(
            context,
            "review_outreach_content",
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "content": iteration.get("content", ""),
                            "citation_ids": citation_ids,
                            "verified_evidence": evidence_records,
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
            system_prompt=(
                "逐项核对正文事实与所附已核验证据，审查引用、问题/方法/贡献、逻辑、创新边界、"
                "硕士阶段可行性和 200–400 字格式。"
            ),
            output_schema=_REVIEW_SCHEMA,
        )
    if result is None:
        issues = []
        if not iteration.get("citation_ids"):
            issues.append("缺少可追溯引用")
        content_length = len(iteration.get("content", ""))
        if not 180 <= content_length <= 800:
            issues.append("正文长度不符合中文研究计划段落范围")
        result = {
            "status": ReviewStatus.PASS.value
            if not issues
            else ReviewStatus.REVISE.value,
            "issues": issues,
            "feedback": "；".join(issues),
        }
    status = _review_status(result.get("status"))
    repo.save_review_result(
        user_id=state["user_id"],
        case_id=state["case_id"],
        iteration_id=state["current_outreach_iteration_id"],
        reviewer_kind="content",
        status=status,
        result=dict(result),
        idempotency_key=f"review:content:{state['current_outreach_iteration_id']}",
    )
    return {"content_review_status": status}


def review_outreach_language(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    context = runtime_context(runtime)
    target_language = context.policies.get("target_language", "zh-CN")
    if target_language in {None, "zh", "zh-CN", "Chinese"}:
        return {
            "language_review_status": ReviewStatus.PASS.value,
        }
    iteration = repository(context).get_outreach_iteration(
        user_id=state["user_id"],
        case_id=state["case_id"],
        iteration_id=state["current_outreach_iteration_id"],
    )
    result = invoke_model(
        context,
        "review_outreach_language",
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": target_language,
                        "content": iteration.get("content", ""),
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        system_prompt=f"审查 {target_language} 版本的礼貌、准确、自然和术语一致性。",
        output_schema=_REVIEW_SCHEMA,
    ) or {
        "status": "BLOCKED",
        "issues": ["非中文版本缺少语言审查模型"],
        "feedback": "请配置语言审查模型后重试。",
    }
    status = _review_status(result.get("status"))
    repository(context).save_review_result(
        user_id=state["user_id"],
        case_id=state["case_id"],
        iteration_id=state["current_outreach_iteration_id"],
        reviewer_kind="language",
        status=status,
        result=dict(result),
        idempotency_key=f"review:language:{state['current_outreach_iteration_id']}",
    )
    return {"language_review_status": status}


def route_review_result(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    statuses = {
        state.get("content_review_status"),
        state.get("language_review_status"),
    }
    if ReviewStatus.BLOCKED.value in statuses:
        decision = "interrupt"
    elif ReviewStatus.REVISE.value in statuses:
        decision = "revise" if int(state.get("review_round", 0)) < 2 else "interrupt"
    elif statuses == {ReviewStatus.PASS.value}:
        decision = "finalize"
    else:
        decision = "interrupt"
    update: dict[str, Any] = {"review_decision": decision}
    if decision == "revise":
        repository(runtime).record_stage_transition(
            user_id=state["user_id"],
            case_id=state["case_id"],
            transition={
                "from_stage": WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN.value,
                "to_stage": WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN.value,
                "run_status": RunStatus.RUNNING.value,
                "node_name": "route_review_result",
            },
            idempotency_key=(
                f"stage:{state['case_id']}:redraft:"
                f"{state.get('current_outreach_iteration_id')}"
            ),
        )
        update.update(
            review_round=int(state.get("review_round", 0)) + 1,
            revision_input_ref=None,
            workflow_stage=WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN.value,
        )
    elif decision == "interrupt":
        update.update(
            run_status=RunStatus.WAITING_FOR_USER.value,
            pending_interrupt={
                "kind": InterruptKind.REVISION_INPUT_REQUIRED.value,
                "payload_ref": f"revision:{state.get('current_outreach_iteration_id')}",
                "resume_schema_version": 1,
                "issued_at": "pending",
            },
        )
    return update


def await_revision_input(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    reviews = repository(runtime).list_review_results(
        user_id=state["user_id"],
        case_id=state["case_id"],
        iteration_id=state["current_outreach_iteration_id"],
    )
    resume = interrupt(
        {
            "kind": InterruptKind.REVISION_INPUT_REQUIRED.value,
            "user_id": state.get("user_id"),
            "case_id": state.get("case_id"),
            "workflow_stage": WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN.value,
            "prompt": "自动返修已阻断或达到上限，请提供反馈、事实或编辑稿。",
            "current_iteration_id": state.get("current_outreach_iteration_id"),
            "reviews": reviews,
            "allowed_actions": ["feedback", "provide_facts", "edit"],
            "resume_schema_version": 1,
        }
    )
    return {"revision_resume_payload": resume, "pending_interrupt": None}


def apply_revision_input(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    resume = state.get("revision_resume_payload")
    if not isinstance(resume, Mapping):
        return _revision_error(state, "返修载荷必须为对象")
    action = resume.get("action")
    key = resume.get("idempotency_key")
    if action not in {"feedback", "provide_facts", "edit"}:
        return _revision_error(state, "请选择有效返修操作")
    if not isinstance(key, str) or not key.strip():
        return _revision_error(state, "缺少 idempotency_key")
    feedback = resume.get("feedback") or resume.get("facts")
    iteration_id = state.get("current_outreach_iteration_id")
    if action == "edit":
        edited = resume.get("edited_text")
        if not isinstance(edited, str) or not edited.strip():
            return _revision_error(state, "编辑操作需要非空 edited_text")
        citation_ids = resume.get("citation_ids")
        bundle = repository(runtime).get_verified_evidence_bundle(
            user_id=state["user_id"],
            case_id=state["case_id"],
            bundle_id=state["verified_evidence_bundle_id"],
        )
        if (
            not isinstance(citation_ids, list)
            or not citation_ids
            or not all(isinstance(item, str) for item in citation_ids)
            or not set(citation_ids).issubset(set(bundle.get("evidence_ids", [])))
        ):
            return _revision_error(state, "编辑稿必须显式提供当前已核验的 citation_ids")
        saved = repository(runtime).save_outreach_iteration(
            user_id=state["user_id"],
            case_id=state["case_id"],
            content=edited.strip(),
            citation_ids=list(dict.fromkeys(citation_ids)),
            parent_iteration_id=iteration_id,
            idempotency_key=key,
        )
        iteration_id = saved["iteration_id"]
        feedback = "用户提交了编辑稿"
    elif not isinstance(feedback, str) or not feedback.strip():
        return _revision_error(state, "该操作需要非空反馈或事实")
    revision_ref = write_json_artifact(
        runtime_context(runtime),
        state,
        artifact_kind="revision-input",
        value={"action": action, "feedback": feedback},
    )
    repository(runtime).record_stage_transition(
        user_id=state["user_id"],
        case_id=state["case_id"],
        transition={
            "from_stage": WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN.value,
            "to_stage": WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN.value,
            "run_status": RunStatus.RUNNING.value,
            "node_name": "apply_revision_input",
        },
        idempotency_key=f"stage:{state['case_id']}:user-revision:{key.strip()}",
    )
    return {
        "current_outreach_iteration_id": iteration_id,
        "revision_input_ref": revision_ref,
        "revision_resume_payload": None,
        "review_round": 0,
        "workflow_stage": WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN.value,
        "run_status": RunStatus.RUNNING.value,
    }


def finalize_case(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    if state.get("content_review_status") != ReviewStatus.PASS.value:
        raise ValueError("content review must PASS before finalization")
    if state.get("language_review_status") != ReviewStatus.PASS.value:
        raise ValueError("required language review must PASS before finalization")
    repo = repository(runtime)
    reviews = repo.list_review_results(
        user_id=state["user_id"],
        case_id=state["case_id"],
        iteration_id=state["current_outreach_iteration_id"],
    )
    if not any(
        review.get("reviewer_kind") == "content"
        and review.get("status") == ReviewStatus.PASS.value
        for review in reviews
    ):
        raise ValueError("persisted content review must PASS before finalization")
    target_language = runtime_context(runtime).policies.get("target_language", "zh-CN")
    if target_language not in {None, "zh", "zh-CN", "Chinese"} and not any(
        review.get("reviewer_kind") == "language"
        and review.get("status") == ReviewStatus.PASS.value
        for review in reviews
    ):
        raise ValueError("persisted language review must PASS before finalization")
    repo.record_stage_transition(
        user_id=state["user_id"],
        case_id=state["case_id"],
        transition={
            "from_stage": WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN.value,
            "to_stage": WorkflowStage.COMPLETED.value,
            "run_status": RunStatus.COMPLETED.value,
            "node_name": "finalize_case",
        },
        idempotency_key=f"stage:{state['case_id']}:completed",
    )
    result = repo.complete_case(
        user_id=state["user_id"],
        case_id=state["case_id"],
        final_iteration_id=state["current_outreach_iteration_id"],
        idempotency_key=f"complete:{state['case_id']}",
    )
    return {
        "workflow_stage": result["workflow_stage"],
        "run_status": result["run_status"],
        "pending_interrupt": None,
    }


def _fallback_draft(revision_input: Any) -> str:
    feedback = (
        revision_input.get("feedback") if isinstance(revision_input, Mapping) else None
    )
    revision_sentence = (
        f"同时，我会依据审查意见“{feedback}”进一步收紧研究边界。"
        if isinstance(feedback, str) and feedback
        else ""
    )
    return (
        "我关注目标领域中理论结论与现实应用之间仍存在的落差。"
        "阅读老师公开发表的研究成果后，我理解到该问题不能只以单一指标解释，"
        "还需要结合对象特征、作用机制与具体情境进行审慎分析。"
        "这些成果为识别关键变量和建立可检验的问题提供了可靠起点，也让我认识到"
        "现有证据的适用范围与局限。\n\n"
        "硕士阶段，我希望围绕已选择的方向，先系统整理相关文献与公开数据，"
        "提出边界清晰的研究问题，再采用小规模、可复现的比较分析或实证调查检验假设。"
        "研究将优先保证数据可获得性、方法透明度和一年至两年内的完成可能性，"
        "并把预期贡献限定为对既有机制的验证、补充或情境扩展，而不夸大创新。"
        "我也希望在老师指导下逐步修正研究设计，使问题意识、分析方法与研究室方向"
        f"保持一致。{revision_sentence}"
    )


def _review_status(value: Any) -> str:
    try:
        return ReviewStatus(str(value).upper()).value
    except ValueError as exc:
        raise ValueError("review returned an unknown status") from exc


def _revision_error(state: Mapping[str, Any], message: str) -> dict[str, Any]:
    return {
        "revision_resume_payload": None,
        "run_status": RunStatus.WAITING_FOR_USER.value,
        "pending_interrupt": {
            "kind": InterruptKind.REVISION_INPUT_REQUIRED.value,
            "payload_ref": f"revision:{state.get('current_outreach_iteration_id')}",
            "resume_schema_version": 1,
            "issued_at": "retry",
        },
        "errors": [message],
    }
