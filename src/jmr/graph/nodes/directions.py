"""P5 applicant memory, direction generation, critique, and selection."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from jmr.domain import (
    InterruptKind,
    MemoryStatus,
    ReviewStatus,
    RunStatus,
    WorkflowStage,
)
from jmr.runtime import JMRRuntimeContext

from .common import (
    compact_evidence_for_model,
    invoke_model,
    read_json_artifact,
    repository,
    required_text,
    runtime_context,
    write_json_artifact,
)

_DIRECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "directions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "application_scenario": {"type": "string"},
                    "research_question": {"type": "string"},
                    "method": {"type": "string"},
                    "masters_deliverable": {"type": "string"},
                    "evaluation": {"type": "string"},
                    "lab_fit": {"type": "string"},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                    "applicant_connection": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "title",
                    "summary",
                    "application_scenario",
                    "research_question",
                    "method",
                    "masters_deliverable",
                    "evaluation",
                    "lab_fit",
                    "assumptions",
                    "applicant_connection",
                    "evidence_ids",
                ],
            },
        }
    },
    "required": ["directions"],
}


def load_applicant_memory(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    """Load only ACTIVE summaries from the current user's namespace."""

    context = runtime_context(runtime)
    if state.get("memory_confirmed"):
        return {"memory_confirmation_needed": False}
    bundle = repository(context).get_verified_evidence_bundle(
        user_id=state["user_id"],
        case_id=state["case_id"],
        bundle_id=required_text(
            state.get("verified_evidence_bundle_id"), "verified_evidence_bundle_id"
        ),
    )
    paper_options = _paper_options(bundle, reference_date=context.clock.now().date())
    if not paper_options:
        repository(context).record_stage_transition(
            user_id=state["user_id"],
            case_id=state["case_id"],
            transition={
                "from_stage": WorkflowStage.GENERATING_RESEARCH_DIRECTIONS.value,
                "to_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
                "run_status": RunStatus.WAITING_FOR_USER.value,
                "node_name": "load_applicant_memory",
            },
            idempotency_key=f"stage:{state['case_id']}:no-publications:{bundle['verified_evidence_bundle_id']}",
        )
        return {
            "workflow_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
            "run_status": RunStatus.WAITING_FOR_USER.value,
            "warnings": [
                "尚未找到过去 12 个月内可引用的教授论文，"
                "请补充论文来源后再生成研究 Idea"
            ],
            "pending_interrupt": {
                "kind": InterruptKind.EVIDENCE_SOURCE_REQUIRED.value,
                "payload_ref": f"papers:{state['case_id']}",
                "resume_schema_version": 1,
                "issued_at": context.clock.now().isoformat(),
            },
        }
    paper_options_ref = write_json_artifact(
        context,
        state,
        artifact_kind="paper-options",
        value=paper_options,
    )
    options: list[dict[str, Any]] = []
    if context.memory_store is not None:
        items = context.memory_store.list(
            namespace=(
                "users",
                required_text(state.get("user_id"), "user_id"),
                "profile",
            ),
            filters={"status": MemoryStatus.ACTIVE.value},
            limit=50,
        )
        for item in items:
            memory_id = item.get("memory_id") or item.get("key")
            if not isinstance(memory_id, str):
                continue
            value = (
                item.get("value") if isinstance(item.get("value"), Mapping) else item
            )
            options.append(
                {
                    "memory_id": memory_id,
                    "label": value.get("label") or value.get("title") or memory_id,
                    "summary": value.get("summary") or value.get("content") or "",
                }
            )
    options_ref = write_json_artifact(
        context,
        state,
        artifact_kind="memory-options",
        value=options,
    )
    return {
        "memory_options_ref": options_ref,
        "paper_options_ref": paper_options_ref,
        "memory_confirmation_needed": True,
        "pending_interrupt": {
            "kind": InterruptKind.APPLICANT_MEMORY_CONFIRMATION_REQUIRED.value,
            "payload_ref": f"memory:{state.get('case_id')}",
            "resume_schema_version": 1,
            "issued_at": context.clock.now().isoformat(),
        },
        "run_status": RunStatus.WAITING_FOR_USER.value,
    }


def await_applicant_memory_confirmation(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    options = read_json_artifact(
        runtime_context(runtime), state.get("memory_options_ref")
    )
    paper_options = read_json_artifact(
        runtime_context(runtime), state.get("paper_options_ref")
    )
    resume = interrupt(
        {
            "kind": InterruptKind.APPLICANT_MEMORY_CONFIRMATION_REQUIRED.value,
            "user_id": state.get("user_id"),
            "case_id": state.get("case_id"),
            "workflow_stage": WorkflowStage.GENERATING_RESEARCH_DIRECTIONS.value,
            "prompt": (
                "请先选择 1–3 篇教授过去 12 个月内的论文，"
                "再决定是否使用个人背景记忆。"
                "Agent 随后会生成多条具体的后续研究 Idea 供你选择。"
            ),
            "paper_options": paper_options,
            "max_selected_papers": 3,
            "options": options,
            "allow_unpersonalized": True,
            "resume_schema_version": 1,
        }
    )
    return {"memory_resume_payload": resume, "pending_interrupt": None}


def apply_applicant_memory_confirmation(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    context = runtime_context(runtime)
    resume = state.get("memory_resume_payload")
    if not isinstance(resume, Mapping):
        return _memory_error(state, "背景确认载荷必须为对象")
    selected_papers = resume.get("selected_paper_evidence_ids", [])
    if (
        not isinstance(selected_papers, list)
        or not 1 <= len(selected_papers) <= 3
        or not all(isinstance(item, str) and item for item in selected_papers)
        or len(set(selected_papers)) != len(selected_papers)
    ):
        return _memory_error(state, "请先选择 1–3 篇教授的论文")
    paper_options = read_json_artifact(context, state.get("paper_options_ref"))
    allowed_papers = {item["evidence_id"] for item in paper_options}
    if not set(selected_papers).issubset(allowed_papers):
        return _memory_error(state, "所选论文不属于当前已核验的论文列表")
    selected = resume.get("selected_memory_ids", [])
    allow_unpersonalized = resume.get("allow_unpersonalized") is True
    if not isinstance(selected, list) or not all(
        isinstance(item, str) and item for item in selected
    ):
        return _memory_error(state, "selected_memory_ids 必须为字符串数组")
    selected = list(dict.fromkeys(selected))
    if bool(selected) == allow_unpersonalized:
        return _memory_error(state, "请选择背景信息或非个性化模式，二者不可同时选择")
    namespace = ("users", required_text(state.get("user_id"), "user_id"), "profile")
    if selected:
        if context.memory_store is None:
            return _memory_error(state, "长期背景存储未配置")
        for memory_id in selected:
            item = context.memory_store.get(namespace=namespace, key=memory_id)
            if item is None or item.get("status") != MemoryStatus.ACTIVE.value:
                return _memory_error(state, f"背景记录不可用于当前 Case：{memory_id}")
    idempotency_key = resume.get("idempotency_key")
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        return _memory_error(state, "缺少 idempotency_key")
    repository(context).save_case_memory_selection(
        user_id=state["user_id"],
        case_id=state["case_id"],
        memory_ids=selected,
        idempotency_key=idempotency_key,
    )
    return {
        "selected_memory_ids": selected,
        "selected_paper_evidence_ids": selected_papers,
        "selected_papers": [
            {
                "evidence_id": item["evidence_id"],
                "title": item["title"],
                "publication_date": item.get("publication_date"),
                "source_url": item.get("source_url"),
            }
            for item in paper_options
            if item["evidence_id"] in selected_papers
        ],
        "allow_unpersonalized": allow_unpersonalized,
        "memory_confirmed": True,
        "memory_confirmation_needed": False,
        "memory_resume_payload": None,
        "run_status": RunStatus.RUNNING.value,
    }


def generate_direction_candidates(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    if not state.get("memory_confirmed"):
        raise ValueError("direction generation requires confirmed memory choice")
    context = runtime_context(runtime)
    repo = repository(context)
    bundle = repo.get_verified_evidence_bundle(
        user_id=state["user_id"],
        case_id=state["case_id"],
        bundle_id=required_text(
            state.get("verified_evidence_bundle_id"),
            "verified_evidence_bundle_id",
        ),
    )
    model_evidence = _selected_paper_context(
        bundle, state.get("selected_paper_evidence_ids", [])
    )
    evidence_ids = model_evidence["allowed_evidence_ids"]
    if not evidence_ids:
        raise ValueError("direction generation requires citable evidence")
    memory_values = []
    if context.memory_store is not None:
        namespace = ("users", state["user_id"], "profile")
        for memory_id in state.get("selected_memory_ids", []):
            item = context.memory_store.get(namespace=namespace, key=memory_id)
            if item is not None:
                memory_values.append(item)
    model_result = invoke_model(
        context,
        "generate_direction_candidates",
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "evidence_bundle": model_evidence,
                        "allowed_evidence_ids": evidence_ids,
                        "confirmed_memory": memory_values,
                        "allow_unpersonalized": state.get(
                            "allow_unpersonalized", False
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            }
        ],
        system_prompt=(
            "用户已选择教授过去 12 个月内的论文。"
            "请结合这些论文生成 3–6 条彼此不同、具体可执行的硕士研究 Idea，"
            "随后由用户选择。"
            "每条 Idea 用一个独立研究主题作标题；"
            "summary 要具体说明该主题的研究对象与预期贡献，不能重复同一句模板。"
            "还要写清新的研究问题、可操作方法、评价指标和硕士阶段产出。"
            "论文只作为依据与灵感，不要把论文标题放在 Idea 标题里，"
            "不要写‘基于某论文，某事可以吗’式的泛化问句。"
            "如果记录没有摘要或方法信息，只能依论文标题与已给事实保守推演，明确待验证假设，不能编造论文结论。"
            "每条 Idea 必须引用 1–3 个所选论文的 evidence_id，不得引用未选论文。"
        ),
        output_schema=_DIRECTION_SCHEMA,
    )
    directions = model_result.get("directions", []) if model_result is not None else []
    if not directions:
        raise ValueError("模型未能基于所选论文生成研究 Idea；请重试当前阶段")
    candidate_ref = write_json_artifact(
        context,
        state,
        artifact_kind="direction-candidates",
        value=[dict(item) for item in directions],
    )
    return {"direction_candidates_ref": candidate_ref}


def critique_direction_candidates(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    repo = repository(runtime)
    bundle = repo.get_verified_evidence_bundle(
        user_id=state["user_id"],
        case_id=state["case_id"],
        bundle_id=state["verified_evidence_bundle_id"],
    )
    allowed = set(state.get("selected_paper_evidence_ids", []))
    candidates = read_json_artifact(
        runtime_context(runtime), state.get("direction_candidates_ref")
    )
    issues: list[str] = []
    if not isinstance(candidates, list) or not 3 <= len(candidates) <= 6:
        issues.append("方向数量必须为 3–6 个")
    else:
        for index, item in enumerate(candidates):
            if not isinstance(item, Mapping):
                issues.append(f"方向 {index + 1} 不是对象")
                continue
            citations = item.get("evidence_ids", [])
            if not isinstance(citations, list) or not 1 <= len(citations) <= 3:
                issues.append(f"方向 {index + 1} 必须引用 1–3 条证据")
            elif not set(citations).issubset(allowed):
                issues.append(f"方向 {index + 1} 引用了未核验证据")
            for field in (
                "title",
                "summary",
                "application_scenario",
                "research_question",
                "method",
                "masters_deliverable",
                "evaluation",
                "lab_fit",
            ):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    issues.append(f"方向 {index + 1} 缺少 {field}")
            if not isinstance(item.get("assumptions"), list):
                issues.append(f"方向 {index + 1} 缺少 assumptions")
            if state.get("selected_memory_ids") and not item.get(
                "applicant_connection"
            ):
                issues.append(f"方向 {index + 1} 缺少申请者经历连接")
    model_critique = None
    if not issues:
        model_critique = invoke_model(
            runtime_context(runtime),
            "critique_direction_candidates",
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "directions": candidates,
                            "verified_evidence": _selected_paper_context(
                                bundle,
                                state.get("selected_paper_evidence_ids", []),
                            ),
                            "selected_memory_ids": state.get("selected_memory_ids", []),
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
            system_prompt=(
                "检查每个 Idea 是否从用户所选近一年论文延伸出具体且不同的研究主题，"
                "有可执行实验和硕士阶段可行性，并由引用的 evidence_id 支持。"
                "若只是把论文标题改写为‘基于某论文，某事可以吗’，"
                "或仅提供泛化类别，应判为 REVISE。"
            ),
            output_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["PASS", "REVISE", "BLOCKED"],
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["status", "issues"],
            },
        )
    if model_critique is not None:
        try:
            status = ReviewStatus(str(model_critique.get("status"))).value
        except ValueError as exc:
            raise ValueError("direction critic returned an unknown status") from exc
        issues = [
            item for item in model_critique.get("issues", []) if isinstance(item, str)
        ]
        if (
            status == ReviewStatus.REVISE.value
            and int(state.get("generation_revision_round", 0)) >= 2
        ):
            status = ReviewStatus.BLOCKED.value
    elif not issues:
        status = ReviewStatus.PASS.value
    elif int(state.get("generation_revision_round", 0)) < 2:
        status = ReviewStatus.REVISE.value
    else:
        status = ReviewStatus.BLOCKED.value
    critique_ref = write_json_artifact(
        runtime_context(runtime),
        state,
        artifact_kind="direction-critique",
        value={"status": status, "issues": issues},
    )
    if status == ReviewStatus.BLOCKED.value:
        repo.record_stage_transition(
            user_id=state["user_id"],
            case_id=state["case_id"],
            transition={
                "from_stage": WorkflowStage.GENERATING_RESEARCH_DIRECTIONS.value,
                "to_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
                "run_status": RunStatus.WAITING_FOR_USER.value,
                "node_name": "critique_direction_candidates",
            },
            idempotency_key=(
                f"stage:{state['case_id']}:direction-block:"
                f"{state.get('verified_evidence_bundle_id')}:"
                f"{state.get('generation_revision_round', 0)}"
            ),
        )
    return {
        "direction_critique_status": status,
        "direction_critique_ref": critique_ref,
        "workflow_stage": (
            WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value
            if status == ReviewStatus.BLOCKED.value
            else WorkflowStage.GENERATING_RESEARCH_DIRECTIONS.value
        ),
        "pending_interrupt": (
            {
                "kind": InterruptKind.EVIDENCE_SOURCE_REQUIRED.value,
                "payload_ref": f"direction-critique:{state['case_id']}",
                "resume_schema_version": 1,
                "issued_at": runtime_context(runtime).clock.now().isoformat(),
            }
            if status == ReviewStatus.BLOCKED.value
            else None
        ),
        "run_status": (
            RunStatus.WAITING_FOR_USER.value
            if status == ReviewStatus.BLOCKED.value
            else RunStatus.RUNNING.value
        ),
    }


def revise_direction_candidates(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    context = runtime_context(runtime)
    current = read_json_artifact(context, state.get("direction_candidates_ref"))
    critique = read_json_artifact(context, state.get("direction_critique_ref"))
    model_result = invoke_model(
        context,
        "revise_direction_candidates",
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "directions": current,
                        "critique": critique,
                        "allowed_evidence_ids": state.get(
                            "selected_paper_evidence_ids", []
                        ),
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        system_prompt="按 Critic 意见修订方向，不得改变或编造 evidence_id。",
        output_schema=_DIRECTION_SCHEMA,
    )
    revised = model_result.get("directions", []) if model_result is not None else []
    if not revised:
        raise ValueError("模型未能基于所选论文修订研究 Idea；请重试当前阶段")
    round_number = int(state.get("generation_revision_round", 0)) + 1
    candidates_ref = write_json_artifact(
        runtime_context(runtime),
        state,
        artifact_kind="direction-candidates",
        value=revised,
    )
    return {
        "direction_candidates_ref": candidates_ref,
        "generation_revision_round": round_number,
    }


def persist_direction_batch(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    if state.get("direction_critique_status") != ReviewStatus.PASS.value:
        raise ValueError("only a PASS direction batch can be persisted")
    repo = repository(runtime)
    candidates = read_json_artifact(
        runtime_context(runtime), state.get("direction_candidates_ref")
    )
    result = repo.save_direction_batch(
        user_id=state["user_id"],
        case_id=state["case_id"],
        directions=candidates,
        evidence_bundle_id=state["verified_evidence_bundle_id"],
        revision_round=int(state.get("generation_revision_round", 0)),
        idempotency_key=(
            f"directions:{state['case_id']}:{state.get('generation_revision_round', 0)}"
        ),
    )
    repo.record_stage_transition(
        user_id=state["user_id"],
        case_id=state["case_id"],
        transition={
            "from_stage": WorkflowStage.GENERATING_RESEARCH_DIRECTIONS.value,
            "to_stage": WorkflowStage.AWAITING_DIRECTION_SELECTION.value,
            "run_status": RunStatus.WAITING_FOR_USER.value,
            "node_name": "persist_direction_batch",
        },
        idempotency_key=f"stage:{state['case_id']}:direction-selection",
    )
    return {
        "direction_batch_id": result["direction_batch_id"],
        "workflow_stage": WorkflowStage.AWAITING_DIRECTION_SELECTION.value,
        "run_status": RunStatus.WAITING_FOR_USER.value,
        "pending_interrupt": {
            "kind": InterruptKind.DIRECTION_SELECTION_REQUIRED.value,
            "payload_ref": f"directions:{result['direction_batch_id']}",
            "resume_schema_version": 1,
            "issued_at": runtime_context(runtime).clock.now().isoformat(),
        },
    }


def await_direction_selection(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    batch = repository(runtime).get_direction_batch(
        user_id=state["user_id"],
        case_id=state["case_id"],
        direction_batch_id=state["direction_batch_id"],
    )
    options = [
        {
            "direction_id": item["direction_id"],
            "title": item.get("title"),
            "summary": item.get("summary"),
        }
        for item in batch["directions"]
    ]
    resume = interrupt(
        {
            "kind": InterruptKind.DIRECTION_SELECTION_REQUIRED.value,
            "user_id": state.get("user_id"),
            "case_id": state.get("case_id"),
            "workflow_stage": WorkflowStage.AWAITING_DIRECTION_SELECTION.value,
            "prompt": "请选择 1–2 个方向，或提交一个明确的自定义方向。",
            "direction_batch_id": state.get("direction_batch_id"),
            "options": options,
            "resume_schema_version": 1,
        }
    )
    return {"direction_resume_payload": resume, "pending_interrupt": None}


def apply_direction_selection(
    state: Mapping[str, Any], runtime: Runtime[JMRRuntimeContext]
) -> dict[str, Any]:
    resume = state.get("direction_resume_payload")
    if not isinstance(resume, Mapping):
        return _direction_error(state, "方向选择载荷必须为对象")
    if resume.get("direction_batch_id") != state.get("direction_batch_id"):
        return _direction_error(state, "方向批次已过期或不匹配")
    selected = resume.get("selected_direction_ids", [])
    custom = resume.get("custom_direction")
    if not isinstance(selected, list) or not all(
        isinstance(item, str) and item for item in selected
    ):
        return _direction_error(state, "selected_direction_ids 必须为字符串数组")
    selected = list(dict.fromkeys(selected))
    if isinstance(custom, str):
        custom = {"title": custom.strip()} if custom.strip() else None
    if bool(selected) == bool(custom) or len(selected) > 2:
        return _direction_error(state, "请选择 1–2 个方向或一个自定义方向")
    batch = repository(runtime).get_direction_batch(
        user_id=state["user_id"],
        case_id=state["case_id"],
        direction_batch_id=state["direction_batch_id"],
    )
    allowed = {item["direction_id"] for item in batch["directions"]}
    if selected and not set(selected).issubset(allowed):
        return _direction_error(state, "所选方向不属于当前批次")
    key = resume.get("idempotency_key")
    if not isinstance(key, str) or not key.strip():
        return _direction_error(state, "缺少 idempotency_key")
    repository(runtime).save_direction_selection(
        user_id=state["user_id"],
        case_id=state["case_id"],
        direction_batch_id=state["direction_batch_id"],
        selected_direction_ids=selected,
        custom_direction=custom,
        idempotency_key=key,
    )
    repository(runtime).record_stage_transition(
        user_id=state["user_id"],
        case_id=state["case_id"],
        transition={
            "from_stage": WorkflowStage.AWAITING_DIRECTION_SELECTION.value,
            "to_stage": WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN.value,
            "run_status": RunStatus.RUNNING.value,
            "node_name": "apply_direction_selection",
        },
        idempotency_key=f"stage:{state['case_id']}:drafting",
    )
    custom_ref = None
    if custom is not None:
        custom_ref = write_json_artifact(
            runtime_context(runtime),
            state,
            artifact_kind="custom-direction",
            value=custom,
        )
    return {
        "selected_direction_ids": selected,
        "selected_custom_direction_ref": custom_ref,
        "direction_resume_payload": None,
        "workflow_stage": WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN.value,
        "run_status": RunStatus.RUNNING.value,
    }


def _paper_options(
    bundle: Mapping[str, Any], *, reference_date: date
) -> list[dict[str, Any]]:
    """Offer only citable publications from the preceding 12 months."""

    allowed = set(bundle.get("evidence_ids", []))
    summary = bundle.get("summary") or {}
    source = summary.get("bundle") or {}
    records = source.get("records", summary.get("records", []))
    options: list[dict[str, Any]] = []
    for item in records if isinstance(records, list) else []:
        evidence = item.get("evidence", item) if isinstance(item, Mapping) else {}
        if not isinstance(evidence, Mapping):
            continue
        evidence_id = evidence.get("evidence_id")
        if evidence_id not in allowed or evidence.get("evidence_type") != "publication":
            continue
        title = evidence.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        publication_date = evidence.get("publication_date")
        if not isinstance(publication_date, str):
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", publication_date):
            try:
                published = date.fromisoformat(publication_date)
            except ValueError:
                continue
            recent = reference_date - timedelta(days=365) <= published <= reference_date
        elif re.fullmatch(r"\d{4}", publication_date):
            # A year-only record cannot place a previous-year paper inside the
            # rolling window, so include only the current calendar year.
            recent = int(publication_date) == reference_date.year
        else:
            recent = False
        if not recent:
            continue
        options.append(
            {
                "evidence_id": evidence_id,
                "title": title.strip(),
                "publication_date": publication_date,
                "venue": evidence.get("venue"),
                "source_url": evidence.get("source_url"),
                "verification_status": evidence.get("verification_status"),
                "recent": True,
            }
        )
    options.sort(key=lambda item: item["publication_date"], reverse=True)
    return options[:40]


def _selected_paper_context(
    bundle: Mapping[str, Any], selected_ids: list[str]
) -> dict[str, Any]:
    if not selected_ids:
        raise ValueError("生成研究 Idea 前必须选择教授的论文")
    selected = set(selected_ids)
    context = compact_evidence_for_model(bundle, priority_ids=selected_ids)
    records = [item for item in context["records"] if item["evidence_id"] in selected]
    if {item["evidence_id"] for item in records} != selected:
        raise ValueError("所选论文在已核验证据中不可用，请重新选择")
    context["records"] = records
    context["allowed_evidence_ids"] = list(selected_ids)
    context["selected_publication_count"] = len(records)
    return context


def _memory_error(state: Mapping[str, Any], message: str) -> dict[str, Any]:
    return {
        "memory_resume_payload": None,
        "memory_confirmation_needed": True,
        "run_status": RunStatus.WAITING_FOR_USER.value,
        "pending_interrupt": {
            "kind": InterruptKind.APPLICANT_MEMORY_CONFIRMATION_REQUIRED.value,
            "payload_ref": f"memory:{state.get('case_id')}",
            "resume_schema_version": 1,
            "issued_at": "retry",
        },
        "errors": [message],
    }


def _direction_error(state: Mapping[str, Any], message: str) -> dict[str, Any]:
    return {
        "direction_resume_payload": None,
        "run_status": RunStatus.WAITING_FOR_USER.value,
        "pending_interrupt": {
            "kind": InterruptKind.DIRECTION_SELECTION_REQUIRED.value,
            "payload_ref": f"directions:{state.get('direction_batch_id')}",
            "resume_schema_version": 1,
            "issued_at": "retry",
        },
        "errors": [message],
    }
