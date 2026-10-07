"""The single production StateGraph for one JMR research case."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from jmr.domain import ReviewStatus, RunStatus, WorkflowStage
from jmr.runtime import JMRRuntimeContext
from jmr.runtime.observability import instrument_node

from .nodes.application import (
    apply_application_inputs,
    await_application_inputs,
    extract_application_inputs,
    initialize_case,
    load_case_context,
    validate_application_inputs,
)
from .nodes.directions import (
    apply_applicant_memory_confirmation,
    apply_direction_selection,
    await_applicant_memory_confirmation,
    await_direction_selection,
    critique_direction_candidates,
    generate_direction_candidates,
    load_applicant_memory,
    persist_direction_batch,
    revise_direction_candidates,
)
from .nodes.outreach import (
    apply_revision_input,
    await_revision_input,
    draft_outreach_paragraph,
    finalize_case,
    persist_outreach_iteration,
    plan_outreach_paragraph,
    review_outreach_content,
    review_outreach_language,
    route_review_result,
)
from .nodes.retrieval import (
    apply_evidence_source_input,
    apply_target_identity_confirmation,
    await_evidence_source_input,
    await_target_identity_confirmation,
    dispatch_retrieval_workers,
    join_retrieval_results,
    kaken_research_agent,
    merge_and_verify_evidence,
    plan_research,
    publication_research_agent,
    route_retrieval_workers,
)
from .state import JMRGraphState, create_jmr_graph_state


class FullGraphState(JMRGraphState, total=False):
    """Durable IDs plus bounded, phase-local node envelopes."""

    application_fields: dict[str, Any]
    persisted_application_fields: dict[str, Any]
    application_extraction: dict[str, Any]
    application_validation: dict[str, Any]
    application_resume_payload: Any
    application_case_version: int
    application_stage_done: bool

    retrieval_workers_needed: list[str]
    retrieval_worker_kind: str
    retrieval_run_ids: list[str]
    retrieval_join_decision: str
    retrieval_resume_payload: Any
    retrieval_cancelled: bool
    memory_options_ref: str
    paper_options_ref: str
    memory_confirmation_needed: bool
    memory_resume_payload: Any
    memory_confirmed: bool
    allow_unpersonalized: bool
    selected_paper_evidence_ids: list[str]
    selected_papers: list[dict[str, Any]]
    direction_candidates_ref: str
    direction_critique_ref: str
    direction_critique_status: str
    direction_resume_payload: Any
    selected_custom_direction_ref: str | None
    outreach_plan_ref: str
    outreach_draft_ref: str
    content_review_status: str | None
    language_review_status: str | None
    review_decision: str
    revision_input_ref: str | None
    revision_resume_payload: Any


def _route_initialize(state: FullGraphState) -> str:
    if not state.get("application_stage_done"):
        return "load_case_context"
    stage = state.get("workflow_stage")
    return {
        WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value: "plan_research",
        WorkflowStage.MERGING_AND_VERIFYING_EVIDENCE.value: (
            "merge_and_verify_evidence"
        ),
        WorkflowStage.GENERATING_RESEARCH_DIRECTIONS.value: "load_applicant_memory",
        WorkflowStage.AWAITING_DIRECTION_SELECTION.value: ("await_direction_selection"),
        WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN.value: (
            "plan_outreach_paragraph"
        ),
        WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN.value: (
            "review_outreach_content"
        ),
        WorkflowStage.COMPLETED.value: END,
    }.get(stage, "load_case_context")


def _route_validation(state: FullGraphState) -> str:
    return (
        "apply_application_inputs"
        if (state.get("application_validation") or {}).get("ready")
        else "await_application_inputs"
    )


def _route_application_apply(state: FullGraphState) -> str:
    if state.get("run_status") == RunStatus.WAITING_FOR_USER.value:
        return "await_application_inputs"
    return "plan_research"


def _route_join(state: FullGraphState) -> str:
    return {
        "identity": "await_target_identity_confirmation",
        "evidence": "await_evidence_source_input",
        "verify": "merge_and_verify_evidence",
    }[state["retrieval_join_decision"]]


def _route_identity_apply(state: FullGraphState) -> str:
    if state.get("retrieval_join_decision") == "verify":
        return "merge_and_verify_evidence"
    if state.get("run_status") == RunStatus.WAITING_FOR_USER.value:
        return "await_target_identity_confirmation"
    return "plan_research"


def _route_evidence_apply(state: FullGraphState) -> str:
    if state.get("retrieval_cancelled"):
        return END
    if state.get("run_status") == RunStatus.WAITING_FOR_USER.value:
        return "await_evidence_source_input"
    return "plan_research"


def _route_verification(state: FullGraphState) -> str:
    if state.get("workflow_stage") == WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value:
        return "await_evidence_source_input"
    return "load_applicant_memory"


def _route_memory(state: FullGraphState) -> str:
    if state.get("workflow_stage") == WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value:
        return "await_evidence_source_input"
    return (
        "await_applicant_memory_confirmation"
        if state.get("memory_confirmation_needed")
        else "generate_direction_candidates"
    )


def _route_memory_apply(state: FullGraphState) -> str:
    if state.get("run_status") == RunStatus.WAITING_FOR_USER.value:
        return "await_applicant_memory_confirmation"
    return "generate_direction_candidates"


def _route_direction_critique(state: FullGraphState) -> str:
    status = state.get("direction_critique_status")
    if status == ReviewStatus.PASS.value:
        return "persist_direction_batch"
    if status == ReviewStatus.REVISE.value:
        return "revise_direction_candidates"
    return "await_evidence_source_input"


def _route_direction_apply(state: FullGraphState) -> str:
    if state.get("run_status") == RunStatus.WAITING_FOR_USER.value:
        return "await_direction_selection"
    return "plan_outreach_paragraph"


def _route_review(state: FullGraphState) -> str:
    return {
        "finalize": "finalize_case",
        "revise": "plan_outreach_paragraph",
        "interrupt": "await_revision_input",
    }[state["review_decision"]]


def _route_revision_apply(state: FullGraphState) -> str:
    if state.get("run_status") == RunStatus.WAITING_FOR_USER.value:
        return "await_revision_input"
    return "plan_outreach_paragraph"


def build_graph() -> StateGraph:
    """Register all 34 manifest nodes and their deterministic routes."""

    graph = StateGraph(FullGraphState, context_schema=JMRRuntimeContext)
    nodes = {
        "initialize_case": initialize_case,
        "load_case_context": load_case_context,
        "extract_application_inputs": extract_application_inputs,
        "validate_application_inputs": validate_application_inputs,
        "await_application_inputs": await_application_inputs,
        "apply_application_inputs": apply_application_inputs,
        "plan_research": plan_research,
        "dispatch_retrieval_workers": dispatch_retrieval_workers,
        "publication_research_agent": publication_research_agent,
        "kaken_research_agent": kaken_research_agent,
        "join_retrieval_results": join_retrieval_results,
        "await_target_identity_confirmation": await_target_identity_confirmation,
        "apply_target_identity_confirmation": apply_target_identity_confirmation,
        "await_evidence_source_input": await_evidence_source_input,
        "apply_evidence_source_input": apply_evidence_source_input,
        "merge_and_verify_evidence": merge_and_verify_evidence,
        "load_applicant_memory": load_applicant_memory,
        "await_applicant_memory_confirmation": await_applicant_memory_confirmation,
        "apply_applicant_memory_confirmation": apply_applicant_memory_confirmation,
        "generate_direction_candidates": generate_direction_candidates,
        "critique_direction_candidates": critique_direction_candidates,
        "revise_direction_candidates": revise_direction_candidates,
        "persist_direction_batch": persist_direction_batch,
        "await_direction_selection": await_direction_selection,
        "apply_direction_selection": apply_direction_selection,
        "plan_outreach_paragraph": plan_outreach_paragraph,
        "draft_outreach_paragraph": draft_outreach_paragraph,
        "persist_outreach_iteration": persist_outreach_iteration,
        "review_outreach_content": review_outreach_content,
        "review_outreach_language": review_outreach_language,
        "route_review_result": route_review_result,
        "await_revision_input": await_revision_input,
        "apply_revision_input": apply_revision_input,
        "finalize_case": finalize_case,
    }
    for name, node in nodes.items():
        graph.add_node(name, instrument_node(name, node))

    graph.add_edge(START, "initialize_case")
    graph.add_conditional_edges("initialize_case", _route_initialize)
    graph.add_edge("load_case_context", "extract_application_inputs")
    graph.add_edge("extract_application_inputs", "validate_application_inputs")
    graph.add_conditional_edges("validate_application_inputs", _route_validation)
    graph.add_edge("await_application_inputs", "apply_application_inputs")
    graph.add_conditional_edges("apply_application_inputs", _route_application_apply)

    graph.add_edge("plan_research", "dispatch_retrieval_workers")
    graph.add_conditional_edges(
        "dispatch_retrieval_workers",
        route_retrieval_workers,
        ["publication_research_agent", "kaken_research_agent"],
    )
    graph.add_edge("publication_research_agent", "join_retrieval_results")
    graph.add_edge("kaken_research_agent", "join_retrieval_results")
    graph.add_conditional_edges("join_retrieval_results", _route_join)
    graph.add_edge(
        "await_target_identity_confirmation",
        "apply_target_identity_confirmation",
    )
    graph.add_conditional_edges(
        "apply_target_identity_confirmation", _route_identity_apply
    )
    graph.add_edge("await_evidence_source_input", "apply_evidence_source_input")
    graph.add_conditional_edges("apply_evidence_source_input", _route_evidence_apply)
    graph.add_conditional_edges("merge_and_verify_evidence", _route_verification)

    graph.add_conditional_edges("load_applicant_memory", _route_memory)
    graph.add_edge(
        "await_applicant_memory_confirmation",
        "apply_applicant_memory_confirmation",
    )
    graph.add_conditional_edges(
        "apply_applicant_memory_confirmation", _route_memory_apply
    )
    graph.add_edge("generate_direction_candidates", "critique_direction_candidates")
    graph.add_conditional_edges(
        "critique_direction_candidates", _route_direction_critique
    )
    graph.add_edge("revise_direction_candidates", "critique_direction_candidates")
    graph.add_edge("persist_direction_batch", "await_direction_selection")
    graph.add_edge("await_direction_selection", "apply_direction_selection")
    graph.add_conditional_edges("apply_direction_selection", _route_direction_apply)

    graph.add_edge("plan_outreach_paragraph", "draft_outreach_paragraph")
    graph.add_edge("draft_outreach_paragraph", "persist_outreach_iteration")
    graph.add_edge("persist_outreach_iteration", "review_outreach_content")
    graph.add_edge("review_outreach_content", "review_outreach_language")
    graph.add_edge("review_outreach_language", "route_review_result")
    graph.add_conditional_edges("route_review_result", _route_review)
    graph.add_edge("await_revision_input", "apply_revision_input")
    graph.add_conditional_edges("apply_revision_input", _route_revision_apply)
    graph.add_edge("finalize_case", END)
    return graph


def compile_graph(checkpointer: Any = None) -> Any:
    """Compile the production graph with an injectable durable saver."""

    return build_graph().compile(
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver()
    )


def create_initial_state(
    user_id: str,
    case_id: str,
    *,
    messages: list[dict[str, Any]] | None = None,
) -> FullGraphState:
    state = create_jmr_graph_state(user_id, case_id, messages=messages or [])
    state.update(
        application_fields={},
        persisted_application_fields={},
        application_extraction={},
        application_validation={},
        application_resume_payload=None,
        application_case_version=1,
        application_stage_done=False,
        retrieval_workers_needed=["publication", "kaken"],
        retrieval_run_ids=[],
        retrieval_resume_payload=None,
        retrieval_cancelled=False,
        memory_confirmation_needed=False,
        memory_resume_payload=None,
        memory_confirmed=False,
        allow_unpersonalized=False,
        selected_paper_evidence_ids=[],
        selected_papers=[],
        direction_resume_payload=None,
        selected_custom_direction_ref=None,
        content_review_status=None,
        language_review_status=None,
        revision_input_ref=None,
        revision_resume_payload=None,
    )
    return state
