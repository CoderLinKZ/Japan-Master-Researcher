"""Static node metadata and legal business-stage transitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from jmr.domain import WorkflowStage


class NodeKind(StrEnum):
    """Whether a node is model-driven, deterministic, or interrupting."""

    LLM = "LLM"
    DETERMINISTIC = "DETERMINISTIC"
    INTERRUPT = "INTERRUPT"


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Implementation contract for one target graph node."""

    name: str
    stage: WorkflowStage
    kind: NodeKind
    paradigm: str
    agent_tools: tuple[str, ...] = ()
    host_tools: tuple[str, ...] = ()
    max_model_calls: int = 0


def _node(
    name: str,
    stage: WorkflowStage,
    kind: NodeKind,
    paradigm: str,
    *,
    agent_tools: tuple[str, ...] = (),
    host_tools: tuple[str, ...] = (),
    max_model_calls: int = 0,
) -> NodeSpec:
    return NodeSpec(
        name=name,
        stage=stage,
        kind=kind,
        paradigm=paradigm,
        agent_tools=agent_tools,
        host_tools=host_tools,
        max_model_calls=max_model_calls,
    )


_NODE_SPECS = (
    _node(
        "initialize_case",
        WorkflowStage.VALIDATING_APPLICATION_INPUTS,
        NodeKind.DETERMINISTIC,
        "Deterministic Pipeline",
        host_tools=("mcp__case_data__create_case",),
    ),
    _node(
        "load_case_context",
        WorkflowStage.VALIDATING_APPLICATION_INPUTS,
        NodeKind.DETERMINISTIC,
        "Deterministic Retrieval",
        host_tools=("mcp__case_data__get_case_context",),
    ),
    _node(
        "extract_application_inputs",
        WorkflowStage.VALIDATING_APPLICATION_INPUTS,
        NodeKind.LLM,
        "Structured Extraction",
        max_model_calls=2,
    ),
    _node(
        "validate_application_inputs",
        WorkflowStage.VALIDATING_APPLICATION_INPUTS,
        NodeKind.DETERMINISTIC,
        "Deterministic Pipeline",
    ),
    _node(
        "await_application_inputs",
        WorkflowStage.VALIDATING_APPLICATION_INPUTS,
        NodeKind.INTERRUPT,
        "Human-in-the-loop",
    ),
    _node(
        "apply_application_inputs",
        WorkflowStage.VALIDATING_APPLICATION_INPUTS,
        NodeKind.DETERMINISTIC,
        "Deterministic Persistence",
        host_tools=(
            "mcp__case_data__update_case_target",
            "mcp__case_data__record_stage_transition",
        ),
    ),
    _node(
        "plan_research",
        WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
        NodeKind.LLM,
        "Plan-and-Execute",
        host_tools=("mcp__case_data__save_research_plan",),
        max_model_calls=2,
    ),
    _node(
        "dispatch_retrieval_workers",
        WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
        NodeKind.DETERMINISTIC,
        "Deterministic Router",
    ),
    _node(
        "publication_research_agent",
        WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
        NodeKind.LLM,
        "Bounded ReAct",
        agent_tools=(
            "mcp__scholar__discover_official_sources",
            "mcp__scholar__search_publications",
        ),
        host_tools=(
            "mcp__case_data__save_retrieval_batch",
            "mcp__case_data__save_source_object_metadata",
        ),
        max_model_calls=6,
    ),
    _node(
        "kaken_research_agent",
        WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
        NodeKind.LLM,
        "Bounded ReAct",
        agent_tools=("mcp__kaken__search_projects",),
        host_tools=("mcp__case_data__save_retrieval_batch",),
        max_model_calls=6,
    ),
    _node(
        "join_retrieval_results",
        WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
        NodeKind.DETERMINISTIC,
        "Deterministic Router",
    ),
    _node(
        "await_target_identity_confirmation",
        WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
        NodeKind.INTERRUPT,
        "Human-in-the-loop",
    ),
    _node(
        "apply_target_identity_confirmation",
        WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
        NodeKind.DETERMINISTIC,
        "Deterministic Persistence",
        host_tools=("mcp__case_data__update_case_target",),
    ),
    _node(
        "await_evidence_source_input",
        WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
        NodeKind.INTERRUPT,
        "Human-in-the-loop",
    ),
    _node(
        "apply_evidence_source_input",
        WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
        NodeKind.DETERMINISTIC,
        "Deterministic Persistence",
        host_tools=("mcp__case_data__update_case_target",),
    ),
    _node(
        "merge_and_verify_evidence",
        WorkflowStage.MERGING_AND_VERIFYING_EVIDENCE,
        NodeKind.DETERMINISTIC,
        "Deterministic Map-Reduce",
        host_tools=(
            "mcp__case_data__get_case_evidence",
            "mcp__case_data__save_verified_evidence",
        ),
    ),
    _node(
        "load_applicant_memory",
        WorkflowStage.GENERATING_RESEARCH_DIRECTIONS,
        NodeKind.DETERMINISTIC,
        "Deterministic Retrieval",
        host_tools=(
            "mcp__memory__list_memories",
            "mcp__memory__get_memory",
        ),
    ),
    _node(
        "await_applicant_memory_confirmation",
        WorkflowStage.GENERATING_RESEARCH_DIRECTIONS,
        NodeKind.INTERRUPT,
        "Human-in-the-loop",
    ),
    _node(
        "apply_applicant_memory_confirmation",
        WorkflowStage.GENERATING_RESEARCH_DIRECTIONS,
        NodeKind.DETERMINISTIC,
        "Deterministic Persistence",
        host_tools=(
            "mcp__memory__get_memory",
            "mcp__case_data__save_case_memory_selection",
        ),
    ),
    _node(
        "generate_direction_candidates",
        WorkflowStage.GENERATING_RESEARCH_DIRECTIONS,
        NodeKind.LLM,
        "Generator",
        max_model_calls=2,
    ),
    _node(
        "critique_direction_candidates",
        WorkflowStage.GENERATING_RESEARCH_DIRECTIONS,
        NodeKind.LLM,
        "Critic",
        max_model_calls=2,
    ),
    _node(
        "revise_direction_candidates",
        WorkflowStage.GENERATING_RESEARCH_DIRECTIONS,
        NodeKind.LLM,
        "Reflection",
        max_model_calls=2,
    ),
    _node(
        "persist_direction_batch",
        WorkflowStage.GENERATING_RESEARCH_DIRECTIONS,
        NodeKind.DETERMINISTIC,
        "Deterministic Persistence",
        host_tools=(
            "mcp__case_data__save_direction_batch",
            "mcp__case_data__record_stage_transition",
        ),
    ),
    _node(
        "await_direction_selection",
        WorkflowStage.AWAITING_DIRECTION_SELECTION,
        NodeKind.INTERRUPT,
        "Human-in-the-loop",
    ),
    _node(
        "apply_direction_selection",
        WorkflowStage.AWAITING_DIRECTION_SELECTION,
        NodeKind.DETERMINISTIC,
        "Deterministic Persistence",
        host_tools=(
            "mcp__case_data__save_direction_selection",
            "mcp__case_data__record_stage_transition",
        ),
    ),
    _node(
        "plan_outreach_paragraph",
        WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN,
        NodeKind.LLM,
        "Plan-to-Execute",
        max_model_calls=2,
    ),
    _node(
        "draft_outreach_paragraph",
        WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN,
        NodeKind.LLM,
        "Prompt Chaining",
        max_model_calls=2,
    ),
    _node(
        "persist_outreach_iteration",
        WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN,
        NodeKind.DETERMINISTIC,
        "Deterministic Persistence",
        host_tools=(
            "mcp__case_data__save_outreach_iteration",
            "mcp__case_data__record_stage_transition",
        ),
    ),
    _node(
        "review_outreach_content",
        WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN,
        NodeKind.LLM,
        "Evaluator",
        host_tools=("mcp__case_data__save_review_result",),
        max_model_calls=2,
    ),
    _node(
        "review_outreach_language",
        WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN,
        NodeKind.LLM,
        "Evaluator",
        host_tools=("mcp__case_data__save_review_result",),
        max_model_calls=2,
    ),
    _node(
        "route_review_result",
        WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN,
        NodeKind.DETERMINISTIC,
        "Deterministic Router",
    ),
    _node(
        "await_revision_input",
        WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN,
        NodeKind.INTERRUPT,
        "Human-in-the-loop",
    ),
    _node(
        "apply_revision_input",
        WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN,
        NodeKind.DETERMINISTIC,
        "Deterministic Persistence",
        host_tools=("mcp__case_data__save_review_result",),
    ),
    _node(
        "finalize_case",
        WorkflowStage.COMPLETED,
        NodeKind.DETERMINISTIC,
        "Deterministic Finalizer",
        host_tools=(
            "mcp__case_data__complete_case",
            "mcp__case_data__record_stage_transition",
        ),
    ),
)


def _build_manifest(specs: tuple[NodeSpec, ...]) -> Mapping[str, NodeSpec]:
    manifest: dict[str, NodeSpec] = {}
    for spec in specs:
        if spec.name in manifest:
            raise ValueError(f"duplicate node name: {spec.name}")
        if spec.kind is not NodeKind.LLM and spec.agent_tools:
            raise ValueError(f"non-LLM node cannot expose agent tools: {spec.name}")
        manifest[spec.name] = spec
    return MappingProxyType(manifest)


NODE_MANIFEST = _build_manifest(_NODE_SPECS)


LEGAL_STAGE_TRANSITIONS: Mapping[
    WorkflowStage,
    tuple[WorkflowStage, ...],
] = MappingProxyType(
    {
        WorkflowStage.VALIDATING_APPLICATION_INPUTS: (
            WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
        ),
        WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE: (
            WorkflowStage.MERGING_AND_VERIFYING_EVIDENCE,
        ),
        WorkflowStage.MERGING_AND_VERIFYING_EVIDENCE: (
            WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
            WorkflowStage.GENERATING_RESEARCH_DIRECTIONS,
        ),
        WorkflowStage.GENERATING_RESEARCH_DIRECTIONS: (
            WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
            WorkflowStage.AWAITING_DIRECTION_SELECTION,
        ),
        WorkflowStage.AWAITING_DIRECTION_SELECTION: (
            WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN,
        ),
        WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN: (
            WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN,
        ),
        WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN: (
            WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN,
            WorkflowStage.COMPLETED,
        ),
        WorkflowStage.COMPLETED: (),
    }
)


def nodes_for_stage(stage: WorkflowStage | str) -> tuple[NodeSpec, ...]:
    """Return all registered nodes for one valid business stage."""

    normalized = WorkflowStage(stage)
    return tuple(spec for spec in _NODE_SPECS if spec.stage is normalized)


def agent_tools_for(node_name: str) -> tuple[str, ...]:
    """Return the exact model-visible MCP allowlist for a node."""

    try:
        return NODE_MANIFEST[node_name].agent_tools
    except KeyError as exc:
        raise ValueError(f"unknown node: {node_name}") from exc


def can_transition(
    current: WorkflowStage | str,
    target: WorkflowStage | str,
) -> bool:
    """Return whether a business-stage edge is declared by the target graph."""

    normalized_current = WorkflowStage(current)
    normalized_target = WorkflowStage(target)
    return normalized_target in LEGAL_STAGE_TRANSITIONS[normalized_current]
