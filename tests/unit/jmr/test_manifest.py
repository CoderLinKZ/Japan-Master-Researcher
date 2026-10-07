"""Target node manifest and routing-contract tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.domain import WorkflowStage  # noqa: E402
from jmr.graph.manifest import (  # noqa: E402
    LEGAL_STAGE_TRANSITIONS,
    NODE_MANIFEST,
    NodeKind,
    agent_tools_for,
    can_transition,
    nodes_for_stage,
)


class NodeManifestTests(unittest.TestCase):
    def test_all_documented_nodes_are_registered_once(self) -> None:
        expected_by_stage = {
            WorkflowStage.VALIDATING_APPLICATION_INPUTS: (
                "initialize_case",
                "load_case_context",
                "extract_application_inputs",
                "validate_application_inputs",
                "await_application_inputs",
                "apply_application_inputs",
            ),
            WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE: (
                "plan_research",
                "dispatch_retrieval_workers",
                "publication_research_agent",
                "kaken_research_agent",
                "join_retrieval_results",
                "await_target_identity_confirmation",
                "apply_target_identity_confirmation",
                "await_evidence_source_input",
                "apply_evidence_source_input",
            ),
            WorkflowStage.MERGING_AND_VERIFYING_EVIDENCE: (
                "merge_and_verify_evidence",
            ),
            WorkflowStage.GENERATING_RESEARCH_DIRECTIONS: (
                "load_applicant_memory",
                "await_applicant_memory_confirmation",
                "apply_applicant_memory_confirmation",
                "generate_direction_candidates",
                "critique_direction_candidates",
                "revise_direction_candidates",
                "persist_direction_batch",
            ),
            WorkflowStage.AWAITING_DIRECTION_SELECTION: (
                "await_direction_selection",
                "apply_direction_selection",
            ),
            WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN: (
                "plan_outreach_paragraph",
                "draft_outreach_paragraph",
                "persist_outreach_iteration",
            ),
            WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN: (
                "review_outreach_content",
                "review_outreach_language",
                "route_review_result",
                "await_revision_input",
                "apply_revision_input",
            ),
            WorkflowStage.COMPLETED: ("finalize_case",),
        }
        expected = {
            node_name
            for node_names in expected_by_stage.values()
            for node_name in node_names
        }

        self.assertEqual(set(NODE_MANIFEST), expected)
        for stage, expected_names in expected_by_stage.items():
            with self.subTest(stage=stage.value):
                self.assertEqual(
                    tuple(spec.name for spec in nodes_for_stage(stage)),
                    expected_names,
                )
                self.assertTrue(
                    all(
                        NODE_MANIFEST[node_name].stage is stage
                        for node_name in expected_names
                    )
                )

    def test_only_retrieval_workers_receive_model_visible_tools(self) -> None:
        self.assertEqual(
            agent_tools_for("publication_research_agent"),
            (
                "mcp__scholar__discover_official_sources",
                "mcp__scholar__search_publications",
            ),
        )
        self.assertEqual(
            agent_tools_for("kaken_research_agent"),
            ("mcp__kaken__search_projects",),
        )
        for name, spec in NODE_MANIFEST.items():
            if name in {
                "publication_research_agent",
                "kaken_research_agent",
            }:
                continue
            with self.subTest(node=name):
                self.assertEqual(spec.agent_tools, ())

    def test_host_write_tools_are_never_model_visible(self) -> None:
        expected_host_tools = {
            "initialize_case": ("mcp__case_data__create_case",),
            "load_case_context": ("mcp__case_data__get_case_context",),
            "apply_application_inputs": (
                "mcp__case_data__update_case_target",
                "mcp__case_data__record_stage_transition",
            ),
            "plan_research": ("mcp__case_data__save_research_plan",),
            "publication_research_agent": (
                "mcp__case_data__save_retrieval_batch",
                "mcp__case_data__save_source_object_metadata",
            ),
            "kaken_research_agent": ("mcp__case_data__save_retrieval_batch",),
            "apply_target_identity_confirmation": (
                "mcp__case_data__update_case_target",
            ),
            "apply_evidence_source_input": ("mcp__case_data__update_case_target",),
            "merge_and_verify_evidence": (
                "mcp__case_data__get_case_evidence",
                "mcp__case_data__save_verified_evidence",
            ),
            "load_applicant_memory": (
                "mcp__memory__list_memories",
                "mcp__memory__get_memory",
            ),
            "apply_applicant_memory_confirmation": (
                "mcp__memory__get_memory",
                "mcp__case_data__save_case_memory_selection",
            ),
            "persist_direction_batch": (
                "mcp__case_data__save_direction_batch",
                "mcp__case_data__record_stage_transition",
            ),
            "apply_direction_selection": (
                "mcp__case_data__save_direction_selection",
                "mcp__case_data__record_stage_transition",
            ),
            "persist_outreach_iteration": (
                "mcp__case_data__save_outreach_iteration",
                "mcp__case_data__record_stage_transition",
            ),
            "review_outreach_content": ("mcp__case_data__save_review_result",),
            "review_outreach_language": ("mcp__case_data__save_review_result",),
            "apply_revision_input": ("mcp__case_data__save_review_result",),
            "finalize_case": (
                "mcp__case_data__complete_case",
                "mcp__case_data__record_stage_transition",
            ),
        }

        for spec in NODE_MANIFEST.values():
            with self.subTest(node=spec.name):
                self.assertEqual(
                    spec.host_tools,
                    expected_host_tools.get(spec.name, ()),
                )
                self.assertTrue(set(spec.agent_tools).isdisjoint(spec.host_tools))
                if spec.kind is not NodeKind.LLM:
                    self.assertEqual(spec.agent_tools, ())

    def test_model_call_limits_match_agent_paradigms(self) -> None:
        react_workers = {
            "publication_research_agent",
            "kaken_research_agent",
        }
        for spec in NODE_MANIFEST.values():
            with self.subTest(node=spec.name):
                if spec.name in react_workers:
                    self.assertEqual(spec.max_model_calls, 6)
                elif spec.kind is NodeKind.LLM:
                    # One normal call plus at most one schema/format retry.
                    self.assertEqual(spec.max_model_calls, 2)
                else:
                    self.assertEqual(spec.max_model_calls, 0)

    def test_node_kind_and_paradigm_assignments_are_exact(self) -> None:
        llm_paradigms = {
            "extract_application_inputs": "Structured Extraction",
            "plan_research": "Plan-and-Execute",
            "publication_research_agent": "Bounded ReAct",
            "kaken_research_agent": "Bounded ReAct",
            "generate_direction_candidates": "Generator",
            "critique_direction_candidates": "Critic",
            "revise_direction_candidates": "Reflection",
            "plan_outreach_paragraph": "Plan-to-Execute",
            "draft_outreach_paragraph": "Prompt Chaining",
            "review_outreach_content": "Evaluator",
            "review_outreach_language": "Evaluator",
        }
        interrupt_nodes = {
            "await_application_inputs",
            "await_target_identity_confirmation",
            "await_evidence_source_input",
            "await_applicant_memory_confirmation",
            "await_direction_selection",
            "await_revision_input",
        }
        deterministic_subtypes = {
            "load_case_context": "Deterministic Retrieval",
            "apply_application_inputs": "Deterministic Persistence",
            "dispatch_retrieval_workers": "Deterministic Router",
            "join_retrieval_results": "Deterministic Router",
            "apply_target_identity_confirmation": "Deterministic Persistence",
            "apply_evidence_source_input": "Deterministic Persistence",
            "merge_and_verify_evidence": "Deterministic Map-Reduce",
            "load_applicant_memory": "Deterministic Retrieval",
            "apply_applicant_memory_confirmation": "Deterministic Persistence",
            "persist_direction_batch": "Deterministic Persistence",
            "apply_direction_selection": "Deterministic Persistence",
            "persist_outreach_iteration": "Deterministic Persistence",
            "route_review_result": "Deterministic Router",
            "apply_revision_input": "Deterministic Persistence",
            "finalize_case": "Deterministic Finalizer",
        }

        for name, spec in NODE_MANIFEST.items():
            with self.subTest(node=name):
                if name in llm_paradigms:
                    self.assertIs(spec.kind, NodeKind.LLM)
                    self.assertEqual(spec.paradigm, llm_paradigms[name])
                elif name in interrupt_nodes:
                    self.assertIs(spec.kind, NodeKind.INTERRUPT)
                    self.assertEqual(spec.paradigm, "Human-in-the-loop")
                else:
                    self.assertIs(spec.kind, NodeKind.DETERMINISTIC)
                    self.assertEqual(
                        spec.paradigm,
                        deterministic_subtypes.get(
                            name,
                            "Deterministic Pipeline",
                        ),
                    )

    def test_legal_business_stage_edges_are_complete_and_bounded(self) -> None:
        self.assertEqual(set(LEGAL_STAGE_TRANSITIONS), set(WorkflowStage))
        self.assertTrue(
            can_transition(
                WorkflowStage.VALIDATING_APPLICATION_INPUTS,
                WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
            )
        )
        self.assertTrue(
            can_transition(
                WorkflowStage.MERGING_AND_VERIFYING_EVIDENCE,
                WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
            )
        )
        self.assertTrue(
            can_transition(
                WorkflowStage.GENERATING_RESEARCH_DIRECTIONS,
                WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE,
            )
        )
        self.assertTrue(
            can_transition(
                WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN,
                WorkflowStage.DRAFTING_OUTREACH_RESEARCH_PLAN,
            )
        )
        self.assertTrue(
            can_transition(
                WorkflowStage.REVIEWING_OUTREACH_RESEARCH_PLAN,
                WorkflowStage.COMPLETED,
            )
        )
        self.assertFalse(
            can_transition(
                WorkflowStage.VALIDATING_APPLICATION_INPUTS,
                WorkflowStage.COMPLETED,
            )
        )
        self.assertEqual(
            LEGAL_STAGE_TRANSITIONS[WorkflowStage.COMPLETED],
            (),
        )


if __name__ == "__main__":
    unittest.main()
