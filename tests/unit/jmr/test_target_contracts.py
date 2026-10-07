"""Target domain, graph-state, reducer, and invocation identity contracts."""

from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from typing import get_args, get_type_hints

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.domain import (  # noqa: E402
    EvidenceVerificationStatus,
    InterruptKind,
    MCPStatus,
    MemoryStatus,
    NodeExecutionStatus,
    ReviewStatus,
    RunStatus,
    WorkflowStage,
)
from jmr.graph.identity import (  # noqa: E402
    InvocationIdentityError,
    validate_invocation_identity,
)
from jmr.graph.state import (  # noqa: E402
    GRAPH_STATE_SCHEMA_VERSION,
    JMRGraphState,
    add_bounded_messages,
    create_jmr_graph_state,
    deserialize_jmr_graph_state,
    keep_immutable_identifier,
    merge_retrieval_refs,
    merge_unique_strings,
    messages_as_dicts,
    normalize_jmr_graph_state,
    serialize_jmr_graph_state,
)
from jmr.runtime import JMRRuntimeContext  # noqa: E402


class DomainEnumContractTests(unittest.TestCase):
    def test_domain_enums_have_the_complete_documented_values(self) -> None:
        expected_values = {
            WorkflowStage: {
                "VALIDATING_APPLICATION_INPUTS",
                "RETRIEVING_RESEARCH_EVIDENCE",
                "MERGING_AND_VERIFYING_EVIDENCE",
                "GENERATING_RESEARCH_DIRECTIONS",
                "AWAITING_DIRECTION_SELECTION",
                "DRAFTING_OUTREACH_RESEARCH_PLAN",
                "REVIEWING_OUTREACH_RESEARCH_PLAN",
                "COMPLETED",
            },
            RunStatus: {
                "READY",
                "RUNNING",
                "WAITING_FOR_USER",
                "RETRYING",
                "BLOCKED",
                "FAILED",
                "COMPLETED",
                "CANCELLED",
            },
            NodeExecutionStatus: {
                "PENDING",
                "RUNNING",
                "SUCCEEDED",
                "PARTIAL",
                "INTERRUPTED",
                "SKIPPED",
                "BLOCKED",
                "FAILED",
            },
            MCPStatus: {
                "SUCCESS",
                "PARTIAL",
                "NO_RESULT",
                "NEEDS_USER_CONFIRMATION",
                "BLOCKED",
                "CONFLICT",
                "FAILED",
            },
            EvidenceVerificationStatus: {
                "UNVERIFIED",
                "PARTIAL",
                "VERIFIED",
                "REJECTED",
            },
            ReviewStatus: {"PASS", "REVISE", "BLOCKED"},
            MemoryStatus: {"ACTIVE", "SUPERSEDED", "DELETED"},
            InterruptKind: {
                "APPLICATION_INPUT_REQUIRED",
                "TARGET_IDENTITY_CONFIRMATION_REQUIRED",
                "EVIDENCE_SOURCE_REQUIRED",
                "APPLICANT_MEMORY_CONFIRMATION_REQUIRED",
                "DIRECTION_SELECTION_REQUIRED",
                "REVISION_INPUT_REQUIRED",
            },
        }

        for enum_type, expected in expected_values.items():
            with self.subTest(enum=enum_type.__name__):
                self.assertEqual({member.value for member in enum_type}, expected)
                with self.assertRaises(ValueError):
                    enum_type("UNKNOWN_VALUE")


class GraphStateContractTests(unittest.TestCase):
    def test_state_schema_contains_only_the_documented_minimal_fields(self) -> None:
        expected_fields = {
            "schema_version",
            "user_id",
            "case_id",
            "workflow_stage",
            "run_status",
            "messages",
            "case_context_id",
            "research_plan_id",
            "retrieval_refs",
            "verified_evidence_bundle_id",
            "selected_memory_ids",
            "direction_batch_id",
            "selected_direction_ids",
            "current_outreach_iteration_id",
            "generation_revision_round",
            "review_round",
            "pending_interrupt",
            "warnings",
            "errors",
        }

        self.assertEqual(set(JMRGraphState.__annotations__), expected_fields)

    def test_new_state_is_id_only_and_has_deterministic_defaults(self) -> None:
        state = create_jmr_graph_state(user_id="user-1", case_id="case-1")

        self.assertEqual(state["schema_version"], GRAPH_STATE_SCHEMA_VERSION)
        self.assertEqual(state["user_id"], "user-1")
        self.assertEqual(state["case_id"], "case-1")
        self.assertEqual(
            state["workflow_stage"],
            WorkflowStage.VALIDATING_APPLICATION_INPUTS.value,
        )
        self.assertEqual(state["run_status"], RunStatus.READY.value)
        self.assertIs(type(state["workflow_stage"]), str)
        self.assertIs(type(state["run_status"]), str)
        self.assertEqual(state["messages"], [])
        self.assertEqual(state["retrieval_refs"], [])
        self.assertEqual(state["selected_memory_ids"], [])
        self.assertEqual(state["selected_direction_ids"], [])
        self.assertEqual(state["generation_revision_round"], 0)
        self.assertEqual(state["review_round"], 0)
        self.assertEqual(state["warnings"], [])
        self.assertEqual(state["errors"], [])
        for reference_field in (
            "case_context_id",
            "research_plan_id",
            "verified_evidence_bundle_id",
            "direction_batch_id",
            "current_outreach_iteration_id",
            "pending_interrupt",
        ):
            self.assertIsNone(state.get(reference_field))

        forbidden_legacy_fields = {
            "phase",
            "status",
            "workflow_context",
            "artifact_refs",
            "incoming_user_text",
            "answer",
            "planned_phase",
        }
        self.assertTrue(forbidden_legacy_fields.isdisjoint(state))

    def test_unknown_state_fields_are_rejected(self) -> None:
        state = dict(create_jmr_graph_state(user_id="user-1", case_id="case-1"))
        state["unexpected_field"] = "not allowed"

        with self.assertRaises((TypeError, ValueError)):
            normalize_jmr_graph_state(state)

    def test_business_bodies_cannot_be_embedded_in_state(self) -> None:
        forbidden_business_fields = {
            "publication_records": [{"title": "paper"}],
            "kaken_projects": [{"title": "project"}],
            "memory_content": {"education": "full profile"},
            "direction_text": "full research direction",
            "outreach_paragraph": "full outreach draft",
            "tool_result": {"records": ["full result"]},
        }

        for field_name, value in forbidden_business_fields.items():
            with self.subTest(field=field_name):
                state = dict(
                    create_jmr_graph_state(
                        user_id="user-1",
                        case_id="case-1",
                    )
                )
                state[field_name] = value
                with self.assertRaises((TypeError, ValueError)):
                    normalize_jmr_graph_state(state)

    def test_runtime_resources_cannot_be_embedded_in_state(self) -> None:
        for field_name in (
            "runtime_context",
            "model_client",
            "mcp_client",
            "database_connection",
            "repository",
            "file_handle",
        ):
            with self.subTest(field=field_name):
                state = dict(
                    create_jmr_graph_state(
                        user_id="user-1",
                        case_id="case-1",
                    )
                )
                state[field_name] = object()
                with self.assertRaises((TypeError, ValueError)):
                    normalize_jmr_graph_state(state)

        state = dict(create_jmr_graph_state(user_id="user-1", case_id="case-1"))
        state["messages"] = [{"role": "user", "content": object()}]
        with self.assertRaises((TypeError, ValueError)):
            normalize_jmr_graph_state(state)

    def test_state_round_trips_through_plain_json(self) -> None:
        state = dict(create_jmr_graph_state(user_id="user-1", case_id="case-1"))
        state.update(
            {
                "workflow_stage": WorkflowStage.RETRIEVING_RESEARCH_EVIDENCE.value,
                "run_status": RunStatus.WAITING_FOR_USER.value,
                "case_context_id": "context-1",
                "research_plan_id": "plan-1",
                "retrieval_refs": [
                    {
                        "worker_kind": "publication",
                        "retrieval_run_id": "retrieval-1",
                        "mcp_status": MCPStatus.PARTIAL.value,
                    }
                ],
                "selected_memory_ids": ["memory-1"],
                "pending_interrupt": {
                    "kind": InterruptKind.EVIDENCE_SOURCE_REQUIRED.value,
                    "payload_ref": "interrupt-payload-1",
                    "resume_schema_version": 1,
                    "issued_at": "2026-09-19T00:00:00+00:00",
                },
                "warnings": ["source coverage is partial"],
            }
        )

        normalized = normalize_jmr_graph_state(state)
        payload = serialize_jmr_graph_state(normalized)
        encoded = json.dumps(payload, ensure_ascii=False)
        restored = deserialize_jmr_graph_state(json.loads(encoded))

        self.assertEqual(restored, normalized)

    def test_state_round_trips_through_langgraph_msgpack(self) -> None:
        state = create_jmr_graph_state(
            user_id="user-1",
            case_id="case-1",
            messages=[{"role": "user", "content": "research request"}],
        )
        serializer = JsonPlusSerializer()

        payload = serializer.dumps_typed(serialize_jmr_graph_state(state))
        restored_payload = serializer.loads_typed(payload)
        restored = deserialize_jmr_graph_state(restored_payload)

        self.assertEqual(payload[0], "msgpack")
        self.assertEqual(restored, state)

    def test_real_add_messages_output_is_checkpoint_safe(self) -> None:
        reduced_messages = add_messages(
            [],
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        )
        state = dict(
            create_jmr_graph_state(
                user_id="user-1",
                case_id="case-1",
            )
        )
        state["messages"] = reduced_messages

        payload = serialize_jmr_graph_state(state)
        restored = deserialize_jmr_graph_state(payload)

        restored_messages = messages_as_dicts(restored["messages"])
        self.assertEqual(
            [message["role"] for message in restored_messages],
            ["user", "assistant"],
        )
        self.assertEqual(
            [message["content"] for message in restored_messages],
            ["question", "answer"],
        )

    def test_unknown_schema_and_out_of_range_rounds_are_rejected(self) -> None:
        for field_name, value in (
            ("schema_version", 999),
            ("schema_version", True),
            ("schema_version", float(GRAPH_STATE_SCHEMA_VERSION)),
            ("schema_version", str(GRAPH_STATE_SCHEMA_VERSION)),
            ("generation_revision_round", 3),
            ("review_round", -1),
        ):
            with self.subTest(field=field_name):
                state = dict(
                    create_jmr_graph_state(
                        user_id="user-1",
                        case_id="case-1",
                    )
                )
                state[field_name] = value
                with self.assertRaises((TypeError, ValueError)):
                    normalize_jmr_graph_state(state)

    def test_large_business_body_cannot_hide_in_short_term_messages(self) -> None:
        with self.assertRaises(ValueError):
            create_jmr_graph_state(
                user_id="user-1",
                case_id="case-1",
                messages=[{"role": "user", "content": "x" * (17 * 1024)}],
            )

    def test_missing_required_checkpoint_fields_are_rejected(self) -> None:
        complete = dict(
            create_jmr_graph_state(
                user_id="user-1",
                case_id="case-1",
            )
        )
        for field_name in ("schema_version", "workflow_stage", "review_round"):
            with self.subTest(field=field_name):
                incomplete = dict(complete)
                incomplete.pop(field_name)
                with self.assertRaises(ValueError):
                    deserialize_jmr_graph_state(incomplete)


class StateReducerContractTests(unittest.TestCase):
    def test_identity_reducer_accepts_replay_and_rejects_replacement(self) -> None:
        self.assertEqual(keep_immutable_identifier("", "case-1"), "case-1")
        self.assertEqual(keep_immutable_identifier(None, "case-1"), "case-1")
        self.assertEqual(
            keep_immutable_identifier("case-1", "case-1"),
            "case-1",
        )
        self.assertEqual(keep_immutable_identifier("case-1", None), "case-1")
        with self.assertRaises(ValueError):
            keep_immutable_identifier("case-1", "case-2")

    def test_identity_reducer_works_in_a_real_checkpointed_state_graph(
        self,
    ) -> None:
        builder = StateGraph(JMRGraphState)
        builder.add_node("noop", lambda _state: {})
        builder.add_edge(START, "noop")
        builder.add_edge("noop", END)
        graph = builder.compile(checkpointer=InMemorySaver())
        config = {
            "configurable": {
                "user_id": "user-1",
                "thread_id": "case-1",
            }
        }

        result = graph.invoke(
            create_jmr_graph_state("user-1", "case-1"),
            config,
        )
        self.assertEqual(result["user_id"], "user-1")
        self.assertEqual(result["case_id"], "case-1")

        with self.assertRaises(ValueError):
            graph.invoke({"user_id": "user-2"}, config)

    def test_state_annotations_bind_the_declared_langgraph_reducers(self) -> None:
        hints = get_type_hints(JMRGraphState, include_extras=True)

        self.assertIn(add_bounded_messages, get_args(hints["messages"]))
        self.assertIn(merge_retrieval_refs, get_args(hints["retrieval_refs"]))
        self.assertIn(merge_unique_strings, get_args(hints["warnings"]))
        self.assertIn(merge_unique_strings, get_args(hints["errors"]))
        self.assertIn(
            keep_immutable_identifier,
            get_args(hints["user_id"]),
        )
        self.assertIn(
            keep_immutable_identifier,
            get_args(hints["case_id"]),
        )

    def test_unique_string_reducer_is_stable_and_does_not_mutate_inputs(
        self,
    ) -> None:
        left = ["first", "shared"]
        right = ["shared", "second", "first"]
        left_before = list(left)
        right_before = list(right)

        merged = merge_unique_strings(left, right)

        self.assertEqual(merged, ["first", "shared", "second"])
        self.assertEqual(left, left_before)
        self.assertEqual(right, right_before)

    def test_retrieval_ref_reducer_deduplicates_exact_replays_in_order(
        self,
    ) -> None:
        publication_ref = {
            "worker_kind": "publication",
            "retrieval_run_id": "retrieval-publication-1",
            "mcp_status": MCPStatus.SUCCESS.value,
        }
        kaken_ref = {
            "worker_kind": "kaken",
            "retrieval_run_id": "retrieval-kaken-1",
            "mcp_status": MCPStatus.PARTIAL.value,
        }
        left = [publication_ref]
        right = [deepcopy(publication_ref), kaken_ref]
        left_before = deepcopy(left)
        right_before = deepcopy(right)

        merged = merge_retrieval_refs(left, right)

        self.assertEqual(merged, [publication_ref, kaken_ref])
        self.assertEqual(left, left_before)
        self.assertEqual(right, right_before)


class InvocationIdentityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = create_jmr_graph_state(
            user_id="user-1",
            case_id="case-1",
        )

    def test_matching_user_case_and_thread_identity_is_accepted(self) -> None:
        validate_invocation_identity(
            self.state,
            {
                "configurable": {
                    "user_id": "user-1",
                    "thread_id": "case-1",
                }
            },
        )

    def test_mismatched_or_missing_invocation_identity_is_rejected(self) -> None:
        invalid_configs = {
            "different user": {
                "configurable": {
                    "user_id": "user-2",
                    "thread_id": "case-1",
                }
            },
            "different thread": {
                "configurable": {
                    "user_id": "user-1",
                    "thread_id": "case-2",
                }
            },
            "missing user": {"configurable": {"thread_id": "case-1"}},
            "missing thread": {"configurable": {"user_id": "user-1"}},
            "missing configurable": {},
        }

        for scenario, config in invalid_configs.items():
            with self.subTest(scenario=scenario):
                with self.assertRaises(InvocationIdentityError):
                    validate_invocation_identity(self.state, config)


class RuntimeContextContractTests(unittest.TestCase):
    def test_runtime_capabilities_are_injected_without_entering_state(
        self,
    ) -> None:
        capabilities = {
            "model_registry": object(),
            "mcp_gateway": object(),
            "case_repository": object(),
            "memory_store": object(),
            "object_store": object(),
            "retry_policy": object(),
            "security_policy": object(),
            "event_sink": object(),
        }
        context = JMRRuntimeContext(
            principal_user_id="user-1",
            **capabilities,
        )

        self.assertEqual(context.principal_user_id, "user-1")
        for field_name, capability in capabilities.items():
            with self.subTest(field=field_name):
                self.assertIs(getattr(context, field_name), capability)

        state = create_jmr_graph_state(
            user_id="user-1",
            case_id="case-1",
        )
        self.assertTrue(set(capabilities).isdisjoint(state))


if __name__ == "__main__":
    unittest.main()
