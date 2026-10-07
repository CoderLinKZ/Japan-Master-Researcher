"""P4-P6 formal graph integration tests using only injected fakes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from jmr.graph import compile_graph, create_initial_state  # noqa: E402
from jmr.graph.nodes.common import read_json_artifact  # noqa: E402
from jmr.persistence import InMemoryCaseRepository, InMemoryObjectStore  # noqa: E402
from jmr.runtime import JMRRuntimeContext  # noqa: E402
from tests.fixtures.jmr_runtime import (  # noqa: E402
    DeterministicModelRegistry,
    FakeRetrievalGateway,
)


class FullWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryCaseRepository()
        self.gateway = FakeRetrievalGateway()
        self.runtime = JMRRuntimeContext(
            case_repository=self.repository,
            model_registry=DeterministicModelRegistry(),
            mcp_gateway=self.gateway,
            object_store=InMemoryObjectStore(),
        )
        self.graph = compile_graph(InMemorySaver())
        self.config = {"configurable": {"user_id": "user-1", "thread_id": "case-1"}}

    def _start(self):
        return self.graph.invoke(
            create_initial_state(
                "user-1",
                "case-1",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "大学：东京大学，研究科：工学系研究科，" + "教授：山田太郎"
                        ),
                    }
                ],
            ),
            self.config,
            context=self.runtime,
        )

    def test_complete_case_crosses_parallel_retrieval_and_three_later_phases(
        self,
    ) -> None:
        first = self._start()
        self.assertEqual(
            first["pending_interrupt"]["kind"],
            "APPLICANT_MEMORY_CONFIRMATION_REQUIRED",
        )
        self.assertEqual(
            set(self.gateway.calls),
            {
                "mcp__scholar__search_publications",
                "mcp__kaken__search_projects",
            },
        )
        self.assertEqual(len(first["retrieval_refs"]), 2)
        paper_id = read_json_artifact(self.runtime, first["paper_options_ref"])[0][
            "evidence_id"
        ]

        second = self.graph.invoke(
            Command(
                resume={
                    "selected_paper_evidence_ids": [paper_id],
                    "selected_memory_ids": [],
                    "allow_unpersonalized": True,
                    "idempotency_key": "memory-choice-1",
                }
            ),
            self.config,
            context=self.runtime,
        )
        self.assertEqual(second["workflow_stage"], "AWAITING_DIRECTION_SELECTION")
        self.assertEqual(len(second["selected_papers"]), 1)
        self.assertEqual(
            second["selected_papers"][0]["title"],
            "Evaluating Conversational Search with Large Language Models",
        )
        batch = self.repository.get_direction_batch(
            user_id="user-1",
            case_id="case-1",
            direction_batch_id=second["direction_batch_id"],
        )
        self.assertEqual(len(batch["directions"]), 3)
        self.assertTrue(
            all(
                set(item["evidence_ids"]).issubset(
                    set(second["selected_paper_evidence_ids"])
                )
                for item in batch["directions"]
            )
        )
        self.assertTrue(
            all(
                "Evaluating Conversational Search" not in item["title"]
                for item in batch["directions"]
            )
        )

        final = self.graph.invoke(
            Command(
                resume={
                    "direction_batch_id": second["direction_batch_id"],
                    "selected_direction_ids": [batch["directions"][0]["direction_id"]],
                    "custom_direction": None,
                    "idempotency_key": "direction-choice-1",
                }
            ),
            self.config,
            context=self.runtime,
        )
        self.assertEqual(final["workflow_stage"], "COMPLETED")
        self.assertEqual(final["run_status"], "COMPLETED")
        self.assertNotIn("direction_candidates", final)
        self.assertNotIn("outreach_draft", final)
        self.assertNotIn("content_review", final)
        self.assertTrue(final["direction_candidates_ref"].startswith("memory://"))
        self.assertTrue(final["outreach_draft_ref"].startswith("memory://"))
        iteration = self.repository.get_outreach_iteration(
            user_id="user-1",
            case_id="case-1",
            iteration_id=final["current_outreach_iteration_id"],
        )
        self.assertGreaterEqual(len(iteration["content"]), 180)
        reviews = self.repository.list_review_results(
            user_id="user-1",
            case_id="case-1",
            iteration_id=final["current_outreach_iteration_id"],
        )
        self.assertEqual(reviews[-1]["status"], "PASS")

    def test_paper_selection_cannot_be_skipped_or_forged(self) -> None:
        self._start()
        invalid = self.graph.invoke(
            Command(
                resume={
                    "selected_paper_evidence_ids": [],
                    "selected_memory_ids": [],
                    "allow_unpersonalized": True,
                    "idempotency_key": "empty-paper-choice",
                }
            ),
            self.config,
            context=self.runtime,
        )
        self.assertEqual(
            invalid["pending_interrupt"]["kind"],
            "APPLICANT_MEMORY_CONFIRMATION_REQUIRED",
        )
        self.assertIn("请先选择 1–3 篇", invalid["errors"][-1])

    def test_invalid_direction_selection_repeats_interrupt_without_a_write(self):
        first = self._start()
        paper_id = read_json_artifact(self.runtime, first["paper_options_ref"])[0][
            "evidence_id"
        ]
        waiting = self.graph.invoke(
            Command(
                resume={
                    "selected_paper_evidence_ids": [paper_id],
                    "selected_memory_ids": [],
                    "allow_unpersonalized": True,
                    "idempotency_key": "memory-choice-1",
                }
            ),
            self.config,
            context=self.runtime,
        )
        invalid = self.graph.invoke(
            Command(
                resume={
                    "direction_batch_id": waiting["direction_batch_id"],
                    "selected_direction_ids": ["another-batch-direction"],
                    "custom_direction": None,
                    "idempotency_key": "bad-choice",
                }
            ),
            self.config,
            context=self.runtime,
        )
        self.assertEqual(invalid["run_status"], "WAITING_FOR_USER")
        self.assertEqual(
            invalid["pending_interrupt"]["kind"],
            "DIRECTION_SELECTION_REQUIRED",
        )

    def test_no_official_site_feedback_advances_with_existing_papers(self) -> None:
        class AmbiguousGateway(FakeRetrievalGateway):
            def call_tool(self, **kwargs):
                result = dict(super().call_tool(**kwargs))
                if kwargs["tool_name"] == "mcp__scholar__search_publications":
                    result["status"] = "NEEDS_USER_CONFIRMATION"
                return result

        runtime = JMRRuntimeContext(
            case_repository=self.repository,
            model_registry=DeterministicModelRegistry(),
            mcp_gateway=AmbiguousGateway(),
            object_store=InMemoryObjectStore(),
        )
        first = self.graph.invoke(
            create_initial_state(
                "user-1",
                "case-1",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "大学：东京大学，研究科：工学系研究科，" + "教授：山田太郎"
                        ),
                    }
                ],
            ),
            self.config,
            context=runtime,
        )
        self.assertEqual(
            first["pending_interrupt"]["kind"],
            "TARGET_IDENTITY_CONFIRMATION_REQUIRED",
        )

        resumed = self.graph.invoke(
            Command(
                resume={
                    "action": "no_official_site",
                    "confirmed_by_user": True,
                    "idempotency_key": "no-site-full-1",
                }
            ),
            self.config,
            context=runtime,
        )

        self.assertEqual(
            resumed["pending_interrupt"]["kind"],
            "APPLICANT_MEMORY_CONFIRMATION_REQUIRED",
        )
        self.assertEqual(
            self.repository.get_case(user_id="user-1", case_id="case-1")["target"][
                "official_site_status"
            ],
            "USER_REPORTED_NONE",
        )

    def test_no_site_and_no_other_evidence_requests_sources(self) -> None:
        class AmbiguousEmptyGateway(FakeRetrievalGateway):
            def __init__(self):
                super().__init__(with_records=False)

            def call_tool(self, **kwargs):
                result = dict(super().call_tool(**kwargs))
                if kwargs["tool_name"] == "mcp__scholar__search_publications":
                    result["status"] = "NEEDS_USER_CONFIRMATION"
                return result

        runtime = JMRRuntimeContext(
            case_repository=self.repository,
            model_registry=DeterministicModelRegistry(),
            mcp_gateway=AmbiguousEmptyGateway(),
            object_store=InMemoryObjectStore(),
        )
        first = self.graph.invoke(
            create_initial_state(
                "user-1",
                "case-1",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "大学：东京大学，研究科：工学系研究科，" + "教授：山田太郎"
                        ),
                    }
                ],
            ),
            self.config,
            context=runtime,
        )
        self.assertEqual(
            first["pending_interrupt"]["kind"],
            "TARGET_IDENTITY_CONFIRMATION_REQUIRED",
        )

        resumed = self.graph.invoke(
            Command(
                resume={
                    "action": "no_official_site",
                    "confirmed_by_user": True,
                    "idempotency_key": "no-site-empty-1",
                }
            ),
            self.config,
            context=runtime,
        )

        self.assertEqual(
            resumed["pending_interrupt"]["kind"], "EVIDENCE_SOURCE_REQUIRED"
        )

    def test_no_evidence_cannot_advance_to_direction_generation(self) -> None:
        repository = InMemoryCaseRepository()
        graph = compile_graph(InMemorySaver())
        result = graph.invoke(
            create_initial_state(
                "user-2",
                "case-2",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "大学：东京大学，研究科：工学系研究科，教授：山田太郎"
                        ),
                    }
                ],
            ),
            {"configurable": {"user_id": "user-2", "thread_id": "case-2"}},
            context=JMRRuntimeContext(
                case_repository=repository,
                model_registry=DeterministicModelRegistry(),
                mcp_gateway=FakeRetrievalGateway(with_records=False),
                object_store=InMemoryObjectStore(),
            ),
        )
        self.assertEqual(result["workflow_stage"], "RETRIEVING_RESEARCH_EVIDENCE")
        self.assertEqual(
            result["pending_interrupt"]["kind"], "EVIDENCE_SOURCE_REQUIRED"
        )

        invalid = graph.invoke(
            Command(
                resume={
                    "action": "adjust_scope",
                    "date_from": "2026-12-01",
                    "date_to": "2026-01-01",
                    "idempotency_key": "bad-dates",
                }
            ),
            {"configurable": {"user_id": "user-2", "thread_id": "case-2"}},
            context=JMRRuntimeContext(
                case_repository=repository,
                model_registry=DeterministicModelRegistry(),
                mcp_gateway=FakeRetrievalGateway(with_records=False),
                object_store=InMemoryObjectStore(),
            ),
        )
        self.assertEqual(invalid["run_status"], "WAITING_FOR_USER")
        self.assertEqual(
            invalid["pending_interrupt"]["kind"], "EVIDENCE_SOURCE_REQUIRED"
        )

    def test_blocked_direction_critic_returns_to_evidence_input(self) -> None:
        class BlockedRegistry(DeterministicModelRegistry):
            def for_node(self, node_name):
                if node_name == "critique_direction_candidates":
                    return self
                return super().for_node(node_name)

            def invoke_structured(self, **_kwargs):
                return {"status": "BLOCKED", "issues": ["证据不足"]}

        runtime = JMRRuntimeContext(
            case_repository=self.repository,
            model_registry=BlockedRegistry(),
            mcp_gateway=self.gateway,
            object_store=InMemoryObjectStore(),
        )
        first = self.graph.invoke(
            create_initial_state(
                "user-1",
                "case-1",
                messages=[
                    {
                        "role": "user",
                        "content": "大学：东京大学，研究科：工学系研究科，"
                        + "教授：山田太郎",
                    }
                ],
            ),
            self.config,
            context=runtime,
        )
        self.assertEqual(
            first["pending_interrupt"]["kind"], "APPLICANT_MEMORY_CONFIRMATION_REQUIRED"
        )
        paper_id = read_json_artifact(runtime, first["paper_options_ref"])[0][
            "evidence_id"
        ]
        blocked = self.graph.invoke(
            Command(
                resume={
                    "selected_paper_evidence_ids": [paper_id],
                    "selected_memory_ids": [],
                    "allow_unpersonalized": True,
                    "idempotency_key": "memory-choice-1",
                }
            ),
            self.config,
            context=runtime,
        )
        self.assertEqual(blocked["workflow_stage"], "RETRIEVING_RESEARCH_EVIDENCE")
        self.assertEqual(
            blocked["pending_interrupt"]["kind"], "EVIDENCE_SOURCE_REQUIRED"
        )


if __name__ == "__main__":
    unittest.main()
