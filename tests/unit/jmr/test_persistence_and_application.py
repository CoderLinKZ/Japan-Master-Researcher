"""P2 repository invariants and P3 interrupt/resume contract tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from jmr.cli import run_turn  # noqa: E402
from jmr.graph import compile_graph, create_initial_state  # noqa: E402
from jmr.graph.nodes.application import extract_application_inputs  # noqa: E402
from jmr.persistence import (  # noqa: E402
    IdempotencyConflictError,
    InMemoryCaseRepository,
    InMemoryObjectStore,
    OwnershipError,
    VersionConflictError,
)
from jmr.runtime import JMRRuntimeContext  # noqa: E402
from tests.fixtures.jmr_runtime import (  # noqa: E402
    DeterministicModelRegistry,
    FakeRetrievalGateway,
)


class _IncompleteApplicationModel:
    def invoke_structured(self, **_kwargs):
        return {
            "extracted": {},
            "explicit_corrections": [],
            "ambiguous_fragments": [],
        }


class _IncompleteApplicationModelRegistry:
    def for_node(self, _node_name):
        return _IncompleteApplicationModel()


class PersistenceContractTests(unittest.TestCase):
    def test_case_writes_are_idempotent_and_optimistically_locked(self) -> None:
        repository = InMemoryCaseRepository()
        created = repository.create_case(
            user_id="user-1",
            case_id="case-1",
            schema_version=2,
            idempotency_key="create-1",
        )
        replay = repository.create_case(
            user_id="user-1",
            case_id="case-1",
            schema_version=2,
            idempotency_key="create-1",
        )
        self.assertEqual(created, replay)

        with self.assertRaises(IdempotencyConflictError):
            repository.create_case(
                user_id="user-1",
                case_id="case-1",
                schema_version=999,
                idempotency_key="create-1",
            )

        updated = repository.update_target(
            user_id="user-1",
            case_id="case-1",
            target={"professor_name": "A"},
            expected_version=1,
            idempotency_key="target-1",
        )
        self.assertEqual(updated["version"], 2)
        with self.assertRaises(VersionConflictError):
            repository.update_target(
                user_id="user-1",
                case_id="case-1",
                target={"professor_name": "B"},
                expected_version=1,
                idempotency_key="target-2",
            )

    def test_case_reads_and_writes_are_user_scoped(self) -> None:
        repository = InMemoryCaseRepository()
        repository.create_case(
            user_id="user-1",
            case_id="case-1",
            schema_version=2,
            idempotency_key="create-1",
        )
        with self.assertRaises(OwnershipError):
            repository.get_case(user_id="user-2", case_id="case-1")

    def test_object_store_keeps_large_content_outside_business_state(self) -> None:
        store = InMemoryObjectStore()
        uri = store.put(
            object_key="case-1/source.html",
            content=b"body",
            content_type="text/html",
        )
        self.assertEqual(store.get(object_uri=uri), b"body")
        self.assertEqual(
            store.metadata(object_uri=uri, content_type="text/html")["byte_size"], 4
        )

    def test_business_versions_and_selection_records_are_kept_separately(self) -> None:
        repository = InMemoryCaseRepository()
        repository.create_case(
            user_id="user-1",
            case_id="case-1",
            schema_version=2,
            idempotency_key="create-1",
        )
        batch = repository.save_direction_batch(
            user_id="user-1",
            case_id="case-1",
            directions=[{"title": "方向 A"}, {"title": "方向 B"}],
            evidence_bundle_id=None,
            revision_round=0,
            idempotency_key="batch-1",
        )
        selection = repository.save_direction_selection(
            user_id="user-1",
            case_id="case-1",
            direction_batch_id=batch["direction_batch_id"],
            selected_direction_ids=[batch["direction_ids"][0]],
            custom_direction=None,
            idempotency_key="selection-1",
        )
        iteration = repository.save_outreach_iteration(
            user_id="user-1",
            case_id="case-1",
            content="草稿",
            citation_ids=["evidence-1"],
            parent_iteration_id=None,
            idempotency_key="iteration-1",
        )
        completed = repository.complete_case(
            user_id="user-1",
            case_id="case-1",
            final_iteration_id=iteration["iteration_id"],
            idempotency_key="complete-1",
        )
        self.assertEqual(selection["direction_batch_id"], batch["direction_batch_id"])
        self.assertEqual(completed["run_status"], "COMPLETED")


class ApplicationInterruptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryCaseRepository()
        self.runtime = JMRRuntimeContext(
            case_repository=self.repository,
            model_registry=_IncompleteApplicationModelRegistry(),
        )
        self.config = {"configurable": {"user_id": "user-1", "thread_id": "case-1"}}
        self.graph = compile_graph(InMemorySaver())

    def test_missing_inputs_interrupt_and_valid_resume_hands_off_to_retrieval(
        self,
    ) -> None:
        first = self.graph.invoke(
            create_initial_state(
                "user-1",
                "case-1",
                messages=[{"role": "user", "content": "请帮我调查东京大学"}],
            ),
            self.config,
            context=self.runtime,
        )
        self.assertEqual(first["run_status"], "WAITING_FOR_USER")
        self.assertEqual(
            first["pending_interrupt"]["kind"], "APPLICATION_INPUT_REQUIRED"
        )
        self.assertTrue(first["__interrupt__"])

        resumed = self.graph.invoke(
            Command(
                resume={
                    "fields": {
                        "university_name": "东京大学",
                        "graduate_school_name": "工学系研究科",
                        "professor_name": "山田太郎",
                        "date_from": "2023-01-01",
                        "date_to": "2026-01-01",
                    },
                    "idempotency_key": "resume-1",
                }
            ),
            self.config,
            context=JMRRuntimeContext(
                case_repository=self.repository,
                model_registry=DeterministicModelRegistry(),
                mcp_gateway=FakeRetrievalGateway(),
                object_store=InMemoryObjectStore(),
            ),
        )
        self.assertEqual(
            resumed["pending_interrupt"]["kind"],
            "APPLICANT_MEMORY_CONFIRMATION_REQUIRED",
        )
        self.assertEqual(resumed["run_status"], "WAITING_FOR_USER")
        persisted = self.repository.get_case(user_id="user-1", case_id="case-1")
        self.assertEqual(persisted["target"]["professor_name"], "山田太郎")
        self.assertEqual(persisted["target"]["date_source"], "user")

    def test_extraction_passes_chat_roles_to_model(self) -> None:
        class CapturingModel:
            def __init__(self):
                self.messages = None

            def invoke_structured(self, **kwargs):
                self.messages = kwargs["messages"]
                return {
                    "extracted": {"university": "早稲田大学"},
                    "explicit_corrections": [],
                    "ambiguous_fragments": [],
                }

        model = CapturingModel()

        class CapturingRegistry:
            def for_node(self, _node_name):
                return model

        result = extract_application_inputs(
            {"messages": [HumanMessage(content="早稲田大学を調査して")]},
            JMRRuntimeContext(model_registry=CapturingRegistry()),
        )

        self.assertEqual(model.messages[0]["role"], "user")
        self.assertEqual(model.messages[0]["content"], "早稲田大学を調査して")
        self.assertEqual(result["application_fields"]["university_name"], "早稲田大学")

    def test_invalid_resume_loops_to_the_same_interrupt_without_a_write(self) -> None:
        self.graph.invoke(
            create_initial_state(
                "user-1",
                "case-1",
                messages=[{"role": "user", "content": "需要调查目标"}],
            ),
            self.config,
            context=self.runtime,
        )
        invalid = self.graph.invoke(
            Command(resume={"fields": {"professor_name": "A"}}),
            self.config,
            context=self.runtime,
        )
        self.assertEqual(invalid["run_status"], "WAITING_FOR_USER")
        self.assertEqual(
            invalid["pending_interrupt"]["kind"], "APPLICATION_INPUT_REQUIRED"
        )
        self.assertIsNone(
            self.repository.get_case(user_id="user-1", case_id="case-1")["target"]
        )

    def test_cli_helper_rejects_a_normal_message_while_interrupted(self) -> None:
        run_turn(
            self.graph,
            user_id="user-1",
            case_id="case-1",
            user_input="目标信息还不完整",
            runtime_context=self.runtime,
        )
        with self.assertRaisesRegex(ValueError, "interrupt is pending"):
            run_turn(
                self.graph,
                user_id="user-1",
                case_id="case-1",
                user_input="绕过中断的普通消息",
                runtime_context=self.runtime,
            )


if __name__ == "__main__":
    unittest.main()
