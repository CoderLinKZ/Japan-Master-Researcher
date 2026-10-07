"""Human-resume inputs cannot expand identity or reuse stale citations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.graph.nodes.outreach import apply_revision_input  # noqa: E402
from jmr.graph.nodes.retrieval import (  # noqa: E402
    apply_target_identity_confirmation,
)
from jmr.persistence import InMemoryCaseRepository, InMemoryObjectStore  # noqa: E402
from jmr.runtime import JMRRuntimeContext  # noqa: E402


class ResumeBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = InMemoryCaseRepository()
        self.repo.create_case(
            user_id="u", case_id="c", schema_version=3, idempotency_key="create"
        )
        self.context = JMRRuntimeContext(
            case_repository=self.repo, object_store=InMemoryObjectStore()
        )

    def test_identity_confirmation_rejects_unoffered_url(self) -> None:
        run = self.repo.save_retrieval_run(
            user_id="u",
            case_id="c",
            worker_kind="publication",
            status="NEEDS_USER_CONFIRMATION",
            result_summary={
                "discovered_sources": [{"url": "https://allowed.example.edu/lab"}]
            },
            idempotency_key="run",
        )
        state = {
            "user_id": "u",
            "case_id": "c",
            "retrieval_join_decision": "identity",
            "retrieval_refs": [
                {
                    "worker_kind": "publication",
                    "retrieval_run_id": run["retrieval_run_id"],
                }
            ],
            "retrieval_resume_payload": {
                "selected_candidate_id": "https://other.example.edu/lab",
                "idempotency_key": "choice",
            },
        }
        rejected = apply_target_identity_confirmation(state, self.context)
        self.assertEqual(rejected["run_status"], "WAITING_FOR_USER")
        self.assertIsNone(self.repo.get_case(user_id="u", case_id="c")["target"])

    def test_explicit_manual_url_can_replace_unoffered_candidate(self) -> None:
        run = self.repo.save_retrieval_run(
            user_id="u",
            case_id="c",
            worker_kind="publication",
            status="NEEDS_USER_CONFIRMATION",
            result_summary={
                "discovered_sources": [{"url": "https://wrong.example.edu/lab"}]
            },
            idempotency_key="run-manual",
        )
        state = {
            "user_id": "u",
            "case_id": "c",
            "retrieval_join_decision": "identity",
            "retrieval_refs": [
                {
                    "worker_kind": "publication",
                    "retrieval_run_id": run["retrieval_run_id"],
                }
            ],
            "retrieval_resume_payload": {
                "selected_candidate_id": "https://right.example.edu/lab",
                "manual_url": True,
                "confirmed_by_user": True,
                "idempotency_key": "manual-choice",
            },
        }

        result = apply_target_identity_confirmation(state, self.context)

        self.assertEqual(result["retrieval_workers_needed"], ["publication"])
        self.assertEqual(
            self.repo.get_case(user_id="u", case_id="c")["target"]["official_urls"],
            ["https://right.example.edu/lab"],
        )

    def test_edit_requires_explicit_current_evidence_ids(self) -> None:
        self.repo.record_stage_transition(
            user_id="u",
            case_id="c",
            transition={
                "from_stage": "VALIDATING_APPLICATION_INPUTS",
                "to_stage": "REVIEWING_OUTREACH_RESEARCH_PLAN",
                "run_status": "RUNNING",
            },
            idempotency_key="review-stage",
        )
        bundle = self.repo.save_verified_evidence(
            user_id="u",
            case_id="c",
            evidence_ids=["ev-1"],
            summary={},
            idempotency_key="evidence",
        )
        iteration = self.repo.save_outreach_iteration(
            user_id="u",
            case_id="c",
            content="原稿",
            citation_ids=["ev-1"],
            parent_iteration_id=None,
            idempotency_key="draft",
        )
        state = {
            "user_id": "u",
            "case_id": "c",
            "verified_evidence_bundle_id": bundle["verified_evidence_bundle_id"],
            "current_outreach_iteration_id": iteration["iteration_id"],
            "revision_resume_payload": {
                "action": "edit",
                "edited_text": "新稿",
                "idempotency_key": "edit-1",
            },
        }
        rejected = apply_revision_input(state, self.context)
        self.assertEqual(rejected["run_status"], "WAITING_FOR_USER")
        self.assertEqual(len(self.repo._iterations), 1)


if __name__ == "__main__":
    unittest.main()
