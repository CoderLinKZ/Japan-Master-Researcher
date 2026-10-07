"""An empty official-site candidate list must remain actionable."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from langgraph.runtime import Runtime

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.graph.full import _route_identity_apply  # noqa: E402
from jmr.graph.nodes.retrieval import (  # noqa: E402
    apply_target_identity_confirmation,
    join_retrieval_results,
)
from jmr.runtime import JMRRuntimeContext  # noqa: E402


class _Repository:
    def __init__(self):
        self.updated_target = None
        self.transitions = []
        self.runs = [
            {
                "status": "NEEDS_USER_CONFIRMATION",
                "result_summary": {"discovered_sources": []},
            }
        ]

    def get_retrieval_runs(self, *, user_id, case_id, retrieval_run_ids):
        return self.runs

    def get_case(self, *, user_id, case_id):
        return {
            "target": self.updated_target or {"professor_name": "炭親良"},
            "version": 1,
        }

    def update_target(
        self, *, user_id, case_id, target, expected_version, idempotency_key
    ):
        self.updated_target = target
        return {"target_id": "target-1"}

    def record_stage_transition(self, *, user_id, case_id, transition, idempotency_key):
        self.transitions.append(transition)
        return transition


class EmptyIdentityConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.repository = _Repository()
        self.runtime = Runtime(
            context=JMRRuntimeContext(case_repository=self.repository)
        )
        self.state = {
            "user_id": "root",
            "case_id": "case-1",
            "retrieval_join_decision": "identity",
            "retrieval_refs": [
                {"worker_kind": "publication", "retrieval_run_id": "run-1"}
            ],
        }

    def test_manual_url_can_resolve_empty_candidate_interrupt(self):
        self.state["retrieval_resume_payload"] = {
            "selected_candidate_id": "https://example.ac.jp/lab/",
            "confirmed_by_user": True,
            "idempotency_key": "confirm-url-1",
        }

        result = apply_target_identity_confirmation(self.state, self.runtime)

        self.assertEqual(result["retrieval_workers_needed"], ["publication"])
        self.assertEqual(
            self.repository.updated_target["official_urls"],
            ["https://example.ac.jp/lab/"],
        )

    def test_manual_url_requires_explicit_confirmation(self):
        self.state["retrieval_resume_payload"] = {
            "selected_candidate_id": "https://example.ac.jp/lab/",
            "idempotency_key": "confirm-url-1",
        }

        result = apply_target_identity_confirmation(self.state, self.runtime)

        self.assertEqual(result["run_status"], "WAITING_FOR_USER")
        self.assertIsNone(self.repository.updated_target)

    def test_explicit_no_site_moves_to_evidence_verification(self):
        self.state["retrieval_resume_payload"] = {
            "action": "no_official_site",
            "confirmed_by_user": True,
            "idempotency_key": "no-site-1",
        }

        result = apply_target_identity_confirmation(self.state, self.runtime)

        self.assertEqual(result["retrieval_join_decision"], "verify")
        self.assertEqual(result["workflow_stage"], "MERGING_AND_VERIFYING_EVIDENCE")
        self.assertEqual(_route_identity_apply(result), "merge_and_verify_evidence")
        self.assertEqual(
            self.repository.updated_target["official_site_status"],
            "USER_REPORTED_NONE",
        )
        self.assertEqual(self.repository.updated_target["official_urls"], [])
        self.assertEqual(len(self.repository.transitions), 1)

    def test_no_site_requires_explicit_confirmation(self):
        self.state["retrieval_resume_payload"] = {
            "action": "no_official_site",
            "idempotency_key": "no-site-1",
        }

        result = apply_target_identity_confirmation(self.state, self.runtime)

        self.assertEqual(result["run_status"], "WAITING_FOR_USER")
        self.assertIsNone(self.repository.updated_target)

    def test_retry_discovery_restarts_only_publication_worker(self):
        self.state["retrieval_resume_payload"] = {
            "action": "retry_discovery",
            "idempotency_key": "retry-search-1",
        }

        result = apply_target_identity_confirmation(self.state, self.runtime)

        self.assertEqual(result["retrieval_workers_needed"], ["publication"])
        self.assertEqual(
            _route_identity_apply({**self.state, **result}), "plan_research"
        )
        self.assertIsNone(self.repository.updated_target)

    def test_reported_no_site_does_not_repeat_identity_interrupt(self):
        self.repository.updated_target = {
            "professor_name": "炭親良",
            "official_site_status": "USER_REPORTED_NONE",
        }
        self.repository.runs = [
            {
                "retrieval_run_id": "run-1",
                "status": "NEEDS_USER_CONFIRMATION",
                "result_summary": {"records": [{"evidence_id": "ev-1"}]},
            }
        ]

        result = join_retrieval_results(self.state, self.runtime)

        self.assertEqual(result["retrieval_join_decision"], "verify")


if __name__ == "__main__":
    unittest.main()
