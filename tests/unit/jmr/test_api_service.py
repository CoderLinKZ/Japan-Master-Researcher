"""Service-level lifecycle, concurrency, and projection tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.api.service import (  # noqa: E402
    CaseBusyError,
    ProductionAgentAPIService,
    ServiceUnavailableError,
    _project_values,
    _snapshot_failure,
    _snapshot_retry_available,
)


class ProductionAgentAPIServiceTests(unittest.TestCase):
    def _service(self, root: str, *, capacity: int = 1):
        return ProductionAgentAPIService(
            {
                "JMR_API_MAX_CONCURRENT_RUNS": str(capacity),
                "JMR_EVENT_LOG_PATH": f"{root}/logs/events.jsonl",
                "JMR_OBJECT_STORE_DIR": f"{root}/objects",
            }
        )

    def test_lifecycle_initializes_and_closes_local_resources(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = self._service(root)
            service.startup()
            self.assertTrue(Path(root, "objects").is_dir())
            self.assertTrue(Path(root, "logs", "events.jsonl").is_file())
            self.assertIsNotNone(service._event_sink)
            service.shutdown()
            self.assertIsNone(service._event_sink)

    def test_same_case_is_non_blocking_busy_but_scopes_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = self._service(root)
            with service._exclusive_guard("case", "case-1"):
                with self.assertRaises(CaseBusyError):
                    with service._exclusive_guard("case", "case-1"):
                        self.fail("the same case must not enter twice")
                with service._exclusive_guard("case", "case-2"):
                    pass
                with service._exclusive_guard("memory", "case-1"):
                    pass

    def test_turn_capacity_is_bounded_without_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = self._service(root, capacity=1)
            with service._turn_slot():
                with self.assertRaises(ServiceUnavailableError):
                    with service._turn_slot():
                        self.fail("capacity exhaustion must fail fast")

    def test_projection_does_not_return_messages_or_private_state(self) -> None:
        projected = _project_values(
            {
                "user_id": "user-1",
                "case_id": "case-1",
                "workflow_stage": "DRAFTING_OUTREACH",
                "run_status": "RUNNING",
                "messages": [{"role": "user", "content": "private"}],
                "application_fields": {"email": "private@example.com"},
                "warnings": [],
                "errors": [],
                "research_plan_id": "plan-1",
            },
            pending=None,
        )
        self.assertNotIn("messages", projected)
        self.assertNotIn("application_fields", projected)
        self.assertEqual(projected["artifacts"], {"research_plan_id": "plan-1"})

    def test_projection_hides_resolved_application_warnings(self) -> None:
        projected = _project_values(
            {
                "workflow_stage": "RETRIEVING_RESEARCH_EVIDENCE",
                "run_status": "WAITING_FOR_USER",
                "application_validation": {
                    "ready": True,
                    "current_values": {"date_source": "user"},
                },
                "warnings": [
                    "缺少大学名称",
                    "缺少研究科或学院名称",
                    "缺少教授姓名",
                    "检索时间范围采用系统默认的过去十二个月",
                    "检索结果较少",
                ],
            },
            pending=None,
        )
        self.assertEqual(projected["warnings"], ["检索结果较少"])

    def test_projection_hides_superseded_site_search_warning(self) -> None:
        projected = _project_values(
            {
                "warnings": [
                    "SerpApi 未发现满足身份信号要求的研究室官网",
                    "搜索结果仅匹配教授姓氏及学校，尚未核实教授全名",
                ]
            },
            pending={
                "kind": "TARGET_IDENTITY_CONFIRMATION_REQUIRED",
                "options": [{"url": "https://sakailab.com/"}],
            },
        )

        self.assertEqual(
            projected["warnings"],
            ["搜索结果仅匹配教授姓氏及学校，尚未核实教授全名"],
        )

    def test_readiness_fails_cleanly_without_database_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._service(root).readiness()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["checks"]["database"], "failed")

    def test_only_failed_nonterminal_checkpoint_is_retryable(self) -> None:
        failed = SimpleNamespace(error="TimeoutError", interrupts=())
        snapshot = SimpleNamespace(values={"run_status": "RUNNING"}, tasks=(failed,))
        self.assertTrue(_snapshot_retry_available(snapshot))
        self.assertEqual(_snapshot_failure(snapshot)["code"], "MODEL_TIMEOUT")
        failed.name = "plan_research"
        failed.error = "ValueError('plan_research failed structured output validation')"
        self.assertEqual(_snapshot_failure(snapshot)["code"], "MODEL_OUTPUT_INVALID")
        snapshot.values["run_status"] = "COMPLETED"
        self.assertFalse(_snapshot_retry_available(snapshot))
        snapshot.values["run_status"] = "RUNNING"
        failed.interrupts = (object(),)
        self.assertFalse(_snapshot_retry_available(snapshot))


if __name__ == "__main__":
    unittest.main()
