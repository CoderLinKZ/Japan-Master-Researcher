"""Durable HTTP operation receipts and deletion plans against PostgreSQL."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.api.service import (  # noqa: E402
    OperationConflictError,
    ProductionAgentAPIService,
    _RuntimeSession,
)
from jmr.persistence import open_postgres_repository  # noqa: E402
from jmr.runtime import open_postgres_checkpointer  # noqa: E402


class _FakeGraph:
    def __init__(self) -> None:
        self.snapshots: dict[str, SimpleNamespace] = {}

    def get_state(self, config: dict) -> SimpleNamespace:
        case_id = config["configurable"]["thread_id"]
        return self.snapshots.get(case_id, _snapshot())


def _snapshot(
    *,
    user_id: str = "",
    case_id: str = "",
    checkpoint_id: str = "",
    interrupt_id: str | None = None,
) -> SimpleNamespace:
    tasks = ()
    if interrupt_id is not None:
        tasks = (
            SimpleNamespace(
                interrupts=(
                    SimpleNamespace(
                        id=interrupt_id,
                        value={"kind": "APPLICATION_INPUT_REQUIRED"},
                    ),
                )
            ),
        )
    values = (
        {
            "user_id": user_id,
            "case_id": case_id,
            "workflow_stage": "VALIDATING_APPLICATION_INPUTS",
            "run_status": "WAITING_FOR_USER" if interrupt_id else "READY",
        }
        if case_id
        else {}
    )
    return SimpleNamespace(
        values=values,
        tasks=tasks,
        config={"configurable": {"checkpoint_id": checkpoint_id}},
    )


@unittest.skipUnless(
    os.getenv("JMR_RUN_POSTGRES_TESTS") == "1" and os.getenv("DATABASE_URL"),
    "requires JMR_RUN_POSTGRES_TESTS=1 and DATABASE_URL",
)
class APIReceiptPostgresTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database_url = os.environ["DATABASE_URL"]
        suffix = uuid.uuid4().hex
        self.user_id = f"api-receipt-user-{suffix}"
        self.case_id = f"api-receipt-case-{suffix}"
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = {
            "DATABASE_URL": self.database_url,
            "JMR_OBJECT_STORE_DIR": str(Path(self.temporary.name) / "objects"),
            "JMR_EVENT_LOG_PATH": str(Path(self.temporary.name) / "events.jsonl"),
        }
        self.service = ProductionAgentAPIService(self.environment)
        with open_postgres_repository(self.database_url, setup=True):
            pass
        with open_postgres_checkpointer(self.database_url, setup=True):
            pass

    def tearDown(self) -> None:
        try:
            self.service.shutdown()
            with open_postgres_checkpointer(self.database_url) as saver:
                saver.delete_thread(self.case_id)
            with open_postgres_repository(self.database_url) as repository:
                connection = repository.connection
                connection.execute(
                    "DELETE FROM jmr.case_deletions WHERE case_id=%s AND user_id=%s",
                    (self.case_id, self.user_id),
                )
                connection.execute(
                    "DELETE FROM jmr.research_cases WHERE case_id=%s AND user_id=%s",
                    (self.case_id, self.user_id),
                )
                connection.execute(
                    "DELETE FROM jmr.idempotency_keys WHERE user_id=%s",
                    (self.user_id,),
                )
                connection.execute(
                    "DELETE FROM jmr.audit_events WHERE user_id=%s",
                    (self.user_id,),
                )
                connection.execute(
                    "DELETE FROM jmr.users WHERE user_id=%s", (self.user_id,)
                )
                connection.commit()
        finally:
            self.temporary.cleanup()

    def _runtime_for(self, graph: _FakeGraph):
        @contextmanager
        def open_runtime(_user_id: str):
            with open_postgres_repository(self.database_url) as repository:
                yield _RuntimeSession(graph, object(), repository)

        return open_runtime

    def _reader_for(self, graph: _FakeGraph):
        @contextmanager
        def open_reader():
            with open_postgres_repository(self.database_url) as repository:
                yield graph, repository

        return open_reader

    def _create_business_case(self) -> None:
        with open_postgres_repository(self.database_url) as repository:
            repository.create_case(
                user_id=self.user_id,
                case_id=self.case_id,
                schema_version=3,
                idempotency_key="business-create",
            )

    def test_create_receipt_survives_new_service_and_rejects_changed_payload(
        self,
    ) -> None:
        graph = _FakeGraph()
        invocations: list[str] = []

        def run_create(_graph, **kwargs):
            case_id = kwargs["case_id"]
            invocations.append(case_id)
            with open_postgres_repository(self.database_url) as repository:
                repository.create_case(
                    user_id=self.user_id,
                    case_id=case_id,
                    schema_version=3,
                    idempotency_key="business-create",
                )
            graph.snapshots[case_id] = _snapshot(
                user_id=self.user_id,
                case_id=case_id,
                checkpoint_id="create-checkpoint",
            )

        key = "http-create-one"
        with (
            patch.object(self.service, "_open_runtime", self._runtime_for(graph)),
            patch.object(self.service, "_open_reader", self._reader_for(graph)),
            patch("jmr.api.service.run_turn", side_effect=run_create),
        ):
            created = self.service.start_case(
                user_id=self.user_id, message="Find professors", idempotency_key=key
            )
            self.case_id = created["case_id"]
            replay = self.service.start_case(
                user_id=self.user_id, message="Find professors", idempotency_key=key
            )
        self.assertEqual(replay, created)
        self.assertEqual(invocations, [self.case_id])
        with patch.object(self.service, "_open_reader", self._reader_for(graph)):
            conversation = self.service.get_conversation(
                user_id=self.user_id, case_id=self.case_id
            )
        self.assertEqual(
            conversation["messages"],
            [{"role": "user", "content": "Find professors"}],
        )
        self.assertEqual(
            [event["kind"] for event in conversation["events"]],
            ["user_message"],
        )

        restarted = ProductionAgentAPIService(self.environment)
        with patch.object(restarted, "_open_reader", self._reader_for(graph)):
            self.assertEqual(
                restarted.start_case(
                    user_id=self.user_id,
                    message="Find professors",
                    idempotency_key=key,
                ),
                created,
            )
            with self.assertRaises(OperationConflictError):
                restarted.start_case(
                    user_id=self.user_id, message="Find labs", idempotency_key=key
                )
        self.assertEqual(invocations, [self.case_id])

    def test_resume_rejects_stale_token_and_replays_durable_receipt(self) -> None:
        self._create_business_case()
        graph = _FakeGraph()
        graph.snapshots[self.case_id] = _snapshot(
            user_id=self.user_id,
            case_id=self.case_id,
            checkpoint_id="pending-checkpoint",
            interrupt_id="interrupt-1",
        )
        token = hashlib.sha256(b"pending-checkpoint:interrupt-1").hexdigest()
        invocations: list[str] = []

        def run_resume(_graph, **kwargs):
            invocations.append(kwargs["resume_interrupt_id"])
            graph.snapshots[self.case_id] = _snapshot(
                user_id=self.user_id,
                case_id=self.case_id,
                checkpoint_id="resumed-checkpoint",
            )

        payload = {"fields": {"professor_name": "Example"}}
        with (
            patch.object(self.service, "_open_runtime", self._runtime_for(graph)),
            patch("jmr.api.service.run_turn", side_effect=run_resume),
        ):
            with self.assertRaises(OperationConflictError):
                self.service.resume_case(
                    user_id=self.user_id,
                    case_id=self.case_id,
                    payload=payload,
                    interrupt_token="stale-token",
                )
            self.assertEqual(invocations, [])
            resumed = self.service.resume_case(
                user_id=self.user_id,
                case_id=self.case_id,
                payload=payload,
                interrupt_token=token,
            )
            replay = self.service.resume_case(
                user_id=self.user_id,
                case_id=self.case_id,
                payload=payload,
                interrupt_token=token,
            )
            with self.assertRaises(OperationConflictError):
                self.service.resume_case(
                    user_id=self.user_id,
                    case_id=self.case_id,
                    payload={"fields": {"professor_name": "Different"}},
                    interrupt_token=token,
                )
        self.assertEqual(replay, resumed)
        self.assertEqual(invocations, ["interrupt-1"])

        restarted = ProductionAgentAPIService(self.environment)
        with patch.object(restarted, "_open_runtime", self._runtime_for(graph)):
            self.assertEqual(
                restarted.resume_case(
                    user_id=self.user_id,
                    case_id=self.case_id,
                    payload=payload,
                    interrupt_token=token,
                ),
                resumed,
            )

    def test_deletion_plan_detects_object_and_case_version_changes(self) -> None:
        self._create_business_case()
        object_store = self.service._require_object_store()
        prefix = f"users/{self.user_id}/cases/{self.case_id}/workflow"
        first_uri = object_store.put(
            object_key=f"{prefix}/one.json",
            content=b"{}",
            content_type="application/json",
        )
        first_plan = self.service.plan_case_deletion(
            user_id=self.user_id, case_id=self.case_id
        )
        self.assertEqual(first_plan["object_uris"], [first_uri])

        second_uri = object_store.put(
            object_key=f"{prefix}/two.json",
            content=b"[]",
            content_type="application/json",
        )
        with self.assertRaises(OperationConflictError):
            self.service.confirm_case_deletion(
                user_id=self.user_id,
                case_id=self.case_id,
                plan_token=first_plan["plan_token"],
            )
        self.assertEqual(object_store.get(object_uri=first_uri), b"{}")

        second_plan = self.service.plan_case_deletion(
            user_id=self.user_id, case_id=self.case_id
        )
        with open_postgres_repository(self.database_url) as repository:
            repository.connection.execute(
                "UPDATE jmr.research_cases SET version=version+1 WHERE case_id=%s",
                (self.case_id,),
            )
            repository.connection.commit()
        with self.assertRaises(OperationConflictError):
            self.service.confirm_case_deletion(
                user_id=self.user_id,
                case_id=self.case_id,
                plan_token=second_plan["plan_token"],
            )

        current_plan = self.service.plan_case_deletion(
            user_id=self.user_id, case_id=self.case_id
        )
        deleted = self.service.confirm_case_deletion(
            user_id=self.user_id,
            case_id=self.case_id,
            plan_token=current_plan["plan_token"],
        )
        self.assertEqual(deleted["status"], "COMPLETED")
        self.assertEqual(deleted["object_count"], 2)
        self.assertEqual(
            self.service.confirm_case_deletion(
                user_id=self.user_id,
                case_id=self.case_id,
                plan_token=current_plan["plan_token"],
            )["status"],
            "COMPLETED",
        )
        for uri in (first_uri, second_uri):
            with self.assertRaises(KeyError):
                object_store.get(object_uri=uri)


if __name__ == "__main__":
    unittest.main()
