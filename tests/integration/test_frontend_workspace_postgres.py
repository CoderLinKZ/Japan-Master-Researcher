"""User-owned browser read models against the real PostgreSQL schema."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.api import create_app  # noqa: E402
from jmr.api.auth import BearerTokenAuthenticator  # noqa: E402
from jmr.api.service import ProductionAgentAPIService  # noqa: E402
from jmr.api.workspace import case_timeline, case_workspace, list_cases  # noqa: E402
from jmr.persistence import (  # noqa: E402
    IdempotencyConflictError,
    OwnershipError,
    open_postgres_repository,
    open_postgres_store,
)


@unittest.skipUnless(
    os.getenv("JMR_RUN_POSTGRES_TESTS") == "1" and os.getenv("DATABASE_URL"),
    "requires JMR_RUN_POSTGRES_TESTS=1 and DATABASE_URL",
)
class FrontendWorkspacePostgresTests(unittest.TestCase):
    def test_case_timeline_persists_user_turns_stages_and_mcp_tools(self) -> None:
        database_url = os.environ["DATABASE_URL"]
        suffix = uuid4().hex
        owner = f"timeline-owner-{suffix}"
        case_id = f"timeline-case-{suffix}"
        with open_postgres_repository(database_url, setup=True) as repository:
            repository.create_case(
                user_id=owner,
                case_id=case_id,
                schema_version=3,
                idempotency_key=f"create-{suffix}",
            )
        try:
            with open_postgres_repository(database_url) as repository:
                first = repository.save_user_message(
                    user_id=owner,
                    case_id=case_id,
                    content="请调查目标研究室",
                    idempotency_key="initial-question",
                )
                replay = repository.save_user_message(
                    user_id=owner,
                    case_id=case_id,
                    content="请调查目标研究室",
                    idempotency_key="initial-question",
                )
                self.assertEqual(first, replay)
                with self.assertRaises(IdempotencyConflictError):
                    repository.save_user_message(
                        user_id=owner,
                        case_id=case_id,
                        content="另一条提问",
                        idempotency_key="initial-question",
                    )
                repository.record_stage_transition(
                    user_id=owner,
                    case_id=case_id,
                    transition={
                        "to_stage": "RETRIEVING_RESEARCH_EVIDENCE",
                        "run_status": "RUNNING",
                        "node_name": "plan_research",
                    },
                    idempotency_key="stage-one",
                )
                repository.record_audit_event(
                    user_id=owner,
                    case_id=case_id,
                    event_type="mcp.tool_called",
                    payload={
                        "node": "publication_research_agent",
                        "tool": "mcp__scholar__search_publications",
                        "status": "SUCCESS",
                    },
                )
                timeline = case_timeline(repository.connection, owner, case_id)
                self.assertEqual(
                    [event["kind"] for event in timeline],
                    ["user_message", "stage", "mcp_tool"],
                )
                self.assertEqual(timeline[0]["content"], "请调查目标研究室")
                self.assertEqual(
                    timeline[2]["tool"], "mcp__scholar__search_publications"
                )
                progress = ProductionAgentAPIService(
                    {"DATABASE_URL": database_url}
                ).get_progress(user_id=owner, case_id=case_id)
                self.assertEqual(
                    progress["workflow_stage"], "RETRIEVING_RESEARCH_EVIDENCE"
                )
                self.assertEqual(progress["events"], timeline)
                with self.assertRaises(OwnershipError):
                    ProductionAgentAPIService(
                        {"DATABASE_URL": database_url}
                    ).get_progress(user_id=f"other-{suffix}", case_id=case_id)
                self.assertEqual(
                    case_timeline(repository.connection, f"other-{suffix}", case_id),
                    [],
                )
                with self.assertRaises(OwnershipError):
                    repository.save_user_message(
                        user_id=f"other-{suffix}",
                        case_id=case_id,
                        content="不应保存",
                        idempotency_key="foreign-message",
                    )
        finally:
            with open_postgres_repository(database_url) as repository:
                repository.connection.execute(
                    "DELETE FROM jmr.audit_events WHERE case_id=%s AND user_id=%s",
                    (case_id, owner),
                )
                repository.connection.execute(
                    "DELETE FROM jmr.research_cases WHERE case_id=%s AND user_id=%s",
                    (case_id, owner),
                )
                repository.connection.execute(
                    "DELETE FROM jmr.idempotency_keys WHERE user_id=%s", (owner,)
                )
                repository.connection.execute(
                    "DELETE FROM jmr.users WHERE user_id=%s", (owner,)
                )
                repository.connection.commit()

    def test_uploaded_document_persists_and_is_user_scoped(self) -> None:
        database_url = os.environ["DATABASE_URL"]
        suffix = uuid4().hex
        owner = f"upload-owner-{suffix}"
        stranger = f"upload-stranger-{suffix}"
        token = f"upload-owner-token-{suffix}"
        other_token = f"upload-stranger-token-{suffix}"
        with open_postgres_repository(database_url, setup=True):
            pass
        with open_postgres_store(database_url, setup=True):
            pass
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                "DATABASE_URL": database_url,
                "ANTHROPIC_API_KEY": "integration-test-dummy-key",
                "MODEL_ID": "integration-test-dummy-model",
                "JMR_OBJECT_STORE_DIR": str(Path(temporary) / "objects"),
                "JMR_EVENT_LOG_PATH": str(Path(temporary) / "events.jsonl"),
            }
            app = create_app(
                service=ProductionAgentAPIService(environment),
                authenticator=BearerTokenAuthenticator(
                    {token: owner, other_token: stranger}
                ),
                environment=environment,
            )
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/api/v1/memories/upload",
                    headers={"Authorization": f"Bearer {token}"},
                    files={"file": ("profile.txt", "研究经历：机器学习".encode())},
                )
                self.assertEqual(response.status_code, 201, response.text)
                memory_id = response.json()["memory"]["memory_id"]
                download = client.get(
                    f"/api/v1/memories/{memory_id}/download",
                    headers={"Authorization": f"Bearer {token}"},
                )
                self.assertEqual(download.content, "研究经历：机器学习".encode())
                blocked = client.get(
                    f"/api/v1/memories/{memory_id}/download",
                    headers={"Authorization": f"Bearer {other_token}"},
                )
                self.assertEqual(blocked.status_code, 404)

    def test_case_listing_and_artifact_groups_are_user_scoped(self) -> None:
        database_url = os.environ["DATABASE_URL"]
        environment = {"DATABASE_URL": database_url}
        suffix = uuid4().hex
        owner = f"browser-owner-{suffix}"
        stranger = f"browser-stranger-{suffix}"
        case_id = f"browser-case-{suffix}"
        with open_postgres_repository(database_url, setup=True) as repository:
            repository.create_case(
                user_id=owner,
                case_id=case_id,
                schema_version=3,
                idempotency_key=f"create-{suffix}",
            )
        try:
            owner_cases = list_cases(environment, owner)["items"]
            self.assertEqual([item["case_id"] for item in owner_cases], [case_id])
            self.assertEqual(list_cases(environment, stranger)["items"], [])
            workspace = case_workspace(environment, owner, case_id)
            self.assertEqual(workspace["user_id"], owner)
            for field in (
                "plans",
                "retrieval_runs",
                "evidence",
                "directions",
                "selections",
                "drafts",
                "reviews",
                "history",
                "selected_memory_ids",
            ):
                self.assertEqual(workspace[field], [], field)
            with self.assertRaises(OwnershipError):
                case_workspace(environment, stranger, case_id)
        finally:
            with open_postgres_repository(database_url) as repository:
                repository.connection.execute(
                    "DELETE FROM jmr.research_cases WHERE case_id=%s AND user_id=%s",
                    (case_id, owner),
                )
                repository.connection.commit()


if __name__ == "__main__":
    unittest.main()
