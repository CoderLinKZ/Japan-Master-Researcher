"""Real PostgreSQL smoke test for the production FastAPI memory boundary."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.api import create_app  # noqa: E402
from jmr.api.auth import BearerTokenAuthenticator  # noqa: E402
from jmr.api.service import ProductionAgentAPIService  # noqa: E402
from jmr.persistence import (  # noqa: E402
    open_postgres_repository,
    open_postgres_store,
)
from jmr.runtime import open_postgres_checkpointer  # noqa: E402


@unittest.skipUnless(
    os.getenv("JMR_RUN_POSTGRES_TESTS") == "1" and os.getenv("DATABASE_URL"),
    "requires JMR_RUN_POSTGRES_TESTS=1 and DATABASE_URL",
)
class FastAPIPostgresIntegrationTests(unittest.TestCase):
    def test_readiness_and_user_scoped_memory_lifecycle(self) -> None:
        database_url = os.environ["DATABASE_URL"]
        suffix = uuid.uuid4().hex
        user_id = f"api-memory-user-{suffix}"
        other_user_id = f"api-memory-other-{suffix}"
        token = f"api-memory-token-{suffix}"
        other_token = f"api-memory-other-token-{suffix}"

        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            object_root = temporary_root / "objects"
            event_path = temporary_root / "events" / "events.jsonl"
            object_root.mkdir(parents=True)
            environment = {
                "DATABASE_URL": database_url,
                "ANTHROPIC_API_KEY": "integration-test-dummy-key",
                "MODEL_ID": "integration-test-dummy-model",
                "JMR_OBJECT_STORE_DIR": str(object_root),
                "JMR_EVENT_LOG_PATH": str(event_path),
                "LANGGRAPH_STRICT_MSGPACK": "true",
            }
            authenticator = BearerTokenAuthenticator(
                {token: user_id, other_token: other_user_id}
            )
            headers = {"Authorization": f"Bearer {token}"}
            other_headers = {"Authorization": f"Bearer {other_token}"}

            # Setup is explicit and idempotent.  The test never drops shared
            # Checkpointer/Store tables or the shared JMR schema.
            with open_postgres_repository(database_url, setup=True):
                pass
            with open_postgres_checkpointer(database_url, setup=True):
                pass
            with open_postgres_store(database_url, setup=True):
                pass

            try:
                service = ProductionAgentAPIService(environment)
                app = create_app(
                    service=service,
                    authenticator=authenticator,
                    environment=environment,
                )
                with TestClient(app, raise_server_exceptions=False) as client:
                    ready = client.get("/health/ready")
                    self.assertEqual(ready.status_code, 200, ready.text)
                    self.assertEqual(ready.json()["status"], "ok")
                    checks = ready.json()["checks"]
                    for dependency in (
                        "database",
                        "schema",
                        "object_store",
                        "checkpointer",
                        "memory_store",
                        "model_configuration",
                        "authentication",
                    ):
                        with self.subTest(dependency=dependency):
                            self.assertEqual(checks[dependency], "ok")

                    created = client.post(
                        "/api/v1/memories",
                        headers=headers,
                        json={
                            "kind": "research_experience",
                            "content": {
                                "summary": "Built a durable LangGraph agent",
                                "skills": ["Python", "PostgreSQL"],
                            },
                            "tags": ["agent", "langgraph"],
                            "confirmed_by_user": True,
                            "idempotency_key": f"create-{suffix}",
                        },
                    )
                    self.assertEqual(created.status_code, 201, created.text)
                    created_memory = created.json()["memory"]
                    memory_id = created_memory["memory_id"]
                    self.assertEqual(created_memory["status"], "ACTIVE")
                    self.assertEqual(created_memory["version"], 1)

                    fetched = client.get(
                        f"/api/v1/memories/{memory_id}", headers=headers
                    )
                    self.assertEqual(fetched.status_code, 200, fetched.text)
                    self.assertEqual(
                        fetched.json()["memory"]["value"]["content"]["summary"],
                        "Built a durable LangGraph agent",
                    )

                    listed = client.get(
                        "/api/v1/memories",
                        headers=headers,
                        params=[
                            ("kind", "research_experience"),
                            ("tags", "agent"),
                            ("limit", "10"),
                        ],
                    )
                    self.assertEqual(listed.status_code, 200, listed.text)
                    self.assertEqual(listed.json()["status"], "SUCCESS")
                    self.assertEqual(
                        [item["memory_id"] for item in listed.json()["items"]],
                        [memory_id],
                    )

                    isolated_get = client.get(
                        f"/api/v1/memories/{memory_id}", headers=other_headers
                    )
                    self.assertEqual(isolated_get.status_code, 404, isolated_get.text)
                    isolated_list = client.get(
                        "/api/v1/memories", headers=other_headers
                    )
                    self.assertEqual(isolated_list.status_code, 200, isolated_list.text)
                    self.assertEqual(isolated_list.json()["status"], "NO_RESULT")
                    self.assertEqual(isolated_list.json()["items"], [])

                    updated = client.patch(
                        f"/api/v1/memories/{memory_id}",
                        headers=headers,
                        json={
                            "content": {
                                "summary": "Built and exposed a durable agent API"
                            },
                            "tags": ["agent", "fastapi"],
                            "expected_version": 1,
                            "confirmed_by_user": True,
                            "idempotency_key": f"update-{suffix}",
                        },
                    )
                    self.assertEqual(updated.status_code, 200, updated.text)
                    updated_memory = updated.json()["memory"]
                    replacement_id = updated_memory["memory_id"]
                    self.assertNotEqual(replacement_id, memory_id)
                    self.assertEqual(updated_memory["supersedes"], memory_id)
                    self.assertEqual(updated_memory["status"], "ACTIVE")

                    old_version = client.get(
                        f"/api/v1/memories/{memory_id}", headers=headers
                    )
                    self.assertEqual(old_version.status_code, 200, old_version.text)
                    self.assertEqual(
                        old_version.json()["memory"]["status"], "SUPERSEDED"
                    )

                    deleted = client.request(
                        "DELETE",
                        f"/api/v1/memories/{replacement_id}",
                        headers=headers,
                        json={
                            "expected_version": 1,
                            "confirmed_by_user": True,
                            "idempotency_key": f"delete-{suffix}",
                        },
                    )
                    self.assertEqual(deleted.status_code, 200, deleted.text)
                    self.assertEqual(deleted.json()["memory"]["status"], "DELETED")
                    self.assertEqual(deleted.json()["memory"]["version"], 2)

                    active = client.get("/api/v1/memories", headers=headers)
                    self.assertEqual(active.status_code, 200, active.text)
                    self.assertEqual(active.json()["status"], "NO_RESULT")
                    self.assertEqual(active.json()["items"], [])

                    deleted_list = client.get(
                        "/api/v1/memories",
                        headers=headers,
                        params={"status": "DELETED"},
                    )
                    self.assertEqual(deleted_list.status_code, 200, deleted_list.text)
                    self.assertEqual(
                        [item["memory_id"] for item in deleted_list.json()["items"]],
                        [replacement_id],
                    )
            finally:
                self._delete_test_rows(
                    database_url,
                    user_id=user_id,
                    other_user_id=other_user_id,
                )

    @staticmethod
    def _delete_test_rows(
        database_url: str, *, user_id: str, other_user_id: str
    ) -> None:
        """Remove only records owned by this test's UUID-scoped users."""

        prefixes = (
            f"users.{user_id}.profile",
            f"users.{other_user_id}.profile",
        )
        with open_postgres_store(database_url) as store:
            store.conn.execute(
                "DELETE FROM store WHERE prefix IN (%s, %s)",
                prefixes,
            )
            store.conn.commit()
        with open_postgres_repository(database_url) as repository:
            connection = repository.connection
            connection.execute(
                "DELETE FROM jmr.profile_memory_operations WHERE user_id IN (%s, %s)",
                (user_id, other_user_id),
            )
            connection.execute(
                "DELETE FROM jmr.audit_events WHERE user_id IN (%s, %s)",
                (user_id, other_user_id),
            )
            connection.execute(
                "DELETE FROM jmr.users WHERE user_id IN (%s, %s)",
                (user_id, other_user_id),
            )
            connection.commit()


if __name__ == "__main__":
    unittest.main()
