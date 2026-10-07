"""Real PostgreSQL regression for confirmed profile Memory and audit."""

from __future__ import annotations

import os
import sys
import unittest
import uuid
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.persistence import (  # noqa: E402
    PostgresMemoryStore,
    open_postgres_repository,
    open_postgres_store,
)
from mcp_servers.memory import create_memory_server  # noqa: E402


@unittest.skipUnless(
    os.getenv("JMR_RUN_POSTGRES_TESTS") == "1" and os.getenv("DATABASE_URL"),
    "requires JMR_RUN_POSTGRES_TESTS=1 and DATABASE_URL",
)
class PostgresMemoryIntegrationTests(unittest.TestCase):
    def test_mutation_replay_and_audit_survive_reconnect(self) -> None:
        database_url = os.environ["DATABASE_URL"]
        user_id = f"memory-user-{uuid.uuid4().hex}"
        payload = {
            "kind": "research_experience",
            "content": {"summary": "A"},
            "confirmed_by_user": True,
            "idempotency_key": "create-1",
        }
        try:
            with open_postgres_repository(database_url, setup=True):
                pass
            with open_postgres_store(database_url, setup=True) as store:
                server = create_memory_server(
                    PostgresMemoryStore(store), user_id=user_id
                )
                created = server.call_tool("create_memory", payload)
            with open_postgres_store(database_url) as store:
                server = create_memory_server(
                    PostgresMemoryStore(store), user_id=user_id
                )
                self.assertEqual(server.call_tool("create_memory", payload), created)
                self.assertEqual(len(server.call_tool("list_memories", {})["items"]), 1)
                count = store.conn.execute(
                    "SELECT count(*) FROM jmr.audit_events WHERE user_id=%s",
                    (user_id,),
                ).fetchone()["count"]
                self.assertEqual(count, 1)
                store.conn.execute(
                    "DELETE FROM store WHERE prefix=%s",
                    (f"users.{user_id}.profile",),
                )
                store.conn.commit()
        finally:
            with open_postgres_repository(database_url) as repository:
                repository.connection.execute("DROP SCHEMA IF EXISTS jmr CASCADE")
                repository.connection.commit()


if __name__ == "__main__":
    unittest.main()
