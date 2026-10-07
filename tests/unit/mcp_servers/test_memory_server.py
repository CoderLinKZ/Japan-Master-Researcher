"""Memory MCP lifecycle and user isolation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.persistence import InMemoryMemoryStore  # noqa: E402
from mcp_servers.memory import create_memory_server  # noqa: E402


class MemoryMCPServerTests(unittest.TestCase):
    def test_crud_is_versioned_and_user_scoped(self) -> None:
        store = InMemoryMemoryStore()
        first = create_memory_server(store, user_id="user-1")
        second = create_memory_server(store, user_id="user-2")
        created = first.call_tool(
            "create_memory",
            {
                "kind": "research_experience",
                "content": {"summary": "A"},
                "confirmed_by_user": True,
                "idempotency_key": "create-1",
            },
        )["memory"]
        self.assertEqual(len(first.call_tool("list_memories", {})["items"]), 1)
        self.assertEqual(second.call_tool("list_memories", {})["items"], [])

        updated = first.call_tool(
            "update_memory",
            {
                "memory_id": created["memory_id"],
                "content": {"summary": "B"},
                "expected_version": 1,
                "confirmed_by_user": True,
                "idempotency_key": "update-1",
            },
        )["memory"]
        self.assertNotEqual(updated["memory_id"], created["memory_id"])
        self.assertEqual(len(first.call_tool("list_memories", {})["items"]), 1)
        first.call_tool(
            "delete_memory",
            {
                "memory_id": updated["memory_id"],
                "expected_version": 1,
                "confirmed_by_user": True,
                "idempotency_key": "delete-1",
            },
        )
        self.assertEqual(first.call_tool("list_memories", {})["items"], [])

    def test_mutation_requires_confirmation_and_replays_idempotently(self) -> None:
        server = create_memory_server(InMemoryMemoryStore(), user_id="user-1")
        payload = {
            "kind": "research_experience",
            "content": {"summary": "A"},
            "confirmed_by_user": True,
            "idempotency_key": "create-1",
        }
        with self.assertRaisesRegex(ValueError, "confirmation"):
            server.call_tool("create_memory", dict(payload, confirmed_by_user=False))
        first = server.call_tool("create_memory", payload)
        self.assertEqual(server.call_tool("create_memory", payload), first)
        with self.assertRaisesRegex(ValueError, "idempotency key reused"):
            server.call_tool(
                "create_memory",
                dict(payload, content={"summary": "different"}),
            )


if __name__ == "__main__":
    unittest.main()
