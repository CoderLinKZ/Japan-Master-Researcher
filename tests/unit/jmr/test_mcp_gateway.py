"""Contracts for the native node-scoped MCP gateway."""

from __future__ import annotations

import sys
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.runtime import (  # noqa: E402
    MCPAccessDeniedError,
    MCPServerRegistry,
    MCPToolDefinitionError,
    NodeScopedMCPGateway,
)


class _Server:
    def __init__(self, definitions: list[dict[str, Any]]) -> None:
        self.definitions = definitions
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_tools(self) -> list[dict[str, Any]]:
        return self.definitions

    def call_tool(self, name: str, arguments: Mapping[str, Any]):
        self.calls.append((name, dict(arguments)))
        return {"status": "SUCCESS", "request_id": "request-1"}


def _definition(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"test {name}",
        "inputSchema": {"type": "object", "additionalProperties": True},
    }


class NodeScopedMCPGatewayTests(unittest.TestCase):
    def test_tool_call_writes_redacted_audit_without_arguments(self) -> None:
        server = _Server([_definition("search_projects")])
        events: list[dict[str, Any]] = []
        gateway = NodeScopedMCPGateway(
            MCPServerRegistry({"kaken": lambda: server}),
            audit_writer=lambda **event: events.append(event),
        )

        gateway.call_tool(
            node_name="kaken_research_agent",
            tool_name="mcp__kaken__search_projects",
            arguments={"professor_name": "private-name"},
            trusted_context={"user_id": "u", "case_id": "c"},
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "mcp.tool_called")
        self.assertEqual(events[0]["payload"]["status"], "SUCCESS")
        self.assertNotIn("private-name", str(events))

    def test_publication_node_sees_only_its_manifest_tools_and_server_is_cached(
        self,
    ) -> None:
        server = _Server(
            [
                _definition("discover_official_sources"),
                _definition("search_publications"),
            ]
        )
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return server

        gateway = NodeScopedMCPGateway(MCPServerRegistry({"scholar": factory}))
        first = gateway.tools_for_node("publication_research_agent")
        second = gateway.tools_for_node("publication_research_agent")

        self.assertEqual(
            {tool["name"] for tool in first},
            {
                "mcp__scholar__discover_official_sources",
                "mcp__scholar__search_publications",
            },
        )
        self.assertEqual(second, first)
        self.assertEqual(factory_calls, 1)

    def test_cross_node_tool_call_is_rejected_before_dispatch(self) -> None:
        gateway = NodeScopedMCPGateway(MCPServerRegistry({}))

        with self.assertRaises(MCPAccessDeniedError):
            gateway.call_tool(
                node_name="publication_research_agent",
                tool_name="mcp__kaken__search_projects",
                arguments={},
                trusted_context={"user_id": "u", "case_id": "c"},
            )

    def test_malformed_server_definition_is_rejected(self) -> None:
        server = _Server(
            [
                {
                    "name": "search_projects",
                    "description": "test",
                    "inputSchema": {"type": "array"},
                }
            ]
        )
        gateway = NodeScopedMCPGateway(MCPServerRegistry({"kaken": lambda: server}))

        with self.assertRaises(MCPToolDefinitionError):
            gateway.tools_for_node("kaken_research_agent")

    def test_trusted_user_and_case_context_are_required(self) -> None:
        server = _Server([_definition("search_projects")])
        gateway = NodeScopedMCPGateway(MCPServerRegistry({"kaken": lambda: server}))

        with self.assertRaisesRegex(ValueError, "case_id"):
            gateway.call_tool(
                node_name="kaken_research_agent",
                tool_name="mcp__kaken__search_projects",
                arguments={},
                trusted_context={"user_id": "user-1"},
            )
        self.assertEqual(server.calls, [])


if __name__ == "__main__":
    unittest.main()
