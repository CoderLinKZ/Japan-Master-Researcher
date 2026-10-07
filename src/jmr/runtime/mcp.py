"""Node-scoped MCP registry and gateway for the production workflow."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from threading import RLock
from typing import Any, Protocol

from .reliability import BoundedRetryPolicy, call_with_retry
from .schema_validation import validate_json_schema


class MCPServer(Protocol):
    """Minimal in-process MCP server contract used by the runtime."""

    def list_tools(self) -> list[dict[str, Any]]: ...

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any: ...


MCPServerFactory = Callable[[], MCPServer]


class MCPGatewayError(RuntimeError):
    """Base class for MCP discovery and invocation failures."""


class MCPAccessDeniedError(MCPGatewayError):
    """Raised when a node requests a capability outside its manifest."""


class MCPToolDefinitionError(MCPGatewayError):
    """Raised when a server advertises an unsafe or malformed tool."""


class MCPServerRegistry:
    """Own validated MCP server factories without global mutable clients."""

    def __init__(self, factories: Mapping[str, MCPServerFactory]) -> None:
        if not isinstance(factories, Mapping):
            raise TypeError("factories must be a mapping")
        self._factories: dict[str, MCPServerFactory] = {}
        for raw_name, factory in factories.items():
            name = _server_name(raw_name)
            if not callable(factory):
                raise TypeError(f"MCP factory for {name!r} must be callable")
            if name in self._factories:
                raise ValueError(f"duplicate MCP server name: {name}")
            self._factories[name] = factory

    @property
    def server_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def create(self, name: str) -> MCPServer:
        normalized = _server_name(name)
        factory = self._factories.get(normalized)
        if factory is None:
            raise MCPGatewayError(f"unknown MCP server: {normalized}")
        server = factory()
        for attribute in ("list_tools", "call_tool"):
            if not callable(getattr(server, attribute, None)):
                raise MCPGatewayError(
                    f"MCP server {normalized!r} is missing {attribute}"
                )
        return server


class NodeScopedMCPGateway:
    """Expose exactly the agent tools declared for the current graph node."""

    def __init__(
        self,
        registry: MCPServerRegistry,
        *,
        retry_policy: BoundedRetryPolicy | None = None,
        audit_writer: Callable[..., None] | None = None,
    ) -> None:
        if not isinstance(registry, MCPServerRegistry):
            raise TypeError("registry must be an MCPServerRegistry")
        self._registry = registry
        self._servers: dict[str, MCPServer] = {}
        self._tools: dict[str, dict[str, Mapping[str, Any]]] = {}
        self._lock = RLock()
        self._retry_policy = retry_policy or BoundedRetryPolicy()
        self._audit_writer = audit_writer

    def tools_for_node(self, node_name: str) -> Sequence[Mapping[str, Any]]:
        return [
            self._model_tool(tool_name) for tool_name in _agent_tools_for(node_name)
        ]

    def call_tool(
        self,
        *,
        node_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        trusted_context: Mapping[str, str],
    ) -> Mapping[str, Any]:
        if not isinstance(arguments, Mapping):
            raise TypeError("MCP tool arguments must be an object")
        if not isinstance(trusted_context, Mapping):
            raise TypeError("trusted_context must be an object")
        if tool_name not in set(_agent_tools_for(node_name)):
            raise MCPAccessDeniedError(
                f"tool {tool_name!r} is not allowed for node {node_name!r}"
            )
        server_name, raw_tool_name = _split_exposed_name(tool_name)
        server, definitions = self._connect(server_name)
        definition = definitions.get(raw_tool_name)
        if definition is None:
            raise MCPGatewayError(f"MCP tool is unavailable: {tool_name}")
        _validate_trusted_context(trusted_context)
        bounded_arguments = dict(arguments)
        validate_json_schema(bounded_arguments, definition["input_schema"])
        try:
            result = call_with_retry(
                lambda: server.call_tool(raw_tool_name, bounded_arguments),
                policy=self._retry_policy,
                idempotent=True,
            )
        except Exception:
            self._audit_tool_call(trusted_context, node_name, tool_name, "FAILED")
            raise
        if not isinstance(result, Mapping):
            self._audit_tool_call(trusted_context, node_name, tool_name, "FAILED")
            raise MCPGatewayError("MCP tool response must be an object")
        self._audit_tool_call(
            trusted_context,
            node_name,
            tool_name,
            str(result.get("status", "UNKNOWN")),
        )
        return dict(result)

    def _audit_tool_call(
        self,
        trusted_context: Mapping[str, str],
        node_name: str,
        tool_name: str,
        status: str,
    ) -> None:
        if self._audit_writer is not None:
            self._audit_writer(
                user_id=trusted_context["user_id"],
                case_id=trusted_context["case_id"],
                event_type="mcp.tool_called",
                payload={"node": node_name, "tool": tool_name, "status": status},
            )

    def _model_tool(self, exposed_name: str) -> dict[str, Any]:
        server_name, raw_name = _split_exposed_name(exposed_name)
        _server, definitions = self._connect(server_name)
        definition = definitions.get(raw_name)
        if definition is None:
            raise MCPGatewayError(f"MCP tool is unavailable: {exposed_name}")
        return {
            "name": exposed_name,
            "description": definition["description"],
            # Target identity, institution, date range, and result limits are
            # host-controlled. The model chooses a capability, never its
            # security-sensitive arguments.
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }

    def _connect(
        self,
        server_name: str,
    ) -> tuple[MCPServer, dict[str, Mapping[str, Any]]]:
        with self._lock:
            existing = self._servers.get(server_name)
            if existing is not None:
                return existing, self._tools[server_name]
            server = self._registry.create(server_name)
            definitions = _validate_tool_definitions(
                server_name,
                server.list_tools(),
            )
            self._servers[server_name] = server
            self._tools[server_name] = definitions
            return server, definitions


def _validate_tool_definitions(
    server_name: str,
    raw_definitions: Any,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(raw_definitions, list):
        raise MCPToolDefinitionError("MCP tools/list result must be an array")
    definitions: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(raw_definitions):
        if not isinstance(item, Mapping):
            raise MCPToolDefinitionError(
                f"{server_name} tool[{index}] must be an object"
            )
        raw_name = item.get("name")
        description = item.get("description")
        schema = item.get("inputSchema", item.get("input_schema"))
        if not isinstance(raw_name, str) or not _NAME_PATTERN.fullmatch(raw_name):
            raise MCPToolDefinitionError(
                f"{server_name} tool[{index}] has an invalid name"
            )
        if raw_name in definitions:
            raise MCPToolDefinitionError(
                f"{server_name} advertises duplicate tool {raw_name!r}"
            )
        if not isinstance(description, str) or not description.strip():
            raise MCPToolDefinitionError(
                f"{server_name}.{raw_name} requires a description"
            )
        if not isinstance(schema, Mapping) or schema.get("type") != "object":
            raise MCPToolDefinitionError(
                f"{server_name}.{raw_name} requires an object input schema"
            )
        definitions[raw_name] = {
            "name": raw_name,
            "description": description.strip(),
            "input_schema": dict(schema),
        }
    return definitions


_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def _server_name(value: Any) -> str:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise ValueError("MCP server name contains invalid characters")
    return value


def _split_exposed_name(name: str) -> tuple[str, str]:
    if not isinstance(name, str):
        raise TypeError("tool_name must be a string")
    parts = name.split("__", 2)
    if (
        len(parts) != 3
        or parts[0] != "mcp"
        or not _NAME_PATTERN.fullmatch(parts[1])
        or not _NAME_PATTERN.fullmatch(parts[2])
    ):
        raise MCPGatewayError(f"invalid exposed MCP tool name: {name!r}")
    return parts[1], parts[2]


def _validate_trusted_context(context: Mapping[str, str]) -> None:
    for name in ("user_id", "case_id"):
        value = context.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"trusted_context.{name} is required")


def _agent_tools_for(node_name: str) -> tuple[str, ...]:
    from jmr.graph.manifest import agent_tools_for

    return agent_tools_for(node_name)


__all__ = [
    "MCPAccessDeniedError",
    "MCPGatewayError",
    "MCPServer",
    "MCPServerRegistry",
    "MCPToolDefinitionError",
    "NodeScopedMCPGateway",
]
