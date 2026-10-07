"""MCP tools for explicit, versioned applicant-memory management."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from psycopg.types.json import Jsonb

from jmr.domain import MemoryStatus


class MemoryMCPServer:
    """Bind every operation to one trusted user namespace."""

    def __init__(self, store: Any, *, user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be non-empty")
        self.store = store
        self.user_id = user_id.strip()
        self.namespace = ("users", self.user_id, "profile")
        self._replays: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}

    def list_tools(self) -> list[dict[str, Any]]:
        content_schema = {"type": "object", "additionalProperties": True}
        tags_schema = {"type": "array", "items": {"type": "string"}}
        confirmation = {
            "confirmed_by_user": {"type": "boolean", "enum": [True]},
            "idempotency_key": {"type": "string", "minLength": 1},
        }
        return [
            _tool(
                "list_memories",
                {
                    "kind": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "status": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                [],
            ),
            _tool(
                "get_memory",
                {"memory_id": {"type": "string", "minLength": 1}},
                ["memory_id"],
            ),
            _tool(
                "create_memory",
                {
                    "kind": {"type": "string", "minLength": 1},
                    "content": content_schema,
                    "tags": tags_schema,
                    **confirmation,
                },
                ["kind", "content", "confirmed_by_user", "idempotency_key"],
            ),
            _tool(
                "update_memory",
                {
                    "memory_id": {"type": "string", "minLength": 1},
                    "content": content_schema,
                    "tags": tags_schema,
                    "expected_version": {"type": "integer", "minimum": 1},
                    **confirmation,
                },
                [
                    "memory_id",
                    "content",
                    "expected_version",
                    "confirmed_by_user",
                    "idempotency_key",
                ],
            ),
            _tool(
                "delete_memory",
                {
                    "memory_id": {"type": "string", "minLength": 1},
                    "expected_version": {"type": "integer", "minimum": 1},
                    **confirmation,
                },
                [
                    "memory_id",
                    "expected_version",
                    "confirmed_by_user",
                    "idempotency_key",
                ],
            ),
        ]

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if not isinstance(arguments, Mapping):
            raise TypeError("memory tool arguments must be an object")
        if name == "list_memories":
            limit = arguments.get("limit", 100)
            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or not 1 <= limit <= 100
            ):
                raise ValueError("limit must be between 1 and 100")
            status = arguments.get("status", MemoryStatus.ACTIVE.value)
            if status not in {item.value for item in MemoryStatus}:
                raise ValueError("invalid memory status")
            kind = arguments.get("kind")
            tags = arguments.get("tags", [])
            if kind is not None:
                kind = _text(kind, "kind")
            if not isinstance(tags, list) or not all(
                isinstance(tag, str) for tag in tags
            ):
                raise ValueError("tags must be a string array")
            memories = list(
                self.store.list(
                    namespace=self.namespace,
                    filters={"status": status},
                    limit=100,
                )
            )
            selected = [
                item
                for item in memories
                if (kind is None or (item.get("value") or {}).get("kind") == kind)
                and set(tags).issubset(set((item.get("value") or {}).get("tags", [])))
            ][:limit]
            return {
                "status": "SUCCESS" if selected else "NO_RESULT",
                "items": selected,
            }
        memory_id = arguments.get("memory_id")
        if name == "get_memory":
            item = self.store.get(namespace=self.namespace, key=memory_id)
            return {
                "status": "SUCCESS" if item is not None else "NO_RESULT",
                "memory": item,
            }
        if name == "create_memory":
            return self._mutate(name, arguments)
        if name == "update_memory":
            return self._mutate(name, arguments)
        if name == "delete_memory":
            return self._mutate(name, arguments)
        raise ValueError(f"Unknown memory tool {name!r}")

    def _mutate(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if arguments.get("confirmed_by_user") is not True:
            raise ValueError("memory mutation requires explicit user confirmation")
        idempotency_key = _text(arguments.get("idempotency_key"), "idempotency_key")
        payload = dict(arguments)
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        connection = getattr(self.store, "connection", None)
        if connection is None:
            replay = self._replays.get((name, idempotency_key))
            if replay is not None:
                if replay[0] != digest:
                    raise ValueError("idempotency key reused with different payload")
                return dict(replay[1])
            result = self._execute_mutation(name, payload)
            self._replays[(name, idempotency_key)] = (digest, result)
            return result

        with connection.transaction():
            connection.execute(
                "INSERT INTO jmr.users(user_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (self.user_id,),
            )
            row = connection.execute(
                """
                SELECT payload_hash, result FROM jmr.profile_memory_operations
                WHERE user_id=%s AND operation=%s AND idempotency_key=%s
                FOR UPDATE
                """,
                (self.user_id, name, idempotency_key),
            ).fetchone()
            if row is not None:
                stored_hash = (
                    row["payload_hash"] if isinstance(row, Mapping) else row[0]
                )
                stored_result = row["result"] if isinstance(row, Mapping) else row[1]
                if stored_hash != digest:
                    raise ValueError("idempotency key reused with different payload")
                result = dict(stored_result)
            else:
                result = self._execute_mutation(name, payload)
                connection.execute(
                    """
                    INSERT INTO jmr.profile_memory_operations
                        (user_id, operation, idempotency_key, payload_hash, result)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (self.user_id, name, idempotency_key, digest, Jsonb(result)),
                )
                connection.execute(
                    """
                    INSERT INTO jmr.audit_events
                        (audit_event_id, user_id, event_type, payload)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        uuid4(),
                        self.user_id,
                        f"memory.{name}",
                        Jsonb({"memory_id": result["memory"]["memory_id"]}),
                    ),
                )
        connection.commit()
        return result

    def _execute_mutation(
        self, name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        memory_id = arguments.get("memory_id")
        if name == "create_memory":
            memory = self.store.create(
                namespace=self.namespace,
                value=_value(arguments, kind=_text(arguments.get("kind"), "kind")),
            )
        elif name == "update_memory":
            current = self.store.get(
                namespace=self.namespace,
                key=_text(memory_id, "memory_id"),
            )
            if current is None:
                raise KeyError(memory_id)
            current_value = current.get("value") or {}
            memory = self.store.update(
                namespace=self.namespace,
                key=_text(memory_id, "memory_id"),
                value=_value(
                    arguments,
                    kind=_text(current_value.get("kind"), "kind"),
                    default_tags=current_value.get("tags", []),
                ),
                expected_version=_version(arguments),
            )
        else:
            memory = self.store.delete(
                namespace=self.namespace,
                key=_text(memory_id, "memory_id"),
                expected_version=_version(arguments),
            )
        return {"status": "SUCCESS", "memory": dict(memory)}


def create_memory_server(store: Any, *, user_id: str) -> MemoryMCPServer:
    return MemoryMCPServer(store, user_id=user_id)


def _tool(
    name: str, properties: Mapping[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Explicitly manage applicant profile memory: {name}.",
        "inputSchema": {
            "type": "object",
            "properties": dict(properties),
            "required": required,
            "additionalProperties": False,
        },
    }


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value.strip()


def _value(
    arguments: Mapping[str, Any],
    *,
    kind: str,
    default_tags: Any = (),
) -> Mapping[str, Any]:
    content = arguments.get("content")
    if not isinstance(content, Mapping):
        raise TypeError("content must be an object")
    tags = arguments.get("tags", default_tags)
    if not isinstance(tags, (list, tuple)):
        raise TypeError("tags must be an array")
    return {
        "kind": kind,
        "content": dict(content),
        "tags": list(dict.fromkeys(_text(tag, "tag") for tag in tags)),
    }


def _version(arguments: Mapping[str, Any]) -> int:
    value = arguments.get("expected_version")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("expected_version must be a positive integer")
    return value
