"""Versioned, user-scoped memory facade used by P5 nodes and tests."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from psycopg.types.json import Jsonb

from jmr.domain import MemoryStatus
from jmr.runtime.ports import SystemClock, UUIDGenerator


class InMemoryMemoryStore:
    """Small PostgresStore-compatible fake with explicit lifecycle states."""

    def __init__(self, *, clock: Any | None = None, id_generator: Any | None = None):
        self.clock = clock or SystemClock()
        self.id_generator = id_generator or UUIDGenerator()
        self._lock = threading.RLock()
        self._items: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}

    def list(
        self,
        *,
        namespace: tuple[str, ...],
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[Mapping[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            items = [
                dict(value)
                for (item_namespace, _), value in self._items.items()
                if item_namespace == namespace
                and all(value.get(key) == expected for key, expected in filters.items())
            ]
        items.sort(key=lambda item: (item["created_at"], item["memory_id"]))
        return items[:limit]

    def get(
        self,
        *,
        namespace: tuple[str, ...],
        key: str,
    ) -> Mapping[str, Any] | None:
        with self._lock:
            item = self._items.get((namespace, key))
            return dict(item) if item is not None else None

    def create(
        self,
        *,
        namespace: tuple[str, ...],
        value: Mapping[str, Any],
        key: str | None = None,
    ) -> Mapping[str, Any]:
        memory_id = key or self.id_generator.new_id("memory")
        item_key = (namespace, memory_id)
        with self._lock:
            if item_key in self._items:
                raise ValueError("memory key already exists")
            now = self.clock.now().isoformat()
            item = {
                "memory_id": memory_id,
                "status": MemoryStatus.ACTIVE.value,
                "version": 1,
                "value": dict(value),
                "created_at": now,
                "updated_at": now,
            }
            self._items[item_key] = item
            return dict(item)

    def update(
        self,
        *,
        namespace: tuple[str, ...],
        key: str,
        value: Mapping[str, Any],
        expected_version: int,
    ) -> Mapping[str, Any]:
        with self._lock:
            current = self._items.get((namespace, key))
            if current is None:
                raise KeyError(key)
            if current["status"] != MemoryStatus.ACTIVE.value:
                raise ValueError("only ACTIVE memories can be updated")
            if current["version"] != expected_version:
                raise ValueError("memory version conflict")
            current["status"] = MemoryStatus.SUPERSEDED.value
            current["updated_at"] = self.clock.now().isoformat()
            replacement = self.create(namespace=namespace, value=value)
            replacement = dict(replacement, supersedes=key)
            self._items[(namespace, replacement["memory_id"])] = replacement
            return replacement

    def delete(
        self,
        *,
        namespace: tuple[str, ...],
        key: str,
        expected_version: int,
    ) -> Mapping[str, Any]:
        with self._lock:
            current = self._items.get((namespace, key))
            if current is None:
                raise KeyError(key)
            if current["version"] != expected_version:
                raise ValueError("memory version conflict")
            current["status"] = MemoryStatus.DELETED.value
            current["version"] += 1
            current["updated_at"] = self.clock.now().isoformat()
            return dict(current)


class PostgresMemoryStore:
    """Versioned applicant-memory facade over LangGraph PostgresStore."""

    def __init__(
        self,
        store: Any,
        *,
        clock: Any | None = None,
        id_generator: Any | None = None,
    ):
        for attribute in ("search", "get", "put"):
            if not callable(getattr(store, attribute, None)):
                raise TypeError(f"store is missing {attribute}")
        self.store = store
        self.connection = getattr(store, "conn", None)
        if not callable(getattr(self.connection, "transaction", None)) or not callable(
            getattr(self.connection, "execute", None)
        ):
            raise TypeError(
                "PostgresMemoryStore requires a direct PostgreSQL connection"
            )
        self.clock = clock or SystemClock()
        self.id_generator = id_generator or UUIDGenerator()

    def list(
        self,
        *,
        namespace: tuple[str, ...],
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[Mapping[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        return [
            _postgres_item(item)
            for item in self.store.search(
                namespace,
                filter=dict(filters),
                limit=limit,
            )
        ]

    def get(
        self,
        *,
        namespace: tuple[str, ...],
        key: str,
    ) -> Mapping[str, Any] | None:
        item = self.store.get(namespace, key)
        return None if item is None else _postgres_item(item)

    def create(
        self,
        *,
        namespace: tuple[str, ...],
        value: Mapping[str, Any],
        key: str | None = None,
    ) -> Mapping[str, Any]:
        memory_id = key or self.id_generator.new_id("memory")
        now = self.clock.now().isoformat()
        item = {
            "memory_id": memory_id,
            "status": MemoryStatus.ACTIVE.value,
            "version": 1,
            "value": dict(value),
            "created_at": now,
            "updated_at": now,
        }
        with self.connection.transaction():
            inserted = self.connection.execute(
                """
                INSERT INTO store (prefix, key, value)
                VALUES (%s, %s, %s)
                ON CONFLICT (prefix, key) DO NOTHING
                RETURNING key
                """,
                (_namespace_text(namespace), memory_id, Jsonb(item)),
            ).fetchone()
            if inserted is None:
                raise ValueError("memory key already exists")
        return dict(item)

    def update(
        self,
        *,
        namespace: tuple[str, ...],
        key: str,
        value: Mapping[str, Any],
        expected_version: int,
    ) -> Mapping[str, Any]:
        prefix = _namespace_text(namespace)
        with self.connection.transaction():
            current = _locked_item(self.connection, prefix, key)
            if current is None:
                raise KeyError(key)
            if current.get("status") != MemoryStatus.ACTIVE.value:
                raise ValueError("only ACTIVE memories can be updated")
            if current.get("version") != expected_version:
                raise ValueError("memory version conflict")
            now = self.clock.now().isoformat()
            superseded = dict(current)
            superseded["status"] = MemoryStatus.SUPERSEDED.value
            superseded["updated_at"] = now
            replacement_id = self.id_generator.new_id("memory")
            replacement = {
                "memory_id": replacement_id,
                "status": MemoryStatus.ACTIVE.value,
                "version": 1,
                "value": dict(value),
                "created_at": now,
                "updated_at": now,
                "supersedes": key,
            }
            inserted = self.connection.execute(
                """
                INSERT INTO store (prefix, key, value)
                VALUES (%s, %s, %s)
                ON CONFLICT (prefix, key) DO NOTHING
                RETURNING key
                """,
                (prefix, replacement_id, Jsonb(replacement)),
            ).fetchone()
            if inserted is None:
                raise ValueError("replacement memory key already exists")
            self.connection.execute(
                "UPDATE store SET value = %s, updated_at = CURRENT_TIMESTAMP "
                "WHERE prefix = %s AND key = %s",
                (Jsonb(superseded), prefix, key),
            )
        return replacement

    def delete(
        self,
        *,
        namespace: tuple[str, ...],
        key: str,
        expected_version: int,
    ) -> Mapping[str, Any]:
        prefix = _namespace_text(namespace)
        with self.connection.transaction():
            current = _locked_item(self.connection, prefix, key)
            if current is None:
                raise KeyError(key)
            if current.get("version") != expected_version:
                raise ValueError("memory version conflict")
            deleted = dict(current)
            deleted["status"] = MemoryStatus.DELETED.value
            deleted["version"] = expected_version + 1
            deleted["updated_at"] = self.clock.now().isoformat()
            self.connection.execute(
                "UPDATE store SET value = %s, updated_at = CURRENT_TIMESTAMP "
                "WHERE prefix = %s AND key = %s",
                (Jsonb(deleted), prefix, key),
            )
        return deleted


def _namespace_text(namespace: tuple[str, ...]) -> str:
    if (
        not isinstance(namespace, tuple)
        or not namespace
        or not all(isinstance(part, str) and part for part in namespace)
    ):
        raise ValueError("memory namespace must contain non-empty string parts")
    return ".".join(namespace)


def _locked_item(connection: Any, prefix: str, key: str) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT value FROM store WHERE prefix = %s AND key = %s FOR UPDATE",
        (prefix, key),
    ).fetchone()
    if row is None:
        return None
    value = row["value"] if isinstance(row, Mapping) else row[0]
    if not isinstance(value, Mapping):
        raise TypeError("stored memory value must be an object")
    return dict(value)


def _postgres_item(item: Any) -> dict[str, Any]:
    value = getattr(item, "value", None)
    if not isinstance(value, Mapping):
        if isinstance(item, Mapping):
            value = item.get("value", item)
        else:
            raise TypeError("PostgresStore item value must be an object")
    result = dict(value)
    key = getattr(item, "key", None)
    if isinstance(key, str):
        result.setdefault("memory_id", key)
    return result
