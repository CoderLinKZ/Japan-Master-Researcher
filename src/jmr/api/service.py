"""Application service behind the HTTP transport."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from psycopg.types.json import Jsonb

from jmr.cli import load_runtime_environment, run_turn
from jmr.domain import RunStatus
from jmr.graph import compile_graph
from jmr.operations.cases import (
    CaseOwnershipError,
    CaseSchemaNotReadyError,
    DeletionPlanStaleError,
    delete_case,
)
from jmr.operations.health import readiness as dependency_readiness
from jmr.persistence import (
    FileObjectStore,
    PostgresMemoryStore,
    open_postgres_repository,
    open_postgres_store,
)
from jmr.runtime import (
    BoundedRetryPolicy,
    CompositeEventSink,
    ConsoleEventSink,
    JMRRuntimeContext,
    JsonLineEventSink,
    MCPServerRegistry,
    NodeScopedMCPGateway,
    create_anthropic_model_registry,
    open_postgres_checkpointer,
)
from jmr.runtime.security import PrincipalSecurityPolicy
from mcp_servers import create_kaken_server, create_scholar_server
from mcp_servers.memory import create_memory_server

from .uploads import extract_document
from .workspace import case_timeline
from .workspace import case_workspace as read_case_workspace
from .workspace import list_cases as read_cases

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CREATE_SCOPE_CASE_ID = "__http_create_case__"
_DELETION_PLAN_TTL_SECONDS = 15 * 60


class APIServiceError(RuntimeError):
    """Base class for errors safe to translate at the HTTP boundary."""


class ResourceNotFoundError(APIServiceError):
    pass


class OperationConflictError(APIServiceError):
    pass


class CaseBusyError(OperationConflictError):
    pass


class ServiceUnavailableError(APIServiceError):
    pass


class AgentAPIService(Protocol):
    def startup(self) -> None: ...

    def shutdown(self) -> None: ...

    def readiness(self) -> Mapping[str, Any]: ...

    def start_case(
        self,
        *,
        user_id: str,
        message: str,
        case_id: str | None = None,
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def send_message(
        self, *, user_id: str, case_id: str, message: str
    ) -> Mapping[str, Any]: ...

    def resume_case(
        self,
        *,
        user_id: str,
        case_id: str,
        payload: Mapping[str, Any],
        interrupt_token: str,
    ) -> Mapping[str, Any]: ...

    def retry_case(self, *, user_id: str, case_id: str) -> Mapping[str, Any]: ...

    def get_case(self, *, user_id: str, case_id: str) -> Mapping[str, Any]: ...

    def get_progress(self, *, user_id: str, case_id: str) -> Mapping[str, Any]: ...

    def list_cases(self, *, user_id: str) -> Mapping[str, Any]: ...

    def get_workspace(self, *, user_id: str, case_id: str) -> Mapping[str, Any]: ...

    def get_conversation(self, *, user_id: str, case_id: str) -> Mapping[str, Any]: ...

    def get_result(self, *, user_id: str, case_id: str) -> Mapping[str, Any]: ...

    def call_memory(
        self, *, user_id: str, tool_name: str, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def upload_memory_document(
        self, *, user_id: str, filename: str, content: bytes
    ) -> Mapping[str, Any]: ...

    def get_memory_document(
        self, *, user_id: str, memory_id: str
    ) -> tuple[str, bytes]: ...

    def plan_case_deletion(
        self, *, user_id: str, case_id: str
    ) -> Mapping[str, Any]: ...

    def confirm_case_deletion(
        self, *, user_id: str, case_id: str, plan_token: str
    ) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class _RuntimeSession:
    graph: Any
    runtime: JMRRuntimeContext
    repository: Any


class ProductionAgentAPIService:
    """Open request-scoped database resources around one graph turn."""

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        load_runtime_environment()
        self._environment = os.environ if environment is None else environment
        self._object_root = Path(
            self._environment.get(
                "JMR_OBJECT_STORE_DIR", str(PROJECT_ROOT / ".jmr" / "objects")
            )
        )
        self._event_path = Path(
            self._environment.get(
                "JMR_EVENT_LOG_PATH", str(PROJECT_ROOT / ".jmr" / "events.jsonl")
            )
        )
        self._event_stream: Any | None = None
        self._event_sink: CompositeEventSink | None = None
        self._object_store: FileObjectStore | None = None
        self._lifecycle_lock = threading.RLock()
        self._case_locks_guard = threading.Lock()
        self._exclusive_locks: dict[tuple[str, str], threading.Lock] = {}
        raw_concurrency = self._environment.get("JMR_API_MAX_CONCURRENT_RUNS", "4")
        try:
            max_concurrent_runs = int(raw_concurrency)
        except ValueError as exc:
            raise ValueError("JMR_API_MAX_CONCURRENT_RUNS must be an integer") from exc
        if not 1 <= max_concurrent_runs <= 64:
            raise ValueError("JMR_API_MAX_CONCURRENT_RUNS must be between 1 and 64")
        self._turn_slots = threading.BoundedSemaphore(max_concurrent_runs)

    def startup(self) -> None:
        with self._lifecycle_lock:
            if self._event_stream is not None:
                return
            self._event_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self._event_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                os.fchmod(descriptor, 0o600)
                event_stream = os.fdopen(descriptor, "a", encoding="utf-8")
            except Exception:
                os.close(descriptor)
                raise
            try:
                object_store = FileObjectStore(self._object_root)
                console_sink = ConsoleEventSink()
                event_sink = CompositeEventSink(
                    [JsonLineEventSink(event_stream), console_sink]
                )
            except Exception:
                event_stream.close()
                raise
            self._event_stream = event_stream
            self._event_sink = event_sink
            self._object_store = object_store

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            if self._event_stream is not None:
                self._event_stream.close()
            self._event_stream = None
            self._event_sink = None
            self._object_store = None

    def readiness(self) -> Mapping[str, Any]:
        database_url = self._environment.get("DATABASE_URL", "").strip()
        if not database_url:
            return {
                "status": "failed",
                "checks": {"database": "failed", "schema": "unknown"},
            }
        import psycopg

        try:
            with psycopg.connect(database_url) as connection:
                report = dict(
                    dependency_readiness(
                        connection,
                        object_store_root=self._object_root,
                    )
                )
                runtime_tables = connection.execute(
                    "SELECT to_regclass('public.checkpoints'), "
                    "to_regclass('public.store')"
                ).fetchone()
                checks = dict(report.get("checks", {}))
                checks["checkpointer"] = (
                    "ok" if runtime_tables and runtime_tables[0] else "failed"
                )
                checks["memory_store"] = (
                    "ok" if runtime_tables and runtime_tables[1] else "failed"
                )
                report["checks"] = checks
        except psycopg.Error:
            report = {
                "status": "failed",
                "checks": {"database": "failed", "schema": "unknown"},
            }
        checks = dict(report.get("checks", {}))
        checks["model_configuration"] = (
            "ok"
            if all(
                self._environment.get(name, "").strip()
                for name in ("ANTHROPIC_API_KEY", "MODEL_ID")
            )
            else "failed"
        )
        report["checks"] = checks
        report["status"] = (
            "ok"
            if checks and all(value == "ok" for value in checks.values())
            else "failed"
        )
        return report

    def start_case(
        self,
        *,
        user_id: str,
        message: str,
        case_id: str | None = None,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        submitted_at = datetime.now(UTC)
        request_hash = _json_hash({"message": message, "case_id": case_id})
        receipt, fresh = self._reserve_create_case(
            user_id=user_id,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            case_id=case_id,
        )
        selected_case_id = str(receipt["case_id"])
        with self._exclusive_guard("case", selected_case_id):
            self._read_operation(
                user_id=user_id,
                case_id=_CREATE_SCOPE_CASE_ID,
                operation="http_create_case",
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            with self._open_reader() as (graph, repository):
                existing = repository.get_case(
                    user_id=user_id, case_id=selected_case_id
                )
                snapshot = graph.get_state(_config(user_id, selected_case_id))
                deleting = _case_has_deletion_job(repository, selected_case_id)
                if fresh and (existing is not None or snapshot.values or deleting):
                    self._discard_create_receipt(user_id, idempotency_key)
                    raise OperationConflictError("case already exists")
                if deleting:
                    raise OperationConflictError("case deletion is in progress")
                if snapshot.values:
                    if existing is None:
                        raise OperationConflictError(
                            "case checkpoint exists without a business record"
                        )
                    repository.save_user_message(
                        user_id=user_id,
                        case_id=selected_case_id,
                        content=message,
                        idempotency_key=f"create:{idempotency_key}",
                        created_at=submitted_at,
                    )
                    return _project_snapshot(snapshot)
            with self._turn_slot():
                with self._open_runtime(user_id) as session:
                    try:
                        run_turn(
                            session.graph,
                            user_id=user_id,
                            case_id=selected_case_id,
                            user_input=message,
                            runtime_context=session.runtime,
                        )
                    except Exception:
                        # The graph can fail after creating the business case.
                        # Preserve the submitted question for diagnosis/retry.
                        try:
                            if (
                                session.repository.get_case(
                                    user_id=user_id, case_id=selected_case_id
                                )
                                is not None
                            ):
                                session.repository.save_user_message(
                                    user_id=user_id,
                                    case_id=selected_case_id,
                                    content=message,
                                    idempotency_key=f"create:{idempotency_key}",
                                    created_at=submitted_at,
                                )
                        except Exception:
                            pass
                        raise
                    session.repository.save_user_message(
                        user_id=user_id,
                        case_id=selected_case_id,
                        content=message,
                        idempotency_key=f"create:{idempotency_key}",
                        created_at=submitted_at,
                    )
                    return _project_snapshot(
                        session.graph.get_state(_config(user_id, selected_case_id))
                    )

    def _reserve_create_case(
        self,
        *,
        user_id: str,
        idempotency_key: str,
        request_hash: str,
        case_id: str | None,
    ) -> tuple[Mapping[str, Any], bool]:
        selected_case_id = case_id or str(uuid.uuid4())
        with open_postgres_repository(environment=self._environment) as repository:
            connection = repository.connection
            with connection.transaction():
                connection.execute(
                    "INSERT INTO jmr.users(user_id) VALUES (%s) ON CONFLICT DO NOTHING",
                    (user_id,),
                )
                inserted = connection.execute(
                    """
                    INSERT INTO jmr.idempotency_keys
                        (user_id, case_id, operation, idempotency_key,
                         payload_hash, result)
                    VALUES (%s, %s, 'http_create_case', %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING idempotency_key
                    """,
                    (
                        user_id,
                        _CREATE_SCOPE_CASE_ID,
                        idempotency_key,
                        request_hash,
                        Jsonb({"case_id": selected_case_id}),
                    ),
                ).fetchone()
            connection.commit()
        receipt = self._read_operation(
            user_id=user_id,
            case_id=_CREATE_SCOPE_CASE_ID,
            operation="http_create_case",
            idempotency_key=idempotency_key,
            request_hash=request_hash,
        )
        return receipt, inserted is not None

    def _discard_create_receipt(self, user_id: str, idempotency_key: str) -> None:
        with open_postgres_repository(environment=self._environment) as repository:
            repository.connection.execute(
                """
                DELETE FROM jmr.idempotency_keys
                WHERE user_id=%s AND case_id=%s AND operation='http_create_case'
                  AND idempotency_key=%s
                """,
                (user_id, _CREATE_SCOPE_CASE_ID, idempotency_key),
            )
            repository.connection.commit()

    def _read_operation(
        self,
        *,
        user_id: str,
        case_id: str,
        operation: str,
        idempotency_key: str,
        request_hash: str,
    ) -> Mapping[str, Any]:
        result = self._lookup_operation(
            user_id=user_id,
            case_id=case_id,
            operation=operation,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
        )
        if result is None:
            raise OperationConflictError("operation receipt is unavailable")
        return result

    def _lookup_operation(
        self,
        *,
        user_id: str,
        case_id: str,
        operation: str,
        idempotency_key: str,
        request_hash: str,
    ) -> Mapping[str, Any] | None:
        with open_postgres_repository(environment=self._environment) as repository:
            row = repository.connection.execute(
                """
                SELECT payload_hash, result FROM jmr.idempotency_keys
                WHERE user_id=%s AND case_id=%s AND operation=%s
                  AND idempotency_key=%s
                """,
                (user_id, case_id, operation, idempotency_key),
            ).fetchone()
        if row is None:
            return None
        if row[0] != request_hash:
            raise OperationConflictError("idempotency key was used for another request")
        if not isinstance(row[1], Mapping):
            raise RuntimeError("operation receipt is invalid")
        return dict(row[1])

    def send_message(
        self, *, user_id: str, case_id: str, message: str
    ) -> Mapping[str, Any]:
        with self._exclusive_guard("case", case_id):
            with self._turn_slot():
                with self._open_runtime(user_id) as session:
                    return self._send_message_in_session(
                        session=session,
                        user_id=user_id,
                        case_id=case_id,
                        message=message,
                    )

    def _send_message_in_session(
        self,
        *,
        session: _RuntimeSession,
        user_id: str,
        case_id: str,
        message: str,
    ) -> Mapping[str, Any]:
        snapshot = self._owned_snapshot(session, user_id, case_id)
        status = snapshot.values.get("run_status")
        if status in {RunStatus.COMPLETED.value, RunStatus.CANCELLED.value}:
            raise OperationConflictError("case is already closed")
        if _snapshot_interrupt(snapshot) is not None:
            raise OperationConflictError(
                "case is waiting for an interrupt resume payload"
            )
        if _snapshot_retry_available(snapshot):
            raise OperationConflictError(
                "case has a failed stage; retry it before sending another message"
            )
        session.repository.save_user_message(
            user_id=user_id,
            case_id=case_id,
            content=message,
            idempotency_key=f"message:{uuid.uuid4().hex}",
        )
        run_turn(
            session.graph,
            user_id=user_id,
            case_id=case_id,
            user_input=message,
            runtime_context=session.runtime,
        )
        return _project_snapshot(session.graph.get_state(_config(user_id, case_id)))

    def resume_case(
        self,
        *,
        user_id: str,
        case_id: str,
        payload: Mapping[str, Any],
        interrupt_token: str,
    ) -> Mapping[str, Any]:
        with self._exclusive_guard("case", case_id):
            with self._turn_slot():
                with self._open_runtime(user_id) as session:
                    return self._resume_in_session(
                        session=session,
                        user_id=user_id,
                        case_id=case_id,
                        payload=payload,
                        interrupt_token=interrupt_token,
                    )

    def retry_case(self, *, user_id: str, case_id: str) -> Mapping[str, Any]:
        """Continue only a failed checkpoint without injecting a new user turn."""

        with self._exclusive_guard("case", case_id):
            with self._turn_slot():
                with self._open_runtime(user_id) as session:
                    snapshot = self._owned_snapshot(session, user_id, case_id)
                    if not _snapshot_retry_available(snapshot):
                        raise OperationConflictError(
                            "case has no failed stage to retry"
                        )
                    session.graph.invoke(
                        None,
                        config=_config(user_id, case_id),
                        context=session.runtime,
                    )
                    return _project_snapshot(
                        session.graph.get_state(_config(user_id, case_id))
                    )

    def _resume_in_session(
        self,
        *,
        session: _RuntimeSession,
        user_id: str,
        case_id: str,
        payload: Mapping[str, Any],
        interrupt_token: str,
    ) -> Mapping[str, Any]:
        snapshot = self._owned_snapshot(session, user_id, case_id)
        request_hash = _json_hash({"payload": payload})
        receipt = self._lookup_operation(
            user_id=user_id,
            case_id=case_id,
            operation="http_resume",
            idempotency_key=interrupt_token,
            request_hash=request_hash,
        )
        if receipt is not None:
            if isinstance(receipt.get("view"), Mapping):
                return dict(receipt["view"])
            raise OperationConflictError(
                "resume outcome is uncertain; inspect the current case state"
            )
        identity = _snapshot_interrupt_identity(snapshot)
        if identity is None or identity[2] != interrupt_token:
            raise OperationConflictError("interrupt token is stale")
        run_turn(
            session.graph,
            user_id=user_id,
            case_id=case_id,
            resume=payload,
            resume_interrupt_id=identity[0],
            runtime_context=session.runtime,
        )
        view = _project_snapshot(session.graph.get_state(_config(user_id, case_id)))
        self._save_resume_receipt(
            user_id=user_id,
            case_id=case_id,
            interrupt_token=interrupt_token,
            request_hash=request_hash,
            view=view,
        )
        return view

    def _save_resume_receipt(
        self,
        *,
        user_id: str,
        case_id: str,
        interrupt_token: str,
        request_hash: str,
        view: Mapping[str, Any],
    ) -> None:
        with open_postgres_repository(environment=self._environment) as repository:
            connection = repository.connection
            with connection.transaction():
                connection.execute(
                    """
                    INSERT INTO jmr.idempotency_keys
                        (user_id, case_id, operation, idempotency_key,
                         payload_hash, result)
                    VALUES (%s, %s, 'http_resume', %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        user_id,
                        case_id,
                        interrupt_token,
                        request_hash,
                        Jsonb({"view": dict(view)}),
                    ),
                )
            connection.commit()
        receipt = self._read_operation(
            user_id=user_id,
            case_id=case_id,
            operation="http_resume",
            idempotency_key=interrupt_token,
            request_hash=request_hash,
        )
        if not isinstance(receipt.get("view"), Mapping):
            raise OperationConflictError("resume receipt is incomplete")

    def get_case(self, *, user_id: str, case_id: str) -> Mapping[str, Any]:
        with self._open_reader() as (graph, repository):
            case = repository.get_case(user_id=user_id, case_id=case_id)
            if case is None:
                raise ResourceNotFoundError("case not found")
            if _case_has_deletion_job(repository, case_id):
                raise OperationConflictError("case deletion is in progress")
            snapshot = graph.get_state(_config(user_id, case_id))
            if not snapshot.values:
                raise ResourceNotFoundError("case checkpoint not found")
            return _project_snapshot(snapshot)

    def list_cases(self, *, user_id: str) -> Mapping[str, Any]:
        return read_cases(self._environment, user_id)

    def get_progress(self, *, user_id: str, case_id: str) -> Mapping[str, Any]:
        """Read committed progress without blocking an active graph turn."""

        with open_postgres_repository(environment=self._environment) as repository:
            case = repository.get_case(user_id=user_id, case_id=case_id)
            if case is None:
                raise ResourceNotFoundError("case not found")
            if _case_has_deletion_job(repository, case_id):
                raise OperationConflictError("case deletion is in progress")
            return {
                "case_id": case_id,
                "workflow_stage": case["workflow_stage"],
                "run_status": case["run_status"],
                "events": case_timeline(repository.connection, user_id, case_id),
            }

    def get_workspace(self, *, user_id: str, case_id: str) -> Mapping[str, Any]:
        result = read_case_workspace(self._environment, user_id, case_id)
        if result is None:
            raise ResourceNotFoundError("case not found")
        return result

    def get_conversation(self, *, user_id: str, case_id: str) -> Mapping[str, Any]:
        with self._open_reader() as (graph, repository):
            case = repository.get_case(user_id=user_id, case_id=case_id)
            if case is None:
                raise ResourceNotFoundError("case not found")
            if _case_has_deletion_job(repository, case_id):
                raise OperationConflictError("case deletion is in progress")
            snapshot = graph.get_state(_config(user_id, case_id))
            if not snapshot.values:
                raise ResourceNotFoundError("case checkpoint not found")
            events = case_timeline(repository.connection, user_id, case_id)
            messages = [
                {"role": "user", "content": event["content"]}
                for event in events
                if event["kind"] == "user_message"
            ]
            if not messages:
                for index, message in enumerate(snapshot.values.get("messages", [])):
                    if getattr(message, "type", "") != "human":
                        continue
                    content = getattr(message, "content", "")
                    if isinstance(content, str):
                        messages.append({"role": "user", "content": content})
                        events.append(
                            {
                                "id": f"checkpoint-user:{index:03d}",
                                "kind": "user_message",
                                "content": content,
                                "created_at": case["created_at"],
                            }
                        )
                events.sort(key=lambda event: (event["created_at"], event["id"]))
            return {"case_id": case_id, "messages": messages, "events": events}

    def get_result(self, *, user_id: str, case_id: str) -> Mapping[str, Any]:
        with self._exclusive_guard("case", case_id):
            with self._open_reader() as (graph, repository):
                case = repository.get_case(user_id=user_id, case_id=case_id)
                if case is None:
                    raise ResourceNotFoundError("case not found")
                if _case_has_deletion_job(repository, case_id):
                    raise OperationConflictError("case deletion is in progress")
                snapshot = graph.get_state(_config(user_id, case_id))
                values = dict(snapshot.values or {})
                if values.get("run_status") != RunStatus.COMPLETED.value:
                    raise OperationConflictError("case result is not ready")
                iteration_id = values.get("current_outreach_iteration_id")
                if not isinstance(iteration_id, str) or not iteration_id:
                    raise OperationConflictError("completed case has no final result")
                iteration = repository.get_outreach_iteration(
                    user_id=user_id,
                    case_id=case_id,
                    iteration_id=iteration_id,
                )
                return {
                    "user_id": user_id,
                    "case_id": case_id,
                    "iteration_id": iteration_id,
                    "content": str(iteration.get("content", "")),
                    "citation_ids": list(iteration.get("citation_ids", [])),
                }

    def call_memory(
        self,
        *,
        user_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if tool_name in {"create_memory", "update_memory", "delete_memory"}:
            try:
                with self._exclusive_guard("memory", user_id):
                    return self._call_memory_once(
                        user_id=user_id,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
            except CaseBusyError as exc:
                raise OperationConflictError(
                    "another memory mutation is already active"
                ) from exc
        return self._call_memory_once(
            user_id=user_id,
            tool_name=tool_name,
            arguments=arguments,
        )

    def upload_memory_document(
        self, *, user_id: str, filename: str, content: bytes
    ) -> Mapping[str, Any]:
        extracted = extract_document(filename, content)
        object_store = self._require_object_store()
        user_hash = hashlib.sha256(user_id.encode()).hexdigest()
        key = f"uploads/{user_hash}/{uuid.uuid4().hex}"
        uri = object_store.put(
            object_key=key, content=content, content_type=extracted["mime_type"]
        )
        try:
            return self.call_memory(
                user_id=user_id,
                tool_name="create_memory",
                arguments={
                    "kind": "uploaded_document",
                    "content": {**extracted, "object_uri": uri},
                    "tags": ["upload"],
                    "confirmed_by_user": True,
                    "idempotency_key": f"upload-{uuid.uuid4().hex}",
                },
            )
        except Exception:
            object_store.delete(object_uri=uri)
            raise

    def get_memory_document(self, *, user_id: str, memory_id: str) -> tuple[str, bytes]:
        result = self.call_memory(
            user_id=user_id,
            tool_name="get_memory",
            arguments={"memory_id": memory_id},
        )
        memory = result.get("memory")
        if not isinstance(memory, Mapping) or memory.get("status") != "ACTIVE":
            raise ResourceNotFoundError("uploaded document not found")
        value = memory.get("value") or {}
        if value.get("kind") != "uploaded_document":
            raise ResourceNotFoundError("uploaded document not found")
        document = value.get("content") or {}
        uri = document.get("object_uri")
        filename = document.get("filename")
        if not isinstance(uri, str) or not isinstance(filename, str):
            raise ResourceNotFoundError("uploaded document not found")
        user_prefix = (
            f"jmr-object:///uploads/{hashlib.sha256(user_id.encode()).hexdigest()}/"
        )
        if not uri.startswith(user_prefix):
            raise ResourceNotFoundError("uploaded document not found")
        try:
            return filename, self._require_object_store().get(object_uri=uri)
        except (KeyError, ValueError) as exc:
            raise ResourceNotFoundError("uploaded document not found") from exc

    def _call_memory_once(
        self,
        *,
        user_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            with open_postgres_store(environment=self._environment) as store:
                server = create_memory_server(
                    PostgresMemoryStore(store), user_id=user_id
                )
                result = server.call_tool(tool_name, arguments)
        except KeyError as exc:
            raise ResourceNotFoundError("memory not found") from exc
        except ValueError as exc:
            raise OperationConflictError(str(exc)) from exc
        if not isinstance(result, Mapping):
            raise RuntimeError("memory service returned an invalid response")
        return dict(result)

    def plan_case_deletion(self, *, user_id: str, case_id: str) -> Mapping[str, Any]:
        return self._delete_case(user_id=user_id, case_id=case_id, plan_token=None)

    def confirm_case_deletion(
        self, *, user_id: str, case_id: str, plan_token: str
    ) -> Mapping[str, Any]:
        return self._delete_case(
            user_id=user_id, case_id=case_id, plan_token=plan_token
        )

    def _delete_case(
        self, *, user_id: str, case_id: str, plan_token: str | None
    ) -> Mapping[str, Any]:
        database_url = self._environment.get("DATABASE_URL", "").strip()
        if not database_url:
            raise ServiceUnavailableError("database is not configured")
        import psycopg

        with self._exclusive_guard("case", case_id):
            try:
                with psycopg.connect(database_url) as connection:
                    with open_postgres_checkpointer(
                        database_url, environment=self._environment
                    ) as checkpointer:
                        object_store = self._require_object_store()
                        if plan_token is None:
                            report = delete_case(
                                connection,
                                checkpointer,
                                object_store,
                                user_id=user_id,
                                case_id=case_id,
                                confirm=False,
                            )
                            fingerprint = _deletion_fingerprint(
                                connection,
                                checkpointer,
                                user_id=user_id,
                                case_id=case_id,
                                object_uris=report["object_uris"],
                            )
                            token = uuid.uuid4().hex
                            connection.execute(
                                """
                                DELETE FROM jmr.idempotency_keys
                                WHERE user_id=%s AND case_id=%s
                                  AND operation='http_deletion_plan'
                                """,
                                (user_id, case_id),
                            )
                            connection.execute(
                                """
                                INSERT INTO jmr.idempotency_keys
                                    (user_id, case_id, operation,
                                     idempotency_key, payload_hash, result)
                                VALUES (%s, %s, 'http_deletion_plan', %s, %s, %s)
                                """,
                                (
                                    user_id,
                                    case_id,
                                    token,
                                    fingerprint,
                                    Jsonb(
                                        {
                                            "fingerprint": fingerprint,
                                            "object_uris": report["object_uris"],
                                            "expires_at": time.time()
                                            + _DELETION_PLAN_TTL_SECONDS,
                                        }
                                    ),
                                ),
                            )
                            connection.commit()
                            return {**report, "plan_token": token}

                        plan_row = connection.execute(
                            """
                            SELECT result FROM jmr.idempotency_keys
                            WHERE user_id=%s AND case_id=%s
                              AND operation='http_deletion_plan'
                              AND idempotency_key=%s
                            """,
                            (user_id, case_id, plan_token),
                        ).fetchone()
                        if plan_row is None or not isinstance(plan_row[0], Mapping):
                            raise OperationConflictError("deletion plan is not valid")
                        plan = dict(plan_row[0])
                        tombstone = connection.execute(
                            "SELECT status FROM jmr.case_deletions WHERE case_id=%s "
                            "AND user_id=%s",
                            (case_id, user_id),
                        ).fetchone()
                        if tombstone is None:
                            if float(plan.get("expires_at", 0)) < time.time():
                                raise OperationConflictError("deletion plan expired")
                            preview = delete_case(
                                connection,
                                checkpointer,
                                object_store,
                                user_id=user_id,
                                case_id=case_id,
                                confirm=False,
                            )
                            actual_fingerprint = _deletion_fingerprint(
                                connection,
                                checkpointer,
                                user_id=user_id,
                                case_id=case_id,
                                object_uris=preview["object_uris"],
                            )
                            if actual_fingerprint != plan.get("fingerprint"):
                                raise OperationConflictError("deletion plan is stale")
                        report = delete_case(
                            connection,
                            checkpointer,
                            object_store,
                            user_id=user_id,
                            case_id=case_id,
                            confirm=True,
                            expected_object_uris=plan.get("object_uris", []),
                        )
                        return {**report, "plan_token": plan_token}
            except KeyError as exc:
                raise ResourceNotFoundError("case not found") from exc
            except CaseOwnershipError as exc:
                raise ResourceNotFoundError("case not found") from exc
            except DeletionPlanStaleError as exc:
                raise OperationConflictError("deletion plan is stale") from exc
            except CaseSchemaNotReadyError as exc:
                raise ServiceUnavailableError("business schema is not ready") from exc
            except psycopg.Error as exc:
                raise ServiceUnavailableError("database operation failed") from exc

    @contextmanager
    def _open_runtime(self, user_id: str) -> Iterator[_RuntimeSession]:
        missing = [
            name
            for name in ("ANTHROPIC_API_KEY", "MODEL_ID")
            if not self._environment.get(name, "").strip()
        ]
        if missing:
            raise ServiceUnavailableError("model configuration is incomplete")
        self.startup()
        with ExitStack() as stack:
            checkpointer = stack.enter_context(
                open_postgres_checkpointer(environment=self._environment)
            )
            repository = stack.enter_context(
                open_postgres_repository(environment=self._environment)
            )
            store = stack.enter_context(
                open_postgres_store(environment=self._environment)
            )
            retry_policy = BoundedRetryPolicy()
            model_registry = create_anthropic_model_registry(
                environment=self._environment,
                retry_policy=retry_policy,
            )
            close_model = getattr(model_registry.model.client, "close", None)
            if callable(close_model):
                stack.callback(close_model)
            runtime = JMRRuntimeContext(
                principal_user_id=user_id,
                model_registry=model_registry,
                mcp_gateway=NodeScopedMCPGateway(
                    MCPServerRegistry(
                        {
                            "scholar": create_scholar_server,
                            "kaken": create_kaken_server,
                        }
                    ),
                    retry_policy=retry_policy,
                    audit_writer=repository.record_audit_event,
                ),
                case_repository=repository,
                memory_store=PostgresMemoryStore(store),
                postgres_store=store,
                object_store=self._require_object_store(),
                retry_policy=retry_policy,
                security_policy=PrincipalSecurityPolicy(),
                event_sink=self._event_sink,
                metadata={"transport": "http"},
            )
            yield _RuntimeSession(
                graph=compile_graph(checkpointer),
                runtime=runtime,
                repository=repository,
            )

    @contextmanager
    def _open_reader(self) -> Iterator[tuple[Any, Any]]:
        with ExitStack() as stack:
            checkpointer = stack.enter_context(
                open_postgres_checkpointer(environment=self._environment)
            )
            repository = stack.enter_context(
                open_postgres_repository(environment=self._environment)
            )
            yield compile_graph(checkpointer), repository

    def _owned_snapshot(
        self, session: _RuntimeSession, user_id: str, case_id: str
    ) -> Any:
        case = session.repository.get_case(user_id=user_id, case_id=case_id)
        if case is None:
            raise ResourceNotFoundError("case not found")
        if _case_has_deletion_job(session.repository, case_id):
            raise OperationConflictError("case deletion is in progress")
        snapshot = session.graph.get_state(_config(user_id, case_id))
        if not snapshot.values:
            raise ResourceNotFoundError("case checkpoint not found")
        return snapshot

    def _require_object_store(self) -> FileObjectStore:
        self.startup()
        if self._object_store is None:  # pragma: no cover - defensive
            raise ServiceUnavailableError("object store is not initialized")
        return self._object_store

    @contextmanager
    def _exclusive_guard(self, scope: str, identifier: str) -> Iterator[None]:
        key = (scope, identifier)
        with self._case_locks_guard:
            lock = self._exclusive_locks.setdefault(key, threading.Lock())
            if not lock.acquire(blocking=False):
                raise CaseBusyError("another operation is active for this resource")
        connection: Any | None = None
        try:
            database_url = self._environment.get("DATABASE_URL", "").strip()
            if database_url:
                import psycopg

                lock_id = int.from_bytes(
                    hashlib.sha256(f"jmr.api:{scope}:{identifier}".encode()).digest()[
                        :8
                    ],
                    byteorder="big",
                    signed=True,
                )
                try:
                    connection = psycopg.connect(
                        database_url, autocommit=True, connect_timeout=5
                    )
                    acquired = connection.execute(
                        "SELECT pg_try_advisory_lock(%s)", (lock_id,)
                    ).fetchone()
                except psycopg.Error as exc:
                    raise ServiceUnavailableError("database lock unavailable") from exc
                if not acquired or not acquired[0]:
                    raise CaseBusyError("another operation is active for this resource")
            yield
        finally:
            try:
                if connection is not None:
                    connection.close()
            finally:
                with self._case_locks_guard:
                    lock.release()
                    self._exclusive_locks.pop(key, None)

    @contextmanager
    def _turn_slot(self) -> Iterator[None]:
        if not self._turn_slots.acquire(blocking=False):
            raise ServiceUnavailableError("agent execution capacity is exhausted")
        try:
            yield
        finally:
            self._turn_slots.release()


def _config(user_id: str, case_id: str) -> dict[str, Any]:
    return {
        "configurable": {
            "user_id": user_id,
            "thread_id": case_id,
        }
    }


def _json_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _deletion_fingerprint(
    connection: Any,
    checkpointer: Any,
    *,
    user_id: str,
    case_id: str,
    object_uris: list[str],
) -> str:
    row = connection.execute(
        "SELECT version FROM jmr.research_cases WHERE case_id=%s AND user_id=%s",
        (case_id, user_id),
    ).fetchone()
    snapshot = compile_graph(checkpointer).get_state(_config(user_id, case_id))
    checkpoint_id = (
        getattr(snapshot, "config", {}).get("configurable", {}).get("checkpoint_id")
    )
    return _json_hash(
        {
            "user_id": user_id,
            "case_id": case_id,
            "case_version": row[0] if row is not None else None,
            "checkpoint_id": checkpoint_id,
            "object_uris": sorted(object_uris),
        }
    )


def _case_has_deletion_job(repository: Any, case_id: str) -> bool:
    connection = repository.connection
    row = connection.execute(
        "SELECT 1 FROM jmr.case_deletions WHERE case_id=%s", (case_id,)
    ).fetchone()
    connection.commit()
    return row is not None


def _project_values(
    values: Mapping[str, Any],
    *,
    pending: Mapping[str, Any] | None,
    interrupt_token: str | None = None,
    retry_available: bool = False,
    failure: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    for field in (
        "research_plan_id",
        "verified_evidence_bundle_id",
        "direction_batch_id",
        "current_outreach_iteration_id",
    ):
        value = values.get(field)
        if isinstance(value, str) and value:
            artifacts[field] = value
    warnings = [str(item) for item in values.get("warnings", [])]
    application = values.get("application_validation") or {}
    if isinstance(application, Mapping) and application.get("ready"):
        warnings = [
            item
            for item in warnings
            if item not in {"缺少大学名称", "缺少研究科或学院名称", "缺少教授姓名"}
        ]
        current_values = application.get("current_values") or {}
        if (
            isinstance(current_values, Mapping)
            and current_values.get("date_source") == "user"
        ):
            warnings = [
                item
                for item in warnings
                if item != "检索时间范围采用系统默认的过去十二个月"
            ]
    candidate_options = pending.get("options") if isinstance(pending, Mapping) else None
    has_official_candidate = isinstance(candidate_options, list) and any(
        isinstance(item, Mapping) and isinstance(item.get("url"), str)
        for item in candidate_options
    )
    user_reported_no_site = any(
        item.startswith("用户报告没有研究室官网") for item in warnings
    )
    if has_official_candidate or user_reported_no_site:
        warnings = [
            item
            for item in warnings
            if not item.startswith(
                (
                    "SerpApi 未发现",
                    "Web Search 未发现",
                    "当前查询没有可供抓取的官网候选",
                )
            )
        ]
    return {
        "user_id": str(values.get("user_id", "")),
        "case_id": str(values.get("case_id", "")),
        "workflow_stage": str(values.get("workflow_stage", "")),
        "run_status": str(values.get("run_status", "")),
        "pending_interrupt": dict(pending) if pending is not None else None,
        "interrupt_token": interrupt_token,
        "retry_available": retry_available,
        "failure": dict(failure) if failure is not None else None,
        "warnings": warnings,
        "errors": [str(item) for item in values.get("errors", [])],
        "result_available": (
            values.get("run_status") == RunStatus.COMPLETED.value
            and isinstance(values.get("current_outreach_iteration_id"), str)
        ),
        "selected_paper_evidence_ids": list(
            values.get("selected_paper_evidence_ids") or []
        ),
        "selected_papers": list(values.get("selected_papers") or []),
        "artifacts": artifacts,
    }


def _project_snapshot(snapshot: Any) -> dict[str, Any]:
    identity = _snapshot_interrupt_identity(snapshot)
    return _project_values(
        snapshot.values,
        pending=_snapshot_interrupt(snapshot),
        interrupt_token=identity[2] if identity is not None else None,
        retry_available=_snapshot_retry_available(snapshot),
        failure=_snapshot_failure(snapshot),
    )


def _snapshot_retry_available(snapshot: Any) -> bool:
    values = getattr(snapshot, "values", {}) or {}
    if values.get("run_status") in {
        RunStatus.COMPLETED.value,
        RunStatus.CANCELLED.value,
    }:
        return False
    return any(
        getattr(task, "error", None) is not None and not getattr(task, "interrupts", ())
        for task in getattr(snapshot, "tasks", ())
    )


def _snapshot_failure(snapshot: Any) -> dict[str, str] | None:
    if not _snapshot_retry_available(snapshot):
        return None
    for task in getattr(snapshot, "tasks", ()):
        error = getattr(task, "error", None)
        if error is None:
            continue
        marker = str(error)
        if "structured output validation" in marker:
            message = "模型输出格式校验失败，请重试该阶段。"
            code = "MODEL_OUTPUT_INVALID"
        elif "APITimeoutError" in marker or "TimeoutError" in marker:
            message = "模型或网络请求超时，请检查模型服务后重试。"
            code = "MODEL_TIMEOUT"
        elif "APIConnectionError" in marker:
            message = "无法连接模型服务，请检查接口配置后重试。"
            code = "MODEL_CONNECTION_FAILED"
        else:
            message = "当前阶段执行失败；可重试，详情见后端控制台。"
            code = "STAGE_FAILED"
        return {
            "node": str(getattr(task, "name", "")),
            "code": code,
            "message": message,
        }
    return None


def _snapshot_interrupt_identity(snapshot: Any) -> tuple[str, str, str] | None:
    interrupts = [
        interrupt
        for task in getattr(snapshot, "tasks", ())
        for interrupt in getattr(task, "interrupts", ())
    ]
    if not interrupts:
        return None
    if len(interrupts) != 1:
        raise OperationConflictError("case has multiple pending interrupts")
    interrupt_id = getattr(interrupts[0], "id", None)
    checkpoint_id = (
        getattr(snapshot, "config", {}).get("configurable", {}).get("checkpoint_id")
    )
    if not isinstance(interrupt_id, str) or not isinstance(checkpoint_id, str):
        raise RuntimeError("pending interrupt is missing a stable identity")
    token = hashlib.sha256(f"{checkpoint_id}:{interrupt_id}".encode()).hexdigest()
    return interrupt_id, checkpoint_id, token


def _snapshot_interrupt(snapshot: Any) -> Mapping[str, Any] | None:
    for task in getattr(snapshot, "tasks", ()):
        interrupts = getattr(task, "interrupts", ())
        if interrupts:
            return _interrupt_value(interrupts[0])
    return None


def _interrupt_value(interrupt: Any) -> Mapping[str, Any] | None:
    value = getattr(interrupt, "value", interrupt)
    return dict(value) if isinstance(value, Mapping) else None
