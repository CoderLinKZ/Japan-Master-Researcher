"""PostgreSQL and in-memory implementations of the JMR case repository.

The repository is deliberately a host-side component.  Agents receive only
small IDs and read-only projections; every write below checks ownership,
optimistic versions, and an idempotency key in one database transaction.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from psycopg.types.json import Jsonb

from jmr.domain import RunStatus, WorkflowStage
from jmr.runtime.ports import CaseRepository, SystemClock, UUIDGenerator

from .errors import (
    ConstraintError,
    IdempotencyConflictError,
    NotFoundError,
    OwnershipError,
    VersionConflictError,
)
from .schema import apply_schema

REPOSITORY_DATABASE_URL_ENV = "DATABASE_URL"


class RepositoryConfigurationError(RuntimeError):
    """Raised when a PostgreSQL business repository cannot be opened."""


@contextmanager
def open_postgres_repository(
    database_url: str | None = None,
    *,
    setup: bool = False,
    environment: Mapping[str, str] | None = None,
):
    """Open a repository connection with an explicit lifecycle."""

    selected_environment = os.environ if environment is None else environment
    selected_url = database_url or selected_environment.get(REPOSITORY_DATABASE_URL_ENV)
    if not isinstance(selected_url, str) or not selected_url.strip():
        raise RepositoryConfigurationError(f"{REPOSITORY_DATABASE_URL_ENV} is missing")
    try:
        import psycopg

        connection = psycopg.connect(selected_url.strip())
    except Exception as exc:  # pragma: no cover - exercised by deployment
        raise RepositoryConfigurationError(
            "Unable to connect to the PostgreSQL business database"
        ) from exc
    try:
        if setup:
            apply_schema(connection)
            connection.commit()
        yield PostgresCaseRepository(connection)
    finally:
        connection.close()


def _required(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if field_name in {"user_id", "case_id"} and len(normalized) > 255:
        raise ValueError(f"{field_name} cannot exceed 255 characters")
    return normalized


def _key(value: str) -> str:
    return _required(value, "idempotency_key")


def _unique_identifiers(values: Any, field_name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    normalized = [_required(value, field_name) for value in values]
    return list(dict.fromkeys(normalized))


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_record_payloads(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    bundle = summary.get("bundle")
    if not isinstance(bundle, Mapping):
        return []
    records = bundle.get("records")
    if not isinstance(records, list):
        return []
    return [dict(item) for item in records if isinstance(item, Mapping)]


def _canonical_source_evidence_ids(
    summary: Mapping[str, Any],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for item in _verified_record_payloads(summary):
        evidence = item.get("evidence")
        source_ids = item.get("source_evidence_ids")
        if not isinstance(evidence, Mapping) or not isinstance(source_ids, list):
            continue
        canonical_id = evidence.get("evidence_id")
        if not isinstance(canonical_id, str):
            continue
        result[canonical_id] = [
            source_id
            for source_id in source_ids
            if isinstance(source_id, str) and source_id
        ]
    return result


class PostgresCaseRepository(CaseRepository):
    """Transactional repository for the JMR business facts."""

    def __init__(
        self,
        connection: Any,
        *,
        clock: Any | None = None,
        id_generator: Any | None = None,
        initialize_schema: bool = False,
    ) -> None:
        self.connection = connection
        self.clock = clock or SystemClock()
        self.id_generator = id_generator or UUIDGenerator()
        # A repository built around one psycopg connection may be called by
        # parallel LangGraph Send workers.  psycopg connections cannot commit
        # sibling transactions concurrently, so serialize only the short DB
        # transaction while leaving network retrieval itself parallel.
        self._lock = threading.RLock()
        if initialize_schema:
            apply_schema(connection)

    @contextmanager
    def _transaction(self):
        with self._lock:
            with self.connection.transaction():
                yield

    def _new_id(self, prefix: str) -> str:
        generated = self.id_generator.new_id(prefix)
        if not isinstance(generated, str) or not generated.strip():
            raise TypeError("id_generator must return a non-empty string")
        return generated

    def record_audit_event(
        self,
        *,
        user_id: str,
        case_id: str | None,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Persist a small, caller-redacted audit event without business bodies."""

        user_id = _required(user_id, "user_id")
        event_type = _required(event_type, "event_type")
        if case_id is not None:
            case_id = _required(case_id, "case_id")
        if not isinstance(payload, Mapping):
            raise TypeError("audit payload must be a mapping")
        with self._transaction():
            if case_id is not None:
                self._require_case(user_id=user_id, case_id=case_id)
            self.connection.execute(
                "INSERT INTO jmr.users(user_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (user_id,),
            )
            self.connection.execute(
                """
                INSERT INTO jmr.audit_events
                    (audit_event_id, user_id, case_id, event_type, payload)
                VALUES (%s::uuid, %s, %s, %s, %s)
                """,
                (
                    _as_uuid(self._new_id("audit")),
                    user_id,
                    case_id,
                    event_type,
                    Jsonb(dict(payload)),
                ),
            )

    def save_user_message(
        self,
        *,
        user_id: str,
        case_id: str,
        content: str,
        idempotency_key: str,
        created_at: datetime | None = None,
    ) -> Mapping[str, Any]:
        """Keep the full user turn after checkpoint message compaction."""

        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        content = _required(content, "content")
        idempotency_key = _key(idempotency_key)
        if len(content) > 12_000:
            raise ValueError("message cannot exceed 12000 characters")
        submitted_at = created_at or datetime.now(UTC)
        with self._transaction():
            self._require_case(user_id=user_id, case_id=case_id)
            inserted = self.connection.execute(
                """
                INSERT INTO jmr.case_messages
                    (message_id, case_id, user_id, role, content,
                     idempotency_key, created_at)
                VALUES (%s::uuid, %s, %s, 'user', %s, %s, %s)
                ON CONFLICT (user_id, case_id, idempotency_key) DO NOTHING
                RETURNING message_id, created_at
                """,
                (
                    _as_uuid(self._new_id("message")),
                    case_id,
                    user_id,
                    content,
                    idempotency_key,
                    submitted_at,
                ),
            ).fetchone()
            if inserted is None:
                existing = self.connection.execute(
                    """
                    SELECT message_id, content, created_at FROM jmr.case_messages
                    WHERE user_id=%s AND case_id=%s AND idempotency_key=%s
                    """,
                    (user_id, case_id, idempotency_key),
                ).fetchone()
                if existing is None or existing[1] != content:
                    raise IdempotencyConflictError(
                        "message idempotency key was used for another request"
                    )
                return {
                    "message_id": str(existing[0]),
                    "content": existing[1],
                    "created_at": existing[2].isoformat(),
                }
            return {
                "message_id": str(inserted[0]),
                "content": content,
                "created_at": inserted[1].isoformat(),
            }

    def _lookup_idempotency(
        self,
        *,
        user_id: str,
        case_id: str,
        operation: str,
        idempotency_key: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        payload_hash = _json_hash(payload)
        row = self.connection.execute(
            """
            SELECT payload_hash, result
            FROM jmr.idempotency_keys
            WHERE user_id=%s AND case_id=%s AND operation=%s
              AND idempotency_key=%s
            """,
            (user_id, case_id, operation, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if row[0] != payload_hash:
            raise IdempotencyConflictError(
                f"idempotency key already used for {operation} with another payload"
            )
        result = row[1]
        return dict(result) if isinstance(result, Mapping) else result

    def _save_idempotency(
        self,
        *,
        user_id: str,
        case_id: str,
        operation: str,
        idempotency_key: str,
        payload: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO jmr.idempotency_keys
                (user_id, case_id, operation, idempotency_key, payload_hash, result)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                case_id,
                operation,
                idempotency_key,
                _json_hash(payload),
                Jsonb(dict(result)),
            ),
        )

    def _require_case(
        self,
        *,
        user_id: str,
        case_id: str,
        for_update: bool = False,
    ) -> dict[str, Any]:
        lock_clause = " FOR UPDATE" if for_update else ""
        row = self.connection.execute(
            f"""
            SELECT case_id, user_id, case_context_id, schema_version,
                   workflow_stage, run_status, version, created_at, updated_at,
                   completed_at
            FROM jmr.research_cases WHERE case_id=%s
            {lock_clause}
            """,
            (case_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"case not found: {case_id}")
        if row[1] != user_id:
            raise OwnershipError("case does not belong to the requested user")
        return {
            "case_id": row[0],
            "user_id": row[1],
            "case_context_id": str(row[2]),
            "schema_version": row[3],
            "workflow_stage": row[4],
            "run_status": row[5],
            "version": row[6],
            "created_at": row[7].isoformat() if row[7] else None,
            "updated_at": row[8].isoformat() if row[8] else None,
            "completed_at": row[9].isoformat() if row[9] else None,
        }

    def create_case(
        self,
        *,
        user_id: str,
        case_id: str,
        schema_version: int,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise TypeError("schema_version must be an integer")
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "schema_version": schema_version,
        }
        with self._transaction():
            replay = self._lookup_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="create_case",
                idempotency_key=idempotency_key,
                payload=payload,
            )
            if replay is not None:
                return replay

            existing = self.connection.execute(
                "SELECT user_id, schema_version, version FROM jmr.research_cases "
                "WHERE case_id=%s",
                (case_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != user_id:
                    raise OwnershipError("case ID belongs to another user")
                raise ConstraintError("case_id already exists; use a new case ID")

            self.connection.execute(
                "INSERT INTO jmr.users(user_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (user_id,),
            )
            case_context_id = _as_uuid(self._new_id("ctx"))
            self.connection.execute(
                """
                INSERT INTO jmr.research_cases
                    (case_id, user_id, case_context_id, schema_version)
                VALUES (%s, %s, %s::uuid, %s)
                """,
                (case_id, user_id, case_context_id, schema_version),
            )
            result = {
                "case_id": case_id,
                "user_id": user_id,
                "case_context_id": case_context_id,
                "version": 1,
                "workflow_stage": WorkflowStage.VALIDATING_APPLICATION_INPUTS.value,
                "run_status": RunStatus.READY.value,
            }
            self._save_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="create_case",
                idempotency_key=idempotency_key,
                payload=payload,
                result=result,
            )
            return result

    def get_case(
        self,
        *,
        user_id: str,
        case_id: str,
    ) -> Mapping[str, Any] | None:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        with self._transaction():
            try:
                result = self._require_case(user_id=user_id, case_id=case_id)
            except NotFoundError:
                return None
            target = self.connection.execute(
                """
                SELECT target_id, target, version
                FROM jmr.case_targets WHERE case_id=%s AND user_id=%s
                """,
                (case_id, user_id),
            ).fetchone()
            if target is not None:
                result["target_id"] = str(target[0])
                result["target"] = dict(target[1])
                result["target_version"] = target[2]
            else:
                result["target_id"] = None
                result["target"] = None
                result["target_version"] = None
            return result

    def update_target(
        self,
        *,
        user_id: str,
        case_id: str,
        target: Mapping[str, Any],
        expected_version: int,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        if not isinstance(target, Mapping):
            raise TypeError("target must be a mapping")
        if not isinstance(expected_version, int) or isinstance(expected_version, bool):
            raise TypeError("expected_version must be an integer")
        target_payload = json.loads(json.dumps(dict(target), ensure_ascii=False))
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "target": target_payload,
            "expected_version": expected_version,
        }
        with self._transaction():
            replay = self._lookup_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="update_target",
                idempotency_key=idempotency_key,
                payload=payload,
            )
            if replay is not None:
                return replay
            current = self._require_case(
                user_id=user_id,
                case_id=case_id,
                for_update=True,
            )
            if current["version"] != expected_version:
                raise VersionConflictError(
                    f"case version {current['version']} does not match "
                    f"expected {expected_version}"
                )
            target_id = self.connection.execute(
                "SELECT target_id FROM jmr.case_targets WHERE case_id=%s",
                (case_id,),
            ).fetchone()
            if target_id is None:
                target_id_value = self._new_id("target")
                self.connection.execute(
                    """
                    INSERT INTO jmr.case_targets
                        (target_id, case_id, user_id, target, version)
                    VALUES (%s::uuid, %s, %s, %s, 1)
                    """,
                    (
                        _as_uuid(target_id_value),
                        case_id,
                        user_id,
                        Jsonb(target_payload),
                    ),
                )
                target_version = 1
            else:
                target_id_value = str(target_id[0])
                current_target_version = self.connection.execute(
                    "SELECT version FROM jmr.case_targets WHERE case_id=%s",
                    (case_id,),
                ).fetchone()[0]
                target_version = current_target_version + 1
                self.connection.execute(
                    """
                    UPDATE jmr.case_targets
                    SET target=%s, version=%s, updated_at=now()
                    WHERE case_id=%s AND user_id=%s
                    """,
                    (Jsonb(target_payload), target_version, case_id, user_id),
                )
            new_case_version = expected_version + 1
            self.connection.execute(
                "UPDATE jmr.research_cases SET version=%s, updated_at=now() "
                "WHERE case_id=%s AND user_id=%s",
                (new_case_version, case_id, user_id),
            )
            result = {
                "case_id": case_id,
                "user_id": user_id,
                "target_id": target_id_value,
                "target": target_payload,
                "target_version": target_version,
                "version": new_case_version,
            }
            self._save_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="update_target",
                idempotency_key=idempotency_key,
                payload=payload,
                result=result,
            )
            return result

    def record_stage_transition(
        self,
        *,
        user_id: str,
        case_id: str,
        transition: Mapping[str, Any],
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        if not isinstance(transition, Mapping):
            raise TypeError("transition must be a mapping")
        transition_payload = json.loads(json.dumps(dict(transition), default=str))
        target_stage = WorkflowStage(transition_payload.get("to_stage"))
        run_status = RunStatus(transition_payload.get("run_status", RunStatus.RUNNING))
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "transition": transition_payload,
        }
        with self._transaction():
            replay = self._lookup_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="record_stage_transition",
                idempotency_key=idempotency_key,
                payload=payload,
            )
            if replay is not None:
                return replay
            current = self._require_case(
                user_id=user_id,
                case_id=case_id,
                for_update=True,
            )
            from_stage = transition_payload.get("from_stage", current["workflow_stage"])
            if from_stage != current["workflow_stage"]:
                raise VersionConflictError("transition source stage is stale")
            history_id = self._new_id("stage")
            self.connection.execute(
                """
                INSERT INTO jmr.case_status_history
                    (history_id, case_id, user_id, from_stage, to_stage,
                     run_status, node_name, metadata, idempotency_key)
                VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    _as_uuid(history_id),
                    case_id,
                    user_id,
                    from_stage,
                    target_stage.value,
                    run_status.value,
                    transition_payload.get("node_name"),
                    Jsonb(dict(transition_payload.get("metadata", {}))),
                    idempotency_key,
                ),
            )
            new_version = current["version"] + 1
            self.connection.execute(
                """
                UPDATE jmr.research_cases
                SET workflow_stage=%s, run_status=%s, version=%s, updated_at=now(),
                    completed_at=CASE WHEN %s='COMPLETED' THEN now()
                                      ELSE completed_at END
                WHERE case_id=%s AND user_id=%s
                """,
                (
                    target_stage.value,
                    run_status.value,
                    new_version,
                    run_status.value,
                    case_id,
                    user_id,
                ),
            )
            result = {
                "history_id": history_id,
                "case_id": case_id,
                "from_stage": from_stage,
                "to_stage": target_stage.value,
                "run_status": run_status.value,
                "version": new_version,
            }
            self._save_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="record_stage_transition",
                idempotency_key=idempotency_key,
                payload=payload,
                result=result,
            )
            return result

    def save_research_plan(
        self,
        *,
        user_id: str,
        case_id: str,
        plan: Mapping[str, Any],
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        return self._save_json_record(
            table="research_plans",
            id_field="research_plan_id",
            payload_field="plan",
            user_id=user_id,
            case_id=case_id,
            payload=plan,
            idempotency_key=idempotency_key,
            prefix="plan",
        )

    def get_research_plan(
        self, *, user_id: str, case_id: str, research_plan_id: str
    ) -> Mapping[str, Any]:
        with self._transaction():
            self._require_case(user_id=user_id, case_id=case_id)
            row = self.connection.execute(
                """
                SELECT plan FROM jmr.research_plans
                WHERE research_plan_id=%s::uuid AND case_id=%s AND user_id=%s
                """,
                (_as_uuid(research_plan_id), case_id, user_id),
            ).fetchone()
            if row is None:
                raise NotFoundError("research plan not found for Case")
            return {"research_plan_id": research_plan_id, "plan": dict(row[0])}

    def save_retrieval_run(
        self,
        *,
        user_id: str,
        case_id: str,
        worker_kind: str,
        status: str,
        result_summary: Mapping[str, Any] | None = None,
        source_object: Mapping[str, Any] | None = None,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        worker_kind = _required(worker_kind, "worker_kind")
        status = _required(status, "status")
        idempotency_key = _key(idempotency_key)
        summary = dict(result_summary or {})
        source_metadata = dict(source_object) if source_object is not None else None
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "worker_kind": worker_kind,
            "status": status,
            "result_summary": summary,
            "source_object": source_metadata,
        }
        with self._transaction():
            replay = self._lookup_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_retrieval_run",
                idempotency_key=idempotency_key,
                payload=payload,
            )
            if replay is not None:
                return replay
            self._require_case(user_id=user_id, case_id=case_id)
            run_id = self._new_id("retrieval")
            self.connection.execute(
                """
                INSERT INTO jmr.retrieval_runs
                    (retrieval_run_id, case_id, user_id, worker_kind, status,
                     result_summary, idempotency_key)
                VALUES (%s::uuid, %s, %s, %s, %s, %s, %s)
                """,
                (
                    _as_uuid(run_id),
                    case_id,
                    user_id,
                    worker_kind,
                    status,
                    Jsonb(summary),
                    idempotency_key,
                ),
            )
            source_object_id = None
            if source_metadata is not None:
                for field in ("object_uri", "sha256", "mime_type", "byte_size"):
                    if field not in source_metadata:
                        raise ValueError(f"source_object.{field} is required")
                source_object_id = _as_uuid(self._new_id("source_object"))
                fetched_at = source_metadata.get("fetched_at")
                if not isinstance(fetched_at, str):
                    raise ValueError("source_object.fetched_at is required")
                row = self.connection.execute(
                    """
                    INSERT INTO jmr.source_objects
                        (source_object_id, user_id, case_id, object_uri, sha256,
                         mime_type, byte_size, fetched_at, metadata)
                    VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s::timestamptz, %s)
                    ON CONFLICT (case_id, sha256) DO UPDATE
                    SET object_uri = EXCLUDED.object_uri
                    RETURNING source_object_id
                    """,
                    (
                        source_object_id,
                        user_id,
                        case_id,
                        source_metadata["object_uri"],
                        source_metadata["sha256"],
                        source_metadata["mime_type"],
                        source_metadata["byte_size"],
                        fetched_at,
                        Jsonb(dict(source_metadata.get("metadata") or {})),
                    ),
                ).fetchone()
                source_object_id = row[0]
            for record in summary.get("records", []):
                if not isinstance(record, Mapping):
                    continue
                external_id = record.get("evidence_id") or record.get("id")
                if not isinstance(external_id, str) or not external_id.strip():
                    continue
                evidence_kind = str(record.get("evidence_type") or worker_kind)
                verification_status = str(
                    record.get("verification_status", "UNVERIFIED")
                ).upper()
                if verification_status not in {
                    "UNVERIFIED",
                    "PARTIAL",
                    "VERIFIED",
                    "REJECTED",
                }:
                    verification_status = "UNVERIFIED"
                evidence_id = _as_uuid(self._new_id("evidence"))
                row = self.connection.execute(
                    """
                    INSERT INTO jmr.evidence_records
                        (evidence_id, case_id, user_id, retrieval_run_id,
                         evidence_kind, external_id, title, record,
                         verification_status)
                    VALUES (%s::uuid, %s, %s, %s::uuid, %s, %s, %s, %s, %s)
                    ON CONFLICT (case_id, evidence_kind, external_id)
                    DO UPDATE SET record=EXCLUDED.record,
                                  verification_status=EXCLUDED.verification_status
                    RETURNING evidence_id
                    """,
                    (
                        evidence_id,
                        case_id,
                        user_id,
                        _as_uuid(run_id),
                        evidence_kind,
                        external_id.strip(),
                        record.get("title"),
                        Jsonb(dict(record)),
                        verification_status,
                    ),
                ).fetchone()
                source_url = record.get("source_url")
                if source_object_id is not None and isinstance(source_url, str):
                    self.connection.execute(
                        """
                        INSERT INTO jmr.evidence_sources
                            (evidence_id, source_object_id, source_url,
                             source_type, citation)
                        VALUES (%s::uuid, %s::uuid, %s, %s, %s)
                        ON CONFLICT (evidence_id, source_url) DO UPDATE
                        SET source_object_id = EXCLUDED.source_object_id
                        """,
                        (
                            row[0],
                            source_object_id,
                            source_url,
                            evidence_kind,
                            Jsonb({"source_name": record.get("source_name")}),
                        ),
                    )
            result = {
                "retrieval_run_id": run_id,
                "worker_kind": worker_kind,
                "status": status,
                "result_summary": summary,
            }
            self._save_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_retrieval_run",
                idempotency_key=idempotency_key,
                payload=payload,
                result=result,
            )
            return result

    def save_verified_evidence(
        self,
        *,
        user_id: str,
        case_id: str,
        evidence_ids: list[str] | tuple[str, ...],
        summary: Mapping[str, Any],
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        ids = _unique_identifiers(evidence_ids, "evidence_ids")
        body = json.loads(json.dumps(dict(summary), ensure_ascii=False, default=str))
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "evidence_ids": ids,
            "summary": body,
        }
        with self._transaction():
            replay = self._lookup_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_verified_evidence",
                idempotency_key=idempotency_key,
                payload=payload,
            )
            if replay is not None:
                return replay
            self._require_case(user_id=user_id, case_id=case_id)
            for verified in _verified_record_payloads(body):
                status = str(
                    (verified.get("evidence") or {}).get(
                        "verification_status", "UNVERIFIED"
                    )
                ).upper()
                conflicts = verified.get("conflicts", [])
                for source_external_id in verified.get("source_evidence_ids", []):
                    evidence_row = self.connection.execute(
                        """
                        SELECT evidence_id FROM jmr.evidence_records
                        WHERE case_id=%s AND user_id=%s AND external_id=%s
                        ORDER BY created_at LIMIT 1
                        """,
                        (case_id, user_id, source_external_id),
                    ).fetchone()
                    if evidence_row is None:
                        continue
                    self.connection.execute(
                        """
                        UPDATE jmr.evidence_records
                        SET verification_status=%s
                        WHERE evidence_id=%s::uuid
                        """,
                        (status, evidence_row[0]),
                    )
                    self.connection.execute(
                        """
                        INSERT INTO jmr.evidence_verifications
                            (verification_id, evidence_id, case_id, user_id,
                             status, conflicts)
                        VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s)
                        """,
                        (
                            _as_uuid(self._new_id("verification")),
                            evidence_row[0],
                            case_id,
                            user_id,
                            status,
                            Jsonb(list(conflicts)),
                        ),
                    )
            bundle_id = _as_uuid(self._new_id("bundle"))
            self.connection.execute(
                """
                INSERT INTO jmr.verified_evidence_bundles
                    (bundle_id, case_id, user_id, evidence_ids, summary)
                VALUES (%s::uuid, %s, %s, %s, %s)
                """,
                (bundle_id, case_id, user_id, Jsonb(ids), Jsonb(body)),
            )
            result = {
                "verified_evidence_bundle_id": bundle_id,
                "evidence_ids": ids,
                "summary": body,
            }
            self._save_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_verified_evidence",
                idempotency_key=idempotency_key,
                payload=payload,
                result=result,
            )
            return result

    def save_case_memory_selection(
        self,
        *,
        user_id: str,
        case_id: str,
        memory_ids: list[str] | tuple[str, ...],
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        ids = _unique_identifiers(memory_ids, "memory_ids")
        payload = {"user_id": user_id, "case_id": case_id, "memory_ids": ids}
        with self._transaction():
            replay = self._lookup_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_case_memory_selection",
                idempotency_key=idempotency_key,
                payload=payload,
            )
            if replay is not None:
                return replay
            self._require_case(user_id=user_id, case_id=case_id)
            for memory_id in ids:
                self.connection.execute(
                    """
                    INSERT INTO jmr.case_memory_links(case_id, user_id, memory_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (case_id, memory_id) DO NOTHING
                    """,
                    (case_id, user_id, memory_id),
                )
            result = {"case_id": case_id, "memory_ids": ids}
            self._save_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_case_memory_selection",
                idempotency_key=idempotency_key,
                payload=payload,
                result=result,
            )
            self.connection.execute(
                """
                INSERT INTO jmr.audit_events
                    (audit_event_id, user_id, case_id, event_type, payload)
                VALUES (%s::uuid, %s, %s, %s, %s)
                """,
                (
                    _as_uuid(self._new_id("audit")),
                    user_id,
                    case_id,
                    "memory.selected",
                    Jsonb({"memory_ids": ids}),
                ),
            )
            return result

    def save_direction_batch(
        self,
        *,
        user_id: str,
        case_id: str,
        directions: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
        evidence_bundle_id: str | None,
        revision_round: int,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        if not isinstance(revision_round, int) or isinstance(revision_round, bool):
            raise TypeError("revision_round must be an integer")
        normalized = [
            json.loads(json.dumps(dict(item), ensure_ascii=False, default=str))
            for item in directions
        ]
        if not normalized:
            raise ValueError("directions cannot be empty")
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "directions": normalized,
            "evidence_bundle_id": evidence_bundle_id,
            "revision_round": revision_round,
        }
        with self._transaction():
            replay = self._lookup_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_direction_batch",
                idempotency_key=idempotency_key,
                payload=payload,
            )
            if replay is not None:
                return replay
            self._require_case(user_id=user_id, case_id=case_id)
            if evidence_bundle_id:
                bundle_row = self.connection.execute(
                    """
                    SELECT case_id, user_id
                    FROM jmr.verified_evidence_bundles
                    WHERE bundle_id=%s::uuid
                    """,
                    (_as_uuid(evidence_bundle_id),),
                ).fetchone()
                if bundle_row is None:
                    raise NotFoundError("verified evidence bundle not found")
                if bundle_row[0] != case_id or bundle_row[1] != user_id:
                    raise OwnershipError("evidence bundle belongs to another case")
                evidence_row = self.connection.execute(
                    """
                    SELECT evidence_ids, summary
                    FROM jmr.verified_evidence_bundles
                    WHERE bundle_id=%s::uuid
                    """,
                    (_as_uuid(evidence_bundle_id),),
                ).fetchone()
                allowed_evidence_ids = set(evidence_row[0])
                canonical_sources = _canonical_source_evidence_ids(
                    dict(evidence_row[1])
                )
                for direction in normalized:
                    citations = direction.get("evidence_ids", [])
                    if (
                        not isinstance(citations, list)
                        or not 1 <= len(citations) <= 3
                        or not set(citations).issubset(allowed_evidence_ids)
                    ):
                        raise ConstraintError(
                            "each direction must cite 1-3 evidence IDs from its bundle"
                        )
            batch_id = _as_uuid(self._new_id("direction-batch"))
            evidence_uuid = _as_uuid(evidence_bundle_id) if evidence_bundle_id else None
            self.connection.execute(
                """
                INSERT INTO jmr.direction_batches
                    (direction_batch_id, case_id, user_id, evidence_bundle_id,
                     directions, revision_round, idempotency_key)
                VALUES (%s::uuid, %s, %s, %s::uuid, %s, %s, %s)
                """,
                (
                    batch_id,
                    case_id,
                    user_id,
                    evidence_uuid,
                    Jsonb(normalized),
                    revision_round,
                    idempotency_key,
                ),
            )
            direction_ids: list[str] = []
            for ordinal, direction in enumerate(normalized, start=1):
                direction_id = _as_uuid(self._new_id("direction"))
                self.connection.execute(
                    """
                    INSERT INTO jmr.research_directions
                        (direction_id, direction_batch_id, case_id, user_id,
                         direction, ordinal)
                    VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s)
                    """,
                    (
                        direction_id,
                        batch_id,
                        case_id,
                        user_id,
                        Jsonb(direction),
                        ordinal,
                    ),
                )
                for canonical_id in direction.get("evidence_ids", []):
                    source_ids = canonical_sources.get(canonical_id, [canonical_id])
                    linked_evidence = None
                    for source_id in source_ids:
                        linked_evidence = self.connection.execute(
                            """
                            SELECT evidence_id FROM jmr.evidence_records
                            WHERE case_id=%s AND user_id=%s AND external_id=%s
                            ORDER BY created_at LIMIT 1
                            """,
                            (case_id, user_id, source_id),
                        ).fetchone()
                        if linked_evidence is not None:
                            break
                    if linked_evidence is not None:
                        self.connection.execute(
                            """
                            INSERT INTO jmr.direction_evidence_links
                                (direction_id, evidence_id)
                            VALUES (%s::uuid, %s::uuid)
                            ON CONFLICT DO NOTHING
                            """,
                            (direction_id, linked_evidence[0]),
                        )
                direction_ids.append(direction_id)
            result = {
                "direction_batch_id": batch_id,
                "direction_ids": direction_ids,
                "revision_round": revision_round,
            }
            self._save_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_direction_batch",
                idempotency_key=idempotency_key,
                payload=payload,
                result=result,
            )
            return result

    def save_direction_selection(
        self,
        *,
        user_id: str,
        case_id: str,
        direction_batch_id: str,
        selected_direction_ids: list[str] | tuple[str, ...],
        custom_direction: Mapping[str, Any] | None,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        direction_batch_id = _required(direction_batch_id, "direction_batch_id")
        idempotency_key = _key(idempotency_key)
        selected = _unique_identifiers(selected_direction_ids, "selected_direction_ids")
        if bool(selected) == bool(custom_direction):
            raise ConstraintError("select directions or provide a custom direction")
        if len(selected) > 2:
            raise ConstraintError("select at most two directions")
        custom = (
            json.loads(json.dumps(dict(custom_direction), ensure_ascii=False))
            if custom_direction is not None
            else None
        )
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "direction_batch_id": direction_batch_id,
            "selected_direction_ids": selected,
            "custom_direction": custom,
        }
        with self._transaction():
            replay = self._lookup_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_direction_selection",
                idempotency_key=idempotency_key,
                payload=payload,
            )
            if replay is not None:
                return replay
            self._require_case(user_id=user_id, case_id=case_id)
            batch_row = self.connection.execute(
                """
                SELECT case_id, user_id
                FROM jmr.direction_batches
                WHERE direction_batch_id=%s::uuid
                """,
                (_as_uuid(direction_batch_id),),
            ).fetchone()
            if batch_row is None:
                raise NotFoundError("direction batch not found")
            if batch_row[0] != case_id or batch_row[1] != user_id:
                raise OwnershipError("direction batch belongs to another case")
            if selected:
                rows = self.connection.execute(
                    """
                    SELECT direction_id FROM jmr.research_directions
                    WHERE direction_batch_id=%s::uuid
                    """,
                    (_as_uuid(direction_batch_id),),
                ).fetchall()
                allowed_ids = {str(row[0]) for row in rows}
                if not set(selected).issubset(allowed_ids):
                    raise ConstraintError(
                        "selected directions must belong to the current batch"
                    )
            selection_id = _as_uuid(self._new_id("selection"))
            self.connection.execute(
                """
                INSERT INTO jmr.direction_selections
                    (selection_id, case_id, user_id, direction_batch_id,
                     selected_direction_ids, custom_direction, idempotency_key)
                VALUES (%s::uuid, %s, %s, %s::uuid, %s, %s, %s)
                """,
                (
                    selection_id,
                    case_id,
                    user_id,
                    _as_uuid(direction_batch_id),
                    Jsonb(selected),
                    Jsonb(custom) if custom is not None else None,
                    idempotency_key,
                ),
            )
            result = {
                "selection_id": selection_id,
                "direction_batch_id": direction_batch_id,
                "selected_direction_ids": selected,
                "custom_direction": custom,
            }
            self._save_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_direction_selection",
                idempotency_key=idempotency_key,
                payload=payload,
                result=result,
            )
            return result

    def save_outreach_iteration(
        self,
        *,
        user_id: str,
        case_id: str,
        content: str,
        citation_ids: list[str] | tuple[str, ...],
        parent_iteration_id: str | None,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        content = _required(content, "content")
        idempotency_key = _key(idempotency_key)
        citations = _unique_identifiers(citation_ids, "citation_ids")
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "content": content,
            "citation_ids": citations,
            "parent_iteration_id": parent_iteration_id,
        }
        with self._transaction():
            replay = self._lookup_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_outreach_iteration",
                idempotency_key=idempotency_key,
                payload=payload,
            )
            if replay is not None:
                return replay
            self._require_case(
                user_id=user_id,
                case_id=case_id,
                for_update=True,
            )
            if parent_iteration_id:
                parent_row = self.connection.execute(
                    """
                    SELECT case_id, user_id
                    FROM jmr.outreach_iterations
                    WHERE iteration_id=%s::uuid
                    """,
                    (_as_uuid(parent_iteration_id),),
                ).fetchone()
                if parent_row is None:
                    raise NotFoundError("parent outreach iteration not found")
                if parent_row[0] != case_id or parent_row[1] != user_id:
                    raise OwnershipError("parent iteration belongs to another case")
            latest = self.connection.execute(
                "SELECT COALESCE(MAX(iteration_no), 0) FROM jmr.outreach_iterations "
                "WHERE case_id=%s AND user_id=%s",
                (case_id, user_id),
            ).fetchone()[0]
            iteration_id = _as_uuid(self._new_id("iteration"))
            self.connection.execute(
                """
                INSERT INTO jmr.outreach_iterations
                    (iteration_id, case_id, user_id, parent_iteration_id,
                     iteration_no, content, citation_ids, idempotency_key)
                VALUES (%s::uuid, %s, %s, %s::uuid, %s, %s, %s, %s)
                """,
                (
                    iteration_id,
                    case_id,
                    user_id,
                    _as_uuid(parent_iteration_id) if parent_iteration_id else None,
                    int(latest) + 1,
                    content,
                    Jsonb(citations),
                    idempotency_key,
                ),
            )
            result = {
                "iteration_id": iteration_id,
                "iteration_no": int(latest) + 1,
                "parent_iteration_id": parent_iteration_id,
                "citation_ids": citations,
            }
            self._save_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_outreach_iteration",
                idempotency_key=idempotency_key,
                payload=payload,
                result=result,
            )
            return result

    def save_review_result(
        self,
        *,
        user_id: str,
        case_id: str,
        iteration_id: str,
        reviewer_kind: str,
        status: str,
        result: Mapping[str, Any],
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        iteration_id = _required(iteration_id, "iteration_id")
        reviewer_kind = _required(reviewer_kind, "reviewer_kind")
        status = _required(status, "status")
        idempotency_key = _key(idempotency_key)
        body = json.loads(json.dumps(dict(result), ensure_ascii=False, default=str))
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "iteration_id": iteration_id,
            "reviewer_kind": reviewer_kind,
            "status": status,
            "result": body,
        }
        with self._transaction():
            replay = self._lookup_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_review_result",
                idempotency_key=idempotency_key,
                payload=payload,
            )
            if replay is not None:
                return replay
            self._require_case(user_id=user_id, case_id=case_id)
            iteration_row = self.connection.execute(
                """
                SELECT case_id, user_id
                FROM jmr.outreach_iterations
                WHERE iteration_id=%s::uuid
                """,
                (_as_uuid(iteration_id),),
            ).fetchone()
            if iteration_row is None:
                raise NotFoundError("outreach iteration not found")
            if iteration_row[0] != case_id or iteration_row[1] != user_id:
                raise OwnershipError("iteration belongs to another case")
            review_id = _as_uuid(self._new_id("review"))
            self.connection.execute(
                """
                INSERT INTO jmr.outreach_reviews
                    (review_id, iteration_id, case_id, user_id, reviewer_kind,
                     status, result)
                VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s, %s)
                """,
                (
                    review_id,
                    _as_uuid(iteration_id),
                    case_id,
                    user_id,
                    reviewer_kind,
                    status,
                    Jsonb(body),
                ),
            )
            result_payload = {"review_id": review_id, "status": status}
            self._save_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="save_review_result",
                idempotency_key=idempotency_key,
                payload=payload,
                result=result_payload,
            )
            return result_payload

    def complete_case(
        self,
        *,
        user_id: str,
        case_id: str,
        final_iteration_id: str,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        final_iteration_id = _required(final_iteration_id, "final_iteration_id")
        idempotency_key = _key(idempotency_key)
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "final_iteration_id": final_iteration_id,
        }
        with self._transaction():
            replay = self._lookup_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="complete_case",
                idempotency_key=idempotency_key,
                payload=payload,
            )
            if replay is not None:
                return replay
            current = self._require_case(
                user_id=user_id,
                case_id=case_id,
                for_update=True,
            )
            iteration_row = self.connection.execute(
                """
                SELECT case_id, user_id
                FROM jmr.outreach_iterations
                WHERE iteration_id=%s::uuid
                """,
                (_as_uuid(final_iteration_id),),
            ).fetchone()
            if iteration_row is None:
                raise NotFoundError("final outreach iteration not found")
            if iteration_row[0] != case_id or iteration_row[1] != user_id:
                raise OwnershipError("final iteration belongs to another case")
            new_version = current["version"] + 1
            self.connection.execute(
                """
                UPDATE jmr.research_cases
                SET workflow_stage=%s, run_status=%s, version=%s,
                    updated_at=now(), completed_at=now()
                WHERE case_id=%s AND user_id=%s
                """,
                (
                    WorkflowStage.COMPLETED.value,
                    RunStatus.COMPLETED.value,
                    new_version,
                    case_id,
                    user_id,
                ),
            )
            result = {
                "case_id": case_id,
                "final_iteration_id": final_iteration_id,
                "workflow_stage": WorkflowStage.COMPLETED.value,
                "run_status": RunStatus.COMPLETED.value,
                "version": new_version,
            }
            self._save_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation="complete_case",
                idempotency_key=idempotency_key,
                payload=payload,
                result=result,
            )
            return result

    def get_retrieval_runs(
        self,
        *,
        user_id: str,
        case_id: str,
        retrieval_run_ids: list[str] | tuple[str, ...],
    ) -> list[Mapping[str, Any]]:
        """Read persisted worker results in caller-supplied order."""

        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        ids = _unique_identifiers(retrieval_run_ids, "retrieval_run_ids")
        with self._transaction():
            self._require_case(user_id=user_id, case_id=case_id)
            results: list[Mapping[str, Any]] = []
            for run_id in ids:
                row = self.connection.execute(
                    """
                    SELECT retrieval_run_id, worker_kind, status, result_summary,
                           version
                    FROM jmr.retrieval_runs
                    WHERE retrieval_run_id=%s::uuid AND case_id=%s AND user_id=%s
                    """,
                    (_as_uuid(run_id), case_id, user_id),
                ).fetchone()
                if row is None:
                    raise NotFoundError(f"retrieval run not found: {run_id}")
                results.append(
                    {
                        "retrieval_run_id": str(row[0]),
                        "worker_kind": row[1],
                        "status": row[2],
                        "result_summary": dict(row[3]),
                        "version": row[4],
                    }
                )
            return results

    def get_verified_evidence_bundle(
        self,
        *,
        user_id: str,
        case_id: str,
        bundle_id: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        bundle_id = _required(bundle_id, "bundle_id")
        with self._transaction():
            self._require_case(user_id=user_id, case_id=case_id)
            row = self.connection.execute(
                """
                SELECT evidence_ids, summary
                FROM jmr.verified_evidence_bundles
                WHERE bundle_id=%s::uuid AND case_id=%s AND user_id=%s
                """,
                (_as_uuid(bundle_id), case_id, user_id),
            ).fetchone()
            if row is None:
                raise NotFoundError("verified evidence bundle not found")
            return {
                "verified_evidence_bundle_id": bundle_id,
                "evidence_ids": list(row[0]),
                "summary": dict(row[1]),
            }

    def get_direction_batch(
        self,
        *,
        user_id: str,
        case_id: str,
        direction_batch_id: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        direction_batch_id = _required(direction_batch_id, "direction_batch_id")
        with self._transaction():
            self._require_case(user_id=user_id, case_id=case_id)
            row = self.connection.execute(
                """
                SELECT directions, evidence_bundle_id, revision_round
                FROM jmr.direction_batches
                WHERE direction_batch_id=%s::uuid AND case_id=%s AND user_id=%s
                """,
                (_as_uuid(direction_batch_id), case_id, user_id),
            ).fetchone()
            if row is None:
                raise NotFoundError("direction batch not found")
            direction_rows = self.connection.execute(
                """
                SELECT direction_id, direction, ordinal
                FROM jmr.research_directions
                WHERE direction_batch_id=%s::uuid
                ORDER BY ordinal
                """,
                (_as_uuid(direction_batch_id),),
            ).fetchall()
            directions = []
            for direction_id, direction, ordinal in direction_rows:
                item = dict(direction)
                item["direction_id"] = str(direction_id)
                item["ordinal"] = ordinal
                directions.append(item)
            return {
                "direction_batch_id": direction_batch_id,
                "evidence_bundle_id": str(row[1]) if row[1] else None,
                "revision_round": row[2],
                "directions": directions,
            }

    def get_outreach_iteration(
        self,
        *,
        user_id: str,
        case_id: str,
        iteration_id: str,
    ) -> Mapping[str, Any]:
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        iteration_id = _required(iteration_id, "iteration_id")
        with self._transaction():
            self._require_case(user_id=user_id, case_id=case_id)
            row = self.connection.execute(
                """
                SELECT iteration_no, parent_iteration_id, content, language,
                       citation_ids
                FROM jmr.outreach_iterations
                WHERE iteration_id=%s::uuid AND case_id=%s AND user_id=%s
                """,
                (_as_uuid(iteration_id), case_id, user_id),
            ).fetchone()
            if row is None:
                raise NotFoundError("outreach iteration not found")
            return {
                "iteration_id": iteration_id,
                "iteration_no": row[0],
                "parent_iteration_id": str(row[1]) if row[1] else None,
                "content": row[2],
                "language": row[3],
                "citation_ids": list(row[4]),
            }

    def list_review_results(
        self,
        *,
        user_id: str,
        case_id: str,
        iteration_id: str,
    ) -> list[Mapping[str, Any]]:
        self.get_outreach_iteration(
            user_id=user_id, case_id=case_id, iteration_id=iteration_id
        )
        rows = self.connection.execute(
            """
            SELECT review_id, reviewer_kind, status, result
            FROM jmr.outreach_reviews
            WHERE iteration_id=%s::uuid AND case_id=%s AND user_id=%s
            ORDER BY created_at, review_id
            """,
            (_as_uuid(iteration_id), case_id, user_id),
        ).fetchall()
        return [
            {
                "review_id": str(row[0]),
                "reviewer_kind": row[1],
                "status": row[2],
                "result": dict(row[3]),
            }
            for row in rows
        ]

    def _save_json_record(
        self,
        *,
        table: str,
        id_field: str,
        payload_field: str,
        user_id: str,
        case_id: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
        prefix: str,
    ) -> Mapping[str, Any]:
        # This helper is intentionally restricted to static table/column names
        # used by this module; no caller-controlled SQL identifier is accepted.
        allowed = {
            ("research_plans", "research_plan_id", "plan", "plan"),
        }
        if (table, id_field, payload_field, prefix) not in allowed:
            raise ValueError("unsupported JSON repository record")
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        body = json.loads(json.dumps(dict(payload), ensure_ascii=False))
        request = {"user_id": user_id, "case_id": case_id, payload_field: body}
        with self._transaction():
            replay = self._lookup_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation=f"save_{prefix}",
                idempotency_key=idempotency_key,
                payload=request,
            )
            if replay is not None:
                return replay
            self._require_case(user_id=user_id, case_id=case_id)
            record_id = self._new_id(prefix)
            self.connection.execute(
                f"""
                INSERT INTO jmr.{table}
                    ({id_field}, case_id, user_id, {payload_field}, idempotency_key)
                VALUES (%s::uuid, %s, %s, %s, %s)
                """,
                (_as_uuid(record_id), case_id, user_id, Jsonb(body), idempotency_key),
            )
            result = {id_field: record_id, payload_field: body}
            self._save_idempotency(
                user_id=user_id,
                case_id=case_id,
                operation=f"save_{prefix}",
                idempotency_key=idempotency_key,
                payload=request,
                result=result,
            )
            return result


def _as_uuid(value: str) -> str:
    """Validate UUID-shaped repository IDs before handing them to PostgreSQL."""

    # UUIDGenerator emits hex IDs with a prefix.  Strip the prefix for UUID
    # columns while retaining the opaque prefixed value in API responses.
    candidate = value.rsplit("_", 1)[-1]
    if len(candidate) != 32:
        # Custom deterministic generators may return a real UUID directly.
        try:
            return str(uuid4() if not value else value)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("repository ID is not UUID-compatible") from exc
    return (
        f"{candidate[0:8]}-{candidate[8:12]}-{candidate[12:16]}-"
        f"{candidate[16:20]}-{candidate[20:]}"
    )


class InMemoryCaseRepository(CaseRepository):
    """Thread-safe fake with the same ownership/idempotency semantics."""

    def __init__(self, *, clock: Any | None = None, id_generator: Any | None = None):
        self.clock = clock or SystemClock()
        self.id_generator = id_generator or UUIDGenerator()
        self._lock = threading.RLock()
        self._cases: dict[str, dict[str, Any]] = {}
        self._idempotency: dict[
            tuple[str, str, str, str], tuple[str, dict[str, Any]]
        ] = {}
        self._history: list[dict[str, Any]] = []
        self._plans: dict[str, dict[str, Any]] = {}
        self._retrieval_runs: dict[str, dict[str, Any]] = {}
        self._bundles: dict[str, dict[str, Any]] = {}
        self._memory_links: dict[str, list[str]] = {}
        self._direction_batches: dict[str, dict[str, Any]] = {}
        self._selections: dict[str, dict[str, Any]] = {}
        self._iterations: dict[str, dict[str, Any]] = {}
        self._reviews: dict[str, dict[str, Any]] = {}
        self.audit_events: list[dict[str, Any]] = []

    def record_audit_event(
        self,
        *,
        user_id: str,
        case_id: str | None,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        user_id = _required(user_id, "user_id")
        if case_id is not None:
            self._case(user_id, _required(case_id, "case_id"))
        with self._lock:
            self.audit_events.append(
                {
                    "user_id": user_id,
                    "case_id": case_id,
                    "event_type": _required(event_type, "event_type"),
                    "payload": dict(payload),
                }
            )

    def _replay_or_none(
        self,
        user_id: str,
        case_id: str,
        operation: str,
        key: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        item = self._idempotency.get((user_id, case_id, operation, key))
        if item is None:
            return None
        digest = _json_hash(payload)
        if item[0] != digest:
            raise IdempotencyConflictError(
                f"idempotency key already used for {operation} with another payload"
            )
        return dict(item[1])

    def _save_replay(
        self,
        user_id: str,
        case_id: str,
        operation: str,
        key: str,
        payload: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        self._idempotency[(user_id, case_id, operation, key)] = (
            _json_hash(payload),
            dict(result),
        )

    def _case(self, user_id: str, case_id: str) -> dict[str, Any]:
        case = self._cases.get(case_id)
        if case is None:
            raise NotFoundError(f"case not found: {case_id}")
        if case["user_id"] != user_id:
            raise OwnershipError("case does not belong to the requested user")
        return case

    def create_case(self, *, user_id, case_id, schema_version, idempotency_key):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "schema_version": schema_version,
        }
        with self._lock:
            replay = self._replay_or_none(
                user_id, case_id, "create_case", idempotency_key, payload
            )
            if replay is not None:
                return replay
            if case_id in self._cases:
                if self._cases[case_id]["user_id"] != user_id:
                    raise OwnershipError("case ID belongs to another user")
                raise ConstraintError("case_id already exists; use a new case ID")
            result = {
                "case_id": case_id,
                "user_id": user_id,
                "case_context_id": self.id_generator.new_id("ctx"),
                "version": 1,
                "workflow_stage": WorkflowStage.VALIDATING_APPLICATION_INPUTS.value,
                "run_status": RunStatus.READY.value,
            }
            self._cases[case_id] = dict(
                result, schema_version=schema_version, target=None
            )
            self._save_replay(
                user_id, case_id, "create_case", idempotency_key, payload, result
            )
            return result

    def get_case(self, *, user_id, case_id):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        with self._lock:
            try:
                case = self._case(user_id, case_id)
            except NotFoundError:
                return None
            return dict(case)

    def update_target(
        self, *, user_id, case_id, target, expected_version, idempotency_key
    ):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "target": json.loads(json.dumps(dict(target), ensure_ascii=False)),
            "expected_version": expected_version,
        }
        with self._lock:
            replay = self._replay_or_none(
                user_id, case_id, "update_target", idempotency_key, payload
            )
            if replay is not None:
                return replay
            case = self._case(user_id, case_id)
            if case["version"] != expected_version:
                raise VersionConflictError(
                    "case version does not match expected version"
                )
            case["target"] = payload["target"]
            case["target_id"] = case.get("target_id") or self.id_generator.new_id(
                "target"
            )
            case["target_version"] = case.get("target_version", 0) + 1
            case["version"] += 1
            result = {
                "case_id": case_id,
                "user_id": user_id,
                "target_id": case["target_id"],
                "target": case["target"],
                "target_version": case["target_version"],
                "version": case["version"],
            }
            self._save_replay(
                user_id, case_id, "update_target", idempotency_key, payload, result
            )
            return result

    def record_stage_transition(self, *, user_id, case_id, transition, idempotency_key):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        body = json.loads(json.dumps(dict(transition), default=str))
        payload = {"user_id": user_id, "case_id": case_id, "transition": body}
        with self._lock:
            replay = self._replay_or_none(
                user_id, case_id, "record_stage_transition", idempotency_key, payload
            )
            if replay is not None:
                return replay
            case = self._case(user_id, case_id)
            from_stage = body.get("from_stage", case["workflow_stage"])
            if from_stage != case["workflow_stage"]:
                raise VersionConflictError("transition source stage is stale")
            to_stage = WorkflowStage(body["to_stage"]).value
            status = RunStatus(body.get("run_status", RunStatus.RUNNING)).value
            case["workflow_stage"] = to_stage
            case["run_status"] = status
            case["version"] += 1
            result = {
                "history_id": self.id_generator.new_id("stage"),
                "case_id": case_id,
                "from_stage": from_stage,
                "to_stage": to_stage,
                "run_status": status,
                "version": case["version"],
            }
            self._history.append(dict(result, metadata=body.get("metadata", {})))
            self._save_replay(
                user_id,
                case_id,
                "record_stage_transition",
                idempotency_key,
                payload,
                result,
            )
            return result

    def save_research_plan(self, *, user_id, case_id, plan, idempotency_key):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        body = json.loads(json.dumps(dict(plan), ensure_ascii=False))
        payload = {"user_id": user_id, "case_id": case_id, "plan": body}
        with self._lock:
            replay = self._replay_or_none(
                user_id, case_id, "save_plan", idempotency_key, payload
            )
            if replay is not None:
                return replay
            self._case(user_id, case_id)
            result = {
                "research_plan_id": self.id_generator.new_id("plan"),
                "plan": body,
            }
            self._plans[result["research_plan_id"]] = dict(
                result, case_id=case_id, user_id=user_id
            )
            self._save_replay(
                user_id, case_id, "save_plan", idempotency_key, payload, result
            )
            return result

    def get_research_plan(self, *, user_id, case_id, research_plan_id):
        with self._lock:
            self._case(user_id, case_id)
            item = self._plans.get(research_plan_id)
            if item is None or item["case_id"] != case_id or item["user_id"] != user_id:
                raise NotFoundError("research plan not found for Case")
            return dict(item)

    def save_retrieval_run(
        self,
        *,
        user_id,
        case_id,
        worker_kind,
        status,
        result_summary=None,
        source_object=None,
        idempotency_key,
    ):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        summary = dict(result_summary or {})
        source_metadata = dict(source_object) if source_object is not None else None
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "worker_kind": worker_kind,
            "status": status,
            "result_summary": summary,
            "source_object": source_metadata,
        }
        with self._lock:
            replay = self._replay_or_none(
                user_id, case_id, "save_retrieval_run", idempotency_key, payload
            )
            if replay is not None:
                return replay
            self._case(user_id, case_id)
            result = {
                "retrieval_run_id": self.id_generator.new_id("retrieval"),
                "worker_kind": worker_kind,
                "status": status,
                "result_summary": summary,
            }
            self._retrieval_runs[result["retrieval_run_id"]] = dict(
                result, case_id=case_id, user_id=user_id
            )
            self._save_replay(
                user_id, case_id, "save_retrieval_run", idempotency_key, payload, result
            )
            return result

    def save_verified_evidence(
        self,
        *,
        user_id,
        case_id,
        evidence_ids,
        summary,
        idempotency_key,
    ):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        ids = _unique_identifiers(evidence_ids, "evidence_ids")
        body = json.loads(json.dumps(dict(summary), ensure_ascii=False, default=str))
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "evidence_ids": ids,
            "summary": body,
        }
        with self._lock:
            replay = self._replay_or_none(
                user_id, case_id, "save_verified_evidence", idempotency_key, payload
            )
            if replay is not None:
                return replay
            self._case(user_id, case_id)
            bundle_id = self.id_generator.new_id("bundle")
            result = {
                "verified_evidence_bundle_id": bundle_id,
                "evidence_ids": ids,
                "summary": body,
            }
            self._bundles[bundle_id] = dict(result, user_id=user_id, case_id=case_id)
            self._save_replay(
                user_id,
                case_id,
                "save_verified_evidence",
                idempotency_key,
                payload,
                result,
            )
            return result

    def save_case_memory_selection(
        self,
        *,
        user_id,
        case_id,
        memory_ids,
        idempotency_key,
    ):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        ids = _unique_identifiers(memory_ids, "memory_ids")
        payload = {"user_id": user_id, "case_id": case_id, "memory_ids": ids}
        with self._lock:
            replay = self._replay_or_none(
                user_id,
                case_id,
                "save_case_memory_selection",
                idempotency_key,
                payload,
            )
            if replay is not None:
                return replay
            self._case(user_id, case_id)
            selected = list(dict.fromkeys(self._memory_links.get(case_id, []) + ids))
            self._memory_links[case_id] = selected
            result = {"case_id": case_id, "memory_ids": selected}
            self._save_replay(
                user_id,
                case_id,
                "save_case_memory_selection",
                idempotency_key,
                payload,
                result,
            )
            self.audit_events.append(
                {
                    "user_id": user_id,
                    "case_id": case_id,
                    "event_type": "memory.selected",
                    "payload": {"memory_ids": ids},
                }
            )
            return result

    def save_direction_batch(
        self,
        *,
        user_id,
        case_id,
        directions,
        evidence_bundle_id,
        revision_round,
        idempotency_key,
    ):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        normalized = [
            json.loads(json.dumps(dict(item), ensure_ascii=False, default=str))
            for item in directions
        ]
        if not normalized:
            raise ValueError("directions cannot be empty")
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "directions": normalized,
            "evidence_bundle_id": evidence_bundle_id,
            "revision_round": revision_round,
        }
        with self._lock:
            replay = self._replay_or_none(
                user_id,
                case_id,
                "save_direction_batch",
                idempotency_key,
                payload,
            )
            if replay is not None:
                return replay
            self._case(user_id, case_id)
            if evidence_bundle_id:
                bundle = self._bundles.get(evidence_bundle_id)
                if bundle is None:
                    raise NotFoundError("verified evidence bundle not found")
                if bundle["case_id"] != case_id or bundle["user_id"] != user_id:
                    raise OwnershipError("evidence bundle belongs to another case")
                allowed_evidence_ids = set(bundle["evidence_ids"])
                for direction in normalized:
                    citations = direction.get("evidence_ids", [])
                    if (
                        not isinstance(citations, list)
                        or not 1 <= len(citations) <= 3
                        or not set(citations).issubset(allowed_evidence_ids)
                    ):
                        raise ConstraintError(
                            "each direction must cite 1-3 evidence IDs from its bundle"
                        )
            batch_id = self.id_generator.new_id("direction-batch")
            direction_ids = [self.id_generator.new_id("direction") for _ in normalized]
            result = {
                "direction_batch_id": batch_id,
                "direction_ids": direction_ids,
                "revision_round": revision_round,
            }
            self._direction_batches[batch_id] = dict(
                result,
                user_id=user_id,
                case_id=case_id,
                directions=normalized,
            )
            self._save_replay(
                user_id,
                case_id,
                "save_direction_batch",
                idempotency_key,
                payload,
                result,
            )
            return result

    def save_direction_selection(
        self,
        *,
        user_id,
        case_id,
        direction_batch_id,
        selected_direction_ids,
        custom_direction,
        idempotency_key,
    ):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        direction_batch_id = _required(direction_batch_id, "direction_batch_id")
        idempotency_key = _key(idempotency_key)
        selected = _unique_identifiers(selected_direction_ids, "selected_direction_ids")
        if bool(selected) == bool(custom_direction):
            raise ConstraintError("select directions or provide a custom direction")
        if len(selected) > 2:
            raise ConstraintError("select at most two directions")
        custom = (
            json.loads(json.dumps(dict(custom_direction), ensure_ascii=False))
            if custom_direction is not None
            else None
        )
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "direction_batch_id": direction_batch_id,
            "selected_direction_ids": selected,
            "custom_direction": custom,
        }
        with self._lock:
            replay = self._replay_or_none(
                user_id,
                case_id,
                "save_direction_selection",
                idempotency_key,
                payload,
            )
            if replay is not None:
                return replay
            case = self._case(user_id, case_id)
            batch = self._direction_batches.get(direction_batch_id)
            if (
                batch is None
                or batch["case_id"] != case["case_id"]
                or batch["user_id"] != user_id
            ):
                raise NotFoundError("direction batch not found")
            if selected and not set(selected).issubset(batch["direction_ids"]):
                raise ConstraintError(
                    "selected directions must belong to the current batch"
                )
            result = {
                "selection_id": self.id_generator.new_id("selection"),
                "direction_batch_id": direction_batch_id,
                "selected_direction_ids": selected,
                "custom_direction": custom,
            }
            self._selections[result["selection_id"]] = dict(
                result, user_id=user_id, case_id=case_id
            )
            self._save_replay(
                user_id,
                case_id,
                "save_direction_selection",
                idempotency_key,
                payload,
                result,
            )
            return result

    def save_outreach_iteration(
        self,
        *,
        user_id,
        case_id,
        content,
        citation_ids,
        parent_iteration_id,
        idempotency_key,
    ):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        content = _required(content, "content")
        idempotency_key = _key(idempotency_key)
        citations = _unique_identifiers(citation_ids, "citation_ids")
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "content": content,
            "citation_ids": citations,
            "parent_iteration_id": parent_iteration_id,
        }
        with self._lock:
            replay = self._replay_or_none(
                user_id,
                case_id,
                "save_outreach_iteration",
                idempotency_key,
                payload,
            )
            if replay is not None:
                return replay
            self._case(user_id, case_id)
            if parent_iteration_id:
                parent = self._iterations.get(parent_iteration_id)
                if parent is None:
                    raise NotFoundError("parent outreach iteration not found")
                if parent["case_id"] != case_id or parent["user_id"] != user_id:
                    raise OwnershipError("parent iteration belongs to another case")
            iteration_no = 1 + max(
                (
                    item["iteration_no"]
                    for item in self._iterations.values()
                    if item["case_id"] == case_id
                ),
                default=0,
            )
            iteration_id = self.id_generator.new_id("iteration")
            result = {
                "iteration_id": iteration_id,
                "iteration_no": iteration_no,
                "parent_iteration_id": parent_iteration_id,
                "citation_ids": citations,
            }
            self._iterations[iteration_id] = dict(
                result, user_id=user_id, case_id=case_id, content=content
            )
            self._save_replay(
                user_id,
                case_id,
                "save_outreach_iteration",
                idempotency_key,
                payload,
                result,
            )
            return result

    def save_review_result(
        self,
        *,
        user_id,
        case_id,
        iteration_id,
        reviewer_kind,
        status,
        result,
        idempotency_key,
    ):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        idempotency_key = _key(idempotency_key)
        body = json.loads(json.dumps(dict(result), ensure_ascii=False, default=str))
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "iteration_id": iteration_id,
            "reviewer_kind": reviewer_kind,
            "status": status,
            "result": body,
        }
        with self._lock:
            replay = self._replay_or_none(
                user_id, case_id, "save_review_result", idempotency_key, payload
            )
            if replay is not None:
                return replay
            self._case(user_id, case_id)
            iteration = self._iterations.get(iteration_id)
            if iteration is None:
                raise NotFoundError("outreach iteration not found")
            if iteration["case_id"] != case_id or iteration["user_id"] != user_id:
                raise OwnershipError("iteration belongs to another case")
            review_id = self.id_generator.new_id("review")
            result_payload = {"review_id": review_id, "status": status}
            self._reviews[review_id] = dict(
                result_payload,
                user_id=user_id,
                case_id=case_id,
                iteration_id=iteration_id,
                reviewer_kind=reviewer_kind,
                result=body,
            )
            self._save_replay(
                user_id,
                case_id,
                "save_review_result",
                idempotency_key,
                payload,
                result_payload,
            )
            return result_payload

    def complete_case(
        self,
        *,
        user_id,
        case_id,
        final_iteration_id,
        idempotency_key,
    ):
        user_id = _required(user_id, "user_id")
        case_id = _required(case_id, "case_id")
        final_iteration_id = _required(final_iteration_id, "final_iteration_id")
        idempotency_key = _key(idempotency_key)
        payload = {
            "user_id": user_id,
            "case_id": case_id,
            "final_iteration_id": final_iteration_id,
        }
        with self._lock:
            replay = self._replay_or_none(
                user_id, case_id, "complete_case", idempotency_key, payload
            )
            if replay is not None:
                return replay
            case = self._case(user_id, case_id)
            iteration = self._iterations.get(final_iteration_id)
            if iteration is None:
                raise NotFoundError("final outreach iteration not found")
            if iteration["case_id"] != case_id or iteration["user_id"] != user_id:
                raise OwnershipError("final iteration belongs to another case")
            case["workflow_stage"] = WorkflowStage.COMPLETED.value
            case["run_status"] = RunStatus.COMPLETED.value
            case["version"] += 1
            result = {
                "case_id": case_id,
                "final_iteration_id": final_iteration_id,
                "workflow_stage": WorkflowStage.COMPLETED.value,
                "run_status": RunStatus.COMPLETED.value,
                "version": case["version"],
            }
            self._save_replay(
                user_id,
                case_id,
                "complete_case",
                idempotency_key,
                payload,
                result,
            )
            return result

    def get_retrieval_runs(self, *, user_id, case_id, retrieval_run_ids):
        with self._lock:
            self._case(user_id, case_id)
            results = []
            for run_id in _unique_identifiers(retrieval_run_ids, "retrieval_run_ids"):
                item = self._retrieval_runs.get(run_id)
                if item is None:
                    raise NotFoundError(f"retrieval run not found: {run_id}")
                if item["case_id"] != case_id or item["user_id"] != user_id:
                    raise OwnershipError("retrieval run belongs to another case")
                results.append(dict(item))
            return results

    def get_verified_evidence_bundle(self, *, user_id, case_id, bundle_id):
        with self._lock:
            self._case(user_id, case_id)
            item = self._bundles.get(bundle_id)
            if item is None:
                raise NotFoundError("verified evidence bundle not found")
            if item["case_id"] != case_id or item["user_id"] != user_id:
                raise OwnershipError("evidence bundle belongs to another case")
            return dict(item)

    def get_direction_batch(self, *, user_id, case_id, direction_batch_id):
        with self._lock:
            self._case(user_id, case_id)
            item = self._direction_batches.get(direction_batch_id)
            if item is None:
                raise NotFoundError("direction batch not found")
            if item["case_id"] != case_id or item["user_id"] != user_id:
                raise OwnershipError("direction batch belongs to another case")
            directions = []
            for ordinal, (direction_id, direction) in enumerate(
                zip(item["direction_ids"], item["directions"], strict=True),
                start=1,
            ):
                directions.append(
                    dict(direction, direction_id=direction_id, ordinal=ordinal)
                )
            return dict(item, directions=directions)

    def get_outreach_iteration(self, *, user_id, case_id, iteration_id):
        with self._lock:
            self._case(user_id, case_id)
            item = self._iterations.get(iteration_id)
            if item is None:
                raise NotFoundError("outreach iteration not found")
            if item["case_id"] != case_id or item["user_id"] != user_id:
                raise OwnershipError("iteration belongs to another case")
            return dict(item)

    def list_review_results(self, *, user_id, case_id, iteration_id):
        self.get_outreach_iteration(
            user_id=user_id, case_id=case_id, iteration_id=iteration_id
        )
        return [
            dict(item)
            for item in self._reviews.values()
            if item["case_id"] == case_id
            and item["user_id"] == user_id
            and item["iteration_id"] == iteration_id
        ]
