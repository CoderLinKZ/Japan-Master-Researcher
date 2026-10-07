"""Report-first, resumable deletion of one user-owned Case."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from jmr.persistence import SCHEMA_VERSION, FileObjectStore, current_schema_version
from jmr.runtime import open_postgres_checkpointer

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CaseOwnershipError(PermissionError):
    """Raised only for a Case deletion ownership mismatch."""


class DeletionPlanStaleError(ValueError):
    """Raised when the approved object list changed before deletion."""


class CaseSchemaNotReadyError(RuntimeError):
    """Raised when the business schema has not reached this release."""


def delete_case(
    connection: Any,
    checkpointer: Any,
    object_store: Any,
    *,
    user_id: str,
    case_id: str,
    confirm: bool = False,
    expected_object_uris: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Delete exact Case resources; a durable tombstone supports retry.

    Use a dedicated connection and quiesce workers for this Case first. Audit
    events remain under their own retention policy.
    """

    user_id = _identifier(user_id, "user_id")
    case_id = _identifier(case_id, "case_id")
    if current_schema_version(connection) != SCHEMA_VERSION:
        raise CaseSchemaNotReadyError("business schema is not current")
    row = connection.execute(
        "SELECT user_id, object_uris, checkpoint_deleted, business_deleted, status "
        "FROM jmr.case_deletions WHERE case_id=%s",
        (case_id,),
    ).fetchone()
    if row is not None:
        job = _job(row)
        if job["user_id"] != user_id:
            raise CaseOwnershipError("Case deletion belongs to another user")
    else:
        owner = connection.execute(
            "SELECT user_id FROM jmr.research_cases WHERE case_id=%s", (case_id,)
        ).fetchone()
        if owner is None:
            raise KeyError(f"Case not found: {case_id}")
        if _field(owner, "user_id", 0) != user_id:
            raise CaseOwnershipError("Case belongs to another user")
        prefix = f"users/{quote(user_id, safe='')}/cases/{quote(case_id, safe='')}/"
        uris = list(object_store.iter_uris(prefix=prefix))
        job = {
            "user_id": user_id,
            "object_uris": uris,
            "checkpoint_deleted": False,
            "business_deleted": False,
            "status": "PENDING",
        }
    connection.commit()
    if not confirm:
        return _report(case_id, job, dry_run=True)

    if row is None and expected_object_uris is not None:
        if sorted(job["object_uris"]) != sorted(expected_object_uris):
            raise DeletionPlanStaleError("deletion plan is stale")

    if row is None:
        with connection.transaction():
            connection.execute(
                """
                INSERT INTO jmr.case_deletions
                    (case_id, user_id, object_uris)
                VALUES (%s, %s, %s)
                """,
                (case_id, user_id, Jsonb(job["object_uris"])),
            )
            _audit(connection, user_id, case_id, "case.deletion_requested")
        connection.commit()

    if not job["checkpoint_deleted"]:
        checkpointer.delete_thread(case_id)
        with connection.transaction():
            connection.execute(
                "UPDATE jmr.case_deletions SET checkpoint_deleted=true, "
                "updated_at=now() WHERE case_id=%s AND user_id=%s",
                (case_id, user_id),
            )
        connection.commit()
        job["checkpoint_deleted"] = True

    if not job["business_deleted"]:
        with connection.transaction():
            owner = connection.execute(
                "SELECT user_id FROM jmr.research_cases WHERE case_id=%s FOR UPDATE",
                (case_id,),
            ).fetchone()
            if owner is not None:
                if _field(owner, "user_id", 0) != user_id:
                    raise CaseOwnershipError("Case belongs to another user")
                connection.execute(
                    """
                    DELETE FROM jmr.direction_evidence_links
                    WHERE direction_id IN (
                        SELECT direction_id FROM jmr.research_directions
                        WHERE case_id=%s
                    )
                    """,
                    (case_id,),
                )
                connection.execute(
                    """
                    DELETE FROM jmr.evidence_sources
                    WHERE evidence_id IN (
                        SELECT evidence_id FROM jmr.evidence_records
                        WHERE case_id=%s AND user_id=%s
                    )
                    """,
                    (case_id, user_id),
                )
                # Several Case children also reference one another without
                # ON DELETE CASCADE. Remove those edges in dependency order
                # before deleting the owning Case.
                for table in (
                    "direction_selections",
                    "direction_batches",
                    "evidence_records",
                    "retrieval_runs",
                    "source_objects",
                    "outreach_iterations",
                    "verified_evidence_bundles",
                ):
                    connection.execute(
                        f"DELETE FROM jmr.{table} WHERE case_id=%s AND user_id=%s",
                        (case_id, user_id),
                    )
                connection.execute(
                    "DELETE FROM jmr.research_cases WHERE case_id=%s AND user_id=%s",
                    (case_id, user_id),
                )
            connection.execute(
                "DELETE FROM jmr.idempotency_keys WHERE case_id=%s AND user_id=%s "
                "AND operation <> 'http_deletion_plan'",
                (case_id, user_id),
            )
            connection.execute(
                "UPDATE jmr.case_deletions SET business_deleted=true, "
                "updated_at=now() WHERE case_id=%s AND user_id=%s",
                (case_id, user_id),
            )
        connection.commit()
        job["business_deleted"] = True

    for uri in job["object_uris"]:
        try:
            object_store.delete(object_uri=uri)
        except KeyError:
            pass  # Already removed by an earlier attempt.
    with connection.transaction():
        connection.execute(
            "UPDATE jmr.case_deletions SET status='COMPLETED', "
            "updated_at=now(), completed_at=now() "
            "WHERE case_id=%s AND user_id=%s",
            (case_id, user_id),
        )
        if job["status"] != "COMPLETED":
            _audit(connection, user_id, case_id, "case.deleted")
    connection.commit()
    job["status"] = "COMPLETED"
    return _report(case_id, job, dry_run=False)


def _field(row: Any, name: str, index: int) -> Any:
    return row[name] if isinstance(row, Mapping) else row[index]


def _job(row: Any) -> dict[str, Any]:
    return {
        "user_id": _field(row, "user_id", 0),
        "object_uris": list(_field(row, "object_uris", 1)),
        "checkpoint_deleted": bool(_field(row, "checkpoint_deleted", 2)),
        "business_deleted": bool(_field(row, "business_deleted", 3)),
        "status": _field(row, "status", 4),
    }


def _report(case_id: str, job: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "user_id": job["user_id"],
        "status": job["status"],
        "dry_run": dry_run,
        "object_count": len(job["object_uris"]),
        "object_uris": list(job["object_uris"]),
        "checkpoint_deleted": job["checkpoint_deleted"],
        "business_deleted": job["business_deleted"],
        "audit_retained": True,
    }


def _audit(connection: Any, user_id: str, case_id: str, event_type: str) -> None:
    connection.execute(
        """
        INSERT INTO jmr.audit_events
            (audit_event_id, user_id, case_id, event_type, payload)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (uuid4(), user_id, case_id, event_type, Jsonb({})),
    )


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 255:
        raise ValueError(f"{name} must be a non-empty identifier of at most 255 chars")
    return value.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is required")
    import psycopg

    object_root = Path(os.getenv("JMR_OBJECT_STORE_DIR", PROJECT_ROOT / ".jmr/objects"))
    with psycopg.connect(database_url) as connection:
        with open_postgres_checkpointer(database_url) as checkpointer:
            result = delete_case(
                connection,
                checkpointer,
                FileObjectStore(object_root),
                user_id=args.user_id,
                case_id=args.case_id,
                confirm=args.confirm,
            )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by operators
    raise SystemExit(main())
