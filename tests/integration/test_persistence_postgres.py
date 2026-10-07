"""Optional real PostgreSQL checks for the P2 business repository."""

from __future__ import annotations

import os
import sys
import unittest
import uuid
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.persistence import (  # noqa: E402
    IdempotencyConflictError,
    OwnershipError,
    open_postgres_repository,
)


@unittest.skipUnless(
    os.getenv("JMR_RUN_POSTGRES_TESTS") == "1",
    "set JMR_RUN_POSTGRES_TESTS=1 to use the local integration database",
)
class PostgresRepositoryIntegrationTests(unittest.TestCase):
    def test_business_records_survive_transactions_and_enforce_scope(self) -> None:
        database_url = os.environ["DATABASE_URL"]
        user_id = f"p2-user-{uuid.uuid4()}"
        case_id = f"p2-case-{uuid.uuid4()}"
        try:
            with open_postgres_repository(database_url, setup=True) as repository:
                created = repository.create_case(
                    user_id=user_id,
                    case_id=case_id,
                    schema_version=2,
                    idempotency_key="create-1",
                )
                self.assertEqual(
                    repository.create_case(
                        user_id=user_id,
                        case_id=case_id,
                        schema_version=2,
                        idempotency_key="create-1",
                    ),
                    created,
                )
                with self.assertRaises(IdempotencyConflictError):
                    repository.create_case(
                        user_id=user_id,
                        case_id=case_id,
                        schema_version=999,
                        idempotency_key="create-1",
                    )
                target = repository.update_target(
                    user_id=user_id,
                    case_id=case_id,
                    target={"professor_name": "Example"},
                    expected_version=1,
                    idempotency_key="target-1",
                )
                self.assertEqual(target["version"], 2)
                repository.record_stage_transition(
                    user_id=user_id,
                    case_id=case_id,
                    transition={
                        "from_stage": "VALIDATING_APPLICATION_INPUTS",
                        "to_stage": "RETRIEVING_RESEARCH_EVIDENCE",
                        "run_status": "READY",
                    },
                    idempotency_key="stage-1",
                )
                repository.record_audit_event(
                    user_id=user_id,
                    case_id=case_id,
                    event_type="mcp.tool_called",
                    payload={
                        "tool": "mcp__scholar__search_publications",
                        "status": "SUCCESS",
                    },
                )
                with self.assertRaises(OwnershipError):
                    repository.get_case(user_id="another-user", case_id=case_id)
            with open_postgres_repository(database_url) as reopened:
                self.assertEqual(
                    reopened.get_case(user_id=user_id, case_id=case_id)["version"],
                    3,
                )
                audit_count = reopened.connection.execute(
                    "SELECT count(*) FROM jmr.audit_events WHERE case_id=%s",
                    (case_id,),
                ).fetchone()[0]
                self.assertEqual(audit_count, 1)
        finally:
            with open_postgres_repository(database_url) as repository:
                with repository.connection.cursor() as cursor:
                    cursor.execute("DROP SCHEMA IF EXISTS jmr CASCADE")
                repository.connection.commit()


if __name__ == "__main__":
    unittest.main()
