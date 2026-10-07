"""Real PostgreSQL test for resumable Case deletion."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

from psycopg.types.json import Jsonb

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.operations.cases import delete_case  # noqa: E402
from jmr.persistence import FileObjectStore, open_postgres_repository  # noqa: E402
from jmr.runtime import open_postgres_checkpointer  # noqa: E402


@unittest.skipUnless(
    os.getenv("JMR_RUN_POSTGRES_TESTS") == "1" and os.getenv("DATABASE_URL"),
    "requires JMR_RUN_POSTGRES_TESTS=1 and DATABASE_URL",
)
class CaseDeletionPostgresTests(unittest.TestCase):
    def test_dry_run_owner_scope_and_idempotent_deletion(self) -> None:
        database_url = os.environ["DATABASE_URL"]
        suffix = uuid.uuid4().hex
        user_id = f"delete-user-{suffix}"
        case_id = f"delete-case-{suffix}"
        with tempfile.TemporaryDirectory() as temporary:
            object_store = FileObjectStore(Path(temporary))
            uri = object_store.put(
                object_key=f"users/{user_id}/cases/{case_id}/workflow/one.json",
                content=b"{}",
                content_type="application/json",
            )
            try:
                with open_postgres_repository(database_url, setup=True) as repo:
                    repo.create_case(
                        user_id=user_id,
                        case_id=case_id,
                        schema_version=3,
                        idempotency_key="create",
                    )
                    source_object_id = uuid.uuid4()
                    evidence_id = uuid.uuid4()
                    repo.connection.execute(
                        """
                        INSERT INTO jmr.source_objects
                            (source_object_id, user_id, case_id, object_uri,
                             sha256, mime_type, byte_size, fetched_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                        """,
                        (
                            source_object_id,
                            user_id,
                            case_id,
                            uri,
                            "a" * 64,
                            "application/json",
                            2,
                        ),
                    )
                    repo.connection.execute(
                        """
                        INSERT INTO jmr.evidence_records
                            (evidence_id, case_id, user_id, evidence_kind, record)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            evidence_id,
                            case_id,
                            user_id,
                            "publication",
                            Jsonb({"title": "test evidence"}),
                        ),
                    )
                    repo.connection.execute(
                        """
                        INSERT INTO jmr.evidence_sources
                            (evidence_id, source_object_id, source_url)
                        VALUES (%s, %s, %s)
                        """,
                        (evidence_id, source_object_id, "https://example.org/paper"),
                    )
                    repo.connection.commit()
                    with open_postgres_checkpointer(database_url, setup=True) as saver:
                        with self.assertRaises(PermissionError):
                            delete_case(
                                repo.connection,
                                saver,
                                object_store,
                                user_id="another-user",
                                case_id=case_id,
                            )
                        planned = delete_case(
                            repo.connection,
                            saver,
                            object_store,
                            user_id=user_id,
                            case_id=case_id,
                        )
                        self.assertTrue(planned["dry_run"])
                        self.assertEqual(planned["object_count"], 1)
                        self.assertEqual(object_store.get(object_uri=uri), b"{}")
                        completed = delete_case(
                            repo.connection,
                            saver,
                            object_store,
                            user_id=user_id,
                            case_id=case_id,
                            confirm=True,
                        )
                        self.assertEqual(completed["status"], "COMPLETED")
                        self.assertIsNone(
                            repo.get_case(user_id=user_id, case_id=case_id)
                        )
                        with self.assertRaises(KeyError):
                            object_store.get(object_uri=uri)
                        replay = delete_case(
                            repo.connection,
                            saver,
                            object_store,
                            user_id=user_id,
                            case_id=case_id,
                            confirm=True,
                        )
                        self.assertEqual(replay["status"], "COMPLETED")
                        audit_count = repo.connection.execute(
                            "SELECT count(*) FROM jmr.audit_events WHERE case_id=%s",
                            (case_id,),
                        ).fetchone()[0]
                        self.assertEqual(audit_count, 2)
            finally:
                with open_postgres_repository(database_url) as repo:
                    repo.connection.execute("DROP SCHEMA IF EXISTS jmr CASCADE")
                    repo.connection.commit()


if __name__ == "__main__":
    unittest.main()
