"""Offline tests for P7 observability, resilience, backup, and retention."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import psycopg

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.graph.nodes.common import compact_messages_for_model  # noqa: E402
from jmr.operations.backup import (  # noqa: E402
    BackupError,
    create_backup,
    restore_backup,
    verify_backup,
)
from jmr.operations.health import exit_code, liveness, main, readiness  # noqa: E402
from jmr.operations.retention import (  # noqa: E402
    RetentionConfirmationError,
    RetentionPolicy,
    cleanup_audit_events,
    cleanup_checkpoint,
    cleanup_expired_backups,
    cleanup_orphan_objects,
    expired_backup_directories,
    referenced_object_uris,
)
from jmr.persistence import (  # noqa: E402
    FileObjectStore,
    SchemaMigrationError,
    rollback_schema,
)
from jmr.runtime import (  # noqa: E402
    AlertingEventSink,
    BoundedRetryPolicy,
    CompositeEventSink,
    ConsoleEventSink,
    InMemoryEventSink,
    JMRRuntimeContext,
    JsonLineEventSink,
    MetricsEventSink,
    call_with_retry,
    emit_runtime_event,
    is_transient_error,
)


class FixedClock:
    def now(self):
        return datetime(2026, 9, 21, 8, 0, tzinfo=UTC)


class ObservabilityTests(unittest.TestCase):
    def test_event_is_complete_redacted_and_projects_metrics_and_alerts(self):
        events = InMemoryEventSink()
        metrics = MetricsEventSink()
        alerts = []
        output = io.StringIO()
        sink = CompositeEventSink(
            [
                events,
                metrics,
                JsonLineEventSink(output),
                AlertingEventSink(alerts.append),
            ]
        )
        context = JMRRuntimeContext(
            event_sink=sink,
            clock=FixedClock(),
            metadata={"run_id": "run-1", "trace_id": "trace-1"},
        )

        emit_runtime_event(
            context=context,
            state={"user_id": "user-1", "case_id": "case-1"},
            node="node-1",
            tool="mcp__scholar__search_publications",
            status="FAILED",
            duration_ms=12.3456,
            retry_count=2,
            failure_id="failure-1",
            details={
                "database_url": "postgresql://user:password@db/private",
                "content": "private body",
                "label": "Bearer top-secret-token",
            },
        )

        event = events.events[0]
        self.assertEqual(
            set(event),
            {
                "event_version",
                "timestamp",
                "user_id",
                "case_id",
                "thread_id",
                "run_id",
                "trace_id",
                "node",
                "tool",
                "status",
                "duration_ms",
                "retry_count",
                "failure_id",
                "details",
            },
        )
        serialized = output.getvalue()
        self.assertNotIn("password", serialized)
        self.assertNotIn("private body", serialized)
        self.assertNotIn("top-secret-token", serialized)
        self.assertEqual(metrics.snapshot()["retry_counts"]["node-1"], 2)
        self.assertEqual(alerts[0]["failure_id"], "failure-1")

    def test_console_reports_stage_and_mcp_without_payloads(self):
        output = io.StringIO()
        sink = ConsoleEventSink(output)
        context = JMRRuntimeContext(event_sink=sink, clock=FixedClock())

        emit_runtime_event(
            context=context,
            state={"user_id": "root", "case_id": "case-1"},
            node="publication_research_agent",
            tool="mcp__scholar__search_publications",
            status="FAILED",
            details={
                "workflow_stage": "RETRIEVING_RESEARCH_EVIDENCE",
                "reason_code": "MCP_CALL_FAILED",
                "content": "private research prompt",
            },
        )

        printed = output.getvalue()
        self.assertIn("stage=RETRIEVING_RESEARCH_EVIDENCE", printed)
        self.assertIn("mcp=mcp__scholar__search_publications", printed)
        self.assertIn("reason=MCP_CALL_FAILED", printed)
        self.assertNotIn("private research prompt", printed)

    def test_sink_failure_never_fails_business_execution(self):
        class BrokenSink:
            def emit(self, event):
                raise OSError("telemetry unavailable and contains no business data")

        emit_runtime_event(
            context=JMRRuntimeContext(event_sink=BrokenSink(), clock=FixedClock()),
            state={"user_id": "u", "case_id": "c"},
            node="safe",
            status="SUCCEEDED",
        )


class ReliabilityTests(unittest.TestCase):
    def test_retry_after_is_capped_and_only_idempotent_calls_retry(self):
        class TemporaryError(RuntimeError):
            status_code = 429
            response = SimpleNamespace(headers={"Retry-After": "999"})

        attempts = 0
        delays = []

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise TemporaryError()
            return "ok"

        result = call_with_retry(
            operation,
            policy=BoundedRetryPolicy(),
            idempotent=True,
            sleep=delays.append,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual(delays, [30.0, 30.0])

        attempts = 0
        with self.assertRaises(TemporaryError):
            call_with_retry(
                operation,
                policy=BoundedRetryPolicy(),
                idempotent=False,
                sleep=delays.append,
            )
        self.assertEqual(attempts, 1)

    def test_auth_schema_and_ssrf_failures_are_not_transient(self):
        class AuthenticationError(RuntimeError):
            pass

        class DatabaseRetry(RuntimeError):
            sqlstate = "40001"

        self.assertFalse(is_transient_error(AuthenticationError()))
        self.assertFalse(is_transient_error(ValueError("schema mismatch")))
        self.assertTrue(is_transient_error(DatabaseRetry()))


class ObjectStoreAndBackupTests(unittest.TestCase):
    def test_file_object_store_is_persistent_immutable_and_traversal_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FileObjectStore(directory)
            uri = store.put(
                object_key="case-1/workflow/item.json",
                content=b"{}",
                content_type="application/json",
            )
            self.assertEqual(FileObjectStore(directory).get(object_uri=uri), b"{}")
            self.assertEqual(store.iter_uris(prefix="case-1/"), [uri])
            with self.assertRaises(FileExistsError):
                store.put(
                    object_key="case-1/workflow/item.json",
                    content=b"different",
                    content_type="application/json",
                )
            with self.assertRaises(ValueError):
                store.put(
                    object_key="../escape",
                    content=b"bad",
                    content_type="text/plain",
                )

    def test_backup_verification_and_restore_cover_all_layers_without_dsn_argv(self):
        commands = []

        def runner(command, *, env, check, capture_output):
            del check, capture_output
            commands.append((list(command), dict(env)))
            if command[0] == "pg_dump":
                output = next(
                    item.split("=", 1)[1]
                    for item in command
                    if item.startswith("--file=")
                )
                Path(output).write_bytes(b"database dump")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            objects = root / "live-objects"
            objects.mkdir()
            (objects / "artifact.json").write_text("evidence", encoding="utf-8")
            backup = create_backup(
                root / "backup-1",
                database_url="postgresql://secret@db/jmr",
                object_store_root=objects,
                confirm_quiesced=True,
                runner=runner,
                now=lambda: datetime(2026, 9, 21, tzinfo=UTC),
            )
            manifest = verify_backup(backup)
            self.assertEqual(len(manifest["database_scope"]), 3)
            self.assertNotIn("postgresql://secret", " ".join(commands[0][0]))
            self.assertEqual(commands[0][1]["PGDATABASE"], "postgresql://secret@db/jmr")

            restored = root / "restored-objects"
            restore_backup(
                backup,
                database_url="postgresql://secret@db/jmr",
                object_store_root=restored,
                confirm_destructive=True,
                runner=runner,
            )
            self.assertEqual(
                (restored / "artifact.json").read_text(encoding="utf-8"),
                "evidence",
            )
            (backup / "postgres.dump").write_bytes(b"tampered")
            with self.assertRaisesRegex(BackupError, "checksum"):
                verify_backup(backup)


class RetentionMigrationAndHealthTests(unittest.TestCase):
    @staticmethod
    def _valid_backup(root: Path, name: str) -> Path:
        objects = root / "objects"
        objects.mkdir(exist_ok=True)

        def runner(command, **_kwargs):
            dump = next(
                item.split("=", 1)[1] for item in command if item.startswith("--file=")
            )
            Path(dump).write_bytes(b"database dump")

        return create_backup(
            root / name,
            database_url="postgresql:///test",
            object_store_root=objects,
            confirm_quiesced=True,
            runner=runner,
            now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        )

    def test_orphan_object_cleanup_is_report_first_and_confirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FileObjectStore(directory)
            retained = store.put(
                object_key="users/u/cases/c/retained.json",
                content=b"{}",
                content_type="application/json",
            )
            orphan = store.put(
                object_key="users/u/cases/c/orphan.json",
                content=b"[]",
                content_type="application/json",
            )
            now = datetime(2026, 9, 21, tzinfo=UTC)
            report = cleanup_orphan_objects(
                store,
                referenced_object_uris=[retained],
                now=now,
            )
            self.assertEqual(report.candidates, (orphan,))
            self.assertEqual(report.deleted, ())
            with self.assertRaises(RetentionConfirmationError):
                cleanup_orphan_objects(
                    store,
                    referenced_object_uris=[retained],
                    now=now,
                    dry_run=False,
                )
            executed = cleanup_orphan_objects(
                store,
                referenced_object_uris=[retained],
                now=now,
                dry_run=False,
                confirm=True,
            )
            self.assertEqual(executed.deleted, (orphan,))
            self.assertEqual(store.get(object_uri=retained), b"{}")
            with self.assertRaises(KeyError):
                store.get(object_uri=orphan)

    def test_object_inventory_combines_business_rows_and_checkpoints(self):
        class Result:
            def fetchall(self):
                return [("jmr-object:///sources/source.json",)]

        class Connection:
            def execute(self, statement):
                if statement != "SELECT object_uri FROM jmr.source_objects":
                    raise AssertionError(statement)
                return Result()

        class Checkpointer:
            def list(self, config):
                if config is not None:
                    raise AssertionError(config)
                return [
                    SimpleNamespace(
                        checkpoint={
                            "channel_values": {
                                "outreach_plan_ref": "jmr-object:///workflow/plan.json"
                            }
                        },
                        pending_writes=(),
                    )
                ]

        self.assertEqual(
            referenced_object_uris(Connection(), Checkpointer()),
            (
                "jmr-object:///sources/source.json",
                "jmr-object:///workflow/plan.json",
            ),
        )

    def test_expired_backup_cleanup_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self._valid_backup(root, "old")
            now = datetime(2026, 3, 1, tzinfo=UTC)
            report = cleanup_expired_backups(
                root, now=now, policy=RetentionPolicy(backup_days=30)
            )
            self.assertEqual(report.candidate_count, 1)
            self.assertTrue(old.exists())
            deleted = cleanup_expired_backups(
                root,
                now=now,
                policy=RetentionPolicy(backup_days=30),
                dry_run=False,
                confirm=True,
            )
            self.assertEqual(deleted.deleted_count, 1)
            self.assertFalse(old.exists())

    def test_audit_cleanup_reports_then_deletes_exact_ids(self):
        class Result:
            def __init__(self, rows):
                self.rows = rows

            def fetchall(self):
                return self.rows

        class Connection:
            def __init__(self):
                self.ids = ["00000000-0000-0000-0000-000000000001"]

            def execute(self, statement, parameters):
                if "SELECT audit_event_id" in statement:
                    return Result([(value,) for value in self.ids])
                self.ids = []
                return Result([(value,) for value in parameters[0]])

        connection = Connection()
        now = datetime(2026, 9, 21, tzinfo=UTC)
        policy = RetentionPolicy(audit_days=365)
        report = cleanup_audit_events(connection, now=now, policy=policy)
        self.assertEqual(report.candidate_count, 1)
        self.assertEqual(len(connection.ids), 1)
        executed = cleanup_audit_events(
            connection,
            now=now,
            policy=policy,
            dry_run=False,
            confirm=True,
        )
        self.assertEqual(executed.deleted_count, 1)
        self.assertEqual(connection.ids, [])

    def test_checkpoint_cleanup_is_due_then_requires_non_dry_run(self):
        class Saver:
            def __init__(self):
                self.deleted = []

            def delete_thread(self, thread_id):
                self.deleted.append(thread_id)

        saver = Saver()
        policy = RetentionPolicy(completed_checkpoint_days=30)
        completed = datetime(2026, 1, 1, tzinfo=UTC)
        now = completed + timedelta(days=31)
        self.assertTrue(
            cleanup_checkpoint(
                saver,
                thread_id="case-1",
                completed_at=completed,
                now=now,
                policy=policy,
            )
        )
        self.assertEqual(saver.deleted, [])
        cleanup_checkpoint(
            saver,
            thread_id="case-1",
            completed_at=completed,
            now=now,
            policy=policy,
            dry_run=False,
            confirm=True,
        )
        self.assertEqual(saver.deleted, ["case-1"])

    def test_backup_retention_only_reports_valid_expired_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self._valid_backup(root, "old")
            invalid = root / "invalid"
            invalid.mkdir()
            (invalid / "manifest.json").write_text("not-json", encoding="utf-8")
            expired = expired_backup_directories(
                root,
                now=datetime(2026, 3, 1, tzinfo=UTC),
                policy=RetentionPolicy(backup_days=30),
            )
            self.assertEqual(expired, [old.resolve()])

    def test_schema_rollback_rejects_destructive_version_zero(self):
        with self.assertRaises(SchemaMigrationError):
            rollback_schema(object(), target_version=0)

    def test_health_probes_are_read_only_and_fail_on_schema_mismatch(self):
        class Result:
            def __init__(self, row):
                self.row = row

            def fetchone(self):
                return self.row

        class Connection:
            def execute(self, statement):
                if statement == "SELECT 1":
                    return Result((1,))
                return Result((999,))

        with tempfile.TemporaryDirectory() as directory:
            report = readiness(Connection(), object_store_root=directory)
        self.assertEqual(liveness()["status"], "ok")
        self.assertEqual(report["checks"]["schema"], "mismatch")
        self.assertEqual(exit_code(report), 1)

    def test_readiness_connection_failure_returns_failed_json(self):
        output = io.StringIO()
        with (
            patch.dict(
                "os.environ",
                {
                    "DATABASE_URL": "postgresql:///missing",
                    "JMR_OBJECT_STORE_ROOT": "/tmp",
                },
            ),
            patch("psycopg.connect", side_effect=psycopg.OperationalError("offline")),
            redirect_stdout(output),
        ):
            self.assertEqual(main(["readiness"]), 1)
        self.assertIn('"database": "failed"', output.getvalue())


class ModelContextTests(unittest.TestCase):
    def test_model_messages_are_trimmed_with_an_explicit_boundary(self):
        messages = [
            {"role": "user", "content": f"message-{index}"} for index in range(30)
        ]
        compacted = compact_messages_for_model(messages, max_messages=5)
        self.assertEqual(len(compacted), 5)
        self.assertIn("26 earlier message", compacted[0]["content"])
        self.assertEqual(compacted[-1]["content"], "message-29")


if __name__ == "__main__":
    unittest.main()
