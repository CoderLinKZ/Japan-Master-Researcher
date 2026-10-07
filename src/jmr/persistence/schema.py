"""The versioned PostgreSQL schema used by the JMR business repositories.

The LangGraph checkpointer and store own their own tables.  This schema is
deliberately limited to business facts and audit data, so a checkpoint can be
deleted or replayed without deleting evidence, directions, or draft history.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SCHEMA_VERSION = 4


SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS jmr;

CREATE TABLE IF NOT EXISTS jmr.schema_migrations (
    version integer PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS jmr.users (
    user_id text PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS jmr.research_cases (
    case_id varchar(255) PRIMARY KEY,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    case_context_id uuid NOT NULL,
    schema_version integer NOT NULL,
    workflow_stage text NOT NULL DEFAULT 'VALIDATING_APPLICATION_INPUTS',
    run_status text NOT NULL DEFAULT 'READY',
    version bigint NOT NULL DEFAULT 1,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz
);
ALTER TABLE jmr.research_cases
    ADD COLUMN IF NOT EXISTS case_context_id uuid;
UPDATE jmr.research_cases
SET case_context_id = gen_random_uuid()
WHERE case_context_id IS NULL;
ALTER TABLE jmr.research_cases
    ALTER COLUMN case_context_id SET NOT NULL;
CREATE INDEX IF NOT EXISTS research_cases_user_idx
    ON jmr.research_cases(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS jmr.case_targets (
    target_id uuid PRIMARY KEY,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    target jsonb NOT NULL,
    version bigint NOT NULL DEFAULT 1,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(case_id),
    UNIQUE(target_id, case_id, user_id)
);

CREATE TABLE IF NOT EXISTS jmr.case_status_history (
    history_id uuid PRIMARY KEY,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    from_stage text,
    to_stage text NOT NULL,
    run_status text NOT NULL,
    node_name text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    idempotency_key text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(user_id, case_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS jmr.research_plans (
    research_plan_id uuid PRIMARY KEY,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    plan jsonb NOT NULL,
    version bigint NOT NULL DEFAULT 1,
    idempotency_key text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(user_id, case_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS jmr.retrieval_runs (
    retrieval_run_id uuid PRIMARY KEY,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    worker_kind text NOT NULL,
    status text NOT NULL,
    result_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    version bigint NOT NULL DEFAULT 1,
    idempotency_key text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(user_id, case_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS jmr.source_objects (
    source_object_id uuid PRIMARY KEY,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    object_uri text NOT NULL,
    sha256 text NOT NULL,
    mime_type text NOT NULL,
    byte_size bigint NOT NULL,
    fetched_at timestamptz NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(case_id, sha256)
);

CREATE TABLE IF NOT EXISTS jmr.evidence_records (
    evidence_id uuid PRIMARY KEY,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    retrieval_run_id uuid REFERENCES jmr.retrieval_runs(retrieval_run_id),
    evidence_kind text NOT NULL,
    external_id text,
    title text,
    record jsonb NOT NULL,
    verification_status text NOT NULL DEFAULT 'UNVERIFIED',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(case_id, evidence_kind, external_id)
);

CREATE TABLE IF NOT EXISTS jmr.evidence_sources (
    evidence_id uuid NOT NULL REFERENCES jmr.evidence_records(evidence_id)
        ON DELETE CASCADE,
    source_object_id uuid REFERENCES jmr.source_objects(source_object_id),
    source_url text NOT NULL,
    source_type text,
    citation jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY(evidence_id, source_url)
);

CREATE TABLE IF NOT EXISTS jmr.evidence_verifications (
    verification_id uuid PRIMARY KEY,
    evidence_id uuid NOT NULL REFERENCES jmr.evidence_records(evidence_id)
        ON DELETE CASCADE,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    status text NOT NULL,
    conflicts jsonb NOT NULL DEFAULT '[]'::jsonb,
    checked_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS jmr.verified_evidence_bundles (
    bundle_id uuid PRIMARY KEY,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    evidence_ids jsonb NOT NULL,
    summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(case_id, bundle_id)
);

CREATE TABLE IF NOT EXISTS jmr.direction_batches (
    direction_batch_id uuid PRIMARY KEY,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    evidence_bundle_id uuid REFERENCES jmr.verified_evidence_bundles(bundle_id),
    directions jsonb NOT NULL,
    revision_round integer NOT NULL DEFAULT 0,
    idempotency_key text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(user_id, case_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS jmr.research_directions (
    direction_id uuid PRIMARY KEY,
    direction_batch_id uuid NOT NULL REFERENCES
        jmr.direction_batches(direction_batch_id)
        ON DELETE CASCADE,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    direction jsonb NOT NULL,
    ordinal integer NOT NULL,
    UNIQUE(direction_batch_id, ordinal)
);

CREATE TABLE IF NOT EXISTS jmr.direction_evidence_links (
    direction_id uuid NOT NULL REFERENCES jmr.research_directions(direction_id)
        ON DELETE CASCADE,
    evidence_id uuid NOT NULL REFERENCES jmr.evidence_records(evidence_id)
        ON DELETE RESTRICT,
    PRIMARY KEY(direction_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS jmr.direction_selections (
    selection_id uuid PRIMARY KEY,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    direction_batch_id uuid NOT NULL REFERENCES
        jmr.direction_batches(direction_batch_id),
    selected_direction_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    custom_direction jsonb,
    idempotency_key text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(user_id, case_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS jmr.case_memory_links (
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    memory_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(case_id, memory_id)
);

CREATE TABLE IF NOT EXISTS jmr.outreach_iterations (
    iteration_id uuid PRIMARY KEY,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    parent_iteration_id uuid REFERENCES jmr.outreach_iterations(iteration_id),
    iteration_no integer NOT NULL,
    content text NOT NULL,
    language text NOT NULL DEFAULT 'zh-CN',
    citation_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    idempotency_key text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(user_id, case_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS jmr.outreach_reviews (
    review_id uuid PRIMARY KEY,
    iteration_id uuid NOT NULL REFERENCES jmr.outreach_iterations(iteration_id)
        ON DELETE CASCADE,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    reviewer_kind text NOT NULL,
    status text NOT NULL,
    result jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS jmr.idempotency_keys (
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    case_id varchar(255) NOT NULL,
    operation text NOT NULL,
    idempotency_key text NOT NULL,
    payload_hash text NOT NULL,
    result jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(user_id, case_id, operation, idempotency_key)
);

CREATE TABLE IF NOT EXISTS jmr.audit_events (
    audit_event_id uuid PRIMARY KEY,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    case_id varchar(255),
    event_type text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
"""

MIGRATION_2_UP_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS outreach_iterations_case_iteration_uidx
    ON jmr.outreach_iterations(case_id, iteration_no);
"""

MIGRATION_2_DOWN_SQL = """
DROP INDEX IF EXISTS jmr.outreach_iterations_case_iteration_uidx;
"""

MIGRATION_3_UP_SQL = """
CREATE TABLE IF NOT EXISTS jmr.profile_memory_operations (
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    operation text NOT NULL,
    idempotency_key text NOT NULL,
    payload_hash text NOT NULL,
    result jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, operation, idempotency_key)
);
CREATE TABLE IF NOT EXISTS jmr.case_deletions (
    case_id varchar(255) PRIMARY KEY,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    object_uris jsonb NOT NULL DEFAULT '[]'::jsonb,
    checkpoint_deleted boolean NOT NULL DEFAULT false,
    business_deleted boolean NOT NULL DEFAULT false,
    status text NOT NULL DEFAULT 'PENDING',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz
);
"""

MIGRATION_3_DOWN_SQL = """
DROP TABLE IF EXISTS jmr.case_deletions;
DROP TABLE IF EXISTS jmr.profile_memory_operations;
"""

MIGRATION_4_UP_SQL = """
CREATE TABLE IF NOT EXISTS jmr.case_messages (
    message_id uuid PRIMARY KEY,
    case_id varchar(255) NOT NULL REFERENCES jmr.research_cases(case_id)
        ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES jmr.users(user_id),
    role text NOT NULL CHECK (role IN ('user')),
    content text NOT NULL CHECK (length(content) BETWEEN 1 AND 12000),
    idempotency_key text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(user_id, case_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS case_messages_timeline_idx
    ON jmr.case_messages(user_id, case_id, created_at, message_id);
"""

MIGRATION_4_DOWN_SQL = """
DROP TABLE IF EXISTS jmr.case_messages;
"""


class SchemaMigrationError(RuntimeError):
    """Raised when a requested schema transition is unsupported or unsafe."""


def apply_schema(connection: Any) -> None:
    """Apply the idempotent schema and record the migration version."""

    current = current_schema_version(connection)
    if current > SCHEMA_VERSION:
        raise SchemaMigrationError(
            f"database schema version {current} is newer than this release"
        )
    # Execute one statement at a time.  PostgresStore opens its connection
    # with ``prepare_threshold=0`` and PostgreSQL rejects a multi-command
    # prepared statement; the migration contains no semicolons inside values,
    # so this small splitter is sufficient and keeps the SQL portable.
    with connection.transaction():
        for statement in SCHEMA_SQL.split(";"):
            statement = statement.strip()
            if statement:
                connection.execute(statement)
        connection.execute(
            """
            INSERT INTO jmr.schema_migrations(version)
            VALUES (%s)
            ON CONFLICT (version) DO NOTHING
            """,
            (1,),
        )
        for statement in MIGRATION_2_UP_SQL.split(";"):
            statement = statement.strip()
            if statement:
                connection.execute(statement)
        connection.execute(
            """
            INSERT INTO jmr.schema_migrations(version)
            VALUES (%s)
            ON CONFLICT (version) DO NOTHING
            """,
            (2,),
        )
        for statement in MIGRATION_3_UP_SQL.split(";"):
            statement = statement.strip()
            if statement:
                connection.execute(statement)
        connection.execute(
            """
            INSERT INTO jmr.schema_migrations(version)
            VALUES (%s)
            ON CONFLICT (version) DO NOTHING
            """,
            (3,),
        )
        for statement in MIGRATION_4_UP_SQL.split(";"):
            statement = statement.strip()
            if statement:
                connection.execute(statement)
        connection.execute(
            """
            INSERT INTO jmr.schema_migrations(version)
            VALUES (%s)
            ON CONFLICT (version) DO NOTHING
            """,
            (4,),
        )


def current_schema_version(connection: Any) -> int:
    """Return the latest recorded JMR migration, or zero before bootstrap."""

    exists = connection.execute(
        "SELECT to_regclass('jmr.schema_migrations')"
    ).fetchone()
    if not exists or _first_column(exists) is None:
        return 0
    row = connection.execute(
        "SELECT COALESCE(MAX(version), 0) FROM jmr.schema_migrations"
    ).fetchone()
    return int(_first_column(row)) if row else 0


def _first_column(row: Any) -> Any:
    return next(iter(row.values())) if isinstance(row, Mapping) else row[0]


def rollback_schema(connection: Any, *, target_version: int) -> None:
    """Rollback reversible migrations after an operator-created backup.

    Version 1 creates all business tables and is intentionally not dropped by
    application code.  Restoring below version 1 must use the verified full
    database backup procedure instead of an accidental ``DROP SCHEMA``.
    """

    if target_version < 1 or target_version > SCHEMA_VERSION:
        raise SchemaMigrationError(
            f"target_version must be between 1 and {SCHEMA_VERSION}"
        )
    with connection.transaction():
        current = current_schema_version(connection)
        if current > SCHEMA_VERSION:
            raise SchemaMigrationError(
                f"database schema version {current} is newer than this release"
            )
        if current >= 4 and target_version < 4:
            for statement in MIGRATION_4_DOWN_SQL.split(";"):
                statement = statement.strip()
                if statement:
                    connection.execute(statement)
            connection.execute(
                "DELETE FROM jmr.schema_migrations WHERE version=%s",
                (4,),
            )
        if current >= 3 and target_version < 3:
            for statement in MIGRATION_3_DOWN_SQL.split(";"):
                statement = statement.strip()
                if statement:
                    connection.execute(statement)
            connection.execute(
                "DELETE FROM jmr.schema_migrations WHERE version=%s",
                (3,),
            )
        if current >= 2 and target_version < 2:
            for statement in MIGRATION_2_DOWN_SQL.split(";"):
                statement = statement.strip()
                if statement:
                    connection.execute(statement)
            connection.execute(
                "DELETE FROM jmr.schema_migrations WHERE version=%s",
                (2,),
            )


def schema_statements() -> Iterable[str]:
    """Return all forward migrations in application order."""

    yield SCHEMA_SQL
    yield MIGRATION_2_UP_SQL
    yield MIGRATION_3_UP_SQL
    yield MIGRATION_4_UP_SQL
