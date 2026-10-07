"""P2 durable persistence boundaries and test fakes."""

from .errors import (
    ConstraintError,
    IdempotencyConflictError,
    NotFoundError,
    OwnershipError,
    PersistenceError,
    VersionConflictError,
)
from .memory import InMemoryMemoryStore, PostgresMemoryStore
from .object_store import FileObjectStore, InMemoryObjectStore
from .repository import (
    InMemoryCaseRepository,
    PostgresCaseRepository,
    RepositoryConfigurationError,
    open_postgres_repository,
)
from .schema import (
    MIGRATION_2_DOWN_SQL,
    MIGRATION_2_UP_SQL,
    SCHEMA_SQL,
    SCHEMA_VERSION,
    SchemaMigrationError,
    apply_schema,
    current_schema_version,
    rollback_schema,
)
from .store import StoreConfigurationError, open_postgres_store

__all__ = [
    "MIGRATION_2_DOWN_SQL",
    "MIGRATION_2_UP_SQL",
    "SCHEMA_SQL",
    "SCHEMA_VERSION",
    "ConstraintError",
    "FileObjectStore",
    "IdempotencyConflictError",
    "InMemoryCaseRepository",
    "InMemoryMemoryStore",
    "InMemoryObjectStore",
    "NotFoundError",
    "OwnershipError",
    "PersistenceError",
    "PostgresCaseRepository",
    "PostgresMemoryStore",
    "RepositoryConfigurationError",
    "SchemaMigrationError",
    "StoreConfigurationError",
    "VersionConflictError",
    "apply_schema",
    "current_schema_version",
    "open_postgres_repository",
    "open_postgres_store",
    "rollback_schema",
]
