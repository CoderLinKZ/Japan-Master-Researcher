"""Typed errors raised by the durable JMR repositories.

The graph uses these errors to distinguish an invalid user request from a
retryable database failure.  Keeping the errors in the persistence package
also prevents the graph layer from depending on a particular SQL driver.
"""

from __future__ import annotations


class PersistenceError(RuntimeError):
    """Base class for repository failures."""


class NotFoundError(PersistenceError):
    """The requested record does not exist in the caller's scope."""


class OwnershipError(PersistenceError):
    """A record exists but belongs to another user or case."""


class IdempotencyConflictError(PersistenceError):
    """An idempotency key was reused with a different request payload."""


class VersionConflictError(PersistenceError):
    """An optimistic-lock version is stale."""


class ConstraintError(PersistenceError):
    """A repository invariant or business constraint was violated."""
