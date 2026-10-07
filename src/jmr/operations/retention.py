"""Report-first retention for checkpoints, audits, objects, and backups.

Every cleanup API defaults to a side-effect-free report.  Execution requires
both ``dry_run=False`` and ``confirm=True`` so a caller cannot turn a scheduled
inventory job into deletion by changing only one flag.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .backup import BackupError, verify_backup


class RetentionConfirmationError(RuntimeError):
    """Raised when a destructive retention run lacks explicit confirmation."""


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    completed_checkpoint_days: int = 30
    backup_days: int = 30
    audit_days: int = 365

    def __post_init__(self) -> None:
        for name in (
            "completed_checkpoint_days",
            "backup_days",
            "audit_days",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class RetentionReport:
    """One JSON-safe inventory or execution result for a retention category."""

    category: str
    generated_at: str
    cutoff: str | None
    candidates: tuple[str, ...]
    deleted: tuple[str, ...] = ()
    dry_run: bool = True

    def __post_init__(self) -> None:
        if not self.category.strip():
            raise ValueError("retention report category cannot be empty")
        if self.dry_run and self.deleted:
            raise ValueError("a dry-run report cannot contain deleted targets")
        if not set(self.deleted).issubset(self.candidates):
            raise ValueError("deleted targets must be retention candidates")

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def deleted_count(self) -> int:
        return len(self.deleted)

    def as_dict(self) -> dict[str, Any]:
        """Return a stable payload suitable for an operator approval record."""

        return {
            "category": self.category,
            "generated_at": self.generated_at,
            "cutoff": self.cutoff,
            "dry_run": self.dry_run,
            "candidate_count": self.candidate_count,
            "deleted_count": self.deleted_count,
            "candidates": list(self.candidates),
            "deleted": list(self.deleted),
        }

    def __bool__(self) -> bool:
        """Preserve the useful predicate semantics of the original helpers."""

        return bool(self.candidates)


def checkpoint_cleanup_due(
    *,
    completed_at: datetime,
    now: datetime,
    policy: RetentionPolicy,
) -> bool:
    """Return true only for a completed checkpoint older than retention."""

    _aware_utc(completed_at, "completed_at")
    _aware_utc(now, "now")
    return now.astimezone(UTC) >= completed_at.astimezone(UTC) + timedelta(
        days=policy.completed_checkpoint_days
    )


def cleanup_checkpoint(
    checkpointer: Any,
    *,
    thread_id: str,
    completed_at: datetime,
    now: datetime,
    policy: RetentionPolicy,
    dry_run: bool = True,
    confirm: bool = False,
) -> RetentionReport:
    """Report or delete one completed checkpoint after its retention period."""

    normalized_thread_id = _identifier(thread_id, "thread_id")
    _validate_execution_mode(dry_run=dry_run, confirm=confirm)
    due = checkpoint_cleanup_due(
        completed_at=completed_at,
        now=now,
        policy=policy,
    )
    candidates = (normalized_thread_id,) if due else ()
    deleted: tuple[str, ...] = ()
    if candidates and not dry_run:
        checkpointer.delete_thread(normalized_thread_id)
        deleted = candidates
    return _report(
        category="checkpoint",
        now=now,
        cutoff=now.astimezone(UTC) - timedelta(days=policy.completed_checkpoint_days),
        candidates=candidates,
        deleted=deleted,
        dry_run=dry_run,
    )


def audit_retention_candidates(
    connection: Any,
    *,
    now: datetime,
    policy: RetentionPolicy,
) -> tuple[str, ...]:
    """List exact audit-event IDs older than the configured cutoff."""

    cutoff = _retention_cutoff(now, policy.audit_days)
    rows = connection.execute(
        """
        SELECT audit_event_id
        FROM jmr.audit_events
        WHERE created_at < %s
        ORDER BY created_at, audit_event_id
        """,
        (cutoff,),
    ).fetchall()
    return tuple(_identifier(str(row[0]), "audit_event_id") for row in rows)


def cleanup_audit_events(
    connection: Any,
    *,
    now: datetime,
    policy: RetentionPolicy,
    dry_run: bool = True,
    confirm: bool = False,
) -> RetentionReport:
    """Report or delete expired audit rows by their exact primary keys."""

    _validate_execution_mode(dry_run=dry_run, confirm=confirm)
    cutoff = _retention_cutoff(now, policy.audit_days)
    if dry_run:
        candidates = audit_retention_candidates(
            connection,
            now=now,
            policy=policy,
        )
        return _report(
            category="audit_event",
            now=now,
            cutoff=cutoff,
            candidates=candidates,
            dry_run=True,
        )

    with _transaction(connection):
        candidates = audit_retention_candidates(
            connection,
            now=now,
            policy=policy,
        )
        deleted = _delete_audit_candidates(connection, candidates)
    return _report(
        category="audit_event",
        now=now,
        cutoff=cutoff,
        candidates=candidates,
        deleted=deleted,
        dry_run=False,
    )


def orphan_object_uris(
    object_store: Any,
    *,
    referenced_object_uris: Iterable[str],
) -> tuple[str, ...]:
    """Return stored object URIs absent from the durable reference inventory."""

    referenced = {
        _identifier(uri, "referenced_object_uri") for uri in referenced_object_uris
    }
    stored = {_identifier(uri, "stored_object_uri") for uri in object_store.iter_uris()}
    return tuple(sorted(stored - referenced))


def referenced_object_uris(connection: Any, checkpointer: Any) -> tuple[str, ...]:
    """Inventory source rows and every retained checkpoint's object references.

    Call while writers are quiesced, then feed the result to the report-first
    orphan cleanup. A partial checkpoint inventory is not safe for deletion.
    """

    rows = connection.execute("SELECT object_uri FROM jmr.source_objects").fetchall()
    uris = {
        str(row["object_uri"] if isinstance(row, Mapping) else row[0]) for row in rows
    }
    for item in checkpointer.list(None):
        checkpoint = getattr(item, "checkpoint", None)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("checkpoint inventory contains an invalid checkpoint")
        _collect_object_uris(checkpoint.get("channel_values", {}), uris)
        _collect_object_uris(getattr(item, "pending_writes", ()), uris)
    return tuple(sorted(uris))


def cleanup_orphan_objects(
    object_store: Any,
    *,
    referenced_object_uris: Iterable[str],
    now: datetime,
    dry_run: bool = True,
    confirm: bool = False,
) -> RetentionReport:
    """Report or delete exact object URIs that have no durable DB reference."""

    _validate_execution_mode(dry_run=dry_run, confirm=confirm)
    _aware_utc(now, "now")
    candidates = orphan_object_uris(
        object_store,
        referenced_object_uris=referenced_object_uris,
    )
    deleted: list[str] = []
    if not dry_run:
        for object_uri in candidates:
            object_store.delete(object_uri=object_uri)
            deleted.append(object_uri)
    return _report(
        category="orphan_object",
        now=now,
        cutoff=None,
        candidates=candidates,
        deleted=tuple(deleted),
        dry_run=dry_run,
    )


def expired_backup_directories(
    root: Path | str,
    *,
    now: datetime,
    policy: RetentionPolicy,
) -> list[Path]:
    """List expired, direct-child backup directories without deleting them."""

    cutoff = _retention_cutoff(now, policy.backup_days)
    backup_root = _safe_root(root)
    expired: list[Path] = []
    for manifest_path in sorted(backup_root.glob("*/manifest.json")):
        backup_directory = manifest_path.parent
        if backup_directory.is_symlink() or backup_directory.parent != backup_root:
            continue
        try:
            payload = verify_backup(backup_directory)
            created = datetime.fromisoformat(payload["created_at"])
        except (BackupError, OSError, KeyError, TypeError, ValueError):
            continue
        if created.tzinfo is not None and created.astimezone(UTC) < cutoff:
            expired.append(backup_directory)
    return expired


def cleanup_expired_backups(
    root: Path | str,
    *,
    now: datetime,
    policy: RetentionPolicy,
    dry_run: bool = True,
    confirm: bool = False,
) -> RetentionReport:
    """Report or remove exact expired backup directories under one safe root."""

    _validate_execution_mode(dry_run=dry_run, confirm=confirm)
    backup_root = _safe_root(root)
    cutoff = _retention_cutoff(now, policy.backup_days)
    candidates = tuple(
        str(path)
        for path in expired_backup_directories(
            backup_root,
            now=now,
            policy=policy,
        )
    )
    deleted: list[str] = []
    if not dry_run:
        for candidate_value in candidates:
            candidate = Path(candidate_value)
            _validate_backup_candidate(candidate, backup_root)
            verify_backup(candidate)
            shutil.rmtree(candidate)
            deleted.append(candidate_value)
    return _report(
        category="backup",
        now=now,
        cutoff=cutoff,
        candidates=candidates,
        deleted=tuple(deleted),
        dry_run=dry_run,
    )


def _delete_audit_candidates(
    connection: Any,
    candidates: Sequence[str],
) -> tuple[str, ...]:
    if not candidates:
        return ()
    rows = connection.execute(
        """
        DELETE FROM jmr.audit_events
        WHERE audit_event_id = ANY(%s::uuid[])
        RETURNING audit_event_id
        """,
        (list(candidates),),
    ).fetchall()
    deleted = {str(row[0]) for row in rows}
    return tuple(candidate for candidate in candidates if candidate in deleted)


def _collect_object_uris(value: Any, result: set[str]) -> None:
    if isinstance(value, str):
        if value.startswith(("jmr-object:///", "memory://")):
            result.add(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            _collect_object_uris(child, result)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _collect_object_uris(child, result)


def _transaction(connection: Any) -> Any:
    transaction = getattr(connection, "transaction", None)
    return transaction() if callable(transaction) else nullcontext()


def _validate_execution_mode(*, dry_run: bool, confirm: bool) -> None:
    if not isinstance(dry_run, bool) or not isinstance(confirm, bool):
        raise TypeError("dry_run and confirm must be booleans")
    if dry_run and confirm:
        raise ValueError("confirm cannot be combined with dry_run")
    if not dry_run and not confirm:
        raise RetentionConfirmationError(
            "retention execution requires dry_run=False and confirm=True"
        )


def _retention_cutoff(now: datetime, days: int) -> datetime:
    return _aware_utc(now, "now") - timedelta(days=days)


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value.strip()


def _safe_root(root: Path | str) -> Path:
    raw = Path(root).expanduser()
    if raw.is_symlink():
        raise ValueError("retention root must not be a symlink")
    selected = raw.resolve()
    if selected == Path(selected.anchor) or selected == Path.home().resolve():
        raise ValueError("retention root is too broad")
    if not selected.is_dir() or selected.is_symlink():
        raise ValueError("retention root must be an existing real directory")
    return selected


def _validate_backup_candidate(candidate: Path, root: Path) -> None:
    resolved = candidate.resolve()
    if (
        candidate.is_symlink()
        or not candidate.is_dir()
        or resolved.parent != root
        or candidate.parent != root
    ):
        raise ValueError("backup retention candidate escaped its configured root")


def _report(
    *,
    category: str,
    now: datetime,
    cutoff: datetime | None,
    candidates: Sequence[str],
    deleted: Sequence[str] = (),
    dry_run: bool,
) -> RetentionReport:
    generated = _aware_utc(now, "now")
    return RetentionReport(
        category=category,
        generated_at=generated.isoformat(),
        cutoff=cutoff.astimezone(UTC).isoformat() if cutoff else None,
        candidates=tuple(candidates),
        deleted=tuple(deleted),
        dry_run=dry_run,
    )
