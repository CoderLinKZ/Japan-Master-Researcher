"""Consistent backup and verified restore for all persistence layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jmr.persistence import SCHEMA_VERSION

BACKUP_FORMAT_VERSION = 1


class BackupError(RuntimeError):
    """Raised when a backup is incomplete, corrupt, or unsafe to restore."""


Runner = Callable[..., Any]


def create_backup(
    destination: Path | str,
    *,
    database_url: str,
    object_store_root: Path | str,
    confirm_quiesced: bool,
    runner: Runner = subprocess.run,
    now: Callable[[], datetime] | None = None,
) -> Path:
    """Create a full PostgreSQL dump plus an immutable object archive.

    A full database dump intentionally includes PostgresSaver, PostgresStore,
    and the ``jmr`` business schema.  The DSN is supplied through the child
    environment, never as a command-line argument or manifest value.
    """

    if not confirm_quiesced:
        raise BackupError("backup requires a confirmed quiesced deployment")
    selected_url = _required_secret(database_url, "database_url")
    object_root = Path(object_store_root).expanduser().resolve()
    if not object_root.is_dir():
        raise BackupError("object_store_root must be an existing directory")
    _reject_symlinks(object_root)
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise BackupError("backup destination already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent))
    )
    try:
        database_dump = temporary / "postgres.dump"
        command = [
            "pg_dump",
            "--format=custom",
            "--no-owner",
            "--no-privileges",
            f"--file={database_dump}",
        ]
        runner(
            command,
            env=_database_environment(selected_url),
            check=True,
            capture_output=True,
        )
        if not database_dump.is_file():
            raise BackupError("pg_dump did not create the expected artifact")
        object_archive = temporary / "objects.tar.gz"
        with tarfile.open(object_archive, "w:gz") as archive:
            archive.add(object_root, arcname="objects", recursive=True)
        created_at = (now or (lambda: datetime.now(UTC)))()
        if created_at.tzinfo is None:
            raise BackupError("backup clock must return a timezone-aware datetime")
        manifest = {
            "format_version": BACKUP_FORMAT_VERSION,
            "created_at": created_at.astimezone(UTC).isoformat(),
            "schema_version": SCHEMA_VERSION,
            "consistency": "operator_quiesced",
            "database_scope": [
                "langgraph_checkpoints",
                "langgraph_store",
                "jmr_business_schema",
            ],
            "objects": "objects.tar.gz",
            "files": {
                "postgres.dump": _sha256(database_dump),
                "objects.tar.gz": _sha256(object_archive),
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def verify_backup(backup_directory: Path | str) -> Mapping[str, Any]:
    """Verify manifest version, required layers, paths, and checksums."""

    root = Path(backup_directory).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupError("backup manifest is missing or invalid") from exc
    if manifest.get("format_version") != BACKUP_FORMAT_VERSION:
        raise BackupError("unsupported backup format version")
    schema_version = manifest.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or not 1 <= schema_version <= SCHEMA_VERSION
    ):
        raise BackupError("backup schema version is unsupported")
    if manifest.get("consistency") != "operator_quiesced":
        raise BackupError("backup has no quiesced consistency guarantee")
    required_scopes = {
        "langgraph_checkpoints",
        "langgraph_store",
        "jmr_business_schema",
    }
    if set(manifest.get("database_scope", [])) != required_scopes:
        raise BackupError("backup does not cover all PostgreSQL layers")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise BackupError("backup manifest has no checksums")
    for name in ("postgres.dump", "objects.tar.gz"):
        expected = files.get(name)
        artifact = root / name
        if not isinstance(expected, str) or not artifact.is_file():
            raise BackupError(f"backup artifact is missing: {name}")
        if _sha256(artifact) != expected:
            raise BackupError(f"backup checksum mismatch: {name}")
    with tarfile.open(root / "objects.tar.gz", "r:gz") as archive:
        _validate_archive_members(archive.getmembers())
    return dict(manifest)


def restore_backup(
    backup_directory: Path | str,
    *,
    database_url: str,
    object_store_root: Path | str,
    confirm_destructive: bool,
    runner: Runner = subprocess.run,
) -> None:
    """Restore a verified backup into a quiesced target environment."""

    if not confirm_destructive:
        raise BackupError("restore requires explicit destructive confirmation")
    selected_url = _required_secret(database_url, "database_url")
    backup_root = Path(backup_directory).expanduser().resolve()
    verify_backup(backup_root)
    target_objects = Path(object_store_root).expanduser().resolve()
    if target_objects.exists() and any(target_objects.iterdir()):
        raise BackupError("object store restore target must be empty")
    target_objects.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target_objects.name}.restore.",
            dir=str(target_objects.parent),
        )
    )
    try:
        with tarfile.open(backup_root / "objects.tar.gz", "r:gz") as archive:
            members = archive.getmembers()
            _validate_archive_members(members)
            archive.extractall(temporary, members=members, filter="data")
        restored = temporary / "objects"
        if not restored.is_dir():
            raise BackupError("object archive has no objects root")
        runner(
            [
                "pg_restore",
                "--clean",
                "--if-exists",
                "--no-owner",
                "--no-privileges",
                "--exit-on-error",
                "--single-transaction",
                str(backup_root / "postgres.dump"),
            ],
            env=_database_environment(selected_url),
            check=True,
            capture_output=True,
        )
        if target_objects.exists():
            target_objects.rmdir()
        os.replace(restored, target_objects)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _database_environment(database_url: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PGDATABASE"] = database_url
    return environment


def _required_secret(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BackupError(f"{field_name} must be configured")
    return value.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise BackupError("object store backup refuses symbolic links")


def _validate_archive_members(members: Sequence[tarfile.TarInfo]) -> None:
    for member in members:
        path = Path(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or not path.parts
            or path.parts[0] != "objects"
        ):
            raise BackupError("object archive contains an unsafe member")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup")
    backup.add_argument("destination")
    backup.add_argument("--object-store-root", required=True)
    backup.add_argument("--confirm-quiesced", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("backup_directory")
    restore = commands.add_parser("restore")
    restore.add_argument("backup_directory")
    restore.add_argument("--object-store-root", required=True)
    restore.add_argument("--confirm-destructive-restore", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        verify_backup(args.backup_directory)
        return 0
    database_url = os.environ.get("DATABASE_URL", "")
    if args.command == "backup":
        create_backup(
            args.destination,
            database_url=database_url,
            object_store_root=args.object_store_root,
            confirm_quiesced=args.confirm_quiesced,
        )
        return 0
    restore_backup(
        args.backup_directory,
        database_url=database_url,
        object_store_root=args.object_store_root,
        confirm_destructive=args.confirm_destructive_restore,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by operators
    raise SystemExit(main())
