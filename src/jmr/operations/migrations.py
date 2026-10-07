"""Operator CLI for inspecting, applying, and rolling back JMR migrations."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from jmr.persistence import (
    SCHEMA_VERSION,
    apply_schema,
    current_schema_version,
    rollback_schema,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _connect(database_url: str):
    import psycopg

    return psycopg.connect(database_url)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("migrate")
    rollback = commands.add_parser("rollback")
    rollback.add_argument("--target-version", type=int, required=True)
    rollback.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        parser.error("DATABASE_URL is required")
    with _connect(database_url) as connection:
        if args.command == "status":
            print(
                f"current={current_schema_version(connection)} latest={SCHEMA_VERSION}"
            )
            return 0
        if args.command == "migrate":
            apply_schema(connection)
            print(f"schema migrated to {SCHEMA_VERSION}")
            return 0
        if not args.confirm:
            parser.error("rollback requires --confirm after a verified backup")
        rollback_schema(connection, target_version=args.target_version)
        print(f"schema rolled back to {args.target_version}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by operators
    raise SystemExit(main())
