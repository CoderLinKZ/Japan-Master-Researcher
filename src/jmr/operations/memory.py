"""Local operator CLI for explicit applicant Memory management."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from dotenv import load_dotenv

from jmr.persistence import (
    SCHEMA_VERSION,
    PostgresMemoryStore,
    current_schema_version,
    open_postgres_store,
)
from mcp_servers.memory import create_memory_server

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    parser.add_argument(
        "--payload-file",
        type=Path,
        help="JSON payload file; omitted input is read from stdin",
    )
    parser.add_argument(
        "action",
        choices=("list", "get", "create", "update", "delete"),
    )
    args = parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if args.payload_file is not None:
        raw = args.payload_file.read_text(encoding="utf-8")
    elif args.action == "list" and sys.stdin.isatty():
        raw = "{}"
    else:
        raw = sys.stdin.read()
    payload = json.loads(raw or "{}")
    if not isinstance(payload, Mapping):
        parser.error("payload must be a JSON object")
    with open_postgres_store() as store:
        connection = store.conn
        if current_schema_version(connection) != SCHEMA_VERSION:
            parser.error("business schema is not current; run migrations first")
        connection.commit()
        server = create_memory_server(PostgresMemoryStore(store), user_id=args.user_id)
        result = server.call_tool(
            f"{args.action}_memory" if args.action != "list" else "list_memories",
            payload,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by operators
    raise SystemExit(main())
