"""Read-only liveness and readiness checks for deployment probes."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from jmr.persistence import SCHEMA_VERSION, current_schema_version

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def liveness() -> dict[str, Any]:
    """A process-local probe that never calls external dependencies."""

    return {"status": "ok", "checks": {"process": "ok"}}


def readiness(
    connection: Any,
    *,
    object_store_root: Path | str,
) -> dict[str, Any]:
    """Check database connectivity/version and the mounted object directory."""

    checks: dict[str, str] = {}
    try:
        row = connection.execute("SELECT 1").fetchone()
        if not row or row[0] != 1:
            raise RuntimeError("unexpected database probe result")
        checks["database"] = "ok"
        version = current_schema_version(connection)
        checks["schema"] = "ok" if version == SCHEMA_VERSION else "mismatch"
    except Exception:
        checks["database"] = "failed"
        checks.setdefault("schema", "unknown")
    root = Path(object_store_root).expanduser().resolve()
    checks["object_store"] = (
        "ok"
        if root.is_dir() and os.access(root, os.R_OK | os.W_OK | os.X_OK)
        else "failed"
    )
    status = "ok" if all(value == "ok" for value in checks.values()) else "failed"
    return {"status": status, "checks": checks}


def exit_code(report: Mapping[str, Any]) -> int:
    return 0 if report.get("status") == "ok" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", choices=("liveness", "readiness"))
    args = parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if args.probe == "liveness":
        report = liveness()
    else:
        database_url = os.environ.get("DATABASE_URL", "").strip()
        object_root = os.environ.get("JMR_OBJECT_STORE_ROOT", "").strip()
        if not database_url or not object_root:
            report = {
                "status": "failed",
                "checks": {"configuration": "missing"},
            }
        else:
            import psycopg

            try:
                with psycopg.connect(database_url) as connection:
                    report = readiness(connection, object_store_root=object_root)
            except psycopg.Error:
                report = {
                    "status": "failed",
                    "checks": {"database": "failed", "schema": "unknown"},
                }
    print(json.dumps(report, sort_keys=True))
    return exit_code(report)


if __name__ == "__main__":  # pragma: no cover - exercised by deployment
    raise SystemExit(main())
