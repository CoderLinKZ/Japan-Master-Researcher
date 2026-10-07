#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${JMR_RUN_POSTGRES_TESTS:-0}" == "1" ]]; then
  : "${DATABASE_URL:?DATABASE_URL is required when JMR_RUN_POSTGRES_TESTS=1}"
fi

uv sync --frozen --group dev
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
uv run --frozen python -m compileall -q src tests
PYTHONDONTWRITEBYTECODE=1 uv run --frozen python -m unittest discover \
  -s tests \
  -t . \
  -p 'test_*.py'
