"""Construction boundary for the LangGraph PostgresStore.

Long-lived user Memory is intentionally separate from the JMR business
tables.  The helper mirrors :func:`open_postgres_checkpointer` and keeps the
optional import lazy for state-only unit tests.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from typing import Any

from .schema import apply_schema

STORE_DATABASE_URL_ENV = "DATABASE_URL"


class StoreConfigurationError(RuntimeError):
    """Raised when the PostgresStore cannot be configured."""


StoreContextFactory = Callable[[str], Any]


@contextmanager
def open_postgres_store(
    database_url: str | None = None,
    *,
    setup: bool = False,
    environment: Mapping[str, str] | None = None,
    store_context_factory: StoreContextFactory | None = None,
    setup_business_schema: bool = False,
) -> Iterator[Any]:
    """Yield a configured ``PostgresStore`` without creating global state."""

    selected_environment = os.environ if environment is None else environment
    selected_url = database_url or selected_environment.get(STORE_DATABASE_URL_ENV)
    if not isinstance(selected_url, str) or not selected_url.strip():
        raise StoreConfigurationError(f"{STORE_DATABASE_URL_ENV} is missing")
    selected_url = selected_url.strip()

    if store_context_factory is None:
        try:
            from langgraph.store.postgres import PostgresStore
        except ImportError as exc:  # pragma: no cover - dependency is optional
            raise StoreConfigurationError(
                "langgraph-checkpoint-postgres is not installed"
            ) from exc
        store_context_factory = PostgresStore.from_conn_string

    stack = ExitStack()
    try:
        store = stack.enter_context(store_context_factory(selected_url))
        if setup:
            store.setup()
        if setup_business_schema:
            # PostgresStore exposes its connection as ``conn`` in the current
            # release.  A custom factory can opt into this by exposing the
            # same attribute; otherwise callers should run apply_schema with
            # their own business connection.
            connection = getattr(store, "conn", None)
            if connection is None:
                raise StoreConfigurationError(
                    "business schema setup requires a store connection"
                )
            apply_schema(connection)
    except StoreConfigurationError:
        stack.close()
        raise
    except Exception as exc:
        stack.close()
        raise StoreConfigurationError(
            "Unable to initialize the PostgreSQL store"
        ) from exc

    try:
        yield store
    finally:
        stack.close()
