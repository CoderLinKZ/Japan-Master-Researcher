"""Construction boundary for the official PostgreSQL checkpointer."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from typing import Any

CHECKPOINT_DATABASE_URL_ENV = "DATABASE_URL"


class CheckpointerConfigurationError(RuntimeError):
    """Raised when durable checkpoint storage cannot be configured."""


SaverContextFactory = Callable[[str], Any]


@contextmanager
def open_postgres_checkpointer(
    database_url: str | None = None,
    *,
    setup: bool = False,
    environment: Mapping[str, str] | None = None,
    saver_context_factory: SaverContextFactory | None = None,
) -> Iterator[Any]:
    """Open an official PostgresSaver with an injectable construction seam.

    Imports are deliberately lazy: state and routing tests do not need psycopg.
    ``setup`` defaults to false so schema migration remains an explicit startup
    decision by the CLI or deployment layer.
    """

    selected_environment = os.environ if environment is None else environment
    selected_url = database_url or selected_environment.get(CHECKPOINT_DATABASE_URL_ENV)
    if not isinstance(selected_url, str) or not selected_url.strip():
        raise CheckpointerConfigurationError(
            f"{CHECKPOINT_DATABASE_URL_ENV} is missing"
        )
    selected_url = selected_url.strip()

    if saver_context_factory is None:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise CheckpointerConfigurationError(
                "langgraph-checkpoint-postgres is not installed"
            ) from exc
        saver_context_factory = PostgresSaver.from_conn_string

    stack = ExitStack()
    try:
        context_manager = saver_context_factory(selected_url)
        saver = stack.enter_context(context_manager)
        if setup:
            saver.setup()
    except Exception as exc:
        stack.close()
        raise CheckpointerConfigurationError(
            "Unable to initialize the PostgreSQL checkpointer"
        ) from exc

    try:
        yield saver
    finally:
        stack.close()
