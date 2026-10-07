"""Runtime configuration and PostgreSQL construction-boundary tests."""

from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.cli import validate_model_environment  # noqa: E402
from jmr.runtime import (  # noqa: E402
    CheckpointerConfigurationError,
    open_postgres_checkpointer,
)


class FakeSaver:
    def __init__(self) -> None:
        self.setup_calls = 0

    def setup(self) -> None:
        self.setup_calls += 1


class RuntimeConfigurationTests(unittest.TestCase):
    def test_missing_database_url_fails_without_memory_fallback(self) -> None:
        with self.assertRaisesRegex(
            CheckpointerConfigurationError,
            "DATABASE_URL is missing",
        ):
            with open_postgres_checkpointer(environment={}):
                self.fail("missing DATABASE_URL must not yield a saver")

    def test_setup_is_called_on_injected_postgres_saver(self) -> None:
        saver = FakeSaver()

        @contextmanager
        def factory(database_url):
            self.assertEqual(database_url, "postgresql://example")
            yield saver

        with open_postgres_checkpointer(
            "postgresql://example",
            setup=True,
            saver_context_factory=factory,
        ) as selected:
            self.assertIs(selected, saver)

        self.assertEqual(saver.setup_calls, 1)

    def test_database_connection_failure_is_typed_and_not_hidden(self) -> None:
        def factory(database_url):
            raise OSError("database unavailable")

        with self.assertRaisesRegex(
            CheckpointerConfigurationError,
            "Unable to initialize",
        ):
            with open_postgres_checkpointer(
                "postgresql://unavailable",
                saver_context_factory=factory,
            ):
                self.fail("a failed connection must not yield a saver")

    def test_caller_errors_are_not_mislabeled_as_database_failures(self) -> None:
        @contextmanager
        def factory(database_url):
            yield FakeSaver()

        with self.assertRaisesRegex(RuntimeError, "node failed"):
            with open_postgres_checkpointer(
                "postgresql://example",
                saver_context_factory=factory,
            ):
                raise RuntimeError("node failed")

    def test_missing_model_configuration_is_reported_together(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "ANTHROPIC_API_KEY, MODEL_ID",
            ):
                validate_model_environment()


if __name__ == "__main__":
    unittest.main()
