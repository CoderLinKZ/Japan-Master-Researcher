"""Real PostgreSQL restart/resume smoke test for the production graph."""

from __future__ import annotations

import os
import sys
import unittest
import uuid
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ROOT))

from langgraph.types import Command  # noqa: E402

from jmr.api.service import (  # noqa: E402
    ProductionAgentAPIService,
    _snapshot_interrupt_identity,
)
from jmr.graph import compile_graph, create_initial_state  # noqa: E402
from jmr.persistence import open_postgres_repository  # noqa: E402
from jmr.runtime import JMRRuntimeContext, open_postgres_checkpointer  # noqa: E402


class _IncompleteApplicationModel:
    def invoke_structured(self, **_kwargs):
        return {
            "extracted": {},
            "explicit_corrections": [],
            "ambiguous_fragments": [],
        }


class _IncompleteApplicationModelRegistry:
    def for_node(self, _node_name):
        return _IncompleteApplicationModel()


@unittest.skipUnless(
    os.getenv("JMR_RUN_POSTGRES_TESTS") == "1" and os.getenv("DATABASE_URL"),
    "requires JMR_RUN_POSTGRES_TESTS=1 and DATABASE_URL",
)
class LangGraphPostgresIntegrationTests(unittest.TestCase):
    def test_interrupt_survives_restart_and_cases_remain_isolated(self) -> None:
        database_url = os.environ["DATABASE_URL"]
        suffix = uuid.uuid4().hex
        user_id = f"checkpoint-user-{suffix}"
        case_id = f"checkpoint-case-{suffix}"
        other_case_id = f"checkpoint-other-{suffix}"
        config = {"configurable": {"user_id": user_id, "thread_id": case_id}}
        other_config = {
            "configurable": {"user_id": user_id, "thread_id": other_case_id}
        }

        with open_postgres_repository(database_url, setup=True) as repository:
            runtime = JMRRuntimeContext(
                case_repository=repository,
                model_registry=_IncompleteApplicationModelRegistry(),
            )
            try:
                with open_postgres_checkpointer(
                    database_url,
                    setup=True,
                ) as saver:
                    graph = compile_graph(saver)
                    waiting = graph.invoke(
                        create_initial_state(
                            user_id,
                            case_id,
                            messages=[{"role": "user", "content": "请帮我调查教授"}],
                        ),
                        config,
                        context=runtime,
                    )
                    self.assertEqual(
                        waiting["pending_interrupt"]["kind"],
                        "APPLICATION_INPUT_REQUIRED",
                    )

                # A new connection and graph instance simulate process restart.
                with open_postgres_checkpointer(database_url) as saver:
                    restarted = compile_graph(saver)
                    restored = restarted.get_state(config)
                    isolated = restarted.get_state(other_config)

                    self.assertEqual(restored.values["user_id"], user_id)
                    self.assertEqual(restored.values["case_id"], case_id)
                    self.assertTrue(restored.tasks[0].interrupts)
                    identity = _snapshot_interrupt_identity(restored)
                    self.assertIsNotNone(identity)
                    self.assertEqual(identity[0], restored.tasks[0].interrupts[0].id)
                    case_view = ProductionAgentAPIService(
                        {"DATABASE_URL": database_url}
                    ).get_case(user_id=user_id, case_id=case_id)
                    self.assertEqual(case_view["interrupt_token"], identity[2])
                    self.assertFalse(isolated.values)

                    invalid = restarted.invoke(
                        Command(
                            resume={identity[0]: {"fields": {"professor_name": "A"}}}
                        ),
                        config,
                        context=runtime,
                    )
                    self.assertEqual(invalid["run_status"], "WAITING_FOR_USER")
                    self.assertEqual(
                        invalid["pending_interrupt"]["kind"],
                        "APPLICATION_INPUT_REQUIRED",
                    )
            finally:
                with open_postgres_checkpointer(database_url) as saver:
                    saver.delete_thread(case_id)
                    saver.delete_thread(other_case_id)
                repository.connection.execute(
                    "DELETE FROM jmr.research_cases WHERE case_id=%s",
                    (case_id,),
                )
                repository.connection.commit()


if __name__ == "__main__":
    unittest.main()
