"""Real PostgreSQL regression for parallel worker persistence and P3-P6 flow."""

from __future__ import annotations

import os
import sys
import unittest
import uuid
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ROOT))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from jmr.graph import compile_graph, create_initial_state  # noqa: E402
from jmr.graph.nodes.common import read_json_artifact  # noqa: E402
from jmr.persistence import InMemoryObjectStore, open_postgres_repository  # noqa: E402
from jmr.runtime import JMRRuntimeContext  # noqa: E402
from tests.fixtures.jmr_runtime import (  # noqa: E402
    DeterministicModelRegistry,
    FakeRetrievalGateway,
)


@unittest.skipUnless(
    os.getenv("JMR_RUN_POSTGRES_TESTS") == "1" and os.getenv("DATABASE_URL"),
    "requires JMR_RUN_POSTGRES_TESTS=1 and DATABASE_URL",
)
class FullWorkflowPostgresTests(unittest.TestCase):
    def test_parallel_workers_commit_and_complete_without_transaction_overlap(self):
        database_url = os.environ["DATABASE_URL"]
        suffix = uuid.uuid4().hex
        user_id = f"full-user-{suffix}"
        case_id = f"full-case-{suffix}"

        try:
            with open_postgres_repository(database_url, setup=True) as repository:
                graph = compile_graph(InMemorySaver())
                runtime = JMRRuntimeContext(
                    case_repository=repository,
                    model_registry=DeterministicModelRegistry(),
                    mcp_gateway=FakeRetrievalGateway(),
                    object_store=InMemoryObjectStore(),
                )
                config = {"configurable": {"user_id": user_id, "thread_id": case_id}}
                waiting_memory = graph.invoke(
                    create_initial_state(
                        user_id,
                        case_id,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "大学：东京大学，研究科：工学系研究科，"
                                    "教授：山田太郎"
                                ),
                            }
                        ],
                    ),
                    config,
                    context=runtime,
                )
                self.assertEqual(len(waiting_memory["retrieval_refs"]), 2)
                waiting_direction = graph.invoke(
                    Command(
                        resume={
                            "selected_paper_evidence_ids": [read_json_artifact(runtime, waiting_memory["paper_options_ref"])[0]["evidence_id"]],
                            "selected_memory_ids": [],
                            "allow_unpersonalized": True,
                            "idempotency_key": "memory-choice",
                        }
                    ),
                    config,
                    context=runtime,
                )
                batch = repository.get_direction_batch(
                    user_id=user_id,
                    case_id=case_id,
                    direction_batch_id=waiting_direction["direction_batch_id"],
                )
                completed = graph.invoke(
                    Command(
                        resume={
                            "direction_batch_id": waiting_direction[
                                "direction_batch_id"
                            ],
                            "selected_direction_ids": [
                                batch["directions"][0]["direction_id"]
                            ],
                            "custom_direction": None,
                            "idempotency_key": "direction-choice",
                        }
                    ),
                    config,
                    context=runtime,
                )
                self.assertEqual(completed["run_status"], "COMPLETED")
                evidence_count = repository.connection.execute(
                    "SELECT count(*) FROM jmr.evidence_records WHERE case_id=%s",
                    (case_id,),
                ).fetchone()[0]
                link_count = repository.connection.execute(
                    "SELECT count(*) FROM jmr.direction_evidence_links"
                ).fetchone()[0]
                source_object_count = repository.connection.execute(
                    "SELECT count(*) FROM jmr.source_objects WHERE case_id=%s",
                    (case_id,),
                ).fetchone()[0]
                evidence_source_count = repository.connection.execute(
                    "SELECT count(*) FROM jmr.evidence_sources"
                ).fetchone()[0]
                self.assertEqual(evidence_count, 2)
                self.assertEqual(link_count, 6)
                self.assertEqual(source_object_count, 2)
                self.assertEqual(evidence_source_count, 2)
        finally:
            with open_postgres_repository(database_url) as repository:
                repository.connection.execute("DROP SCHEMA IF EXISTS jmr CASCADE")
                repository.connection.commit()


if __name__ == "__main__":
    unittest.main()
