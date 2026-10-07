"""Advisory model plans are bounded by the host before persistence."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from langgraph.runtime import Runtime

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.graph.nodes.retrieval import plan_research  # noqa: E402
from jmr.runtime import JMRRuntimeContext  # noqa: E402
from jmr.runtime.schema_validation import validate_json_schema  # noqa: E402


class _LongPlanModel:
    def invoke_structured(self, *, messages, system_prompt, output_schema):
        del messages, system_prompt
        result = {
            "publication_queries": [f"论文查询 {index}" for index in range(6)],
            "kaken_queries": [f"科研费查询 {index}" for index in range(5)],
            "stop_conditions": [f"停止条件 {index}" for index in range(7)],
        }
        validate_json_schema(result, output_schema)
        return result


class _Registry:
    def for_node(self, node_name):
        assert node_name == "plan_research"
        return _LongPlanModel()


class _Repository:
    def __init__(self):
        self.saved_plan = None

    def get_case(self, *, user_id, case_id):
        assert (user_id, case_id) == ("root", "case-1")
        return {"target": {"professor_name": "炭親良", "university_name": "上智大学"}}

    def save_research_plan(self, *, user_id, case_id, plan, idempotency_key):
        self.saved_plan = plan
        return {"research_plan_id": "plan-1"}


class ResearchPlanBoundsTests(unittest.TestCase):
    def test_model_overproducing_queries_does_not_stop_case(self) -> None:
        repository = _Repository()
        runtime = Runtime(
            context=JMRRuntimeContext(
                case_repository=repository, model_registry=_Registry()
            )
        )

        result = plan_research(
            {"user_id": "root", "case_id": "case-1", "retrieval_refs": []},
            runtime,
        )

        self.assertEqual(result["research_plan_id"], "plan-1")
        self.assertEqual(
            {
                key: len(value)
                for key, value in repository.saved_plan["model_plan"].items()
            },
            {
                "publication_queries": 4,
                "kaken_queries": 4,
                "stop_conditions": 4,
            },
        )


if __name__ == "__main__":
    unittest.main()
