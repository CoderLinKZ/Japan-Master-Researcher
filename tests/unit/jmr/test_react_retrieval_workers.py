"""Behavior tests for the formal bounded ReAct retrieval workers."""

from __future__ import annotations

import json
import sys
import unittest
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.graph.nodes.retrieval import (  # noqa: E402
    kaken_research_agent,
    publication_research_agent,
)
from jmr.persistence import InMemoryCaseRepository  # noqa: E402
from jmr.runtime import (  # noqa: E402
    JMRRuntimeContext,
    ModelToolCall,
    ToolModelTurn,
)

PUBLICATION_NODE = "publication_research_agent"
KAKEN_NODE = "kaken_research_agent"
DISCOVERY_TOOL = "mcp__scholar__discover_official_sources"
PUBLICATION_TOOL = "mcp__scholar__search_publications"
KAKEN_TOOL = "mcp__kaken__search_projects"


class ScriptedToolModel:
    def __init__(self, turns: list[ToolModelTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    def invoke_tool_step(self, **kwargs: Any) -> ToolModelTurn:
        self.calls.append(kwargs)
        if not self.turns:
            raise AssertionError("model was invoked more times than scripted")
        return self.turns.pop(0)


class ModelRegistry:
    def __init__(self, models: Mapping[str, Any]) -> None:
        self.models = dict(models)

    def for_node(self, node_name: str) -> Any:
        return self.models[node_name]


class FakeGateway:
    def __init__(
        self,
        handler: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def tools_for_node(self, node_name: str) -> list[dict[str, Any]]:
        names = (
            [DISCOVERY_TOOL, PUBLICATION_TOOL]
            if node_name == PUBLICATION_NODE
            else [KAKEN_TOOL]
        )
        return [
            {
                "name": name,
                "description": f"fake {name}",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": True,
                },
            }
            for name in names
        ]

    def call_tool(
        self,
        *,
        node_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        trusted_context: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "node_name": node_name,
                "tool_name": tool_name,
                "arguments": dict(arguments),
                "trusted_context": dict(trusted_context),
            }
        )
        return self.handler(tool_name, arguments)


class BoundedReActRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryCaseRepository()
        created = self.repository.create_case(
            user_id="user-1",
            case_id="case-1",
            schema_version=2,
            idempotency_key="create-case",
        )
        self.repository.update_target(
            user_id="user-1",
            case_id="case-1",
            target={
                "university_name": "Tokyo University",
                "graduate_school_name": "Engineering",
                "laboratory_name": "Trustworthy AI Lab",
                "professor_name": "Taro Yamada",
                "professor_name_variants": ["Yamada Taro"],
                "official_urls": ["https://u.example/professor"],
                "date_from": "2025-09-20",
                "date_to": "2026-09-20",
            },
            expected_version=created["version"],
            idempotency_key="target",
        )
        plan = self.repository.save_research_plan(
            user_id="user-1",
            case_id="case-1",
            plan={
                "model_plan": {
                    "publication_queries": ["trustworthy AI"],
                    "kaken_queries": ["research grant"],
                    "stop_conditions": ["stop after verified sources"],
                }
            },
            idempotency_key="plan",
        )
        self.state = {
            "user_id": "user-1",
            "case_id": "case-1",
            "research_plan_id": plan["research_plan_id"],
        }

    def test_publication_worker_performs_discovery_search_and_grounded_finish(
        self,
    ) -> None:
        discovery = _discovery_result("SUCCESS")
        publication = _retrieval_result("publication", "SUCCESS")
        model = ScriptedToolModel(
            [
                _tool_turn("discover", DISCOVERY_TOOL, {"professor_name": "evil"}),
                _tool_turn("search", PUBLICATION_TOOL, {"date_from": "1900-01-01"}),
                ToolModelTurn(final_result={"request_id": publication["request_id"]}),
            ]
        )
        gateway = FakeGateway(
            lambda tool, _arguments: (
                discovery if tool == DISCOVERY_TOOL else publication
            )
        )

        update = publication_research_agent(
            self.state,
            self._runtime(PUBLICATION_NODE, model, gateway),
        )

        self.assertEqual(update["retrieval_refs"][0]["mcp_status"], "SUCCESS")
        self.assertEqual(len(model.calls), 3)
        self.assertIn("trustworthy AI", str(model.calls[0]["messages"][0]["content"]))
        self.assertEqual(
            {tool["name"] for tool in model.calls[0]["tools"]},
            {DISCOVERY_TOOL, PUBLICATION_TOOL},
        )
        self.assertEqual(len(gateway.calls), 2)
        first_arguments = gateway.calls[0]["arguments"]
        self.assertEqual(first_arguments["professor_name"], "Taro Yamada")
        self.assertEqual(first_arguments["search_start_date"], "2025-09-20")
        second_arguments = gateway.calls[1]["arguments"]
        self.assertEqual(second_arguments["search_end_date"], "2026-09-20")
        self.assertEqual(
            second_arguments["official_sources"][0]["url"],
            "https://u.example/lab",
        )
        second_messages = model.calls[1]["messages"]
        self.assertEqual(second_messages[-2]["content"][0]["type"], "tool_use")
        self.assertEqual(second_messages[-1]["content"][0]["type"], "tool_result")
        observation = json.loads(
            model.calls[2]["messages"][-1]["content"][0]["content"]
        )
        self.assertEqual(
            observation["record_previews"][0]["evidence_id"],
            publication["records"][0]["evidence_id"],
        )
        persisted = self._persisted(update)
        self.assertEqual(
            persisted["result_summary"]["request_id"],
            publication["request_id"],
        )
        self.assertEqual(
            persisted["result_summary"]["records"][0]["evidence_id"],
            publication["records"][0]["evidence_id"],
        )

    def test_kaken_worker_stops_at_six_model_rounds(self) -> None:
        result = _retrieval_result("kaken_project", "SUCCESS")
        model = ScriptedToolModel(
            [_tool_turn(str(index), KAKEN_TOOL, {}) for index in range(6)]
        )
        gateway = FakeGateway(lambda _tool, _arguments: result)

        update = kaken_research_agent(
            self.state,
            self._runtime(KAKEN_NODE, model, gateway),
        )

        self.assertEqual(len(model.calls), 6)
        self.assertEqual(len(gateway.calls), 6)
        self.assertEqual(update["retrieval_refs"][0]["mcp_status"], "PARTIAL")
        self.assertIn("6-round limit", " ".join(update["warnings"]))

    def test_identical_tool_input_and_error_opens_circuit_on_third_failure(
        self,
    ) -> None:
        model = ScriptedToolModel(
            [_tool_turn(str(index), KAKEN_TOOL, {}) for index in range(6)]
        )

        def fail(_tool: str, _arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            raise TimeoutError("upstream timeout")

        gateway = FakeGateway(fail)
        update = kaken_research_agent(
            self.state,
            self._runtime(KAKEN_NODE, model, gateway),
        )

        self.assertEqual(len(model.calls), 3)
        self.assertEqual(len(gateway.calls), 3)
        self.assertEqual(update["retrieval_refs"][0]["mcp_status"], "FAILED")
        self.assertIn("circuit breaker", " ".join(update["errors"]))

    def test_needs_user_confirmation_ends_without_another_model_turn(self) -> None:
        ambiguous = _discovery_result("NEEDS_USER_CONFIRMATION")
        model = ScriptedToolModel([_tool_turn("discover", DISCOVERY_TOOL, {})])
        gateway = FakeGateway(lambda _tool, _arguments: ambiguous)

        update = publication_research_agent(
            self.state,
            self._runtime(PUBLICATION_NODE, model, gateway),
        )

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(
            update["retrieval_refs"][0]["mcp_status"],
            "NEEDS_USER_CONFIRMATION",
        )
        persisted = self._persisted(update)
        self.assertEqual(
            persisted["result_summary"]["discovered_sources"][0]["url"],
            "https://u.example/lab",
        )

    def test_forbidden_tool_request_is_not_dispatched(self) -> None:
        model = ScriptedToolModel([_tool_turn("forbidden", KAKEN_TOOL, {})])
        gateway = FakeGateway(
            lambda _tool, _arguments: self.fail("forbidden tool was called")
        )

        update = publication_research_agent(
            self.state,
            self._runtime(PUBLICATION_NODE, model, gateway),
        )

        self.assertEqual(gateway.calls, [])
        self.assertEqual(update["retrieval_refs"][0]["mcp_status"], "FAILED")
        self.assertIn("allowlist", " ".join(update["errors"]))

    def test_two_ungrounded_final_results_fail_structural_correction(self) -> None:
        invented = _retrieval_result("kaken_project", "SUCCESS")
        model = ScriptedToolModel(
            [
                ToolModelTurn(final_result=invented),
                ToolModelTurn(final_result=invented),
            ]
        )
        gateway = FakeGateway(
            lambda _tool, _arguments: self.fail("no tool should be called")
        )

        update = kaken_research_agent(
            self.state,
            self._runtime(KAKEN_NODE, model, gateway),
        )

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(gateway.calls, [])
        self.assertEqual(update["retrieval_refs"][0]["mcp_status"], "FAILED")
        self.assertIn("ungrounded", " ".join(update["errors"]))

    def test_missing_tool_calling_model_is_a_configuration_error(self) -> None:
        class StructuredOnlyModel:
            def invoke_structured(self, **_kwargs: Any) -> Mapping[str, Any]:
                return {}

        gateway = FakeGateway(
            lambda _tool, _arguments: self.fail("no tool should be called")
        )
        runtime = JMRRuntimeContext(
            case_repository=self.repository,
            model_registry=ModelRegistry({KAKEN_NODE: StructuredOnlyModel()}),
            mcp_gateway=gateway,
        )

        with self.assertRaisesRegex(RuntimeError, "ToolCallingModel"):
            kaken_research_agent(self.state, runtime)
        self.assertEqual(gateway.calls, [])

    def test_tool_turn_contract_rejects_ambiguous_or_empty_turns(self) -> None:
        call = ModelToolCall(call_id="one", name=KAKEN_TOOL, arguments={})
        with self.assertRaisesRegex(ValueError, "either tool_calls or final_result"):
            ToolModelTurn()
        with self.assertRaisesRegex(ValueError, "either tool_calls or final_result"):
            ToolModelTurn(
                tool_calls=(call,),
                final_result=_retrieval_result("kaken_project", "SUCCESS"),
            )

    def _runtime(
        self,
        node_name: str,
        model: ScriptedToolModel,
        gateway: FakeGateway,
    ) -> JMRRuntimeContext:
        return JMRRuntimeContext(
            case_repository=self.repository,
            model_registry=ModelRegistry({node_name: model}),
            mcp_gateway=gateway,
        )

    def _persisted(self, update: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.repository.get_retrieval_runs(
            user_id="user-1",
            case_id="case-1",
            retrieval_run_ids=[update["retrieval_refs"][0]["retrieval_run_id"]],
        )[0]


def _tool_turn(call_id: str, name: str, arguments: Mapping[str, Any]) -> ToolModelTurn:
    return ToolModelTurn(
        tool_calls=(ModelToolCall(call_id=call_id, name=name, arguments=arguments),)
    )


def _discovery_result(status: str) -> dict[str, Any]:
    return {
        "request_id": "discovery-request",
        "status": status,
        "query": "official source query",
        "source_name": "fake-scholar",
        "retrieved_at": "2026-09-20T00:00:00+00:00",
        "sources": [
            {
                "url": "https://u.example/lab",
                "source_type": "laboratory_website",
                "discovery_method": "web_search",
                "verification_status": "PARTIAL",
                "title": "Trustworthy AI Lab",
                "matched_signals": ["professor", "institution"],
            }
        ],
        "warnings": [],
        "errors": [],
    }


def _retrieval_result(evidence_type: str, status: str) -> dict[str, Any]:
    record = {
        "evidence_id": f"{evidence_type}-evidence",
        "evidence_type": evidence_type,
        "title": "Grounded result",
        "source_name": "fake-provider",
        "retrieved_at": "2026-09-20T00:00:00+00:00",
        "authors": ["Taro Yamada"],
        "verification_status": "VERIFIED",
    }
    return {
        "request_id": f"{evidence_type}-request",
        "evidence_type": evidence_type,
        "status": status,
        "query": "bounded target query",
        "source_name": "fake-provider",
        "retrieved_at": "2026-09-20T00:00:00+00:00",
        "records": [record] if status in {"SUCCESS", "PARTIAL"} else [],
        "discovered_sources": [],
        "warnings": [],
        "errors": [],
        "artifact_ref": None,
    }


if __name__ == "__main__":
    unittest.main()
