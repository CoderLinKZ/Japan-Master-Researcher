"""Provider-neutral contract tests for the production Anthropic adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.runtime import (  # noqa: E402
    AnthropicNodeModel,
    create_anthropic_model_registry,
)


class _Messages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)


class _Client:
    def __init__(self, responses):
        self.messages = _Messages(responses)


def _response(*blocks):
    return SimpleNamespace(content=list(blocks))


def _tool_use(name, payload, *, call_id="call-1"):
    return SimpleNamespace(
        type="tool_use",
        id=call_id,
        name=name,
        input=payload,
    )


class AnthropicNodeModelTests(unittest.TestCase):
    def test_structured_call_forces_and_returns_the_schema_tool(self) -> None:
        client = _Client(
            [_response(_tool_use("submit_structured_result", {"answer": 1}))]
        )
        model = AnthropicNodeModel(client, model_id="test-model")

        result = model.invoke_structured(
            messages=[{"role": "user", "content": "question"}],
            system_prompt="Return structured data.",
            output_schema={"type": "object"},
        )

        self.assertEqual(result, {"answer": 1})
        request = client.messages.requests[0]
        self.assertEqual(
            request["tool_choice"],
            {"type": "tool", "name": "submit_structured_result"},
        )

    def test_react_call_translates_mcp_tool_request(self) -> None:
        client = _Client(
            [_response(_tool_use("mcp__kaken__search_projects", {"query": "q"}))]
        )
        model = AnthropicNodeModel(client, model_id="test-model")

        turn = model.invoke_tool_step(
            messages=[{"role": "user", "content": "retrieve"}],
            system_prompt="Use tools.",
            tools=[
                {
                    "name": "mcp__kaken__search_projects",
                    "description": "search",
                    "input_schema": {"type": "object"},
                }
            ],
            output_schema={"type": "object"},
        )

        self.assertEqual(turn.tool_calls[0].name, "mcp__kaken__search_projects")
        self.assertEqual(turn.tool_calls[0].arguments, {"query": "q"})
        self.assertIsNone(turn.final_result)

    def test_host_controlled_tool_discards_dummy_arguments(self) -> None:
        client = _Client(
            [_response(_tool_use("mcp__kaken__search_projects", {"_noargs": ""}))]
        )
        model = AnthropicNodeModel(client, model_id="test-model")

        turn = model.invoke_tool_step(
            messages=[{"role": "user", "content": "retrieve"}],
            system_prompt="Use tools.",
            tools=[
                {
                    "name": "mcp__kaken__search_projects",
                    "description": "host supplies every search argument",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            ],
            output_schema={"type": "object"},
        )

        self.assertEqual(turn.tool_calls[0].arguments, {})

    def test_react_call_translates_final_result(self) -> None:
        payload = {"status": "NO_RESULT"}
        client = _Client([_response(_tool_use("submit_retrieval_result", payload))])
        model = AnthropicNodeModel(client, model_id="test-model")

        turn = model.invoke_tool_step(
            messages=[{"role": "user", "content": "retrieve"}],
            system_prompt="Use tools.",
            tools=[],
            output_schema={"type": "object"},
        )

        self.assertEqual(turn.final_result, payload)
        self.assertEqual(turn.tool_calls, ())

    def test_final_result_cannot_be_mixed_with_an_mcp_request(self) -> None:
        client = _Client(
            [
                _response(
                    _tool_use("submit_retrieval_result", {"status": "SUCCESS"}),
                    _tool_use("mcp__kaken__search_projects", {}),
                )
            ]
        )
        model = AnthropicNodeModel(client, model_id="test-model")

        with self.assertRaisesRegex(ValueError, "mixed final output"):
            model.invoke_tool_step(
                messages=[{"role": "user", "content": "retrieve"}],
                system_prompt="Use tools.",
                tools=[],
                output_schema={"type": "object"},
            )

    def test_registry_factory_reports_all_missing_configuration(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "ANTHROPIC_API_KEY, MODEL_ID",
        ):
            create_anthropic_model_registry(environment={})


if __name__ == "__main__":
    unittest.main()
