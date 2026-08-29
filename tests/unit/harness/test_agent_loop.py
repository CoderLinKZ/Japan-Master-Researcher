# File: test_agent_loop.py
# Author: L1nzhk0
# Purpose: This file verifies the stopping, tool execution, and error-recovery boundaries of the core agent loop.

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from harness.agent_loop import AgentLoop  # noqa: E402


def text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=text)],
    )


def tool_response(name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        # The loop intentionally follows the actual content blocks, even if
        # a provider reports an inconsistent stop reason.
        stop_reason="end_turn",
        content=[
            SimpleNamespace(
                type="tool_use",
                id="tool_1",
                name=name,
                input=arguments,
            )
        ],
    )


class FakeModelCaller:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **request: Any) -> SimpleNamespace:
        self.calls.append(request)
        if not self._responses:
            raise AssertionError("AgentLoop requested an unexpected model turn")
        return self._responses.pop(0)


class AgentLoopTests(unittest.TestCase):
    def test_stops_when_response_has_no_tool_call(self) -> None:
        caller = FakeModelCaller([text_response("done")])
        messages = [{"role": "user", "content": "hello"}]

        answer = AgentLoop(caller).run(messages, "system")

        self.assertEqual(answer, "done")
        self.assertEqual(len(caller.calls), 1)
        self.assertEqual(messages[-1]["role"], "assistant")

    def test_executes_tool_and_feeds_result_back_to_model(self) -> None:
        caller = FakeModelCaller(
            [tool_response("echo", {"value": "hello"}), text_response("complete")]
        )
        pool_calls = 0

        def assemble_tool_pool():
            nonlocal pool_calls
            pool_calls += 1
            tools = [{
                "name": "echo",
                "description": "Return the provided value.",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            }]
            return tools, {"echo": lambda value: value}

        messages = [{"role": "user", "content": "use echo"}]
        loop = AgentLoop(caller, assemble_tool_pool)

        answer = loop.run(messages, "system")

        self.assertEqual(answer, "complete")
        self.assertEqual(len(caller.calls), 2)
        self.assertEqual(pool_calls, 2)
        tool_result = messages[2]["content"][0]
        self.assertEqual(tool_result["tool_use_id"], "tool_1")
        self.assertEqual(tool_result["content"], "hello")
        self.assertNotIn("is_error", tool_result)

    def test_returns_tool_exception_to_model_as_error_result(self) -> None:
        caller = FakeModelCaller(
            [tool_response("broken", {}), text_response("recovered")]
        )

        def broken():
            raise ValueError("test failure")

        loop = AgentLoop(caller, lambda: ([], {"broken": broken}))
        messages = [{"role": "user", "content": "use broken"}]

        answer = loop.run(messages, "system")

        self.assertEqual(answer, "recovered")
        tool_result = messages[2]["content"][0]
        self.assertTrue(tool_result["is_error"])
        self.assertIn("ValueError: test failure", tool_result["content"])

    def test_does_not_append_empty_tool_result_turn(self) -> None:
        caller = FakeModelCaller(
            [SimpleNamespace(stop_reason="tool_use", content=[])]
        )
        messages = [{"role": "user", "content": "hello"}]

        answer = AgentLoop(caller).run(messages, "system")

        self.assertEqual(answer, "")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
