# File: test_main_agent.py
# Author: L1nzhk0
# Purpose: This file verifies conversation ownership and failure rollback in the main agent.

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from agent.main import MainAgent, WorkflowState  # noqa: E402


class StubAgentLoop:
    def __init__(self, answer: str = "done", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls = []

    def run(self, messages, system_prompt):
        self.calls.append((list(messages), system_prompt))
        if self.error is not None:
            raise self.error
        return self.answer


class MainAgentTests(unittest.TestCase):
    def test_adds_user_message_and_delegates_to_agent_loop(self) -> None:
        loop = StubAgentLoop(answer="answer")
        agent = MainAgent(loop, "system")

        answer = agent.run("question")

        self.assertEqual(answer, "answer")
        self.assertEqual(agent.messages, [{"role": "user", "content": "question"}])
        self.assertEqual(agent.state, WorkflowState.VALIDATING_APPLICATION_INPUTS)
        self.assertEqual(loop.calls[0][1], "system")

    def test_rolls_back_partial_history_when_turn_fails(self) -> None:
        loop = StubAgentLoop(error=RuntimeError("model unavailable"))
        agent = MainAgent(loop, "system")
        agent.messages.append({"role": "user", "content": "previous"})

        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            agent.run("new question")

        self.assertEqual(
            agent.messages,
            [{"role": "user", "content": "previous"}],
        )


if __name__ == "__main__":
    unittest.main()
