# File: main.py
# Author: L1nzhk0
# Purpose: This file provides the application entry point and receives user input for the main agent.

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.main import MainAgent
from harness.agent_loop import AgentLoop, empty_tool_pool

# 定义系统提示词文件路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "研究室调查提示词.md"


# 加载系统提示词
def load_system_prompt() -> str:
    """Load the current main-agent prompt from the project document."""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


# 注册 Anthropic 模型调用函数
def create_anthropic_caller() -> Callable[..., Any]:
    """Create the provider-specific callable consumed by AgentLoop."""
    try:
        from anthropic import Anthropic
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependencies. Run: pip install -r requirements.txt"
        ) from exc

    load_dotenv(override=True)

    model = os.getenv("MODEL_ID")
    if not model:
        raise RuntimeError("MODEL_ID is missing; configure it in .env")

    base_url = os.getenv("ANTHROPIC_BASE_URL")
    client = Anthropic(base_url=base_url) if base_url else Anthropic()

    # 调用模型并返回响应
    def call_llm(
        *,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
    ) -> Any:
        request: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": 8_000,
        }
        if tools:
            request["tools"] = tools
        return client.messages.create(**request)

    return call_llm


# 构建主 Agent 实例
def build_agent() -> MainAgent:
    """Assemble the first runnable main agent and its harness dependencies."""
    agent_loop = AgentLoop(
        call_llm=create_anthropic_caller(),
        assemble_tool_pool=empty_tool_pool,
    )
    return MainAgent(
        agent_loop=agent_loop,
        system_prompt=load_system_prompt(),
    )


# 启动命令行交互程序
def main() -> int:
    """Run the command-line conversation loop."""
    try:
        agent = build_agent()
    except (OSError, RuntimeError) as exc:
        print(f"Startup error: {exc}")
        return 1

    print("Japan Master Researcher")
    print("Type q or exit to quit.\n")

    while True:
        try:
            user_input = input("JMR > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.strip().lower() in {"q", "exit"}:
            break
        if not user_input.strip():
            continue

        try:
            answer = agent.run(user_input)
        except Exception as exc:
            print(f"Agent error: {type(exc).__name__}: {exc}")
            continue

        print(answer if answer else "(The model returned no text.)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
