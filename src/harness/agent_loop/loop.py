# File: loop.py
# Author: L1nzhk0
# Purpose: This file implements the core agent loop for model calls, tool execution, and tool-result feedback.

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any


Message = dict[str, Any]
ToolHandler = Callable[..., Any]
ModelCaller = Callable[..., Any]
ToolPoolAssembler = Callable[
    [], tuple[list[dict[str, Any]], Mapping[str, ToolHandler]]
]


class AgentLoopLimitError(RuntimeError):
    """Raised when one user turn exceeds the configured model-call limit."""


# 创建空工具池
def empty_tool_pool() -> tuple[list[dict[str, Any]], Mapping[str, ToolHandler]]:
    """Return an empty tool pool for the first runnable version of the agent."""
    return [], {}


class AgentLoop:
    """Run model and tool turns until the model stops requesting tools."""

    # 初始化 Agent 主循环
    def __init__(
        self,
        call_llm: ModelCaller,
        assemble_tool_pool: ToolPoolAssembler | None = None,
        max_model_calls: int | None = 100,
    ) -> None:
        if max_model_calls is not None and max_model_calls < 1:
            raise ValueError("max_model_calls must be positive or None")

        self._call_llm = call_llm
        self._assemble_tool_pool = assemble_tool_pool or empty_tool_pool
        self._max_model_calls = max_model_calls

    # 执行单轮模型与工具调用循环
    def run(self, messages: list[Message], system_prompt: str) -> str:
        """Execute one user turn and return the model's final text response."""
        model_calls = 0

        while True:
            # 记录当前用户请求中的模型调用次数，防止模型与工具无限循环
            model_calls += 1
            if (
                self._max_model_calls is not None
                and model_calls > self._max_model_calls
            ):
                raise AgentLoopLimitError(
                    "Agent loop exceeded "
                    f"{self._max_model_calls} model calls in one user turn"
                )

            # Reassemble on every round so newly connected MCP tools can appear
            # in the immediately following model call.
            tools, handlers = self._assemble_tool_pool()
            response = self._call_llm(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
            )

            content = _response_content(response)
            messages.append({"role": "assistant", "content": content})

            # Actual tool-use blocks, rather than stop_reason, control whether
            # the loop continues.
            tool_calls = [
                block for block in content if _block_field(block, "type") == "tool_use"
            ]
            if not tool_calls:
                return _extract_text(content)

            tool_results = [
                _dispatch_tool_call(tool_call, handlers)
                for tool_call in tool_calls
            ]
            messages.append({"role": "user", "content": tool_results})


# 提取模型响应中的内容块
def _response_content(response: Any) -> list[Any]:
    """Read content blocks from an SDK response object or a plain dictionary."""
    if isinstance(response, Mapping):
        content = response.get("content", [])
    else:
        content = getattr(response, "content", [])

    if content is None:
        return []
    return list(content)


# 读取内容块中的指定字段
def _block_field(block: Any, name: str, default: Any = None) -> Any:
    """Read one field from either an SDK content block or a dictionary."""
    if isinstance(block, Mapping):
        return block.get(name, default)
    return getattr(block, name, default)


# 分发并执行单个工具调用
def _dispatch_tool_call(
    tool_call: Any,
    handlers: Mapping[str, ToolHandler],
) -> dict[str, Any]:
    """Execute one tool call and always convert its outcome to a tool result."""
    tool_use_id = _block_field(tool_call, "id")
    tool_name = _block_field(tool_call, "name")
    tool_input = _block_field(tool_call, "input", {})

    if not tool_use_id:
        raise ValueError("A tool_use block is missing its id")

    if not isinstance(tool_name, str) or not tool_name:
        return _tool_result(
            tool_use_id,
            "Error: A tool_use block is missing its tool name",
            is_error=True,
        )

    handler = handlers.get(tool_name)
    if handler is None:
        return _tool_result(
            tool_use_id,
            f"Error: Unknown tool '{tool_name}'",
            is_error=True,
        )

    if not isinstance(tool_input, Mapping):
        return _tool_result(
            tool_use_id,
            f"Error: Input for tool '{tool_name}' must be an object",
            is_error=True,
        )

    try:
        output = handler(**dict(tool_input))
    except Exception as exc:  # The model should see tool errors and may recover.
        return _tool_result(
            tool_use_id,
            f"Error: {type(exc).__name__}: {exc}",
            is_error=True,
        )

    return _tool_result(tool_use_id, _stringify_tool_output(output))


# 构造工具调用结果
def _tool_result(
    tool_use_id: str,
    content: str,
    *,
    is_error: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        result["is_error"] = True
    return result


# 将工具输出转换为字符串
def _stringify_tool_output(output: Any) -> str:
    if output is None:
        return "(no output)"
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(output)


# 提取模型响应中的文本内容
def _extract_text(content: list[Any]) -> str:
    text_blocks = [
        str(text)
        for block in content
        if _block_field(block, "type") == "text"
        and (text := _block_field(block, "text"))
    ]
    return "\n".join(text_blocks)
