# File: demo.py
# Author: L1nzhk0
# Purpose: 本文件用于提供仅在自动化测试中验证 MCP 动态连接流程的 Demo Server。

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class DemoMCPServer:
    """提供一个无副作用的 echo 工具作为 MCP 测试夹具。"""

    # 返回测试 Demo Server 提供的工具定义
    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "echo",
                "description": "Return the provided text unchanged.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            }
        ]

    # 执行测试 Demo Server 中的指定工具
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name != "echo":
            raise ValueError(f"Unknown demo tool '{name}'")

        text = arguments.get("text")
        if not isinstance(text, str):
            raise TypeError("echo.text must be a string")
        return text


# 创建一个仅供测试使用的进程内 Demo MCP Server
def create_demo_server() -> DemoMCPServer:
    return DemoMCPServer()
