# File: __init__.py
# Author: L1nzhk0
# Purpose: 本文件用于导出测试专用的 MCP Server 夹具。

from .demo import DemoMCPServer, create_demo_server

__all__ = ["DemoMCPServer", "create_demo_server"]
