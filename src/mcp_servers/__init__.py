# File: __init__.py
# Author: L1nzhk0
# Purpose: 本文件用于导出项目正式提供的 MCP Server 工厂。

from .kaken import create_kaken_server
from .scholar import create_scholar_server

__all__ = ["create_kaken_server", "create_scholar_server"]
