# File: __init__.py
# Author: L1nzhk0
# Purpose: 本文件用于导出 KAKEN MCP Server、查询模型和官方数据源实现。

from .nii import (
    KAKEN_API_URL,
    KakenRequestError,
    NIIKakenSearchProvider,
    UrllibXMLHTTPClient,
    XMLHTTPClient,
)
from .provider import (
    KakenSearchProvider,
    KakenSearchProviderResult,
    KakenSearchQuery,
)
from .server import (
    SEARCH_PROJECTS_TOOL_NAME,
    KakenMCPServer,
    create_kaken_server,
)

__all__ = [
    "KAKEN_API_URL",
    "SEARCH_PROJECTS_TOOL_NAME",
    "KakenMCPServer",
    "KakenRequestError",
    "KakenSearchProvider",
    "KakenSearchProviderResult",
    "KakenSearchQuery",
    "NIIKakenSearchProvider",
    "UrllibXMLHTTPClient",
    "XMLHTTPClient",
    "create_kaken_server",
]
