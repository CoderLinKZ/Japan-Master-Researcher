# File: __init__.py
# Author: L1nzhk0
# Purpose: 本文件用于导出 Scholar MCP Server、论文检索查询和 Provider 接口。

from .adapters import (
    AcademicPublicationAdapter,
    OfficialSiteDiscoveryAdapter,
    OfficialSitePublicationAdapter,
    UnavailableOfficialSiteDiscoveryAdapter,
    UnavailableOfficialSitePublicationAdapter,
)
from .brave_discovery import (
    BraveOfficialSiteDiscoveryAdapter,
    BraveSearchClient,
    BraveSearchError,
    BraveSearchResult,
    BraveWebSearchClient,
)
from .composite import (
    CompositePublicationSearchProvider,
    create_default_publication_search_provider,
)
from .models import (
    PublicationAdapterResult,
    RawPublicationCandidate,
    SourceDiscoveryResult,
)
from .official_site import (
    CrawlingOfficialSitePublicationAdapter,
    ParsedOfficialPage,
    parse_official_page,
)
from .openalex import (
    JSONHTTPClient,
    OpenAlexPublicationAdapter,
    OpenAlexRequestError,
    UrllibJSONHTTPClient,
)
from .provider import (
    PublicationSearchProvider,
    PublicationSearchProviderResult,
    PublicationSearchQuery,
)
from .safe_http import (
    HTTPTextResponse,
    RawHTTPResponse,
    SafeHTTPAccessRestrictedError,
    SafeHTTPBlockedError,
    SafeHTTPError,
    SafeHTTPTextClient,
)
from .serpapi_discovery import (
    SerpApiOfficialSiteDiscoveryAdapter,
    SerpApiSearchError,
    SerpApiWebSearchClient,
)
from .server import (
    DISCOVER_OFFICIAL_SOURCES_TOOL_NAME,
    SEARCH_PUBLICATIONS_TOOL_NAME,
    ScholarMCPServer,
    create_scholar_server,
)

__all__ = [
    "DISCOVER_OFFICIAL_SOURCES_TOOL_NAME",
    "SEARCH_PUBLICATIONS_TOOL_NAME",
    "AcademicPublicationAdapter",
    "BraveOfficialSiteDiscoveryAdapter",
    "BraveSearchClient",
    "BraveSearchError",
    "BraveSearchResult",
    "BraveWebSearchClient",
    "CompositePublicationSearchProvider",
    "CrawlingOfficialSitePublicationAdapter",
    "HTTPTextResponse",
    "JSONHTTPClient",
    "OfficialSiteDiscoveryAdapter",
    "OfficialSitePublicationAdapter",
    "OpenAlexPublicationAdapter",
    "OpenAlexRequestError",
    "ParsedOfficialPage",
    "PublicationAdapterResult",
    "PublicationSearchProvider",
    "PublicationSearchProviderResult",
    "PublicationSearchQuery",
    "RawHTTPResponse",
    "RawPublicationCandidate",
    "SafeHTTPAccessRestrictedError",
    "SafeHTTPBlockedError",
    "SafeHTTPError",
    "SafeHTTPTextClient",
    "ScholarMCPServer",
    "SerpApiOfficialSiteDiscoveryAdapter",
    "SerpApiSearchError",
    "SerpApiWebSearchClient",
    "SourceDiscoveryResult",
    "UnavailableOfficialSiteDiscoveryAdapter",
    "UnavailableOfficialSitePublicationAdapter",
    "UrllibJSONHTTPClient",
    "create_default_publication_search_provider",
    "create_scholar_server",
    "parse_official_page",
]
