# File: adapters.py
# Author: L1nzhk0
# Purpose: 本文件用于定义论文索引、官网发现和官网论文读取 Adapter 的统一接口。

from __future__ import annotations

from typing import Protocol

from jmr.domain import DiscoveredSource, RetrievalStatus

from .models import PublicationAdapterResult, SourceDiscoveryResult
from .provider import PublicationSearchQuery


class AcademicPublicationAdapter(Protocol):
    """定义从结构化学术索引检索论文候选的最小接口。"""

    # 返回学术索引 Adapter 的来源名称
    @property
    def source_name(self) -> str: ...

    # 按规范化查询检索学术索引中的论文候选
    def search(self, query: PublicationSearchQuery) -> PublicationAdapterResult: ...


class OfficialSiteDiscoveryAdapter(Protocol):
    """定义在用户未提供官网时发现候选官网的最小接口。"""

    # 尝试发现与目标研究室或教授匹配的官网来源
    def discover(self, query: PublicationSearchQuery) -> SourceDiscoveryResult: ...


class OfficialSitePublicationAdapter(Protocol):
    """定义从已知官网来源读取论文候选的最小接口。"""

    # 从经过选择的官网来源检索论文候选
    def search(
        self,
        query: PublicationSearchQuery,
        sources: list[DiscoveredSource],
    ) -> PublicationAdapterResult: ...


class UnavailableOfficialSiteDiscoveryAdapter:
    """在本次查询没有已记录官网候选时安全跳过官网读取。"""

    # 返回缺少已记录官网候选的可降级结果
    def discover(self, query: PublicationSearchQuery) -> SourceDiscoveryResult:
        return SourceDiscoveryResult(
            status=RetrievalStatus.BLOCKED,
            warnings=[
                "当前查询没有可供抓取的官网候选；可配置 SERPAPI_API_KEY "
                "自动发现官网，或由用户提供官网 URL；已继续使用结构化学术索引检索"
            ],
        )


class UnavailableOfficialSitePublicationAdapter:
    """在尚未配置官网解析能力时安全跳过官网论文读取。"""

    # 返回缺少官网解析服务的可降级结果
    def search(
        self,
        query: PublicationSearchQuery,
        sources: list[DiscoveredSource],
    ) -> PublicationAdapterResult:
        return PublicationAdapterResult(
            status=RetrievalStatus.BLOCKED,
            warnings=[
                "已记录官网来源，但当前尚未配置官网论文页面解析服务；"
                "已继续使用结构化学术索引结果"
            ],
        )
