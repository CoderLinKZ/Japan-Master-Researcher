# File: test_safe_http.py
# Author: L1nzhk0
# Purpose: 本文件用于验证官网 HTTP Client 的公网地址、重定向和内容类型安全边界。

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(SRC_ROOT))

from mcp_servers.scholar import (  # noqa: E402
    RawHTTPResponse,
    SafeHTTPBlockedError,
    SafeHTTPTextClient,
)


class FakeResolver:
    """按主机名返回固定 IP 地址。"""

    # 初始化固定 DNS 映射
    def __init__(self, records: dict[str, list[str]]) -> None:
        self._records = records

    # 返回指定主机的固定 IP 地址
    def resolve(self, hostname: str) -> list[str]:
        return list(self._records[hostname])


class FakeTransport:
    """按顺序返回固定 HTTP 响应并记录连接 IP。"""

    # 初始化固定 HTTP 响应队列
    def __init__(self, responses: list[RawHTTPResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    # 记录请求并返回下一条固定响应
    def get(
        self,
        url: str,
        resolved_ip: str,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> RawHTTPResponse:
        self.calls.append((url, resolved_ip))
        return self._responses.pop(0)


class SafeHTTPTextClientTests(unittest.TestCase):
    # 验证公网 HTML 能够通过固定 IP Transport 返回文本
    def test_fetches_public_html(self) -> None:
        transport = FakeTransport(
            [
                RawHTTPResponse(
                    status=200,
                    headers={"content-type": "text/html; charset=utf-8"},
                    body="研究室".encode(),
                )
            ]
        )
        client = SafeHTTPTextClient(
            resolver=FakeResolver({"lab.example": ["8.8.8.8"]}),
            transport=transport,
        )

        response = client.fetch("https://lab.example/publications")

        self.assertEqual(response.text, "研究室")
        self.assertEqual(
            transport.calls,
            [("https://lab.example/publications", "8.8.8.8")],
        )

    # 验证解析到私网地址的官网会在发起请求前被阻止
    def test_blocks_private_ip_before_transport(self) -> None:
        transport = FakeTransport([])
        client = SafeHTTPTextClient(
            resolver=FakeResolver({"internal.example": ["192.168.1.5"]}),
            transport=transport,
        )

        with self.assertRaisesRegex(SafeHTTPBlockedError, "非公网 IP"):
            client.fetch("https://internal.example/")

        self.assertEqual(transport.calls, [])

    # 验证重定向目标会重新解析并阻止跳转到本机地址
    def test_revalidates_redirect_target(self) -> None:
        transport = FakeTransport(
            [
                RawHTTPResponse(
                    status=302,
                    headers={"location": "http://localhost/private"},
                    body=b"",
                )
            ]
        )
        client = SafeHTTPTextClient(
            resolver=FakeResolver({"lab.example": ["8.8.8.8"]}),
            transport=transport,
        )

        with self.assertRaisesRegex(SafeHTTPBlockedError, "本地或内部"):
            client.fetch("https://lab.example/")

        self.assertEqual(len(transport.calls), 1)

    # 验证非网页内容类型不会进入 HTML 解析层
    def test_blocks_non_text_content_type(self) -> None:
        transport = FakeTransport(
            [
                RawHTTPResponse(
                    status=200,
                    headers={"content-type": "application/pdf"},
                    body=b"pdf",
                )
            ]
        )
        client = SafeHTTPTextClient(
            resolver=FakeResolver({"lab.example": ["8.8.8.8"]}),
            transport=transport,
        )

        with self.assertRaisesRegex(SafeHTTPBlockedError, "内容类型"):
            client.fetch("https://lab.example/paper.pdf")


if __name__ == "__main__":
    unittest.main()
