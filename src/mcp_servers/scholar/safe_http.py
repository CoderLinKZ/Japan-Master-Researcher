# File: safe_http.py
# Author: L1nzhk0
# Purpose: 本文件用于在访问研究室官网时限制协议、目标地址、重定向、响应大小和超时。

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urljoin, urlparse

DEFAULT_SITE_TIMEOUT_SECONDS = 10.0
DEFAULT_SITE_MAX_RESPONSE_BYTES = 1 * 1024 * 1024
DEFAULT_SITE_MAX_REDIRECTS = 3
ALLOWED_SITE_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
}


class SafeHTTPError(RuntimeError):
    """表示官网请求发生网络、协议或响应解析错误。"""


class SafeHTTPBlockedError(SafeHTTPError):
    """表示官网请求因 SSRF 或资源边界规则被主动阻止。"""


class SafeHTTPAccessRestrictedError(SafeHTTPBlockedError):
    """表示来源要求认证、订阅或因法律策略不可公开读取。"""


@dataclass(frozen=True, slots=True)
class RawHTTPResponse:
    """保存底层固定 IP HTTP Transport 返回的原始响应。"""

    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True, slots=True)
class HTTPTextResponse:
    """保存经过安全检查和文本解码后的官网响应。"""

    url: str
    status: int
    content_type: str
    text: str


class HostResolver(Protocol):
    """定义安全 HTTP Client 所需的主机名解析接口。"""

    # 将主机名解析为一个或多个 IP 地址
    def resolve(self, hostname: str) -> list[str]: ...


class HTTPTransport(Protocol):
    """定义使用已校验固定 IP 发起 GET 请求的接口。"""

    # 向已经过校验的 IP 地址发送单次 HTTP GET 请求
    def get(
        self,
        url: str,
        resolved_ip: str,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> RawHTTPResponse: ...


class SocketHostResolver:
    """使用系统 DNS 将官网主机名解析为去重后的 IP 地址。"""

    # 解析主机名并保持地址首次出现的顺序
    def resolve(self, hostname: str) -> list[str]:
        try:
            records = socket.getaddrinfo(
                hostname,
                None,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise SafeHTTPError(f"无法解析官网主机名：{hostname}") from exc
        result: list[str] = []
        seen: set[str] = set()
        for record in records:
            address = record[4][0]
            if address not in seen:
                seen.add(address)
                result.append(address)
        if not result:
            raise SafeHTTPError(f"官网主机名没有可用地址：{hostname}")
        return result


class PinnedIPHTTPTransport:
    """将请求连接固定到预先校验的 IP, 并保留原主机名用于 Host 和 TLS。"""

    # 向固定 IP 发送请求并限制读取的响应字节数
    def get(
        self,
        url: str,
        resolved_ip: str,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> RawHTTPResponse:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname is None:
            raise SafeHTTPError("官网 URL 缺少主机名")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        headers = {
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8",
            "Host": parsed.netloc,
            "User-Agent": "Japan-Master-Researcher/0.1",
        }
        if parsed.scheme == "https":
            connection: http.client.HTTPConnection = _PinnedHTTPSConnection(
                hostname=hostname,
                resolved_ip=resolved_ip,
                port=port,
                timeout=timeout_seconds,
            )
        else:
            connection = http.client.HTTPConnection(
                resolved_ip,
                port=port,
                timeout=timeout_seconds,
            )
        try:
            connection.request("GET", path, headers=headers)
            response = connection.getresponse()
            body = response.read(max_response_bytes + 1)
            response_headers = {
                key.casefold(): value for key, value in response.getheaders()
            }
        except (OSError, http.client.HTTPException) as exc:
            raise SafeHTTPError(f"官网请求失败：{url}") from exc
        finally:
            connection.close()
        if len(body) > max_response_bytes:
            raise SafeHTTPBlockedError(f"官网响应超过大小限制：{url}")
        return RawHTTPResponse(
            status=response.status,
            headers=response_headers,
            body=body,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """建立到固定 IP 的 TLS 连接并使用原主机名校验证书。"""

    # 初始化固定 IP HTTPS 连接
    def __init__(
        self,
        *,
        hostname: str,
        resolved_ip: str,
        port: int,
        timeout: float,
    ) -> None:
        super().__init__(
            hostname,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._resolved_ip = resolved_ip

    # 连接固定 IP 并以原始域名执行 TLS 握手
    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._resolved_ip, self.port),
            self.timeout,
            self.source_address,
        )
        self.sock = self._context.wrap_socket(
            raw_socket,
            server_hostname=self.host,
        )


class SafeHTTPTextClient:
    """在每次请求和重定向前校验目标并返回受限文本内容。"""

    # 初始化官网 HTTP Client 的安全边界和可替换依赖
    def __init__(
        self,
        resolver: HostResolver | None = None,
        transport: HTTPTransport | None = None,
        timeout_seconds: float = DEFAULT_SITE_TIMEOUT_SECONDS,
        max_response_bytes: int = DEFAULT_SITE_MAX_RESPONSE_BYTES,
        max_redirects: int = DEFAULT_SITE_MAX_REDIRECTS,
    ) -> None:
        self._resolver = resolver or SocketHostResolver()
        self._transport = transport or PinnedIPHTTPTransport()
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._max_redirects = max_redirects

    # 获取官网文本并对每一次重定向重新执行目标校验
    def fetch(self, url: str) -> HTTPTextResponse:
        current_url = url
        for redirect_count in range(self._max_redirects + 1):
            hostname, addresses = self._validate_target(current_url)
            raw_response = self._request_available_address(
                current_url,
                addresses,
            )
            if raw_response.status in {301, 302, 303, 307, 308}:
                location = raw_response.headers.get("location")
                if not location:
                    raise SafeHTTPError(f"官网重定向缺少 Location：{current_url}")
                if redirect_count >= self._max_redirects:
                    raise SafeHTTPBlockedError("官网重定向次数超过限制")
                current_url = urljoin(current_url, location)
                continue
            if raw_response.status in {401, 402, 403, 407, 451}:
                raise SafeHTTPAccessRestrictedError(
                    f"官网内容不是无需认证即可读取的公开来源：{hostname}"
                )
            if not 200 <= raw_response.status < 300:
                raise SafeHTTPError(f"官网返回 HTTP {raw_response.status}：{hostname}")
            content_type, charset = _parse_content_type(
                raw_response.headers.get("content-type", "")
            )
            if content_type not in ALLOWED_SITE_CONTENT_TYPES:
                raise SafeHTTPBlockedError(
                    f"官网返回了不允许的内容类型：{content_type or '(missing)'}"
                )
            try:
                text = raw_response.body.decode(charset, errors="replace")
            except LookupError as exc:
                raise SafeHTTPError(f"官网声明了未知字符编码：{charset}") from exc
            return HTTPTextResponse(
                url=current_url,
                status=raw_response.status,
                content_type=content_type,
                text=text,
            )
        raise SafeHTTPBlockedError("官网重定向次数超过限制")

    # 校验 URL、端口和解析出的全部 IP 均属于公网
    def _validate_target(self, url: str) -> tuple[str, list[str]]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise SafeHTTPBlockedError("官网只允许 HTTP 或 HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise SafeHTTPBlockedError("官网 URL 不允许包含用户凭据")
        hostname = parsed.hostname
        if hostname is None:
            raise SafeHTTPBlockedError("官网 URL 缺少主机名")
        normalized_hostname = hostname.rstrip(".").casefold()
        if (
            normalized_hostname == "localhost"
            or normalized_hostname.endswith(".localhost")
            or normalized_hostname.endswith(".local")
            or normalized_hostname.endswith(".internal")
        ):
            raise SafeHTTPBlockedError("官网 URL 指向本地或内部主机")
        try:
            port = parsed.port
        except ValueError as exc:
            raise SafeHTTPBlockedError("官网 URL 端口无效") from exc
        if port is not None and port not in {80, 443}:
            raise SafeHTTPBlockedError("官网 URL 只允许 80 或 443 端口")
        addresses = self._resolver.resolve(normalized_hostname)
        for address in addresses:
            try:
                parsed_address = ipaddress.ip_address(address)
            except ValueError as exc:
                raise SafeHTTPBlockedError(f"官网主机解析出无效 IP：{address}") from exc
            if not parsed_address.is_global:
                raise SafeHTTPBlockedError(f"官网主机解析到非公网 IP：{address}")
        return normalized_hostname, addresses

    # 依次尝试经过校验的公网 IP 并返回第一个成功响应
    def _request_available_address(
        self,
        url: str,
        addresses: list[str],
    ) -> RawHTTPResponse:
        last_error: SafeHTTPError | None = None
        for address in addresses:
            try:
                return self._transport.get(
                    url,
                    address,
                    self._timeout_seconds,
                    self._max_response_bytes,
                )
            except SafeHTTPBlockedError:
                raise
            except SafeHTTPError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise SafeHTTPError("官网主机没有可请求的公网地址")


# 解析 Content-Type 头部的媒体类型与字符编码
def _parse_content_type(value: str) -> tuple[str, str]:
    parts = [part.strip() for part in value.split(";")]
    content_type = parts[0].casefold() if parts else ""
    charset = "utf-8"
    for part in parts[1:]:
        if part.casefold().startswith("charset="):
            charset = part.split("=", 1)[1].strip(' "') or "utf-8"
    return content_type, charset
