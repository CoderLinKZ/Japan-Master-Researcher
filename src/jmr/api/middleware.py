"""Pure ASGI middleware for bounding HTTP request bodies."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

DEFAULT_MAX_BODY_BYTES = 128 * 1024
UPLOAD_MAX_BODY_BYTES = 2 * 1024 * 1024 + 32 * 1024
DEFAULT_BODY_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
)


class RequestBodyLimitMiddleware:
    """Reject oversized request bodies before an application parses them.

    A declared oversized ``Content-Length`` is rejected without reading the
    body.  Otherwise, body-bearing requests are buffered up to the configured
    limit and replayed to the downstream ASGI application as one request
    event. Later disconnect events still come from the original connection.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        body_methods: Iterable[str] = DEFAULT_BODY_METHODS,
    ) -> None:
        if (
            isinstance(max_body_bytes, bool)
            or not isinstance(max_body_bytes, int)
            or max_body_bytes < 1
        ):
            raise ValueError("max_body_bytes must be a positive integer")
        methods = frozenset(_method(method) for method in body_methods)
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.body_methods = methods

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http" or str(scope.get("method", "")).upper() not in (
            self.body_methods
        ):
            await self.app(scope, receive, send)
            return

        limit = (
            UPLOAD_MAX_BODY_BYTES
            if scope.get("path") == "/api/v1/memories/upload"
            else self.max_body_bytes
        )
        if _declared_length_exceeds(
            scope.get("headers", ()),
            limit=limit,
        ):
            await self._reject(send, limit=limit)
            return

        buffered = bytearray()
        terminal_message: Message | None = None
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "http.disconnect":
                terminal_message = message
                break
            if message_type != "http.request":
                terminal_message = message
                break
            body = message.get("body", b"")
            if not isinstance(body, bytes):
                body = bytes(body)
            if len(buffered) + len(body) > limit:
                await self._reject(send, limit=limit)
                return
            buffered.extend(body)
            if not message.get("more_body", False):
                break

        delivered_body = False
        delivered_terminal = False

        async def replay_receive() -> Message:
            nonlocal delivered_body, delivered_terminal
            if not delivered_body:
                delivered_body = True
                return {
                    "type": "http.request",
                    "body": bytes(buffered),
                    "more_body": terminal_message is not None,
                }
            if terminal_message is not None and not delivered_terminal:
                delivered_terminal = True
                return terminal_message
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _reject(self, send: Send, *, limit: int) -> None:
        body = json.dumps(
            {
                "error": {
                    "code": "PAYLOAD_TOO_LARGE",
                    "message": f"Request body exceeds {limit} bytes",
                }
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )


def _method(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("body_methods must contain non-empty strings")
    return value.strip().upper()


def _declared_length_exceeds(
    headers: Iterable[tuple[bytes, bytes]],
    *,
    limit: int,
) -> bool:
    """Return true when any syntactically valid declared length is too large."""

    for raw_name, raw_value in headers:
        if raw_name.lower() != b"content-length":
            continue
        for candidate in raw_value.split(b","):
            try:
                declared = int(candidate.strip().decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                continue
            if declared > limit:
                return True
    return False


__all__ = [
    "DEFAULT_BODY_METHODS",
    "DEFAULT_MAX_BODY_BYTES",
    "RequestBodyLimitMiddleware",
]
