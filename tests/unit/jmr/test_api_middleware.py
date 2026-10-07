"""Contract tests for the pure ASGI request-body limiter."""

from __future__ import annotations

import json
import unittest
from collections import deque
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from jmr.api.middleware import RequestBodyLimitMiddleware


class _RecordingApp:
    def __init__(self, *, read_after_body: bool = False) -> None:
        self.calls = 0
        self.messages: list[Message] = []
        self.read_after_body = read_after_body

    async def __call__(self, _: Scope, receive: Receive, send: Send) -> None:
        self.calls += 1
        while True:
            message = await receive()
            self.messages.append(message)
            if message["type"] == "http.disconnect" or not message.get(
                "more_body", False
            ):
                break
        if self.read_after_body:
            self.messages.append(await receive())
        await send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [],
            }
        )
        await send({"type": "http.response.body", "body": b""})


class RequestBodyLimitMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_declared_oversized_body_is_rejected_without_receiving_it(
        self,
    ) -> None:
        downstream = _RecordingApp()
        middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=10)

        sent, receive_calls = await _invoke(
            middleware,
            headers=[(b"content-length", b"11")],
            messages=[
                {"type": "http.request", "body": b"not-consumed", "more_body": False}
            ],
        )

        self.assertEqual(downstream.calls, 0)
        self.assertEqual(receive_calls, 0)
        self.assertEqual(_status(sent), 413)
        self.assertEqual(_response_json(sent), _expected_error(10))

    async def test_chunked_body_is_rejected_when_cumulative_size_exceeds_limit(
        self,
    ) -> None:
        downstream = _RecordingApp()
        middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=10)

        sent, receive_calls = await _invoke(
            middleware,
            headers=[(b"transfer-encoding", b"chunked")],
            messages=[
                {"type": "http.request", "body": b"123456", "more_body": True},
                {"type": "http.request", "body": b"78901", "more_body": True},
                {"type": "http.request", "body": b"ignored", "more_body": False},
            ],
        )

        self.assertEqual(downstream.calls, 0)
        self.assertEqual(receive_calls, 2)
        self.assertEqual(_status(sent), 413)
        self.assertEqual(_response_json(sent), _expected_error(10))

    async def test_body_within_limit_is_replayed_and_disconnect_remains_visible(
        self,
    ) -> None:
        downstream = _RecordingApp(read_after_body=True)
        middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=10)
        original = [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ]

        sent, receive_calls = await _invoke(
            middleware,
            headers=[(b"content-length", b"6")],
            messages=original,
        )

        self.assertEqual(receive_calls, 3)
        self.assertEqual(downstream.calls, 1)
        self.assertEqual(
            downstream.messages,
            [
                {"type": "http.request", "body": b"abcdef", "more_body": False},
                {"type": "http.disconnect"},
            ],
        )
        self.assertEqual(_status(sent), 204)

    async def test_get_body_is_limited_too(self) -> None:
        downstream = _RecordingApp()
        middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=3)
        original = [{"type": "http.request", "body": b"oversized", "more_body": False}]

        sent, receive_calls = await _invoke(
            middleware,
            method="GET",
            headers=[(b"content-length", b"9")],
            messages=original,
        )

        self.assertEqual(receive_calls, 0)
        self.assertEqual(downstream.calls, 0)
        self.assertEqual(downstream.messages, [])
        self.assertEqual(_status(sent), 413)


class FastAPIRequestBodyLimitTests(unittest.TestCase):
    def test_json_parsing_and_body_free_health_check(self) -> None:
        app = FastAPI()

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/echo")
        def echo(payload: dict[str, str]) -> dict[str, str]:
            return payload

        app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=24)
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            accepted = client.post("/echo", json={"item": "small"})
            self.assertEqual(accepted.status_code, 200)
            self.assertEqual(accepted.json(), {"item": "small"})
            rejected = client.post("/echo", json={"item": "x" * 25})
            self.assertEqual(rejected.status_code, 413)
            self.assertEqual(rejected.json(), _expected_error(24))


async def _invoke(
    app: Any,
    *,
    headers: list[tuple[bytes, bytes]],
    messages: list[Message],
    method: str = "POST",
) -> tuple[list[Message], int]:
    queued = deque(messages)
    sent: list[Message] = []
    receive_calls = 0

    async def receive() -> Message:
        nonlocal receive_calls
        receive_calls += 1
        if queued:
            return queued.popleft()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    return sent, receive_calls


def _status(messages: list[Message]) -> int:
    start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    return int(start["status"])


def _response_json(messages: list[Message]) -> dict[str, Any]:
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return json.loads(body)


def _expected_error(limit: int) -> dict[str, Any]:
    return {
        "error": {
            "code": "PAYLOAD_TOO_LARGE",
            "message": f"Request body exceeds {limit} bytes",
        }
    }


if __name__ == "__main__":
    unittest.main()
