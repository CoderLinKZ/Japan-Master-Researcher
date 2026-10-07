"""Offline security gate spanning URL policy and structured logging."""

from __future__ import annotations

import io
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.runtime import (  # noqa: E402
    JMRRuntimeContext,
    JsonLineEventSink,
    emit_runtime_event,
)
from mcp_servers.scholar import (  # noqa: E402
    RawHTTPResponse,
    SafeHTTPAccessRestrictedError,
    SafeHTTPBlockedError,
    SafeHTTPTextClient,
)


class Resolver:
    def __init__(self, records):
        self.records = records

    def resolve(self, hostname):
        return list(self.records[hostname])


class Transport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, resolved_ip, timeout_seconds, max_response_bytes):
        self.calls.append((url, resolved_ip, timeout_seconds, max_response_bytes))
        return self.responses.pop(0)


class Clock:
    def now(self):
        return datetime(2026, 9, 21, tzinfo=UTC)


class SecurityBoundaryTests(unittest.TestCase):
    def test_credentials_illegal_ports_and_mixed_private_dns_are_blocked(self):
        cases = (
            "https://user:password@lab.example/",
            "https://lab.example:8443/",
            "https://mixed.example/",
        )
        resolver = Resolver(
            {
                "lab.example": ["8.8.8.8"],
                "mixed.example": ["8.8.8.8", "127.0.0.1"],
            }
        )
        transport = Transport([])
        client = SafeHTTPTextClient(resolver=resolver, transport=transport)
        for url in cases:
            with self.subTest(url=url), self.assertRaises(SafeHTTPBlockedError):
                client.fetch(url)
        self.assertEqual(transport.calls, [])

    def test_redirect_is_revalidated_and_transport_stays_pinned_to_checked_ip(self):
        transport = Transport(
            [
                RawHTTPResponse(302, {"location": "https://internal.example/"}, b""),
            ]
        )
        client = SafeHTTPTextClient(
            resolver=Resolver(
                {"lab.example": ["8.8.8.8"], "internal.example": ["10.0.0.8"]}
            ),
            transport=transport,
        )
        with self.assertRaises(SafeHTTPBlockedError):
            client.fetch("https://lab.example/")
        self.assertEqual(transport.calls[0][1], "8.8.8.8")
        self.assertEqual(len(transport.calls), 1)

    def test_authentication_and_paywall_statuses_are_policy_blocks(self):
        for status in (401, 402, 403, 407, 451):
            with self.subTest(status=status):
                client = SafeHTTPTextClient(
                    resolver=Resolver({"lab.example": ["8.8.8.8"]}),
                    transport=Transport([RawHTTPResponse(status, {}, b"")]),
                )
                with self.assertRaises(SafeHTTPAccessRestrictedError):
                    client.fetch("https://lab.example/")

    def test_structured_log_contains_no_secret_or_business_body(self):
        output = io.StringIO()
        context = JMRRuntimeContext(
            event_sink=JsonLineEventSink(output),
            clock=Clock(),
            metadata={"run_id": "security-gate"},
        )
        emit_runtime_event(
            context=context,
            state={"user_id": "u", "case_id": "c"},
            node="security_test",
            status="FAILED",
            details={
                "api_key": "sk-secretvalue",
                "tool_arguments": {"professor_name": "Sensitive Name"},
                "error_label": "postgresql://root:secret@localhost/private",
            },
        )
        log = output.getvalue()
        for forbidden in ("sk-secretvalue", "Sensitive Name", "root:secret"):
            self.assertNotIn(forbidden, log)


if __name__ == "__main__":
    unittest.main()
