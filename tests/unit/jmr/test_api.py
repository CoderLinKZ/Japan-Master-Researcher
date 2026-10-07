"""HTTP contract tests for the FastAPI boundary.

These tests deliberately inject an application-service fake.  The transport
contract can therefore be exercised without opening PostgreSQL connections or
calling a model/provider.
"""

from __future__ import annotations

import sys
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx2
from anthropic import APITimeoutError
from fastapi.testclient import TestClient

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.api import create_app  # noqa: E402
from jmr.api.auth import BearerTokenAuthenticator  # noqa: E402
from jmr.api.service import (  # noqa: E402
    CaseBusyError,
    OperationConflictError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from jmr.graph.nodes.common import ModelOutputValidationError  # noqa: E402
from jmr.persistence import OwnershipError  # noqa: E402

TOKEN = "test-token-for-user-0001"
AUTHORIZATION = {"Authorization": f"Bearer {TOKEN}"}
CASE_IDEMPOTENCY_KEY = "create-case-1"
INTERRUPT_TOKEN = "a" * 64
PLAN_TOKEN = "b" * 32


def _case_view(
    user_id: str,
    case_id: str,
    *,
    run_status: str = "WAITING_FOR_USER",
    pending_interrupt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "case_id": case_id,
        "workflow_stage": "VALIDATING_APPLICATION_INPUTS",
        "run_status": run_status,
        "pending_interrupt": dict(pending_interrupt)
        if pending_interrupt is not None
        else None,
        "interrupt_token": INTERRUPT_TOKEN if pending_interrupt is not None else None,
        "warnings": [],
        "errors": [],
        "result_available": run_status == "COMPLETED",
        "artifacts": {},
    }


class FakeService:
    """Record the trusted principal and return bounded public projections."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.failures: dict[str, Exception] = {}
        self.startup_calls = 0
        self.shutdown_calls = 0
        self.readiness_report: dict[str, Any] = {
            "status": "ok",
            "checks": {
                "database": "ok",
                "schema": "ok",
                "object_store": "ok",
                "model_configuration": "ok",
            },
        }

    def startup(self) -> None:
        self.startup_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def _record(self, operation: str, **arguments: Any) -> None:
        self.calls.append((operation, dict(arguments)))
        failure = self.failures.get(operation)
        if failure is not None:
            raise failure

    def readiness(self) -> Mapping[str, Any]:
        self._record("readiness")
        return self.readiness_report

    def start_case(
        self,
        *,
        user_id: str,
        message: str,
        case_id: str | None = None,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        self._record(
            "start_case",
            user_id=user_id,
            message=message,
            case_id=case_id,
            idempotency_key=idempotency_key,
        )
        return _case_view(
            user_id,
            case_id or "generated-case",
            pending_interrupt={"kind": "APPLICATION_INPUT_REQUIRED"},
        )

    def send_message(
        self, *, user_id: str, case_id: str, message: str
    ) -> Mapping[str, Any]:
        self._record("send_message", user_id=user_id, case_id=case_id, message=message)
        return _case_view(user_id, case_id, run_status="RUNNING")

    def resume_case(
        self,
        *,
        user_id: str,
        case_id: str,
        payload: Mapping[str, Any],
        interrupt_token: str,
    ) -> Mapping[str, Any]:
        self._record(
            "resume_case",
            user_id=user_id,
            case_id=case_id,
            payload=dict(payload),
            interrupt_token=interrupt_token,
        )
        if interrupt_token != INTERRUPT_TOKEN:
            raise OperationConflictError("interrupt token is stale")
        result = _case_view(user_id, case_id, run_status="RUNNING")
        result["artifacts"] = {"research_plan_id": "plan-1"}
        return result

    def get_case(self, *, user_id: str, case_id: str) -> Mapping[str, Any]:
        self._record("get_case", user_id=user_id, case_id=case_id)
        return _case_view(user_id, case_id)

    def get_result(self, *, user_id: str, case_id: str) -> Mapping[str, Any]:
        self._record("get_result", user_id=user_id, case_id=case_id)
        return {
            "user_id": user_id,
            "case_id": case_id,
            "iteration_id": "iteration-1",
            "content": "grounded final content",
            "citation_ids": ["evidence-1"],
        }

    def call_memory(
        self, *, user_id: str, tool_name: str, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self._record(
            "call_memory",
            user_id=user_id,
            tool_name=tool_name,
            arguments=dict(arguments),
        )
        if tool_name == "list_memories":
            return {"status": "SUCCESS", "items": []}
        if tool_name == "get_memory" and arguments.get("memory_id") == "missing":
            return {"status": "NO_RESULT", "memory": None}
        return {
            "status": "SUCCESS",
            "memory": {
                "memory_id": str(arguments.get("memory_id") or "memory-1"),
                "status": "ACTIVE",
                "version": 1,
                "value": {
                    "kind": arguments.get("kind", "research_interest"),
                    "content": dict(arguments.get("content", {})),
                    "tags": list(arguments.get("tags", [])),
                },
            },
        }

    def plan_case_deletion(self, *, user_id: str, case_id: str) -> Mapping[str, Any]:
        self._record("plan_case_deletion", user_id=user_id, case_id=case_id)
        return self._deletion(user_id, case_id, dry_run=True)

    def confirm_case_deletion(
        self, *, user_id: str, case_id: str, plan_token: str
    ) -> Mapping[str, Any]:
        self._record(
            "confirm_case_deletion",
            user_id=user_id,
            case_id=case_id,
            plan_token=plan_token,
        )
        if plan_token != PLAN_TOKEN:
            raise OperationConflictError("deletion plan has expired")
        return self._deletion(user_id, case_id, dry_run=False)

    @staticmethod
    def _deletion(user_id: str, case_id: str, *, dry_run: bool) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "user_id": user_id,
            "status": "PENDING" if dry_run else "COMPLETED",
            "dry_run": dry_run,
            "object_count": 0,
            "object_uris": [],
            "checkpoint_deleted": not dry_run,
            "business_deleted": not dry_run,
            "audit_retained": True,
            "plan_token": PLAN_TOKEN if dry_run else None,
        }


class FastAPIContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeService()
        self.authenticator = BearerTokenAuthenticator({TOKEN: "user-1"})
        self.app = create_app(
            service=self.service,
            authenticator=self.authenticator,
        )
        self.client_context = TestClient(
            self.app,
            raise_server_exceptions=False,
        )
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def test_lifespan_liveness_readiness_and_openapi(self) -> None:
        self.assertEqual(self.service.startup_calls, 1)
        live = self.client.get("/health/live")
        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.json(), {"status": "ok", "checks": {"process": "ok"}})

        ready = self.client.get("/health/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["checks"]["authentication"], "ok")

        schema = self.client.get("/openapi.json")
        self.assertEqual(schema.status_code, 200)
        paths = schema.json()["paths"]
        self.assertIn("/api/v1/cases", paths)
        self.assertIn("/api/v1/cases/{case_id}/resume", paths)
        self.assertIn("/api/v1/memories/{memory_id}", paths)
        self.assertIn("HTTPBearer", schema.json()["components"]["securitySchemes"])
        self.assertTrue(
            {"201", "401", "409", "413", "422", "503"}.issubset(
                paths["/api/v1/cases"]["post"]["responses"]
            )
        )

    def test_shutdown_runs_when_lifespan_exits(self) -> None:
        service = FakeService()
        app = create_app(service=service, authenticator=self.authenticator)
        with TestClient(app):
            self.assertEqual(service.startup_calls, 1)
            self.assertEqual(service.shutdown_calls, 0)
        self.assertEqual(service.shutdown_calls, 1)

    def test_authentication_is_fail_closed_and_principal_is_server_derived(
        self,
    ) -> None:
        missing = self.client.get("/api/v1/cases/case-1")
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.json()["error"]["code"], "UNAUTHORIZED")
        self.assertEqual(missing.headers["WWW-Authenticate"], "Bearer")

        invalid = self.client.get(
            "/api/v1/cases/case-1",
            headers={"Authorization": "Bearer invalid-token-value"},
        )
        self.assertEqual(invalid.status_code, 401)

        valid = self.client.get("/api/v1/cases/case-1", headers=AUTHORIZATION)
        self.assertEqual(valid.status_code, 200)
        self.assertEqual(
            self.service.calls[-1],
            ("get_case", {"user_id": "user-1", "case_id": "case-1"}),
        )

        unconfigured = create_app(
            service=FakeService(),
            authenticator=BearerTokenAuthenticator(
                {}, configuration_error="tokens are missing"
            ),
        )
        with TestClient(unconfigured, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/cases/case-1")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json()["error"]["message"],
                "API authentication is not configured",
            )

    def test_case_create_read_message_resume_and_result_contracts(self) -> None:
        created = self.client.post(
            "/api/v1/cases",
            headers={
                **AUTHORIZATION,
                "X-Request-ID": "request-1",
                "Idempotency-Key": CASE_IDEMPOTENCY_KEY,
            },
            json={"message": "  investigate this professor  "},
        )
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.headers["Location"], "/api/v1/cases/generated-case")
        self.assertEqual(created.headers["X-Request-ID"], "request-1")
        self.assertEqual(
            set(created.json()),
            {
                "user_id",
                "case_id",
                "workflow_stage",
                "run_status",
                "pending_interrupt",
                "interrupt_token",
                "retry_available",
                "failure",
                "warnings",
                "errors",
                "result_available",
                "artifacts",
            },
        )
        self.assertEqual(created.json()["interrupt_token"], INTERRUPT_TOKEN)
        self.assertEqual(
            self.service.calls[-1],
            (
                "start_case",
                {
                    "user_id": "user-1",
                    "message": "investigate this professor",
                    "case_id": None,
                    "idempotency_key": CASE_IDEMPOTENCY_KEY,
                },
            ),
        )

        fetched = self.client.get("/api/v1/cases/generated-case", headers=AUTHORIZATION)
        self.assertEqual(fetched.status_code, 200)

        sent = self.client.post(
            "/api/v1/cases/generated-case/messages",
            headers=AUTHORIZATION,
            json={"message": "continue"},
        )
        self.assertEqual(sent.status_code, 200)
        self.assertEqual(sent.json()["run_status"], "RUNNING")

        resumed = self.client.post(
            "/api/v1/cases/generated-case/resume",
            headers=AUTHORIZATION,
            json={
                "payload": {"fields": {"professor_name": "Yamada"}},
                "interrupt_token": INTERRUPT_TOKEN,
            },
        )
        self.assertEqual(resumed.status_code, 200)
        self.assertEqual(resumed.json()["artifacts"], {"research_plan_id": "plan-1"})
        self.assertEqual(
            self.service.calls[-1][1]["payload"],
            {"fields": {"professor_name": "Yamada"}},
        )
        self.assertEqual(self.service.calls[-1][1]["interrupt_token"], INTERRUPT_TOKEN)

        result = self.client.get(
            "/api/v1/cases/generated-case/result", headers=AUTHORIZATION
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["citation_ids"], ["evidence-1"])

    def test_case_creation_requires_idempotency_key_before_service(self) -> None:
        for headers in (AUTHORIZATION, {**AUTHORIZATION, "Idempotency-Key": ""}):
            with self.subTest(headers=headers):
                response = self.client.post(
                    "/api/v1/cases",
                    headers=headers,
                    json={"message": "research this program"},
                )
                self.assertEqual(response.status_code, 422)
        self.assertFalse(any(call[0] == "start_case" for call in self.service.calls))

    def test_resume_requires_a_current_interrupt_token(self) -> None:
        url = "/api/v1/cases/case-1/resume"
        payload = {"payload": {"fields": {"professor_name": "Yamada"}}}
        for body in (
            payload,
            {**payload, "interrupt_token": "short"},
            {**payload, "interrupt_token": "z" * 64},
        ):
            with self.subTest(body=body):
                response = self.client.post(url, headers=AUTHORIZATION, json=body)
                self.assertEqual(response.status_code, 422)
        self.assertFalse(any(call[0] == "resume_case" for call in self.service.calls))

        stale = self.client.post(
            url,
            headers=AUTHORIZATION,
            json={**payload, "interrupt_token": "c" * 64},
        )
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["error"]["code"], "OPERATION_CONFLICT")

    def test_blank_messages_and_unknown_fields_are_rejected_before_service(
        self,
    ) -> None:
        for path in ("/api/v1/cases", "/api/v1/cases/case-1/messages"):
            with self.subTest(path=path):
                before = len(self.service.calls)
                response = self.client.post(
                    path,
                    headers={**AUTHORIZATION, "Idempotency-Key": CASE_IDEMPOTENCY_KEY},
                    json={"message": "   "},
                )
                self.assertEqual(response.status_code, 422)
                self.assertEqual(len(self.service.calls), before)

        extra = self.client.post(
            "/api/v1/cases",
            headers={**AUTHORIZATION, "Idempotency-Key": CASE_IDEMPOTENCY_KEY},
            json={"message": "valid", "user_id": "attacker-selected"},
        )
        self.assertEqual(extra.status_code, 422)

    def test_resume_payload_is_size_bounded(self) -> None:
        response = self.client.post(
            "/api/v1/cases/case-1/resume",
            headers=AUTHORIZATION,
            json={
                "payload": {"text": "x" * (64 * 1024)},
                "interrupt_token": INTERRUPT_TOKEN,
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertFalse(any(call[0] == "resume_case" for call in self.service.calls))

    def test_request_body_is_rejected_before_json_validation_at_128_kib(self) -> None:
        response = self.client.post(
            "/api/v1/cases",
            headers={**AUTHORIZATION, "Idempotency-Key": CASE_IDEMPOTENCY_KEY},
            json={"message": "x" * (128 * 1024)},
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "PAYLOAD_TOO_LARGE")
        self.assertTrue(response.headers["X-Request-ID"].startswith("req_"))
        self.assertFalse(any(call[0] == "start_case" for call in self.service.calls))

    def test_memory_mutations_require_literal_confirmation(self) -> None:
        body = {
            "kind": "research_interest",
            "content": {"topic": "reliable AI"},
            "tags": ["ai", " ai "],
            "idempotency_key": "memory-create-1",
        }
        for confirmation in (None, False):
            with self.subTest(confirmation=confirmation):
                candidate = dict(body)
                if confirmation is not None:
                    candidate["confirmed_by_user"] = confirmation
                before = len(self.service.calls)
                rejected = self.client.post(
                    "/api/v1/memories",
                    headers=AUTHORIZATION,
                    json=candidate,
                )
                self.assertEqual(rejected.status_code, 422)
                self.assertEqual(len(self.service.calls), before)

        accepted = self.client.post(
            "/api/v1/memories",
            headers=AUTHORIZATION,
            json={**body, "confirmed_by_user": True},
        )
        self.assertEqual(accepted.status_code, 201)
        call = self.service.calls[-1]
        self.assertEqual(call[0], "call_memory")
        self.assertEqual(call[1]["user_id"], "user-1")
        self.assertEqual(call[1]["tool_name"], "create_memory")
        self.assertEqual(call[1]["arguments"]["tags"], ["ai"])

        listed = self.client.get(
            "/api/v1/memories?kind=research_interest&tags=ai&limit=10",
            headers=AUTHORIZATION,
        )
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(self.service.calls[-1][1]["tool_name"], "list_memories")

        missing = self.client.get("/api/v1/memories/missing", headers=AUTHORIZATION)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "NOT_FOUND")

    def test_case_deletion_is_report_first_and_requires_confirmation_body(self) -> None:
        plan = self.client.get(
            "/api/v1/cases/case-1/deletion-plan", headers=AUTHORIZATION
        )
        self.assertEqual(plan.status_code, 200)
        self.assertTrue(plan.json()["dry_run"])
        self.assertEqual(plan.json()["plan_token"], PLAN_TOKEN)

        no_body = self.client.request(
            "DELETE", "/api/v1/cases/case-1", headers=AUTHORIZATION
        )
        self.assertEqual(no_body.status_code, 422)

        rejected = self.client.request(
            "DELETE",
            "/api/v1/cases/case-1",
            headers=AUTHORIZATION,
            json={"confirmed_by_user": False, "plan_token": PLAN_TOKEN},
        )
        self.assertEqual(rejected.status_code, 422)

        missing_plan = self.client.request(
            "DELETE",
            "/api/v1/cases/case-1",
            headers=AUTHORIZATION,
            json={"confirmed_by_user": True},
        )
        self.assertEqual(missing_plan.status_code, 422)

        stale_plan = self.client.request(
            "DELETE",
            "/api/v1/cases/case-1",
            headers=AUTHORIZATION,
            json={"confirmed_by_user": True, "plan_token": "c" * 32},
        )
        self.assertEqual(stale_plan.status_code, 409)
        self.assertEqual(stale_plan.json()["error"]["code"], "OPERATION_CONFLICT")

        deleted = self.client.request(
            "DELETE",
            "/api/v1/cases/case-1",
            headers=AUTHORIZATION,
            json={"confirmed_by_user": True, "plan_token": PLAN_TOKEN},
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse(deleted.json()["dry_run"])
        self.assertEqual(
            self.service.calls[-1],
            (
                "confirm_case_deletion",
                {"user_id": "user-1", "case_id": "case-1", "plan_token": PLAN_TOKEN},
            ),
        )

    def test_error_translation_is_stable_and_does_not_leak_internal_details(
        self,
    ) -> None:
        cases = (
            (ResourceNotFoundError("secret record detail"), 404, "NOT_FOUND"),
            (OwnershipError("belongs to user-2"), 404, "NOT_FOUND"),
            (
                OperationConflictError("case result is not ready"),
                409,
                "OPERATION_CONFLICT",
            ),
            (CaseBusyError("lock detail"), 409, "CASE_BUSY"),
            (
                ServiceUnavailableError("database dsn detail"),
                503,
                "SERVICE_UNAVAILABLE",
            ),
            (
                APITimeoutError(
                    request=httpx2.Request("POST", "https://model.invalid")
                ),
                503,
                "MODEL_UNAVAILABLE",
            ),
            (ModelOutputValidationError("plan_research"), 502, "MODEL_OUTPUT_INVALID"),
        )
        for error, expected_status, expected_code in cases:
            with self.subTest(error=type(error).__name__):
                self.service.failures["get_case"] = error
                response = self.client.get(
                    "/api/v1/cases/case-1", headers=AUTHORIZATION
                )
                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(response.json()["error"]["code"], expected_code)
                if not isinstance(error, OperationConflictError) or isinstance(
                    error, CaseBusyError
                ):
                    self.assertNotIn(str(error), response.text)

        self.service.failures["get_case"] = RuntimeError(
            "postgresql://admin:secret@localhost/private"
        )
        unexpected = self.client.get("/api/v1/cases/case-1", headers=AUTHORIZATION)
        self.assertEqual(unexpected.status_code, 500)
        payload = unexpected.json()["error"]
        self.assertEqual(payload["code"], "INTERNAL_ERROR")
        self.assertTrue(payload["failure_id"].startswith("failure_"))
        self.assertNotIn("admin:secret", unexpected.text)

    def test_failed_readiness_includes_authentication_and_returns_503(self) -> None:
        self.service.readiness_report = {
            "status": "failed",
            "checks": {"database": "failed", "schema": "unknown"},
        }
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["checks"]["authentication"], "ok")


if __name__ == "__main__":
    unittest.main()
