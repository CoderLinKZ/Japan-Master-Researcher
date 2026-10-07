"""Browser-facing transport and bounded upload contracts."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.api import create_app  # noqa: E402
from jmr.api.auth import BearerTokenAuthenticator  # noqa: E402
from jmr.api.uploads import UploadValidationError, extract_document  # noqa: E402


class WorkspaceServiceFake:
    def startup(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def list_cases(self, *, user_id: str) -> dict:
        return {"items": [{"case_id": "case-one", "user_id": user_id}]}

    def get_workspace(self, *, user_id: str, case_id: str) -> dict:
        return {
            "user_id": user_id,
            "case_id": case_id,
            "target": None,
            "plans": [],
            "retrieval_runs": [],
            "evidence": [],
            "directions": [],
            "selections": [],
            "drafts": [],
            "reviews": [],
            "history": [],
        }

    def get_conversation(self, *, user_id: str, case_id: str) -> dict:
        return {"case_id": case_id, "messages": [{"role": "user", "content": user_id}]}

    def get_progress(self, *, user_id: str, case_id: str) -> dict:
        return {
            "case_id": case_id,
            "workflow_stage": "RETRIEVING_RESEARCH_EVIDENCE",
            "run_status": "RUNNING",
            "events": [
                {
                    "kind": "mcp_tool",
                    "tool": "mcp__scholar__search_publications",
                    "created_at": "2026-09-25T00:00:00+00:00",
                }
            ],
        }

    def retry_case(self, *, user_id: str, case_id: str) -> dict:
        return {
            "user_id": user_id,
            "case_id": case_id,
            "workflow_stage": "GENERATING_RESEARCH_DIRECTIONS",
            "run_status": "WAITING_FOR_USER",
            "retry_available": False,
        }

    def upload_memory_document(
        self, *, user_id: str, filename: str, content: bytes
    ) -> dict:
        extracted = extract_document(filename, content)
        return {"status": "SUCCESS", "memory": {"user_id": user_id, **extracted}}

    def get_memory_document(self, *, user_id: str, memory_id: str) -> tuple[str, bytes]:
        return f"{user_id}-{memory_id}.txt", b"document"


class FrontendWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(
            create_app(
                service=WorkspaceServiceFake(),
                authenticator=BearerTokenAuthenticator(
                    {"test-token-for-root-0001": "root"}
                ),
            ),
            raise_server_exceptions=False,
        )
        self.headers = {"Authorization": "Bearer test-token-for-root-0001"}

    def tearDown(self) -> None:
        self.client.close()

    def test_browser_shell_and_authenticated_read_models(self) -> None:
        shell = self.client.get("/frontend")
        self.assertEqual(shell.status_code, 200)
        self.assertIn("/frontend/assets/app.js", shell.text)
        self.assertEqual(shell.headers["cache-control"], "no-store")
        self.assertEqual(
            self.client.get("/frontend/assets/app.js").headers["cache-control"],
            "no-store",
        )
        self.assertIn('id="delete-case-dialog"', shell.text)
        self.assertEqual(self.client.get("/api/v1/cases").status_code, 401)
        self.assertEqual(
            self.client.get("/api/v1/session", headers=self.headers).json(),
            {"user_id": "root"},
        )
        self.assertEqual(
            self.client.get("/api/v1/cases", headers=self.headers).json()["items"][0][
                "case_id"
            ],
            "case-one",
        )
        self.assertEqual(
            self.client.get(
                "/api/v1/cases/case-one/workspace", headers=self.headers
            ).json()["user_id"],
            "root",
        )
        self.assertEqual(
            self.client.get(
                "/api/v1/cases/case-one/conversation", headers=self.headers
            ).json()["messages"][0]["content"],
            "root",
        )
        self.assertEqual(
            self.client.get("/api/v1/cases/case-one/progress").status_code,
            401,
        )
        self.assertEqual(
            self.client.get(
                "/api/v1/cases/case-one/progress", headers=self.headers
            ).json()["events"][0]["tool"],
            "mcp__scholar__search_publications",
        )

    def test_upload_and_download_are_authenticated(self) -> None:
        content = "个人研究经历".encode()
        response = self.client.post(
            "/api/v1/memories/upload",
            headers=self.headers,
            files={"file": ("profile.txt", io.BytesIO(content), "text/plain")},
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["memory"]["user_id"], "root")
        self.assertEqual(
            self.client.get("/api/v1/memories/id/download").status_code, 401
        )
        download = self.client.get("/api/v1/memories/id/download", headers=self.headers)
        self.assertEqual(download.content, b"document")
        self.assertIn("attachment", download.headers["Content-Disposition"])

    def test_failed_stage_retry_is_authenticated(self) -> None:
        url = "/api/v1/cases/case-one/retry"
        self.assertEqual(self.client.post(url).status_code, 401)
        response = self.client.post(url, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["case_id"], "case-one")

    def test_upload_route_accepts_more_than_the_json_body_limit(self) -> None:
        response = self.client.post(
            "/api/v1/memories/upload",
            headers=self.headers,
            files={"file": ("large.txt", b"x" * (150 * 1024), "text/plain")},
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.assertTrue(response.json()["memory"]["truncated"])

    def test_extraction_rejects_unsupported_and_oversized_files(self) -> None:
        with self.assertRaises(UploadValidationError):
            extract_document("file.exe", b"text")
        with self.assertRaises(UploadValidationError):
            extract_document("file.txt", b"x" * (2 * 1024 * 1024 + 1))
        with self.assertRaises(UploadValidationError):
            extract_document("file.pdf", b"not pdf")


if __name__ == "__main__":
    unittest.main()
