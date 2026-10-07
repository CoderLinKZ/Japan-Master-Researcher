"""Boundary validation for sensitive HTTP mutations and JSON payloads."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.api.models import (  # noqa: E402
    ConfirmCaseDeletionRequest,
    CreateMemoryRequest,
    DeleteMemoryRequest,
    ResumeCaseRequest,
    UpdateMemoryRequest,
)

CREATE_MEMORY = {
    "kind": "research_interest",
    "content": {"topic": "agents"},
    "idempotency_key": "create-1",
}
UPDATE_MEMORY = {
    "content": {"topic": "agent reliability"},
    "expected_version": 1,
    "idempotency_key": "update-1",
}
DELETE_MEMORY = {
    "expected_version": 1,
    "idempotency_key": "delete-1",
}
INTERRUPT_TOKEN = "a" * 64
PLAN_TOKEN = "b" * 32


class APIBoundarySecurityTests(unittest.TestCase):
    def test_confirmed_by_user_requires_the_boolean_true_value(self) -> None:
        requests = (
            (CreateMemoryRequest, CREATE_MEMORY),
            (UpdateMemoryRequest, UPDATE_MEMORY),
            (DeleteMemoryRequest, DELETE_MEMORY),
            (ConfirmCaseDeletionRequest, {"plan_token": PLAN_TOKEN}),
        )
        for model, fields in requests:
            with self.subTest(model=model.__name__, accepted=True):
                accepted = model.model_validate({**fields, "confirmed_by_user": True})
                self.assertIs(accepted.confirmed_by_user, True)
            for invalid in (1, 1.0, "true", "1", False, None):
                with self.subTest(model=model.__name__, invalid=invalid):
                    with self.assertRaises(ValidationError):
                        model.model_validate({**fields, "confirmed_by_user": invalid})

    def test_expected_version_requires_a_positive_integer_not_bool(self) -> None:
        for model, fields in (
            (UpdateMemoryRequest, UPDATE_MEMORY),
            (DeleteMemoryRequest, DELETE_MEMORY),
        ):
            for invalid in (True, False, 0, -1, 1.0, "1"):
                with self.subTest(model=model.__name__, invalid=invalid):
                    with self.assertRaises(ValidationError):
                        model.model_validate(
                            {
                                **fields,
                                "confirmed_by_user": True,
                                "expected_version": invalid,
                            }
                        )

    def test_nested_nonfinite_numbers_are_rejected(self) -> None:
        for number in (float("nan"), float("inf"), float("-inf")):
            for model, fields, payload_field in (
                (CreateMemoryRequest, CREATE_MEMORY, "content"),
                (UpdateMemoryRequest, UPDATE_MEMORY, "content"),
                (
                    ResumeCaseRequest,
                    {"interrupt_token": INTERRUPT_TOKEN},
                    "payload",
                ),
            ):
                with self.subTest(model=model.__name__, number=str(number)):
                    candidate = {
                        **fields,
                        payload_field: {"nested": [{"number": number}]},
                    }
                    if model is not ResumeCaseRequest:
                        candidate["confirmed_by_user"] = True
                    with self.assertRaises(ValidationError):
                        model.model_validate(candidate)

    def test_interrupt_and_deletion_plan_tokens_are_required_and_hex_shaped(
        self,
    ) -> None:
        requests = (
            (ResumeCaseRequest, {"payload": {}}, "interrupt_token", INTERRUPT_TOKEN),
            (
                ConfirmCaseDeletionRequest,
                {"confirmed_by_user": True},
                "plan_token",
                PLAN_TOKEN,
            ),
        )
        for model, fields, token_name, token in requests:
            with self.subTest(model=model.__name__, token="missing"):
                with self.assertRaises(ValidationError):
                    model.model_validate(fields)
            for invalid in ("short", "z" * len(token), None):
                with self.subTest(model=model.__name__, token=invalid):
                    with self.assertRaises(ValidationError):
                        model.model_validate({**fields, token_name: invalid})
            self.assertEqual(
                getattr(
                    model.model_validate({**fields, token_name: token}), token_name
                ),
                token,
            )

    def test_omitted_tags_can_preserve_existing_tags_and_empty_tags_clear_them(
        self,
    ) -> None:
        base = {**UPDATE_MEMORY, "confirmed_by_user": True}
        omitted = UpdateMemoryRequest.model_validate(base)
        self.assertIsNone(omitted.tags)
        self.assertNotIn("tags", omitted.model_dump(exclude_none=True))

        cleared = UpdateMemoryRequest.model_validate({**base, "tags": []})
        self.assertEqual(cleared.tags, [])
        self.assertIn("tags", cleared.model_dump(exclude_none=True))

        with self.assertRaises(ValidationError):
            UpdateMemoryRequest.model_validate({**base, "tags": None})


if __name__ == "__main__":
    unittest.main()
