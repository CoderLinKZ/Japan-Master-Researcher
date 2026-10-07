"""Fail-closed bearer-token authentication for the HTTP boundary."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

API_TOKENS_ENV = "JMR_API_TOKENS_JSON"
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True, eq=False)
class BearerTokenAuthenticator:
    """Resolve an opaque bearer token to one trusted user identifier."""

    token_to_user: Mapping[str, str]
    configuration_error: str | None = None

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> BearerTokenAuthenticator:
        selected = os.environ if environment is None else environment
        raw = selected.get(API_TOKENS_ENV, "").strip()
        if not raw:
            return cls({}, f"{API_TOKENS_ENV} is missing")
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return cls({}, f"{API_TOKENS_ENV} must be a JSON object")
        if not isinstance(decoded, dict):
            return cls({}, f"{API_TOKENS_ENV} must be a JSON object")
        normalized: dict[str, str] = {}
        for token, user_id in decoded.items():
            if (
                not isinstance(token, str)
                or len(token) < 16
                or not isinstance(user_id, str)
                or not user_id.strip()
                or len(user_id.strip()) > 255
            ):
                return cls(
                    {},
                    f"{API_TOKENS_ENV} contains an invalid token or user identifier",
                )
            normalized[token] = user_id.strip()
        if not normalized:
            return cls({}, f"{API_TOKENS_ENV} cannot be empty")
        return cls(normalized)

    @property
    def configured(self) -> bool:
        return self.configuration_error is None and bool(self.token_to_user)

    def __call__(
        self,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(_bearer),
        ],
    ) -> str:
        if not self.configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API authentication is not configured",
            )
        if credentials is None or credentials.scheme.casefold() != "bearer":
            raise _unauthorized()
        for token, user_id in self.token_to_user.items():
            if secrets.compare_digest(credentials.credentials, token):
                return user_id
        raise _unauthorized()


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )
