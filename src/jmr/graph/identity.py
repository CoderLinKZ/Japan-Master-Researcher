"""Trusted invocation identity checks for every target graph entrypoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class InvocationIdentityError(ValueError):
    """Raised when state and trusted invocation identifiers do not agree."""


@dataclass(frozen=True, slots=True)
class InvocationIdentity:
    """Validated identity shared by graph state and RunnableConfig."""

    user_id: str
    case_id: str
    thread_id: str


def validate_invocation_identity(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> InvocationIdentity:
    """Require explicit user identity and ``case_id == thread_id``.

    This check is deliberately independent of checkpoint contents.  A caller
    must be authorized for the case before reading a thread, then target graph
    boundary nodes call this function again to protect against mixed inputs.
    """

    if not isinstance(state, Mapping):
        raise InvocationIdentityError("state must be a mapping")
    if not isinstance(config, Mapping):
        raise InvocationIdentityError("config must be a mapping")
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        raise InvocationIdentityError("config.configurable must be a mapping")

    state_user_id = _identifier(state.get("user_id"), "state.user_id")
    state_case_id = _identifier(state.get("case_id"), "state.case_id")
    config_user_id = _identifier(configurable.get("user_id"), "configurable.user_id")
    thread_id = _identifier(configurable.get("thread_id"), "configurable.thread_id")

    if state_user_id != config_user_id:
        raise InvocationIdentityError("state.user_id must match configurable.user_id")
    if state_case_id != thread_id:
        raise InvocationIdentityError("state.case_id must match configurable.thread_id")
    return InvocationIdentity(
        user_id=state_user_id,
        case_id=state_case_id,
        thread_id=thread_id,
    )


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvocationIdentityError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > 255:
        raise InvocationIdentityError(f"{field_name} cannot exceed 255 characters")
    return normalized
