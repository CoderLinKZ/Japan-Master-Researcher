"""Host-side authorization policy for user-owned research cases."""

from __future__ import annotations


class PrincipalSecurityPolicy:
    """Allow access only when the authenticated principal owns the case."""

    def authorize_case(
        self,
        *,
        principal_user_id: str,
        user_id: str,
        case_id: str,
    ) -> None:
        del case_id
        if not isinstance(principal_user_id, str) or not principal_user_id.strip():
            raise PermissionError("an authenticated principal is required")
        if principal_user_id.strip() != user_id:
            raise PermissionError("principal does not own this user namespace")


__all__ = ["PrincipalSecurityPolicy"]
