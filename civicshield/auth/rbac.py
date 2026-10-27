"""
CivicShield Auth — Role-Based Access Control
require_role() returns a FastAPI dependency that enforces minimum role level.

Role hierarchy (ascending privilege):
  viewer < analyst < admin
"""
from __future__ import annotations
from fastapi import Depends, HTTPException, status

from civicshield.auth.models import Role, UserMe
from civicshield.auth.jwt import get_current_user

# Map role → numeric level (higher = more privilege)
_LEVEL = {Role.viewer: 0, Role.analyst: 1, Role.admin: 2}


def require_role(*roles: Role):
    """
    Dependency factory. Usage:

        @router.get("/admin-only")
        async def endpoint(user=Depends(require_role(Role.admin))):
            ...
    """
    allowed_levels = {_LEVEL[r] for r in roles}
    minimum_level = min(allowed_levels)

    async def _check(user: UserMe = Depends(get_current_user)) -> UserMe:
        if _LEVEL[user.role] < minimum_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {[r.value for r in roles]}",
            )
        return user

    return _check


# Convenience shortcuts
require_viewer = require_role(Role.viewer, Role.analyst, Role.admin)
require_analyst = require_role(Role.analyst, Role.admin)
require_admin = require_role(Role.admin)
