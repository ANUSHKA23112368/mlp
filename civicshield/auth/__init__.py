"""
CivicShield Auth — __init__.py
Re-exports the public surface of the auth package.
"""
from civicshield.auth.models import Role, LoginRequest, TokenResponse, UserMe
from civicshield.auth.jwt import (
    create_token,
    verify_token,
    get_current_user,
    hash_password,
    verify_password,
    EXPIRE_MINUTES,
)
from civicshield.auth.rbac import require_role, require_viewer, require_analyst, require_admin

__all__ = [
    "Role",
    "LoginRequest",
    "TokenResponse",
    "UserMe",
    "create_token",
    "verify_token",
    "get_current_user",
    "hash_password",
    "verify_password",
    "EXPIRE_MINUTES",
    "require_role",
    "require_viewer",
    "require_analyst",
    "require_admin",
]
