"""
CivicShield — Auth Router
Mounts at /auth with endpoints:
  POST /auth/login  → issue JWT
  GET  /auth/me     → current user info
  POST /auth/logout → client-side hint (stateless JWT — just returns ok)

Demo users are seeded in memory for development.
In production, replace USER_STORE with a real DB query.
"""
from __future__ import annotations
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from civicshield.auth.models import LoginRequest, TokenResponse, UserMe, UserRecord, Role
from civicshield.auth.jwt import (
    create_token,
    get_current_user,
    hash_password,
    verify_password,
    EXPIRE_MINUTES,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# ── In-memory user store (dev only) ──────────────────────────────
# Passwords are pre-hashed; plain-text is shown only in this comment for dev.
#   admin   / admin123
#   analyst / analyst123
#   viewer  / viewer123
USER_STORE: dict[str, UserRecord] = {
    "admin": UserRecord(
        user_id="u-001",
        username="admin",
        email="admin@civicshield.local",
        hashed_password=hash_password("admin123"),
        role=Role.admin,
    ),
    "analyst": UserRecord(
        user_id="u-002",
        username="analyst",
        email="analyst@civicshield.local",
        hashed_password=hash_password("analyst123"),
        role=Role.analyst,
    ),
    "viewer": UserRecord(
        user_id="u-003",
        username="viewer",
        email="viewer@civicshield.local",
        hashed_password=hash_password("viewer123"),
        role=Role.viewer,
    ),
}


def _authenticate(username: str, password: str) -> UserRecord:
    user = USER_STORE.get(username)
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")
    return user


@router.post("/login", response_model=TokenResponse, summary="Obtain a JWT access token")
async def login(body: LoginRequest):
    user = _authenticate(body.username, body.password)
    token = create_token(
        subject=user.user_id,
        role=user.role,
        extra={"username": user.username, "email": user.email},
    )
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=EXPIRE_MINUTES * 60,
        role=user.role,
    )


@router.get("/me", response_model=UserMe, summary="Return the currently authenticated user")
async def me(current_user: UserMe = Depends(get_current_user)):
    return current_user


@router.post("/logout", summary="Client-side logout hint (stateless)")
async def logout(current_user: UserMe = Depends(get_current_user)):
    # Stateless JWT — instruct client to discard the token.
    return {"detail": "Logged out. Discard your token client-side.", "username": current_user.username}
