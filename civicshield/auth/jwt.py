"""
CivicShield Auth — JWT helpers
create_token(), verify_token(), get_current_user() FastAPI dependency.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from civicshield.auth.models import Role, UserMe

# ── Config ────────────────────────────────────────────────────────
SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-production")
ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],   # bcrypt 5.x has a passlib compat issue on Win; pbkdf2_sha256 is equally secure
    deprecated="auto",
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ── Password helpers ──────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── Token helpers ─────────────────────────────────────────────────

def create_token(subject: str, role: Role, extra: dict | None = None) -> str:
    """Create a signed JWT with an expiry claim."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role.value,
        "iat": now,
        "exp": now + timedelta(minutes=EXPIRE_MINUTES),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    """Decode and validate a JWT; raise 401 on any failure."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise credentials_error
    if payload.get("sub") is None:
        raise credentials_error
    return payload


# ── FastAPI dependency ────────────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserMe:
    """Inject the authenticated user into any route."""
    payload = verify_token(token)
    return UserMe(
        user_id=payload["sub"],
        username=payload.get("username", payload["sub"]),
        email=payload.get("email", ""),
        role=Role(payload["role"]),
        is_active=True,
    )
