"""
CivicShield Auth — User & Role Models
Pydantic schemas for request/response payloads; no ORM dependency yet.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class Role(str, Enum):
    admin = "admin"
    analyst = "analyst"
    viewer = "viewer"


# ── Stored user representation (in-memory / DB row) ─────────────
class UserRecord(BaseModel):
    user_id: str
    username: str
    email: str
    hashed_password: str
    role: Role = Role.viewer
    is_active: bool = True


# ── API request / response schemas ───────────────────────────────
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int          # seconds
    role: Role


class UserMe(BaseModel):
    user_id: str
    username: str
    email: str
    role: Role
    is_active: bool
