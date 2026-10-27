"""
Tests for CivicShield Auth endpoints.
Covers: login success, login failure, /auth/me, /auth/logout, RBAC.
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app, raise_server_exceptions=True)


# ── Helpers ───────────────────────────────────────────────────────────────
def _login(username: str, password: str) -> dict:
    resp = client.post("/auth/login", json={"username": username, "password": password})
    return resp


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Login ─────────────────────────────────────────────────────────────────
def test_login_admin_success():
    resp = _login("admin", "admin123")
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["role"] == "admin"
    assert body["expires_in"] > 0


def test_login_analyst_success():
    resp = _login("analyst", "analyst123")
    assert resp.status_code == 200
    assert resp.json()["role"] == "analyst"


def test_login_viewer_success():
    resp = _login("viewer", "viewer123")
    assert resp.status_code == 200
    assert resp.json()["role"] == "viewer"


def test_login_wrong_password():
    resp = _login("admin", "wrongpassword")
    assert resp.status_code == 401


def test_login_unknown_user():
    resp = _login("nobody", "doesnotmatter")
    assert resp.status_code == 401


def test_login_missing_fields():
    resp = client.post("/auth/login", json={"username": "admin"})
    assert resp.status_code == 422   # Unprocessable Entity


# ── /auth/me ──────────────────────────────────────────────────────────────
def test_me_returns_user_info():
    token = _login("admin", "admin123").json()["access_token"]
    resp = client.get("/auth/me", headers=_bearer(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "admin"
    assert body["role"] == "admin"
    assert body["is_active"] is True


def test_me_without_token_is_401():
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_me_with_invalid_token_is_401():
    resp = client.get("/auth/me", headers=_bearer("not.a.valid.jwt"))
    assert resp.status_code == 401


# ── /auth/logout ──────────────────────────────────────────────────────────
def test_logout_success():
    token = _login("viewer", "viewer123").json()["access_token"]
    resp = client.post("/auth/logout", headers=_bearer(token))
    assert resp.status_code == 200
    assert "Logged out" in resp.json()["detail"]


# ── Response headers ──────────────────────────────────────────────────────
def test_request_id_header_present():
    resp = client.get("/auth/me")   # will 401 but headers still set
    assert "x-request-id" in resp.headers


def test_custom_request_id_echoed():
    custom_id = "test-req-12345"
    resp = client.get(
        "/auth/me",
        headers={"X-Request-ID": custom_id},
    )
    assert resp.headers.get("x-request-id") == custom_id
