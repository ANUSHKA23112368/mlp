"""
CivicShield Shared — ASGI Middleware
Adds:
  • Request-ID header injection (X-Request-ID)
  • Structured request/response logging
  • CORS (configured from env)
  • Basic rate-limit headers (slowapi integration point)
"""
from __future__ import annotations
import os
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from civicshield.shared.logging import get_logger

log = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique X-Request-ID into every request/response."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        # Attach to structlog context for this request
        import structlog
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers["X-Request-ID"] = request_id
        log.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response


def register_middleware(app: FastAPI) -> None:
    """Apply all middleware to a FastAPI app instance."""
    cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501")
    cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)
