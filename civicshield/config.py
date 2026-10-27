"""CivicShield configuration — loads from environment variables with safe defaults."""
from __future__ import annotations

import os
from pathlib import Path


# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

# ── Application identity ────────────────────────────────────────────────────────
APP_NAME: str = os.getenv("CIVICSHIELD_APP_NAME", "CivicShield AI")
APP_VERSION: str = "0.1.0"
ENVIRONMENT: str = os.getenv("CIVICSHIELD_ENV", "development")   # development | production | demo
DEBUG: bool = os.getenv("CIVICSHIELD_DEBUG", "true").lower() == "true"

# ── Authentication ──────────────────────────────────────────────────────────────
SECRET_KEY: str = os.getenv(
    "CIVICSHIELD_SECRET_KEY",
    "CHANGE_THIS_IN_PRODUCTION_use_at_least_32_random_chars_abc123",
)
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("CIVICSHIELD_TOKEN_EXPIRE_MINUTES", "60"))

# ── Database ────────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "CIVICSHIELD_DATABASE_URL",
    f"sqlite:///{PROJECT_ROOT / 'data' / 'civicshield_dev.db'}",
)

# ── Rate limiting ───────────────────────────────────────────────────────────────
RATE_LIMIT_DEFAULT: str = os.getenv("CIVICSHIELD_RATE_LIMIT", "60/minute")
RATE_LIMIT_PREDICT: str = os.getenv("CIVICSHIELD_RATE_LIMIT_PREDICT", "30/minute")
RATE_LIMIT_AUTH: str = os.getenv("CIVICSHIELD_RATE_LIMIT_AUTH", "10/minute")

# ── CORS ────────────────────────────────────────────────────────────────────────
CORS_ORIGINS: list[str] = os.getenv(
    "CIVICSHIELD_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8501,http://127.0.0.1:3000",
).split(",")

# ── Security monitoring ─────────────────────────────────────────────────────────
SECURITY_EVENT_RETENTION_DAYS: int = int(os.getenv("CIVICSHIELD_SEC_RETENTION_DAYS", "90"))
FAILED_LOGIN_THRESHOLD: int = int(os.getenv("CIVICSHIELD_FAILED_LOGIN_THRESHOLD", "5"))
RATE_ABUSE_THRESHOLD: int = int(os.getenv("CIVICSHIELD_RATE_ABUSE_THRESHOLD", "100"))
ANOMALY_WINDOW_SECONDS: int = int(os.getenv("CIVICSHIELD_ANOMALY_WINDOW_SECS", "300"))

# ── ML settings ─────────────────────────────────────────────────────────────────
DEFAULT_ALGORITHM: str = os.getenv("CIVICSHIELD_ML_ALGORITHM", "logistic_regression")
MODEL_STORE_DIR: Path = PROJECT_ROOT / "model_store"
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", f"file:///{PROJECT_ROOT / 'mlruns'}")

# ── Streaming ───────────────────────────────────────────────────────────────────
SSE_MAX_QUEUE_SIZE: int = int(os.getenv("CIVICSHIELD_SSE_QUEUE_SIZE", "500"))
SSE_HEARTBEAT_SECONDS: int = int(os.getenv("CIVICSHIELD_SSE_HEARTBEAT", "15"))


def is_production() -> bool:
    return ENVIRONMENT.lower() == "production"


def is_demo() -> bool:
    return ENVIRONMENT.lower() == "demo"
