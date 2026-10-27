"""
CivicShield Shared — Structured JSON Logging
Configures structlog to emit JSON-formatted log records compatible with
log-aggregation systems (ELK, Cloud Logging, etc.).

Usage
-----
    from civicshield.shared.logging import get_logger
    log = get_logger(__name__)
    log.info("event", user_id="u1", action="predict")
"""
from __future__ import annotations
import logging
import os
import sys

import structlog


def configure_logging() -> None:
    """Call once at application startup."""
    log_level_name: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level: int = getattr(logging, log_level_name, logging.INFO)
    log_format: str = os.getenv("LOG_FORMAT", "json").lower()

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)


def get_logger(name: str = __name__):
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
