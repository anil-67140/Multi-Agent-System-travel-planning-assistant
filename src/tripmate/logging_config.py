"""Structured logging: JSON in production, readable in development, always redacted and request-correlated."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

from tripmate.observability.redaction import redact_text, redaction_processor

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
plan_id_var: ContextVar[str | None] = ContextVar("plan_id", default=None)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.plan_id = plan_id_var.get()
        # Third-party libraries format messages with secrets in them; scrub at the source.
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        return True


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    import structlog

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.ExtraAdder(),
        redaction_processor,
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "LiteLLM", "litellm", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
