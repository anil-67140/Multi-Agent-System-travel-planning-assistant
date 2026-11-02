"""Application error hierarchy.

Every error that can reach an API client derives from :class:`TripMateError`
and carries a stable machine-readable ``code`` plus a *safe* public message.
Internal details (stack traces, provider payloads, SQL) are logged, never
returned to the client.
"""

from __future__ import annotations


class TripMateError(Exception):
    code: str = "internal_error"
    http_status: int = 500
    public_message: str = "Something went wrong on our side. Please try again."

    def __init__(self, message: str | None = None, *, detail: str | None = None) -> None:
        super().__init__(message or self.public_message)
        self.message = message or self.public_message
        # ``detail`` is for logs only.
        self.detail = detail


class BadRequestError(TripMateError):
    code = "bad_request"
    http_status = 400
    public_message = "The request could not be understood."


class GuardrailBlockedError(BadRequestError):
    code = "guardrail_blocked"
    public_message = "TripMate can only help with travel-planning requests."


class UnauthorizedError(TripMateError):
    code = "unauthorized"
    http_status = 401
    public_message = "Authentication is required."


class ForbiddenError(TripMateError):
    code = "forbidden"
    http_status = 403
    public_message = "You do not have access to this resource."


class NotFoundError(TripMateError):
    code = "not_found"
    http_status = 404
    public_message = "The requested resource was not found."


class ConflictError(TripMateError):
    code = "conflict"
    http_status = 409
    public_message = "The request conflicts with the current state of the resource."


class ValidationFailure(TripMateError):
    code = "validation_failed"
    http_status = 422
    public_message = "The request failed validation."


class RateLimitedError(TripMateError):
    code = "rate_limited"
    http_status = 429
    public_message = "Too many requests. Please slow down."

    def __init__(self, message: str | None = None, *, retry_after: int = 1) -> None:
        super().__init__(message)
        self.retry_after = max(1, int(retry_after))


class UpstreamUnavailableError(TripMateError):
    code = "upstream_unavailable"
    http_status = 503
    public_message = "A required upstream service is temporarily unavailable."


class ProviderError(TripMateError):
    """A data provider (weather, FX, search ...) failed.

    ``retryable`` tells the resilience layer whether another attempt is
    sensible; ``retry_after`` carries a server-supplied back-off hint.
    """

    code = "provider_error"
    http_status = 502
    public_message = "A data provider failed."

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        retryable: bool = True,
        retry_after: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(f"{provider}: {message}", detail=message)
        self.provider = provider
        self.retryable = retryable
        self.retry_after = retry_after
        self.status_code = status_code


class LLMError(TripMateError):
    code = "llm_error"
    http_status = 502
    public_message = "The language model could not produce a valid answer."


class LLMOutputError(LLMError):
    """The model answered, but not with data matching the requested schema."""

    code = "llm_invalid_output"
