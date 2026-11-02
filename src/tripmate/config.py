"""Typed configuration from environment variables (12-factor). Secrets are ``SecretStr``."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

INSECURE_JWT_DEFAULT = "dev-only-insecure-secret-change-me-0123456789"


def _csv(value: object) -> object:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return value


CsvTuple = Annotated[tuple[str, ...], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRIPMATE_", env_file=".env", extra="ignore", case_sensitive=False
    )

    env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # --- persistence -------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./tripmate.db"
    # LangGraph checkpoints use a plain libpq URL; empty -> in-memory (dev/test only).
    checkpoint_postgres_url: str | None = None
    redis_url: str | None = None

    # --- security ----------------------------------------------------------
    jwt_secret: SecretStr = SecretStr(INSECURE_JWT_DEFAULT)
    jwt_ttl_minutes: int = Field(default=60, ge=5, le=60 * 24 * 7)
    allow_registration: bool = True
    cors_origins: CsvTuple = ()
    max_body_bytes: int = Field(default=64 * 1024, ge=1024)
    rate_limit_per_minute: int = Field(default=60, ge=1)
    plan_rate_limit_per_hour: int = Field(default=10, ge=1)  # LLM calls cost real quota

    # --- LLM gateway (free tiers) -------------------------------------------
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    tavily_api_key: SecretStr | None = Field(default=None, validation_alias="TAVILY_API_KEY")
    llm_smart_models: CsvTuple = ("groq/openai/gpt-oss-120b", "gemini/gemini-3.8-flash")
    llm_fast_models: CsvTuple = ("groq/openai/gpt-oss-20b", "gemini/gemini-3.5-flash-lite")
    litellm_proxy_url: str | None = None
    litellm_proxy_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=45.0, gt=0)
    llm_max_concurrency: int = Field(default=4, ge=1)

    # --- tools / reliability ------------------------------------------------
    tool_timeout_seconds: float = Field(default=10.0, gt=0)
    tool_max_attempts: int = Field(default=3, ge=1, le=6)
    tool_max_concurrency: int = Field(default=8, ge=1)

    # --- workflow -----------------------------------------------------------
    run_timeout_seconds: float = Field(default=180.0, gt=0)
    stale_run_seconds: int = Field(default=600, ge=60)
    max_revisions: int = Field(default=3, ge=0, le=10)
    queue_backend: Literal["inprocess", "arq"] = "inprocess"
    worker_concurrency: int = Field(default=4, ge=1)

    # --- observability ------------------------------------------------------
    langfuse_public_key: SecretStr | None = Field(
        default=None, validation_alias="LANGFUSE_PUBLIC_KEY"
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_host: str | None = Field(default=None, validation_alias="LANGFUSE_HOST")
    sentry_dsn: SecretStr | None = Field(default=None, validation_alias="SENTRY_DSN")
    metrics_token: SecretStr | None = None  # if set, /metrics requires it as a bearer token

    @field_validator("llm_smart_models", "llm_fast_models", "cors_origins", mode="before")
    @classmethod
    def _split(cls, value: object) -> object:
        return _csv(value)

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @model_validator(mode="after")
    def _production_guards(self) -> Settings:
        if self.is_production:
            problems: list[str] = []
            secret = self.jwt_secret.get_secret_value()
            if secret == INSECURE_JWT_DEFAULT or len(secret) < 32:
                problems.append(
                    "TRIPMATE_JWT_SECRET must be set to a random value of at least 32 characters"
                )
            if self.database_url.startswith("sqlite"):
                problems.append("TRIPMATE_DATABASE_URL must point to PostgreSQL")
            if not self.checkpoint_postgres_url:
                problems.append(
                    "TRIPMATE_CHECKPOINT_POSTGRES_URL is required so workflows survive restarts"
                )
            if "*" in self.cors_origins:
                problems.append("wildcard CORS origins are not allowed")
            if not (self.groq_api_key or self.gemini_api_key or self.litellm_proxy_url):
                problems.append("set GROQ_API_KEY and/or GEMINI_API_KEY")
            if problems:
                raise ValueError("Unsafe production configuration: " + "; ".join(problems))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
