from __future__ import annotations

import pytest

from tests.helpers import require_dep

require_dep("pydantic_settings")
from tripmate.config import INSECURE_JWT_DEFAULT, Settings  # noqa: E402


def prod(**kw):
    base = {
        "env": "production",
        "jwt_secret": "x" * 40,
        "database_url": "postgresql+psycopg://u:p@h/db",
        "checkpoint_postgres_url": "postgresql://u:p@h/db",
        "groq_api_key": "k",
        "_env_file": None,
    }
    base.update(kw)
    return Settings(**base)


def test_dev_defaults_work_without_any_env():
    s = Settings(_env_file=None)
    assert not s.is_production and s.jwt_secret.get_secret_value() == INSECURE_JWT_DEFAULT


def test_production_happy_path():
    assert prod().is_production


@pytest.mark.parametrize(
    "override,needle",
    [
        ({"jwt_secret": INSECURE_JWT_DEFAULT}, "JWT_SECRET"),
        ({"jwt_secret": "short"}, "JWT_SECRET"),
        ({"database_url": "sqlite+aiosqlite:///x.db"}, "PostgreSQL"),
        ({"checkpoint_postgres_url": None}, "CHECKPOINT"),
        ({"cors_origins": ("*",)}, "CORS"),
        ({"groq_api_key": None}, "GROQ_API_KEY"),
    ],
)
def test_unsafe_production_configs_are_rejected(override, needle):
    with pytest.raises(ValueError, match=needle):
        prod(**override)


def test_csv_env_parsing(monkeypatch):
    monkeypatch.setenv("TRIPMATE_LLM_SMART_MODELS", "groq/a, gemini/b ,")
    assert Settings(_env_file=None).llm_smart_models == ("groq/a", "gemini/b")


def test_secrets_do_not_leak_in_repr():
    assert "x" * 40 not in repr(prod())
