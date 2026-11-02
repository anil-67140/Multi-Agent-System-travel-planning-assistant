.PHONY: install lint format typecheck test eval security run worker migrate up down

install:
	pip install -e ".[dev,scale,observability]"
lint:
	ruff check . && ruff format --check .
format:
	ruff check --fix . && ruff format .
typecheck:
	mypy src
test:
	REQUIRE_ALL_DEPS=1 pytest --cov=tripmate --cov-report=term-missing
eval:
	tripmate-eval
security:
	bandit -r src -ll -q && pip-audit --skip-editable
run:
	tripmate-api
worker:
	tripmate-worker
migrate:
	alembic upgrade head
up:
	docker compose up --build
down:
	docker compose down -v
