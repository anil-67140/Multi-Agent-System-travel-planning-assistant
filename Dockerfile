# syntax=docker/dockerfile:1.7
# ---- build stage: wheels only, so the runtime image has no compiler or pip cache ----
FROM python:3.12-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels ".[scale,observability]"

# ---- runtime ----
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    TRIPMATE_ENV=production TRIPMATE_LOG_JSON=true PORT=8000
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links=/wheels tripmate && rm -rf /wheels
COPY alembic.ini ./
COPY migrations ./migrations
COPY evals ./evals
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz',timeout=3)" || exit 1
# Run migrations separately (docker compose `migrate` service / release command), never at app start,
# so several replicas do not race on the schema.
CMD ["tripmate-api"]
