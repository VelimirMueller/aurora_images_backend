# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev \
 && mkdir -p models && .venv/bin/python scripts/fetch_model.py

FROM python:3.12-slim
RUN useradd --system --uid 10001 --home /app app
WORKDIR /app
COPY --from=build --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 AURORA_UPLOAD_DIR=/data/uploads
RUN mkdir -p /data/uploads && chown app:app /data/uploads
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "aurora_images.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
