FROM node:24-slim AS frontend
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json vite.config.ts ./
COPY frontend ./frontend
RUN npm run build


FROM python:3.14-slim AS deps
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


FROM python:3.14-slim
ENV PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY --from=deps /app/.venv ./.venv
COPY paths.py ./
COPY api ./api
COPY db ./db
COPY indexing ./indexing
COPY llm ./llm
COPY sql ./sql
COPY templates ./templates
COPY frontend ./frontend
COPY --from=frontend /app/static ./static

VOLUME /app/uploads
EXPOSE 8000

# Jobs run in-process, so keep a single worker
CMD ["sh", "-c", "python -m db.setup && exec uvicorn api.server:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'"]
