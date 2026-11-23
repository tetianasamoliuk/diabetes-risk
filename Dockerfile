FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Проєкт не встановлюємо (потрібен лише src/ через PYTHONPATH), лише залежності
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

FROM python:3.12-slim
# Без обмеження арен malloc пам'ять процесу росте з кожним сеансом, що рахує на великому файлі користувача, і не повертається системі
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    TZ=Europe/Kyiv
RUN useradd --create-home app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app .streamlit/ .streamlit/
COPY --chown=app:app src/ src/
COPY --chown=app:app app/ app/
COPY --chown=app:app data/processed/ data/processed/
COPY --chown=app:app data/external/ data/external/
COPY --chown=app:app models/ models/
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/Головна.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
