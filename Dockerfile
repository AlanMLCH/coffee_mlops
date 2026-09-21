# Prediction API. Pipelines run natively with uv; only services are containerized.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# LightGBM links against the OpenMP runtime, which the slim image does not ship.
RUN apt-get update     && apt-get install -y --no-install-recommends libgomp1     && rm -rf /var/lib/apt/lists/*

# Dependencies first: they change far less often than the source.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra serving --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra serving --no-dev

# Never serve as root. /cache exists in the image so the named volume mounted there
# inherits its ownership instead of being created as root.
RUN useradd --create-home --uid 10001 api     && mkdir /cache     && chown -R api:api /app /cache
USER api

EXPOSE 8000
CMD ["uvicorn", "mlops_core.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
