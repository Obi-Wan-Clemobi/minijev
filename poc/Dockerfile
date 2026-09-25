# minijev API: FastAPI + the Qwen model on the CPU. Build from the repository root (the POC depends on ../src/minijev):
#   docker build -f poc/Dockerfile .       (docker-compose.yml does this)
# Model weights live in the HF cache volume (see docker-compose.yml).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 HF_HOME=/hf PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY poc/pyproject.toml poc/uv.lock ./poc/
WORKDIR /app/poc
RUN uv sync --frozen --no-dev
COPY poc/minijev_poc.py poc/experiments.py poc/server.py poc/data.py poc/minijev.env ./
COPY poc/datasets ./datasets
COPY poc/templates ./templates
COPY poc/results ./results
COPY poc/calibration ./calibration
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "minijev", "serve", "--host", "0.0.0.0", "--port", "8000"]
