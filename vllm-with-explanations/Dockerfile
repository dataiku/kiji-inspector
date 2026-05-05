FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV PATH="/root/.local/bin:/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    patch \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml uv.lock README.md ./
COPY patches ./patches

RUN uv sync --frozen --no-dev --no-install-project \
    && PYTHON=.venv/bin/python patches/0.20.1/apply_patch.sh

COPY activation_processor_sidecar.py ./
COPY docker-entrypoint.sh ./

RUN uv run --script activation_processor_sidecar.py --help >/dev/null \
    && chmod +x docker-entrypoint.sh activation_processor_sidecar.py

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
