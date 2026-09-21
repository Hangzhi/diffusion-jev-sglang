FROM nvidia/cuda:13.0.1-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git libnuma1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
ENV UV_LINK_MODE=copy UV_PYTHON=3.12
RUN uv sync --locked --extra engine --no-dev
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "diffusion-jev", "serve", "--host", "0.0.0.0"]
