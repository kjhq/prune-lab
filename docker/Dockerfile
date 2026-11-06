FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv
RUN uv pip install --system --no-cache numpy==2.2.6 pytest==8.4.2 && adduser --disabled-password --gecos '' runner
USER runner
WORKDIR /work
