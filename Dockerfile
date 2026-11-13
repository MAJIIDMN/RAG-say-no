# CPU image for the API. Default extras run the offline toy config; add `ml` for real embedders:
#   docker build --build-arg EXTRAS="--extra llm --extra api --extra ml" -t ragkit .
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH

ARG EXTRAS="--extra llm --extra api"
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project $EXTRAS
COPY src ./src
RUN uv sync --frozen --no-dev $EXTRAS

COPY configs ./configs
COPY data/toy ./data/toy
RUN useradd --create-home app && mkdir -p artifacts && chown app artifacts
USER app

ENV CONFIG=configs/toy.yaml
EXPOSE 8000
# The index is cached by config hash, so ingest is a no-op after the first start.
CMD ["sh", "-c", "ragkit ingest -c \"$CONFIG\" >/dev/null && exec ragkit serve -c \"$CONFIG\" --host 0.0.0.0 --port 8000"]
