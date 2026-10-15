# AgentCore Runtime image: CPU-only linux/arm64, dependency versions from uv.lock.
# The runtime-cpu extra retains the existing data APIs while selecting the exact
# upstream PyTorch version's verified CPU distribution. General data installs
# retain their existing dependency source and are not changed by this image.
# Build on a native ARM builder, or an explicitly configured ARM buildx builder.
FROM public.ecr.aws/docker/library/python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93 AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN pip install --no-cache-dir uv==0.8.8
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Cache-free installation keeps wheel archives out of the image. The retained
# xgboost CPU path does not use NCCL; its optional GPU collective library is omitted.
RUN uv sync --frozen --no-cache --no-dev --no-install-project \
      --extra agents --extra runtime-cpu --no-install-package nvidia-nccl-cu12

FROM public.ecr.aws/docker/library/python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93
# OpenMP is required by the retained numerical libraries; build compilers stay
# in the dependency stage. Ownership is applied during COPY, avoiding a second
# full dependency layer from recursively chowning an already-populated /app.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    useradd --create-home --uid 10001 appuser
WORKDIR /app
# Runtime writes relative output beneath /app; change only the parent directory.
RUN chown 10001:10001 /app
COPY --chown=10001:10001 --from=dependencies /app/.venv /app/.venv
COPY --chown=10001:10001 src/ ./src/
COPY --chown=10001:10001 prompts/ ./prompts/
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
# Region and identity arrive from deployment. No named developer profile,
# account identifier, credential or target region is baked into the image.
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ping', timeout=3).close()"]
CMD ["python", "-c", "from amlas.app import app; app.run()"]
