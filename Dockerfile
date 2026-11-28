# HyperInkling-Turing: Inkling 2 Mini on one Turing GPU (a 22 GB 2080 Ti), as a container.
#
# The vLLM layer is vLLM's own docker/Dockerfile built from our patched checkout for one
# arch, sm75 by default (scripts/build-image.sh does both steps, ARCH=8.6 for a 3090); this
# file adds the launcher on top.
#
#   scripts/build-image.sh                 # ~1 h the first time, then cached
#   docker compose up -d                   # see docker-compose.yml and scripts/wizard.py
ARG VLLM_IMAGE=hyperinkling-vllm:sm75
FROM ${VLLM_IMAGE}

LABEL org.opencontainers.image.source=https://github.com/freckles4680/HyperInkling-Turing \
      org.opencontainers.image.description="Inkling 2 Mini on one Turing GPU, 22 GB 2080 Ti (vLLM, sm75)" \
      org.opencontainers.image.licenses=Apache-2.0
WORKDIR /app
# bench/quality_battery.py and bench/extract_bench.py read parquet.
RUN python3 -m pip install --no-cache-dir pyarrow
COPY start.sh LICENSE README.md ./
COPY scripts/ scripts/
COPY bench/ bench/
COPY docs/ docs/

# Triton, FlashInfer and the encoder JIT caches live under HOME, a volume in compose, so
# a restart does not recompile them.
ENV HOME=/cache VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_XET_HIGH_PERFORMANCE=1 \
    HOST=0.0.0.0 PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python3 -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)" || exit 1
ENTRYPOINT ["/app/start.sh"]
