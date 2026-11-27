#!/bin/bash
# Serve Cloudflare Clef, a decision model, from the 2080 Ti: POST /v1/systemone (the
# Jev/SystemOne API; docs/clef.md has the request format). It does not chat, so it is not
# a model for Open WebUI's chat; scripts/serve-switch.sh puts it on the same address as
# Inkling and loads whichever a request names.
#
#   scripts/setup-clef.sh                     # once: venv + download (+ NF4 copy)
#   scripts/clef.sh                           # foreground, 127.0.0.1:18090
#   CLEF_MODEL=clef-flash scripts/clef.sh     # the 9B
#   BIND=0.0.0.0 PORT=8090 API_KEY=secret scripts/clef.sh
#
# Check it:
#   curl -s http://127.0.0.1:18090/v1/systemone -H 'Content-Type: application/json' -d \
#     '{"model":"clef","state":"The site is down!","questions":{"outage":{"type":"noul"}}}'
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- which model ----------------------------------------------------------------------
# clef = 27B (Qwen3.8-27B backbone), clef-flash = 9B (Qwen3.5-9B backbone). Same API.
export CLEF_MODEL=${CLEF_MODEL:-clef}
# The checkpoint. The NF4 copy setup-clef.sh writes loads in about half the time and needs
# no bf16 download kept around; the original is quantized at every load otherwise.
if [ -z "${CLEF_PATH:-}" ]; then
	CLEF_PATH=models/$CLEF_MODEL
	[ -d "models/$CLEF_MODEL-nf4" ] && CLEF_PATH=models/$CLEF_MODEL-nf4
fi
export CLEF_PATH
# The name(s) it answers to, comma-separated (the request's "model" is echoed back).
export CLEF_NAME=${CLEF_NAME:-$CLEF_MODEL}

# --- memory ---------------------------------------------------------------------------
# Weights: nf4 (4-bit, ~19 GB for the 27B, ~8.5 GB for the 9B), int8 (~30 / ~12 GB: the
# 27B does not fit), none (fp16: only the 9B, ~19 GB). The LM head and the vision tower
# stay fp16 in every mode (the decision head reads the LM head's rows). Ignored for a
# pre-quantized CLEF_PATH. Estimates from the parameter counts, not measured on the card.
export CLEF_QUANT=${CLEF_QUANT:-nf4}
# Token embeddings in host RAM (1 = 2.5 GB less on the GPU for the 27B, 2 GB for the 9B;
# the lookup is a few rows per token). The first thing to try on "out of GPU memory".
export CLEF_EMBED_CPU=${CLEF_EMBED_CPU:-0}
# Longest input in tokens (state + questions + images). Longer states are cut at the end,
# as the release does; its default is 16384. Memory for one forward grows with it.
export CLEF_MAX_LEN=${CLEF_MAX_LEN:-16384}
# float16 on Turing (no bf16 tensor cores). Every answer checks the largest value the
# layers produce and refuses one that overflowed; bfloat16 avoids that, much slower here.
export CLEF_DTYPE=${CLEF_DTYPE:-float16}

# --- network --------------------------------------------------------------------------
# 127.0.0.1 by default: behind serve-switch.sh, which holds the public port and the key.
export BIND=${BIND:-127.0.0.1}
export PORT=${PORT:-18090}
export API_KEY=${API_KEY:-}

# --- checks in the log ----------------------------------------------------------------
# A fixed request at boot (expected: technical / urgency ~2 / outage true), and optionally
# one of about this many tokens, with its time and peak memory: what to send back.
export CLEF_SELFTEST=${CLEF_SELFTEST:-1}
export CLEF_SELFTEST_TOKENS=${CLEF_SELFTEST_TOKENS:-0}

PY=${CLEF_PYTHON:-.local/clef-venv/bin/python}
if [ ! -x "$PY" ]; then
	echo "clef: $PY is missing: run scripts/setup-clef.sh first" >&2
	exit 1
fi
if [ ! -f "$CLEF_PATH/joint_schema_model.py" ]; then
	echo "clef: no checkpoint at $CLEF_PATH: CLEF_MODEL=$CLEF_MODEL scripts/setup-clef.sh" >&2
	exit 1
fi
# The shared Hugging Face cache is not needed: everything is in CLEF_PATH.
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec "$PY" scripts/clef_server.py
