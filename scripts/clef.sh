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
# Everything stays on the GPU. For the 27B on a 22 GiB card (bench/clef_memory.py, from the
# real module tree; the boot log has the measured numbers):
#   NF4 layers 11.70 + INT8 embeddings/LM head 2.37 + vision 0.86 + joint head 0.24
#   + CUDA context ~0.4 = 15.6 GiB before any input; a 16k-token input adds 2.7 GiB
#   (4k: 0.8, 8k: 1.4), so 18.2 GiB at 16k with 3.8 GiB free. The 9B: 6.7 + 1.9 GiB.
# Weights of the layers: nf4. int8 (8-bit) does not fit the 27B (24 GiB of layers);
# none (fp16) only fits the 9B. Ignored for a pre-quantized CLEF_PATH (the -nf4 copy).
export CLEF_QUANT=${CLEF_QUANT:-nf4}
# Token embeddings and LM head as INT8 tables (int8, 2.4 GiB less for the 27B) or fp16.
# Clef only looks rows up in both (its decision head reads LM head rows, no logits).
export CLEF_TABLES=${CLEF_TABLES:-int8}
# Longest input in tokens (state + questions + images); longer states are cut at the end,
# as the release does (its default: 16384). auto = at boot, run the longest input that
# fits (16384, then 12288, 8192, ... 2048) and keep that, so no request can run out of
# memory later; the log says "FIT: ... up to N tokens", or "DOES NOT FIT" and stops. A
# number = that length or stop.
export CLEF_MAX_LEN=${CLEF_MAX_LEN:-auto}
# GPU memory that must stay free at the longest input (a desktop's windows, a browser).
export CLEF_HEADROOM_GB=${CLEF_HEADROOM_GB:-1.0}
# Linear-attention heads computed at once (8 of 48: 2.3 GiB less at 16k than all at once,
# bitwise the same answer). 0 = all at once.
export CLEF_DELTA_HEADS=${CLEF_DELTA_HEADS:-8}
# Images larger than this are scaled down (one token per 32x32 pixels: 1 MP = 1,024).
export CLEF_IMAGE_PIXELS=${CLEF_IMAGE_PIXELS:-1048576}
# float16 on Turing (no bf16 tensor cores). Every answer checks the largest value the
# layers produce and refuses one that overflowed; bfloat16 avoids that, much slower here.
export CLEF_DTYPE=${CLEF_DTYPE:-float16}

# --- network --------------------------------------------------------------------------
# 127.0.0.1 by default: behind serve-switch.sh, which holds the public port and the key.
export BIND=${BIND:-127.0.0.1}
export PORT=${PORT:-18090}
export API_KEY=${API_KEY:-}

# --- checks in the log ----------------------------------------------------------------
# A fixed request after the fit check (expected: technical / urgency ~2 / outage true).
export CLEF_SELFTEST=${CLEF_SELFTEST:-1}

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
