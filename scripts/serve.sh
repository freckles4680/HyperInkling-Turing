#!/bin/bash
# Serve Inkling 2 Mini to the local network from a 22 GB RTX 2080 Ti (Turing), with the
# settings measured on that card. An OpenAI-compatible API: point a front end
# (Open WebUI, Unsloth Studio, any OpenAI client) at  http://<this machine>:<PORT>/v1
#
#   scripts/serve.sh                          # foreground; run it inside tmux to keep it up
#   PORT=9000 API_KEY=secret scripts/serve.sh # anything below can be set on the command line
#
# This is a thin wrapper: it sets the variables below and runs ./start.sh, which holds the
# full list of knobs with their measurements. Edit the defaults here to keep a change.
#
# Check from another machine (replace the address, and drop the header without a key):
#   curl http://192.168.1.50:8080/v1/models -H "Authorization: Bearer $API_KEY"
#
# Firewall: the port must be open to the LAN.
#   Fedora/RHEL:   sudo firewall-cmd --add-port=8080/tcp --permanent && sudo firewall-cmd --reload
#   Ubuntu/Debian: sudo ufw allow from 192.168.1.0/24 to any port 8080 proto tcp
#
# Speed: a desktop on the same GPU costs about 3 ms of every decode step, 11% of prefill and
# 25k tokens of context (measured: 36.5 ms a step headless against 39.7 at 90k context). For
# the fastest server stop the display manager first (this closes the graphical session):
#   sudo systemctl isolate multi-user.target      # back: sudo systemctl isolate graphical.target
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- network ---------------------------------------------------------------------------
# 0.0.0.0 listens on every interface (the LAN). 127.0.0.1 = this machine only.
export HOST=${HOST:-0.0.0.0}
# The port clients connect to. Change it if 8080 is taken (start.sh refuses a busy port).
export PORT=${PORT:-8080}
# A shared secret every client must send ("Authorization: Bearer <key>"; front ends call it
# the API key). Empty = anyone who can reach the port can use the model: fine on a home
# network you trust, not on anything shared. Make one with: openssl rand -hex 16
API_KEY=${API_KEY:-}
# The model name clients ask for. "local" also answers (SERVED_ALIASES in start.sh).
export SERVED_NAME=${SERVED_NAME:-inkling-2-mini}

# --- capacity --------------------------------------------------------------------------
# Requests decoded at the same time. Each slot reserves cache memory, so fewer slots leave
# more context. 2 = one chat plus the side request front ends send (titles, tags). Raise it
# for several users; one request alone is fastest per user.
export MAX_SEQS=${MAX_SEQS:-2}
# Longest request in tokens (prompt + answer). auto = as much as the card holds after
# warmup, printed at boot ("max_model_len"): about 145k headless, 120k with a desktop, at
# one slot. A number caps it (and the boot fails if it does not fit).
export MAX_LEN=${MAX_LEN:-auto}
# Share of the GPU's memory vLLM may use. 0.95 leaves room for a desktop; headless, 0.97
# gives a little more context (not measured on this card: lower it again if the boot runs
# out of memory).
export GPU_UTIL=${GPU_UTIL:-0.95}
# Prefix cache in RAM (/dev/shm), in GiB: a conversation that left the GPU's cache comes
# back without a full prefill. 0 = off, the safe choice with 32 GB of RAM; "auto" sizes it
# from free RAM on machines with more.
export KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-0}
# Images per request, and whether the vision tower lives in host RAM (1 frees GPU memory
# for context at the cost of slower image encoding; not measured on this card).
export IMAGES=${IMAGES:-50}
export HYPERINKLING_VISION_HOST=${HYPERINKLING_VISION_HOST:-0}

# --- speed -----------------------------------------------------------------------------
# Speculative decoding with the bundled DSpark drafter: 2 to 3.5 tokens per step in place of
# one, same output distribution. "off" = plain decoding (to compare, or to debug).
export SPEC=${SPEC:-dspark}
# Verify lengths the server picks from at each step, from what the text accepts and what a
# step costs on this card. 2,4,7 suits mixed use. Empty = the fixed schedule SPEC_K
# (4 drafts for 1-2 requests): steadier step times, for benchmarks.
export HYPERINKLING_SPEC_ADAPTIVE=${HYPERINKLING_SPEC_ADAPTIVE-2,4,7}
# Drafts copied from the request's own context when the answer repeats it (quoting, code
# edits). Lossless; 0 disables.
export HYPERINKLING_LOOKUP=${HYPERINKLING_LOOKUP-1}
# Prefill chunk in tokens. 2048 is the fastest prefill; 1024 frees about 20k tokens of
# context.
export NBT=${NBT:-2048}
# The Turing tensor-core attention kernel (scripts/build-sm75-kernels.sh). start.sh finds
# it at this path by itself; without it the server falls back to a kernel about 4x slower
# at long context, so refuse to start instead.
SM75_LIB=${HYPERINKLING_SM75_LIB-venv/hyperinkling_sm75.so}

# --- sanity ----------------------------------------------------------------------------
# The wizard's .env is skipped so that this file is the whole configuration. Set
# ENV_FILE=.env to layer it underneath (values here still win).
export ENV_FILE=${ENV_FILE-}
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -1 || true)
if [ "$cc" = 7.5 ] && [ -n "$SM75_LIB" ] && [ ! -f "$SM75_LIB" ]; then
	echo "serve: $SM75_LIB is missing: run scripts/build-sm75-kernels.sh" \
		"(or HYPERINKLING_SM75_LIB= to serve on the slow kernel)" >&2
	exit 1
fi
if [ -n "$API_KEY" ]; then export VLLM_API_KEY=$API_KEY; fi
if nvidia-smi --query-compute-apps=name --format=csv,noheader 2> /dev/null | grep -q .; then
	echo "serve: other programs are computing on the GPU (nvidia-smi); the server shares it" >&2
fi
address=$(hostname -I 2> /dev/null | awk '{print $1}')
echo "serve: http://${address:-<this machine>}:$PORT/v1  model '$SERVED_NAME'" \
	"$([ -n "$API_KEY" ] && echo '(API key required)' || echo '(no API key)')"
echo "serve: ready when  curl -sf http://127.0.0.1:$PORT/health  succeeds (a minute or two)"
exec ./start.sh
