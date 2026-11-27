#!/bin/bash
# HyperInkling-Turing: Inkling 2 Mini on one Turing GPU (sm75, measured on a 22 GB 2080 Ti),
# single entry point. Turing runs in fp16: see GPU_CC below. A 3090 (sm86) still works.
# scripts/serve.sh wraps this with the 2080 Ti settings for the LAN.
#
#   ./start.sh                      # runs in the foreground
#   PORT=18020 KV_OFFLOAD_GB=0 ./start.sh
#
# Every knob below is an env var with the measured default; ./.env (scripts/wizard.py)
# sets the ones the command line does not. The weights are
# models/Inkling-2-Mini-W4A16-g128-gptq (experts INT4 g128 GPTQ, see prepare/);
# attention, LM head and token embeddings become INT8 at load time.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# The wizard's .env, for the variables not already set (the command line wins). UID/GID
# are only for Docker's prepare service (UID is read-only in bash). ENV_FILE= skips it
# (the benches measure the defaults).
ENV_FILE=${ENV_FILE-.env}
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
	while IFS='=' read -r key value; do
		case "$key" in ''|'#'*|UID|GID) continue ;; esac
		if [ -z "${!key+set}" ]; then export "$key=$value"; fi
	done < "$ENV_FILE"
fi

# The GPU's compute capability (nvidia-smi; GPU_CC=7.5 forces it). Turing (sm75: RTX 20xx,
# 2080 Ti 22 GB mods) has no bf16 tensor cores: the model runs in fp16 there
# (HYPERINKLING_FP16=1 forces that path on any GPU). docs/turing-bringup.md: how to check it.
# DSpark runs in fp16 there too (its bf16 weights cast at load, as the target's).
GPU_CC=${GPU_CC:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)}
if [ "$GPU_CC" = 7.5 ]; then
	export HYPERINKLING_FP16=1
	# Triton has no tensor-core MMA on sm75: the drafter's INT4 Triton linears took
	# ~40 ms a step on a 2080 Ti. fp16 weights on cuBLAS instead (~0.4 GB more), and
	# the Markov fold in INT8 on the dot-free GEMV (half the bytes of fp16).
	: "${HYPERINKLING_DRAFT_W4:=0}" "${HYPERINKLING_DSPARK_FOLD:=w8}"
	# Attention on the fp16 tensor cores, when scripts/build-sm75-kernels.sh has built
	# it (HYPERINKLING_SM75_LIB= for the Triton kernel: 74% of an 8k prefill and
	# ~43 ms of a decode step at 90k tokens on a 2080 Ti).
	if [ -z "${HYPERINKLING_SM75_LIB+x}" ] && [ -f "$REPO/venv/hyperinkling_sm75.so" ]; then
		export HYPERINKLING_SM75_LIB=$REPO/venv/hyperinkling_sm75.so
	fi
fi
export HYPERINKLING_FP16=${HYPERINKLING_FP16:-0}
DTYPE=auto
if [ "$HYPERINKLING_FP16" = 1 ]; then
	DTYPE=float16
	echo "fp16 activations (GPU ${GPU_CC:-?}): validate it as docs/turing-bringup.md says" >&2
fi

MODEL=${MODEL:-$REPO/models/Inkling-2-Mini-W4A16-g128-gptq}
SERVED_NAME=${SERVED_NAME:-inkling-2-mini}
# Extra names the same model answers to: "local" is the backend-neutral id Hermes uses.
SERVED_ALIASES=${SERVED_ALIASES:-local}
# Local only by default. HOST=0.0.0.0 serves the network: set VLLM_API_KEY too, or put
# the server behind something that authenticates (a VPN, a reverse proxy).
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
MAX_SEQS=${MAX_SEQS:-8}
# Speculative decoding with the bundled DSpark drafter (on unless SPEC=off). The
# block drafter always drafts 7 tokens; SPEC_K says how many of them the target
# verifies, by batch size: [first, last, k] ranges. k=4 for
# 1-2 requests, k=3 from 3 (bench/dspark_k_sweep.sh; a verify step touches ~6x the
# experts of a plain step, so long windows only pay on predictable text).
SPEC=${SPEC:-dspark}
SPEC_K=${SPEC_K:-[[1,2,4],[3,8,3]]}
# Drafts sampled from the draft distribution (as the reference) rather than
# greedy: at temperature 0.2-0.5 +15-20% on structured output.
SPEC_DRAFT=${SPEC_DRAFT:-probabilistic}
# The verify length chosen online per step (adaptive_k.py) from these candidates;
# empty = the fixed SPEC_K schedule. Each candidate costs the context of one CUDA graph
# per batch size: 2,4,7 ~5k tokens (still 262k with 6 requests and vision on the GPU),
# 1,2,3,4,5,7 ~14k. With the lookup, against SPEC_K (temperature 0): +43% copying, +10% code
# and struct, +4-5% chat and prose, 6 requests within noise.
export HYPERINKLING_SPEC_ADAPTIVE=${HYPERINKLING_SPEC_ADAPTIVE-2,4,7}
# Drafts from the request's own context in place of DSpark's (ops/lookup.py): +10% on
# top of a long verify length while the answer copies the prompt; lossless. 0 disables.
export HYPERINKLING_LOOKUP=${HYPERINKLING_LOOKUP-1}
# Extra speculative-config fields, as a JSON fragment starting with a comma.
SPEC_OPTS=${SPEC_OPTS:-}
# Longest request: auto = as much as the KV cache holds after warmup (vLLM fits it);
# a number caps it (the boot then fails if it does not fit).
MAX_LEN=${MAX_LEN:-auto}
# Async scheduling keeps two steps in flight: +3-4% decode for ~12k tokens of context
# (each in-flight step holds its tail of the sliding-window and sconv caches).
ASYNC=${ASYNC:-1}
# Prefill chunk: 2048 prefills ~8% faster than 1024 (and encodes two 1024-token images
# per step); 1024 frees ~20k tokens of context.
NBT=${NBT:-2048}
# Most tokens of one prompt per step while other requests are there (none when it is alone):
# short prompts do not wait for a long prefill. 0 disables.
LONG_PREFILL=${LONG_PREFILL:-$((NBT / 2))}
GPU_UTIL=${GPU_UTIL:-0.95}
KV_DTYPE=${KV_DTYPE:-fp8}        # unit-scale E4M3, emulated in software on sm75/sm86 (ops/e4m3.py)
# CPU prefix-cache tier in /dev/shm, GiB; 0 disables. auto: 60% of RAM minus 8 GiB, at most
# 36 and at most what /dev/shm has free (the wizard's default).
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-auto}
# SSE comment every N s while a request waits for its first token: a 262k prefill takes
# over a minute, and clients with a read timeout would drop it. 0 disables.
SSE_KEEPALIVE=${SSE_KEEPALIVE:-30}
IMAGES=${IMAGES:-50}             # images per request
# Largest image profiled: 1,014 tokens (40 px cells), an A4 page at ~128 ppi; NBT=2048
# encodes two per step. Larger images are scaled down to it (HYPERINKLING_IMAGE_MAX_CELLS)
# instead of hanging. A4 at 300 ppi: IMAGE_MAX_W=2480 IMAGE_MAX_H=3508 ENCODER_CACHE=16384.
IMAGE_MAX_W=${IMAGE_MAX_W:-1040}
IMAGE_MAX_H=${IMAGE_MAX_H:-1560}
MM_CACHE_GB=${MM_CACHE_GB:-6}
# Encoder output kept on the GPU, in tokens (~4.6 KB each): ~6 pages. It must hold
# the largest image; vLLM's default (the prefill chunk) would reject a page.
ENCODER_CACHE=${ENCODER_CACHE:-8192}
CPUS=${CPUS:-0-$(( $(nproc) > 1 ? $(nproc) - 2 : 0 ))}   # leave one hardware thread free

[ -d "$MODEL" ] || { echo "[start] model not found: $MODEL" >&2; exit 1; }

# HyperInkling load-time conversions (vllm/models/inkling2/nvidia, branch hyperinkling/sm75).
export HYPERINKLING_ATTN_W8=${HYPERINKLING_ATTN_W8:-1}
export HYPERINKLING_HEAD_W8=${HYPERINKLING_HEAD_W8:-1}
export HYPERINKLING_EMBED_W8=${HYPERINKLING_EMBED_W8:-1}
export HYPERINKLING_EMBED_UVA=${HYPERINKLING_EMBED_UVA:-1}
# Vision tower weights in host RAM, copied in layer by layer per image: ~34k tokens of
# context, +27 ms per A4 page.
export HYPERINKLING_VISION_HOST=${HYPERINKLING_VISION_HOST:-0}
export HYPERINKLING_IMAGE_MAX_CELLS=$(( ((IMAGE_MAX_W + 39) / 40) * ((IMAGE_MAX_H + 39) / 40) ))
# DSpark drafter: layers in INT4 g128 (round to nearest), and its Markov head's
# folded LM-head projection in INT4; the drafts come from the shared INT8 head.
export HYPERINKLING_DRAFT_W4=${HYPERINKLING_DRAFT_W4:-1}
export HYPERINKLING_DSPARK_FOLD=${HYPERINKLING_DSPARK_FOLD:-w4}
# Log a warning when no engine step completes for this many seconds while requests are
# live (a wedged engine is otherwise silent); 0 disables.
export VLLM_ENGINE_STALL_SENTINEL_S=${VLLM_ENGINE_STALL_SENTINEL_S:-60}
# Drafter KV as the target's unit-scale E4M3 (bf16 otherwise): 2 KiB/token back.
export HYPERINKLING_DRAFT_KV_FP8=${HYPERINKLING_DRAFT_KV_FP8:-1}
# nvcc 13.3 rejects gcc 16 (Fedora 44): FlashInfer's JIT builds use gcc-15 where the
# system compiler is newer and gcc-15 is installed (not needed in the container).
if command -v gcc-15 > /dev/null && command -v g++-15 > /dev/null; then
	export CC=gcc-15 CXX=g++-15 NVCC_PREPEND_FLAGS="-ccbin $(command -v g++-15)"
fi

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}$"; then
	echo "[start] port $PORT is already in use: another server is up. Not starting." >&2
	exit 1
fi
if ! pgrep -f '[v]llm serve' > /dev/null; then
	for f in /dev/shm/vllm_offload_*.mmap /dev/shm/VLLM_OBJECT_STORAGE_SHM_BUFFER_*; do
		[ -e "$f" ] || continue
		echo "[start] removing orphaned $f ($(du -h "$f" | cut -f1))"
		rm -f "$f"
	done
fi

if [ "$KV_OFFLOAD_GB" = auto ]; then
	ram=$(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo)
	shm=$(( $(df -B1 --output=avail /dev/shm | tail -n1 | tr -d ' ') / 1073741824 ))
	KV_OFFLOAD_GB=$(( ram * 6 / 10 - 8 ))
	[ "$KV_OFFLOAD_GB" -gt 36 ] && KV_OFFLOAD_GB=36
	[ "$KV_OFFLOAD_GB" -gt "$shm" ] && KV_OFFLOAD_GB=$shm
	[ "$KV_OFFLOAD_GB" -lt 0 ] && KV_OFFLOAD_GB=0
fi

args=(--sse-keep-alive-interval "$SSE_KEEPALIVE")
if [ "$KV_OFFLOAD_GB" != 0 ]; then
	need=$((KV_OFFLOAD_GB * 1073741824))
	avail=$(df -B1 --output=avail /dev/shm | tail -n1 | tr -d ' ')
	if [ "$avail" -lt "$need" ]; then
		echo "[start] KV_OFFLOAD_GB=$KV_OFFLOAD_GB needs that much free in /dev/shm," \
			"only $((avail / 1073741824)) GiB is free." >&2
		exit 1
	fi
	args+=(--kv-offloading-size "$KV_OFFLOAD_GB")
fi
[ -n "${VLLM_API_KEY:-}" ] && args+=(--api-key "$VLLM_API_KEY")
# In the container the port mapping (compose BIND) decides who can reach it.
if [ "$HOST" != 127.0.0.1 ] && [ "$HOST" != localhost ] && [ -z "${VLLM_API_KEY:-}" ] \
	&& [ ! -f /.dockerenv ]; then
	echo "[start] warning: serving $HOST without VLLM_API_KEY: anyone who reaches this port can use it." >&2
fi
[ "$LONG_PREFILL" != 0 ] && args+=(--long-prefill-token-threshold "$LONG_PREFILL")
[ "$ASYNC" = 0 ] && args+=(--no-async-scheduling)
# HYPERINKLING_FP16_CHECK=1: log how close the fp16 residual stream runs to fp16's limit
# (loop.py); it syncs after every layer visit, so no CUDA graphs. Diagnostics only.
[ "${HYPERINKLING_FP16_CHECK:-0}" = 1 ] && args+=(--enforce-eager)
# Decode CUDA graphs by request count: every count up to 8, then coarser steps up to
# MAX_SEQS (a batch runs in the next larger graph; one without a graph runs eagerly).
if [ -z "${GRAPH_SIZES:-}" ]; then
	GRAPH_SIZES=$(for n in 1 2 3 4 5 6 7 8 12 16 24 32 40 48 56 64; do
		[ "$n" -lt "$MAX_SEQS" ] && echo "$n"; done; echo "$MAX_SEQS")
	GRAPH_SIZES=$(echo $GRAPH_SIZES | tr ' ' ,)
fi
case "$SPEC" in
dspark) args+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"num_speculative_tokens_per_batch_size\":$SPEC_K,\"draft_sample_method\":\"$SPEC_DRAFT\"$SPEC_OPTS}") ;;
off) ;;
*) echo "[start] SPEC must be dspark or off, not '$SPEC'" >&2; exit 1 ;;
esac

# PROFILE=<name>: run under Nsight Systems. Nothing is recorded until
# `curl -X POST http://127.0.0.1:$PORT/start_profile`; `/stop_profile` ends the
# capture and writes .local/prof/<name>.nsys-rep (CUDA kernels, including the
# ones replayed inside CUDA graphs, and NVTX).
wrap=()
if [ -n "${PROFILE:-}" ]; then
	mkdir -p "$REPO/.local/prof"
	wrap=(nsys profile -o "$REPO/.local/prof/$PROFILE" --force-overwrite true
		--trace cuda,nvtx --cuda-graph-trace node
		--capture-range cudaProfilerApi --capture-range-end stop)
	args+=(--profiler-config '{"profiler":"cuda"}')
fi

echo "=================================================="
echo "HyperInkling: $(basename "$MODEL") as '$SERVED_NAME' on $HOST:$PORT"
echo "Context $([ "$MAX_LEN" = auto ] && echo "auto (what the GPU holds)" || echo "$MAX_LEN")/request, $MAX_SEQS concurrent, prefill chunk $NBT, async scheduling $ASYNC"
echo "KV $KV_DTYPE, prefix cache on, CPU tier ${KV_OFFLOAD_GB} GiB"
if [ -n "$HYPERINKLING_SPEC_ADAPTIVE" ]; then verify="adaptive over $HYPERINKLING_SPEC_ADAPTIVE"
else verify="by batch size $SPEC_K"; fi
echo "Speculation: $SPEC$([ "$SPEC" = dspark ] && echo ", $SPEC_DRAFT drafts, verify k $verify, context lookup $HYPERINKLING_LOOKUP")"
echo "INT8: attention=$HYPERINKLING_ATTN_W8 head=$HYPERINKLING_HEAD_W8" \
	"embed=$HYPERINKLING_EMBED_W8 (host RAM: $HYPERINKLING_EMBED_UVA)"
[ "$GPU_CC" = 7.5 ] && echo "Turing: fp16, tensor-core attention ${HYPERINKLING_SM75_LIB:-not built (scripts/build-sm75-kernels.sh)}"
echo "Images: up to $IMAGES/request, ${IMAGE_MAX_W}x${IMAGE_MAX_H} profiled, larger ones scaled to $HYPERINKLING_IMAGE_MAX_CELLS cells, vision tower $([ "$HYPERINKLING_VISION_HOST" = 1 ] && echo 'in host RAM' || echo 'on the GPU')"
echo "=================================================="

# The native install's venv; the container has vllm on PATH.
[ -f "$REPO/venv/bin/activate" ] && source "$REPO/venv/bin/activate"
# The CUDA toolkit the venv was built with (scripts/build-vllm.sh records it): FlashInfer
# compiles its JIT kernels at the first boot with $CUDA_HOME/bin/nvcc, else whatever nvcc
# is first on PATH, which may be an older system CUDA. CUDA_HOME set by hand wins.
if [ -z "${CUDA_HOME:-}" ] && [ -f "$REPO/venv/cuda_home" ]; then
	CUDA_HOME=$(cat "$REPO/venv/cuda_home")
fi
if [ -n "${CUDA_HOME:-}" ]; then
	export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
fi
# Requests at the shapes vLLM's warmup leaves out, once /health answers: their kernels
# compile (first boot after an update) or load from the JIT cache before real traffic.
WARMUP=${WARMUP:-1}
if [ "$WARMUP" = 1 ] && [ -z "${PROFILE:-}" ]; then
	python3 "$REPO/scripts/warmup.py" "$PORT" "$SERVED_NAME" "$HOST" &
fi
exec taskset -c "$CPUS" "${wrap[@]}" vllm serve "$MODEL" \
	--served-model-name "$SERVED_NAME" $SERVED_ALIASES --host "$HOST" --port "$PORT" \
	--max-model-len "$([ "$MAX_LEN" = auto ] && echo -1 || echo "$MAX_LEN")" --max-num-seqs "$MAX_SEQS" \
	--max-num-batched-tokens "$NBT" --gpu-memory-utilization "$GPU_UTIL" --dtype "$DTYPE" \
	--kv-cache-dtype "$KV_DTYPE" --enable-prefix-caching \
	--reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
	--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
	--limit-mm-per-prompt "{\"image\":{\"count\":$IMAGES,\"width\":$IMAGE_MAX_W,\"height\":$IMAGE_MAX_H}}" \
	--mm-processor-cache-type shm --mm-processor-cache-gb "$MM_CACHE_GB" \
	--encoder-cache-size "$ENCODER_CACHE" \
	--compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"decode_cudagraph_capture_request_sizes\":[$GRAPH_SIZES]}" \
	--disable-uvicorn-access-log --uvicorn-log-level warning \
	"${args[@]}" ${EXTRA_ARGS:-}
