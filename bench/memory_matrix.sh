#!/bin/bash
# The longest request each knob combination can serve, measured: one boot of start.sh with
# MAX_LEN=auto, which vLLM fits after warmup and CUDA graph capture. Feeds the wizard's
# table (scripts/wizard.py).
#
#   bench/memory_matrix.sh [out.tsv]        # ~2 min per combination, GPU must be free
#   COMBOS="gpu:dspark:1:6:1024" bench/memory_matrix.sh x.tsv   # vision:spec:async:seqs:nbt
#
# The defaults otherwise (ENV_FILE=: not the .env). Columns: vision spec async max_seqs nbt
# max_len_fits pool_tokens weights_gib gpu_gib (the GPU's memory, rounded: the wizard scales
# other sizes from the nearest measured one)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
OUT=${1:-.local/memory_matrix.tsv}; PORT=${PORT:-18030}
mkdir -p .local/memmatrix
source scripts/server-lib.sh
mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_GIB=$(( (mib + 512) / 1024 ))
COMBOS=${COMBOS:-"gpu:dspark:1:6 gpu:dspark:0:6 gpu:off:1:6 gpu:off:0:6 host:dspark:1:6 host:off:1:6 none:dspark:1:6 none:off:1:6 gpu:dspark:1:8 gpu:dspark:0:8 gpu:off:1:8 host:dspark:1:8 none:dspark:1:8 host:off:1:8 none:off:1:8 gpu:dspark:1:1 gpu:dspark:1:16 gpu:dspark:1:32 gpu:dspark:1:64 gpu:off:1:64 gpu:dspark:1:6:1024 gpu:dspark:0:6:1024 gpu:off:1:6:1024"}
[ -f "$OUT" ] || printf 'vision\tspec\tasync\tmax_seqs\tnbt\tmax_len_fits\tpool_tokens\tweights_gib\tgpu_gib\n' > "$OUT"
for c in $COMBOS; do
	IFS=: read -r vision spec async seqs nbt <<< "$c"
	nbt=${nbt:-2048}
	log=.local/memmatrix/$vision-$spec-a$async-s$seqs-n$nbt.log
	env_args=(ENV_FILE= PORT=$PORT KV_OFFLOAD_GB=0 WARMUP=0 SPEC=$spec ASYNC=$async MAX_SEQS=$seqs NBT=$nbt)
	case $vision in
	host) env_args+=(HYPERINKLING_VISION_HOST=1) ;;
	none) env_args+=(IMAGES=0) ;;
	esac
	start_server "$PORT" "$log" "${env_args[@]}" MAX_LEN=auto ./start.sh
	for _ in $(seq 200); do
		grep -qE 'GPU KV cache size|Engine core initialization failed' "$log" && break
		server_alive || break
		sleep 3
	done
	fits=$(grep -oE 'reduced max_model_len from [0-9]+ to [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
	pool=$(grep -m1 -oE 'GPU KV cache size: [0-9,]+' "$log" | grep -oE '[0-9,]+$' | tr -d ,)
	weights=$(grep -m1 -oE 'Model loading took [0-9.]+ GiB' "$log" | grep -oE '[0-9.]+')
	stop_server
	printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$vision" "$spec" "$async" "$seqs" "$nbt" "${fits:-?}" "${pool:--}" "${weights:-?}" "$GPU_GIB" | tee -a "$OUT"
done
