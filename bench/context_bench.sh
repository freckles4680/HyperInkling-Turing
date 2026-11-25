#!/bin/bash
# bench/context_bench.py with and without DSpark, one server each with start.sh's defaults
# (not the .env), no CPU tier. Then the README table.
#
#   bench/context_bench.sh <name>        # labels <name>-off, <name>-dspark; GPU must be free
#   SPECS=dspark CONTEXTS="8192 90000" CONC=1 MAX_SEQS=1 bench/context_bench.sh <name>
#       one long request: MAX_LEN stays auto, and the context that fits is printed
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/server-lib.sh
name=$1; PORT=${PORT:-18030}
SPECS=${SPECS:-off dspark}
opts=()
[ -n "${CONTEXTS:-}" ] && opts+=(--contexts $CONTEXTS)
[ -n "${CONC:-}" ] && opts+=(--conc $CONC)
for spec in $SPECS; do
	log=.local/context-bench-$name-$spec.log
	start_server "$PORT" "$log" SPEC=$spec PORT=$PORT KV_OFFLOAD_GB=0 WARMUP=0 ENV_FILE= ./start.sh
	if wait_health "$PORT"; then
		grep -oE 'reduced max_model_len from [0-9]+ to [0-9]+|GPU KV cache size: [0-9,]+ tokens' "$log" | tail -2
		venv/bin/python bench/context_bench.py run "$name-$spec" --port "$PORT" "${opts[@]}"
	else
		echo "server failed: $log"
	fi
	stop_server
done
# shellcheck disable=SC2046
venv/bin/python bench/context_bench.py table $(for spec in $SPECS; do echo "$name-$spec"; done)
