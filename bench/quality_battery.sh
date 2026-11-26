#!/bin/bash
# bench/quality_battery.py on its own server with start.sh's defaults (not the .env), the
# settings of the README's "Task accuracy" table. Downloads the data on the first run.
#
#   bench/quality_battery.sh <name> [tasks]     # GPU must be free; N=200 JOBS=6
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/server-lib.sh
name=${1:?usage: bench/quality_battery.sh <name> [tasks]}; shift
PORT=${PORT:-18030}
[ -d bench/quality-data/gsm8k ] || venv/bin/python bench/quality_battery.py download || exit 1
log=.local/quality-$name.log
start_server "$PORT" "$log" PORT=$PORT MAX_LEN=${MAX_LEN:-32768} KV_OFFLOAD_GB=0 WARMUP=0 ENV_FILE= ./start.sh
if wait_health "$PORT"; then
	venv/bin/python bench/quality_battery.py run "$@" --port "$PORT" --n "${N:-200}" \
		--jobs "${JOBS:-6}" --label "$name"
	echo "README, HyperInkling as served (3090 Ti): gsm8k 0.970, tools 0.615 (names 0.710)," \
		"docvqa 0.935 (json 0.995); noise about 1 to 3 points at n=200"
else
	echo "server failed: $log"; tail -5 "$log"
fi
stop_server
grep -ho "HyperInkling: sm75.*\|^Turing:.*\|clamped.*" "$log" | sort -u
