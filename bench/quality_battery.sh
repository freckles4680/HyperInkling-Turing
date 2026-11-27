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
# The data are parquet files; pyarrow is not among vLLM's requirements.
venv/bin/python -c "import pyarrow" 2> /dev/null \
	|| uv pip install --python venv/bin/python pyarrow \
	|| venv/bin/python -m pip install pyarrow \
	|| { echo "install pyarrow into venv/ first" >&2; exit 1; }
[ -d bench/quality-data/gsm8k ] || venv/bin/python bench/quality_battery.py download || exit 1
log=.local/quality-$name.log
start_server "$PORT" "$log" PORT=$PORT MAX_LEN=${MAX_LEN:-32768} KV_OFFLOAD_GB=0 WARMUP=0 ENV_FILE= ./start.sh
if wait_health "$PORT"; then
	venv/bin/python bench/quality_battery.py run "$@" --port "$PORT" --n "${N:-200}" \
		--jobs "${JOBS:-6}" --label "$name"
	echo "README, as served on the 22 GB 2080 Ti: gsm8k 0.965, tools 0.605 (names 0.695)," \
		"docvqa 0.936 (json 0.990); noise about 1 to 3 points at n=200"
else
	echo "server failed: $log"; tail -5 "$log"
fi
stop_server
grep -ho "HyperInkling: sm75.*\|^Turing:.*\|clamped.*" "$log" | sort -u
