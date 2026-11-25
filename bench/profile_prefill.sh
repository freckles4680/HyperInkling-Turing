#!/bin/bash
# Kernel time of one prefill under nsys, with start.sh's own settings. One server on PORT
# (18030), an unmeasured prompt of the same length (kernels compile on first use), then a
# profiled prompt the server has not seen (max_tokens 1), and the top kernels per 1000
# prompt tokens.
#
#   bench/profile_prefill.sh <name> [tokens=8192] [EXTRA_ARGS for start.sh]
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/server-lib.sh
name=$1; tokens=${2:-8192}; extra=${3:-}
PORT=${PORT:-18030}; log=.local/prof-$name.log
mkdir -p .local/prof
start_server "$PORT" "$log" PROFILE=$name PORT=$PORT KV_OFFLOAD_GB=0 WARMUP=0 ENV_FILE= MAX_SEQS=${MAX_SEQS:-1} EXTRA_ARGS="$extra" ./start.sh
wait_health "$PORT" || { echo "server failed: $log"; exit 1; }
prefill() {  # prefill <seed>: prompt tokens and seconds
	PYTHONPATH=bench venv/bin/python -c "
import sys, time
import context_bench as cb
base = 'http://127.0.0.1:$PORT'
content, n = cb.Prompts(base).make($tokens, $1)
t0 = time.perf_counter()
cb._at_once(base, [content], 1)
print(n, time.perf_counter() - t0)
" | tail -1
}
prefill 1 > /dev/null
curl -s -X POST "localhost:$PORT/start_profile"
read -r n seconds < <(prefill 2)
curl -s -X POST --max-time 600 "localhost:$PORT/stop_profile"
stop_server
nsys stats -q --report cuda_gpu_kern_sum --format csv ".local/prof/$name.nsys-rep" 2> /dev/null | python3 -c "
import csv, sys
rows = list(csv.DictReader(sys.stdin)); k = $n / 1000
tot = sum(float(r['Total Time (ns)']) for r in rows)
# Wall time includes the profiler's overhead; the gap to the kernel time is the GPU idle.
print(f'$n tokens in $seconds s: {$n / $seconds:.0f} tok/s, wall {$seconds / k * 1e3:.1f} ms per 1000 tokens, kernels {tot / k / 1e6:.1f} ms')
for r in rows[:${TOP:-30}]:
    print(f\"{float(r['Total Time (ns)']) / k / 1e6:8.2f} ms/1k {int(r['Instances']) / k:8.1f}x {float(r['Avg (ns)']) / 1e3:9.1f}us  {r['Name'][:90]}\")
"
