#!/bin/bash
# Kernel time per decode step at a long context under nsys, with start.sh's own settings.
# One server on PORT (18030), a prompt of <tokens> prefilled twice unprofiled (the second
# a prefix hit: the decode kernels compile), then a profiled decode of <gen> tokens on the
# same prompt, and the top kernels per engine step with the tokens accepted per step.
#
#   bench/profile_decode.sh <name> [tokens=90000] [gen=256] [EXTRA_ARGS for start.sh]
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/server-lib.sh
name=$1; tokens=${2:-90000}; gen=${3:-256}; extra=${4:-}
PORT=${PORT:-18030}; log=.local/prof-$name.log
mkdir -p .local/prof
start_server "$PORT" "$log" PROFILE=$name PORT=$PORT KV_OFFLOAD_GB=0 WARMUP=0 ENV_FILE= MAX_SEQS=${MAX_SEQS:-1} EXTRA_ARGS="$extra" ./start.sh
wait_health "$PORT" || { echo "server failed: $log"; exit 1; }
metric() { curl -s "localhost:$PORT/metrics" | awk -v m="vllm:$1{" 'index($0,m)==1 {s+=$NF} END {print s+0}'; }
decode() {  # decode <max_tokens>: prompt tokens
	PYTHONPATH=bench venv/bin/python -c "
import context_bench as cb
base = 'http://127.0.0.1:$PORT'
content, n = cb.Prompts(base).make($tokens, 1)
cb._at_once(base, [content], $1)
print(n)
" | tail -1
}
decode 1 > /dev/null
decode "$gen" > /dev/null
curl -s -X POST "localhost:$PORT/start_profile"
s0=$(metric iteration_tokens_total_count); g0=$(metric generation_tokens_total); t0=$(date +%s.%N)
n=$(decode "$gen")
s1=$(metric iteration_tokens_total_count); g1=$(metric generation_tokens_total); t1=$(date +%s.%N)
curl -s -X POST --max-time 600 "localhost:$PORT/stop_profile"
stop_server
grep -ho "HyperInkling: sm75.*\|^Turing:.*" "$log" | sort -u
nsys stats -q --report cuda_gpu_kern_sum --format csv ".local/prof/$name.nsys-rep" 2> /dev/null | python3 -c "
import csv, sys
rows = list(csv.DictReader(sys.stdin)); steps = max($s1 - $s0, 1); gen = $g1 - $g0
tot = sum(float(r['Total Time (ns)']) for r in rows)
wall = ($t1 - $t0) / steps * 1e3
# Wall time includes the profiler's overhead and the prefix lookup; the gap to the kernel
# time is the GPU idle.
print(f'context $n tokens: {gen:.0f} tokens in {steps} steps ({gen / steps:.2f} tokens/step), wall {wall:.2f} ms/step, kernels {tot / steps / 1e6:.2f} ms/step')
for r in rows[:${TOP:-30}]:
    print(f\"{float(r['Total Time (ns)']) / steps / 1e3:8.1f} us/step {int(r['Instances']) / steps:6.1f}x {float(r['Avg (ns)']) / 1e3:7.1f}us  {r['Name'][:90]}\")
"
