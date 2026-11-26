#!/bin/bash
# Turing (sm75): every measurement of the port in one file to share,
# .local/turing-report-<name>.txt (about 20 minutes; the GPU must be free).
#
#   scripts/turing-report.sh <name>
#   STEPS="system test prefill decode bench" CONTEXT=90000 scripts/turing-report.sh <name>
#
# system:  the commit, the GPU (clocks, temperature, power limit, PCIe link), other GPU users
# test:    tests/test_sm75_attention.py with its timings
# prefill: bench/profile_prefill.sh, kernel time of an 8k prefill
# decode:  bench/profile_decode.sh, kernel time per decode step at 8k and at CONTEXT
# bench:   bench/context_bench.sh at 8k and CONTEXT (prefill and decode tok/s, fitted context)
# The GPU's temperature, clock and power are sampled every 5 s meanwhile: a throttled card
# shows in the summary at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
name=${1:?usage: scripts/turing-report.sh <name>}
STEPS=${STEPS:-system test prefill decode bench}
CONTEXT=${CONTEXT:-90000}
mkdir -p .local
out=.local/turing-report-$name.txt
samples=.local/turing-report-$name.gpu.csv
section() { printf '\n===== %s (%s) =====\n' "$1" "$(date +%H:%M:%S)"; }
gpu_query=timestamp,temperature.gpu,clocks.gr,clocks.mem,power.draw,utilization.gpu,memory.used,fan.speed
nvidia-smi --query-gpu=$gpu_query --format=csv,nounits -l 5 > "$samples" 2> /dev/null &
sampler=$!
trap 'kill $sampler 2> /dev/null' EXIT
{
	echo "turing-report $name, $(date -Is)"
	for step in $STEPS; do
		case $step in
		system)
			section system
			git log --oneline -1
			git status --short | head -20
			git -C vllm log --oneline -1 2> /dev/null
			ls -l venv/hyperinkling_sm75.so 2> /dev/null
			cat venv/cuda_home 2> /dev/null
			nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit,clocks.gr,clocks.mem,pcie.link.gen.current,pcie.link.width.current,pstate --format=csv
			nvidia-smi -q | grep -i -A12 "clocks\|throttle\|event reasons" | head -60
			nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
			free -h | head -3
			cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2> /dev/null
			;;
		test)
			section "kernel tests and timings"
			venv/bin/python -m pytest -q -s tests/test_sm75_attention.py 2>&1 | grep -v "Warning\|init_node\|^$"
			;;
		prefill)
			section "prefill profile, 8192 tokens"
			bench/profile_prefill.sh "report-$name-prefill" 8192 2>&1
			;;
		decode)
			for ctx in 8192 $CONTEXT; do
				section "decode profile, $ctx tokens"
				bench/profile_decode.sh "report-$name-decode-$ctx" "$ctx" 2>&1
			done
			;;
		bench)
			section "context bench"
			SPECS=dspark CONTEXTS="8192 $CONTEXT" CONC=1 MAX_SEQS=1 bench/context_bench.sh "report-$name" 2>&1
			;;
		*) echo "unknown step $step" ;;
		esac
	done
	kill $sampler 2> /dev/null
	section "GPU while it ran (5 s samples: $samples)"
	python3 - "$samples" <<-'PY'
	import csv, sys
	rows = list(csv.reader(open(sys.argv[1])))
	if len(rows) > 1:
	    head, rows = [h.strip() for h in rows[0]], rows[1:]
	    for i, h in enumerate(head[1:], 1):
	        vals = []
	        for r in rows:
	            try:
	                vals.append(float(r[i]))
	            except (ValueError, IndexError):
	                pass
	        if vals:
	            print(f"{h:28s} min {min(vals):8.0f}  mean {sum(vals) / len(vals):8.0f}  max {max(vals):8.0f}")
	PY
} 2>&1 | tee "$out"
echo "report: $out"
