#!/bin/bash
# INT4 attention / LM head (HYPERINKLING_ATTN_W4, HYPERINKLING_HEAD_W4) against the INT8
# defaults, in one file to share: .local/attn-w4-<name>.txt. The GPU must be free.
#
#   bench/attn_w4.sh <name>                          # every step, ~1-1.5 hours
#   STEPS="system test kernels" bench/attn_w4.sh k1  # no server, ~5 minutes
#   VARIANTS="w8 head4 attn4g32" STEPS="speed kl" bench/attn_w4.sh v2
#
# system:  the commit, the GPU, other GPU users
# test:    tests/test_linear_w4_gemv.py (stops the run if it fails)
# kernels: bench/sm75_w4_gemv.py, INT4 against INT8 per call and per decode step
# speed:   bench/context_bench.sh per variant: 8k and CONTEXT, one request, DSpark at a
#          fixed verify length (HYPERINKLING_SPEC_ADAPTIVE=, so ms/step compares), the
#          context that fits
# kl:      teacher-forced KL of each variant against the first one (w8, the served
#          defaults) on bench/kl/corpus.json, SPEC=off, one request (bench/kl_eval.py)
# Variants (VARIANTS, names from the table below; the first is the reference):
VARIANTS=${VARIANTS:-w8 head4 attn4 all4}
declare -A ENVS=(
	[w8]=""
	[head4]="HYPERINKLING_HEAD_W4=1"
	[head4g32]="HYPERINKLING_HEAD_W4=1 HYPERINKLING_HEAD_W4_GROUP=32"
	[attn4]="HYPERINKLING_ATTN_W4=1"
	[attn4g32]="HYPERINKLING_ATTN_W4=1 HYPERINKLING_ATTN_W4_GROUP=32"
	[attn4g128]="HYPERINKLING_ATTN_W4=1 HYPERINKLING_ATTN_W4_GROUP=128"
	[edge4]="HYPERINKLING_ATTN_W4=0-5,18-23"
	[core4]="HYPERINKLING_ATTN_W4=6-17"
	[all4]="HYPERINKLING_ATTN_W4=1 HYPERINKLING_HEAD_W4=1"
	[all4g32]="HYPERINKLING_ATTN_W4=1 HYPERINKLING_ATTN_W4_GROUP=32 HYPERINKLING_HEAD_W4=1"
)
# Groups default to 64 (HYPERINKLING_{ATTN,HEAD}_W4_GROUP); a variant may also be given
# inline as name=VAR=value+VAR=value (e.g. "mine=HYPERINKLING_ATTN_W4=0-5,18-23+HYPERINKLING_ATTN_W4_GROUP=32").
set -uo pipefail
cd "$(dirname "$0")/.."
name=${1:?usage: bench/attn_w4.sh <name>}
STEPS=${STEPS:-system test kernels speed kl}
CONTEXT=${CONTEXT:-90000}
PORT=${PORT:-18030}
mkdir -p .local/attn-w4
out=.local/attn-w4-$name.txt
section() { printf '\n===== %s (%s) =====\n' "$1" "$(date +%H:%M:%S)"; }

variant_env() {  # the VAR=value words of a variant
	local v=$1
	if [[ $v == *=* ]]; then
		echo "${v#*=}" | tr '+' ' '
	elif [[ -v ENVS[$v] ]]; then
		echo "${ENVS[$v]}"
	else
		echo "unknown variant $v" >&2; return 1
	fi
}
variant_name() { echo "${1%%=*}"; }

{
	echo "attn-w4 $name, $(date -Is), variants: $VARIANTS"
	for v in $VARIANTS; do echo "  $(variant_name "$v"): $(variant_env "$v" || true)"; done
	for step in $STEPS; do
		case $step in
		system)
			section system
			git log --oneline -1
			git status --short | head -20
			git -C vllm log --oneline -1 2> /dev/null
			git -C vllm status --short | head -10
			nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit,clocks.gr,clocks.mem,pcie.link.gen.current,pcie.link.width.current --format=csv
			nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
			ls -l venv/hyperinkling_sm75.so 2> /dev/null
			;;
		test)
			section "kernel tests"
			venv/bin/python -m pytest -q -s tests/test_linear_w4_gemv.py > .local/attn-w4/test-$name.log 2>&1
			rc=$?
			grep -v "Warning\|^$" .local/attn-w4/test-$name.log | tail -40
			[ $rc = 0 ] || { echo "kernel tests FAILED: stopping before the server runs"; break; }
			;;
		kernels)
			section "INT4 vs INT8 GEMV per call"
			venv/bin/python bench/sm75_w4_gemv.py 2>&1 | grep -v "Warning"
			;;
		speed)
			for v in $VARIANTS; do
				vn=$(variant_name "$v")
				section "speed: $vn ($(variant_env "$v"))"
				# shellcheck disable=SC2046
				env $(variant_env "$v") HYPERINKLING_SPEC_ADAPTIVE= SPECS=dspark \
					CONTEXTS="8192 $CONTEXT" CONC=1 MAX_SEQS=1 PORT=$PORT \
					bench/context_bench.sh "attnw4-$name-$vn" 2>&1 | grep -v "^$"
				grep -h "HyperInkling\|INT8:\|Error\|error" ".local/context-bench-attnw4-$name-$vn-dspark.log" 2> /dev/null \
					| grep -iv "warn" | sort -u | head -8
			done
			section "speed: all variants"
			# shellcheck disable=SC2046
			venv/bin/python bench/context_bench.py table $(for v in $VARIANTS; do echo "attnw4-$name-$(variant_name "$v")-dspark"; done)
			;;
		kl)
			source scripts/server-lib.sh
			ref=
			for v in $VARIANTS; do
				vn=$(variant_name "$v")
				section "kl: $vn ($(variant_env "$v"))"
				log=.local/attn-w4/kl-$name-$vn.log
				# shellcheck disable=SC2046
				start_server "$PORT" "$log" $(variant_env "$v") SPEC=off MAX_SEQS=1 MAX_LEN=16384 \
					PORT=$PORT WARMUP=0 ENV_FILE= ./start.sh
				if wait_health "$PORT"; then
					grep -h "INT8:\|INT4:\|GPU KV cache size" "$log" | sort -u | head -5
					PORT=$PORT venv/bin/python bench/kl_eval.py score bench/kl/corpus.json \
						".local/attn-w4/kl-$name-$vn.json"
				else
					echo "server failed: $log"; tail -20 "$log"
				fi
				stop_server
				if [ -z "$ref" ]; then
					ref=.local/attn-w4/kl-$name-$vn.json
				elif [ -f "$ref" ] && [ -f ".local/attn-w4/kl-$name-$vn.json" ]; then
					echo -n "KL $vn vs $(basename "$ref" .json): "
					venv/bin/python bench/kl_eval.py compare "$ref" ".local/attn-w4/kl-$name-$vn.json"
				fi
			done
			;;
		*) echo "unknown step $step" ;;
		esac
	done
} 2>&1 | tee "$out"
echo "report: $out"
