#!/bin/bash
# General chatbot intelligence of the model as served, in one file to share:
# .local/intelligence-<name>.txt. Runs against a server that is ALREADY UP (scripts/serve.sh
# or ./start.sh); it does not boot or stop one.
#
#   bench/intelligence.sh <name>                            # standard preset, about 1.5-3 h
#   PRESET=quick bench/intelligence.sh <name>               # smoke test, about 15-30 min
#   STEPS="ifeval mtbench" bench/intelligence.sh <name>     # a subset
#   JUDGE_MODEL=gpt-4.1 JUDGE_API_KEY=sk-... bench/intelligence.sh <name>   # MT-Bench judged
#
# ifeval:   IFEval (google/IFEval): does the answer follow verifiable instructions ("no
#           commas", "at least 300 words", "answer in JSON"...). lm-evaluation-harness.
# mmlu_pro: MMLU-Pro (TIGER-Lab/MMLU-Pro): 10-choice questions in 14 subjects, 5-shot with
#           chain of thought. lm-evaluation-harness. LIMIT counts per subject.
# mtbench:  MT-Bench-style two-turn chat in 8 categories, scored 1-10 by a JUDGE model you
#           choose (bench/mtbench.py). Without JUDGE_MODEL the answers are saved but NOT
#           scored, and the report says so in capitals.
#
# lm-evaluation-harness goes into its own venv (.local/eval-venv, ~300 MB, no torch), so the
# server's venv/ is never touched. The machine needs internet for the first run: Hugging
# Face for the datasets, GitHub for the MT-Bench questions. Times above are guesses for a
# 2080 Ti at MAX_SEQS=2 (the model reasons before it answers); the report prints real ones.
#
# Scores are comparable between two runs only at the same PRESET/limits, JOBS-independent
# settings and (for MT-Bench) the same judge. With limits these are samples, not the
# official full-set numbers: expect a few points of noise.
set -uo pipefail
cd "$(dirname "$0")/.."
name=${1:?usage: bench/intelligence.sh <name>   (see the comments at the top)}

# --- settings (each can be set on the command line: VAR=value bench/intelligence.sh ...) ----
# The server. 8080 is serve.sh's API; do NOT use 8081 (the speed-display proxy adds a line
# to every answer, which would be scored as part of it).
# (HOST_ADDR, not HOST: login shells often set HOST to the machine name.)
HOST=${HOST_ADDR:-127.0.0.1}
PORT=${PORT:-8080}
MODEL=${MODEL:-inkling-2-mini}
# The server's API key, if serve.sh was started with API_KEY=... (empty = none).
API_KEY=${API_KEY:-}
# Which benchmarks, in this order.
STEPS=${STEPS:-ifeval mmlu_pro mtbench}
# Sample sizes. quick = a smoke test that everything works; standard = enough to compare two
# servers to within a few points; full = the whole sets (MMLU-Pro alone is 12k questions:
# days on one GPU). Any of the three limits below overrides the preset.
PRESET=${PRESET:-standard}
case $PRESET in
quick) d_ifeval=30 d_mmlu=2 d_mt=1 ;;      # 30 + 28 + 8 questions
standard) d_ifeval=150 d_mmlu=15 d_mt=10 ;; # 150 of 541, 210 of 12,032, all 80
full) d_ifeval=0 d_mmlu=0 d_mt=10 ;;        # 0 = no limit
*) echo "PRESET must be quick, standard or full" >&2; exit 1 ;;
esac
IFEVAL_LIMIT=${IFEVAL_LIMIT:-$d_ifeval}     # prompts (541 in all)
MMLU_PRO_LIMIT=${MMLU_PRO_LIMIT:-$d_mmlu}   # questions PER SUBJECT (14 subjects)
MTBENCH_LIMIT=${MTBENCH_LIMIT:-$d_mt}       # questions PER CATEGORY (8 categories x 10)
# Requests sent at once. Match the server's MAX_SEQS (serve.sh: 2); more only queue.
JOBS=${JOBS:-2}
# Token budget per answer, reasoning included. An answer that spends it all thinking comes
# back empty and scores 0, so keep it generous ("cut by max_tokens" in the report).
MAX_TOKENS=${MAX_TOKENS:-8192}
# Extra request fields for lm-eval, space separated key=value, e.g.
#   GEN_EXTRA='chat_template_kwargs={"enable_thinking":False}'   (Python literal syntax)
GEN_EXTRA=${GEN_EXTRA:-}
# The MT-Bench judge: any OpenAI-compatible endpoint (see bench/mtbench.py). Empty
# JUDGE_MODEL = answers only, judging SKIPPED. Pointing it at this same server works but is
# self-grading (biased upward, and slower): prefer a different, stronger model.
export JUDGE_MODEL=${JUDGE_MODEL:-}
export JUDGE_BASE_URL=${JUDGE_BASE_URL:-https://api.openai.com/v1}
export JUDGE_API_KEY=${JUDGE_API_KEY:-}
JUDGE_JOBS=${JUDGE_JOBS:-4}
# lm-evaluation-harness version (pinned so two runs score the same way).
LM_EVAL_VERSION=${LM_EVAL_VERSION:-0.4.13}
EVAL_VENV=${EVAL_VENV:-.local/eval-venv}

# --- setup -------------------------------------------------------------------------------------
mkdir -p .local
out=.local/intelligence-$name.txt
raw=.local/intelligence-$name # lm-eval logs, every sample and answer
mkdir -p "$raw"
base=http://$HOST:$PORT/v1
section() { printf '\n===== %s (%s) =====\n' "$1" "$(date +%H:%M:%S)"; }
auth=()
[ -n "$API_KEY" ] && auth=(-H "Authorization: Bearer $API_KEY")
lim() { [ "$1" -gt 0 ] 2> /dev/null && echo "--limit $1"; } # 0 = no --limit

ensure_lm_eval() {
	[ -x "$EVAL_VENV/bin/lm-eval" ] \
		&& "$EVAL_VENV/bin/python" -c "import lm_eval, langdetect, immutabledict, nltk" 2> /dev/null \
		&& return 0
	echo "installing lm-evaluation-harness $LM_EVAL_VERSION into $EVAL_VENV (once)"
	if command -v uv > /dev/null; then
		uv venv -q "$EVAL_VENV" && uv pip install -q --python "$EVAL_VENV/bin/python" \
			"lm_eval[api,ifeval]==$LM_EVAL_VERSION"
	else
		python3 -m venv "$EVAL_VENV" && "$EVAL_VENV/bin/pip" install -q \
			"lm_eval[api,ifeval]==$LM_EVAL_VERSION"
	fi || { echo "lm-eval install failed"; return 1; }
}

# run_lm_eval <task> <limit> [extra lm-eval args]: one task, its table and timing in the report.
run_lm_eval() {
	local task=$1 limit=$2; shift 2
	local dir=$raw/$task t0=$SECONDS
	ensure_lm_eval || return 1
	# temperature 0 and no stop strings: MMLU-Pro's "Question:" stop would also cut the
	# reasoning, which the server applies stop strings to.
	# shellcheck disable=SC2086
	OPENAI_API_KEY=${API_KEY:-none} HF_DATASETS_TRUST_REMOTE_CODE=1 "$EVAL_VENV/bin/lm-eval" run \
		--model local-chat-completions \
		--model_args "model=$MODEL,base_url=$base/chat/completions,num_concurrent=$JOBS,max_retries=3,timeout=3600,tokenized_requests=False" \
		--tasks "$task" $(lim "$limit") --apply_chat_template "$@" \
		--gen_kwargs temperature=0 "max_gen_toks=$MAX_TOKENS" "until=[]" $GEN_EXTRA \
		--output_path "$dir" --log_samples > "$dir.log" 2>&1
	local rc=$?
	echo "lm-eval exit $rc, $(((SECONDS - t0) / 60)) min; log: $dir.log, samples under $dir/"
	# lm-eval's own table (the lines that start with "|").
	grep '^|' "$dir.log" || { echo "no results table: last lines of the log:"; tail -25 "$dir.log"; }
	# Answers that came back empty: errors, or the token budget spent on reasoning.
	python3 - "$dir" <<-'PY'
	import glob, json, sys
	n = empty = 0
	for f in glob.glob(f"{sys.argv[1]}/**/samples_*.jsonl", recursive=True):
	    for line in open(f):
	        s = json.loads(line)
	        r = s.get("resps") or [[""]]
	        n += 1
	        empty += not (r[0][0] if r and r[0] else "").strip()
	if n:
	    print(f"{n} answers, {empty} empty (request errors or max_tokens spent on reasoning)")
	PY
	return $rc
}

{
	echo "intelligence $name, $(date -Is), preset $PRESET"
	section setup
	git log --oneline -1
	echo "server $base, model $MODEL, jobs $JOBS, max_tokens $MAX_TOKENS${GEN_EXTRA:+, extra $GEN_EXTRA}"
	echo "limits: ifeval $IFEVAL_LIMIT, mmlu_pro $MMLU_PRO_LIMIT per subject, mtbench $MTBENCH_LIMIT per category (0 = all)"
	echo "judge: ${JUDGE_MODEL:-NONE (MT-Bench judging will be SKIPPED)}${JUDGE_MODEL:+ at $JUDGE_BASE_URL}"
	[ "$PORT" = 8081 ] && echo "WARNING: port 8081 is the speed-display proxy: its extra line is scored. Use 8080."
	if ! models=$(curl -sf --max-time 10 "${auth[@]}" "$base/models"); then
		echo "the server does not answer at $base/models (not started, wrong PORT, or API_KEY needed)"
		exit 1
	fi
	echo "served models: $(echo "$models" | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))')"
	echo "$models" | grep -q "\"$MODEL\"" || { echo "model $MODEL is not served (set MODEL=)"; exit 1; }
	nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2> /dev/null
	for step in $STEPS; do
		case $step in
		ifeval)
			section "IFEval (instruction following; strict = exactly as asked, loose = minor slips forgiven)"
			run_lm_eval ifeval "$IFEVAL_LIMIT"
			;;
		mmlu_pro)
			section "MMLU-Pro (knowledge and reasoning, 10 choices: chance is 0.10)"
			run_lm_eval mmlu_pro "$MMLU_PRO_LIMIT" --fewshot_as_multiturn
			;;
		mtbench)
			section "MT-Bench-style judged chat (1-10 per turn)"
			t0=$SECONDS
			API_KEY=$API_KEY python3 bench/mtbench.py run --label "$name" --host "$HOST" --port "$PORT" \
				--model "$MODEL" --limit "$MTBENCH_LIMIT" --jobs "$JOBS" --judge-jobs "$JUDGE_JOBS" \
				--max-tokens "$MAX_TOKENS" > "$raw/mtbench.log" 2>&1
			echo "mtbench exit $?, $(((SECONDS - t0) / 60)) min; log: $raw/mtbench.log"
			# The summary (everything after the per-answer progress lines).
			sed -n '/^MT-Bench (/,$p' "$raw/mtbench.log"
			grep -m5 -i "error\|traceback" "$raw/mtbench.log" | grep -v "^MT-Bench (" | head -5
			echo "answers and judgments: bench/quality-data/mt_bench/{answers,judgments}-$name.jsonl"
			;;
		*) echo "unknown step $step" ;;
		esac
	done
	section "for comparison"
	echo "same card, bench/quality_battery.sh (2026-11-26): gsm8k 0.965, tools 0.605, docvqa 0.936"
	echo "published IFEval/MMLU-Pro/MT-Bench numbers use the full sets and other prompts:"
	echo "use this file to compare two runs of this script, not against leaderboards"
	section done
} 2>&1 | tee "$out"
echo "report: $out"
