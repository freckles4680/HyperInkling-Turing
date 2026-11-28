# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A port of Inkling 2 Mini (upstream vLLM runs it on Hopper/Blackwell; Ampere is PR #69165, open) to
one Turing GPU: a modded RTX 2080 Ti (sm75, 22 GB), the only card it is measured on. It is a
fork of Pento95/HyperInkling, the port to one RTX 3090 Ti (sm86, 24 GiB), whose code paths
and requantized checkpoint it keeps; most numbers in NOTES.md are from that 3090 Ti. Three parts: a pinned
vLLM with one patch, a model-preparation pipeline (`prepare/`), and a server with measured
defaults (`start.sh`). README.md has the numbers, NOTES.md the reasons behind every choice and
the gotchas already met, FOLLOWUPS.md what comes next, by phase. Read NOTES.md
before changing a design decision: most alternatives were already tried and measured.

## The vLLM checkout and the patch (most important workflow)

`vllm/` is not in git here. It is a clone of vllm-project/vllm at the commit in `patches/BASE`,
with `patches/vllm-hyperinkling-sm75.patch` applied as **one commit** on branch
`hyperinkling/sm75` (`scripts/setup-vllm.sh` recreates it). It is an editable install in
`venv/`, so Python/Triton edits under `vllm/` take effect on the next server start, no rebuild.

All the real model code lives in `vllm/vllm/models/inkling2/nvidia/` (`model.py`, `loop.py`,
`moe.py`, `attention.py`, `ops/*.py`). Changes made there exist only in `vllm/` until the patch
is regenerated. Keep it a single commit:

```bash
cd vllm && git add -A && git commit --amend --no-edit   # update the commit message if scope changed
git format-patch -1 --stdout > ../patches/vllm-hyperinkling-sm75.patch
```

The commit message is the authoritative list of what the port changes; keep it current.

Key mechanisms of the port (spread across several files):
- `patches/BASE` is the head of vLLM PR #69165 (Inkling 2 on SM80/86/89: PDL/TMA gating,
  software e4m3 `ops/e4m3.py`, Triton conv/mix fallback, FA2 vision). On top, sm8x takes
  the SM12x (consumer Blackwell) Triton path (`use_sm12x_path`).
- Routed experts: INT4 g128 asymmetric in AWQ layout (quant format `int4_awq`), GPTQ-calibrated,
  run by Marlin MoE (`ops/moe_marlin.py`).
- Attention, LM head and token embeddings are converted to INT8 **at load time**
  (`ops/linear_w8.py`, `model.py`), toggled by `HYPERINKLING_{ATTN,HEAD,EMBED}_W8` and
  `HYPERINKLING_EMBED_UVA` (embeddings in pinned host RAM).
- FP8 E4M3 KV cache emulated in software (`ops/e4m3.py`; decode `e4m3_f16_scaled`): bytes
  shifted into fp16 are value·2⁻⁸, so MMAs run in fp16 and the 2⁸ goes into the softmax
  scale / output.
- sconv conv-state cache: only each step's tail has real blocks (`SconvStateManager` in
  `inkling/nvidia/sconv_swa_attn.py`); the conv skips stores into the null block.
- Loop controllers and sconv taps on Triton batched GEMVs (`ops/bgemv.py`).
- Marlin rows in a fixed order per expert (`order_rows`, `HYPERINKLING_MOE_ORDER`): the same
  prefill alone gives the same numbers bit for bit (generation with DSpark does not repeat).
- `HYPERINKLING_CAPTURE_DIR` enables the MoE input/routing capture hook used by GPTQ.
- DSpark speculative decoding (on by default, `SPEC`): the block drafter always drafts 7;
  `SPEC_K` sets how many the target verifies per batch size (vLLM dynamic speculative
  decoding). Drafter layers and Markov fold INT4 (`ops/linear_w4.py`), base logits from the
  shared INT8 head, draft KV FP8 through a KV-write-only FlashAttention impl (`dspark.py`).
  Online verify length (`HYPERINKLING_SPEC_ADAPTIVE`, default 2,4,7):
  `vllm/v1/spec_decode/dynamic/adaptive_k.py`. Context lookup drafts (`HYPERINKLING_LOOKUP`,
  default on): `ops/lookup.py`.
- Images over the profiled page are scaled down in `processing.py` (`fit_to_budget`);
  `HYPERINKLING_VISION_HOST=1` streams the vision tower from host RAM per layer (`model.py`).
- Turing (sm75, a 22 GB 2080 Ti): fp16 activations (`ops.use_fp16`, `ops.act_dtype`;
  `HYPERINKLING_FP16=1` forces the path elsewhere). Kernels take their dtype from their
  pointers, so the bf16 path is the same code; DSpark too (weights cast at load). Tiles over Turing's
  64 KB of shared memory per block are smaller under `is_sm75()`; `bench/sm75_compile.py`
  checks every kernel offline (no GPU). `HYPERINKLING_FP16_CHECK=1` logs the residual
  stream's fp16 headroom. Bring-up on the card: `docs/turing-bringup.md`.
  Triton emits no tensor-core MMA for sm75 (`tl.dot` runs on FMA): decode-size linears go
  through the dot-free GEMV (`ops/linear_w8.py`, `gemv_sm75`), and attention (prefill and
  split decode) through a CUDA `wmma` kernel outside the vLLM build (`kernels/sm75/`,
  `scripts/build-sm75-kernels.sh` -> `venv/hyperinkling_sm75.so`, `ops/attention_sm75.py`;
  it also compiles for the CPU, `tests/test_sm75_attention.py`).
- The small loop kernels carry the reference numerics (fixed summation orders, no FMA
  contraction): any fusion must be checked bitwise against the unfused path.

## Commands

The machine needs `CC=gcc-15 CXX=g++-15 NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-15"` for
anything that compiles CUDA (nvcc 13.3 rejects Fedora 44's gcc 16); the scripts set it.
Long-running work is pinned to CPUs 0-30 with `taskset`.

```bash
scripts/setup-vllm.sh                 # recreate vllm/ from BASE + patch
scripts/build-vllm.sh                 # build into venv/ (sm75, ~30-60 min; ARCH=8.6 for a 3090)
scripts/build-sm75-kernels.sh         # Turing attention kernel -> venv/hyperinkling_sm75.so (~1 min)
./start.sh                            # production server, port 8080; every knob is an env var
scripts/serve.sh                      # Turing: start.sh on the LAN with the 2080 Ti settings, commented;
                                      # scripts/speed_proxy.py (:8081) adds tok/s under each chat answer
scripts/serve-switch.sh               # Turing: one address, loads the model each request names
                                      # (scripts/models.toml: Inkling, Clef); one log,
                                      # .local/model-switch.log; Clef: docs/clef.md, own venv
PORT=18030 ./start.sh                 # test instance on another port
sandbox/setup.sh                      # Claude Code in a GPU container (docs/claude-sandbox.md);
                                      # scripts/claude_loop.py: change, measure, keep or reset
PROFILE=<name> ./start.sh             # Nsight Systems; POST /start_profile, /stop_profile
scripts/serve-try.sh <log> [args]     # short-context eager boot of the original checkpoint, :18030

venv/bin/python -m pytest -q tests/                          # kernel tests (GPU; test_auth.py, test_wizard.py, test_model_switch.py, test_clef_server.py not)
venv/bin/python -m pytest -q -s tests/test_bgemv.py -k <name>  # one test, with timings
ruff check bench prepare tests scripts                        # lint: ruff 0.17.6 (as CI), rules in ruff.toml
```

CI (`.github/workflows/checks.yml`, no GPU): the patch applies to BASE, ruff, shell syntax,
the wizard. Release side: `scripts/wizard.py` (questions -> `.env`, from `docs/memory_matrix.tsv`),
`Dockerfile` + `docker-compose.yml` + `scripts/build-image.sh` (vLLM's own Dockerfile on
`./vllm`, sm75 by default, `ARCH=8.6` for a 3090; compose runs `hyperinkling:sm75`),
`bench/memory_matrix.sh` (longest request per knob combination: one boot each with
`MAX_LEN=auto`, which vLLM fits after warmup). Benches that boot `start.sh` pass `ENV_FILE=`
(measure the defaults, not the `.env`) and `WARMUP=0`.

Benchmarks and quality (all need a running server):
- `bench/context_bench.sh <name>`: the README speed table (8k/32k context, 1 and 6 requests,
  with and without DSpark; prefill on unseen prompts, decode on prefix hits).
- `bench/profile_prefill.sh <name> [tokens]`, `bench/profile_decode.sh <name> [tokens]`: kernel
  time of one prefill, and per decode step at a long context, under nsys.
  `scripts/turing-report.sh <name>` (Turing): GPU state, kernel tests, both profiles and the
  long-context bench in one file, `.local/turing-report-<name>.txt`.
- `bench/metrics_window.py [seconds] [port]`: server-side throughput/TTFT; use this, not
  client-side numbers (client TTFT hides prefix-cache hits).
- `scripts/quick_bench.py [concurrency] [max_tokens]`: quick client-side decode check.
- `bench/kl_eval.py gen|score|compare`, `bench/kl_sweep.sh`: teacher-forced KL against the
  original checkpoint (`data/calib/eval.jsonl`, 22 held-out transcripts), one request at a
  time so runs are comparable.
- `bench/spec_bench.py [port] [max_tokens] [temp|user]`: decode and draft acceptance per kind
  of text, at concurrency 1 and 6 (`user` = 0.2 for structured output, 0.5 otherwise).
  Sampling noise is about +-5%: decide on temperature 0 and step times.
- `bench/loop_eval.py cases.jsonl out.json`: share of answers in a repetition loop per
  sampling setting; cases are chat requests (tools are declared, never run).
- `bench/evalscope_perf.sh <name> [port]`: 10 minutes of AA-LCR prompts (256-64k tokens) with
  `metrics_window.py` alongside. `bench/profile_spec.sh`: kernel time per step under nsys.
- `bench/quality_battery.py download|run` (`bench/quality_battery.sh <name>` boots its own server): GSM8K, Hermes tool calls, DocVQA as strict JSON,
  on public data (`bench/quality-data/`, not in git); compare two servers at the same settings.
- `bench/intelligence.sh <name>`: general chatbot intelligence of a server that is already up (8080,
  not the 8081 proxy): IFEval and MMLU-Pro via lm-evaluation-harness (own venv `.local/eval-venv`),
  MT-Bench-style chat judged by `JUDGE_MODEL`/`JUDGE_BASE_URL`/`JUDGE_API_KEY` (`bench/mtbench.py`;
  no judge = flagged SKIPPED and exported to paste into a chat, `mtbench.py import` reads the
  scores back). `PRESET=quick|standard|full`; one report,
  `.local/intelligence-<name>.txt`.
- `bench/extract_bench.py`: document extraction on public data (shared Italian system
  prompt, 1-3 DocVQA pages, JSON or summary), closed loop with `metrics_window.py` alongside.
- `bench/greedy_ref.py save|compare`, `bench/copy_test.py`: output and acceptance checks.
- `bench/prefix_hit.py [port]`: prefix hits of requests sharing only a system prompt, and warm
  answers against cold ones.
- `bench/diverge.py`: first layer visit where two runs of a request differ (capture hook).
- `bench/w4_gemv.py`, `bench/w8_gemv.py`: W4/W8 GEMV tiles against a grid search (no server).
- `bench/long_decode.py`, `bench/offload_test.py`, `bench/moe_kernels.py`.

Model preparation (`prepare/`): `build_calib.py` → `replay_capture.py` → `gptq_experts.py`
(see README "Model preparation"); `quantize_experts.py` is the round-to-nearest baseline,
`int4_awq.py` the packing shared by both.

## Operational rules

Machine-specific rules (other engines on the GPU, who runs the server, ...) are in
CLAUDE.local.md, not in git.

- Servers started from a Claude session die with it: run long-lived ones in tmux.
- `start.sh` refuses to start if the port is taken. Two engines cannot share the 22 GB.
- `--uvicorn-log-level warning` hides "Application startup complete": wait on
  `GET /health` with a deadline. `GET /` is 404 by design.
- Scripts that boot a server use `scripts/server-lib.sh`: it refuses a busy GPU or port and
  stops only its own server.
- Never `pkill -f '<pattern>'` (it matches and kills the calling shell); use
  `pgrep -f '[v]env/bin/vllm serve'`.
- After killing a server (especially under nsys), check `nvidia-smi --query-compute-apps`
  and the port owner before trusting a restart.
- Not in git: `vllm/`, `venv/`, `models/`, `data/` (~70 GB of captures), `.local/` (logs,
  profiles). Pushing to GitHub or Hugging Face needs the owner's go-ahead.
