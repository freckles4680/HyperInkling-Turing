# Notes

Facts, gotchas and the reasons behind the choices, so they are not re-derived.

## Machine

- This fork: a modded RTX 2080 Ti (sm75), 22 GB, native Linux, 32 GB of RAM, PCIe 3 x16.
  616 GB/s, fp16 tensor cores only (no bf16, no FP8), 64 KB of shared memory per block.
  Most measurements below this list (and in NOTES generally) come from upstream's machine
  and say so where they matter; Turing ones name the 2080 Ti.
- Upstream's machine: RTX 3090 Ti (sm86), 24 GiB, undervolted to 350 W; bf16 tensor cores unlocked: 102 TFLOPS
  measured on a cuBLAS GEMM (stock ~40), 904 GB/s copy bandwidth. Decode is
  bandwidth-bound; the unlock helps prefill.
- 16 cores / 32 threads: builds and servers are pinned to CPUs 0-30 (`taskset`).
- Fedora 44: gcc 16 is the default and nvcc 13.3 rejects it; everything that compiles
  CUDA (the vLLM build, FlashInfer's JIT at first use) needs `CC=gcc-15 CXX=g++-15
  NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-15"` (set in `start.sh` and the scripts).
- `/dev/shm` is 52 GB, enough for the 36 GiB KV CPU tier. RAM 62 GB.
- The GPU drives no display: all 24 GiB are free when nothing runs.

## Model (config.json, vLLM's inkling2 code)

- 24 layers, layers 6-17 run 3 passes with shared weights: 48 layer visits per token.
  Each pass has its own KV and conv-state caches; passes 2-3 blend the previous pass's
  messages through rank-64 controllers.
- 20 sliding-window layers (1024) + 4 global (5, 11, 17, 23): 8 global layer visits
  hold the long-context KV, 16 KiB/token in FP8.
- 128 experts, top 8, hidden 2304, expert intermediate 1152 (w13 interleaves gate/up
  rows). Released experts: NVFP4 layers 0-19, MXFP8 layers 20-23.
- Reasoning is always on to some degree (`reasoning_effort`, minimum "none" in the
  chat template still reasons a little); the model reasons in English even for
  Italian prompts. Tool calls: `--tool-call-parser inkling`; reasoning:
  `--reasoning-parser inkling`.
- Images: 40 px patches, 2x2 merge: tokens ~ (w/40)*(h/40). A4 at 147 ppi ~1.3k
  tokens, at 300 ppi 5.5k.

## Why these choices

- **GPTQ, not AWQ**, for the experts: AWQ's input-channel scaling cannot be folded on
  this architecture (w13's input is shared by all 128 experts and all 3 passes behind
  one norm; w2's would fold into the up rows through a tanh softcap). The format is
  still AWQ-layout INT4 g128 asymmetric for Marlin. No bf16 master exists: GPTQ runs on
  the dequantized NVFP4/MXFP8 weights, ~10-13% weight-space error, judged by output KL.
- **FP8 E4M3 KV, emulated, instead of int8 per-token-head**: the model is built for a
  unit-scale FP8 cache (K L2-clipped, V RMS-normalized), the authors' own default on
  DGX Spark; reuses their layout and numerics. The decode trick that made it cheap:
  E4M3 bits shifted into FP16 fields are exactly value * 2^-8 (subnormals too), so the
  attention MMAs run in fp16 and the 2^8 goes into the softmax scale (K) and the
  output (V). Two earlier decodes (via fp32, via integer bf16 bits) were 2-7x slower.
- **Where a decode step goes** (nsys, `bench/profile_spec.sh`, 6 requests, 22.4 ms): Marlin experts 12.0 ms (at bandwidth: ~60 of 128 experts per layer visit, measured
  on captured routing; same-domain requests share only ~7% more), `_w8_kernel` 3.6 ms
  (qkvr, o, the LM head: the small matrices stay at 300-500 GB/s whatever the tile),
  drafter W4 1.4 ms, attention 0.8 ms, the loop's small kernels ~3 ms. The GPU is never
  idle (wall within 1 ms of kernel time) except the 1.5 ms per step of synchronous
  scheduling at one request (async off at 262k).
- **Decode attention, full layers: at least two CTAs per SM.** One request (8 KV heads) had
  10 splits, one CTA per SM, ~570 GB/s on the KV; 21 splits: +3% decode at 32k, +10% at
  128k. 40 or 80 splits gain nothing more.
- **Warm-up requests at boot** (`scripts/warmup.py`, `WARMUP=1`): vLLM's warmup leaves out
  prefills of 65-256 tokens, over 256 not a multiple of 16, a full chunk, and the LM head at
  verify row counts (outside the graphs). Requests at those shapes after `/health`: 14 s on a
  cold Triton cache, 4 s warm; afterwards real traffic compiled one small kernel.
- **INT8 attention on a Triton GEMV, not Marlin**: Marlin W8A16 at one token reached
  ~45% of bandwidth on these small matrices, slower than the bf16 GEMV it replaced.
  The Triton kernel (`ops/linear_w8.py`) reaches 80-90%, and
  projects a full layer's q/k/v (three different inputs) in one launch.
- **Marlin for the experts**: the best MoE W4A16 kernel available here; at decode it reads
  ~900 GB/s.
- **The sconv conv-state cache holds only each step's tail** (`SconvStateManager`): the
  conv reads cached taps only for a step's first 3 tokens, so a step needs real blocks for
  the window, the drafts in flight and one prefix-hit alignment (64 tokens), not for its
  whole prefill chunk. Plus the window below the shared-prefix junction vLLM detects (where
  requests with the same system prompt diverge): without it they would hit only when a step
  happens to end there (`bench/prefix_hit.py`).
- **Host memory in the KV's address range**: `cuMemCreate` with a `HOST_NUMA` location maps
  beside device pages (VMM), and Triton kernels read it unchanged at ~25 GB/s. Decode reads a
  block once per step; prefill attention reads the KV once per query tile, so host pages must
  be staged on the GPU first (FOLLOWUPS, "Tried and dropped").
- **Embeddings via UVA** (HyperQwen's `qwen3_5-embed-uva` mechanism): only the looked-up
  rows cross PCIe; frees ~23k tokens of KV.

## DSpark (speculative decoding)

- **It is a block drafter: it always drafts 7 tokens** (`num_speculative_tokens` must be
  the checkpoint's block size - 1). How many the target *verifies* is set separately with
  vLLM's dynamic speculative decoding, `num_speculative_tokens_per_batch_size`
  (`start.sh` `SPEC_K`); the runner truncates the drafts.
- **A verify step is expensive on this MoE**: 1 + k tokens per request touch far more
  experts than one token (k=7 at one request: Marlin 6.5 ms vs 2.3 ms; the whole step
  15.2 ms vs 7.5). Long windows only pay on predictable text (struct, quoting); chat and
  prose want k=2-3. Hence the per-batch-size schedule.
- **Drafter cost, one request, before the HyperInkling changes: ~2.5 ms of a 7.5 ms
  step.** Base logits over the full bf16 head (0.93 GB, ~1 ms), seven Markov steps each
  reading the folded `(vocab, 256)` projection (7 x 103 MB, 0.85 ms), three dense bf16
  layers (~0.6 ms). Now: base logits through the shared INT8 head, the fold in INT4
  (`HYPERINKLING_DSPARK_FOLD=w4`), the layers in INT4 g128 RTN (`HYPERINKLING_DRAFT_W4`,
  `ops/linear_w4.py`). The bf16 head is no longer kept (0.86 GiB back).
- **Drafter quantization only moves acceptance, never output** (the target verifies every
  draft). Measured on counting (bench/copy_test.py): 6.7 tokens/step all-bf16, 6.6 with
  INT8 head + INT4 layers + INT4 fold.
- **Do not screen the Markov head on the base logits' top-k.** The base logits of later
  block positions are weak; the right token comes from the Markov bias and is often
  outside the base top-64 (counting fell from 6.6 to 2.9 tokens/step). And the head has
  pad rows past `vocab_size` (200,058 of 201,024) that the full path slices off.
- **Acceptance on free text is low by nature**, not a port bug: the original NVFP4
  checkpoint with everything bf16 gives the same 2.3 tokens/step at k=7 as ours, and
  counting reaches 6.7 of 8. Most served tokens are reasoning (English, free text).
- **Memory with DSpark**: +0.3 GiB of weights (INT4 drafter and fold; its Markov
  embedding goes to host RAM) and the draft's global layer in the KV (+2 KiB/token in
  FP8 on the target's 16).
- **Decode CUDA graphs must be given in requests**, `decode_cudagraph_capture_request_sizes`:
  the deprecated token sizes `[1..6]` cover no 8-token speculative decode batch, and vLLM
  then silently runs decode without graphs. A batch larger than every captured size runs
  eagerly too: `start.sh` derives the sizes from `MAX_SEQS` (every count to 8, then steps).
  At 12 requests that is 650 vs 381 tok/s against sizes 1-6, for ~1.5k tokens of KV.
- **A drafter-only INT4 copy of the LM head for the base logits does not pay**:
  393 vs 550 us at 7 rows, worse from 21 rows up (the W4 kernel reaches
  ~500 GB/s against INT8's ~900 on this 201k x 2304 shape), -1.5% step time at one
  request for 0.23 GiB (~12k tokens of context). Revisit only with a faster W4 GEMV.
- **Drafter KV in FP8 needed the drafter's `Attention` to get a cache config** (without
  one it is always bf16, whatever `kv_cache_dtype` the speculative config says) and a
  FlashAttention impl that only writes KV (FlashAttention refuses fp8 on SM8x; the draft
  attention runs in `block_decode_attention`, which reads the emulated E4M3 cache). Its
  pages then pack with the target's instead of being padded to them.
- **Longest request** (DSpark, vision tower on the GPU, 8 requests, FP8 KV): 268k,
  285k with async off (`docs/memory_matrix.tsv`). The global layers hold almost all of it; async scheduling costs
  ~17k tokens (two in-flight tails) for +3-4% decode.
- **Measure the context limit with `MAX_LEN=auto`.** vLLM checks a fixed limit before the
  warmup (without the transient activations and CUDA graphs, ~1.5 GiB here, so its estimate
  is optimistic) and again after it; `--max-model-len -1` fits the limit after warmup in one
  boot. That needs nothing sized by `max_model_len`: the drafter computes its RoPE angles
  instead of a table (a 1M table would cost ~25k tokens).
- **Vision tower in host RAM** (`HYPERINKLING_VISION_HOST=1`, each layer copied in before
  it runs): +27 ms per A4 page at 150 ppi, and as much context as no vision at all.
- **vLLM's adaptive verification does not run on Inkling**: it needs decode graphs with
  per-request query lengths, and `InklingSconvBackend` declares none. What would have to
  change (sconv builder, the decode combine kernel's row bound, ...) is mapped in a code
  investigation (docs/varlen-decode-investigation.md, FOLLOWUPS 7). `adaptive_k.py` chooses one length per step instead,
  on by default with candidates 2,4,7 (~5k tokens of decode graphs; 1,2,3,4,5,7 costs
  ~14k and no longer fits 262k with the vision tower on the GPU, for little more speed).
- basalt (jesdga95/basalt, MIT, Blackwell only) runs the same drafter: its draft policy
  picks each window by expected tokens per ms, from stated per-draft chances
  (`include/basalt/spec/draft_policy.hpp`); vLLM's `enable_adaptive_verification` is the
  same idea.

## Gotchas met

- **Padding rows broke the sconv tile skip** (upstream code): above 256 tokens `_sconv_kernel`
  skips the cache loads of a tile unless its last row's request starts within W-1 rows, and
  padding rows had `query_start = 0`. A padded batch whose real tokens end mid-tile lost
  the cache taps of the last request. Padding rows now start at `num_actual_tokens`
  (`inkling/nvidia/ops/sconv.py`; `tests/test_sconv_padding.py`).
- **`git am` drops a bracketed subject prefix**: `scripts/setup-vllm.sh` uses
  `--keep-non-patch`, or the commit loses `[hyperinkling]` and `build-image.sh` refuses it.

- **Images above the profiled size hang forever.** vLLM accepts an image only if it fits
  the encoder compute budget, max(prefill chunk, largest profiled image). Inkling
  profiled 2048 px squares (2704 tokens) and `--limit-mm-per-prompt` width/height can
  only shrink that, so an A4 page at 300 ppi (5488 tokens) passed validation and then
  waited with no error. Fixed in `processing.py` (the override can grow the profiled
  page up to 4096 px); `start.sh` sets `IMAGE_MAX_W`/`_H` and `--encoder-cache-size`.
- **`disable_any_whitespace` did nothing for Inkling**: its parser wraps JSON in a structural
  tag, where the xgrammar backend ignores the flag (vllm#68990; drop our part when it merges).
- **The KV CPU tier needs prefix caching on**; it worked first time with Inkling's
  hybrid groups.
- **Greedy output depends on the verify length.** Each `SPEC_K` runs different kernel
  shapes, so near-ties resolve differently: compare a speculative change against `SPEC=off`
  (or the same `SPEC_K`), never across k.
- **Run-to-run repeatability**: `moe_align_block_size` puts an expert's rows in atomic order;
  past `block_m` rows a row lands in another Marlin block and its result moves by 1 bf16 ulp,
  which flips experts down the sequence. `order_rows` (`ops/moe_marlin.py`,
  `HYPERINKLING_MOE_ORDER`, 0.44% of a step at 6 requests, nothing at 1) sorts them: the
  original against itself is KL 0 at every position (prefill, one request at a time, no
  speculation, no prefix cache). Generation with the served defaults is not repeatable, even
  one request at a time (8 greedy prompts twice: 2 with the same tokens): the adaptive verify
  length follows wall time and the drafts are sampled, so kernel shapes change. Not isolated. `bench/diverge.py` finds the first differing layer visit.
- **KL: compare medians.** A small numeric difference flips an expert and the rest of the
  sequence follows, so a few positions dominate the mean. Image-pad positions must be excluded
  (token id 200054; prompt_logprobs are keyed by token id).
- **Doubling calibration (332k tokens) changed nothing** measurable vs 166k.
- **`--uvicorn-log-level warning` hides "Application startup complete"**: wait on
  `GET /health` instead. Every wait gets a deadline.
- **`pkill -f '<pattern>'` kills the calling shell** when the pattern is in its own
  command line: use `pgrep -f '[v]env/bin/vllm serve'`.
- **Nsight `stop_profile` blocks for a minute or more** while the report is written; the
  server keeps serving. A killed API server under nsys can linger without the GPU:
  check `nvidia-smi --query-compute-apps` and the port owner before trusting a restart.
- **Client-side benchmarks with a shared prefix under-report prefill**: most of the prompt
  is served by the cache. Use `bench/metrics_window.py` for server-side numbers.
- **Mean TTFT under 6-way load is mostly queueing** when more than `MAX_SEQS` requests
  are offered.

## Data

- Calibration sources (one file each, `data/calib-src`): NousResearch/hermes-function-
  calling-v1 (Apache-2.0), pixparse/docvqa-single-page-questions (MIT),
  ise-uiuc/Magicoder-Evol-Instruct-110K (Apache-2.0), wikimedia/wikipedia 20231101.it
  (CC-BY-SA), HuggingFaceH4/ultrachat_200k (MIT).
- `bench/quality_battery.py` and `bench/extract_bench.py` run on public data
  (`bench/quality-data/`, downloaded by `quality_battery.py download`).
