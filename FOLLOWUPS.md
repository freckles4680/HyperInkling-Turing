# Follow-ups

What comes next, by phase. Any item that trades something for something else ships as a
wizard option, not a default. Alternative tensors go in the same HF repo as separate files,
fetched with `hf download --include` only when chosen.

| phase | items |
|---|---|
| T, Turing (sm75) | T1, T2, T3, T4 (fp16 and shared memory in code) |
| 2, the checkpoint (new files on HF) | 33 |
| 3, context (best effort) | 18, 13, 55 |
| 4, speed and upstream | 10, 7, 21, 22, 32 |

Released: `ghcr.io/pento95/hyperinkling:0.1.4` (compose default) and two HF checkpoints, the
original (README = docs/MODEL_CARD.md) and the uncensored one (README =
docs/MODEL_CARD_UNCENSORED.md): re-upload a card when it changes. A new image goes with a new
tag (`scripts/build-image.sh` refuses a published one) and the compose default.

Where each idea comes from: **[HQ]** HyperQwen (github.com/syv-ai/HyperQwen, `patches/`, docs),
**[BS]** basalt (github.com/jesdga95/basalt, MIT), **[AR]** Ar4ikov's vLLM PRs
[#69165](https://github.com/vllm-project/vllm/pull/69165) (Inkling 2 on SM80/86/89) and
[#69141](https://github.com/vllm-project/vllm/pull/69141) (compressed-tensors experts),
**[AX]** github.com/agent2x0r/inkling2-sm86-kernels (2x3090 bring-up, `labs/`, `csrc/`),
**[C3]** club-3090 (github.com/noonghunna/club-3090, its `vllm-inkling` image recipe),
**[CD]** crazydiamond-on-ampere (github.com/Mr-Moonsilver/crazydiamond-on-ampere, KV in
host RAM), **[LC]** [llm-compressor](https://github.com/vllm-project/llm-compressor), **[IK]** the model's
own structure, **[NS]** ours (measurements in this repo, nsys profiles).

## Phase T: Turing (sm75, a 22 GB 2080 Ti)

In place: `is_sm75()` routes sm75 like sm8x, the quant config admits 7.5, the vision tower
runs on per-image SDPA (FlashAttention 2 is SM80+), `ARCH=7.5` for `scripts/build-vllm.sh`
and `scripts/build-image.sh`, and the wizard scales the context to the card's memory
(`--gpu-gib`, `gpu_gib` column of `docs/memory_matrix.tsv`).

**fp16 activations** (T1, T2 done in code): `ops.use_fp16()` (Turing, or
`HYPERINKLING_FP16=1` on any GPU) and `ops.act_dtype()`; buffers, norms, controllers, the
router, vision and Marlin's activations and INT4 scales follow it, the Triton kernels take
the dtype from their pointers (bitwise the same code on bf16), `_situ` unpacks fp16 pairs,
the vision rounding uses `cvt.rn.f16.f32`, and bf16 checkpoint tensors are cast at load,
clamped at 65504 (logged). `start.sh` passes `--dtype float16` on that path. DSpark runs in fp16
too: its bf16 weights cast at load, its kernels storing in the activation dtype (the Markov
gate keeps the reference's bf16 roundings, as bit operations). Unsloth's llama.cpp DSpark
GGUFs run the same drafter on RTX 20xx in F16/F32 math. `tests/test_fp16.py` checks each changed kernel in fp16 against fp32 (W8/W4/bgemv
also ran in Triton's CPU interpreter). Shared memory (64 KB per block on Turing against
Ampere's ~100 KB) is checked offline by `bench/sm75_compile.py`: the attention tiles
(72-96 KB) and the W8 prefill tile (128 KB) were over and are smaller on sm75 (attention
BLOCK_M <= 64 and BLOCK_N 64/32, W8 BN <= 128); every kernel now fits. Nothing has run on a
Turing GPU yet. The bring-up on the card, step by step, is `docs/turing-bringup.md`. Left:

T1. **fp16 on the card**: there is no bf16 GPU to compare against, so the check is fp16's
    own range and the published scores. `HYPERINKLING_FP16_CHECK=1` (eager) logs each new
    record of the residual stream's max |value| above 4096 and any non-finite value, by
    layer and pass; `bench/quality_battery.py` against README "Task accuracy". If the
    residual overflows, keep the residual blocks in fp32 in `Inkling2LoopModel.forward`.
T2. **Speed on the card**: Triton emits no tensor-core MMA for sm75, so every `tl.dot`
    runs as FMA. First profile (2080 Ti, DSpark, 1 request): 275 ms a step, 218 ms in
    `_w8_kernel` and 39 ms in the drafter's W4 kernels. W8 now has a dot-free GEMV (<= 64
    rows) and cuBLAS above; on Turing the drafter runs fp16 on cuBLAS
    (`HYPERINKLING_DRAFT_W4=0`, `HYPERINKLING_DSPARK_FOLD=bf16`). Third profile: 37 ms a
    step, 50.5 tok/s (1 request, 8k, DSpark), of which `_w8_gemv_kernel` 13.8 ms (138 us a
    call against ~10 us at the bandwidth bound), Marlin 8.3, `_k_group_kernel` 2.9, the
    controller GEMVs 2.8, `_rel_attention_kernel` 1.5. Fourth (3D-tile GEMV at untuned
    tiles, controllers on cuBLAS): 42.7 ms, worse. cuBLAS picks a 128x128 tile for the
    controllers' few rows (181 us a call); the 2.8 ms `_k_group_kernel` is the MoE router.
    `bench/sm75_w8_gemv.py` found tiles 2-4x faster (565 GB/s on the LM head), now
    `sm75_gemv_tile`'s; the controllers are back on Triton, and the router and the other
    decode-size K-group linears run on the same GEMV with fp16 weights. Fifth: 32.7 ms,
    58.4 tok/s; W8 GEMV 10.6 ms (149 calls with the router), Marlin 10.0, the controllers
    2.7, the fp16 Markov fold on cuBLAS 1.4 (7 x 202 us). Since: the controllers on the
    GEMV (batched, the down projection in 8 K slices), the fold in INT8 on Turing. Sixth:
    31.7 ms, 64.6 tok/s (GEMV 13.0 ms over 250 calls, Marlin 9.5). All of these are at a
    128-token context: decode at 90k (attention over the whole KV, `tl.dot` on FMA) is not
    measured yet (`CONTEXTS="8192 90000"` on `bench/context_bench.sh`). Measured then (one
    request, DSpark, prompts of code): 97.5k tokens fit; decode 85 tok/s at 8k and 35 at
    90k, prefill 692 and 159 tok/s. The 9 full-attention launches of a step (8 global layer
    visits and the draft's) read 1.6 GB of KV at 90k, ~3 ms at the card's bandwidth, and
    take ~43 ms on Triton's FMA dots. Since: `kernels/sm75/rel_attention.cu`, the split
    decode attention on `wmma` (one-warp blocks, 32 keys an iteration, 4 blocks per SM),
    checked on the CPU against the Triton kernel. On the card its tests pass; first timing
    2.2 ms a call at 90k (85 GB/s) against Triton's 3.5-5.2 ms, decode at 90k 41 tok/s: each
    lane read its own key, 32 scattered rows a load. Now the warp reads a key row at a time
    (one contiguous transaction), not yet timed. Prefill profile at 8k (668 tok/s, 1491 ms
    of kernels per 1000 tokens): `_rel_attention_kernel` 1097, Marlin 149, the router on
    `_k_group_kernel` 97, cuBLAS (W8 above 64 rows, dequantized) ~58, `_qr_rel_kernel` 24.
    Since: the wmma kernel also does prefill (MODE_OUT_LSE, four-warp blocks of 64 rows
    sharing the decoded keys, sliding-window layers) and the decode partials of local
    layers; K-group linears above 64 rows on cuBLAS. Not yet timed. Target (prefill:
    1000 tok/s): 70 tok/s at 1 request (~26 ms a step) with 90k+ context.
T3. **DSpark on the card** (in code): the draft's attention runs in `block_decode_attention`
    (Triton `rel_attention` on sm75, compiled in `bench/sm75_compile.py`). Its KV write is
    vLLM's `reshape_and_cache_flash` (core op, no FlashAttention kernel), and its RMSNorms
    run on Triton on sm75 (flashinfer's JIT-compile with the host's nvcc). Left: acceptance on the card
    (`bench/spec_bench.py`, `docs/turing-bringup.md` step 4).
T4. **A 22 GiB memory matrix**: `bench/memory_matrix.sh` on the card replaces the scaled
    estimate (it writes `gpu_gib`).

## Phase 2: the checkpoint

33. **llm-compressor / compressed-tensors** [LC, AR #69141]: llm-compressor calibrates GPTQ,
    AWQ and SmoothQuant schemes on the HF model and writes compressed-tensors checkpoints;
    #69141 runs such experts (W4A16, W4A8, W8A8, FP8) on Humming. Three uses here:
    (a) a second calibration of the experts, attention and head to compare with
    `prepare/gptq_experts.py` by KL median; (b) loading compressed-tensors W4A16 experts by
    repacking to the Marlin layout, so community quantizations run unchanged;
    (c) W8A8 INT8 for prefill: sm86's INT8 tensor cores run at twice the bf16 rate and
    prefill is compute-bound (vLLM has W8A8 int8 GEMMs for sm80; the MoE side needs a kernel).
    #69141 also fixes the DSpark draft quantization config and casts fp32 checkpoint tensors
    to bf16: check both against our drafter loader at the rebase (32).

## Phase 3: context (best effort: no DSpark, vision off the GPU)

18. **4-bit V in the global KV** [HQ KVarN]: KL median 0.0034 (0.0031 as served) with V in
    4 bits, groups of 32, asymmetric, K left in E4M3: ~+20% context. K is the sensitive half
    (K and V 4-bit g128 0.0047), a fixed codebook without scales fails on V's outlier dimensions
    (0.0042 at best), and V 4-bit on the windowed layers too costs 0.0046, so vLLM's equal pages
    per group need a planner change.

13. **Drafter W4 at 6-16 requests** [NS]: past ~21 rows the W4 kernels
    are tensor-core bound (int4 halves the bytes per MAC, so the bound comes at half the
    rows of INT8): 1.3 ms per step at 6 requests. fp16 accumulation doubles the MMA rate on
    GeForce; for the drafter it moves only acceptance (est. +2-3% at 6-16 requests).
    Building bf16 from bits (`code | 0x4300` = 128 + code, the zero point corrected per
    group from the row sums) gave +15-20% at 1-7 rows on split-K, worth ~0.03 ms a step.
55. **Sources in bf16** (other fine-tunes): `prepare/quantize_experts.py` and
    `gptq_experts.py` read only the release's NVFP4/MXFP8 experts (as the uncensored source
    is), and a bf16 Inkling 2 Mini does not fit 24 GiB to generate the corpus and capture.

## Phase 4: speed and polish

7. **Per-request verify length** [BS draft policy, vLLM adaptive verification]: needs decode
   graphs with per-request query lengths; docs/varlen-decode-investigation.md maps the
   changes (sconv builder varlen bound, the decode combine kernel's row bound from the
   device total, the CPU `max_query_len` as an upper bound, a varlen test). Today
   `adaptive_k.py` picks one length per step for the whole batch.
10. **Small-kernel fusion in the loop** [IK, AX `labs/lab1`, `csrc/convmix_sm86.cu`]: ~3 ms
    of a 22 ms step at 6 requests (48 layer visits x ~15 kernels of 2-5 us):
    `_attn_res_row_kernel`, `_sconv_kernel`, `_k_group_kernel` (router), `_attn_prologue`,
    `_blend_rms_norm`, routing. AX fuses short conv + residual + RMSNorm in one kernel
    (`cp.async` staging) and blend + RMSNorm with the row kept in registers. The reference
    numerics (fixed summation orders, no FMA contraction, exact `tanh`: AX measured
    `tanh.approx` flipping 0.23% of bf16 outputs) must hold: check bitwise against the
    unfused path.
    Measured at 6 requests: the controller chain (`pair_rms`, `down_split`,
    `up_from_partials`, `blend_rms_norm`) is 18 us per visit, but `blend`'s RMSNorm sums the
    whole row in one `tl.sum` and `up` must stay a `tl.dot`, so they cannot merge bitwise;
    `pair_rms` into `down_split` saves ~0.1 ms, `attn_res_row` (0.53 ms) is the other
    candidate: ~1-2% in all.
21. **Rust frontend** for the native build: vendored OpenSSL needs Perl `FindBin` (the
    container builds it).
22. **Upstream PRs**: the processing.py dummy-size fix and the oversized-image downscale are
    real upstream bugs; so is the sconv padding-row fix (NOTES).
32. **BASE to vLLM main when #69165 merges** [AR]: `patches/BASE` is its head today
    (`scripts/setup-vllm.sh` stops if a force-push removed it). #69165 hit a sampler-warmup
    OOM at `max_num_seqs` 256: keep the wizard's cap.

## To consider

56. **A server-side brake on reasoning loops** [NS]: vLLM has `repetition_detection` (per
    request) and `thinking_token_budget` (needs `--reasoning-config`). As wizard options, after
    checking them with DSpark and tool calls, if loops remain at temperature 1.0.
57. **Public cases for `bench/loop_eval.py`** [NS]: loops were measured on one private
    session, too few cases to publish rates; synthetic agent transcripts (8k-100k tokens) would
    make them repeatable.

## Tried and dropped

- Fewer bits, KL median (as served 0.0031, round to nearest with clip search): attention INT4
  0.0081 (g32 0.0064, only layers 0-5 and 18-23 0.0048, g32 there 0.0042), INT6 0.0036; LM head
  INT4 0.0042 (g64 0.0039) for ~13k tokens of context and a slower kernel; 3-bit for the 32 / 64
  / 128 least-routed experts 0.0040 / 0.0060 / 0.0138, 2-bit for 16 0.0040.
- Least-routed experts in host RAM [BS]: routing is flat (the 8 rarest of 128 take 1.7% of the
  routes, 32 take 11%, 64 29%): 8 experts free ~0.75 GiB and cost ~1 ms per decoded token.
- DSpark tuned for temperature 1.0: adaptive candidates 1,2,4,7 / 1,2,3,4,5,7 / 1,2,3, greedy
  drafts and no lookup are within noise of the defaults or worse; `SPEC=off` is 5% (chat) to
  2.1x (structured) slower.
- A bf16 drafter (a bound for drafter GPTQ): 1-9% slower at one request, +4% at 6 (noise).
- Padding of the drafter's sliding-window layers: ~15 MiB per live request (~850 tokens).
- Against club-3090's image [C3] (`single-fast`, release checkpoint, 100k limit, NBT 512, 2 requests), decode of one
  request at temperature 1.0, `bench/spec_bench.py`: chat 86 tok/s (here 144), code 144 (227),
  struct 195 (288), prose 90 (157), doc 178 (246); KV pool 113k tokens (268k).
- `NBT` above 2048: 3072 costs ~27k tokens of context and gives +1-2% prefill at 8k and 32k;
  4096 ~65k and +5-6%. With the KV overflow in host RAM (2 GiB) the context goes 370k -> 342k and a 299k request decodes
  at 35 tok/s instead of 52 (more of it in RAM) with no faster prefill. With 18 the cost per
  chunk is the same and its KL does not depend on the chunk (0.0034 at 2048 and 3072).
- KV overflow in host RAM [CD]: works, left out by choice (branch `hyperinkling/kv-host` of the
  local `vllm/`). Blocks past the GPU's in host memory mapped in the same address range, prefill
  through a GPU stage; lossless (KL scores and greedy text identical to the GPU-only ones).
  Longest request 318k with 1 GiB, 370k with 2, 421k with 3. Past the GPU's share decode pays
  ~0.7 ms per step per 1k tokens in RAM: at 1 GiB 130 tok/s at 266k and 63 at 299k, at 3 GiB 49
  at 299k and 23 at 401k (154 at 266k all on the GPU, 76 at 308k with `SPEC=off`).
- The whole KV in host RAM: dense global attention reads all of it at each step (25 GB/s over
  PCIe against 900); CD's engine can because its attention is sparse.
- Loop-core attention (layers 6-17) in bf16, as [hxyulin's recipe](https://huggingface.co/hxyulin/Inkling-2-Mini-Outer-Attention-FP8)
  does: all load-time conversions and the FP8 KV together cost 0.0001 of KL median (0.0030 ->
  0.0031), for ~17k tokens of context.
- Verify-side MoE kernel work [NS]: Marlin already reads ~900 GB/s; a second verified token
  costs less than double because tokens share experts. A warp-GEMV MoE for 1-4 tokens [AX]
  reaches ~390 GB/s per expert: below Marlin here.
- W8 attention/head kernels [NS, item 35]: up to 32 rows they read 75-90% of bandwidth
  (qkvr 765-800 GB/s, the head 870-930); past 32 rows they are bound by the dequant and
  MMAs (qkvr at 96 rows ~62 TFLOPS). Ordering the grid so row tiles of a column run
  together (weights from L2) changed nothing. Left: o (few column tiles), <1% of a step.
- SiTU fused into the experts' w13 epilogue [AX, item 34]: `_swiglu_kernel` is 0.12 ms of a
  22 ms step at 6 requests (0.5%), and each try rebuilds Marlin's C++.
- Screening the Markov head on the base logits' top-k [BS]: acceptance collapsed (counting
  6.6 -> 2.9 tokens/step); the right token comes from the Markov bias.
- A drafter-only INT4 copy of the LM head [HQ / BS 2-bit screen]: -1.5% step time for 0.23
  GiB; revisit with 13.
- vLLM's adaptive verification as shipped: the sconv backend refuses varlen decode graphs
  (7 is the way).
- HyperQwen's `kv-cache-seat-budget` (its `KVCacheGroup` API differs here),
  `terminal-request-progress` (does not apply), `tokenize-v1-route` (not needed).
- The release's MXFP8 experts on layers 20-23 [BS]: KL median
  0.0027 vs 0.0030 (noise floor 0.0003), top-1 92.5% vs 92.35%, for +1.94 GiB (~110k tokens
  of context); layer 23 alone changes nothing.
- A draft vocabulary subset [HQ `qwen3_5-mtp-draft-vocab`]: base logits and Markov fold on
  the 32k or 48k most frequent head rows. Steps 5% shorter, acceptance lower (extraction
  bench 42.5% -> 34-35%, struct -5 to -12%); document text needs the full vocabulary.
- Context lookup at the default verify length: +2% while copying (the drafter copies tables
  well on its own); it pays with the adaptive length (on by default, `ops/lookup.py`).
