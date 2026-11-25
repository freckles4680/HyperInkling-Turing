# HyperInkling

**[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines
Lab, Apache-2.0) on one RTX 3090 / 3090 Ti, with vLLM.** Text and images (document pages),
tool calls, strict JSON output, ~268k tokens of context per request, several requests at
once, and the model's own DSpark drafter for speculative decoding.

Upstream vLLM runs Inkling 2 Mini on Hopper and Blackwell. This repository is the port to
one Ampere card (sm86) plus a requantized checkpoint that fits in 24 GB: a pinned vLLM with
one patch, a model-preparation pipeline, and a launcher with measured defaults.

It does not aim to beat other inference engines on one request at a time. It is a solid base
for serving the new Inkling 2 Mini to **several requests in parallel** on one 3090: vLLM's
batching, prefix cache and CPU tier, a long context next to the vision tower, and DSpark
that still pays at 6 requests.

> **Scope, honestly.** This is a personal project, written almost entirely with an AI coding
> assistant ("vibe coded"), published because it may be useful to other 3090 owners. It
> targets **one RTX 3090 or 3090 Ti and nothing else**: no other GPU generation, no
> multi-GPU, no promise of support. Issues with a reproduction are welcome; feature requests
> for other hardware will be closed.

## Quick start (Docker)

```bash
git clone https://github.com/Pento95/HyperInkling hyperinkling && cd hyperinkling
python3 scripts/wizard.py         # a few questions -> .env (checkpoint, vision, DSpark, ...)
docker compose up -d              # first start pulls the image (~26 GB) and the checkpoint (~16 GB)
docker compose logs -f hyperinkling
```

The first question picks the checkpoint: Inkling 2 Mini as released
([Pentium95/Inkling-2-Mini-W4A16-g128-gptq](https://huggingface.co/Pentium95/Inkling-2-Mini-W4A16-g128-gptq)),
or [OpenUncensored's abliterated version](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored)
quantized the same way
([Pentium95/Inkling-2-Mini-Uncensored-W4A16-g128-gptq](https://huggingface.co/Pentium95/Inkling-2-Mini-Uncensored-W4A16-g128-gptq),
KL median 0.0027 against its source as served). Compose downloads it at the first start.

The wizard's answers are in `.env`, which compose reads at every start: next time, just
`docker compose up -d`. To keep several setups, save copies and pick one before starting:
`mkdir -p profiles && cp .env profiles/long-context.env`, later
`cp profiles/long-context.env .env && docker compose up -d`.

To build the image yourself instead (~1 hour, compiles vLLM for sm86):
`scripts/setup-vllm.sh && scripts/build-image.sh`, then `IMAGE=hyperinkling:latest` in `.env`.

The OpenAI-compatible API is at `http://127.0.0.1:8080/v1`, model `inkling-2-mini` (alias `local`, see `SERVED_ALIASES`)
(`GET /health` when it is up; `GET /` is a 404 by design).

**Sampling:** temperature 1.0 is the model's default. Its card reports that it can get stuck in
long reasoning loops: cap `max_tokens` and the agent's turns.

The port is published on 127.0.0.1 only: to serve other machines set `BIND=0.0.0.0` and a
`VLLM_API_KEY` in `.env` (the wizard does both). With a key, every path except `/health`, `/ping`, `/load` and
`/version` needs it, `/metrics` and `/tokenize` included.

Requirements: an RTX 3090 or 3090 Ti with nothing else on it (a display on the same GPU costs
about 1 GiB: tell the wizard), an NVIDIA driver recent enough for CUDA 13 (580 or newer),
Docker with the NVIDIA Container Toolkit, ~20 GB of disk for the checkpoint. 32 GB of RAM
works; 64 GB lets the prefix-cache CPU tier hold long shared prompts.

### Without Docker

```bash
scripts/setup-vllm.sh            # vllm/ = vLLM PR #69165 at patches/BASE + our patch
scripts/build-vllm.sh            # venv/, sm86 only (~30-60 min)
hf download Pentium95/Inkling-2-Mini-W4A16-g128-gptq --local-dir models/Inkling-2-Mini-W4A16-g128-gptq
python3 scripts/wizard.py         # the same questions, writes .env
./start.sh                       # foreground; reads .env, every knob is also an env var
```

Tested on Fedora 44 (gcc 15 next to the default gcc 16, CUDA 13.3). Other distributions
should work where CUDA 13 and a gcc it accepts are installed; that is best effort.

### Next to an existing CUDA setup (llama.cpp and others)

HyperInkling can be installed without touching an existing CUDA toolkit, Python or
llama.cpp build. The one thing it shares with them is the NVIDIA driver, which must be 580
or newer. A newer driver still runs programs built for older CUDA versions, so llama.cpp
builds against CUDA 12 keep working after the update. Both engines cannot hold the GPU at
the same time: stop one before starting the other.

**In a container (the most isolated).** CUDA 13, Python and vLLM all live in the image. The
host needs only the driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
On Windows, use Docker Desktop with the WSL2 backend and the Windows NVIDIA driver; don't
install a Linux driver inside WSL. The published image is built for sm86. On a Turing card
(RTX 20xx, 2080 Ti), build an sm75 image once, which takes about an hour:

```bash
scripts/setup-vllm.sh
ARCH=7.5 scripts/build-image.sh hyperinkling:sm75   # vLLM's Dockerfile, CUDA 13 inside
python3 scripts/wizard.py                            # on Turing it writes IMAGE=hyperinkling:sm75
docker compose up -d
```

To remove it, run `docker compose down`, then `docker image rm hyperinkling:sm75 hyperinkling-vllm:sm75`.

**In a uv environment (native).** `scripts/build-vllm.sh` builds everything into `./venv`
with [uv](https://docs.astral.sh/uv/). uv fetches its own Python 3.14, so the system Python is
not used. The torch wheels bring their own CUDA runtime, and `start.sh` activates the venv
only in its own process. Building vLLM still needs a CUDA 13 `nvcc`. If the system toolkit
is another version, install CUDA 13 beside it and point the build at it:

```bash
# NVIDIA's runfile, toolkit only, in its own directory (no driver, system toolkit untouched)
sh cuda_13.2.1_*_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-13.2
CUDA_HOME=$HOME/cuda-13.2 ARCH=7.5 scripts/build-vllm.sh   # ARCH=7.5 on Turing, default 8.6
./start.sh
```

Afterwards, check that `/usr/local/cuda` still points where it did. The build refuses an
`nvcc` that is not CUDA 13, and records the toolkit it used in `venv/cuda_home`. `start.sh`
puts that toolkit first on `PATH` for the server, because FlashInfer compiles kernels with
`nvcc` at the first boot. A system CUDA 12 `nvcc` fails there with
`Unknown option '--compress-mode=size'`. For a venv built before this was recorded, run
`echo $HOME/cuda-13.2 > venv/cuda_home` and then `rm -rf ~/.cache/flashinfer`. To remove
everything, run `rm -rf venv vllm $HOME/cuda-13.2 ~/.cache/flashinfer`.

## Speed

RTX 3090 Ti undervolted to 350 W with [gpu-unlock](https://github.com/zagitta/gpu-unlock)
active; the defaults (KV cache FP8 E4M3, prefill chunk 2048, async scheduling). Prompts of
exactly 8k or 32k tokens (AA-LCR documents, a detailed summary asked), 512 tokens out, greedy.
Prefill on prompts never seen; decode on the same prompts again as prefix-cache hits,
counted only while every request generates, so no prefill is inside it. "Aggregate" is the
sum over the requests running at once (`bench/context_bench.sh`).

| | 1 request, no DSpark | 1 request, DSpark | 6 requests, no DSpark | 6 requests, DSpark |
|---|---|---|---|---|
| prefill, per request, 8k context | 6,283 tok/s | 6,191 tok/s | 2,292 tok/s | 2,417 tok/s |
| prefill, aggregate, 8k context | 6,283 tok/s | 6,191 tok/s | 6,243 tok/s | 6,191 tok/s |
| decode, per request, 8k context | 131 tok/s | 267 tok/s | 68 tok/s | 114 tok/s |
| decode, aggregate, 8k context | 131 tok/s | 267 tok/s | **407 tok/s** | **686 tok/s** |
| prefill, per request, 32k context | 5,255 tok/s | 5,209 tok/s | 2,123 tok/s | 2,114 tok/s |
| prefill, aggregate, 32k context | 5,255 tok/s | 5,209 tok/s | 5,267 tok/s | 5,223 tok/s |
| decode, per request, 32k context | 124 tok/s | 188 tok/s | 58 tok/s | 108 tok/s |
| decode, aggregate, 32k context | 124 tok/s | 188 tok/s | **350 tok/s** | **646 tok/s** |

One prompt already fills the GPU (prefill is compute-bound), so 6 at once share the same
~6,200 tok/s. gpu-unlock raises the bf16 tensor-core rate (102 TFLOPS on a cuBLAS GEMM instead
of ~40): it changes prefill, not decode, which is bandwidth-bound. DSpark is lossless (the
target verifies every draft); how much it gains depends on the text: most on code, structured
output and answers that quote the prompt, least on free prose like these summaries.

A field report on a **3090 (not Ti)**, power limit 320 W, gpu-unlock active, 0.1.2, same bench:

| | 1 request, no DSpark | 1 request, DSpark | 6 requests, no DSpark | 6 requests, DSpark |
|---|---|---|---|---|
| prefill, per request, 8k context | 5,745 tok/s | 5,770 tok/s | 2,124 tok/s | 2,093 tok/s |
| prefill, aggregate, 8k context | 5,745 tok/s | 5,770 tok/s | 5,803 tok/s | 5,800 tok/s |
| decode, per request, 8k context | 123 tok/s | 268 tok/s | 63 tok/s | 102 tok/s |
| decode, aggregate, 8k context | 123 tok/s | 268 tok/s | 381 tok/s | 611 tok/s |
| prefill, per request, 32k context | 4,805 tok/s | 4,876 tok/s | 1,979 tok/s | 1,973 tok/s |
| prefill, aggregate, 32k context | 4,805 tok/s | 4,876 tok/s | 4,888 tok/s | 4,877 tok/s |
| decode, per request, 32k context | 116 tok/s | 180 tok/s | 54 tok/s | 98 tok/s |
| decode, aggregate, 32k context | 116 tok/s | 180 tok/s | 326 tok/s | 590 tok/s |

## Context and the knobs that trade it

Longest single request, 8 requests at once, KV at 8 bits, the defaults otherwise
(`docs/memory_matrix.tsv`, `bench/memory_matrix.sh`). The server fits it at boot
(`MAX_LEN=auto`); the wizard shows it for every choice.

| vision tower | DSpark (default) | no DSpark |
|---|---|---|
| on the GPU (default) | **268k** | 345k |
| in host RAM | 303k | 385k |
| none (text only) | 305k | 394k |

With DSpark and the vision tower on the GPU: 289k at 1 request, 274k at 6, 256k at 16, 235k at 32,
197k at 64. Async scheduling off gives ~17k more; a prefill chunk of 1024 ~24k more. Without
DSpark, without the vision tower, with async scheduling off and a 1024 prefill chunk: **432k**
at 1 request (427k at 6), as far as the 24 GiB allow, not to the model's 1M.

| knob | default | what it does | context / VRAM | speed |
|---|---|---|---|---|
| `SPEC` | `dspark` | speculative decoding with the bundled drafter | ~78k tokens less (268k vs 345k) | 1.5-2x decode at one request, 1.7-1.8x aggregate at 6 |
| `MAX_LEN` | `auto` | longest request: `auto` is what the KV cache holds after warmup; a number caps it (and the boot fails if it does not fit) | none | none |
| `ASYNC` | 1 | async scheduling (two steps in flight) | off: ~17k tokens more | off: -3 to -4% decode |
| `NBT` | 2048 | prefill chunk (and image encodings per step) | 1024: ~24k tokens more | 1024: -8% prefill, same decode |
| `LONG_PREFILL` | `NBT`/2 | most tokens of one prompt per step while other requests are there (no limit when it is alone); 0 = off | none | short prompts beside a 120k prefill: TTFT 26.6 -> 1.9 s; that prefill +19% |
| `IMAGES` | 50 | images per request; 0 = no vision tower | 0: ~37k tokens more | no images |
| `HYPERINKLING_VISION_HOST` | 0 | vision tower weights in host RAM, copied to the GPU layer by layer per image | 1: ~36k tokens more | +27 ms per A4 page |
| `IMAGE_MAX_W` / `_H` | 1040 x 1560 | largest image: 1,014 tokens, an A4 page at ~126 ppi (two per 2048 chunk); larger ones are scaled down to it | A4 at 300 ppi (2480 x 3508, with `ENCODER_CACHE=16384`): ~5.5x the tokens per page | 300 ppi: ~5.5x the prefill per page |
| `MAX_SEQS` | 8 | requests at once | 289k at 1, 274k at 6, 268k at 8, 256k at 16, 197k at 64 | more aggregate decode |
| `KV_OFFLOAD_GB` | `auto` (60% of RAM - 8 GiB, at most 36) | prefix-cache tier in host RAM (`/dev/shm`) | host RAM only | evicted prefixes come back in 0.1 s instead of seconds |
| `HYPERINKLING_SPEC_ADAPTIVE` | `2,4,7` | verify length chosen per step from these; empty = the fixed `SPEC_K` schedule | ~5k tokens | with the lookup, against `SPEC_K`: +43% copying, +4-10% on free text and code, within noise at 6 requests |
| `HYPERINKLING_LOOKUP` | 1 | drafts copied from the request's own context instead of DSpark's on a match of 8 tokens or more (lossless) | 6 MB | +10% while copying |
| `SPEC_K` | `[[1,2,4],[3,6,3]]` | drafts verified per step, by batch size, when the adaptive length is off | none | |
| `SPEC_DRAFT` | `probabilistic` | drafts sampled from the draft distribution | none | +15-20% on structured output at temperature > 0 |
| `HYPERINKLING_MOE_ORDER` | 1 | each expert's rows in a fixed order: the same prefill gives the same numbers bit for bit (generation with DSpark still varies between runs) | none | 0: +0.4% decode at 6 requests |
| `WARMUP` | 1 | requests at the shapes vLLM's warmup leaves out, right after `/health` | none | no compile stalls on the first requests (4 s at boot, 14 s after an update) |
| `HOST` / `VLLM_API_KEY` | 127.0.0.1 / none | who can connect | none | none |

## What runs where

| part | format | kernel |
|---|---|---|
| routed experts | INT4 g128 asymmetric (AWQ layout), GPTQ-calibrated | Marlin MoE |
| attention projections | INT8 g128, quantized at load | Triton W8A16 GEMV/GEMM |
| LM head | INT8 g128, fp32 logits (shared with the drafter) | same |
| token embeddings | INT8 g128, in pinned host RAM (UVA) | lookup |
| KV cache | unit-scale FP8 E4M3, emulated (sm86 has no FP8) | Triton attention, fp16 MMAs |
| loop controllers, sconv taps | bf16 | Triton batched GEMV |
| short-conv state | only each step's last tokens hold real blocks | Triton |
| DSpark drafter | layers INT4 g128; Markov fold INT4; KV FP8 | Triton W4A16 split-K |
| vision tower | bf16, on the GPU (or streamed from host RAM) | FlashAttention-2 + cuBLAS |
| prefix cache | GPU + CPU tier | vLLM OffloadingConnector |

The patch's commit message (`patches/vllm-hyperinkling-sm86.patch`) lists every change.
NOTES.md has the reasons, the measurements and the gotchas; FOLLOWUPS.md what comes next.

## Quality

Teacher-forced KL against the original NVFP4/MXFP8 checkpoint on 22 held-out transcripts
(images, tool calls, code, Italian, chat; `bench/kl_eval.py`, one request at a time). One
request at a time, without speculation, scoring a prompt repeats bit for bit
(`HYPERINKLING_MOE_ORDER`); the mean
stays dominated by the few positions where a small difference flips an expert, so compare
medians.

| | KL median | KL mean | top-1 agreement |
|---|---|---|---|
| the original against itself | 0 | 0 | 100% |
| experts INT4, round to nearest (older build) | 0.0039 | 0.133 | 91.2% |
| experts INT4 GPTQ | 0.0030 | 0.117 | 92.5% |
| **as served**: + INT8 attention, head, embeddings, FP8 KV | **0.0031** | 0.123 | 92.1% |

Speculative decoding does not
change the output distribution.

Task accuracy, the same server settings for both (`bench/quality_battery.py`, greedy, 200
items each; about +-1 to +-3 points of noise):

| | original checkpoint | HyperInkling as served |
|---|---|---|
| GSM8K | 0.970 | 0.970 |
| tool calls, exact (names only) | 0.645 (0.735) | 0.615 (0.710) |
| DocVQA as strict JSON, ANLS (valid JSON) | 0.938 (99.5%) | 0.935 (99.5%) |

## Quantizing a model for HyperInkling

The published checkpoint is the output of this pipeline. It turns a checkpoint with Inkling 2
Mini's architecture into one HyperInkling serves: routed experts INT4, group 128, asymmetric,
GPTQ-calibrated on the model's own text (AWQ layout, for Marlin); every other tensor and the
DSpark drafter unchanged (attention, LM head and embeddings become INT8 at load time).

The source must be in the release's format (experts NVFP4 on layers 0-19, MXFP8 on 20-23,
`hf_quant_config.json` from modelopt), like `thinkingmachines/Inkling-2-Mini`; other formats
(a bf16 fine-tune) are not supported yet. Needs the native install, one RTX 3090 and ~75 GB of
disk (source ~20, captured activations ~35, output ~17).

```bash
V=venv/bin/python
SRC=models/Inkling-2-Mini        # the source checkpoint
OUT=models/Inkling-2-Mini-W4A16-g128-gptq
# 0. the public datasets the prompts come from (licenses in NOTES.md)
hf download --repo-type dataset NousResearch/hermes-function-calling-v1 \
    func-calling.json func-calling-singleturn.json --local-dir data/calib-src/hermes-fc
hf download --repo-type dataset ise-uiuc/Magicoder-Evol-Instruct-110K \
    data-evol_instruct-decontaminated.jsonl --local-dir data/calib-src/magicoder
hf download --repo-type dataset pixparse/docvqa-single-page-questions \
    data/train-00002-of-00036.parquet --local-dir data/calib-src/docvqa
hf download --repo-type dataset HuggingFaceH4/ultrachat_200k \
    data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet --local-dir data/calib-src/ultrachat
hf download --repo-type dataset wikimedia/wikipedia \
    20231101.it/train-00002-of-00010.parquet --local-dir data/calib-src/wiki-it
# 1. the source answers the prompts itself: 82 calibration transcripts (~166k tokens) and
#    22 held out for the check (server on :18030 with the tool and reasoning parsers)
MODEL=$SRC MM_LIMIT='{"image":1}' MAX_LEN=16384 MAX_SEQS=6 EAGER= scripts/serve-try.sh gen \
    --max-num-batched-tokens 512 --reasoning-parser inkling --enable-auto-tool-choice \
    --tool-call-parser inkling &        # wait for curl -sf 127.0.0.1:18030/health
$V prepare/build_calib.py data/calib/calib.jsonl data/calib/eval.jsonl   # then stop the server
# 2. capture the experts' inputs and routing while the source rereads them (eager server)
HYPERINKLING_CAPTURE_DIR=$PWD/data/capture MODEL=$SRC MM_LIMIT='{"image":1}' MAX_LEN=16384 \
    MAX_SEQS=2 scripts/serve-try.sh capture --max-num-batched-tokens 512 \
    --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling &   # wait
$V prepare/replay_capture.py data/calib/calib.jsonl data/capture          # then stop it
# 3. GPTQ every layer's experts (~5 min on the GPU)
$V prepare/gptq_experts.py $SRC data/capture $OUT
# 4. check: teacher-forced KL against the source on the held-out transcripts
bench/kl_sweep.sh data/calib/eval.jsonl bench/kl/mine ref=$SRC int4=$OUT
$V bench/kl_eval.py compare bench/kl/mine/ref.json bench/kl/mine/int4.json data/calib/eval.jsonl
```

Serve the result with `MODEL=$OUT ./start.sh`. For reference, the published checkpoint scores
a KL median of 0.0030 against the release (round to nearest,
`prepare/quantize_experts.py`, 0.0039). `prepare/calib_round.sh` runs steps 1-4 for a further
round with a new seed; a second round of 166k tokens changed nothing measurable here.

## Repository

| path | what |
|---|---|
| `start.sh` | the launcher; every knob is an env var |
| `scripts/wizard.py` | asks the questions, writes `.env` |
| `Dockerfile`, `docker-compose.yml`, `scripts/build-image.sh` | the container |
| `patches/` | the vLLM patch and its base commit (`BASE`) |
| `scripts/` | `setup-vllm.sh`, `build-vllm.sh`, `serve-try.sh`, `quick_bench.py`, `warmup.py`, `verify-image.py` |
| `prepare/` | INT4 packing, RTN and GPTQ converters, calibration corpus and capture |
| `bench/` | speed (fixed contexts, per kind of text, long prompts, server-side metrics), memory matrix, KL eval, quality battery, profiles |
| `tests/` | kernel and processor tests (`venv/bin/python -m pytest -q tests/`) |
| `docs/` | measured tables |

## Credits

- [HyperQwen](https://github.com/syv-ai/HyperQwen): the idea, the name, and several of the
  approaches here (a pinned vLLM plus a patch, embeddings in host RAM, the launcher and
  wizard style, the one-step-in-flight trade for context).
- [basalt](https://github.com/jesdga95/basalt) (MIT): its DSpark draft policy (verify window
  by expected tokens per millisecond) and its handling of oversized images informed ours.
- vLLM's Inkling 2 support ([#69028](https://github.com/vllm-project/vllm/pull/69028)), and
  Thinking Machines Lab for the model and its drafter.
- Ar4ikov's vLLM PR [#69165](https://github.com/vllm-project/vllm/pull/69165) (Inkling 2 on
  SM80/86/89), our vLLM base; his [#69141](https://github.com/vllm-project/vllm/pull/69141)
  (compressed-tensors experts) and
  [inkling2-sm86-kernels](https://github.com/agent2x0r/inkling2-sm86-kernels) (a 2x3090
  bring-up with SM86 kernel labs): ideas for the next steps (FOLLOWUPS.md).
- [club-3090](https://github.com/noonghunna/club-3090), which serves Inkling 2 Mini on 3090s
  too: its release guards (a published tag never moves, the compiled ops checked on the GPU).
- [crazydiamond-on-ampere](https://github.com/Mr-Moonsilver/crazydiamond-on-ampere) (Apache-2.0):
  its scheduler (short turns ahead of a long prefill) and its KV in host RAM prompted
  `LONG_PREFILL` and our host-RAM experiments (FOLLOWUPS.md).
- [llm-compressor](https://github.com/vllm-project/llm-compressor): the route for calibrated
  quantization of the parts we quantize at load today.

## License

Apache-2.0, as the model and vLLM.
