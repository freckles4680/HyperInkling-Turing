# HyperInkling-Turing

**[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines
Lab, Apache-2.0) on one Turing GPU, with vLLM.** Built and measured on a modded RTX 2080 Ti
with 22 GB. Text and images (document pages), tool calls, strict JSON output, up to ~145k
tokens of context, and the model's own DSpark drafter for speculative decoding.

This is a fork of [Pento95/HyperInkling](https://github.com/Pento95/HyperInkling), which
ports Inkling 2 Mini to one RTX 3090 / 3090 Ti (Ampere, sm86). Upstream vLLM runs the model
on Hopper and Blackwell only. Turing (sm75) is missing a lot of what the Ampere port relies
on: bf16, FP8, Triton tensor-core code, and more than 64 KB of shared memory per block.
This fork runs the model in fp16 and replaces the kernels Triton cannot build well for
Turing:

- **Attention.** Prefill and long-context decode run on the fp16 tensor cores, in a CUDA
  `wmma` kernel built outside vLLM (`kernels/sm75/`, `scripts/build-sm75-kernels.sh`).
- **Linear layers at decode size.** INT8 GEMVs that use no `tl.dot`, with tiles measured on
  a 2080 Ti (`gemv_sm75`, `gemv3_sm75`).
- **Tile sizes.** Smaller tiles wherever the Ampere ones went past 64 KB of shared memory.
  `bench/sm75_compile.py` checks every kernel offline, with no GPU.

The sm86 path is unchanged, so the same tree still serves a 3090.

> **Scope, honestly.** A personal project, written almost entirely with an AI coding
> assistant. It is tested on **one modded 22 GB RTX 2080 Ti on native Linux** and nothing
> else. Other Turing cards below should work if they have the memory, but nobody has run
> them yet.

## Hardware

| card | memory | status |
|---|---|---|
| RTX 2080 Ti, 22 GB mod | 22 GB | **measured** (everything on this page) |
| Titan RTX, Quadro RTX 6000 / 8000 | 24 / 24 / 48 GB | should work, untested |
| RTX 2080 Ti (stock), 2080, 2070, 2060, T4 | 11 GB or less, T4 16 GB | **do not fit**: the checkpoint alone is ~17 GB |
| RTX 3090 / 3090 Ti (Ampere) | 24 GB | upstream's target, still supported here |

You also need an NVIDIA driver for CUDA 13 (580 or newer), and 32 GB of system RAM: the
token embeddings live in pinned host RAM. A display on the same card costs speed and
context (see below), so run headless when you can.

## Results on the 2080 Ti (22 GB, native Linux)

One request, DSpark on, KV cache in 8 bits, default settings
(`SPECS=dspark CONTEXTS="8192 90000" CONC=1 MAX_SEQS=1 bench/context_bench.sh`).

| | headless | desktop running on the card |
|---|---|---|
| decode at 90k context | **93 tok/s** (36.5 ms a step) | 83-85 tok/s (39.7-40.8 ms a step) |
| prefill at 90k context | **1,154 tok/s** | 1,071 tok/s |
| decode at 8k context | 84 tok/s (28.0 ms a step) | 85 tok/s (31.0 ms a step) |
| prefill at 8k context | 2,445 tok/s | 2,250 tok/s |
| longest request (`MAX_LEN=auto`, one slot) | ~145k tokens | ~120k tokens |

Decode speed in tokens per second depends on how many drafts the model accepts. DSpark gives
2.7 to 3.6 tokens a step, depending on the text and the run, so 8k and 90k can look alike in
tok/s even though the step at 8k is faster. Step times are the stable number, and the one
to compare between builds. For reference, the same model in Unsloth on this card gives about
40 tok/s.

A desktop on the same GPU takes turns with the server. It costs about 3 ms a step, 11% of
prefill and 25k tokens of context. To stop it,
run `sudo systemctl isolate multi-user.target` (this closes the graphical session;
`graphical.target` brings it back), or drive the display from another GPU.

**Quality on the card** (`bench/quality_battery.py`, greedy, 200 items each, about +-1 to +-3
points of noise). It is within noise of the 3090 Ti, so fp16 costs nothing measurable:

| | original checkpoint (3090 Ti) | upstream as served (3090 Ti) | **this fork (2080 Ti, fp16)** |
|---|---|---|---|
| GSM8K | 0.970 | 0.970 | **0.965** |
| tool calls, exact (names only) | 0.645 (0.735) | 0.615 (0.710) | **0.605** (0.695) |
| DocVQA as strict JSON, ANLS (valid JSON) | 0.938 (99.5%) | 0.935 (99.5%) | **0.936** (99.0%) |

### Why a 3090 is so much faster

Upstream reports **267 tok/s** decode on a 3090 Ti. That is at 8k context, one request,
DSpark on, greedy decoding, with the card undervolted to 350 W and
[gpu-unlock](https://github.com/zagitta/gpu-unlock) on. Without DSpark it is 131 tok/s, and
at 32k context 188 tok/s. Upstream publishes no 90k figure. Their 686 tok/s is the total
over 6 requests at once, not one chat. The model, checkpoint, drafter and acceptance are
the same as here, so the gap comes from how long one step takes:

- **Memory bandwidth, about 1.5-1.6x.** Decode reads the weights once a step: 1,008 GB/s on a
  3090 Ti and 936 on a 3090, against 616 on a 2080 Ti.
- **Kernels, roughly another 1.5-2x (an estimate: the rest of the gap).** On Ampere, Triton puts the verify step's 3-8 rows on
  tensor cores, uses bf16, async copies and ~100 KB of shared memory, and Marlin is tuned
  for it. On Turing, Triton emits no tensor-core instructions at all. Attention needed a
  hand-written CUDA kernel, the linears a dot-free GEMV, and every tile had to shrink to
  fit 64 KB.
- **Prefill compute.** gpu-unlock lifts the 3090 Ti's tensor cores to ~102 TFLOPS, and
  prefill is compute-bound, which is why upstream reaches ~6,200 tok/s against our
  2,445 at 8k.

At 8k with a desktop running, a step takes ~28-31 ms here and, from upstream's figures, roughly 10-11 ms on the 3090 Ti, so
84 tok/s against 267. Upstream's full tables, as published, are in
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## What runs where (Turing)

| part | format | kernel on sm75 |
|---|---|---|
| activations | fp16 (Turing has no bf16); bf16 checkpoint tensors cast at load | |
| routed experts (most of the weights) | INT4 g128 asymmetric (AWQ layout), GPTQ-calibrated | Marlin MoE, fp16 |
| attention projections | INT8 g128, quantized at load | up to 64 rows: dot-free Triton GEMV; above: dequantized blocks on cuBLAS |
| LM head | INT8 g128, fp32 logits (shared with the drafter) | same |
| token embeddings | INT8 g128, in pinned host RAM | lookup |
| KV cache | FP8 E4M3, emulated in software (no FP8 hardware) | CUDA `wmma` attention on fp16 tensor cores (`kernels/sm75/`) |
| loop controllers, sconv taps, router | fp16 | dot-free GEMV / batched GEMV, cuBLAS above 64 rows |
| DSpark drafter | fp16 layers on cuBLAS (INT4 on Ampere: Triton's INT4 took ~40 ms a step here); Markov fold INT8; KV FP8 | cuBLAS, W8 GEMV |
| vision tower | fp16, on the GPU (or streamed from host RAM) | SDPA per image (FlashAttention 2 needs SM80) |

The checkpoint is the same as upstream's
([Pentium95/Inkling-2-Mini-W4A16-g128-gptq](https://huggingface.co/Pentium95/Inkling-2-Mini-W4A16-g128-gptq)).
The INT8 parts are made from it when the server loads. The patch's commit message
(`patches/vllm-hyperinkling-sm75.patch`) lists every change, NOTES.md has the reasons and the
measurements, and FOLLOWUPS.md what comes next.

## Setup (native Linux)

```bash
git clone https://github.com/freckles4680/HyperInkling-Turing hyperinkling && cd hyperinkling
scripts/setup-vllm.sh                  # vllm/ = vLLM PR #69165 at patches/BASE + this repo's patch
ARCH=7.5 scripts/build-vllm.sh         # into ./venv, sm75 only (30-60 min)
scripts/build-sm75-kernels.sh          # venv/hyperinkling_sm75.so: Turing tensor-core attention (~1 min)
hf download Pentium95/Inkling-2-Mini-W4A16-g128-gptq --local-dir models/Inkling-2-Mini-W4A16-g128-gptq
```

Building vLLM needs a CUDA 13 `nvcc`. If the system has another toolkit, for example CUDA 12
for llama.cpp, install CUDA 13 beside it without touching the system one, and pass it to the
build. The build records it in `venv/cuda_home`, and `start.sh` serves with it:

```bash
sh cuda_13.*_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-13
CUDA_HOME=$HOME/cuda-13 ARCH=7.5 scripts/build-vllm.sh
```

uv brings its own Python, and the torch wheels bring their own CUDA runtime, so the system
Python and llama.cpp are left alone. The two engines cannot hold the GPU at the same time,
so stop one before you start the other. Run `scripts/build-sm75-kernels.sh` again after a
`git pull` that changes `kernels/sm75/`. Tested on Fedora 44 (gcc 15 next to the default
gcc 16). Other distributions are best effort.

The first bring-up on a new card, step by step with what to check at each stage, is in
[`docs/turing-bringup.md`](docs/turing-bringup.md).

## Serving

```bash
scripts/serve.sh                       # foreground: run it inside tmux to keep it up
BIND=127.0.0.1 PORT=9000 API_KEY=secret scripts/serve.sh
```

`scripts/serve.sh` holds the 2080 Ti settings, each one commented at the top of the file,
and runs `./start.sh`, the launcher that lists every knob with its measurements. The defaults:

- **`BIND`** is `0.0.0.0` (the LAN). Use `127.0.0.1` for this machine only. It is not called
  `HOST`, because some shells already set `HOST` to the machine's name.
- **`PORT`** is `8080`. The OpenAI-compatible API is at `http://<machine>:8080/v1`, model
  `inkling-2-mini` (alias `local`). `GET /health` answers once it is up, after a minute or
  two, and `GET /` is a 404 by design.
- **`API_KEY`** is empty, so anyone who can reach the port can use it. Set one
  (`openssl rand -hex 16`) on any network you do not fully trust.
- **`MAX_SEQS=2`** is one chat plus the side requests front ends send (titles, tags), and
  **`MAX_LEN=auto`** gives the longest request the card holds.

Open the port in the firewall for the LAN (`sudo firewall-cmd --add-port=8080/tcp --permanent
&& sudo firewall-cmd --reload` on Fedora). Point Open WebUI, Unsloth Studio or any OpenAI
client at the base URL above.

**Speed shown in the chat.** `serve.sh` also starts
`scripts/speed_proxy.py` on port **8081**. It is the same API, with one line under each
streamed answer giving the prefill and decode speed of that request. Point the front end at
`:8081/v1` to see it, or at `:8080/v1` for the plain API, and open 8081 in the firewall too.

**Several models on one address.** `scripts/serve-switch.sh` replaces `serve.sh` with the
same ports. It loads whichever model a request names, one at a time, because only one fits
on the card. Open WebUI keeps chatting with `inkling-2-mini`. Cloudflare's
[Clef](https://huggingface.co/Cloudflare/clef), a decision model with no chat, answers
`POST /v1/systemone` on the same address. The 27B takes 15.6 GiB plus 4.6 GiB at its 64k
context, all on the GPU, and a fit check at boot confirms it; the 9B Clef-Flash is the
smaller option. `docs/clef.md` covers
the setup, the request format and the single log to send back.

**Sampling:** temperature 1.0 is the model's default. Its card says it can get stuck in long
reasoning loops, so cap `max_tokens` and the agent's turns.

### Docker

Build the image once (about an hour). It compiles vLLM for sm75, and compose runs it as
`hyperinkling:sm75`. Upstream's published image is sm86 only and does not run on Turing.

```bash
scripts/setup-vllm.sh
scripts/build-image.sh                 # hyperinkling:sm75 (ARCH=8.6 ... hyperinkling:sm86 for a 3090)
python3 scripts/wizard.py              # questions -> .env, with the 2080 Ti's defaults
docker compose up -d
```

The native route is the one measured above.

## Benchmarks

Every benchmark needs the GPU to itself. Scripts that boot their own server refuse a busy GPU or port.

```bash
scripts/turing-report.sh <name>        # GPU state, kernel tests, prefill/decode profiles, long-context bench:
                                       #   one file to share, .local/turing-report-<name>.txt
SPECS=dspark CONTEXTS="8192 90000" CONC=1 MAX_SEQS=1 bench/context_bench.sh <name>   # the results table
STEPS="system quality" scripts/turing-report.sh quality    # GSM8K, tool calls, DocVQA on their own server
venv/bin/python bench/spec_bench.py 8080 256               # draft acceptance per kind of text
venv/bin/python bench/sm75_w8_gemv.py                      # W8 GEMV tiles on this card (no server)
python3 bench/sm75_compile.py                              # every Turing kernel compiled offline, no GPU
```

To compare two builds, fix the verify length (`HYPERINKLING_SPEC_ADAPTIVE=`, so 4 drafts a
step) and compare milliseconds a step. Tokens per second moves with acceptance from run to
run. `bench/intelligence.sh` adds IFEval, MMLU-Pro and a judged MT-Bench
in one report.

## Quality against the original checkpoint

Teacher-forced KL against the original NVFP4/MXFP8 checkpoint on 22 held-out transcripts
(images, tool calls, code, Italian, chat; `bench/kl_eval.py`), measured upstream on a 3090 Ti.
Compare medians: the mean is dominated by the few positions where a small difference flips
an expert.

| | KL median | KL mean | top-1 agreement |
|---|---|---|---|
| experts INT4, round to nearest | 0.0039 | 0.133 | 91.2% |
| experts INT4 GPTQ | 0.0030 | 0.117 | 92.5% |
| **as served**: + INT8 attention, head, embeddings, FP8 KV | **0.0031** | 0.123 | 92.1% |

Speculative decoding does not change the output distribution: the target verifies every
draft. The task scores on the 2080 Ti are in the results section above.

## Making the checkpoint

The published checkpoint comes from upstream's pipeline in `prepare/`: routed experts
GPTQ-calibrated to INT4 g128 on the model's own text, everything else unchanged. It was run
on a 3090 and has not been run on Turing, because you only need it to quantize another
checkpoint with Inkling 2 Mini's architecture. The steps are in
[upstream's README](https://github.com/Pento95/HyperInkling#quantizing-a-model-for-hyperinkling).

## Repository

| path | what |
|---|---|
| `scripts/serve.sh` | the 2080 Ti server on the LAN, every setting commented |
| `scripts/serve-switch.sh`, `model_switch.py`, `models.toml` | one address, the model each request names (`docs/clef.md`) |
| `scripts/clef.sh`, `clef_server.py`, `setup-clef.sh` | Cloudflare Clef (decision model) on Turing |
| `start.sh` | the launcher; every knob is an env var |
| `kernels/sm75/`, `scripts/build-sm75-kernels.sh` | the Turing tensor-core attention kernel |
| `patches/` | the vLLM patch and its base commit (`BASE`) |
| `scripts/` | `setup-vllm.sh`, `build-vllm.sh`, `turing-report.sh`, `wizard.py`, `quick_bench.py`, ... |
| `bench/` | speed, profiles, memory matrix, KL eval, quality battery, `sm75_compile.py` |
| `tests/` | kernel tests (`venv/bin/python -m pytest -q tests/`; `test_fp16.py`, `test_sm75_attention.py` for Turing) |
| `docs/` | `turing-bringup.md`, measured tables, the upstream model cards |

## Credits

- [Pento95/HyperInkling](https://github.com/Pento95/HyperInkling): everything this fork
  starts from: the sm86 port, the checkpoint and its pipeline, the launcher and benches.
- [HyperQwen](https://github.com/syv-ai/HyperQwen), [basalt](https://github.com/jesdga95/basalt),
  [club-3090](https://github.com/noonghunna/club-3090),
  [crazydiamond-on-ampere](https://github.com/Mr-Moonsilver/crazydiamond-on-ampere) and
  [llm-compressor](https://github.com/vllm-project/llm-compressor), credited upstream for
  the ideas this builds on.
- vLLM's Inkling 2 support ([#69028](https://github.com/vllm-project/vllm/pull/69028)) and
  Ar4ikov's PR [#69165](https://github.com/vllm-project/vllm/pull/69165) (Inkling 2 on
  SM80/86/89), the vLLM base; Thinking Machines Lab for the model and its drafter.

## License

Apache-2.0, as the model and vLLM.
