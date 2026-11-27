# Clef on the 2080 Ti, and switching models

## What Clef is

[Clef](https://huggingface.co/Cloudflare/clef) (Cloudflare, Apache-2.0, released 2026-09-30)
is a **decision model**, not a chatbot. You give it a state (text, JSON, optionally images)
and typed questions. It returns a probability for every allowed answer, in one forward pass.
It writes no text, so there is nothing to stream and nothing to parse. Its model card says:
"There is no free-form text generation and no output parsing."

| | Clef | Clef-Flash |
|---|---|---|
| repo | `Cloudflare/clef` | `Cloudflare/clef-flash` |
| backbone | Qwen3.8-27B, 64 layers, hidden 5120 | Qwen3.5-9B, 32 layers, hidden 4096 |
| attention | 3 Gated DeltaNet (linear) layers to 1 full GQA layer (4 KV heads, head 256) | same |
| extra | vision tower (27 blocks), joint schema head (256 MB) | same |
| bf16 download | 55 GB | ~19 GB |
| on the GPU here, before any input | 15.6 GiB | 6.7 GiB |
| + one 16k / 32k / 64k-token input | 1.3 / 2.4 / 4.6 GiB (64k: 20.1 total, 1.9 free) | 0.9 / 1.9 / 3.6 GiB |

Everything stays on the GPU; nothing is offloaded to host RAM. The figures come from
`bench/clef_memory.py`, which builds the released module tree (it reproduces the 27B's
27,356,728,560 parameters) and runs a long forward on PyTorch's meta device, counting
every tensor alive at each step. It assumes 0.4 GiB of CUDA context. The boot fit check
below measures the real peak on the card.

27B budget, in GiB:

| part | GiB |
|---|---|
| 24.35B layer weights in NF4 | 11.70 |
| token embeddings + LM head as INT8 tables (fp16 would be 4.74) | 2.37 |
| vision tower (fp16) | 0.86 |
| joint schema head (fp16) | 0.24 |
| CUDA context (assumed) | 0.40 |
| activations at 16k / 32k / 64k tokens | 1.3 / 2.4 / 4.6 |

Context: Cloudflare gives Clef a 64k context (the backbone's positions go to 262k; 16384 is
only the release code's default cut). At 64k the 27B leaves about 1.9 GiB free on a
headless card, so with a desktop holding more than about 1 GiB the fit check settles on
48k instead. One 64k input is a long prefill on this card (minutes; the `FIT:` line
measures it).

The API is Jev/SystemOne (`POST /v1/systemone`), not OpenAI chat. Open WebUI cannot use it
for chat, so the switcher keeps it out of `/v1/models` and refuses a chat sent to it.

### Why it does not run on this repo's vLLM

The architecture is `Qwen3_5ForConditionalGeneration`. The vLLM in `patches/BASE` would need
three things for it:

- the Qwen3.5 hybrid model;
- its Gated DeltaNet Triton kernels, on a card where Triton has no tensor-core MMA and only
  64 KB of shared memory per block (see CLAUDE.md);
- the joint schema head, which vLLM does not have. It reads the last hidden states and the
  LM head's rows as a table, so it would be a custom pooling model.

Clef also needs no decode loop, KV cache or batching across steps, which is what vLLM is
for. Each request is one prefill.

So Clef runs on the release's own code (`joint_schema_model.py`, Transformers 5.10.2), with
what the 2080 Ti needs added in `scripts/clef_server.py`:

- **fp16 compute.** Turing has no bf16 tensor cores. Every request records the largest value
  each decoder layer outputs. An answer whose layers overflowed fp16 (65504) is refused,
  not returned.
- **NF4 weights** through bitsandbytes, the 4-bit path that runs on Turing. The Marlin and
  NVFP4 kernels need sm80 or newer. The vision tower stays in fp16.
- **Token embeddings and LM head as INT8 tables** (2.4 GiB less). Clef only looks rows up
  in both: the decision head averages LM-head rows of each option's tokens and never
  computes logits. Each row keeps one fp16 scale, and the error is under 0.4% of the row's
  largest value.
- **Every layer over 4,096 tokens at a time** on long inputs (`CLEF_CHUNK`). In one pass
  a 64k input would need about 10 GB of activations (4x the 16k one), mostly the MLP's 17,408-wide
  intermediates and the attention projections. The MLP is per token. The DeltaNet layers
  carry their conv and recurrent state from chunk to chunk, through the same
  continuation path Transformers uses with a cache. Full attention computes K/V for every
  position once, then runs the queries chunk by chunk with a lower-right causal mask, one
  KV head at a time, so K/V are never repeated to all 24 query heads.
  `tests/test_clef_memory.py` checks that the chunked model gives the one-pass output.
- **The DeltaNet layers 8 heads at a time.** These are 48 of the 64 layers. Without fla
  kernels, Transformers' PyTorch path makes about eight float32 copies of q, k and v for all
  48 heads: 4.9 GiB of activations at 16k tokens. The heads are independent, so splitting
  them gives the same answer bit for bit (`tests/test_clef_memory.py`) and uses 2.65 GiB.
- **Full attention on the memory-efficient SDPA kernel.** K and V are repeated to the 24
  query heads. With `enable_gqa` instead, sm75 has no flash kernel and PyTorch would fall
  back to the math kernel, which builds the whole 16k x 16k matrix (13 GB).
- **A fit check at boot.** It runs the longest input once (65536 tokens, then shorter until
  one fits with `CLEF_HEADROOM_GB`, default 1, still free). That length becomes the server's
  limit, so a request can't run out of memory later. The log says `FIT: clef takes inputs
  up to N tokens on this card`. If even 2048 tokens don't fit, it says `DOES NOT FIT` and
  exits. `setup-clef.sh` refuses before the download when the card is too small: about
  17.8 GiB for the 27B, 8.7 GiB for the 9B.
- **A one-time NF4 copy on disk**, so a model switch does not quantize 27B again.

## Setup (once)

```bash
scripts/setup-clef.sh                          # Clef 27B: venv, 55 GB download, NF4 copy (19 GB)
CLEF_MODEL=clef-flash scripts/setup-clef.sh    # and/or the 9B
DROP_ORIGINAL=1 scripts/setup-clef.sh          # delete the bf16 download after the NF4 copy
```

The script needs the GPU free for the NF4 step, so stop Inkling first. Everything it prints
goes to `.local/setup-clef.log`. It checks five things before it downloads anything:

- the card has enough memory for the model chosen;
- the torch wheel has sm_75 kernels (if not, rerun with
  `TORCH_INDEX=https://download.pytorch.org/whl/cu126`);
- the Qwen3.5 class imports;
- a bitsandbytes NF4 matmul runs on the card;
- the memory-efficient SDPA kernel runs with head 256.

## Serving both models on one address

```bash
scripts/serve-switch.sh                        # in tmux; replaces scripts/serve.sh
PRELOAD=inkling-2-mini scripts/serve-switch.sh # boot Inkling at once
```

It listens where `serve.sh` did: `:8080/v1`, and `:8081/v1` with the speed line. Open WebUI
needs no change and still lists `inkling-2-mini`.

`scripts/model_switch.py` reads each request's `model`. When that is not the model running,
it waits for the requests in flight to end, then stops the running server (its whole process
group) and waits for the GPU to be released. Then it starts the requested model, waits for
`/health`, and forwards the request. Requests that arrive during a switch wait for it.
For Inkling it also waits for start.sh's background warm-up (`warm-up done` in the log,
`ready_line` in models.toml): a chat sent during the warm-up shares both decode slots
with it and runs at about half speed.

- A switch costs one boot: Inkling's boot plus its warm-up (about 3.5 min on the card),
  or Clef's load (65 s measured, with a cached fit check less).
- Both models run on 127.0.0.1 without a key. `API_KEY` is checked at the switcher.
- `scripts/models.toml` lists the models and how to start them. Each model's settings stay
  in its own script (`serve.sh`, `clef.sh`). `[switch] idle_unload` frees the GPU after some
  idle time (off by default).

```bash
curl -s http://127.0.0.1:8080/switch/status                 # what runs, what is starting
curl -s -X POST http://127.0.0.1:8080/switch/load -d '{"model":"clef"}'   # switch ahead
curl -s -X POST http://127.0.0.1:8080/switch/unload         # free the GPU
```

**One log for everything:** `.local/model-switch.log` (the previous run's log is `.1`). It
holds the switch events and every line each server printed, prefixed with the model's
name. Send that file back.

Clef alone, without the switcher: `scripts/clef.sh` (127.0.0.1:18090; every setting is
commented at the top).

## Asking Clef

```bash
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "clef",
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle it?",
                   "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is a service down?"}
  }}'
```

The three question types are:

- **`noul`** is yes or no. The answer is `{"noul": <p(true)>}`.
- **`choice`** has named `criteria`. The answer gives `choice`, `confidence` and
  `probabilities`.
- **`score`** has ordered `criteria`. The answer gives `score` (the expected index),
  `confidence`, `legend` and `probabilities`.

`state` can be a string or any JSON. `images` is a list of base64 strings or `data:` URLs.
Videos are not accepted over HTTP; send frames as images instead. The input is cut at
the length the fit check found (`CLEF_MAX_LEN`, at most 65536), from the end of the state.
The fit result is kept in `.local/clef-fit-<checkpoint>.json` and reused on later boots
(every switch to Clef) while the settings match and at least as much GPU memory is free;
`CLEF_FIT=probe` measures again.

## What to send back after a first run

1. `.local/setup-clef.log`.
2. `.local/model-switch.log` after: Inkling chat in Open WebUI, then the curl above, then
   chat again (two switches). The Clef boot logs the fit check (its `FIT:` line, with the
   peak memory and the prefill speed at that length), then a self-test whose expected
   answers are `technical`, urgency near 2 and outage true.

The log answers what is still open: load time, the measured peak against the
`bench/clef_memory.py` estimate, prefill speed and fp16 headroom on the card.
