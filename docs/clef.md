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
| here (NF4) | ~19 GB on the GPU (~16.5 with `CLEF_EMBED_CPU=1`) | ~8.5 GB |

The memory figures are estimates from the parameter counts. They have not been measured on
the card yet, and the boot log prints the real number.

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
  NVFP4 kernels need sm80 or newer. The LM head and the vision tower stay in fp16.
- **Optional token embeddings in host RAM** (`CLEF_EMBED_CPU=1`).
- **A one-time NF4 copy on disk**, so a model switch does not quantize 27B again.
- Full attention through PyTorch SDPA, whose memory-efficient kernel runs in fp16 on sm75.
  The DeltaNet layers use Transformers' PyTorch path. It is slower than the fla kernels,
  but it has no Triton tile limits.

## Setup (once)

```bash
scripts/setup-clef.sh                          # Clef 27B: venv, 55 GB download, NF4 copy (19 GB)
CLEF_MODEL=clef-flash scripts/setup-clef.sh    # and/or the 9B
DROP_ORIGINAL=1 scripts/setup-clef.sh          # delete the bf16 download after the NF4 copy
```

The script needs the GPU free for the NF4 step, so stop Inkling first. Everything it prints
goes to `.local/setup-clef.log`. It checks four things before it downloads anything:

- the torch wheel has sm_75 kernels (if not, rerun with
  `TORCH_INDEX=https://download.pytorch.org/whl/cu126`);
- the Qwen3.5 class imports;
- a bitsandbytes NF4 matmul runs on the card;
- fp16 SDPA runs with head 256 and GQA.

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

- A switch costs one boot: Inkling's minute or two, or Clef's load (not measured yet).
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
`CLEF_MAX_LEN` tokens (16384, the release's default), from the end of the state.

## What to send back after a first run

1. `.local/setup-clef.log`.
2. `.local/model-switch.log` after: Inkling chat in Open WebUI, then the curl above, then
   chat again (two switches). The Clef boot runs a self-test, whose expected answers are
   `technical`, urgency near 2 and outage true. `CLEF_SELFTEST_TOKENS=8000` in
   `scripts/clef.sh` adds one long request, with its time and peak memory.

The log answers what is still open: load time, GPU memory, prefill speed and fp16 headroom
on the card.
