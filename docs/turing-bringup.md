# Bringing HyperInkling up on a 2080 Ti (Turing, sm75)

You need only the 2080 Ti: there is no second GPU to compare against. Every step checks
the card against something fixed: an fp32 reference in the kernel tests, fp16's own
range in the overflow check, and the published task scores in the README. Do the steps
in order, and stop at the first one that fails. Each step says what to send back when it
does.

Assumed: Linux, or WSL2 on Windows, with an NVIDIA driver for CUDA 13 (580 or newer). The
22 GB card shows about 22 GiB in `nvidia-smi`. To keep an existing CUDA or llama.cpp setup
as it is, build in a container or in `./venv` with a CUDA 13 toolkit kept beside the system
one: README, "Next to an existing CUDA setup". The steps below use the native build. In the
container, run the same commands with `docker compose exec hyperinkling`.

## 0. Offline compile check (any machine, no GPU)

```bash
python3 -m pip install torch "triton~=3.8.0" regex numpy   # triton 3.8 = vLLM's torch 2.14
python3 bench/sm75_compile.py
```

This compiles every Triton kernel on the Turing path for sm75, at the model's shapes in
fp16, and checks each one's shared memory against Turing's 64 KB per block. The DSpark
drafter's kernels are included. It should end with `0 over 65536 B, 0 steps failed`. It
found the attention tiles (72-96 KB) and the W8 prefill tile (128 KB) over the limit, and
those are now smaller on sm75.

## 1. Build

```bash
scripts/setup-vllm.sh                 # vllm/ at patches/BASE + the patch
ARCH=7.5 scripts/build-vllm.sh        # into venv/, sm75 only (30-60 min); CUDA_HOME=... for a side toolkit
scripts/build-sm75-kernels.sh         # venv/hyperinkling_sm75.so, about a minute
```

The second build is attention (prefill and decode) on the fp16 tensor cores (Triton
cannot emit them for Turing). Run it again after a `git pull` that changes `kernels/sm75/`.
`start.sh` uses the library when it is there and says so in its banner;
`HYPERINKLING_SM75_LIB= ./start.sh` serves with the Triton kernel instead.
`venv/bin/python -m pytest -q -s tests/test_sm75_attention.py` checks it against the Triton
kernel and times both at 90k tokens.

The last line prints the GPU's capability, which should be `(7, 5)`. The build records its
CUDA toolkit in `venv/cuda_home`, and `start.sh` serves with that toolkit. FlashInfer
compiles kernels with `nvcc` at the first boot, and a CUDA 12 `nvcc` fails there. If the build fails,
send the first `error:` lines.

## 2. Kernel tests on the card (no checkpoint)

```bash
uv pip install --python venv/bin/python pytest   # only for a venv built before the build installed it
venv/bin/python -m pytest -q -s tests/test_fp16.py tests/test_e4m3.py \
    tests/test_sconv_padding.py tests/test_sconv_state_manager.py tests/test_image_budget.py
```

`test_fp16.py` runs each kernel the port changed, in fp16 against fp32, including Marlin's
INT4 experts and the DSpark drafter's kernels. `OutOfResources` means a tile is too big for Turing: send the kernel name.
Some other tests in `tests/` need the original release checkpoint (`models/Inkling-2-Mini`)
and bf16, so leave them out.

## 3. Checkpoint and first boot

```bash
hf download Pentium95/Inkling-2-Mini-W4A16-g128-gptq --local-dir models/Inkling-2-Mini-W4A16-g128-gptq
mkdir -p .local          # not in git: tee cannot create the log without it
HYPERINKLING_FP16_CHECK=1 SPEC=off MAX_LEN=8192 WARMUP=0 ./start.sh 2>&1 | tee .local/fp16-first.log
```

`start.sh` is the server, so it runs in the foreground until Ctrl-C. The log fills while it
runs, so use a second shell for the rest.

`start.sh` sees compute capability 7.5. It turns on fp16 and passes `--dtype float16`.
This first boot leaves DSpark off (`SPEC=off`) so that only the target model is checked.
`HYPERINKLING_FP16_CHECK=1` runs eager (no CUDA graphs) and logs the
residual stream's largest |value| after each layer visit, each time that value sets a new
record above 4096. In another shell, once `curl -s localhost:8080/health` answers:

```bash
python3 scripts/quick_bench.py 1 256
grep -E 'fp16 check|clamped|Error' .local/fp16-first.log
```

What the log means:
- **No `fp16 check` lines, or records well below 65504:** fp16 has headroom.
- **`non-finite residual after layer L pass P`:** the residual stream overflows fp16 there.
  The fix is to keep the residual blocks in fp32 on this path (`Inkling2LoopModel.forward`
  in `loop.py`, about 2x the memory of that buffer). Send the lines.
- **`values past fp16's range clamped`:** a checkpoint tensor was clamped at load. A few
  values are harmless; send the tensor names if there are many.

## 4. DSpark in fp16

Its bf16 weights are cast to fp16 at load like the target's. Its kernels round to fp16
where the bf16 path rounds to bf16, except the Markov gate, which keeps the reference's bf16
roundings. DSpark is on by default, so this boot drops `SPEC=off`:

```bash
HYPERINKLING_FP16_CHECK=1 MAX_LEN=8192 WARMUP=0 ./start.sh 2>&1 | tee .local/fp16-dspark.log
python3 scripts/quick_bench.py 1 256
grep -E 'clamped|Error|Traceback' .local/fp16-dspark.log
venv/bin/python bench/spec_bench.py 8080 256              # draft acceptance per kind of text
```

Acceptance comparable to the README's DSpark rows means the drafter works in fp16. Drafts
cannot change the output: the target verifies every one, so a weak drafter costs only
speed. The drafter's RMSNorms run on a Triton kernel on sm75, so no `nvcc` is needed at
boot.

## 5. Quality against the published scores

Run the same battery the README reports, on its own server (CUDA graphs on, the GPU must
be free), into one file to share:

```bash
STEPS="system quality" scripts/turing-report.sh quality     # .local/turing-report-quality.txt
```

Compare with README "Task accuracy", HyperInkling as served. GSM8K 0.970, tool calls 0.615
(names 0.710), DocVQA ANLS 0.935 (valid JSON 99.5%). Noise is about 1 to 3 points at
n=200. Inside that, fp16 is good. A clear drop (say GSM8K below 0.93) means fp16 is losing
precision somewhere: run step 3's check on a failing prompt and send its lines.

## 6. Memory and speed on the card

```bash
bench/memory_matrix.sh .local/memory_matrix_22g.tsv     # one boot per combination, ~2 min each
bench/context_bench.sh turing                           # the README speed table, on this card
SPECS=dspark CONTEXTS="8192 90000" CONC=1 MAX_SEQS=1 bench/context_bench.sh turing90k  # one long request
scripts/turing-report.sh <name>   # all of it in one file to share: .local/turing-report-<name>.txt
venv/bin/python bench/sm75_w8_gemv.py                   # W8 GEMV tiles on this card (no server)
MAX_LEN=8192 bench/profile_spec.sh t6 1 128             # kernel time per decode step
```

The matrix rows carry `gpu_gib` 22. Append them to `docs/memory_matrix.tsv`, and the
wizard then uses them in place of the 24 GiB estimate.

### A desktop on the same card

A compositor on the 2080 Ti takes turns with the server: measured at a fixed verify length
(`HYPERINKLING_SPEC_ADAPTIVE=`), a decode step took 39.7 ms with the desktop running and
36.5 ms without it at 90k context (31.0 and 28.0 ms at 8k), prefill was 11% faster, and the
fitted context rose from 120k to 145k tokens. For a server that should be as fast as the
card allows, stop the display manager (`sudo systemctl isolate multi-user.target`, which
closes the graphical session; `graphical.target` brings it back) or drive the display
from another GPU.

## 7. INT4 attention and LM head (opt-in)

Off by default. `HYPERINKLING_ATTN_W4=1` puts every attention layer's projections in INT4
(or a list such as `0-5,18-23`; the others stay INT8), `HYPERINKLING_HEAD_W4=1` the LM head.
Groups of 64 inputs by default (`HYPERINKLING_{ATTN,HEAD}_W4_GROUP=32|64|128`), round to
nearest after a clip search. It halves the bytes the decode GEMV reads; on the 3090 Ti,
round to nearest raised the KL against the original checkpoint (FOLLOWUPS "Tried and
dropped"), so it stays off until it is measured here.

```bash
STEPS="system test kernels" bench/attn_w4.sh k1   # no server, ~5 min: per-call INT4 vs INT8
bench/attn_w4.sh v1                               # + speed at 8k/90k and KL per variant, ~1-1.5 h
```

Both write `.local/attn-w4-<name>.txt`. Run it headless (a desktop moves ms/step by ~3 ms).
A variant is kept only if its KL is close to the reference and `bench/quality_battery.sh`
(`STEPS=quality scripts/turing-report.sh <name>` with the variant's variables exported)
stays within noise of GSM8K 0.965, tools 0.605, DocVQA 0.936.

## What is not done

- DSpark in fp16 has not run yet (step 4).
- The kernels' tiles are the 3090's, shrunk only where Turing's shared memory forced it.
  They are not tuned for the 2080 Ti's speed.
