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
```

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

Restart without the check (CUDA graphs back on) and run the same battery the README
reports:

```bash
MAX_LEN=32768 ./start.sh                                   # in tmux
venv/bin/python bench/quality_battery.py download
venv/bin/python bench/quality_battery.py run gsm8k tools docvqa --n 200
```

Compare with README "Task accuracy", HyperInkling as served. GSM8K 0.970, tool calls 0.615
(names 0.710), DocVQA ANLS 0.935 (valid JSON 99.5%). Noise is about 1 to 3 points at
n=200. Inside that, fp16 is good. A clear drop (say GSM8K below 0.93) means fp16 is losing
precision somewhere: run step 3's check on a failing prompt and send its lines.

## 6. Memory and speed on the card

```bash
bench/memory_matrix.sh .local/memory_matrix_22g.tsv     # one boot per combination, ~2 min each
bench/context_bench.sh turing                           # the README speed table, on this card
venv/bin/python bench/sm75_w8_gemv.py                   # W8 GEMV tiles on this card (no server)
MAX_LEN=8192 bench/profile_spec.sh t5 1 128             # kernel time per decode step
```

The matrix rows carry `gpu_gib` 22. Append them to `docs/memory_matrix.tsv`, and the
wizard then uses them in place of the 24 GiB estimate.

## What is not done

- DSpark in fp16 has not run yet (step 4).
- The kernels' tiles are the 3090's, shrunk only where Turing's shared memory forced it.
  They are not tuned for the 2080 Ti's speed.
