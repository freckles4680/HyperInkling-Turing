"""Turing (sm75): the tensor-core attention (kernels/sm75/rel_attention.cu) against
the Triton kernel it replaces, on random paged E4M3 caches.

    scripts/build-sm75-kernels.sh
    venv/bin/python -m pytest -q -s tests/test_sm75_attention.py

- test_device: the built library on the GPU against Triton's `rel_attention`.
- test_cpu_phases: the same source compiled for the CPU (g++, HOST_EMU: the kernel's phases
  one lane after the other) against Triton. If this passes and test_device fails, the fault
  is in what only the GPU build has (tensor-core fragments, warp timing), not in the math.
- test_speed (-s): both kernels for decode at 90k tokens (and blocks per SM for the new
  one), and for a prefill chunk.

The two kernels cut the keys into different blocks (32 against 64 keys), so the split
partials differ: the merged output and LSE are compared, as the combine kernel merges them.
"""

import os
import subprocess
from pathlib import Path

import pytest
import torch

from vllm.models.inkling2.nvidia.ops import attention_sm75, attention_sm90

REPO = Path(__file__).resolve().parent.parent
LIB = Path(os.environ.get("HYPERINKLING_SM75_LIB") or REPO / "venv/hyperinkling_sm75.so")
SOURCE = REPO / "kernels/sm75/rel_attention.cu"
LN2 = 0.6931471805599453

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _inputs(seed, q_lens, seq_lens, extent, heads=16, kv_heads=8, page=16):
    g = torch.Generator().manual_seed(seed)
    dim, tokens, reqs = 128, sum(q_lens), len(q_lens)
    pages = max(seq_lens) // page + 2
    blocks = reqs * pages + 3
    codes = torch.randint(0, 256, (2, blocks, page, kv_heads, dim), generator=g,
                          dtype=torch.uint8)
    # No NaN codes, magnitudes up to 1.75 (as normalized keys and values).
    codes = torch.where((codes & 0x78) > 0x38, codes & 0xC7 | 0x38, codes)
    table = torch.randperm(blocks, generator=g)[: reqs * pages].view(reqs, pages).int()
    return dict(
        q=(torch.randn(tokens, heads, dim, generator=g) * 0.5).half(),
        key_cache=codes[0].contiguous(), value_cache=codes[1].contiguous(),
        rel_logits=(torch.randn(tokens + 128, heads, extent + 256, generator=g) * 2).half(),
        block_table=table,
        cache_seqlens=torch.tensor(seq_lens, dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()],
                                  dtype=torch.int32),
    )


def _outputs(x, splits, mode):
    tokens, heads, dim = x["q"].shape
    if mode == attention_sm90.MODE_PARTIAL:
        shapes = (splits, tokens, heads, dim), (splits, tokens, heads)
    else:
        shapes = (tokens, heads, dim), (heads, tokens)
    return [x["q"].new_zeros(shape, dtype=torch.float32) for shape in shapes]


def _run(kernel, x, splits, draft, window, mode, **extra):
    out, lse = _outputs(x, splits, mode)
    kernel(
        x["q"], x["key_cache"], x["value_cache"], out, lse,
        None if draft else x["rel_logits"],
        block_table=x["block_table"], cache_seqlens=x["cache_seqlens"],
        cu_seqlens_q=x["cu_seqlens_q"],
        max_query_len=int(x["cu_seqlens_q"].diff().max()), softmax_scale=0.08,
        window=window, num_splits=splits, mode=mode, **extra)
    return out, lse


def _triton(x, splits, draft, window=0, mode=attention_sm90.MODE_PARTIAL):
    return _run(attention_sm90.rel_attention, x, splits, draft, window, mode)


def _sm75(x, splits, draft, lib, window=0, mode=attention_sm90.MODE_PARTIAL):
    return _run(attention_sm75.rel_attention_sm75, x, splits, draft, window, mode, lib=lib)


def _merged(out, lse):
    weight = torch.softmax(lse.double() * LN2, 0).nan_to_num(0.0)
    return (out.double() * weight[..., None]).sum(0), torch.logsumexp(lse.double() * LN2, 0)


def _compare(want, got, mode):
    if mode == attention_sm90.MODE_PARTIAL:
        want, got = _merged(*want), _merged(*got)
    (out_a, lse_a), (out_b, lse_b) = want, got
    finite = torch.isfinite(lse_a)
    assert torch.equal(finite, torch.isfinite(lse_b))
    assert (out_a - out_b).norm() / out_a.norm() < 2e-4
    assert (lse_a[finite] - lse_b[finite]).abs().max() < 2e-3


PARTIAL, OUT_LSE = attention_sm90.MODE_PARTIAL, attention_sm90.MODE_OUT_LSE
CASES = [
    # q_lens, seq_lens, splits, draft, extent, window, mode
    ([3], [200], 3, False, 128, 0, PARTIAL),
    ([1], [1], 2, False, 128, 0, PARTIAL),
    ([8], [8], 3, False, 128, 0, PARTIAL),
    ([8, 1], [333, 70], 2, False, 128, 0, PARTIAL),
    ([7], [150], 4, True, 128, 0, PARTIAL),
    ([2, 5], [40, 257], 5, False, 64, 0, PARTIAL),
    ([5, 2], [300, 90], 3, False, 100, 100, PARTIAL),
    ([7], [400], 2, True, 128, 130, PARTIAL),
    ([20, 3], [260, 50], 2, False, 128, 0, PARTIAL),
    ([70], [70], 1, False, 128, 0, OUT_LSE),
    ([40, 9], [300, 9], 1, False, 128, 0, OUT_LSE),
    ([200], [330], 1, False, 96, 96, OUT_LSE),
    ([150, 33], [150, 200], 1, False, 64, 0, OUT_LSE),
    # The model's sizes: decode and a prefill chunk over a long context.
    ([4, 8, 1], [5000, 64, 33], 34, False, 1024, 0, PARTIAL),
    ([8], [20000], 34, False, 1024, 0, PARTIAL),
    ([8], [3000], 9, False, 1024, 1024, PARTIAL),
    ([2048], [6000], 1, False, 1024, 0, OUT_LSE),
    ([2048], [6000], 1, False, 1024, 1024, OUT_LSE),
]
CPU_CASES = 13  # the CPU build takes minutes on the last ones


def _device(x):
    return {name: value.cuda() for name, value in x.items()}


@pytest.mark.parametrize("case", CASES, ids=str)
def test_device(case):
    if not LIB.exists():
        pytest.skip(f"{LIB} is not built (scripts/build-sm75-kernels.sh)")
    q_lens, seq_lens, splits, draft, extent, window, mode = case
    x = _device(_inputs(1, q_lens, seq_lens, extent))
    lib = attention_sm75.load_library(str(LIB))
    _compare(_triton(x, splits, draft, window, mode),
             _sm75(x, splits, draft, lib, window, mode), mode)


@pytest.mark.parametrize("case", CASES[:CPU_CASES], ids=str)
def test_cpu_phases(case, tmp_path_factory):
    path = tmp_path_factory.getbasetemp() / "hyperinkling_sm75_host.so"
    if not path.exists():
        subprocess.run(["g++", "-x", "c++", "-std=c++17", "-O2", "-DHOST_EMU", "-shared",
                        "-fPIC", "-o", str(path), str(SOURCE)], check=True)
    q_lens, seq_lens, splits, draft, extent, window, mode = case
    x = _inputs(1, q_lens, seq_lens, extent)
    want = _triton(_device(x), splits, draft, window, mode)
    got = _sm75(x, splits, draft, attention_sm75.load_library(str(path)), window, mode)
    _compare([t.cpu() for t in want], got, mode)


def _time(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps * 1e3  # us


def test_speed():
    """Decode: one full layer's attention at 90k tokens (9 of them run per step). Prefill:
    a 2048-token chunk at the end of 8k and of 32k tokens, full and sliding-window layers
    (8 and 40 layer visits per chunk)."""
    if not LIB.exists():
        pytest.skip(f"{LIB} is not built (scripts/build-sm75-kernels.sh)")
    lib = attention_sm75.load_library(str(LIB))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    seq = 90000
    megabytes = seq * 8 * 128 * 2 / 1e6
    print(f"\n{sms} SMs; decode at {seq} tokens, {megabytes:.0f} MB of KV per call")
    for q_len in (1, 3, 8):
        x = _device(_inputs(2, [q_len], [seq], 1024))
        splits = max(2 * sms // 8, min(sms // 8, 6 * sms // 8))
        us = _time(lambda: _triton(x, splits, False))
        line = f"{q_len} tokens: Triton {us:8.0f} us ({splits} splits)"
        for per_sm in (2, 3, 4, 5, 8):
            splits = per_sm * sms // 8
            us = _time(lambda: _sm75(x, splits, False, lib))
            line += f" | {per_sm}/SM {us:6.0f} us {megabytes / us * 1e3:4.0f} GB/s"
        print(line)
    print("prefill, a 2048-token chunk:")
    for seq in (8192, 32768):
        x = _device(_inputs(3, [2048], [seq], 1024))
        for window in (0, 1024):
            old = _time(lambda: _triton(x, 1, False, window, OUT_LSE), reps=3)
            new = _time(lambda: _sm75(x, 1, False, lib, window, OUT_LSE), reps=3)
            name = "full layer" if window == 0 else "window 1024"
            print(f"  ending at {seq:5d} tokens, {name:11s}: Triton {old / 1e3:8.1f} ms, "
                  f"tensor cores {new / 1e3:7.1f} ms")
