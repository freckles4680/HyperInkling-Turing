"""short_conv on a padded decode batch above 256 tokens (the tile-skip path).

Padding rows must not hide the cache taps of the real tokens that share their tile:
each real row is compared against the same requests run unpadded, at most 256 tokens
at a time (the path without the skip). In the activation dtype: fp16 on Turing, bf16 otherwise.

    venv/bin/python -m pytest -q tests/test_sconv_padding.py
"""

import pytest
import torch

from vllm.models.inkling.nvidia.ops.sconv import sconv_seq_metadata
from vllm.models.inkling2.nvidia.ops import act_dtype
from vllm.models.inkling2.nvidia.ops.sconv import short_conv

H, WS, W, N = 8, 128, 4, 16  # heads, stream width, taps, cache block size


def _batch(lens, ctx, pad, blocks_per_req):
    """Metadata for requests of `lens` new tokens after `ctx` cached ones; block 0
    is the null block, as in vLLM (the conv never stores there)."""
    qsl = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    total = int(qsl[-1])
    positions, slots = [], []
    for r, (q, c) in enumerate(zip(lens, ctx)):
        for p in range(c, c + q):
            positions.append(p)
            slots.append((1 + r * blocks_per_req + p // N) * N + p % N)
    positions += [0] * pad
    slots += [-1] * pad
    T = total + pad
    seq_idx = torch.empty(T, dtype=torch.int32, device="cuda")
    qstart = torch.empty(T, dtype=torch.int32, device="cuda")
    sconv_seq_metadata(qsl.cuda(), len(lens), total, seq_idx, qstart, T)
    bt = 1 + torch.arange(len(lens) * blocks_per_req, dtype=torch.int32, device="cuda")
    return (
        torch.tensor(positions, dtype=torch.int64, device="cuda"),
        bt.view(len(lens), blocks_per_req),
        seq_idx,
        torch.tensor(slots, dtype=torch.int64, device="cuda"),
        qstart,
    )


@pytest.mark.parametrize("q", [3, 5])
def test_padded_decode_over_256(q):
    torch.manual_seed(0)
    reqs, blocks_per_req, pad = 95, 4, 13  # real tokens not a multiple of the tile
    lens = [q] * reqs
    ctx = torch.randint(W, blocks_per_req * N - q, (reqs,)).tolist()
    cache = torch.randn(1 + reqs * blocks_per_req, H, N, WS, device="cuda").to(act_dtype())
    weight = (torch.randn(H * WS, W, device="cuda") * 0.5).to(act_dtype())
    x = torch.randn(reqs * q, H * WS, device="cuda").to(act_dtype())
    assert reqs * q + pad > 256

    xp = torch.cat([x, torch.zeros(pad, H * WS, device="cuda", dtype=x.dtype)])
    got = short_conv(xp, weight, cache.clone(), *_batch(lens, ctx, pad, blocks_per_req), 0, WS)

    want = []
    per = 256 // q
    for s in range(0, reqs, per):
        sub_lens, sub_ctx = lens[s : s + per], ctx[s : s + per]
        sub_cache = cache[s * blocks_per_req : (s + len(sub_lens)) * blocks_per_req].clone()
        xs = x[s * q : (s + len(sub_lens)) * q]
        want.append(
            short_conv(xs, weight, sub_cache, *_batch(sub_lens, sub_ctx, 0, blocks_per_req), 0, WS)
        )
    want = torch.cat(want)
    torch.testing.assert_close(got[: reqs * q].float(), want.float(), atol=2e-2, rtol=2e-2)
