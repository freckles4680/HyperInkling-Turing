"""Software E4M3 of `ops/e4m3.py` (the KV store and load on SM75 and SM8x, which have no FP8) against PyTorch's
float8_e4m3fn conversion on the CPU, bit for bit.

    venv/bin/python -m pytest -q tests/test_e4m3.py
"""

import torch

from vllm.models.inkling2.nvidia.ops.e4m3 import cast_store, e4m3_f16_scaled
from vllm.triton_utils import tl, triton


@triton.jit
def _encode(X, Codes, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    tl.store(Codes + offs, cast_store(x, tl.uint8), mask=mask)


def _codes(x: torch.Tensor) -> torch.Tensor:
    x = x.cuda()
    codes = torch.empty(x.shape, dtype=torch.uint8, device="cuda")
    _encode[(triton.cdiv(x.numel(), 1024),)](x, codes, x.numel(), BLOCK=1024)
    return codes.cpu()


def _want(x: torch.Tensor) -> torch.Tensor:
    return x.cpu().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def test_encode_random():
    torch.manual_seed(0)
    x = torch.cat([torch.randn(1 << 20) * s for s in (0.003, 0.05, 1.0, 20.0, 150.0)])
    assert torch.equal(_codes(x), _want(x))


def test_encode_ties():
    """Every midpoint between neighbouring e4m3 values, and one fp32 ulp on
    either side: where an fp16 intermediate rounds twice (vLLM #69199)."""
    values = torch.arange(0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    mids = (values[:-1] + values[1:]) / 2
    x = torch.cat([
        mids,
        torch.nextafter(mids, torch.full_like(mids, float("inf"))),
        torch.nextafter(mids, torch.zeros_like(mids)),
    ])
    x = torch.cat([x, -x])
    assert torch.equal(_codes(x), _want(x))


def test_encode_saturation():
    x = torch.tensor([500.0, -1e6, 448.0, 449.0, 464.0, 1e-9, -1e-9])
    assert torch.equal(_codes(x), _want(x))


@triton.jit
def _decode_f16_scaled(Codes, Out, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = e4m3_f16_scaled(tl.load(Codes + offs, mask=mask, other=0))
    tl.store(Out + offs, v.to(tl.float32) * 256.0, mask=mask)


def test_decode_f16_scaled_all_codes():
    codes = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
    codes = codes[(codes & 0x7F) != 0x7F]  # NaN codes are never stored
    out = torch.empty(codes.shape, dtype=torch.float32, device="cuda")
    _decode_f16_scaled[(1,)](codes, out, codes.numel(), BLOCK=256)
    assert torch.equal(out, codes.view(torch.float8_e4m3fn).float())
