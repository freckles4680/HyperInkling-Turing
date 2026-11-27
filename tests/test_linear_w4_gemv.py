"""INT4 attention / LM head linears (`ops/linear_w4_gemv.py`, HYPERINKLING_ATTN_W4,
HYPERINKLING_HEAD_W4) on synthetic weights: the packing, the clip search, the decode
GEMV (every tile the bench searches), the three-input launch and the prefill path, against
the dequantized weight in fp32. No checkpoint needed.

    venv/bin/python -m pytest -q -s tests/test_linear_w4_gemv.py

Without a GPU it runs in Triton's interpreter on the CPU, at smaller shapes:
    TRITON_INTERPRET=1 python -m pytest -q tests/test_linear_w4_gemv.py
"""

import os

import pytest
import torch

from vllm.models.inkling2.nvidia.ops.linear_w4_gemv import (
    W4GLinear,
    dequant_w4g_rows,
    dequantize_w4g,
    quantize_w4g,
    unpack_w4g,
    w4_layer_selected,
    w4g_gemv,
)

GPU = torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1"
DEV = "cuda" if GPU else "cpu"
K = 2304 if GPU else 256
N = 4096 if GPU else 96
DT = torch.float16


def _rel(got, want):
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


def _weight(n=N, k=K, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=g) * 0.02
    w[:, ::97] *= 6  # a few outlier inputs, as attention weights have
    return w.to(DT).to(DEV)


@pytest.mark.parametrize("group", [32, 64, 128])
def test_pack_roundtrip(group):
    w = _weight()
    packed, scale, zero = quantize_w4g(w, group, clip=False)
    assert packed.dtype == torch.int32 and packed.shape == (N, K // 8)
    codes = unpack_w4g(packed)
    assert codes.min() >= 0 and codes.max() <= 15
    # Without clipping, every weight is within half a step of its code.
    deq = dequantize_w4g(packed, scale, zero)
    step = scale.float().repeat_interleave(group, 1)
    assert ((deq - w.float()).abs() <= step * 0.5 + 1e-6 + w.float().abs() * 1e-2).all()
    # The plane layout: input 16 i + j of a tile is nibble i of word j.
    s = scale.float()[0, 0]
    z = zero.float()[0, 0]
    word0 = int(packed[0, 0]) & 0xFFFFFFFF
    for i in range(min(8, group // 16)):
        code = (word0 >> (4 * i)) & 15
        assert abs((code - z) * s - deq[0, 16 * i].item()) < 1e-6


@pytest.mark.parametrize("group", [32, 64, 128])
def test_clip_search_lowers_error(group):
    w = _weight()
    plain = dequantize_w4g(*quantize_w4g(w, group, clip=False))
    clipped = dequantize_w4g(*quantize_w4g(w, group, clip=True))
    e_plain, e_clip = _rel(plain, w), _rel(clipped, w)
    print(f"g{group}: rel error {e_plain:.4f} min/max, {e_clip:.4f} clip search")
    assert e_clip <= e_plain + 1e-6
    assert e_clip < 0.2  # the synthetic outlier inputs widen the groups


TILES = [(1, 16, 2), (1, 8, 1), (1, 32, 4), (2, 8, 1), (4, 8, 1), (4, 4, 1), (8, 4, 1), (8, 8, 2)]


@pytest.mark.parametrize("group", [32, 64, 128])
@pytest.mark.parametrize("tile", TILES if GPU else TILES[:3] + TILES[5:7])
@pytest.mark.parametrize("m", [1, 3, 5] if not GPU else [1, 2, 3, 4, 5, 8, 13, 64])
def test_gemv(group, tile, m):
    w = _weight()
    packed, scale, zero = quantize_w4g(w, group)
    ref_w = dequantize_w4g(packed, scale, zero)
    x = (torch.randn(m, K) * 0.5).to(DT).to(DEV)
    want = x.float() @ ref_w.T
    for out_dtype in (DT, torch.float32):
        out = torch.empty(m, N, dtype=out_dtype, device=DEV)
        w4g_gemv(x, packed, scale, zero, out, tile)
        assert _rel(out, want) < 2e-3, (tile, m, out_dtype)


@pytest.mark.parametrize("m", [1, 4, 65, 300] if GPU else [1, 4, 70])
def test_linear_call(m):
    w = _weight()
    lin = W4GLinear(w, 64)
    x = (torch.randn(m, K) * 0.5).to(DT).to(DEV)
    want = x.float() @ dequantize_w4g(lin.packed, lin.scale, lin.zero).T
    got = lin(x)
    assert got.dtype == DT and got.shape == (m, N)
    assert _rel(got, want) < 2e-3
    head = lin(x, torch.empty(m, N, dtype=torch.float32, device=DEV))  # the LM head's fp32
    assert _rel(head, want) < 2e-3
    # Prefill dequantizes on the GPU kernel: the same weight as the reference.
    rows = slice(0, 40)
    deq = dequant_w4g_rows(lin.packed[rows], lin.scale[rows], lin.zero[rows], DT)
    assert torch.equal(deq, lin.dequantize(rows).to(DT))


@pytest.mark.parametrize("m", [1, 3, 5] if not GPU else [1, 2, 3, 5, 8, 64, 100])
def test_multi_three_inputs(m):
    """A full layer: q side, k and v from three inputs, one launch at decode sizes."""
    n_q, kv = (N - 2 * 32, 32) if not GPU else (8600 - 2 * 1024, 1024)
    n = n_q + 2 * kv
    w = _weight(n)
    lin = W4GLinear(w, 128)
    ref_w = dequantize_w4g(lin.packed, lin.scale, lin.zero)
    xs = [(torch.randn(m, K, generator=torch.Generator().manual_seed(i)) * 0.5).to(DT).to(DEV)
          for i in range(3)]
    ranges = [(0, n_q), (n_q, n_q + kv), (n_q + kv, n)]
    got = lin.multi(xs, ranges)
    for x, (lo, hi), y in zip(xs, ranges, got):
        assert y.shape == (m, hi - lo)
        assert _rel(y, x.float() @ ref_w[lo:hi].T) < 2e-3


def test_layer_selection():
    assert not w4_layer_selected("", 3) and not w4_layer_selected("0", 3)
    assert w4_layer_selected("1", 3) and w4_layer_selected("all", None)
    assert w4_layer_selected("0-5,18-23", 5) and w4_layer_selected("0-5,18-23", 18)
    assert not w4_layer_selected("0-5,18-23", 6) and not w4_layer_selected("0-5", None)
    assert w4_layer_selected("7,9", 9) and not w4_layer_selected("7,9", 8)
