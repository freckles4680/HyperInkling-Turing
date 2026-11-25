"""The fp16 activation path (Turing) on synthetic data: each kernel the port changed
(the DSpark drafter's included),
run with fp16 inputs against an fp32 reference (and in bf16 too on a GPU that has it).
No checkpoint needed. Step 2 of docs/turing-bringup.md.

    venv/bin/python -m pytest -q -s tests/test_fp16.py

The Marlin test needs the fp16 path on (`ops.act_dtype`: always on Turing, else
HYPERINKLING_FP16=1): it is skipped otherwise.
"""

import os
import sys

import pytest
import torch

from vllm.models.inkling2.nvidia.ops import act_dtype
from vllm.models.inkling2.nvidia.ops.bgemv import bgemv, controller_gemv
from vllm.models.inkling2.nvidia.ops.linear_w4 import W4Linear, dequantize_int4
from vllm.models.inkling2.nvidia.ops.linear_w8 import (
    W8Linear,
    dequantize_int8,
    quantize_int8,
)
from vllm.models.inkling2.nvidia.ops.loop import pair_rms_norm
from vllm.models.inkling2.nvidia.ops.moe_sm90 import swiglu
from vllm.models.inkling2.nvidia.ops.vision import scaled_residual_norm_

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prepare"))
import int4_awq  # noqa: E402

HIDDEN, INTER, TOPK, EXPERTS = 2304, 1152, 8, 128
GATE_CAP, UP_CAP = 4.0, 25.0

_BF16_MMA = torch.cuda.get_device_capability() >= (8, 0)
DTYPES = [torch.float16] + ([torch.bfloat16] if _BF16_MMA else [])


def _rel(got, want):
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tokens", [1, 6, 48, 512])
def test_w8(dtype, tokens):
    torch.manual_seed(0)
    w = torch.randn(4096, HIDDEN, device="cuda") * 0.02
    lin = W8Linear(w.to(torch.bfloat16))
    x = torch.randn(tokens, HIDDEN, device="cuda").to(dtype)
    got = lin(x)
    assert got.dtype == dtype and torch.isfinite(got).all()
    want = x.float() @ dequantize_int8(*quantize_int8(w.to(torch.bfloat16))).T
    rel = _rel(got, want)
    print(f"w8 {dtype} x {tokens}: {rel:.1e}")
    assert rel < 1e-2


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tokens", [1, 8, 48, 512])
def test_w4(dtype, tokens):
    torch.manual_seed(0)
    w = (torch.randn(4096, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    lin = W4Linear(w)
    x = torch.randn(tokens, HIDDEN, device="cuda").to(dtype)
    got = lin(x)
    assert got.dtype == dtype and torch.isfinite(got).all()
    want = x.float() @ dequantize_int4(lin.packed, lin.scale, lin.zero).T
    rel = _rel(got, want)
    print(f"w4 {dtype} x {tokens}: {rel:.1e}")
    assert rel < 2e-2


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tokens", [1, 6, 64])
def test_controllers(dtype, tokens):
    torch.manual_seed(0)
    streams, rank = 3, 64
    down = (torch.randn(streams, rank, 2 * HIDDEN, device="cuda") * 0.02).to(dtype)
    up = (torch.randn(streams, 2 * HIDDEN, rank, device="cuda") * 0.02).to(dtype)
    pair = torch.randn(streams, tokens, 2 * HIDDEN, device="cuda").to(dtype)
    want = (pair.float() @ down.float().transpose(1, 2)) @ up.float().transpose(1, 2)
    assert _rel(controller_gemv(pair, down, up), want) < 1e-2
    hidden = bgemv(pair, down)
    assert hidden.dtype == dtype
    assert _rel(bgemv(hidden, up, torch.float32), want) < 1e-2


@pytest.mark.parametrize("tokens", [1, 64])
def test_pair_rms_norm_dtype(tokens):
    dest = torch.randn(3, tokens, HIDDEN, device="cuda")
    message = torch.randn(3, tokens, HIDDEN, device="cuda")
    out = pair_rms_norm(dest, message)
    assert out.dtype == act_dtype()
    want = torch.cat(
        [t * torch.rsqrt(t.square().mean(-1, keepdim=True) + 1e-6) for t in (dest, message)],
        dim=-1,
    )
    assert _rel(out, want) < 1e-2


@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu(dtype):
    torch.manual_seed(0)
    rows = 64
    gate_up = (torch.randn(rows, 2 * INTER, device="cuda") * 3).to(dtype)
    gammas = torch.rand(rows, device="cuda") + 0.5
    src2dst = torch.arange(rows, device="cuda", dtype=torch.int32)
    got = swiglu(gate_up, gammas, src2dst, GATE_CAP, UP_CAP)
    assert got.dtype == dtype
    pairs = gate_up.float().view(rows, INTER, 2)
    gate, up = pairs[..., 0], pairs[..., 1]
    # The kernel's SiTU: softcap(gate) * sigmoid(gate) * softcap(up) * gamma.
    want = (GATE_CAP * torch.tanh(gate / GATE_CAP)) * torch.sigmoid(gate) * (
        UP_CAP * torch.tanh(up / UP_CAP)) * gammas[:, None]
    rel = _rel(got, want)
    print(f"swiglu {dtype}: {rel:.1e}")
    assert rel < 1e-2


@pytest.mark.parametrize("dtype", DTYPES)
def test_vision_residual(dtype):
    torch.manual_seed(0)
    hidden = torch.randn(16, 1024, device="cuda").to(dtype)
    update = torch.randn(16, 1024, device="cuda").to(dtype)
    gamma = torch.rand(1024, device="cuda").to(dtype)
    want = hidden.float() + update.float() * gamma.float()
    scaled_residual_norm_(hidden, update, gamma, None)
    assert _rel(hidden, want) < 1e-2


@pytest.mark.skipif(act_dtype() != torch.float16, reason="fp16 path off")
@pytest.mark.parametrize("tokens", [1, 6, 48, 512])
def test_marlin_int4_fp16(tokens):
    from vllm.models.inkling2.nvidia.ops.moe_marlin import MarlinExperts

    torch.manual_seed(0)
    dense, weights = [], {}
    for name, (n, k) in zip(("w13", "w2"), ((2 * INTER, HIDDEN), (HIDDEN, INTER))):
        w = torch.randn(EXPERTS, n, k, device="cuda") * 0.02
        q, z, s = int4_awq.quantize(w, clip_search=False)
        dense.append(int4_awq.dequantize(q, z, s))
        weights[name] = int4_awq.to_awq_layout(q, z, s)
    experts = MarlinExperts(weights, ((2 * INTER, HIDDEN), (HIDDEN, INTER)), "int4")
    x = (torch.randn(tokens, HIDDEN, device="cuda") * 0.5).to(torch.float16)
    ids = torch.stack(
        [torch.randperm(EXPERTS, device="cuda")[:TOPK] for _ in range(tokens)]
    ).to(torch.int32)
    gammas = torch.rand(tokens * TOPK, device="cuda") + 0.5
    src2dst = torch.arange(tokens * TOPK, device="cuda", dtype=torch.int32)
    got = experts(x, ids, gammas, src2dst, EXPERTS, GATE_CAP, UP_CAP)
    assert got.dtype == torch.float16 and torch.isfinite(got).all()

    w13, w2 = dense
    flat = ids.flatten().long()
    rows = x.repeat_interleave(TOPK, dim=0).float()
    gate_up = torch.empty((tokens * TOPK, 2 * INTER), device="cuda")
    for e in flat.unique():
        gate_up[flat == e] = rows[flat == e] @ w13[e].T
    act = swiglu(gate_up.half(), gammas, src2dst, GATE_CAP, UP_CAP).float()
    want = torch.empty((tokens * TOPK, HIDDEN), device="cuda")
    for e in flat.unique():
        want[flat == e] = act[flat == e] @ w2[e].T
    rel = _rel(got, want)
    print(f"marlin int4 fp16 x {tokens}: {rel:.1e}")
    assert rel < 2e-2


@pytest.mark.parametrize("dtype", DTYPES)
def test_dspark_kernels(dtype):
    """The DSpark drafter's kernels in `dtype` against fp32 math."""
    from vllm.models.inkling2.nvidia.ops.dspark import (
        add_softcap,
        markov_gate,
        mix_rmsnorm,
        qk_norm_rope_,
        silu_and_mul,
    )

    torch.manual_seed(0)
    rows, taps, d, hq, hk, rank = 16, 4, 128, 16, 4, 256
    targets = torch.randn(rows, taps, HIDDEN, device="cuda").to(dtype)
    alpha = torch.softmax(torch.randn(taps, device="cuda"), 0)
    weight = (torch.rand(HIDDEN, device="cuda") + 0.5).to(dtype)
    mixed = (targets.float() * alpha[:, None]).sum(1)
    want = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6) * weight.float()
    assert _rel(mix_rmsnorm(targets, alpha, weight, 1e-6), want) < 1e-2

    qkv = torch.randn(rows, (hq + 2 * hk) * d, device="cuda").to(dtype)
    q_w = (torch.rand(d, device="cuda") + 0.5).to(dtype)
    k_w = (torch.rand(d, device="cuda") + 0.5).to(dtype)
    inv_freq = 1.0 / 10000 ** (torch.arange(0, d, 2, device="cuda").float() / d)
    pos = torch.arange(rows, device="cuda") * 37
    heads = qkv.float()[:, : (hq + hk) * d].view(rows, hq + hk, d)
    w = torch.cat([q_w.float().expand(hq, d), k_w.float().expand(hk, d)])
    normed = heads * torch.rsqrt(heads.square().mean(-1, keepdim=True) + 1e-6) * w
    ang = pos[:, None].float() * inv_freq
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]
    x1, x2 = normed[..., : d // 2], normed[..., d // 2 :]
    want = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
    qk_norm_rope_(qkv, pos, q_w, k_w, inv_freq, hq, hk, 1e-6)
    assert _rel(qkv[:, : (hq + hk) * d].view(rows, hq + hk, d), want) < 1e-2

    gate_up = torch.randn(rows, 2 * 1024, device="cuda").to(dtype)
    g, u = gate_up.float().chunk(2, -1)
    assert _rel(silu_and_mul(gate_up), torch.nn.functional.silu(g) * u) < 1e-2

    a = (torch.randn(rows, 4096, device="cuda") * 20).to(dtype)
    b = torch.randn(rows, 4096, device="cuda").to(dtype)
    want = 30.0 * torch.tanh((a.float() + b.float()) / 30.0)
    got = add_softcap(a, b, 30.0)
    assert got.dtype == dtype and _rel(got, want) < 1e-2

    raw = torch.randn(rows, 3 * rank, device="cuda") * 2
    state = torch.rand(rows, rank, device="cuda").to(dtype) * 2 - 1
    gate, cand, out = raw.chunk(3, -1)
    gate = torch.sigmoid(gate)
    got_state, got_latent = markov_gate(raw, state)
    assert got_state.dtype == dtype
    assert _rel(got_state, gate * state.float() + (1 - gate) * torch.tanh(cand)) < 1e-2
    assert _rel(got_latent, torch.tanh(out)) < 1e-2
