"""The compiled vLLM ops run on this GPU and give right results: one op of each library
(`_C`: rms_norm, `_moe_C`: topk_softmax) against torch. Libraries built without this GPU's
architecture fail to launch, or (moe_align_block_size on stock sm100 builds) leave their
outputs unwritten: both show here. Run by scripts/build-image.sh inside the image.

    python3 scripts/verify-image.py
"""

import sys

import torch

from vllm import _custom_ops as ops


def main():
    cap = torch.cuda.get_device_capability()
    torch.manual_seed(0)
    # Turing (sm75) has no bf16 kernels in vLLM: check the fp16 build there.
    dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16
    x = torch.randn(64, 2304, device="cuda", dtype=dtype)
    w = torch.randn(2304, device="cuda", dtype=dtype)
    out = torch.full_like(x, float("nan"))
    ops.rms_norm(out, x, w, 1e-6)
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
           * w.float())
    ok_norm = torch.allclose(out.float(), ref, atol=5e-2, rtol=2e-2)

    logits = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    weights = torch.full((64, 8), float("nan"), device="cuda")
    ids = torch.full((64, 8), -1, device="cuda", dtype=torch.int32)
    rows = torch.empty((64, 8), device="cuda", dtype=torch.int32)
    ops.topk_softmax(weights, ids, rows, logits)
    want_w, want_ids = logits.softmax(-1).topk(8, dim=-1)
    ok_topk = bool((ids.long().sort(-1).values == want_ids.sort(-1).values).all()) and \
        torch.allclose(weights.sort(-1).values, want_w.sort(-1).values, atol=1e-5)

    print(f"sm{cap[0]}{cap[1]}: rms_norm {'ok' if ok_norm else 'WRONG'}, "
          f"topk_softmax {'ok' if ok_topk else 'WRONG'}")
    sys.exit(0 if ok_norm and ok_topk else 1)


if __name__ == "__main__":
    main()
