"""Turing: the INT4 decode GEMV (`ops/linear_w4_gemv.py`, HYPERINKLING_ATTN_W4 /
HYPERINKLING_HEAD_W4) against the INT8 one it would replace, at the model's decode and
verify shapes. No server; the GPU must be free.

    venv/bin/python bench/sm75_w4_gemv.py            # ~3-5 minutes; send the output back
    QUICK=1 venv/bin/python bench/sm75_w4_gemv.py    # default tiles only, no search (~30 s)

Each row: shape and group, the INT8 GEMV at its tuned tile, the INT4 GEMV at its default
tile, the best INT4 tile found and its time, the INT4 weight read rate, and the speedup of
the best INT4 tile over INT8. Weights are L2-cold, as in a decode step. The last block
adds up one decode step's worth of these calls, so the step-time gain can be estimated
before booting a server.
"""

import itertools
import os

import torch

from vllm.models.inkling2.nvidia.ops.linear_w4_gemv import (
    quantize_w4g,
    sm75_w4_tile,
    w4g_gemv,
)
from vllm.models.inkling2.nvidia.ops.linear_w8 import (
    quantize_int8,
    sm75_gemv_tile,
    w8_gemv_sm75,
)

K = 2304
MS = (1, 3, 5, 8)  # 1 + k rows, k = 2, 4 or 7 verified drafts
NS = (2304, 8600, 201024)  # wo, qkvr, LM head
GROUPS = (32, 64, 128)
QUICK = os.environ.get("QUICK") == "1"


def _time(fn, reps):
    flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")  # L2-cold weights
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    fn()
    torch.cuda.synchronize()
    total = 0.0
    for _ in range(reps):
        flush.zero_()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        total += start.elapsed_time(end)
    return total / reps * 1e3  # us


def _tiles(m):
    bms = sorted({min(1 << (m - 1).bit_length(), 8), 8} if m > 1 else {1, 2})
    for bm, bn, warps in itertools.product(bms, (1, 2, 4, 8, 16, 32, 64), (1, 2, 4, 8)):
        acc = bm * bn * 16  # fp32 accumulators per block (twice that live)
        if acc // (32 * warps) > 64 or acc < 32 * warps // 4:
            continue
        yield bm, bn, warps


def main():
    torch.manual_seed(0)
    print(torch.cuda.get_device_name(), f"K {K}")
    print(f"{'M':>3} {'N':>7} {'G':>4} {'INT8 us':>8} {'INT4 us':>8} {'best us':>8} "
          f"{'best tile':>12} {'GB/s':>5} {'x INT8':>7}")
    best_us = {}
    for n in NS:
        w = (torch.randn(n, K, device="cuda") * 0.02).half()
        q8, s8 = quantize_int8(w)
        quant = {g: quantize_w4g(w, g, clip=False) for g in GROUPS}
        del w
        for m in MS:
            x = torch.randn(m, K, device="cuda").half()
            out = torch.empty(m, n, device="cuda", dtype=torch.half)
            reps = 5 if n > 100000 else 30
            t8 = _time(lambda: w8_gemv_sm75(x, q8, s8, out, sm75_gemv_tile(m, n)), reps)
            for g in GROUPS:
                p, s, z = quant[g]
                default = sm75_w4_tile(m, n)
                results = {default: _time(lambda: w4g_gemv(x, p, s, z, out, default), reps)}
                if not QUICK:
                    for tile in _tiles(m):
                        if tile in results:
                            continue
                        try:
                            results[tile] = _time(
                                lambda t=tile: w4g_gemv(x, p, s, z, out, t), reps)
                        except Exception:  # noqa: BLE001 (a tile the card refuses)
                            pass
                best = min(results, key=results.get)
                best_us[(m, n, g)] = (results[best], t8)
                bytes_w4 = p.numel() * 4 + s.numel() * 3
                print(f"{m:3d} {n:7d} {g:4d} {t8:8.1f} {results[default]:8.1f} "
                      f"{results[best]:8.1f} {str(best):>12} "
                      f"{bytes_w4 / results[best] / 1e3:5.0f} {t8 / results[best]:7.2f}",
                      flush=True)
        torch.cuda.empty_cache()
    # One decode step's projections: 48 attention layer visits (24 layers, 6-17 three
    # times; the local layers' shapes are taken as the full ones') and one LM head call,
    # at 1 + k rows (the target's verify).
    visits = int(os.environ.get("VISITS", "48"))
    print(f"\nper decode step, {visits} attention layer visits + 1 LM head (VISITS= to change):")
    for m in MS:
        for g in GROUPS:
            a4 = visits * (best_us[(m, 8600, g)][0] + best_us[(m, 2304, g)][0])
            a8 = visits * (best_us[(m, 8600, g)][1] + best_us[(m, 2304, g)][1])
            h4, h8 = best_us[(m, 201024, g)]
            print(f"  M {m} g{g}: attention INT8 {a8 / 1e3:5.2f} ms -> INT4 {a4 / 1e3:5.2f} ms, "
                  f"head {h8 / 1e3:5.2f} -> {h4 / 1e3:5.2f} ms")


if __name__ == "__main__":
    main()
