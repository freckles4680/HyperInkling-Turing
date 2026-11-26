"""Turing: time the W8 GEMV tiles (`_w8_gemv_kernel`) on the card, at the model's
decode and verify shapes, against the default tile and an fp16 cuBLAS GEMV of the same
weight (the bandwidth to aim for). No server; the GPU must be free.

    venv/bin/python bench/sm75_w8_gemv.py          # ~1-2 minutes; send the table back

Each row: shape, the default tile's time, the best tile found and its time, and the
int8 weight read rate of the best tile (the 2080 Ti peaks near 600 GB/s).
"""

import itertools

import torch

from vllm.models.inkling2.nvidia.ops.linear_w8 import (
    quantize_int8,
    sm75_gemv_tile,
    w8_gemv_sm75,
)

K = 2304
# A decode step has 1 + k rows: k = 2, 4 or 7 verified drafts (start.sh).
MS = (1, 2, 3, 5, 8)
# INT8 (attention, LM head); fp16 weights (router, sconv taps, loop down projections).
SHAPES = [("int8", m, n) for n in (2304, 8600, 201024) for m in MS]
SHAPES += [("fp16", m, n) for n in (256, 2304) for m in MS]


def _time(fn, reps):
    flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")  # L2-cold weights
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    total = 0.0
    for _ in range(reps):
        flush.zero_()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        total += start.elapsed_time(end)
    return total / reps * 1e3  # us


def main():
    torch.manual_seed(0)
    print(f"{'W':>4} {'M':>3} {'N':>7} {'default us':>10} {'best us':>8} {'best tile':>18} "
          f"{'GB/s':>6} {'cuBLAS fp16 us':>14}")
    for kind, m, n in SHAPES:
        q, s = quantize_int8((torch.randn(n, K, device="cuda") * 0.02).half())
        if kind == "fp16":
            q, s = (q.float() * 0.01).half(), None
        x = torch.randn(m, K, device="cuda").half()
        out = torch.empty(m, n, device="cuda", dtype=torch.half)
        reps = 5 if n > 100000 else 30
        default = sm75_gemv_tile(m, n)
        results = {}
        bms = {min(1 << (m - 1).bit_length(), 8), 8}  # the row tile that fits, and 8
        for bm, bk, acc, warps in itertools.product(
                sorted(bms), (16, 32, 64, 128), (256, 512, 1024, 2048, 4096, 8192), (1, 2, 4, 8)):
            bn = acc // (bm * bk)
            if bn < 1 or acc // (32 * warps) > 128:
                continue
            tile = (bm, bn, bk, warps)
            try:
                w8_gemv_sm75(x, q, s, out, tile)
                results[tile] = _time(lambda t=tile: w8_gemv_sm75(x, q, s, out, t), reps)
            except Exception:  # noqa: BLE001 (a tile the compiler or the card refuses)
                pass
        if default not in results:
            w8_gemv_sm75(x, q, s, out, default)
            results[default] = _time(lambda: w8_gemv_sm75(x, q, s, out, default), reps)
        best = min(results, key=results.get)
        w16 = q if s is None else (q.float() * 0).half()  # same shape, fp16
        cublas = _time(lambda: torch.mm(x, w16.t()), reps)
        rate = n * K * (2 if s is None else 1) / results[best] / 1e3
        print(f"{kind:>4} {m:3d} {n:7d} {results[default]:10.1f} {results[best]:8.1f} "
              f"{str(best):>18} {rate:6.0f} {cublas:14.1f}")


if __name__ == "__main__":
    main()
