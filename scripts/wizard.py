#!/usr/bin/env python3
"""HyperInkling launch wizard: a few questions, then the .env (and the command) that serves
Inkling 2 Mini on one RTX 3090 / 3090 Ti (or a 22 GB 2080 Ti) with those choices.

    python3 scripts/wizard.py            # interactive; writes .env next to docker-compose.yml
    python3 scripts/wizard.py --print    # only print, write nothing
    python3 scripts/wizard.py --gpu-gib 22 --compute-cap 7.5   # answer for another GPU

The server fits the longest request to the GPU at boot (MAX_LEN=auto); the context shown
for each choice comes from docs/memory_matrix.tsv, measured on an RTX 3090 Ti with nothing
else on the GPU (bench/memory_matrix.sh), KV cache at 8 bits per value. For a GPU with
other memory (a 22 GB 2080 Ti) the rows measured at that size are used if there are any,
else the 24 GiB ones minus the KV the missing memory would have held.
Standard library only, so it runs wherever Python 3.9+ does (Linux, macOS, Windows).
"""

import argparse
import csv
import math
import os
import secrets
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TABLE = REPO / "docs" / "memory_matrix.tsv"
MAX_CONTEXT = 1_048_576  # the model's own limit
# Per-token KV of a long request (the 8 full-attention layer visits, 8-bit; the DSpark
# drafter adds its own full-attention layer), to turn GiB into tokens.
KIB_PER_TOKEN = {"dspark": 18, "off": 16}
# Decode tok/s of one request, 8k context, RTX 3090 Ti (README).
DECODE_1 = {"dspark": 254, "off": 131}
GPU_UTIL = 0.95  # start.sh's default --gpu-memory-utilization
REF_GPU_GIB = 24  # the size most of docs/memory_matrix.tsv was measured at
# Context the adaptive verify length (its CUDA graphs) and the context lookup take.
ADAPTIVE_COST = 5_120
CELL = 40  # px per image token side (40 px patches, 2x2 merge)
A4_WIDTH_IN = 8.27
# (HF repo, local directory under models/) of each checkpoint.
CHECKPOINTS = {
    "original": ("Pentium95/Inkling-2-Mini-W4A16-g128-gptq", "Inkling-2-Mini-W4A16-g128-gptq"),
    "uncensored": ("Pentium95/Inkling-2-Mini-Uncensored-W4A16-g128-gptq",
                   "Inkling-2-Mini-Uncensored-W4A16-g128-gptq"),
}


def load_table():
    """{gpu_gib: {(vision, spec, async, max_seqs, nbt): longest request}}."""
    tables = {}
    with TABLE.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["max_len_fits"] in ("", "?"):
                continue
            key = (r["vision"], r["spec"], r["async"] == "1", int(r["max_seqs"]),
                   int(r["nbt"]))
            gib = int(r.get("gpu_gib") or REF_GPU_GIB)
            tables.setdefault(gib, {})[key] = int(r["max_len_fits"])
    return tables


def pick_table(tables, gpu_gib):
    """(rows, measured size) for a GPU of `gpu_gib`: its own rows if measured, else the
    reference size's (shifted later by `memory_shift`)."""
    size = round(gpu_gib)
    if size in tables:
        return tables[size], size
    ref = REF_GPU_GIB if REF_GPU_GIB in tables else max(tables)
    return tables[ref], ref


def memory_shift(gpu_gib, measured_gib, spec):
    """Tokens of KV gained (negative: lost) by `gpu_gib` against rows measured at
    `measured_gib`: the KV pool is what --gpu-memory-utilization leaves after weights and
    activations, which do not change with the card."""
    return int((gpu_gib - measured_gib) * GPU_UTIL * 2**20 / KIB_PER_TOKEN[spec])


def gpu_info():
    """(memory GiB, compute capability "7.5") of the first GPU, from nvidia-smi; Nones
    when there is no GPU or no nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
        mib, cap = (v.strip() for v in out.splitlines()[0].split(","))
        return int(mib) / 1024, cap
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None, None


def expected(rows, vision, spec, async_, seqs, nbt):
    """Longest request for a combination: measured, or the 6-request async-on NBT-2048 row
    of this vision tower and spec shifted by what async, NBT and the request count shift
    on the GPU (a shift not measured for `spec` is taken from DSpark's)."""
    if (vision, spec, async_, seqs, nbt) in rows:
        return rows[(vision, spec, async_, seqs, nbt)]

    def row(v, s, a=True, q=6, n=2048):
        return rows.get((v, s, a, q, n))

    def shift(change):  # change(spec) -> (from, to) row values, or None
        for s in (spec, "dspark"):
            a, b = change(s)
            if a is not None and b is not None:
                return b - a
        return 0

    base = row(vision, spec)
    if base is None:  # this tower measured with the other spec only
        other = "off" if spec == "dspark" else "dspark"
        if None in (row("gpu", spec), row(vision, other), row("gpu", other)):
            return None
        base = row("gpu", spec) + row(vision, other) - row("gpu", other)
    if not async_:
        base += shift(lambda s: (row("gpu", s), row("gpu", s, False)))
    if nbt != 2048:
        base += shift(lambda s: (row("gpu", s, async_), row("gpu", s, async_, n=nbt))) or \
            shift(lambda s: (row("gpu", s), row("gpu", s, n=nbt)))
    if seqs != 6:
        # Piecewise-linear in the request count over the measured gpu series.
        series = sorted((q, v) for (vi, s, a, q, n), v in rows.items()
                        if (vi, s, a, n) == ("gpu", "dspark", True, 2048))
        lo = max((p for p in series if p[0] <= seqs), default=series[0])
        hi = min((p for p in series if p[0] >= seqs), default=series[-1])
        ref = lo[1] if hi[0] == lo[0] else (
            lo[1] + (hi[1] - lo[1]) * (seqs - lo[0]) / (hi[0] - lo[0]))
        base += int(ref - (row("gpu", "dspark") or ref))
    return max(0, min(MAX_CONTEXT, base))


def image_size(tokens):
    """The largest A4-shaped image of at most `tokens` tokens: (width, height) px."""
    w = max(1, math.floor(math.sqrt(tokens / math.sqrt(2))))
    return w * CELL, (tokens // w) * CELL


def ask(prompt, choices=None, default=None, cast=str, check=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            raw = str(default)
        try:
            value = cast(raw)
        except ValueError:
            print("  not a valid answer")
            continue
        if choices and str(value).lower() not in choices:
            print(f"  choose one of: {', '.join(choices)}")
            continue
        if check and not check(value):
            continue
        return str(value).lower() if choices else value


def ram_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def fmt_tokens(n):
    return f"{n / 1000:.0f}k" if n < 1_000_000 else f"{n / 1_048_576:.2f}M"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--print", action="store_true", help="print only, do not write .env")
    ap.add_argument("--gpu-gib", type=float, help="GPU memory in GiB (default: nvidia-smi)")
    ap.add_argument("--compute-cap", help="GPU compute capability, e.g. 7.5 (default: "
                    "nvidia-smi)")
    args = ap.parse_args()
    tables = load_table()
    print(__doc__.split("\n\n")[0], "\n")

    found_gib, found_cap = gpu_info()
    cap = args.compute_cap or found_cap
    turing = cap == "7.5"
    if args.gpu_gib:
        gpu_gib = args.gpu_gib
    else:
        default = round(found_gib, 1) if found_gib else REF_GPU_GIB
        print("GPU memory" + (f" (nvidia-smi: {found_gib:.1f} GiB)" if found_gib else
                              ": no nvidia-smi here; a 22 GB 2080 Ti is 22"))
        gpu_gib = ask("   GiB", default=default, cast=float,
                      check=lambda v: 8 <= v <= 96 or print("  8 to 96"))
    rows, measured_gib = pick_table(tables, gpu_gib)
    if turing:
        print("\nTuring (sm75) runs the model in fp16 (no bf16 tensor cores): ported but not "
              "yet run on a Turing GPU.\nDSpark is not ported to fp16, so answer no to it. "
              "Speeds below are 3090 Ti numbers.")
    if round(gpu_gib) != measured_gib:
        print(f"\nContext below: measured at {measured_gib} GiB, scaled to {gpu_gib:g} GiB.")
    print()

    def fit(vision, spec, *a):
        n = expected(rows, vision, spec, *a)
        if n is None:
            return None
        return max(0, min(MAX_CONTEXT, n + memory_shift(gpu_gib, measured_gib, spec)))

    def ctx(*a):
        n = fit(*a)
        return fmt_tokens(n) if n else "?"

    print("0) Which checkpoint?")
    print("   a) Inkling 2 Mini as released by Thinking Machines")
    print("   b) Inkling 2 Mini Uncensored (OpenUncensored's abliteration: refusals removed, only")
    print("      the attention output projections edited), quantized the same way")
    checkpoint = "uncensored" if ask("   choice", "ab", "a") == "b" else "original"

    seqs = ask("\n1) How many requests should run at the same time? (1-64)", default=8,
               cast=int, check=lambda v: 1 <= v <= 64 or print("  1 to 64"))

    print("\n2) Where should the vision tower (images) run?")
    towers = (("a", "gpu", "on the GPU: fastest image encoding"),
              ("b", "host", "weights in host RAM, streamed per image: ~27 ms more per A4 page"),
              ("c", "none", "no vision: text only"))
    for key, vision, note in towers:
        print(f"   {key}) {note}  (longest request: "
              f"{ctx(vision, 'dspark', True, seqs, 2048)})")
    vision = {k: v for k, v, _ in towers}[ask("   choice", "abc", "a")]

    print("\n3) Speculative decoding with the bundled DSpark drafter?")
    print(f"   y) yes: one request decodes ~{DECODE_1['dspark']} tok/s instead of "
          f"~{DECODE_1['off']} (more on code and structured output, less on free prose); "
          f"longest request {ctx(vision, 'dspark', True, seqs, 2048)}")
    print(f"   n) no: longest request {ctx(vision, 'off', True, seqs, 2048)}")
    spec = "dspark" if ask("   choice", "yn", "n" if turing else "y") == "y" else "off"
    if turing and spec == "dspark":
        print("   DSpark is not ported to fp16: start.sh refuses SPEC=dspark on Turing.")
        spec = "off"

    adaptive = "n"
    if spec == "dspark":
        print("\n3a) Adaptive verify length and drafts from the context: +43% while the "
              "answer copies the prompt, +4-10% on free text and code, for "
              f"~{fmt_tokens(ADAPTIVE_COST)} of context.")
        adaptive = ask("   enable", "yn", "y")

    width = height = None
    if vision != "none":
        print("\n4) Largest image, in tokens (larger images are scaled down to it). 1024 "
              "tokens is an A4 page at ~126 ppi; the prefill chunk of 2048 encodes two per "
              "step.")
        for t in (1024, 1364, 2048, 5488):
            w, h = image_size(t)
            print(f"   {t:5d} tokens = {w * h / 1e6:.1f} MP, A4 at ~{w / A4_WIDTH_IN:.0f} ppi"
                  f" ({w} x {h})")
        tokens = ask("   tokens", default=1024, cast=int,
                     check=lambda v: 256 <= v <= 16384 or print("  256 to 16384"))
        width, height = image_size(tokens)

    nbt_gain = (fit(vision, spec, True, seqs, 1024) or 0) - (
        fit(vision, spec, True, seqs, 2048) or 0)
    print(f"\n5) Prefill chunk (NBT): 2048, or 1024 for ~{fmt_tokens(max(0, nbt_gain))} more "
          "context at ~8% slower prefill (same decode).")
    nbt = int(ask("   chunk", ("2048", "1024"), "2048"))

    on = fit(vision, spec, True, seqs, nbt)
    off = fit(vision, spec, False, seqs, nbt)
    gain = f"~{fmt_tokens(off - on)} more context" if on and off else "more context"
    print(f"\n6) Async scheduling (two steps in flight): +3-4% decode; off: {gain}.")
    async_ = ask("   enable", "yn", "y") == "y"

    headroom = ask("\n7) GiB of GPU memory to leave free (a display on this GPU: 1)",
                   default=0.0, cast=float, check=lambda v: 0 <= v <= 4)

    longest = fit(vision, spec, async_, seqs, nbt)
    if longest == 0:
        print(f"\n   Nothing fits: at {gpu_gib:g} GiB the weights and activations leave no "
              "KV cache with these choices.")
    if longest:
        if adaptive == "n" and spec == "dspark":
            longest += ADAPTIVE_COST
        longest -= int(headroom * 2**20 / KIB_PER_TOKEN[spec])
    print(f"\n8) Longest request: auto fits it to the GPU at boot (expected "
          f"~{fmt_tokens(longest) if longest else '?'}); a number caps it (prompts above it "
          "are refused; a cap that does not fit stops the boot).")

    def ok(v):
        if v == "auto":
            return True
        if not 1024 <= v <= MAX_CONTEXT:
            return print(f"  1024 to {MAX_CONTEXT}, or auto") or False
        if longest and v > longest:
            return print(f"  {fmt_tokens(v)} is above the expected ~{fmt_tokens(longest)}"
                         ) or False
        return True

    max_len = ask("   tokens or auto", default="auto",
                  cast=lambda v: v.lower() if v.lower() == "auto" else int(v), check=ok)

    ram = ram_gib()
    auto = "unknown" if ram is None else f"{int(max(0, min(36, ram * 0.6 - 8)))} GiB"
    print(f"\n9) Prefix-cache tier in host RAM (/dev/shm): prompts evicted from the GPU come "
          f"back in 0.1 s instead of being recomputed. RAM: "
          f"{'unknown' if ram is None else f'{ram:.0f} GiB'}.")
    tier = ask(f"   GiB, 0 = off, auto = 60% of RAM minus 8, at most 36 (here {auto})",
               default="auto", cast=lambda v: v if v.lower() == "auto" else int(v),
               check=lambda v: v == "auto" or v >= 0 or print("  0 or more, or auto"))

    expose = ask("\n10) Reachable from other machines? (needs an API key)", "yn", "n")
    env = {
        "MAX_SEQS": seqs,
        "MAX_LEN": max_len,
        "SPEC": spec,
        "ASYNC": int(async_),
        "NBT": nbt,
        "KV_OFFLOAD_GB": tier,
        "GPU_UTIL": round(GPU_UTIL - headroom / gpu_gib, 3),
    }
    repo, local = CHECKPOINTS[checkpoint]
    if checkpoint != "original":  # compose downloads HF_REPO into models/CHECKPOINT
        env["HF_REPO"], env["CHECKPOINT"], env["MODEL"] = repo, local, f"models/{local}"
    if width:
        env["IMAGE_MAX_W"], env["IMAGE_MAX_H"] = width, height
        if (width // CELL) * (height // CELL) > 2048:
            env["ENCODER_CACHE"] = 4 * (width // CELL) * (height // CELL)
    if vision == "none":
        env["IMAGES"] = 0
    if vision == "host":
        env["HYPERINKLING_VISION_HOST"] = 1
    if spec == "dspark" and adaptive == "n":
        env["HYPERINKLING_SPEC_ADAPTIVE"] = ""
        env["HYPERINKLING_LOOKUP"] = 0
    if tier != "auto" and tier > 36:  # compose's /dev/shm default is 48
        env["SHM_GB"] = tier + 12
    if hasattr(os, "getuid"):  # compose's prepare writes ./models as this user
        env["UID"], env["GID"] = os.getuid(), os.getgid()
    if expose == "y":
        env["BIND"] = "0.0.0.0"  # Docker: the published port
        env["HOST"] = "0.0.0.0"  # native start.sh
        env["VLLM_API_KEY"] = secrets.token_hex(24)

    text = "".join(f"{k}={v}\n" for k, v in env.items())
    print("\n--- .env " + "-" * 50 + f"\n{text}" + "-" * 59)
    if seqs > 1 and longest:
        print(f"{seqs} long requests at once share the KV pool, about "
              f"{fmt_tokens(longest / seqs)} each; shorter ones fit more.")
    if not args.print:
        dest = REPO / ".env"
        if dest.exists() and ask(f"\n{dest} exists. Overwrite?", "yn", "n") != "y":
            print("Not written.")
        else:
            dest.write_text(text)
            # Docker would create a missing bind-mount source as root.
            (REPO / "models").mkdir(exist_ok=True)
            print(f"Written {dest}")
    print("\nStart it:  docker compose up -d     (logs: docker compose logs -f; the first start"
          " downloads the checkpoint)")
    print(f"Native:    hf download {repo} --local-dir models/{local}   (once)")
    exports = " ".join(f"{k}={v}" for k, v in env.items()
                       if k not in ("UID", "GID", "SHM_GB", "BIND", "HF_REPO", "CHECKPOINT"))
    print(f"Native:    {exports} ./start.sh")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)
