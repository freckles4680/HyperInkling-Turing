"""Prefill and decode at fixed context lengths, 1 and N requests at once, on a running server.

    venv/bin/python bench/context_bench.py run <label> [--port 18030] [--contexts 8192 32768]
    venv/bin/python bench/context_bench.py table <label> [<label> ...]

Per context and concurrency, two phases on the same prompts (AA-LCR documents, a unique nonce
first, cut to the exact token count with /tokenize):
- prefill: the prompts never seen, sent at once (max_tokens 1). Per request: prompt tokens /
  its time to first token; aggregate: all prompt tokens / the last first token.
- decode: the same prompts again, now prefix-cache hits, so every request starts generating
  together; 512 tokens each (min_tokens = max_tokens, temperature 0). Measured only in the
  window where all of them generate (from the last first token to the first finish), from
  the streamed usage: no prefill inside. With the tokens accepted per engine step and the
  step time (/metrics): compare step times between runs, as acceptance follows the text.
Each shape runs once unmeasured first (kernels compile on first use). Results:
.local/context_bench/<label>.json.
"""

import argparse
import json
import random
import statistics
import sysconfig
import threading
import time
import urllib.request
from pathlib import Path

OUT = Path(".local/context_bench")
SRC = Path.home() / ".cache/evalscope/aa_lcr/lcr"
INSTRUCTION = ("\n\n---\nWrite a detailed summary of the document above in Italian, "
               "section by section.")
MODEL = "inkling-2-mini"


def _post(url, body, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


class Prompts:
    def __init__(self, base):
        self.base = base
        self.text = "\n\n".join(f.read_text(errors="ignore") for f in sorted(SRC.rglob("*.txt")))
        if not self.text:
            # No AA-LCR download: Python's own library (code accepts more drafts). Not
            # vLLM's sources, which every patch changes, and the prompts with them.
            lib = Path(sysconfig.get_paths()["stdlib"])
            print(f"no documents under {SRC}: prompts from {lib}/*.py instead")
            files = sorted(lib.glob("*.py"))
            self.text = "\n\n".join(f.read_text(errors="ignore") for f in files)

    def count(self, content):
        body = {"model": MODEL, "messages": [{"role": "user", "content": content}]}
        return json.load(_post(f"{self.base}/tokenize", body))["count"]

    def make(self, tokens, seed):
        """A user message whose chat prompt is `tokens` tokens (+- 8)."""
        rng = random.Random(seed)
        start = rng.randrange(0, len(self.text) // 2)
        head = f"[{seed:08x}]\n"
        chars = tokens * 4
        for _ in range(8):
            content = head + self.text[start:start + chars] + INSTRUCTION
            n = self.count(content)
            if abs(n - tokens) <= 8:
                break
            chars = int(chars * tokens / n) if n > tokens else chars + (tokens - n) * 3
        return content, n


def _stream(base, content, max_tokens, rec):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": max_tokens, "min_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    t0 = time.perf_counter()
    rec["points"] = []  # (seconds, completion tokens so far)
    for line in _post(f"{base}/v1/chat/completions", body):
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        usage = json.loads(line[6:]).get("usage")
        if usage:
            rec["prompt"] = usage["prompt_tokens"]
            done = usage.get("completion_tokens") or 0
            if done and (not rec["points"] or done > rec["points"][-1][1]):
                rec["points"].append((time.perf_counter() - t0, done))


def _at_once(base, contents, max_tokens):
    recs = [{} for _ in contents]
    threads = [threading.Thread(target=_stream, args=(base, c, max_tokens, r))
               for c, r in zip(contents, recs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return recs


def _steps(base):
    """Engine steps and generated tokens so far (/metrics)."""
    text = urllib.request.urlopen(f"{base}/metrics", timeout=30).read().decode()

    def total(name):
        return sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
                   if line.startswith(name + "{"))

    return total("vllm:iteration_tokens_total_count"), total("vllm:generation_tokens_total")


def _tokens_at(points, t):
    return max([n for s, n in points if s <= t], default=0)


def measure(base, prompts, ctx, conc, seed, gen):
    contents = [prompts.make(ctx, seed + i)[0] for i in range(conc)]
    pre = _at_once(base, contents, 1)
    ttft = [r["points"][0][0] for r in pre]
    prompt_tokens = [r["prompt"] for r in pre]
    steps0, tokens0 = _steps(base)
    dec = _at_once(base, contents, gen)
    steps1, tokens1 = _steps(base)
    per_step = (tokens1 - tokens0) / max(steps1 - steps0, 1)
    start = max(r["points"][0][0] for r in dec)
    end = min(r["points"][-1][0] for r in dec)
    assert end > start, "no window where every request decodes"
    rates = [(_tokens_at(r["points"], end) - _tokens_at(r["points"], start)) / (end - start)
             for r in dec]
    return {
        "context": statistics.mean(prompt_tokens),
        "prefill_per_request": statistics.mean(p / t for p, t in zip(prompt_tokens, ttft)),
        "prefill_aggregate": sum(prompt_tokens) / max(ttft),
        "decode_per_request": statistics.mean(rates),
        "decode_aggregate": sum(rates),
        # What a code change moves is the step time; the tokens a step accepts follow the
        # text (and DSpark's generation does not repeat), and tok/s is their ratio.
        "tokens_per_step": per_step,
        "ms_per_step": per_step / sum(rates) * 1e3,
    }


def run(args):
    base = f"http://127.0.0.1:{args.port}"
    prompts = Prompts(base)
    results = {}
    for ctx in args.contexts:
        for conc in args.conc:
            # The same documents every run (DSpark's acceptance depends on the text); a
            # server restart and no CPU tier keep runs from hitting each other's prefixes.
            seed = ctx * 10 + conc
            try:
                measure(base, prompts, ctx, conc, seed + 1000, args.gen)  # warm-up, unmeasured
                r = measure(base, prompts, ctx, conc, seed, args.gen)
            except Exception as e:  # noqa: BLE001 (a context the server refuses: keep the rest)
                detail = e.read().decode(errors="ignore")[:300] if hasattr(e, "read") else ""
                print(f"{ctx:6d} tokens x{conc}: failed: {e} {detail}", flush=True)
                continue
            results[f"{ctx}x{conc}"] = r
            print(f"{ctx:6d} tokens x{conc}: " + ", ".join(
                f"{k} {v:,.2f}" if "step" in k else f"{k} {v:,.0f}" for k, v in r.items()),
                flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.label}.json").write_text(json.dumps(results, indent=1))


ROWS = [("prefill_per_request", "prefill, per request"),
        ("prefill_aggregate", "prefill, aggregate"),
        ("decode_per_request", "decode, per request (no prefill)"),
        ("decode_aggregate", "decode, aggregate (no prefill)"),
        ("tokens_per_step", "decode, tokens per step"),
        ("ms_per_step", "decode, ms per step")]


def table(args):
    """Markdown: a row per metric and context, a column per label x concurrency."""
    data = {lb: json.loads((OUT / f"{lb}.json").read_text()) for lb in args.labels}
    keys = sorted({k for d in data.values() for k in d},
                  key=lambda k: tuple(map(int, k.split("x"))))
    concs = sorted({int(k.split("x")[1]) for k in keys})
    ctxs = sorted({int(k.split("x")[0]) for k in keys})
    cols = [(c, lb) for c in concs for lb in args.labels]
    def spec(lb):
        return {"off": "no DSpark", "dspark": "DSpark"}.get(lb.rsplit("-", 1)[-1], lb)

    head = [f"{c} request{'s' if c > 1 else ''}, {spec(lb)}" for c, lb in cols]
    print("| | " + " | ".join(head) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for ctx in ctxs:
        for key, name in ROWS:
            cells = []
            for c, lb in cols:
                r = data[lb].get(f"{ctx}x{c}")
                if not r or key not in r:
                    cells.append("")
                elif "step" in key:
                    cells.append(f"{r[key]:.2f}")
                else:
                    cells.append(f"{r[key]:,.0f} tok/s")
            print(f"| {name}, {ctx // 1024}k context | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("label")
    r.add_argument("--port", default="18030")
    r.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
    r.add_argument("--conc", type=int, nargs="+", default=[1, 6])
    r.add_argument("--gen", type=int, default=512)
    t = sub.add_parser("table")
    t.add_argument("labels", nargs="+")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else table(a)


if __name__ == "__main__":
    main()
