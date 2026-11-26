"""A light quality battery against a running server, on public data.

    gsm8k   grade-school math (openai/gsm8k test): accuracy of the last number
    tools   tool calls (NousResearch/hermes-function-calling-v1, single turn, records not
            in the calibration corpus): exact match of every call's name and arguments
    docvqa  questions on document pages (pixparse/docvqa-single-page-questions validation),
            answered as strict JSON: ANLS, and the share of replies that are valid JSON

    venv/bin/python bench/quality_battery.py download             # ~0.4 GB into bench/quality-data
    venv/bin/python bench/quality_battery.py run [tasks] [--port 8080] [--n 200] [--jobs 6]

Greedy (temperature 0) and reasoning at the template default, so two servers are
comparable; batching still moves a few answers (about +-1 point at n=200). Results are
printed and saved to bench/quality-data/result-<label>.json; every answer goes to
bench/quality-data/answers-<label>.jsonl as it arrives (tail -f it to follow a run).
"""

import argparse
import base64
import json
import math
import re
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(__file__).resolve().parent / "quality-data"
FILES = {
    "openai/gsm8k": ["main/test-00000-of-00001.parquet"],
    "NousResearch/hermes-function-calling-v1": ["func-calling-singleturn.json"],
    "pixparse/docvqa-single-page-questions": ["data/validation-00000-of-00004.parquet"],
}
# Hermes records whose prompt is in the calibration corpus (prepare/build_calib.py).
HERMES_STEP, HERMES_SKIP = 9, {612, 837, 1008, 1035, 1053}
MAX_TOKENS = 8192


def download():
    for repo, files in FILES.items():
        dest = DATA / repo.split("/")[1]
        hf = Path(sys.executable).with_name("hf")  # the venv's, else the one on PATH
        cmd = [str(hf) if hf.exists() else "hf", "download", repo, *files, "--repo-type", "dataset", "--local-dir", str(dest)]
        subprocess.run(cmd, check=True)


def _post(port, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps({"model": "inkling-2-mini", "temperature": 0, "max_tokens": MAX_TOKENS,
                    **body}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["choices"][0]["message"]


# --- gsm8k ---------------------------------------------------------------------------

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _number(text):
    nums = NUM_RE.findall((text or "").replace("$", ""))
    return float(nums[-1].replace(",", "")) if nums else None


def gsm8k_items(n):
    import pyarrow.parquet as pq

    t = pq.read_table(DATA / "gsm8k/main/test-00000-of-00001.parquet").to_pylist()[:n]
    return [(r["question"], float(r["answer"].split("####")[-1].strip().replace(",", "")))
            for r in t]


def gsm8k_one(port, item):
    q, gold = item
    msg = _post(port, {"messages": [{"role": "user", "content": q + "\n\nEnd with the final "
                                     "answer as a plain number on its own line."}]})
    got = _number(msg.get("content"))
    return {"ok": got is not None and math.isclose(got, gold, abs_tol=1e-6),
            "question": q, "gold": gold, "answer": msg.get("content")}


# --- tools ---------------------------------------------------------------------------

TOOLS_RE = re.compile(r"<tools>\n(.*?)\n</tools>", re.S)
CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _norm(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return {k: _norm(x) for k, x in v.items()}


def _call_key(name, args):
    return json.dumps([name, _norm(args)], sort_keys=True)


def tools_items(n):
    ex = json.load(open(DATA / "hermes-function-calling-v1/func-calling-singleturn.json"))
    out = []
    for i in range(0, len(ex), HERMES_STEP):
        conv = ex[i]["conversations"]
        if i in HERMES_SKIP or not (m := TOOLS_RE.search(conv[0]["value"])):
            continue
        try:
            tools = json.loads(m.group(1))
            gold = sorted(_call_key(c["name"], c["arguments"])
                          for c in map(json.loads, CALL_RE.findall(conv[-1]["value"])))
        except (json.JSONDecodeError, KeyError):
            continue
        user = next(c["value"] for c in conv if c["from"] == "human")
        out.append((user, tools, gold))
    return out[:n]


def tools_one(port, item):
    user, tools, gold = item
    msg = _post(port, {"messages": [{"role": "user", "content": user}], "tools": tools})
    calls = []
    for c in msg.get("tool_calls") or []:
        try:
            calls.append(_call_key(c["function"]["name"], json.loads(c["function"]["arguments"])))
        except json.JSONDecodeError:
            calls.append("invalid")
    names = sorted(json.loads(k)[0] for k in gold)
    got_names = sorted(json.loads(k)[0] if k != "invalid" else k for k in calls)
    return {"ok": sorted(calls) == gold, "names": got_names == names,
            "question": user, "gold": gold, "answer": sorted(calls) or msg.get("content")}


# --- docvqa --------------------------------------------------------------------------

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
          "required": ["answer"], "additionalProperties": False}


def _levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _anls(pred, golds, tau=0.5):
    p = " ".join(pred.lower().split())
    best = 0.0
    for g in golds:
        g = " ".join(g.lower().split())
        nl = _levenshtein(p, g) / max(len(p), len(g), 1)
        best = max(best, 1 - nl if nl < tau else 0.0)
    return best


def docvqa_items(n):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(DATA / "docvqa-single-page-questions/data/validation-00000-of-00004.parquet")
    out = []
    for batch in pf.iter_batches(batch_size=64, columns=["image", "question", "answers"]):
        for r in batch.to_pylist():
            url = "data:image/png;base64," + base64.b64encode(r["image"]["bytes"]).decode()
            out.append((url, r["question"], r["answers"]))
            if len(out) == n:
                return out
    return out


def docvqa_one(port, item):
    url, q, golds = item
    msg = _post(port, {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": q + "\nAnswer with the exact words from the document."}]}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "answer", "schema": SCHEMA, "strict": True}},
    })
    try:
        ans = json.loads(msg.get("content") or "")["answer"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"ok": 0.0, "json": False, "question": q, "gold": list(golds),
                "answer": msg.get("content")}
    return {"ok": _anls(str(ans), golds), "json": True, "question": q, "gold": list(golds),
            "answer": ans}


TASKS = {"gsm8k": (gsm8k_items, gsm8k_one), "tools": (tools_items, tools_one),
         "docvqa": (docvqa_items, docvqa_one)}


def run(args):
    report = {}
    # Every answer as it arrives, to follow a run: tail -f bench/quality-data/answers-<label>.jsonl
    answers = (DATA / f"answers-{args.label}.jsonl").open("w")
    lock = threading.Lock()

    def logged(name, one, item):
        r = one(args.port, item)
        with lock:
            answers.write(json.dumps({"task": name, **r}, ensure_ascii=False, default=str) + "\n")
            answers.flush()
        return r

    for name in args.tasks:
        items_fn, one = TASKS[name]
        items = items_fn(args.n)
        with ThreadPoolExecutor(args.jobs) as ex:
            res = list(ex.map(lambda it, name=name, one=one: logged(name, one, it), items))
        n = len(res)
        score = sum(float(r["ok"]) for r in res) / n
        stderr = math.sqrt(max(score * (1 - score), 1e-9) / n)
        line = {"n": n, "score": round(score, 4), "stderr": round(stderr, 4)}
        for extra in ("names", "json"):
            if extra in res[0]:
                line[extra] = round(sum(r[extra] for r in res) / n, 4)
        report[name] = line
        print(f"{name:7s} {score:.3f} +- {stderr:.3f} (n={n})"
              + "".join(f"  {k} {line[k]:.3f}" for k in ("names", "json") if k in line),
              flush=True)
    out = DATA / f"result-{args.label}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    r = sub.add_parser("run")
    r.add_argument("tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    r.add_argument("--port", default="8080")
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--jobs", type=int, default=6)
    r.add_argument("--label", default="run")
    args = ap.parse_args()
    if args.cmd == "download":
        download()
    else:
        run(args)


if __name__ == "__main__":
    sys.exit(main())
