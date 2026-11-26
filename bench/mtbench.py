"""MT-Bench-style judged chat against a running server: 80 two-turn questions in 8
categories (writing, roleplay, reasoning, math, coding, extraction, stem, humanities),
answered by the model, then scored 1 to 10 per turn by a separate judge model.

    python bench/mtbench.py run   --label <name> [--port 8080] [--limit 10] [--jobs 2]
    python bench/mtbench.py judge --label <name>          # judge saved answers again (or later)

Questions and reference answers (for math, reasoning and coding) are lm-sys FastChat's MT-Bench
files (Apache-2.0), downloaded once into bench/quality-data/mt_bench/. The judge prompts follow
FastChat's "single answer grading" prompts; the reference answers are FastChat's GPT-4 ones.

The judge is any OpenAI-compatible chat endpoint, set by environment variables:

    JUDGE_MODEL      model name at the judge endpoint (empty = no judging: answers only)
    JUDGE_BASE_URL   default https://api.openai.com/v1 (OpenRouter, a llama.cpp or vLLM server
                     on another machine, ... anything that serves /v1/chat/completions)
    JUDGE_API_KEY    sent as "Authorization: Bearer <key>" (empty = no header)

Without JUDGE_MODEL the answers are still generated and saved, the summary says loudly that
judging was SKIPPED, and `judge` can score them later. Scores from different judges are not
comparable: compare two runs with the same judge only.

Files (under bench/quality-data/mt_bench/):
    answers-<label>.jsonl     one line per question, both turns (tail -f it to follow a run)
    judgments-<label>.jsonl   one line per judged turn, the judge's full explanation included
    result-<label>.json       the summary that is printed at the end
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(__file__).resolve().parent / "quality-data" / "mt_bench"
FASTCHAT = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench"
FILES = {"question.jsonl": f"{FASTCHAT}/question.jsonl",
         "reference-gpt-4.jsonl": f"{FASTCHAT}/reference_answer/gpt-4.jsonl"}
CATEGORIES = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]
# FastChat's sampling temperature per category (creative text samples, the rest is ~greedy).
TEMPERATURE = {"writing": 0.7, "roleplay": 0.7, "extraction": 0.0, "math": 0.0, "coding": 0.0,
               "reasoning": 0.0, "stem": 0.1, "humanities": 0.1}
NEED_REF = {"math", "reasoning", "coding"}

# --- judge prompts (FastChat llm_judge/data/judge_prompts.jsonl, single-answer grading) ------
_RATE = ('After providing your explanation, you must rate the response on a scale of 1 to 10 by '
         'strictly following this format: "[[rating]]", for example: "Rating: [[5]]".')
_GENERAL = ("Please act as an impartial judge and evaluate the quality of the response provided by "
            "an AI assistant to the user question displayed below. Your evaluation should consider "
            "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of "
            "detail of the response.")
_MATH = ("Please act as an impartial judge and evaluate the quality of the response provided by an "
         "AI assistant to the user question displayed below. Your evaluation should consider "
         "correctness and helpfulness. You will be given a reference answer and the assistant's "
         "answer. Begin your evaluation by comparing the assistant's answer with the reference "
         "answer. Identify and correct any mistakes.")


def judge_messages(q, answers, turn, ref):
    """The judge's chat for one turn (0 or 1) of question q."""
    if turn == 0:
        if ref:
            text = (f"[Instruction]\n{_MATH} Be as objective as possible. {_RATE}\n\n"
                    f"[Question]\n{q['turns'][0]}\n\n"
                    f"[The Start of Reference Answer]\n{ref[0]}\n[The End of Reference Answer]\n\n"
                    f"[The Start of Assistant's Answer]\n{answers[0]}\n[The End of Assistant's Answer]")
        else:
            text = (f"[Instruction]\n{_GENERAL} Begin your evaluation by providing a short "
                    f"explanation. Be as objective as possible. {_RATE}\n\n"
                    f"[Question]\n{q['turns'][0]}\n\n"
                    f"[The Start of Assistant's Answer]\n{answers[0]}\n[The End of Assistant's Answer]")
        return [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}]
    focus = " Your evaluation should focus on the assistant's answer to the second user question."
    if ref:
        system = (f"{_MATH}{focus} Be as objective as possible. {_RATE}")
        refblock = ("<|The Start of Reference Answer|>\n\n"
                    f"### User:\n{q['turns'][0]}\n\n### Reference answer:\n{ref[0]}\n\n"
                    f"### User:\n{q['turns'][1]}\n\n### Reference answer:\n{ref[1]}\n\n"
                    "<|The End of Reference Answer|>\n\n\n")
    else:
        system = (f"{_GENERAL} You evaluation should focus on the assistant's answer to the second "
                  f"user question. Begin your evaluation by providing a short explanation. Be as "
                  f"objective as possible. {_RATE}")
        refblock = ""
    text = (f"{refblock}<|The Start of Assistant A's Conversation with User|>\n\n"
            f"### User:\n{q['turns'][0]}\n\n### Assistant A:\n{answers[0]}\n\n"
            f"### User:\n{q['turns'][1]}\n\n### Assistant A:\n{answers[1]}\n\n"
            "<|The End of Assistant A's Conversation with User|>")
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


# --- HTTP ------------------------------------------------------------------------------------
def chat(base_url, key, body, timeout=3600, retries=3):
    """POST /chat/completions; returns the first choice's message. Retries transient errors."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    for attempt in range(retries + 1):
        req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions",
                                     json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
            return out["choices"][0]["message"], out["choices"][0].get("finish_reason")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code < 500 and e.code != 429 or attempt == retries:
                raise RuntimeError(f"HTTP {e.code} from {base_url}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                raise RuntimeError(f"{base_url}: {e}") from None
        time.sleep(5 * (attempt + 1))


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = DATA / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())


def load_questions(limit):
    download()
    qs = [json.loads(line) for line in open(DATA / "question.jsonl") if line.strip()]
    refs = {}
    for line in open(DATA / "reference-gpt-4.jsonl"):
        if line.strip():
            r = json.loads(line)
            refs[r["question_id"]] = r["choices"][0]["turns"]
    picked, seen = [], {}
    for q in qs:  # the first `limit` questions of each category, in file order
        c = q["category"]
        if seen.get(c, 0) < limit:
            seen[c] = seen.get(c, 0) + 1
            q["ref"] = q.get("reference") or refs.get(q["question_id"]) if c in NEED_REF else None
            picked.append(q)
    return picked


# --- answer ----------------------------------------------------------------------------------
def answer_all(args, qs, path):
    base = f"http://{args.host}:{args.port}/v1"
    key = os.environ.get("API_KEY", "")
    lock, done = threading.Lock(), [0]
    path.unlink(missing_ok=True)

    def one(q):
        msgs, turns, finish, t0 = [], [], [], time.time()
        for text in q["turns"]:
            msgs.append({"role": "user", "content": text})
            try:
                m, fin = chat(base, key, {"model": args.model, "messages": msgs,
                                          "temperature": TEMPERATURE[q["category"]],
                                          "max_tokens": args.max_tokens})
                content = m.get("content") or ""
            except RuntimeError as e:
                content, fin = "", f"error: {e}"
            msgs.append({"role": "assistant", "content": content})
            turns.append(content)
            finish.append(fin)
        rec = {"question_id": q["question_id"], "category": q["category"], "turns": turns,
               "finish_reason": finish, "seconds": round(time.time() - t0, 1)}
        with lock:
            done[0] += 1
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  answered {done[0]}/{len(qs)}  q{q['question_id']} {q['category']}"
                  f"  {rec['seconds']} s  finish={finish}", flush=True)
        return rec

    with ThreadPoolExecutor(args.jobs) as ex:
        return list(ex.map(one, qs))


# --- judge -----------------------------------------------------------------------------------
def judge_all(args, qs, answers, path):
    model = os.environ.get("JUDGE_MODEL", "")
    base = os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1")
    key = os.environ.get("JUDGE_API_KEY", "")
    if not model:
        return None, "JUDGE_MODEL is not set"
    byid = {q["question_id"]: q for q in qs}
    lock = threading.Lock()
    path.unlink(missing_ok=True)

    def one(job):
        a, turn = job
        q = byid[a["question_id"]]
        rec = {"question_id": q["question_id"], "category": q["category"], "turn": turn + 1,
               "score": None}
        if not a["turns"][turn]:
            rec["error"] = "empty answer (error, or max_tokens spent on reasoning): scored 1"
            rec["score"] = 1.0
        else:
            try:
                m, _ = chat(base, key, {"model": model, "temperature": 0, "max_tokens": 2048,
                                        "messages": judge_messages(q, a["turns"], turn, q["ref"])},
                            timeout=600)
                text = m.get("content") or ""
                rec["judgment"] = text
                hit = re.search(r"\[\[(\d+\.?\d*)\]\]", text) or re.search(r"\[(\d+\.?\d*)\]", text)
                if hit:
                    rec["score"] = min(10.0, max(1.0, float(hit.group(1))))
                else:
                    rec["error"] = "no [[rating]] in the judge's reply"
            except RuntimeError as e:
                rec["error"] = str(e)
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  judged q{rec['question_id']} turn {rec['turn']}: {rec['score']}"
                  f"{'  (' + rec['error'] + ')' if rec.get('error') else ''}", flush=True)
        return rec

    jobs = [(a, t) for a in answers for t in range(len(a["turns"]))]
    with ThreadPoolExecutor(args.judge_jobs) as ex:
        out = list(ex.map(one, jobs))
    if all(r["score"] is None for r in out):
        return out, "every judge call failed (see the errors above)"
    return out, None


def summarize(args, answers, judged, skipped):
    res = {"label": args.label, "model": args.model, "questions": len(answers),
           "judge_model": os.environ.get("JUDGE_MODEL", "") or None,
           "judge_base_url": os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1")
           if os.environ.get("JUDGE_MODEL") else None,
           "empty_answers": sum(1 for a in answers for t in a["turns"] if not t),
           "truncated": sum(1 for a in answers for f in a["finish_reason"] if f == "length"),
           "errors": sum(1 for a in answers for f in a["finish_reason"]
                         if str(f).startswith("error"))}
    print(f"\nMT-Bench ({len(answers)} questions, {sum(len(a['turns']) for a in answers)} turns):"
          f" {res['empty_answers']} empty, {res['truncated']} cut by max_tokens,"
          f" {res['errors']} request errors")
    if skipped:
        res["judge"] = f"SKIPPED: {skipped}"
        print("\n" + "!" * 78)
        print(f"!! MT-Bench JUDGING SKIPPED: {skipped}.")
        print("!! The answers were generated and saved, but there are NO MT-Bench scores.")
        print("!! Set JUDGE_MODEL (and JUDGE_BASE_URL, JUDGE_API_KEY), then run:")
        print(f"!!   python bench/mtbench.py judge --label {args.label}")
        print("!" * 78)
    else:
        ok = [r for r in judged if r["score"] is not None]
        failed = len(judged) - len(ok)
        table = {}
        for c in CATEGORIES:
            row = {}
            for t in (1, 2):
                s = [r["score"] for r in ok if r["category"] == c and r["turn"] == t]
                row[f"turn{t}"] = round(sum(s) / len(s), 2) if s else None
            s = [r["score"] for r in ok if r["category"] == c]
            row["avg"] = round(sum(s) / len(s), 2) if s else None
            if s:
                table[c] = row
        overall = {f"turn{t}": [r["score"] for r in ok if r["turn"] == t] for t in (1, 2)}
        overall = {k: round(sum(v) / len(v), 2) if v else None for k, v in overall.items()}
        overall["avg"] = round(sum(r["score"] for r in ok) / len(ok), 2)
        res.update(judge="done", judge_failures=failed, per_category=table, overall=overall)
        print(f"judge: {res['judge_model']} at {res['judge_base_url']}"
              f"{f'  ({failed} turns unscored)' if failed else ''}")
        print(f"{'category':<12} {'turn 1':>7} {'turn 2':>7} {'avg':>6}")
        for c, row in table.items():
            print(f"{c:<12} {row['turn1'] or '-':>7} {row['turn2'] or '-':>7} {row['avg']:>6}")
        print(f"{'OVERALL':<12} {overall['turn1'] or '-':>7} {overall['turn2'] or '-':>7}"
              f" {overall['avg']:>6}   (1 to 10)")
    (DATA / f"result-{args.label}.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["run", "judge", "download"])
    p.add_argument("--label", default="run", help="names the output files")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", default="inkling-2-mini")
    p.add_argument("--limit", type=int, default=10, help="questions per category (10 = all 80)")
    p.add_argument("--jobs", type=int, default=2, help="requests at once to the model")
    p.add_argument("--judge-jobs", type=int, default=4, help="requests at once to the judge")
    p.add_argument("--max-tokens", type=int, default=8192, help="per answer, reasoning included")
    args = p.parse_args()
    if args.cmd == "download":
        download()
        return
    qs = load_questions(args.limit if args.cmd == "run" else 10)  # judge: whatever was answered
    apath = DATA / f"answers-{args.label}.jsonl"
    if args.cmd == "run":
        print(f"MT-Bench: {len(qs)} questions, model {args.model} at {args.host}:{args.port},"
              f" {args.jobs} at once; answers -> {apath}", flush=True)
        answers = answer_all(args, qs, apath)
    else:
        if not apath.exists():
            sys.exit(f"no answers at {apath}: run `mtbench.py run --label {args.label}` first")
        answers = [json.loads(line) for line in open(apath) if line.strip()]
        ids = {a["question_id"] for a in answers}
        qs = [q for q in qs if q["question_id"] in ids]
    judged, skipped = judge_all(args, qs, answers, DATA / f"judgments-{args.label}.jsonl")
    summarize(args, answers, judged, skipped)


if __name__ == "__main__":
    main()
