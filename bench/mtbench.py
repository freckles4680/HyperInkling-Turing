"""MT-Bench-style judged chat against a running server: 80 two-turn questions in 8
categories (writing, roleplay, reasoning, math, coding, extraction, stem, humanities),
answered by the model, then scored 1 to 10 per turn by a separate judge model.

    python bench/mtbench.py run   --label <name> [--port 8080] [--limit 10] [--jobs 2]
    python bench/mtbench.py judge --label <name>          # judge saved answers again (or later)
    python bench/mtbench.py export --label <name> [--batch 10]   # files to grade in a chat
    python bench/mtbench.py import --label <name> replies.txt    # read the chat's scores back

Questions and reference answers (for math, reasoning and coding) are lm-sys FastChat's MT-Bench
files (Apache-2.0), downloaded once into bench/quality-data/mt_bench/. The judge prompts follow
FastChat's "single answer grading" prompts; the reference answers are FastChat's GPT-4 ones.

The judge is any OpenAI-compatible chat endpoint, set by environment variables:

    JUDGE_MODEL      model name at the judge endpoint (empty = no judging: answers only)
    JUDGE_BASE_URL   default https://api.openai.com/v1 (OpenRouter, a llama.cpp or vLLM server
                     on another machine, ... anything that serves /v1/chat/completions)
    JUDGE_API_KEY    sent as "Authorization: Bearer <key>" (empty = no header)

Without JUDGE_MODEL the answers are still generated and saved, the summary says loudly that
judging was SKIPPED, and the answers are exported for grading by hand: paste
export/<name>/batch-*.md one by one into one chat with a strong model (Claude Opus, ...), save
its replies into a text file, and `import` reads the "q81 t1: 7" lines back into the result
and appends the scores to .local/intelligence-<name>.txt. `judge` can still score them later. Scores from different judges are not
comparable: compare two runs with the same judge only.

Files (under bench/quality-data/mt_bench/):
    answers-<label>.jsonl     one line per question, both turns (tail -f it to follow a run)
    judgments-<label>.jsonl   one line per judged turn, the judge's full explanation included
    result-<label>.json       the summary that is printed at the end
"""

import argparse
import contextlib
import io
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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "quality-data" / "mt_bench"
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


def summarize(args, answers, judged, skipped, judge_name=None, judge_where=None):
    if judge_name is None and os.environ.get("JUDGE_MODEL"):
        judge_name = os.environ["JUDGE_MODEL"]
        judge_where = os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1")
    res = {"label": args.label, "model": args.model, "questions": len(answers),
           "judge_model": judge_name, "judge_base_url": judge_where,
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
        print("!! Either grade them by hand in a chat (Claude Opus, ...): the files to paste are")
        print(f"!!   {EXPORT / args.label}/  (python3 bench/mtbench.py export --label {args.label})")
        print("!! then save the chat's replies to one text file and run")
        print(f"!!   python3 bench/mtbench.py import --label {args.label} <replies.txt>")
        print("!! or set JUDGE_MODEL (and JUDGE_BASE_URL, JUDGE_API_KEY) and run:")
        print(f"!!   python3 bench/mtbench.py judge --label {args.label}")
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
        print(f"judge: {res['judge_model']}{' at ' + res['judge_base_url'] if res['judge_base_url'] else ''}"
              f"{f'  ({failed} turns unscored)' if failed else ''}")
        print(f"{'category':<12} {'turn 1':>7} {'turn 2':>7} {'avg':>6}")
        for c, row in table.items():
            print(f"{c:<12} {row['turn1'] or '-':>7} {row['turn2'] or '-':>7} {row['avg']:>6}")
        print(f"{'OVERALL':<12} {overall['turn1'] or '-':>7} {overall['turn2'] or '-':>7}"
              f" {overall['avg']:>6}   (1 to 10)")
    (DATA / f"result-{args.label}.json").write_text(json.dumps(res, indent=1))
    return res


# --- export / import: grade by hand in a chat window ---------------------------------------
# `export` writes the judge instructions, the questions and the answers as Markdown files of
# --batch questions each (10 = about 15-30k tokens: one paste in a Claude chat). Paste them in
# order into ONE chat; each reply ends in a SCORES block. Save all the replies into one text
# file (or one per batch) and `import` it: every "q<id> t<turn>: <score>" line is read, the
# last one wins, and the summary goes into the result and the intelligence report.
EXPORT = DATA / "export"
SCORE_LINE = re.compile(r"\bq\s*(\d+)\s*[-_ ]?\s*t(?:urn)?\s*([12])\s*[:=]\s*\[*(\d+(?:\.\d+)?)", re.I)

CHAT_RULES = """\
# MT-Bench grading: batch {batch} of {nbatch}

You are an impartial judge grading an AI assistant's answers to MT-Bench questions (two-turn
conversations, lm-sys FastChat). Grade every turn of every question below on a scale of 1 to 10.

Rules, the standard MT-Bench single-answer grading:
- Consider helpfulness, relevance, accuracy, depth, creativity and level of detail.
- When a REFERENCE ANSWER is given (math, reasoning, coding), grade correctness first: compare the
  assistant's answer with the reference, identify any mistakes, and score a wrong final answer low
  however well it is written.
- Turn 2: grade only the assistant's answer to the second user message, in the context of the
  whole conversation.
- An empty answer scores 1. Do not let answer length alone raise a score. Be as objective as
  possible and grade each question on its own, not relative to the others.
- Write one or two sentences of explanation per turn, then finish your reply with exactly one
  block in this format, one line per turn, nothing else inside it:

```
SCORES
q<id> t1: <score>
q<id> t2: <score>
```

{more}
---
"""


def export_chat(args, qs, answers, batch):
    byid = {q["question_id"]: q for q in qs}
    answers = sorted(answers, key=lambda a: a["question_id"])
    chunks = [answers[i:i + batch] for i in range(0, len(answers), batch)]
    out = EXPORT / args.label
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("batch-*.md"):
        old.unlink()
    paths = []
    for n, chunk in enumerate(chunks, 1):
        more = ("More batches follow in this chat: grade each batch on its own when it arrives, "
                "with the same rules." if len(chunks) > 1 else "")
        parts = [CHAT_RULES.format(batch=n, nbatch=len(chunks), more=more)]
        for a in chunk:
            q = byid[a["question_id"]]
            parts.append(f"\n## Question q{q['question_id']} ({q['category']})\n")
            for t, text in enumerate(q["turns"]):
                parts.append(f"### Turn {t + 1}, user\n\n{text}\n")
                if q.get("ref"):
                    parts.append(f"### Turn {t + 1}, REFERENCE ANSWER\n\n{q['ref'][t]}\n")
                ans = a["turns"][t] if t < len(a["turns"]) else ""
                parts.append(f"### Turn {t + 1}, assistant\n\n{ans or '(empty answer)'}\n")
        ids = " ".join(f"q{a['question_id']} t1, t2;" for a in chunk)
        parts.append(f"\n---\nEnd of batch {n} of {len(chunks)}. Give the SCORES block for: {ids}\n")
        path = out / f"batch-{n:02d}.md"
        path.write_text("\n".join(parts))
        paths.append(path)
    print(f"exported {len(answers)} questions in {len(paths)} files to {out}/ (~{batch} questions each):")
    for path in paths:
        print(f"  {path}  ({path.stat().st_size // 4000}k tokens, roughly)")
    print("Paste or attach them one by one in ONE chat, save every reply (in order) into one text file, then:")
    print(f"  python3 bench/mtbench.py import --label {args.label} <that file> [--judge-name 'Opus 4.x chat']")
    return paths


def import_chat(args, qs, answers, files):
    scores = {}
    for f in files:
        for qid, turn, score in SCORE_LINE.findall(Path(f).read_text(errors="replace")):
            scores[(int(qid), int(turn))] = min(10.0, max(1.0, float(score)))
    byid = {q["question_id"]: q for q in qs}
    judged, missing = [], []
    for a in answers:
        for t in range(1, len(a["turns"]) + 1):
            rec = {"question_id": a["question_id"], "category": byid[a["question_id"]]["category"],
                   "turn": t, "score": scores.get((a["question_id"], t)), "judge": args.judge_name}
            if rec["score"] is None:
                rec["error"] = "no score in the pasted replies"
                missing.append(f"q{a['question_id']} t{t}")
            judged.append(rec)
    extra = sorted({q for q, _ in scores} - {a["question_id"] for a in answers})
    with open(DATA / f"judgments-{args.label}.jsonl", "w") as f:
        for rec in judged:
            f.write(json.dumps(rec) + "\n")
    print(f"imported {len(scores)} scores from {len(files)} file(s)")
    if missing:
        print(f"MISSING {len(missing)} turns (not counted; paste those batches again): {' '.join(missing)}")
    if extra:
        print(f"ignored scores for questions that were not answered in this run: {extra}")
    if len(missing) == len(judged):
        sys.exit("no 'q<id> t<turn>: <score>' lines found: check the file holds the chat's replies")
    return judged


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["run", "judge", "export", "import", "download"])
    p.add_argument("files", nargs="*", help="import: text file(s) holding the chat's replies")
    p.add_argument("--label", default="run", help="names the output files")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", default="inkling-2-mini")
    p.add_argument("--limit", type=int, default=10, help="questions per category (10 = all 80)")
    p.add_argument("--jobs", type=int, default=2, help="requests at once to the model")
    p.add_argument("--judge-jobs", type=int, default=4, help="requests at once to the judge")
    p.add_argument("--max-tokens", type=int, default=8192, help="per answer, reasoning included")
    p.add_argument("--batch", type=int, default=10, help="export: questions per file to paste")
    p.add_argument("--judge-name", default="manual (chat)", help="import: who graded, for the report")
    p.add_argument("--report", help="import: report to append the summary to"
                   " (default .local/intelligence-<label>.txt when it exists)")
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
    if args.cmd == "export":
        export_chat(args, qs, answers, args.batch)
        return
    if args.cmd == "import":
        if not args.files:
            sys.exit("usage: mtbench.py import --label <name> <replies.txt> [more.txt ...]")
        judged = import_chat(args, qs, answers, args.files)
        report = Path(args.report) if args.report else ROOT / ".local" / f"intelligence-{args.label}.txt"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summarize(args, answers, judged, None, judge_name=args.judge_name)
        print(buf.getvalue(), end="")
        if report.exists():
            with open(report, "a") as f:
                f.write(f"\n===== MT-Bench scores imported from a chat ({time.strftime('%Y-%m-%d %H:%M')}) =====\n")
                f.write(buf.getvalue())
            print(f"appended to {report}")
        return
    judged, skipped = judge_all(args, qs, answers, DATA / f"judgments-{args.label}.jsonl")
    summarize(args, answers, judged, skipped)
    if skipped:  # no judge: leave the files for grading by hand ready
        export_chat(args, qs, answers, args.batch)


if __name__ == "__main__":
    main()
