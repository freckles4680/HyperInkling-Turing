#!/usr/bin/env python3
"""Let Claude Code improve a measured number on this machine until a target is hit.

Runs inside the sandbox (docs/claude-sandbox.md). Each iteration:

  1. Claude (headless, `claude -p`) makes one change toward the goal and commits it.
  2. This script, not Claude, checks the change: protected files untouched, the gate
     command passes (kernel tests), then the measure command runs and the metric is read
     from its output with a regex.
  3. Better than the best so far: kept. Otherwise both repositories go back to the best
     commit, and the attempt (its diff and Claude's summary) is shown to the next iteration.

It stops when the target is met, after --max-iters iterations, when the estimated spend
passes --budget-usd, after two iterations in a row with no change, or when Claude reports
an error (a usage limit, say): --resume carries on later from the saved state.

Everything goes into one file to share, .local/claude-loop-<name>.txt, with each step's
full output under .local/claude-loop-<name>/.

    python3 scripts/claude_loop.py --name dec90k --preset decode-90k --target '<=35' \\
        --goal "Lower the decode step time at 90k context on the 2080 Ti"
    python3 scripts/claude_loop.py --name dec90k --resume          # after a stop

Work happens on branch loop/<name> in this repository and in vllm/; the best commit is
left checked out at the end. Nothing is pushed.
"""

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Ready-made measurements. The command's {name} becomes <name>-i<iteration>.
PRESETS = {
    # Fixed verify length (k=4): steps comparable run to run (adaptive k and DSpark's
    # acceptance move tok/s; ms/step at fixed k does not). About 5 minutes.
    "decode-90k": dict(
        measure="HYPERINKLING_SPEC_ADAPTIVE= bench/profile_decode.sh {name} 90000 256",
        metric=r"wall ([0-9.]+) ms/step",
        gate="venv/bin/python -m pytest -q -x tests/test_fp16.py tests/test_sm75_attention.py",
    ),
    "decode-8k": dict(
        measure="HYPERINKLING_SPEC_ADAPTIVE= bench/profile_decode.sh {name} 8192 256",
        metric=r"wall ([0-9.]+) ms/step",
        gate="venv/bin/python -m pytest -q -x tests/test_fp16.py tests/test_sm75_attention.py",
    ),
    # Prefill throughput of one 8k prompt, under nsys (higher is better: --target '>=2400').
    "prefill-8k": dict(
        measure="bench/profile_prefill.sh {name} 8192",
        metric=r"tokens in [0-9.]+ s: ([0-9.]+) tok/s",
        gate="venv/bin/python -m pytest -q -x tests/test_fp16.py tests/test_sm75_attention.py",
    ),
}

# Files Claude must not change: the measurement, the sandbox, this script. (The sandbox's
# managed settings deny edits there too; this check also catches shell writes.)
PROTECTED = ["bench/", "sandbox/", "scripts/claude_loop.py", "scripts/server-lib.sh", "models/"]

PROMPT = """You are in an unattended optimisation loop on the user's own machine (a sandboxed
container with the GPU: an RTX 2080 Ti 22 GB, sm75). Nobody will answer questions: decide.

Goal: {goal}
Metric: {metric_desc}. Target: {target}. Best so far: {best} (baseline {baseline}).
Iteration {it} of at most {max_iters}.

History (measured by the loop, not by you):
{history}

Last measurement output (tail; full log at {last_log}):
{last_tail}

Rules:
- Read CLAUDE.md, and NOTES.md / FOLLOWUPS.md before changing a design decision: many
  alternatives were already measured. Do not repeat an attempt the history shows failed.
- Make ONE focused change this iteration, the most promising next step. Keep numerics:
  small loop kernels carry the reference numerics, check fusions bitwise as CLAUDE.md says.
- Do not edit {protected}. Do not push, and do not touch git remotes.
- Do not start servers on port 8080 and do not run the benchmark yourself: the loop runs
  `{measure}` after you finish. Quick kernel tests and offline checks are fine and
  encouraged (venv/bin/python -m pytest -q tests/<file> -k <name>, bench/sm75_compile.py).
  The loop also runs the gate `{gate}`: a failing gate throws the change away.
- Model code lives in vllm/ (its own git repository). Commit there with a NEW commit on the
  current branch (no amend, no rebase): the user squashes into the patch afterwards. Commit
  changes to this repository too. Short commit messages that say what and why.
- Finish with three lines: CHANGE: what you changed. WHY: the evidence. EXPECT: the effect
  on the metric.
"""


def now():
    return datetime.now().strftime("%H:%M:%S")


class Log:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, text=""):
        line = text if text.startswith("=====") or not text else f"[{now()}] {text}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def run(cmd, cwd=None, timeout=None, log_file=None, env=None):
    """Shell command; returns (exit code, output). Output also goes to log_file."""
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd or REPO, text=True, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           env={**os.environ, **(env or {})})
        rc, out = p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        rc, out = 124, out + f"\n[timed out after {timeout} s]"
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(f"$ {cmd}\n{out}\n[exit {rc}]\n")
    return rc, out


def git(repo, *args, check=True):
    p = subprocess.run(["git", "-C", str(repo), *args], text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode:
        raise RuntimeError(f"git -C {repo} {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout.strip()


def parse_target(text):
    m = re.fullmatch(r"\s*(<=|>=|<|>)\s*([-+0-9.eE]+)\s*", text)
    if not m:
        raise SystemExit(f"--target must look like '<=35' or '>=1000', not {text!r}")
    return m.group(1), float(m.group(2))


def meets(value, op, bound):
    return {"<=": value <= bound, "<": value < bound,
            ">=": value >= bound, ">": value > bound}[op]


def better(new, old, op, min_gain):
    """new beats old by more than min_gain (a fraction of old)."""
    if old is None:
        return True
    margin = abs(old) * min_gain
    return new < old - margin if op.startswith("<") else new > old + margin


def gpu_busy_mib():
    """Memory in use on the GPU, or None without nvidia-smi. (In a container nvidia-smi
    lists no processes of the host, so scripts/server-lib.sh would not see a server
    running there: memory is the check that does.)"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], text=True,
                             capture_output=True, timeout=30).stdout
        return max(int(x) for x in out.split())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class Loop:
    def __init__(self, a):
        self.a = a
        self.dir = REPO / ".local" / f"claude-loop-{a.name}"
        self.state_path = self.dir / "state.json"
        self.log = Log(REPO / ".local" / f"claude-loop-{a.name}.txt")
        self.repos = [REPO / r for r in a.repos if (REPO / r / ".git").exists()]
        self.branch = f"loop/{a.name}"
        self.op, self.bound = parse_target(a.target)
        self.state = None

    # --- state ---------------------------------------------------------------------------
    def heads(self):
        return {str(r.relative_to(REPO)): git(r, "rev-parse", "HEAD") for r in self.repos}

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def reset_to(self, heads):
        for r in self.repos:
            git(r, "reset", "-q", "--hard", heads[str(r.relative_to(REPO))])
            git(r, "clean", "-q", "-fd")   # untracked only; ignored files (venv, logs) stay

    def check_clean(self):
        for r in self.repos:
            if git(r, "status", "--porcelain"):
                raise SystemExit(f"{r} has uncommitted changes: commit or stash them first "
                                 "(the loop resets to its best commit)")

    def start_branches(self):
        for r in self.repos:
            if git(r, "rev-parse", "--verify", "-q", self.branch, check=False):
                raise SystemExit(f"{r}: branch {self.branch} exists (--resume, or another --name)")
            git(r, "switch", "-q", "-c", self.branch)

    # --- steps ---------------------------------------------------------------------------
    def wait_gpu_free(self):
        if self.a.gpu_free_mib <= 0:
            return
        for _ in range(30):
            used = gpu_busy_mib()
            if used is None:
                self.log("nvidia-smi gave nothing: not checking that the GPU is free")
                return
            if used <= self.a.gpu_free_mib:
                return
            time.sleep(4)
        raise SystemExit(f"the GPU has {used} MiB in use (another server? the host's "
                         f"scripts/serve.sh?): stop it, or raise --gpu-free-mib")

    def measure(self, tag):
        """Gate, then the measurement; returns (value or None, note, log path)."""
        if self.a.gate:
            path = self.dir / f"{tag}-gate.log"
            rc, out = run(self.a.gate, timeout=self.a.step_timeout, log_file=path)
            if rc:
                return None, f"gate failed (exit {rc}), {path.relative_to(REPO)}", path
        values, path = [], None
        for rep in range(self.a.repeat):
            self.wait_gpu_free()
            name = f"{self.a.name}-{tag}" + (f"r{rep}" if self.a.repeat > 1 else "")
            cmd = self.a.measure.replace("{name}", name)
            path = self.dir / f"{tag}-measure{rep if self.a.repeat > 1 else ''}.log"
            rc, out = run(cmd, timeout=self.a.step_timeout, log_file=path)
            found = re.findall(self.a.metric, out)
            if not found:
                return None, f"no metric in the output (exit {rc}), {path.relative_to(REPO)}", path
            last = found[-1]
            values.append(float(last[0] if isinstance(last, tuple) else last))
        value = statistics.median(values)
        spread = f" (runs {', '.join(f'{v:g}' for v in values)})" if len(values) > 1 else ""
        return value, f"{value:g}{spread}", path

    def ask_claude(self, it):
        s = self.state
        hist = "\n".join(
            f"- iter {h['iter']}: {h['outcome']}; metric {h['value']}. "
            + (h.get("summary") or "").replace("\n", " ")[:600]
            for h in s["history"]) or "- (none yet)"
        last_log = s.get("last_log")
        tail = ""
        if last_log and Path(last_log).exists():
            tail = "\n".join(Path(last_log).read_text(errors="replace").splitlines()[-60:])
        prompt = PROMPT.format(
            goal=self.a.goal, metric_desc=f"regex {self.a.metric!r} on the measure output",
            target=self.a.target, best=s["best"], baseline=s["baseline"], it=it,
            max_iters=self.a.max_iters, history=hist, last_log=last_log, last_tail=tail,
            protected=", ".join(PROTECTED), measure=self.a.measure, gate=self.a.gate or "none")
        if self.a.extra_prompt:
            prompt += "\nAlso:\n" + self.a.extra_prompt + "\n"
        cmd = [*shlex.split(self.a.claude), "-p", prompt, "--output-format", "json",
               "--max-turns", str(self.a.max_turns), "--permission-mode", self.a.permission_mode]
        if self.a.model:
            cmd += ["--model", self.a.model]
        if self.a.iter_budget_usd:
            cmd += ["--max-budget-usd", str(self.a.iter_budget_usd)]
        path = self.dir / f"i{it}-claude.json"
        try:
            p = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True,
                               timeout=self.a.claude_timeout)
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            rc, out, err = 124, "", f"claude timed out after {self.a.claude_timeout} s"
        path.write_text(out + ("\n--- stderr ---\n" + err if err else ""))
        try:
            res = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
        except json.JSONDecodeError:
            res = {}
        cost = float(res.get("total_cost_usd") or 0)
        error = rc != 0 or res.get("is_error") or not res
        text = res.get("result") or err.strip() or out.strip()[-500:]
        return error, cost, text, path

    def touched_protected(self, before):
        changed = set()
        for r in self.repos:
            rel = str(r.relative_to(REPO))
            names = git(r, "diff", "--name-only", before[rel], "HEAD").splitlines()
            for n in names:
                changed.add(n if rel == "." else f"{rel}/{n}")
        return sorted(n for n in changed if any(n == p or n.startswith(p) for p in PROTECTED))

    def commit_leftovers(self, it):
        for r in self.repos:
            if git(r, "status", "--porcelain"):
                git(r, "add", "-A")
                git(r, "commit", "-q", "-m", f"loop {self.a.name} iteration {it}: uncommitted changes")

    def diff_stat(self, before):
        parts = []
        for r in self.repos:
            rel = str(r.relative_to(REPO))
            stat = git(r, "diff", "--shortstat", before[rel], "HEAD")
            if stat:
                parts.append(f"{rel}: {stat}")
        return "; ".join(parts) or "no change"

    def save_diff(self, before, it):
        text = ""
        for r in self.repos:
            rel = str(r.relative_to(REPO))
            text += f"### {rel}\n" + git(r, "diff", before[rel], "HEAD") + "\n"
        path = self.dir / f"i{it}.diff"
        path.write_text(text)
        return path

    # --- main ------------------------------------------------------------------------------
    def run(self):
        a, log = self.a, self.log
        if a.resume:
            if not self.state_path.exists():
                raise SystemExit(f"nothing to resume: {self.state_path} does not exist")
            self.state = json.loads(self.state_path.read_text())
            for k in ("goal", "measure", "metric", "gate", "target"):
                setattr(a, k, getattr(a, k) or self.state["args"][k])
            self.op, self.bound = parse_target(a.target)
            for r in self.repos:
                git(r, "switch", "-q", self.branch)
            self.reset_to(self.state["best_heads"])
            log(f"===== resumed {a.name} ({datetime.now().isoformat(timespec='seconds')}) =====")
        else:
            if not (a.goal and a.measure and a.metric and a.target):
                raise SystemExit("needs --goal, --target and --measure/--metric (or --preset)")
            self.check_clean()
            self.start_branches()
            log(f"===== claude-loop {a.name} ({datetime.now().isoformat(timespec='seconds')}) =====")
            log(f"goal: {a.goal}")
            log(f"commit {git(REPO, 'log', '--oneline', '-1')}; claude {run(a.claude + ' --version')[1].strip()}")
            gpu = run("nvidia-smi --query-gpu=name,memory.total,temperature.gpu,power.limit "
                      "--format=csv,noheader")[1].strip()
            log(f"GPU: {gpu.splitlines()[0] if gpu else 'none'}")
            log(f"target: metric {a.target}; measure: {a.measure}; metric regex: {a.metric}")
            log(f"gate: {a.gate or 'none'}; repos: {', '.join(str(r.relative_to(REPO)) for r in self.repos)}"
                f"; branch {self.branch}")
            log(f"limits: {a.max_iters} iterations, ${a.budget_usd} total, "
                f"${a.iter_budget_usd or '-'} and {a.max_turns} turns per iteration; "
                f"permission mode {a.permission_mode}")
            log("baseline measurement")
            value, note, path = self.measure("i0")
            if value is None:
                raise SystemExit(f"the baseline did not measure: {note}")
            log(f"baseline: {note}")
            self.state = dict(args={k: getattr(a, k) for k in ("goal", "measure", "metric", "gate", "target")},
                              baseline=value, best=value, best_iter=0, best_heads=self.heads(),
                              cost=0.0, iter=0, history=[], last_log=str(path), no_change=0)
            self.save()

        s = self.state
        reason = None
        while True:
            if meets(s["best"], self.op, self.bound):
                reason = f"target met: {s['best']:g} {a.target}"
                break
            if s["iter"] >= a.max_iters:
                reason = f"{a.max_iters} iterations done"
                break
            if s["cost"] >= a.budget_usd:
                reason = f"budget spent: ${s['cost']:.2f} of ${a.budget_usd}"
                break
            if s["no_change"] >= 2:
                reason = "two iterations in a row changed nothing"
                break
            it = s["iter"] + 1
            log(f"===== iteration {it} =====")
            before = self.heads()
            error, cost, text, cpath = self.ask_claude(it)
            s["cost"] += cost
            summary = "\n".join(text.strip().splitlines()[-6:]) if text else ""
            log(f"claude: ${cost:.2f} (total ${s['cost']:.2f}), {cpath.relative_to(REPO)}")
            for line in summary.splitlines():
                log(f"  | {line}")
            if error:
                self.reset_to(before)
                self.save()
                reason = ("claude stopped with an error (a usage limit or an API error; "
                          f"see {cpath.relative_to(REPO)}): rerun with --resume later")
                break
            self.commit_leftovers(it)
            entry = dict(iter=it, summary=summary, value=None, cost=round(cost, 4))
            s["iter"] = it
            bad = self.touched_protected(before)
            if bad:
                entry["outcome"] = f"thrown away: changed protected files {', '.join(bad)}"
            elif self.heads() == before:
                entry["outcome"] = "no change"
                s["no_change"] += 1
            else:
                s["no_change"] = 0
                entry["diff"] = str(self.save_diff(before, it).relative_to(REPO))
                log(f"change: {self.diff_stat(before)} ({entry['diff']})")
                value, note, path = self.measure(f"i{it}")
                entry["value"] = value
                s["last_log"] = str(path) if path else s["last_log"]
                if value is None:
                    entry["outcome"] = f"thrown away: {note}"
                elif better(value, s["best"], self.op, a.min_gain):
                    entry["outcome"] = f"kept, new best (was {s['best']:g})"
                    s.update(best=value, best_iter=it, best_heads=self.heads())
                else:
                    entry["outcome"] = f"thrown away: {value:g} not better than {s['best']:g}"
            if not entry["outcome"].startswith("kept"):
                self.reset_to(s["best_heads"])
            log(f"result: {entry['outcome']}; metric {entry['value']}")
            s["history"].append(entry)
            self.save()

        log("===== summary =====")
        log(f"stopped: {reason}")
        log(f"baseline {s['baseline']:g}, best {s['best']:g} at iteration {s['best_iter']}, "
            f"target {a.target}: {'MET' if meets(s['best'], self.op, self.bound) else 'not met'}")
        log(f"estimated spend ${s['cost']:.2f} over {s['iter']} iterations")
        for h in s["history"]:
            log(f"  iter {h['iter']}: {h['outcome']}")
        for rel, head in s["best_heads"].items():
            log(f"best commit in {rel}: {head[:12]} on {self.branch}")
        log("next: review `git log` on the loop branches; to keep a vllm/ change, squash it into "
            "the one patch commit (CLAUDE.md) and regenerate patches/vllm-hyperinkling-sm86.patch")
        return 0 if meets(s["best"], self.op, self.bound) else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.split("\n\n")[1:]))
    p.add_argument("--name", required=True, help="label: branch loop/<name>, log .local/claude-loop-<name>.txt")
    p.add_argument("--goal", help="what to improve, in words (goes to Claude)")
    p.add_argument("--target", help="stop when the metric is e.g. '<=35' or '>=1000'")
    p.add_argument("--preset", choices=sorted(PRESETS), help="measure, metric and gate for a known bench")
    p.add_argument("--measure", help="shell command printing the metric; {name} -> a unique label")
    p.add_argument("--metric", help="regex with one group, the number (last match wins)")
    p.add_argument("--gate", help="shell command that must pass before measuring (tests)")
    p.add_argument("--repeat", type=int, default=1, help="measure N times, take the median")
    p.add_argument("--min-gain", type=float, default=0.005,
                   help="relative gain a change must beat to be kept (noise guard, default 0.5%%)")
    p.add_argument("--max-iters", type=int, default=8)
    p.add_argument("--budget-usd", type=float, default=25.0,
                   help="stop when Claude's estimated spend passes this (total_cost_usd)")
    p.add_argument("--iter-budget-usd", type=float, default=5.0, help="claude --max-budget-usd per iteration")
    p.add_argument("--max-turns", type=int, default=80, help="claude --max-turns per iteration")
    p.add_argument("--model", help="claude --model (default: Claude Code's)")
    p.add_argument("--permission-mode", default="bypassPermissions",
                   help="bypassPermissions only inside the sandbox; dontAsk with allow rules elsewhere")
    p.add_argument("--extra-prompt", help="more instructions for Claude")
    p.add_argument("--repos", nargs="+", default=[".", "vllm"], help="git repositories the loop manages")
    p.add_argument("--gpu-free-mib", type=int, default=1500,
                   help="wait for the GPU to use at most this much before measuring (0: no check)")
    p.add_argument("--step-timeout", type=int, default=3600, help="seconds for the gate or one measurement")
    p.add_argument("--claude-timeout", type=int, default=5400, help="seconds for one claude run")
    p.add_argument("--claude", default="claude", help="the claude command (tests use a stub)")
    p.add_argument("--resume", action="store_true", help="carry on from .local/claude-loop-<name>/state.json")
    a = p.parse_args(argv)
    if a.preset:
        for k, v in PRESETS[a.preset].items():
            if getattr(a, k) is None:
                setattr(a, k, v)
    # CLAUDE_LOOP_ALLOW_BYPASS=1 skips these two checks (tests/test_claude_loop.py's stub).
    if a.permission_mode == "bypassPermissions" and not os.environ.get("CLAUDE_LOOP_ALLOW_BYPASS"):
        if not Path("/.dockerenv").exists():
            raise SystemExit("bypassPermissions outside a container: run this in the sandbox "
                             "(sandbox/agent.sh), or pass --permission-mode dontAsk")
        if os.geteuid() == 0:
            raise SystemExit("running as root: use sandbox/agent.sh (the agent user)")
    return Loop(a).run()


if __name__ == "__main__":
    sys.exit(main())
