"""scripts/claude_loop.py against a stub `claude` in a scratch repository (no GPU, no
network, standard library and git only).

    python3 -m pytest -q tests/test_claude_loop.py
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "claude_loop", Path(__file__).resolve().parent.parent / "scripts" / "claude_loop.py")
claude_loop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(claude_loop)

# The stub plays Claude: each call applies the next planned move to the metric file
# value.txt, commits, and prints a result line like `claude -p --output-format json`.
STUB = r'''
import json, subprocess, sys
from pathlib import Path
if "--version" in sys.argv:
    print("stub 0"); sys.exit()
plan = json.loads(Path("plan.json").read_text())
calls = Path(".calls"); n = int(calls.read_text()) if calls.exists() else 0
calls.write_text(str(n + 1))
move = plan[n] if n < len(plan) else "none"
if move == "error":
    print(json.dumps({"type": "result", "is_error": True, "result": "usage limit reached",
                      "total_cost_usd": 0.1})); sys.exit(1)
if move == "protected":
    Path("bench").mkdir(exist_ok=True); Path("bench/x.sh").write_text("echo fake\n")
elif move != "none":
    Path("value.txt").write_text(move)
if move != "none":
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-qm", f"move {n}"], check=True)
print(json.dumps({"type": "result", "is_error": False, "total_cost_usd": 0.5,
                  "result": f"CHANGE: move {move}\nWHY: test\nEXPECT: lower"}))
'''


def make_repo(tmp_path, plan, start="40"):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "value.txt").write_text(start)
    (repo / "plan.json").write_text(json.dumps(plan))
    (repo / ".gitignore").write_text(".local/\n.calls\n")
    stub = tmp_path / "stub_claude.py"
    stub.write_text(STUB)
    git = ["git", "-C", str(repo)]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], check=True)
    subprocess.run(git + ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"],
                   check=True)
    return repo, stub


def loop(monkeypatch, repo, stub, *extra):
    monkeypatch.setattr(claude_loop, "REPO", repo)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")
    monkeypatch.setenv("CLAUDE_LOOP_ALLOW_BYPASS", "1")
    args = ["--name", "t", "--goal", "lower it", "--target", "<=30",
            "--measure", "echo wall $(cat value.txt) ms/step", "--metric", r"wall ([0-9.]+) ms/step",
            "--claude", f"{sys.executable} {stub}", "--repos", ".", "--gpu-free-mib", "0",
            "--min-gain", "0", *extra]
    rc = claude_loop.main(args)
    state = json.loads((repo / ".local/claude-loop-t/state.json").read_text())
    return rc, state, (repo / ".local/claude-loop-t.txt").read_text()


def test_keeps_gains_reverts_regressions_stops_at_target(tmp_path, monkeypatch):
    repo, stub = make_repo(tmp_path, ["35", "38", "protected", "29"])
    rc, state, log = loop(monkeypatch, repo, stub)
    assert rc == 0
    outcomes = [h["outcome"] for h in state["history"]]
    assert outcomes[0].startswith("kept")
    assert outcomes[1].startswith("thrown away: 38")
    assert "protected" in outcomes[2]
    assert outcomes[3].startswith("kept")
    assert state["best"] == 29 and state["baseline"] == 40
    assert (repo / "value.txt").read_text() == "29"
    assert not (repo / "bench").exists()
    assert "target met" in log and state["cost"] == pytest.approx(2.0)


def test_error_stops_and_resume_continues(tmp_path, monkeypatch):
    repo, stub = make_repo(tmp_path, ["35", "error", "30"])
    rc, state, log = loop(monkeypatch, repo, stub)
    assert rc == 1 and "--resume" in log and state["iter"] == 1
    rc, state, log = loop(monkeypatch, repo, stub, "--resume")
    assert rc == 0 and state["best"] == 30


def test_stops_after_two_idle_iterations_and_on_budget(tmp_path, monkeypatch):
    repo, stub = make_repo(tmp_path, ["none", "none", "35"])
    rc, state, log = loop(monkeypatch, repo, stub)
    assert rc == 1 and "changed nothing" in log and state["iter"] == 2
    repo2, stub2 = make_repo(tmp_path / "b", ["39", "38", "37"])
    rc, state, log = loop(monkeypatch, repo2, stub2, "--budget-usd", "1")
    assert "budget spent" in log and state["iter"] == 2


def test_target_parsing():
    assert claude_loop.parse_target("<=35") == ("<=", 35.0)
    assert claude_loop.better(1100, 1000, ">=", 0.005)
    assert not claude_loop.better(1002, 1000, ">=", 0.005)
    with pytest.raises(SystemExit):
        claude_loop.parse_target("35")
