# Claude Code on the 2080 Ti box, sandboxed, in a loop

This sets up Claude Code on the machine with the GPU so that it can change code, run the
tests and benchmarks itself, and keep going until a number is reached, without you
pasting logs back. It runs in a Docker container that holds only this project, can reach
only Anthropic's API, and has the one GPU.

```
sandbox/setup.sh                        # once: checks, builds, starts, tests the container
sandbox/agent.sh claude                 # Claude Code, interactive, in the sandbox
sandbox/agent.sh python3 scripts/claude_loop.py --name dec90k --preset decode-90k \
    --target '<=35' --goal "Lower the decode step time at 90k context"
```

The loop writes one file to share, `.local/claude-loop-<name>.txt`.

## Why a container, and what it can reach

Claude Code has a built-in sandbox (bubblewrap on Linux), but it is the wrong fit here. It
covers only the commands Claude runs (not its file edits), it lets those commands read the
whole machine (your `~/.ssh` included) unless you list what to hide, and its docs say
nothing about GPU devices. Running unattended also means no permission prompts, and
`--dangerously-skip-permissions` should not run on the host itself. A container fixes all
of this:

| | In the sandbox | Not in reach |
|---|---|---|
| Files | the project, read-write, at its host path; `models/` and `hf-cache/` read-only; uv's Python and the CUDA toolkit read-only | your home, `~/.ssh`, git credentials, the Docker socket, other disks |
| Network | loopback (its test servers), DNS, HTTPS to `api.anthropic.com`, `claude.ai`, `claude.com`, `console.anthropic.com`, `platform.claude.com` | GitHub, Hugging Face, PyPI, your LAN, Tailscale peers |
| Ports | none published: servers it starts are visible inside only and never clash with yours on 8080 | |
| Privileges | your uid, no capabilities, no sudo | the firewall (only root at container start sets it) |
| GPU | the 2080 Ti, through the NVIDIA Container Toolkit | |
| Memory | 26 GB of your 32 (`MEM_LIMIT`) | |

On top of that, `sandbox/managed-settings.json` is installed as Claude Code's managed
settings, which no other settings file can override. It denies edits to the benchmark
(`bench/`), the sandbox, the loop script and `models/`, reading `.env`, and commands like
`git push`, `sudo`, `docker`, `pip install` and `hf`. The loop also checks every change
for edits to those files and throws such a change away, because a deny rule cannot stop
every way a shell can write a file.

Three things it does not stop. Claude can read any file in the project. It can send data
to Anthropic's API, which it has to reach anyway. And DNS lookups go out, as in Anthropic's
own reference container. Keep secrets out of the project folder. The login token lives
in `~/.config/hyperinkling-sandbox/secrets.env`, outside it.

**Why not the repo's Dockerfile and docker-compose.yml?** They build the serving image:
vLLM compiled into the image and `start.sh` as the entrypoint. The loop needs the opposite:
your editable `vllm/` checkout and `venv/`, so that a Python or Triton edit takes effect at
the next server start without a 30 to 60 minute image build. So the sandbox image holds no
Python, CUDA or vLLM of its own. It mounts yours at the same paths as on the host, because
a venv holds absolute paths, and it uses your distribution as its base, so that the vLLM
extension you built finds the same glibc and libstdc++. The GPU reservation is the same
block as in `docker-compose.yml`.

## Steps

### 1. Docker and the NVIDIA Container Toolkit (once, on the host)

Fedora:

```bash
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager addrepo --from-repofile=https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit
```

Ubuntu / Debian:

```bash
# Docker: https://docs.docker.com/engine/install/ubuntu/ (docker-ce + docker-compose-plugin)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

Then, on both:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker && sudo systemctl restart docker
sudo usermod -aG docker "$USER"        # log out and back in once
docker run --rm --gpus all ubuntu:24.04 nvidia-smi    # should show the 2080 Ti
```

Being in the `docker` group is as good as root on this machine. If that matters to you,
use rootless Docker instead; the sandbox has not been tried with it.

### 2. A login for the container

Pick one, and put it in `~/.config/hyperinkling-sandbox/secrets.env` (setup makes the
file, mode 600):

- **Pro or Max subscription:** on any machine with Claude Code, run `claude setup-token`
  and log in. It prints a token valid for a year. Add the line
  `CLAUDE_CODE_OAUTH_TOKEN=<token>`. Usage counts against your plan's limits.
- **API key** from console.anthropic.com: `ANTHROPIC_API_KEY=sk-ant-...`. You pay per
  token. If both are set, the API key wins in `claude -p`, which is what the loop uses.
- Or leave the file empty, and run `/login` once in `sandbox/agent.sh claude`. The login
  is kept in the container's home volume.

### 3. Build and check

```bash
cd /mnt/wdblack/ai_dev/projects/hyperinkling
git pull
sandbox/setup.sh
```

It refuses to go on if something is missing. It checks Docker and the nvidia runtime, finds
the venv's Python and the CUDA toolkit from `venv/cuda_home`, and writes `sandbox/.env`. Then
it builds the image (a few minutes), starts the container, and checks from inside: the GPU is
visible, torch sees `(7, 5)`, the agent has your uid, Hugging Face is blocked, and Claude Code
starts. Options:

- `EXTRA_PACKAGES=...` adds distribution packages, for example the gcc-15 package if your
  host compiles CUDA with a gcc beside the default one (CLAUDE.md).
- `ALLOW_DOMAINS="pypi.org files.pythonhosted.org"` opens hosts, for a `pip install`.
- `MEM_LIMIT=24g` changes the memory cap.
- `REBUILD=1` builds from scratch. Claude Code does not update itself in the sandbox, so do
  this to get a new version.

Run it again after moving the project, or after rebuilding `venv/` with another Python
or CUDA toolkit.

### 4. Free the GPU

Two engines do not fit on the card. Stop `scripts/serve.sh` (or anything else on the
GPU) before a loop, and preferably the desktop too, which costs about 3 ms of every decode
step (`sudo systemctl isolate multi-user.target`). Before every measurement, the loop waits
until the GPU uses less than 1.5 GB (`--gpu-free-mib`). If it stays busy, the loop stops.
It checks memory because from inside a container, `nvidia-smi` does not list the host's
processes.

### 5. Try it interactively first

```bash
sandbox/agent.sh                 # a shell in the sandbox, in the project
sandbox/agent.sh claude          # Claude Code; ask it to run tests/test_fp16.py
```

Inside, Claude Code asks before running commands as usual. For an unattended session,
use `claude --dangerously-skip-permissions`. That is fine in here and nowhere else.

### 6. Run a loop

Run it in tmux, so that it survives the terminal, and so that you can look in from your
phone or laptop over Tailscale (`ssh <box>` then `tmux attach -t loop`):

```bash
tmux new -s loop
sandbox/agent.sh python3 scripts/claude_loop.py --name dec90k --preset decode-90k \
    --target '<=35' --max-iters 6 --budget-usd 20 \
    --goal "Lower the decode step time at 90k context on the 2080 Ti (fixed k=4)"
# Ctrl-b d to detach; Ctrl-c in the window stops the loop
```

What one iteration does:

1. `claude -p` gets the goal, the target, the history of every earlier attempt with its
   measured result, and the tail of the last benchmark log. It makes one change, runs
   quick tests and commits (a new commit in `vllm/`, and in the project if needed).
2. The loop checks that no protected file changed. Then it runs the gate (the kernel tests)
   and the measurement, and reads the number from its output. Claude never reports the
   number itself.
3. A change that beats the best result by more than `--min-gain` (0.5%) is kept. Anything
   else is reset to the best commit, and the next iteration sees what was tried and what
   it measured.

The loop stops when the target is met, after `--max-iters`, when the estimated spend
passes `--budget-usd`, after two iterations in a row with no change, or when Claude fails,
for example at a usage limit. In that last case, carry on later with
`... claude_loop.py --name dec90k --resume`.

The presets are:

| Preset | Measures | About |
|---|---|---|
| `decode-90k` | `HYPERINKLING_SPEC_ADAPTIVE= bench/profile_decode.sh` at 90k, wall ms/step, fixed k=4 | 5 min |
| `decode-8k` | the same at 8k | 3 min |
| `prefill-8k` | `bench/profile_prefill.sh` 8k, tok/s (use `--target '>=2400'`) | 3 min |

All three gate on `tests/test_fp16.py` and `tests/test_sm75_attention.py`. Your own
measurement works too, as any command plus a regex with one group:

```bash
... claude_loop.py --name kern --goal "..." --target '<=30' \
    --measure "HYPERINKLING_SPEC_ADAPTIVE= bench/profile_decode.sh {name} 90000 256" \
    --metric 'kernels ([0-9.]+) ms/step' --gate "venv/bin/python -m pytest -q -x tests/test_fp16.py" \
    --repeat 2
```

`{name}` becomes a label unique to the iteration. Use `--repeat` for a noisy number, and
`--extra-prompt` for more instructions, like "only touch ops/linear_w8.py".

### 7. Afterwards

- Send the log: `.local/claude-loop-<name>.txt` has the commit, GPU, every iteration's
  change, Claude's summary, the measured number and what was kept, and the spend. The
  full outputs are in `.local/claude-loop-<name>/` (each benchmark log, each diff, each
  `claude -p` result).
- The work is on branch `loop/<name>`, in the project and in `vllm/`, with the best commit
  checked out. Nothing is pushed: the sandbox cannot reach GitHub.
- To keep a `vllm/` change, squash the loop commits into the one patch commit and
  regenerate the patch, as CLAUDE.md says, outside the sandbox, after reading the diff.
- `sandbox/agent.sh down` stops the container. The home volume (your login, the
  Triton/FlashInfer caches) is kept. `docker volume rm hyperinkling-sandbox_claude-home`
  removes it.

### Link it to the Claude app (Remote Control)

`claude remote-control` in the sandbox makes the session show up in the Claude app, where a
project can hand it work. It needs a full `/login`: the `claude setup-token` token does not
do Remote Control, and takes precedence over `/login`, so remove `CLAUDE_CODE_OAUTH_TOKEN`
from the secrets file first.

```bash
sandbox/agent.sh down && sandbox/agent.sh up     # after editing the secrets file
sandbox/agent.sh claude                          # /login, paste the code, /exit
tmux new -s rc
sandbox/agent.sh claude remote-control
```

The sandbox leaves `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` unset for this, because
Remote Control refuses to run with it. It sets `DISABLE_TELEMETRY` and
`DISABLE_ERROR_REPORTING` instead. If it cannot connect, the firewall is missing a host:
add it with `ALLOW_DOMAINS` and run `sandbox/setup.sh` again.

## Costs and limits

An unattended loop spends tokens without you watching. Each iteration is a fresh
`claude -p` session that reads CLAUDE.md, NOTES.md and the code it touches, so expect
roughly $1 to $5 of API-equivalent usage per iteration on Opus, more for a big change.
The guards:

- `--budget-usd` (default 25) stops the loop on the summed `total_cost_usd` that Claude
  Code reports. That figure is an estimate made on your machine, not an invoice.
  `--iter-budget-usd` (default 5) and `--max-turns` (80) cap each iteration.
- With a subscription, the loop uses your plan's rolling session and weekly limits, which
  you share with claude.ai. When a limit is hit, the call fails, the loop stops cleanly,
  and `--resume` picks it up after the reset. Check with `/usage` in `sandbox/agent.sh claude`.
- With an API key, you pay per token. Set a spend limit in the Console as well.
- The benchmark takes most of each iteration's time, and while it runs no tokens are spent.
- `--model sonnet` costs less per iteration than Opus, and is usually enough for focused
  tuning.

## A lighter alternative: /goal

Claude Code has `/goal <condition>`. It keeps a session working until a small model judges
the condition met, and it runs headless too: `claude -p "/goal ..."`. It is simpler, but the
judge reads only the transcript and runs nothing. Claude would measure itself, and nothing
resets a regression. For "make the tests pass", that is fine. For "35 ms/step", use the loop
script, which measures outside Claude and keeps only real gains.

## Not tested yet

Nothing here has run on your machine. What was checked in the cloud session:

- The loop script, against a stub `claude` in a scratch repository
  (`tests/test_claude_loop.py`). It keeps gains, resets regressions and edits to protected
  files, stops at the target, the budget and on idle iterations, and resumes after an error.
- The image on `ubuntu:24.04` with Docker. The firewall let `api.anthropic.com` and an
  `ALLOW_DOMAINS` host through, refused GitHub, which was reachable without it, and kept
  loopback open. The agent user ends up with your uid and no capabilities, and cannot
  touch iptables. The compose file renders. Claude Code itself was not installed there:
  that session's network policy blocks its download host.

Still to check on the box, and `sandbox/setup.sh` checks the first four itself:

- the GPU in the container, and the venv running there (uv's Python and CUDA mounted)
- the Fedora image (the cloud session could not reach Fedora's mirrors)
- the native installer in the image, and logging in with `CLAUDE_CODE_OAUTH_TOKEN`
- whether `api.anthropic.com` stays reachable over hours: its addresses are resolved
  once, at container start
- a full `start.sh` boot in the sandbox (FlashInfer's first-boot JIT with the mounted
  toolkit, `nsys` for the profile presets, memory under the 26 GB cap)
- a real `claude -p` iteration: whether the managed deny rules hold in
  `bypassPermissions` mode (the loop's own check catches edits to protected files either
  way), and the real cost of an iteration
