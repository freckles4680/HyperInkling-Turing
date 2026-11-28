#!/bin/bash
# One-time setup of the Claude Code sandbox (docs/claude-sandbox.md): checks the host, writes
# sandbox/.env with this machine's paths, builds the image, starts the container and checks
# the GPU and the firewall from inside. Run it again after moving the project, rebuilding
# venv/ with another Python or CUDA, or to update Claude Code (REBUILD=1).
#
#   sandbox/setup.sh
#   REBUILD=1 sandbox/setup.sh            # fresh image (new Claude Code version)
#   EXTRA_PACKAGES=gcc15-c++ sandbox/setup.sh
#   ALLOW_DOMAINS="pypi.org files.pythonhosted.org" sandbox/setup.sh
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project=$(cd "$here/.." && pwd)
cd "$here"
die() { echo "setup: $*" >&2; exit 1; }
say() { echo "setup: $*"; }

# --- the host --------------------------------------------------------------------------
command -v docker > /dev/null || die "docker is not installed (docs/claude-sandbox.md, step 1)"
docker compose version > /dev/null 2>&1 || die "the docker compose plugin is missing"
docker info > /dev/null 2>&1 || die "cannot talk to the Docker daemon (is it running? is ${USER:-your user} in the docker group?)"
docker info --format '{{json .Runtimes}}' | grep -q nvidia \
	|| die "Docker has no nvidia runtime: install the NVIDIA Container Toolkit and run" \
		"'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker'"
nvidia-smi > /dev/null 2>&1 || die "nvidia-smi fails on the host"

# --- the paths the venv needs at the same place ------------------------------------------
[ -x "$project/venv/bin/python" ] || die "no $project/venv: build it on the host first (scripts/build-vllm.sh)"
py=$(readlink -f "$project/venv/bin/python")
# Every hop of venv/bin/python's symlink chain (and pyvenv.cfg's home) must exist in the
# container. uv links through a minor-version directory (cpython-3.14-...) to the real
# one (cpython-3.14.x-...), so mount the directory that holds all of them.
uv_python=$(python3 - "$project" <<-'PY'
	import os, sys
	project = os.path.realpath(sys.argv[1])
	path, hops = os.path.join(project, "venv", "bin", "python"), []
	for _ in range(40):
	    hops.append(path)
	    if not os.path.islink(path):
	        break
	    path = os.path.join(os.path.dirname(path), os.readlink(path))
	hops.append(os.path.realpath(path))
	cfg = os.path.join(project, "venv", "pyvenv.cfg")
	for line in open(cfg) if os.path.exists(cfg) else ():
	    key, _, value = line.partition("=")
	    if key.strip() == "home":
	        hops.append(os.path.join(value.strip(), "python"))
	outside = [os.path.dirname(os.path.dirname(os.path.abspath(h))) for h in hops
	           if not os.path.abspath(h).startswith(project + os.sep)]
	print(os.path.commonpath(outside) if outside else "")
	PY
)
[ -n "$uv_python" ] || die "venv/bin/python does not lead outside the project ($py): not a uv venv?"
case "$uv_python" in
	/usr | /usr/local | /usr/*) die "venv/ uses the system Python ($py); the sandbox expects uv's (scripts/build-vllm.sh)" ;;
	/ | /home | "$HOME") die "venv/bin/python's links spread over $uv_python, too much to mount: send the output of 'namei -l venv/bin/python'" ;;
esac
cuda_dir=$(cat "$project/venv/cuda_home" 2> /dev/null || echo /usr/local/cuda)
cuda_src=$(readlink -f "$cuda_dir")
[ -x "$cuda_src/bin/nvcc" ] || die "no nvcc under $cuda_dir (venv/cuda_home)"
hf_cache=${HF_CACHE:-$project/hf-cache}
mkdir -p "$project/models" "$hf_cache" "$here/.empty"

# nsys for bench/profile_*.sh: inside the toolkit, or a separate Nsight Systems install.
nsys_src= nsys_dst= nsys_bin=
if [ -x "$cuda_src/bin/nsys" ]; then
	nsys_bin=$cuda_dir/bin
elif command -v nsys > /dev/null; then
	nsys_real=$(readlink -f "$(command -v nsys)")
	nsys_src=$(dirname "$(dirname "$nsys_real")")
	nsys_dst=$nsys_src
	nsys_bin=$(dirname "$nsys_real")
else
	say "no nsys on the host: the bench/profile_*.sh measurements will not run in the sandbox"
fi

# --- the base image: the host's distribution, for the same glibc and compilers ------------
# shellcheck disable=SC1091
. /etc/os-release
base=${BASE:-}
if [ -z "$base" ]; then
	case "$ID" in
		ubuntu | debian | fedora) base=$ID:$VERSION_ID ;;
		*) die "unknown distribution '$ID': set BASE to a matching image, e.g. BASE=ubuntu:24.04" ;;
	esac
fi

# --- the login, outside the project (the agent can read every file of the project) --------
secrets=${SECRETS_FILE:-$HOME/.config/hyperinkling-sandbox/secrets.env}
if [ ! -f "$secrets" ]; then
	mkdir -p "$(dirname "$secrets")"
	(umask 077 && : > "$secrets")
	say "made $secrets (empty): put CLAUDE_CODE_OAUTH_TOKEN=... (from 'claude setup-token')" \
		"or ANTHROPIC_API_KEY=... in it, or log in inside the container (docs/claude-sandbox.md)"
fi
chmod 600 "$secrets"

cat > "$here/.env" <<-EOF
	# Written by sandbox/setup.sh on $(date -I); run it again rather than editing by hand.
	PROJECT=$project
	UV_PYTHON=$uv_python
	CUDA_DIR=$cuda_dir
	CUDA_SRC=$cuda_src
	HF_CACHE=$hf_cache
	NSYS_SRC=${nsys_src:-./.empty}
	NSYS_DST=${nsys_dst:-/opt/no-nsys}
	NSYS_BIN=${nsys_bin:-/usr/local/bin}
	BASE=$base
	UID=$(id -u)
	GID=$(id -g)
	EXTRA_PACKAGES=${EXTRA_PACKAGES:-}
	ALLOW_DOMAINS=${ALLOW_DOMAINS:-}
	MEM_LIMIT=${MEM_LIMIT:-26g}
	SECRETS_FILE=$secrets
EOF
say "wrote sandbox/.env (base image $base, Python $uv_python, CUDA $cuda_dir)"

# --- build, start, check ------------------------------------------------------------------
compose=(docker compose -f "$here/compose.yml")
build_args=()
[ -n "${REBUILD:-}" ] && build_args=(--no-cache --pull)
"${compose[@]}" build "${build_args[@]}"
"${compose[@]}" up -d --force-recreate
for _ in $(seq 30); do
	"${compose[@]}" logs claude 2> /dev/null | grep -q '^.*firewall: outbound limited' && break
	if ! "${compose[@]}" ps --status running -q claude | grep -q .; then
		"${compose[@]}" logs claude >&2
		die "the container stopped (log above)"
	fi
	sleep 1
done
inside() { "${compose[@]}" exec -T -u agent claude "$@"; }
say "checks from inside the container:"
inside nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader \
	|| die "no GPU in the container"
inside "$project/venv/bin/python" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_capability())' \
	|| die "the venv does not run in the container (see the error above)"
inside id -u | grep -qx "$(id -u)" || die "the agent does not have your uid"
if inside curl -s --max-time 5 -o /dev/null https://huggingface.co; then
	die "huggingface.co is reachable from the container: the firewall is not working"
fi
inside claude --version
say "ready. Next: log in once (sandbox/agent.sh claude), then a loop:"
say "  sandbox/agent.sh python3 scripts/claude_loop.py --help"
