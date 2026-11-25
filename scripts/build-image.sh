#!/bin/bash
# Build the HyperInkling container image: vLLM's own Dockerfile on our patched checkout
# (./vllm, from scripts/setup-vllm.sh) for one GPU arch (ARCH, default 8.6; 7.5 for Turing),
# then the launcher layer on top.
#
#   scripts/build-image.sh [tag]            # default tag hyperinkling:latest
#
# A tag already on its registry is refused (published images never move: bump the version).
# The built image runs scripts/verify-image.py on the GPU (the compiled ops work on it).
# Env: ARCH (CUDA arch, default 8.6), MAX_JOBS (compile jobs, default: nproc - 1),
# NVCC_THREADS (2), DOCKER (docker),
# BUILD_ARGS (more flags for the vLLM build, e.g. "--build-arg FINAL_BASE_IMAGE=...").
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-hyperinkling:latest}
DOCKER=${DOCKER:-docker}
ARCH=${ARCH:-8.6}
SM=sm${ARCH//./}
JOBS=${MAX_JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}
[ -d vllm/.git ] || { echo "no ./vllm checkout: run scripts/setup-vllm.sh first" >&2; exit 1; }
git -C vllm log -1 --format='%s' | grep -q '^\[hyperinkling\]' \
	|| { echo "./vllm HEAD is not the HyperInkling commit" >&2; exit 1; }
if [[ "$TAG" == */* ]] && "$DOCKER" manifest inspect "$TAG" > /dev/null 2>&1; then
	echo "$TAG is already published: use a new version" >&2
	exit 1
fi

"$DOCKER" build --network=host -f vllm/docker/Dockerfile --target vllm-openai \
	--build-arg torch_cuda_arch_list="$ARCH" \
	${BUILD_ARGS:-} --build-arg max_jobs="$JOBS" --build-arg nvcc_threads="${NVCC_THREADS:-2}" \
	-t "hyperinkling-vllm:$SM" vllm
"$DOCKER" build --network=host --build-arg VLLM_IMAGE="hyperinkling-vllm:$SM" -t "$TAG" .
"$DOCKER" run --rm --gpus all --entrypoint python3 "$TAG" scripts/verify-image.py \
	|| { echo "$TAG: the compiled ops fail on this GPU" >&2; exit 1; }
echo "built and verified $TAG"
