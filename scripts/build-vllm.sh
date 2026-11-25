#!/bin/bash
# Build vLLM (checkout in ./vllm) for one GPU arch, into ./venv.
#   scripts/build-vllm.sh             # sm86 (RTX 3090 / 3090 Ti)
#   ARCH=7.5 scripts/build-vllm.sh    # sm75 (Turing: RTX 20xx, 2080 Ti 22 GB mods)
set -euo pipefail
cd "$(dirname "$0")/.."
# Leave one of the 32 hardware threads free: pin the whole build (nvcc, cc1plus,
# cargo, which sizes its job count from this affinity) to CPUs 0-30.
if [ -z "${HYPERINKLING_PINNED:-}" ]; then
	HYPERINKLING_PINNED=1 exec taskset -c 0-30 "$0" "$@"
fi
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-31}"
[ -d venv ] || uv venv venv --python 3.14
source venv/bin/activate
cd vllm
uv pip install -r requirements/build/cuda.txt
uv pip install -r requirements/cuda.txt
export TORCH_CUDA_ARCH_LIST="${ARCH:-8.6}" MAX_JOBS="${MAX_JOBS:-16}" NVCC_THREADS=2 CUDA_HOME=/usr/local/cuda
# nvcc 13.3 rejects gcc 16 (the Fedora 44 default); gcc-15 is installed alongside.
export CC=gcc-15 CXX=g++-15 CUDAHOSTCXX=g++-15 NVCC_CCBIN=g++-15
uv pip install -e . --no-build-isolation -v
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_capability())"
