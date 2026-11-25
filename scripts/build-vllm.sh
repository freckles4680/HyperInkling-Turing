#!/bin/bash
# Build vLLM (checkout in ./vllm) for one GPU arch, into ./venv.
#   scripts/build-vllm.sh             # sm86 (RTX 3090 / 3090 Ti)
#   ARCH=7.5 scripts/build-vllm.sh    # sm75 (Turing: RTX 20xx, 2080 Ti 22 GB mods)
#   CUDA_HOME=/opt/cuda-13.2 ...      # a CUDA 13 toolkit kept beside the system one
#                                     # (recorded in venv/cuda_home; start.sh uses it)
# Everything goes into ./venv (uv, its own Python); the host's Python, CUDA toolkit
# and PATH are not changed.
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
export TORCH_CUDA_ARCH_LIST="${ARCH:-8.6}" MAX_JOBS="${MAX_JOBS:-16}" NVCC_THREADS=2
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 13\.' || {
	echo "$CUDA_HOME/bin/nvcc is not CUDA 13: set CUDA_HOME to a CUDA 13 toolkit" >&2
	exit 1
}
# nvcc 13.3 rejects gcc 16 (the Fedora 44 default): gcc-15 where it is installed alongside,
# else the system compiler.
if command -v gcc-15 > /dev/null && command -v g++-15 > /dev/null; then
	export CC=gcc-15 CXX=g++-15 CUDAHOSTCXX=g++-15 NVCC_CCBIN=g++-15
fi
uv pip install -e . --no-build-isolation -v
uv pip install pytest   # tests/ (the kernel tests run in this venv)
# start.sh serves with the same toolkit (FlashInfer's JIT kernels compile at the first boot).
echo "$CUDA_HOME" > ../venv/cuda_home
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_capability())"
