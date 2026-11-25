#!/bin/bash
# Turing (sm75) only: compile kernels/sm75/*.cu into venv/hyperinkling_sm75.so, the
# tensor-core decode attention Triton cannot emit for sm75 (about a minute, no vLLM
# rebuild). start.sh uses the library when it is there (HYPERINKLING_SM75_LIB); without
# it the Triton kernels serve.
#   scripts/build-sm75-kernels.sh          # after scripts/build-vllm.sh, and after a git
#                                          # pull that changes kernels/sm75/
#   CUDA_HOME=/opt/cuda-13.2 scripts/build-sm75-kernels.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${CUDA_HOME:-}" ] && [ -f venv/cuda_home ]; then CUDA_HOME=$(cat venv/cuda_home); fi
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
nvcc=$CUDA_HOME/bin/nvcc
[ -x "$nvcc" ] || { echo "no nvcc at $nvcc: set CUDA_HOME" >&2; exit 1; }
ccbin=()
# nvcc 13.3 rejects gcc 16 (the Fedora 44 default), as in build-vllm.sh.
if command -v g++-15 > /dev/null; then ccbin=(-ccbin g++-15); fi
out=venv/hyperinkling_sm75.so
# --cudart shared: the CUDA runtime torch has already loaded, not a second static copy.
"$nvcc" "${ccbin[@]}" -O3 -arch=sm_75 -shared --cudart shared -Xcompiler -fPIC \
	-o "$out.tmp" kernels/sm75/rel_attention.cu
mv "$out.tmp" "$out"
echo "built $out ($("$nvcc" --version | grep -o 'release [0-9.]*'))"
