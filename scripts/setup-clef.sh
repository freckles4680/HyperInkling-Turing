#!/bin/bash
# Once, before scripts/clef.sh: a venv of its own for Clef, the checkpoint, and an NF4
# copy of it that loads without quantizing. Everything it prints also goes to
# .local/setup-clef.log (send that file back if a step fails).
#
#   scripts/setup-clef.sh                       # Clef 27B: 55 GB download, 19 GB NF4 copy
#   CLEF_MODEL=clef-flash scripts/setup-clef.sh # Clef-Flash 9B: ~19 GB download
#   QUANTIZE=0 scripts/setup-clef.sh            # venv + download only
#   DROP_ORIGINAL=1 scripts/setup-clef.sh       # delete the bf16 download after the NF4 copy
#
# Its own venv (.local/clef-venv) because the release needs transformers 5.10 and the
# vLLM venv pins its own versions; the two never share a process.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p .local models
exec > >(tee .local/setup-clef.log) 2>&1

CLEF_MODEL=${CLEF_MODEL:-clef}
case "$CLEF_MODEL" in
	clef) REPO_ID=Cloudflare/clef ;;
	clef-flash) REPO_ID=Cloudflare/clef-flash ;;
	*) echo "CLEF_MODEL must be clef or clef-flash" >&2; exit 1 ;;
esac
VENV=${VENV:-.local/clef-venv}
PYTHON=${PYTHON:-python3}
# PyTorch wheels built for CUDA 13.0, which still include Turing (sm_75); checked below.
# If the check fails: TORCH_INDEX=https://download.pytorch.org/whl/cu126
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
QUANTIZE=${QUANTIZE:-1}
DROP_ORIGINAL=${DROP_ORIGINAL:-0}

echo "== setup-clef $(date -Is): $CLEF_MODEL ($REPO_ID)"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv || true
df -h models | tail -1

if [ ! -x "$VENV/bin/python" ]; then
	echo "== venv $VENV"
	"$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
echo "== torch from $TORCH_INDEX"
"$VENV/bin/pip" install -q torch --index-url "$TORCH_INDEX"
echo "== transformers and the rest (the release was tested with transformers 5.10.2)"
"$VENV/bin/pip" install -q "transformers==5.10.2" accelerate bitsandbytes safetensors \
	pillow "huggingface_hub>=0.34"

echo "== checks"
"$VENV/bin/python" - << 'EOF'
import torch, transformers, bitsandbytes as bnb
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers",
      transformers.__version__, "bitsandbytes", bnb.__version__)
assert torch.cuda.is_available(), "torch sees no GPU"
cap = torch.cuda.get_device_capability()
arch = f"sm_{cap[0]}{cap[1]}"
print("GPU", torch.cuda.get_device_name(), arch, "wheel archs", torch.cuda.get_arch_list())
assert arch in torch.cuda.get_arch_list(), (
    f"this torch wheel has no {arch} kernels: rerun with "
    "TORCH_INDEX=https://download.pytorch.org/whl/cu126")
from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401 (the release's class)
layer = bnb.nn.Linear4bit(5120, 5120, compute_dtype=torch.float16, quant_type="nf4").cuda()
x = torch.randn(8, 5120, device="cuda", dtype=torch.float16)
y = layer(x)
w = bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state)
ref = x.float() @ w.float().t()
print("bitsandbytes NF4 on the GPU: ok, relative error",
      float((y.float() - ref).norm() / ref.norm()))
q = torch.randn(1, 24, 512, 256, device="cuda", dtype=torch.float16)
k = torch.randn(1, 4, 512, 256, device="cuda", dtype=torch.float16)
out = torch.nn.functional.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=True)
print("fp16 SDPA (GQA, head 256): ok", tuple(out.shape))
EOF

if [ ! -f "models/$CLEF_MODEL/joint_schema_model.py" ]; then
	echo "== download $REPO_ID -> models/$CLEF_MODEL"
	"$VENV/bin/hf" download "$REPO_ID" --local-dir "models/$CLEF_MODEL"
else
	echo "== models/$CLEF_MODEL is already there"
fi

if [ "$QUANTIZE" = 1 ] && [ ! -f "models/$CLEF_MODEL-nf4/joint_schema_model.py" ]; then
	echo "== NF4 copy -> models/$CLEF_MODEL-nf4 (quantizes on the GPU, a few minutes)"
	if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null | grep -q .; then
		echo "the GPU is in use (an Inkling server?): stop it first, then rerun" >&2
		exit 1
	fi
	CLEF_PATH=models/$CLEF_MODEL CLEF_QUANT=nf4 HF_HUB_OFFLINE=1 \
		"$VENV/bin/python" scripts/clef_server.py --save-quantized "models/$CLEF_MODEL-nf4"
	if [ "$DROP_ORIGINAL" = 1 ]; then
		echo "== removing the bf16 download models/$CLEF_MODEL"
		rm -rf "models/$CLEF_MODEL"
	fi
fi
du -sh models/"$CLEF_MODEL"* 2> /dev/null || true
echo "== done: CLEF_MODEL=$CLEF_MODEL scripts/clef.sh  (or scripts/serve-switch.sh)"
