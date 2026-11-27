#!/usr/bin/env python3
"""GPU memory of Clef as scripts/clef_server.py runs it, computed on the CPU (no GPU, no
download): weights from the real module tree, activations from a forward pass on the meta
device that tracks every tensor alive at each op. Needs torch and transformers 5.10
(.local/clef-venv after scripts/setup-clef.sh).

    .local/clef-venv/bin/python bench/clef_memory.py                  # both models, 22 GiB card
    .local/clef-venv/bin/python bench/clef_memory.py --model clef --lengths 8192,16384
    .local/clef-venv/bin/python bench/clef_memory.py --config models/clef/config.json

The configuration is the released one (it reproduces the 27,356,728,560 parameters of
Cloudflare/clef). What it cannot see: the CUDA context and cuBLAS workspaces (--context,
0.4 GiB by default), bitsandbytes dequantizing one weight at a time during a prefill (the
largest is added), and the allocator's fragmentation. The boot fit check of
clef_server.py measures the real peak on the card.
"""

import argparse
import json
import sys
import weakref
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

GiB = 2**30
NF4_BYTES = 0.5 + 1 / 64 + 4 / (64 * 256)  # 4-bit codes + uint8 absmax/64 + fp32 per 256 blocks
JOINT_HEAD_BYTES = 256_125_024             # joint_head.safetensors of Cloudflare/clef

RELEASED = {  # text_config / vision_config of the two releases (config.json)
    "clef": dict(hidden_size=5120, intermediate_size=17408, num_hidden_layers=64,
                 num_attention_heads=24, linear_num_value_heads=48),
    "clef-flash": dict(hidden_size=4096, intermediate_size=12288, num_hidden_layers=32,
                       num_attention_heads=16, linear_num_value_heads=32),
}


def build_config(model, path=None):
    from transformers import Qwen3_5Config

    if path:
        return Qwen3_5Config(**json.loads(Path(path).read_text()))
    t = RELEASED[model]
    layers = t["num_hidden_layers"]
    text = dict(num_key_value_heads=4, head_dim=256, vocab_size=248320,
                layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                             for i in range(layers)],
                full_attention_interval=4, attn_output_gate=True, linear_num_key_heads=16,
                linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                tie_word_embeddings=False, partial_rotary_factor=0.25,
                rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                                 "mrope_section": [11, 11, 10], "mrope_interleaved": True,
                                 "partial_rotary_factor": 0.25}, **t)
    vision = dict(depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16,
                  patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                  out_hidden_size=t["hidden_size"], num_position_embeddings=2304,
                  in_channels=3, hidden_act="gelu_pytorch_tanh")
    return Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=False)


class LiveBytes(TorchDispatchMode):
    """Peak bytes of tensor storage alive during the ops run under it."""

    def __init__(self):
        super().__init__()
        self.live, self.now, self.peak = {}, 0, 0

    def _drop(self, key):
        size, refs = self.live[key]
        if refs == 1:
            del self.live[key]
            self.now -= size
        else:
            self.live[key] = (size, refs - 1)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor):
                storage = t.untyped_storage()
                key = storage._cdata
                size, refs = self.live.get(key, (storage.nbytes(), 0))
                if refs == 0:
                    self.now += size
                self.live[key] = (size, refs + 1)
                weakref.finalize(t, self._drop, key)
        self.peak = max(self.peak, self.now)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["clef", "clef-flash", "both"], default="both")
    ap.add_argument("--config", help="a checkpoint's config.json instead of the built-in one")
    ap.add_argument("--lengths", default="4096,8192,16384")
    ap.add_argument("--card", type=float, default=22.0, help="GiB on the card")
    ap.add_argument("--context", type=float, default=0.4, help="GiB of CUDA context")
    ap.add_argument("--delta-heads", type=int, default=8)
    args = ap.parse_args()

    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    import clef_server
    from transformers import Qwen3_5ForConditionalGeneration

    # The meta device cannot answer data-dependent questions: an all-ones mask means
    # SDPA's is_causal path, as on the card for one request.
    qwen.create_causal_mask = lambda *a, **k: None
    for cls in vars(qwen).values():
        if isinstance(cls, type) and hasattr(cls, "_update_linear_attn_mask"):
            cls._update_linear_attn_mask = lambda self, mask, *a, **k: None
    clef_server.no_gqa_in_sdpa()

    for name in (["clef", "clef-flash"] if args.model == "both" else [args.model]):
        config = build_config(name, args.config)
        config._attn_implementation = config.text_config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = Qwen3_5ForConditionalGeneration(config).half().eval()
        counts = {"vision": 0, "tables": 0, "linear": 0, "other": 0}
        largest = 0
        for mod_name, module in model.named_modules():
            direct = list(module.parameters(recurse=False))
            if not direct:
                continue
            n = sum(p.numel() for p in direct)
            if "visual" in mod_name:
                counts["vision"] += n
            elif mod_name.endswith("embed_tokens") or mod_name == "lm_head":
                counts["tables"] += n
            elif isinstance(module, torch.nn.Linear):
                counts["linear"] += n
                largest = max(largest, n)
            else:
                counts["other"] += n
        total = sum(counts.values())
        hidden = config.text_config.hidden_size
        weights = {
            "NF4 layers": counts["linear"] * NF4_BYTES,
            "INT8 embeddings + LM head": counts["tables"] * (1 + 2 / hidden),
            "vision tower (fp16)": counts["vision"] * 2,
            "norms, conv, gates (fp16)": counts["other"] * 2,
            "joint head (fp16)": JOINT_HEAD_BYTES * (hidden / 5120),
        }
        fixed = sum(weights.values()) / GiB + args.context
        print(f"\n{name}: {total:,} parameters (backbone)")
        for key, value in weights.items():
            print(f"  {key:28s} {value / GiB:6.2f} GiB")
        print(f"  {'CUDA context (assumed)':28s} {args.context:6.2f} GiB")
        print(f"  {'= before any input':28s} {fixed:6.2f} GiB of {args.card:.1f}")
        if args.delta_heads:
            clef_server.split_delta_rule_heads(model, args.delta_heads)
        text = model.model.language_model
        dequant = largest * 2 / GiB  # one weight dequantized to fp16 during a prefill
        print(f"  {'tokens':>8s} {'activations':>12s} {'total':>8s} {'free':>8s}")
        for n in [int(x) for x in args.lengths.split(",")]:
            ids = torch.zeros(1, n, dtype=torch.long, device="meta")
            tracker = LiveBytes()
            with torch.inference_mode(), tracker:
                text(input_ids=ids, attention_mask=None, use_cache=False, return_dict=True)
            act = tracker.peak / GiB + dequant
            used = fixed + act
            print(f"  {n:8,d} {act:9.2f} GiB {used:6.2f} {args.card - used:6.2f} GiB")


if __name__ == "__main__":
    main()
