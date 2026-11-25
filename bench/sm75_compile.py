"""Compile the Triton kernels of the Turing path for sm75, without a GPU, and check each
one's shared memory against Turing's 64 KB per block.

Nothing is launched: Triton's driver is replaced by one that reports an sm75 device,
every kernel launch only compiles (ptxas included, so an instruction sm75 lacks fails
here too), and the kernels run through their real launchers at the model's shapes in
fp16. A kernel over the limit would fail at launch on a 2080 Ti with OutOfResources.

    python3 bench/sm75_compile.py            # needs torch (CPU is enough), triton, regex

Use the Triton version vLLM's torch pins (torch 2.14: triton 3.8): shared memory depends
on it. The modules under ./vllm are loaded by path with the fp16/sm75 switches forced
on; other vllm imports they make are stubbed. Shapes: hidden 2304, 8 KV heads of 128,
16 query heads (a guess where the code needs one; the tiles do not depend on it).
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import traceback
import types
from pathlib import Path

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.runtime import driver as triton_driver
from triton.runtime.jit import JITFunction

OPS = Path(__file__).resolve().parent.parent / "vllm/vllm/models/inkling2/nvidia/ops"
LIMIT = 65536  # Turing: 64 KB of shared memory per block (opt-in maximum)
D, H, HQ, HID = 128, 8, 16, 2304


class _Dummy:
    def __call__(self, *a, **k):
        return False

    def __getattr__(self, name):
        return _Dummy()


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy()


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Any `vllm.*` module that does not import is an empty stub."""

    def find_spec(self, name, path, target=None):
        if name != "vllm" and not name.startswith("vllm."):
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                if finder.find_spec(name, path, target):
                    return None
            except Exception:
                pass
        return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


class _Sm75Driver:
    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def get_current_target(self):
        return GPUTarget("cuda", 75, 32)

    def get_active_torch_device(self):
        return torch.device("cpu")

    def get_device_interface(self):
        return torch.cuda


RESULTS = []
_run = JITFunction.run


def _compile_only(self, *args, grid, warmup, **kwargs):
    kernel = _run(self, *args, grid=grid, warmup=True, **kwargs)
    consts = {k: v for k, v in kwargs.items() if k.isupper()}
    RESULTS.append((self.fn.__name__, kernel.metadata.shared, kernel.metadata.num_warps,
                    consts))
    return kernel


def _setup():
    sys.meta_path.append(_StubFinder())
    if isinstance(importlib.import_module("vllm.triton_utils"), _StubModule):
        utils = types.ModuleType("vllm.triton_utils")
        utils.tl, utils.triton = triton.language, triton
        from triton.language.extra import libdevice

        utils.tldevice = libdevice
        sys.modules["vllm.triton_utils"] = utils
    pkg = types.ModuleType("ops")
    pkg.__path__ = [str(OPS)]
    switches = dict(
        act_dtype=lambda: torch.float16, use_fp16=lambda: True, is_sm75=lambda: True,
        is_sm8x=lambda: False, is_sm75_or_sm8x=lambda: True, use_pdl=lambda: False,
        use_sm12x_path=lambda: True, use_sm100_kernels=lambda: False,
        use_native_e4m3=lambda: False, use_native_conv_mix=lambda: False,
    )
    for name, value in switches.items():
        setattr(pkg, name, value)
    sys.modules["ops"] = pkg
    triton_driver.set_active(_Sm75Driver())
    JITFunction.run = _compile_only


def load(name):
    if f"ops.{name}" in sys.modules:
        return sys.modules[f"ops.{name}"]
    spec = importlib.util.spec_from_file_location(f"ops.{name}", OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"ops.{name}"] = module
    sys.modules[f"vllm.models.inkling2.nvidia.ops.{name}"] = module
    spec.loader.exec_module(module)
    setattr(sys.modules["ops"], name, module)
    return module


def _r(*shape):
    return torch.randn(*shape).half()


def _w8():
    w8 = load("linear_w8")
    for n in (1024, 2304, 4608, 201024):
        q, s = w8.quantize_int8(torch.randn(n, HID).to(torch.bfloat16))
        for m in (1, 6, 12, 24, 48, 100, 512, 2048):
            w8.w8_matmul(_r(m, HID), q, s)


def _k_groups():
    linear = load("linear")
    for t in (1, 6, 16, 48, 128, 256, 512, 2048, 4096):
        for n, k in ((2304, 2304), (128, 2304), (4608, 2304)):
            linear.k_group_partials(_r(t, k), _r(n, k))  # (Turing prefill: linear_k_groups on cuBLAS)


def _controllers():
    bgemv = load("bgemv")
    for t in (1, 6, 16, 32, 64):
        down, up, pair = _r(3, 64, 2 * HID), _r(3, 2 * HID, 64), _r(3, t, 2 * HID)
        bgemv.controller_gemv(pair, down, up)
        bgemv.bgemv(bgemv.bgemv(pair, down), up, torch.float32)


def _norms():
    norm, loop, attn_res = load("norm"), load("loop"), load("attn_res")
    for t in (1, 512):
        norm.rms_norm(_r(t, HID), _r(HID), 1e-6)
        dest, message = torch.randn(3, t, HID), torch.randn(3, t, HID)
        loop.pair_rms_norm(dest, message)
        loop.blend_rms_norm(dest, message, torch.randn(3, t, 2 * HID), _r(3, HID), 1e-6)
        attn_res.attn_res(_r(6, t, HID), _r(t, HID), _r(t, HID), torch.randn(2, HID),
                          _r(2, HID), 1e-6, True)


def _swiglu():
    moe_sm90 = load("moe_sm90")
    for rows in (8, 4096):
        moe_sm90.swiglu(_r(rows, 2304), torch.rand(rows),
                        torch.arange(rows, dtype=torch.int32), 4.0, 25.0)


def _vision():
    vision = load("vision")
    pixels = torch.full((1, 80, 80, 3), 3.0).to(torch.bfloat16)
    vision.gather_patches(pixels, torch.tensor([0, 120]), 40, (0.5,) * 3, (0.25,) * 3,
                          (0.5,) * 3)
    vision.scaled_residual_norm_(_r(16, 1024), _r(16, 1024), _r(1024),
                                 torch.nn.LayerNorm(1024).half())


def _attention():
    att = load("attention_sm90")
    page = 16
    for q_lens, past in (([1] * 8, 500), ([4] * 6, 500), ([512], 0), ([2048], 0)):
        for window in (0, 1024):
            reqs, tokens = len(q_lens), sum(q_lens)
            blocks = -(-(past + max(q_lens)) // page)
            cache = torch.zeros((2, 1 + reqs * blocks, page, H, D), dtype=torch.uint8)
            q, rel = _r(tokens, HQ, D), _r(tokens + 128, HQ, 1024 + 256)
            table = torch.zeros((reqs, blocks), dtype=torch.int32)
            lens = torch.tensor([past + n for n in q_lens], dtype=torch.int32)
            cu = torch.tensor([0, *q_lens]).cumsum(0).to(torch.int32)
            for mode in (att.MODE_OUT_LSE, att.MODE_PARTIAL, att.MODE_GATED):
                splits = 1 if mode == att.MODE_GATED else 4
                if mode == att.MODE_OUT_LSE:
                    out, lse = torch.empty(tokens, HQ, D), torch.empty(HQ, tokens)
                elif mode == att.MODE_PARTIAL:
                    out = torch.empty(splits, tokens, HQ, D)
                    lse = torch.empty(splits, tokens, HQ)
                else:
                    out, lse = _r(tokens, HQ * D), None
                att.rel_attention(
                    q, cache[0], cache[1], out, lse, rel, _r(tokens, HQ), _r(1, HQ),
                    block_table=table, cache_seqlens=lens, cu_seqlens_q=cu,
                    max_query_len=max(q_lens), softmax_scale=D**-0.5, window=window,
                    num_splits=splits, mode=mode,
                )


def _short_conv():
    sconv = load("sconv")
    heads, width, taps, page = 8, 128, 4, 16
    for t in (1, 6, 48, 512):
        cache = _r(8, heads, page, width)
        ints = [torch.zeros(t, dtype=torch.int64), torch.zeros((1, 4), dtype=torch.int32),
                torch.zeros(t, dtype=torch.int32), torch.zeros(t, dtype=torch.int64),
                torch.zeros(t, dtype=torch.int32)]
        x = _r(t, heads * width)
        sconv.short_conv(x, _r(heads * width, taps), cache, *ints, 0, width)
        sconv.short_conv(x, _r(heads * width, taps), cache, *ints, 0, width,
                         addend=_r(t, heads * width), residual=_r(t, heads * width))


def _dspark():
    """The DSpark drafter's own kernels (fp16 on Turing) and its block attention
    (`block_decode_attention` runs `rel_attention` in draft mode on sm75)."""
    ds, att = load("dspark"), load("attention_sm90")
    hq, hkv, d, block, page = 16, 4, 128, 8, 16
    for reqs in (1, 6):
        t = reqs * block
        ds.mix_rmsnorm(_r(t, 4, HID), torch.rand(4), _r(HID), 1e-6)
        ds.add_rms_norm_(_r(t, HID), _r(t, HID), _r(HID), 1e-6)
        ds.add_rms_norm_(_r(t, d), None, _r(d), 1e-6)
        qkv = _r(t, (hq + 2 * hkv) * d)
        inv_freq = torch.rand(d // 2)
        pos = torch.arange(t)
        ds.qk_norm_rope_(qkv, pos, _r(d), _r(d), inv_freq, hq, hkv, 1e-6)
        ds.rope_(_r(t, hkv, d), pos, inv_freq)
        ds.silu_and_mul(_r(t, 2 * 4608))
        ds.add_softcap(_r(t, 201024), _r(t, 201024), 30.0)
        ds.markov_gate(torch.randn(t, 3 * 256), _r(t, 256))
        blocks = 40
        cache = torch.zeros((1 + reqs * blocks, page, hkv, d), dtype=torch.uint8)
        for window, splits in ((0, 1), (0, 4), (520, 1)):
            out = _r(t, hq * d)
            if splits == 1:
                o, lse, mode = out, None, att.MODE_GATED
            else:
                o, lse = torch.empty(splits, t, hq, d), torch.empty(splits, t, hq)
                mode = att.MODE_PARTIAL
            att.rel_attention(
                _r(t, hq, d), cache, cache.clone(), o, lse, None, out, out,
                block_table=torch.zeros((reqs, blocks), dtype=torch.int32),
                cache_seqlens=torch.full((reqs,), 600, dtype=torch.int32),
                cu_seqlens_q=torch.arange(reqs + 1, dtype=torch.int32) * block,
                max_query_len=block, softmax_scale=d**-0.5, window=window,
                num_splits=splits, mode=mode,
            )


def _prologue():
    sconv = load("sconv")
    page, d_rel, extent = 16, 16, 1024
    for t in (1, 48, 512, 4096):
        key_cache = torch.zeros((4, page, H, D), dtype=torch.uint8)
        sconv.attention_prologue(
            _r(t, HQ, D), _r(t, HQ, d_rel), _r(t, H * D), _r(t, H * D), _r(HQ, D),
            _r(HQ, d_rel), None, _r(8, d_rel + 1, extent + sconv._SHIFT_PAD),
            _r(H * D, 4), _r(H * D, 4), None, None, _r(4, H, page, 4 * D), torch.arange(t),
            torch.zeros((1, 4), dtype=torch.int32), torch.zeros(t, dtype=torch.int32),
            torch.zeros(t, dtype=torch.int64), torch.zeros(2, dtype=torch.int32),
            key_cache, key_cache.clone(), torch.zeros(t, dtype=torch.int64),
            _r(t, HQ, D), _r(t + 128, HQ, extent + 256),
            torch.tensor([0, t], dtype=torch.int32), torch.tensor([t], dtype=torch.int32),
            clip_threshold=22.6, rel_threshold=8.0, norm_eps=1e-6,
        )


def main():
    _setup()
    failed = []
    for step in (_w8, _k_groups, _controllers, _norms, _swiglu, _vision, _attention,
                 _prologue, _short_conv, _dspark):
        try:
            step()
        except Exception:
            failed.append(step.__name__)
            print(f"FAILED to compile {step.__name__}:")
            traceback.print_exc(limit=-4)
    worst = {}
    for name, shared, warps, consts in RESULTS:
        if shared > worst.get(name, (-1,))[0]:
            worst[name] = (shared, warps, consts)
    over = [r for r in RESULTS if r[1] > LIMIT]
    for name, (shared, warps, consts) in sorted(worst.items(), key=lambda x: -x[1][0]):
        flag = "OVER" if shared > LIMIT else "ok  "
        print(f"{flag} {name:30s} {shared:7d} B  warps {warps}  {consts}")
    for name, shared, warps, consts in over:
        print(f"over the limit: {name} {shared} B {consts}")
    print(f"{len(RESULTS)} compiles, {len(over)} over {LIMIT} B, {len(failed)} steps failed")
    sys.exit(1 if over or failed else 0)


if __name__ == "__main__":
    main()
