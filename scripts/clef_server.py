#!/usr/bin/env python3
"""Serve Cloudflare Clef (a decision model) on a Turing card, with the Jev/SystemOne API.

Clef does not write text: for a state (any text or JSON, optionally images) and a set of
typed questions it returns a probability per allowed answer, in one forward pass. The
model code is the release's own joint_schema_model.py (loaded from the checkpoint); this
file adds what the 2080 Ti needs and an HTTP server:

  * fp16 compute (Turing has no bf16 tensor cores), with an overflow check
  * 4-bit NF4 weights through bitsandbytes (27B in bf16 is 55 GB), vision tower in fp16
  * token embeddings and LM head as INT8 tables (the model only looks rows up in both:
    the decision head reads the LM head's rows, it never computes logits)
  * the PyTorch Gated DeltaNet run 8 heads at a time (bitwise the same, 2 GB less at 16k)
  * at boot, a fit check: the longest input that fits with headroom, or a clear failure
  * --save-quantized: write the NF4 model once, later loads skip the quantization

Memory of the 27B (bench/clef_memory.py): 15.2 GiB of weights, 2.5 GiB of activations at
16k tokens, on a 22 GiB card.

    POST /v1/systemone   {"model": "clef", "state": "...", "questions": {...}}
    GET  /v1/models, GET /health

Run it through scripts/clef.sh (settings, venv); docs/clef.md has the request format.
"""

import argparse
import base64
import io
import json
import os
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def log(text):
    print(f"{time.strftime('%H:%M:%S')} clef: {text}", flush=True)


# --- loading --------------------------------------------------------------------------
def quantization_config(quant, skip):
    import torch
    from transformers import BitsAndBytesConfig

    if quant == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_use_double_quant=True,
                                  llm_int8_skip_modules=skip)
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
    return None


def backbone_of(model):
    for child in model.children():
        if hasattr(child, "save_pretrained"):
            return child
    raise RuntimeError("no transformers model inside the Clef model")


class Int8Rows:
    """A [vocab, hidden] table as int8 rows with one fp16 scale per row; `table[ids]` gives
    the rows back in the original dtype. Half the bytes of fp16 (1.27 GB less per table
    for the 27B); the rounding error is under 0.4% of each row's largest value."""

    def __init__(self, weight, chunk=8192):
        import torch

        vocab, hidden = weight.shape
        self.dtype, self.device, self.shape = weight.dtype, weight.device, weight.shape
        self.q = torch.empty(vocab, hidden, dtype=torch.int8, device=weight.device)
        self.scale = torch.empty(vocab, 1, dtype=torch.float16, device=weight.device)
        for i in range(0, vocab, chunk):
            w = weight[i:i + chunk].float()
            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127
            self.q[i:i + chunk] = (w / scale).round().clamp(-127, 127).to(torch.int8)
            self.scale[i:i + chunk] = scale.to(torch.float16)

    def __getitem__(self, ids):
        return self.q[ids].to(self.dtype) * self.scale[ids].to(self.dtype)

    def nbytes(self):
        return self.q.numel() + self.scale.numel() * 2


def int8_tables(backbone):
    """Token embeddings and LM head as Int8Rows: both are only ever indexed here."""
    import torch

    class Embedding(torch.nn.Module):
        def __init__(self, rows, padding_idx):
            super().__init__()
            self.rows, self.padding_idx = rows, padding_idx
            self.weight = rows  # for code that reads .weight.dtype / .device

        def forward(self, ids):
            return self.rows[ids]

    class TableOnly(torch.nn.Module):
        def __init__(self, rows):
            super().__init__()
            self.weight = rows

        def forward(self, hidden):
            raise RuntimeError("Clef never computes LM logits; the LM head is a table here")

    saved = 0
    embed = backbone.get_input_embeddings()
    if isinstance(embed, torch.nn.Embedding):
        rows = Int8Rows(embed.weight.data)
        saved += embed.weight.numel() * embed.weight.element_size() - rows.nbytes()
        backbone.set_input_embeddings(Embedding(rows, embed.padding_idx))
    del embed  # one fp16 table at a time: the load's peak stays under the served size
    torch.cuda.empty_cache()
    head = backbone.get_output_embeddings()
    if isinstance(head, torch.nn.Linear) and head.bias is None:
        rows = Int8Rows(head.weight.data)
        saved += head.weight.numel() * head.weight.element_size() - rows.nbytes()
        backbone.set_output_embeddings(TableOnly(rows))
    del head
    torch.cuda.empty_cache()
    return saved


def no_gqa_in_sdpa():
    """K/V repeated to the query heads before SDPA. With enable_gqa, PyTorch on sm75 has
    no flash kernel and falls back to the math kernel, which builds the whole attention
    matrix (13 GB at 16k tokens); without it the memory-efficient kernel runs."""
    import transformers.integrations.sdpa_attention as sdpa

    sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False


def cap_image_pixels(processor, max_pixels):
    """Bigger images are scaled down by the processor (32x32 px per token after merge)."""
    image = getattr(processor, "image_processor", None)
    if image is None or not max_pixels:
        return False
    done = False
    if hasattr(image, "max_pixels"):
        image.max_pixels, done = max_pixels, True
    size = getattr(image, "size", None)  # a dict, or transformers 5's SizeDict
    if isinstance(size, dict) and "longest_edge" in size:
        image.size = {**size, "longest_edge": max_pixels}
        done = True
    elif getattr(size, "longest_edge", None) is not None:
        size.longest_edge = max_pixels
        done = True
    return done


def split_delta_rule_heads(model, group):
    """Transformers' PyTorch Gated DeltaNet (no fla kernels: Triton on Turing has no tensor
    cores and 64 KB of shared memory) makes ~8 float32 copies of q/k/v for all 48 heads at
    once: ~3 GB at 16k tokens. Heads are independent, so run them `group` at a time."""
    import inspect

    import torch

    wrapped = 0
    for module in model.modules():
        fn = getattr(module, "chunk_gated_delta_rule", None)
        if fn is None or getattr(fn, "__name__", "") != "torch_chunk_gated_delta_rule":
            continue  # absent, or the fla kernel (already memory-light)
        signature = inspect.signature(fn)

        def split(*args, _fn=fn, _sig=signature, **kwargs):
            bound = _sig.bind(*args, **kwargs)
            a = bound.arguments
            heads = a["query"].shape[2]
            if a.get("initial_state") is not None or heads <= group:
                return _fn(*args, **kwargs)
            outs, states = [], []
            for h in range(0, heads, group):
                part = dict(a)
                for key in ("query", "key", "value", "g", "beta"):
                    part[key] = a[key][:, :, h:h + group]
                extra = part.pop("kwargs", {})
                out, state = _fn(**part, **extra)
                outs.append(out)
                states.append(state)
            state = None if states[0] is None else torch.cat(states, dim=1)
            return torch.cat(outs, dim=2), state

        split.__name__ = "split_heads"
        module.chunk_gated_delta_rule = split
        wrapped += 1
    return wrapped


def headroom_hooks(backbone):
    """Largest |value| leaving each decoder layer (fp16 saturates at 65504)."""
    import torch

    seen = {"max": 0.0, "layer": -1}
    layers = None
    for name, module in backbone.named_modules():
        if name.endswith("layers") and isinstance(module, torch.nn.ModuleList):
            if layers is None or len(module) > len(layers):  # text layers, not vision
                layers = module
    for i, layer in enumerate(layers or []):
        def hook(module, args, output, i=i):
            out = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(out):
                value = out.detach().abs().amax().float().item()
                if not value < seen["max"]:  # also catches nan
                    seen["max"], seen["layer"] = value, i
        layer.register_forward_hook(hook)
    return seen


class Clef:
    def __init__(self, args):
        import torch

        self.torch = torch
        self.args = args
        self.lock = threading.Lock()
        path = Path(args.model).resolve()
        if not (path / "joint_schema_model.py").is_file():
            raise SystemExit(f"{path} is not a Clef checkpoint (no joint_schema_model.py): "
                             "scripts/setup-clef.sh downloads it")
        sys.path.insert(0, str(path))
        import joint_schema_model as jsm

        self.jsm = jsm
        no_gqa_in_sdpa()
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
        prequantized = "quantization_config" in json.loads((path / "config.json").read_text())
        kwargs = {"attn_implementation": "sdpa"}
        skip = ["lm_head", "visual", *[s for s in args.skip.split(",") if s]]
        if not prequantized:
            config = quantization_config(args.quant, skip)
            if config is not None:
                kwargs["quantization_config"] = config
        gpu = "no GPU"
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            gpu = f"{torch.cuda.get_device_name()} sm{cap[0]}{cap[1]}"
        log(f"torch {torch.__version__}, {gpu}, loading {path.name} "
            f"({'pre-quantized' if prequantized else args.quant}, {args.dtype})")
        started = time.monotonic()
        self.model, self.processor = jsm.load_release_model(path, device=args.device,
                                                            dtype=dtype, **kwargs)
        backbone = backbone_of(self.model)
        if args.save_quantized:
            self._memory(started)
            self.save(path, Path(args.save_quantized), backbone)
            return
        notes = []
        if args.tables == "int8":
            notes.append(f"embeddings + LM head as INT8 tables "
                         f"({int8_tables(backbone) / 2**30:.2f} GiB less)")
        if args.delta_heads:
            notes.append(f"{split_delta_rule_heads(backbone, args.delta_heads)} DeltaNet "
                         f"layers {args.delta_heads} heads at a time")
        if cap_image_pixels(self.processor, args.image_pixels):
            notes.append(f"images scaled to {args.image_pixels:,} px at most")
        if notes:
            log("; ".join(notes))
        self.seen = headroom_hooks(backbone) if args.dtype == "float16" else None
        self._memory(started)
        self.max_len = self.fit(args.max_len, args.headroom_gb)

    def _memory(self, started):
        torch = self.torch
        note = f"loaded in {time.monotonic() - started:.0f} s"
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            note += (f", {torch.cuda.memory_allocated() / 2**30:.2f} GiB on the GPU "
                     f"(card {total / 2**30:.1f} GiB, {free / 2**30:.1f} free)")
        log(note)

    def fit(self, wanted, headroom_gb):
        """Run the longest input once and keep `headroom_gb` free on the card after it, so
        every request this server accepts is known to fit (requests are cut to this
        length, as the release does). auto = the longest that fits, from 16384 down."""
        torch = self.torch
        if not torch.cuda.is_available():
            return 16384 if wanted == "auto" else int(wanted)
        lengths = [16384, 12288, 8192, 6144, 4096, 2048] if wanted == "auto" else [int(wanted)]
        filler = ("Ticket log line: the payment page timed out for a customer in region "
                  "eu-west, retried, and succeeded on the second attempt. ")
        for n in lengths:
            probe = {"model": "fit-check", "state": filler * (n // 20 + 10),
                     "questions": {"outage": {"type": "noul", "instructions": "Is it down?"}}}
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            try:
                with self.lock:
                    self.jsm.systemone(self.model, self.processor, probe, max_length=n)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"fit check: {n:,} tokens: out of GPU memory")
                continue
            seconds = time.monotonic() - started
            free = torch.cuda.mem_get_info()[0] / 2**30  # the cache still holds the peak
            peak = torch.cuda.max_memory_reserved() / 2**30
            if free < headroom_gb:
                log(f"fit check: {n:,} tokens fit but leave {free:.2f} GiB free "
                    f"(CLEF_HEADROOM_GB={headroom_gb}): too tight")
                continue
            torch.cuda.empty_cache()
            log(f"FIT: {self.args.name} takes inputs up to {n:,} tokens on this card: "
                f"{peak:.2f} GiB used by the server at its peak, {free:.2f} GiB still free; "
                f"{n:,} tokens in {seconds:.1f} s ({n / seconds:,.0f} tok/s)")
            return n
        total = torch.cuda.mem_get_info()[1] / 2**30
        raise SystemExit(
            f"clef: DOES NOT FIT: {self.args.name} could not run "
            f"{'even 2,048 tokens' if wanted == 'auto' else f'{lengths[0]:,} tokens'} with "
            f"{headroom_gb} GiB to spare on this {total:.1f} GiB card. Free the GPU (a desktop, "
            "another server: nvidia-smi), or CLEF_MODEL=clef-flash (the 9B), "
            "or a lower CLEF_MAX_LEN.")

    def save(self, source, target, backbone):
        target.mkdir(parents=True, exist_ok=True)
        log(f"writing the quantized model to {target}")
        backbone.save_pretrained(target)
        self.processor.save_pretrained(target)
        for name in ("joint_head.safetensors", "joint_head_config.json",
                     "joint_schema_model.py", "chat_template.jinja", "LICENSE", "README.md"):
            if (source / name).is_file():
                shutil.copy2(source / name, target / name)
        log(f"done: CLEF_PATH={target} scripts/clef.sh  loads it without quantizing")

    def answer(self, request):
        torch = self.torch
        request = dict(request)
        if request.get("images"):
            request["images"] = [decode_image(i) for i in request["images"]]
        if request.get("videos"):
            raise ValueError("videos are not taken over HTTP here; send frames as images")
        with self.lock:  # one forward at a time on one card
            if self.seen is not None:
                self.seen.update(max=0.0, layer=-1)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            try:
                response = self.jsm.systemone(self.model, self.processor, request,
                                              max_length=self.max_len)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise MemoryError("out of GPU memory although the fit check passed at boot "
                                  f"({self.max_len:,} tokens): did something else take GPU "
                                  "memory since (nvidia-smi)? Images also cost memory")
            seconds = time.monotonic() - started
        tokens = response.get("usage", {}).get("input_tokens", 0)
        note = f"{tokens:,} tokens in {seconds:.2f} s ({tokens / max(seconds, 1e-6):,.0f} tok/s)"
        if torch.cuda.is_available():
            note += f", peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        if self.seen is not None:
            peak = self.seen["max"]
            note += f", fp16 peak |x| {peak:,.0f} at layer {self.seen['layer']}"
            if not peak < 65504:
                raise ArithmeticError(
                    f"fp16 overflow at layer {self.seen['layer']} (|x| = {peak}): the "
                    "answer would be wrong. CLEF_DTYPE=bfloat16 avoids it (slower on "
                    "Turing); please send .local/model-switch.log or this server's output")
        log(note)
        return response


def decode_image(item):
    """An image as a data: URL or bare base64 (files and URLs are not fetched)."""
    from PIL import Image

    if not isinstance(item, str):
        raise ValueError("each image must be a base64 string or a data: URL")
    if item.startswith("data:"):
        item = item.split(",", 1)[-1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB")
    except Exception as exc:
        raise ValueError(f"an image could not be decoded: {exc}") from exc


# --- HTTP -----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    service = None   # anything with .answer(dict) -> dict
    names = ()
    api_key = ""

    def log_message(self, *args):
        pass

    def _json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status, message, kind="invalid_request_error"):
        self._json(status, {"error": {"message": message, "type": kind, "code": status}})

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            return self._json(200 if self.service else 503, {"status": "ok"})
        if self.api_key and self.headers.get("Authorization") != f"Bearer {self.api_key}":
            return self._error(401, "missing or wrong API key", "authentication_error")
        if path in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": n, "object": "model", "created": 0, "owned_by": "cloudflare"}
                for n in self.names]})
        self._error(404, f"{path}: this server answers POST /v1/systemone (Clef is a "
                         "decision model, it does not chat)", "not_found_error")

    def do_POST(self):
        path = self.path.split("?")[0]
        if self.api_key and self.headers.get("Authorization") != f"Bearer {self.api_key}":
            return self._error(401, "missing or wrong API key", "authentication_error")
        if path not in ("/v1/systemone", "/systemone"):
            return self._error(404, f"{path}: this server answers POST /v1/systemone (Clef "
                                    "is a decision model, it does not chat)",
                               "not_found_error")
        if self.service is None:
            return self._error(503, "the model is still loading", "server_error")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length) or b"null")
            if not isinstance(request, dict):
                raise ValueError("the body must be a JSON object")
            request.setdefault("model", self.names[0] if self.names else "clef")
            response = self.service.answer(request)
        except (ValueError, KeyError, TypeError) as exc:
            return self._error(400, str(exc))
        except (MemoryError, ArithmeticError) as exc:
            return self._error(503, str(exc), "server_error")
        except Exception as exc:  # keep serving; the trace goes to the log
            import traceback

            traceback.print_exc()
            return self._error(500, f"{type(exc).__name__}: {exc}", "server_error")
        self._json(200, response)


SELFTEST = {
    "state": "Our checkout started returning errors at 09:12 and every order since is "
             "blocked. Customers are calling. Please help ASAP.",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle it?",
                       "criteria": {"billing": "Payments or invoices",
                                    "technical": "Bugs or outages"}},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}


def selftest(service, name):
    """The expected answers are technical / ~2 (today) / outage true."""
    response = service.answer({"model": name, **SELFTEST})
    log("self-test: " + json.dumps(response["answers"]))


def main():
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=env("CLEF_PATH", "models/clef"))
    ap.add_argument("--name", default=env("CLEF_NAME", "clef"),
                    help="model id(s) in /v1/models, comma-separated")
    ap.add_argument("--quant", choices=["nf4", "int8", "none"], default=env("CLEF_QUANT", "nf4"))
    ap.add_argument("--dtype", choices=["float16", "bfloat16"],
                    default=env("CLEF_DTYPE", "float16"))
    ap.add_argument("--skip", default=env("CLEF_SKIP", ""),
                    help="more module names to keep unquantized, comma-separated")
    ap.add_argument("--tables", choices=["int8", "fp16"], default=env("CLEF_TABLES", "int8"),
                    help="token embeddings and LM head")
    ap.add_argument("--delta-heads", type=int, default=int(env("CLEF_DELTA_HEADS", "8")),
                    help="DeltaNet heads per step (0 = all at once)")
    ap.add_argument("--image-pixels", type=int,
                    default=int(env("CLEF_IMAGE_PIXELS", str(1024 * 1024))))
    ap.add_argument("--max-len", default=env("CLEF_MAX_LEN", "auto"),
                    help="longest input in tokens, or auto (the fit check finds it)")
    ap.add_argument("--headroom-gb", type=float, default=float(env("CLEF_HEADROOM_GB", "1.0")),
                    help="GPU memory left free at the longest input")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bind", default=env("BIND", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(env("PORT", "18090")))
    ap.add_argument("--selftest", type=int, default=int(env("CLEF_SELFTEST", "1")))
    ap.add_argument("--save-quantized", default="",
                    help="write the quantized model to this directory and exit")
    args = ap.parse_args()
    names = [n for n in args.name.split(",") if n]
    service = Clef(args)
    if args.save_quantized:
        return 0
    if args.selftest:
        selftest(service, names[0])
    Handler.service, Handler.names = service, names
    Handler.api_key = env("API_KEY", "")
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = True
    log(f"listening on http://{args.bind}:{args.port}/v1/systemone as {', '.join(names)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
